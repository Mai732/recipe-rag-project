import streamlit as st
import json
import os
import numpy as np
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.0-flash-lite")

# Load data
@st.cache_resource
def load_data():
    with open("chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    index = faiss.read_index("recipe_index.faiss")
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return chunks, index, embed_model

chunks, index, embed_model = load_data()

# Functions
def retrieve(query, top_k=5):
    query_vec = embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(query_vec, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        results.append({
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "similarity_score": float(dist),
        })
    return results

def rewrite_query(query):
    try:
        prompt = f"Rewrite this query to be clearer for recipe search. Return ONLY the rewritten query:\n{query}"
        response = gemini.generate_content(prompt)
        return response.text.strip()
    except:
        return query

def classify_query(query):
    try:
        prompt = f"""Classify this query into ONE of: factual, recommendation, general, out_of_scope
Return ONLY the category name.
Query: {query}"""
        response = gemini.generate_content(prompt)
        return response.text.strip().lower()
    except:
        return "general"

def build_prompt(query, chunks):
    context = ""
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        context += f"--- Source {i} ---\nTitle: {meta.get('title','')}\n{chunk['text']}\n\n"
    return f"""You are a recipe assistant. Answer ONLY from the context below.
If not found, say: "I don't have enough information."

Context:
{context}

Question: {query}
Answer:"""

def get_answer(query):
    retrieved = retrieve(query)
    if not retrieved:
        return "No relevant recipes found.", []
    prompt = build_prompt(query, retrieved)
    try:
        response = gemini.generate_content(prompt)
        return response.text, retrieved
    except Exception as e:
        return f"Error: {e}", retrieved

# Streamlit UI
st.title("Recipe RAG Assistant")
st.markdown("Ask any question about recipes!")

query = st.text_input("Enter your question:", placeholder="How do I make pancakes?")

if st.button("Search") and query:
    with st.spinner("Searching..."):
        
        # Query Intelligence
        rewritten = rewrite_query(query)
        query_class = classify_query(query)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Original Query:** {query}")
            st.success(f"**Rewritten Query:** {rewritten}")
        with col2:
            st.warning(f"**Query Class:** {query_class}")
        
        # Get answer
        answer, retrieved = get_answer(rewritten)
        
        # Show answer
        st.markdown("### 📝 Answer")
        st.write(answer)
        
        # Show retrieved chunks
        st.markdown("### 📚 Retrieved Sources")
        for i, r in enumerate(retrieved, 1):
            with st.expander(f"Chunk {i}: {r['metadata']['title']} (Score: {r['similarity_score']:.4f})"):
                st.write(f"**Category:** {r['metadata']['category']}")
                st.write(f"**URL:** {r['metadata']['url']}")
                st.write(f"**Text:** {r['text'][:300]}...")