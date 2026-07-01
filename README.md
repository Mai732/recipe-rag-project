# Recipe RAG System

An AI-powered recipe question-answering assistant built using Retrieval-Augmented Generation (RAG).

## Project Structure

- `recipe_rag.ipynb` — Level 0 & Level 1: Data collection, chunking, embedding, and RAG pipeline
- `Query.ipynb` — Level 2: Query intelligence (rewriting, classification, filter extraction)
- `Hybrid.ipynb` — Level 3: Hybrid search (Dense + BM25 + RRF)

## Setup

1. Clone the repository
2. Install dependencies:
3. Create .gitignore based on `.env` to protect Api_Key
4. Run the notebooks in order

## Data Source

Recipes scraped from [AllRecipes](https://www.allrecipes.com/)

## Technologies Used

- Google Gemini API
- FAISS vector store
- Sentence Transformers (all-MiniLM-L6-v2)
- BM25 (rank-bm25)
- Python, Jupyter Notebook