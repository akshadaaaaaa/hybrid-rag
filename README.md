# HybridRAG — Semantic vs Keyword vs Hybrid Search

🚀 **Live Demo:** [Click here](https://hybrid-rag-fuv5tf4tbewedulenlld37.streamlit.app/)

A RAG system that compares three retrieval methods side by side.

## What it does
- Semantic search using FAISS + sentence transformers
- Keyword search using TF-IDF
- Hybrid search combining both (70% semantic + 30% keyword)
- LLM answer generation using Groq Llama 3
- Streamlit UI for visual comparison

## Real challenge solved
Semantic search misses exact technical terms.
Keyword search misses meaning.
Hybrid search catches both.

## Retrieval Evaluation

Evaluated on a ground-truth set of 10 queries (K=3):

| Method   | Precision@3 | MRR  | NDCG@3 |
|----------|-------------|------|--------|
| Semantic | 0.533*      | 1.00 | 0.992  |
| Keyword  | 0.400       | 1.00 | 1.000  |
| Hybrid   | 0.533*      | 1.00 | 0.992  |

\*0.533 is the maximum achievable P@3 on this ground truth — most queries
have fewer than 3 relevant chunks, so semantic and hybrid retrieved every
relevant chunk available.

**Key finding:** on a small, clean corpus the 70/30 hybrid weighting is
dominated by semantic scores. Next steps: larger/noisier corpus, harder
eval queries, Reciprocal Rank Fusion, and cross-encoder reranking.

## Tech stack
Python · FAISS · scikit-learn · LangChain ·
sentence-transformers · Groq · Streamlit

## How to run
1. Clone the repo
2. pip install -r requirements.txt
3. Add your GROQ_API_KEY to .env file
4. streamlit run app.py

## Try these queries
- "payment declined"
- "login problem"
- "account locked"
- "wrong charge"
