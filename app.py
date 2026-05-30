import streamlit as st
from rag_pipeline import RAGPipeline

# Page config
st.set_page_config(
    page_title="HybridRAG — Search Comparison",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 HybridRAG — Semantic vs Keyword vs Hybrid Search")
st.markdown("Compare how different retrieval methods affect RAG answers.")

# Load pipeline once — cache so it doesn't reload every time
@st.cache_resource
def load_pipeline():
    pipeline = RAGPipeline()
    pipeline.load_documents("data/complaints.txt")
    return pipeline

pipeline = load_pipeline()

# Sidebar
st.sidebar.header("Settings")
k = st.sidebar.slider("Number of chunks to retrieve", 1, 5, 3)
st.sidebar.markdown("---")
st.sidebar.markdown("**How it works:**")
st.sidebar.markdown("- **Semantic** — finds meaning")
st.sidebar.markdown("- **Keyword** — finds exact words")
st.sidebar.markdown("- **Hybrid** — combines both")

# Query input
query = st.text_input(
    "Enter your query:",
    placeholder="e.g. payment declined, app crashing, account locked"
)

if query:
    st.markdown("---")

    # Run all three search types
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🧠 Semantic Search")
        result = pipeline.answer(query, search_type="semantic", k=k)

        st.markdown("**Answer:**")
        st.info(result["answer"])

        st.markdown("**Retrieved chunks:**")
        for i, (chunk, score) in enumerate(result["sources"], 1):
            st.markdown(f"**{i}.** Score: `{score}`")
            st.caption(chunk)

    with col2:
        st.subheader("🔤 Keyword Search (TF-IDF)")
        result = pipeline.answer(query, search_type="keyword", k=k)

        st.markdown("**Answer:**")
        st.info(result["answer"])

        st.markdown("**Retrieved chunks:**")
        for i, (chunk, score) in enumerate(result["sources"], 1):
            st.markdown(f"**{i}.** Score: `{score}`")
            st.caption(chunk)

    with col3:
        st.subheader("⚡ Hybrid Search")
        result = pipeline.answer(query, search_type="hybrid", k=k)

        st.markdown("**Answer:**")
        st.info(result["answer"])

        st.markdown("**Retrieved chunks:**")
        for i, (chunk, score) in enumerate(result["sources"], 1):
            st.markdown(f"**{i}.** Score: `{score}`")
            st.caption(chunk)

    st.markdown("---")
    st.caption("Hybrid search combines semantic (70%) and keyword (30%) for best results.")