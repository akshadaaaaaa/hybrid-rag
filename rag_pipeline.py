import os
from groq import Groq
from dotenv import load_dotenv
from hybrid_search import HybridSearch

load_dotenv()

class RAGPipeline:
    """Connects hybrid_Search to LLM.
    Retrieves relevant chunks then generates grounded answer
    """

    def __init__(self):
        self.searcher = HybridSearch()
        self.client = Groq(api_key = os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def load_documents(self, filepath: str) -> None:
        """Load and index documents from a text file

        Args:
            filepath : Path to text file with one chunk per line
        """
        with open(filepath,"r") as f:
            chunks = [line.strip() for line in f if line.strip()]

        self.searcher.add_documents(chunks)
        print(f"Loaded {len(chunks)} documents")

    def answer(
        self,
        query: str,
        search_type: str ="hybrid",
        k: int=3
    )-> dict:
        """
        Answer a query using RAG pipeline.

        Args:
            query: User question
            search_type: "semantic", "keyword", or "hybrid"
            k: Number of chunks to retrieve

        Returns:
            Dictionary with answer, sources, and search type used

        """
        if not query:
            raise ValueError("Query cannot be empty")

        #Step 1: Retriev relevant chunks

        if search_type == "semantic":
            results = self.searcher.semantic_search(query,k)
        elif search_type == "keyword":
            results = self.searcher.keyword_search(query,k)
        else:
            results = self.searcher.hybrid_search(query,k)


        #Step 2: Build context from retrived chunks

        context = "\n".join([chunk for chunk,score in results])
        sources = [(chunk, round(score,3)) for chunk, score in results]

        #Step 3: Generate answer using LLM
        prompt = f"""You are a customer support analyst

    Answer the question using ONLY the context below.
    If the answer is not in the context, say "I don't have that information."

    Context:
    {context}

Question: {query}

Answer: """
        response = self.client.chat.completions.create(
                model= self.model,
                messages = [{"role":"user","content":prompt}],
                temperature = 0
            )

        answer = response.choices[0].message.content

        return {
            "query": query,
            "search_type": search_type,
            "answer": answer,
            "sources": sources
        }