import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss

class HybridSearch:
    """
    Combines semantic search (FAISS + Embeddings)
    with keyword search (TF-IDF) for better retrieval
    """

    def __init__(self, model_name:str="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name) #ek isse search karenge
        self.tfidf = TfidfVectorizer()  #ek isse search karenge
        self.chunks = []
        self.index = None
        self.tfidf_matrix = None

    #Training both

    def add_documents(self,chunks:list[str])-> None:
        """Index documents for both semantic an dkeyword search
        Args:
            chunks :List of text chunks to index
        """
        if not chunks:
            raise ValueError("Chunks cannot be empty")

        self.chunks = chunks

        #Semantic index - FAISS - train 1 #Librarian 1
        embeddings = self.model.encode(chunks) #converts all chunks to vectors --> 10 chunks → 10 vectors of 384 numbers each.
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings, dtype= np.float32))

        #Keyword index - TF IDF - train 2 #Librarian 2
        self.tfidf_matrix = self.tfidf.fit_transform(chunks)

        print(f"Indexed {len(chunks)} chunks")

#Librarian 1- searches seperately
    def semantic_search(self, query:str, k: int = 3)-> list[tuple[str,float]]:
        """Search using vector embeddings- finds semantically similar chunks

        Args:
            query : Search query
            k : Number of results. Defaults to 3.

        Returns:
            list of (chunk, score) tuples

        """

        query_embedding = self.model.encode([query])
        distances,indices = self.index.search(
            np.array(query_embedding,dtype=np.float32), k
        )

        results = []
        for dist, idx in zip(distances[0],indices[0]):
            if idx != -1:
                #Convert L2 distance to similarity score
                score = 1/(1+dist)
                results.append((self.chunks[idx],float(score)))

        return results

#Librarian 2- searches seperately

    def keyword_search(self,query:str,k:int=3)-> list[tuple[str,float]]:
        """Search using TF-IDF - finds chunks with matching keywords

        Args:
            query (str): Search query
            k : Number of results. Defaults to 3.

        Returns:
            list of (chunk, score) tuples
        """
        query_vec = self.tfidf.transform([query])
        scores = cosine_similarity(query_vec,self.tfidf_matrix).flatten()

        top_indices = np.argsort(scores)[::-1][:k]

        return [
            (self.chunks[i], float(scores[i]))
            for i in top_indices
            if scores[i] >0
        ]


    def hybrid_search(
        self,
        query:str,
        k: int =3,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> list[tuple[str,float]]:
        """Combines semantic and keyword search results
        using weighted scoring

        Args:
            query : search query
            k : Number of results to return
            semantic_weight: Weight of semantic search. Defaults to 0.7.
            keyword_weight: Weight of keyword search. Defaults to 0.3

        Returns:
            list of (chunk, combined score) tuples sorted by relevance
        """

        semantic_results = self.semantic_search(query,k)
        keyword_results = self.keyword_search(query,k)

        #combine scores using a dictionary
        combined = {}

        for chunk, score in semantic_results:
            combined[chunk] = combined.get(chunk,0) + score * semantic_weight

        for chunk, score in keyword_results:
            combined[chunk] = combined.get(chunk,0) + score * keyword_weight

        #sort by combined score descending
        sorted_results = sorted(
            combined.items(),
            key = lambda x: x[1],
            reverse=True
        )

        return sorted_results[:k]