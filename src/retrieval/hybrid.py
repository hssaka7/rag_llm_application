
from src.retrieval.base import RetrievalStrategy

# Hybrid retrieval strategy stub (for future extension)
class HybridRetrievalStrategy(RetrievalStrategy):
    def retrieve(self, collection, query_text, embedding_model, top_k, **kwargs):
        # TODO: Implement hybrid logic (e.g., combine dense and keyword search)
        # For now, fallback to dense retrieval
        embedding = embedding_model.encode(query_text, normalize_embeddings=True).tolist()
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        return results