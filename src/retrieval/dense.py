

from src.retrieval.base import RetrievalStrategy

class DenseRetrievalStrategy(RetrievalStrategy):
    def retrieve(self, collection, query_text, embedding_model, top_k, **kwargs):
        embedding = embedding_model.encode(query_text, normalize_embeddings=True).tolist()
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        return results