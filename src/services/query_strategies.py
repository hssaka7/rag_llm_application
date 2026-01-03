from abc import ABC, abstractmethod
from typing import Any

class QueryStrategy(ABC):
    @abstractmethod
    def query(self, collection, embedding: Any, top_k: int = 3) -> dict:
        pass

class ChromaDBQueryStrategy(QueryStrategy):
    def query(self, collection, embedding: Any, top_k: int = 3) -> dict:
        return collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
