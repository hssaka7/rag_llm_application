
import logging


from abc import ABC, abstractmethod


# Abstract base class for vector store
class VectorStore(ABC):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def add_documents(self, doc_ids: list, contents: list, metadatas: list = None):
        pass

    @abstractmethod
    def query(self, query_text: str, top_k: int = 3):
        pass

    @abstractmethod
    def doc_exists(self, doc_id: str) -> bool:
        pass