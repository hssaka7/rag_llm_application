
import logging

from abc import ABC, abstractmethod

# Retrieval strategy interface
class RetrievalStrategy(ABC):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def retrieve(self, collection, query_text, embedding_model, top_k, **kwargs):
        """
        kwargs can include llm, reranker_model, or any other future parameters.
        """
        pass