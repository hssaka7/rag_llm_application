import os
from typing import Optional

from .milvus import MilvusDBInterface

__all__ = ["MilvusDBInterface", "get_vector_store"]


def get_vector_store(
    llm=None,
    reranker_model=None
):
    """
    Factory function to get the Milvus vector store instance.

    Args:
        llm: LLM service for advanced retrieval
        reranker_model: Reranker model

    Returns:
        Milvus vector store instance
    """
    return MilvusDBInterface(
        llm=llm,
        reranker_model=reranker_model
    )