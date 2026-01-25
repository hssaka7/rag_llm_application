import os

import logging
import unicodedata
from typing import List, Dict, Any, Optional
from pymilvus import MilvusClient, DataType

from sentence_transformers import SentenceTransformer

from src.services.vector_store.base import VectorStore


class MilvusDBInterface(VectorStore):
    def __init__(
        self,
        collection_name: str = "documents",
        uri: str = None,
        embedding_model=None,
        client=None,
        llm=None,
        reranker_model=None
    ):
        """
        Initialize Milvus vector database interface.

        Args:
            collection_name: Name of the collection
            uri: Milvus connection URI (file path for local or connection string for remote)
            embedding_model: Sentence transformer model for embeddings
            client: Pre-initialized Milvus client
            llm: LLM service for advanced retrieval
            reranker_model: Model for reranking results
        """
        super().__init__()

        if uri is None:
            uri = os.getenv("MILVUS_URI")
            if uri is None:
                raise ValueError("MILVUS_URI environment variable must be set")

        # Allow dependency injection for embedding model and client
        if client is not None:
            self.client = client
        else:
            # Initialize MilvusClient
            self.client = MilvusClient(uri=uri)

        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = SentenceTransformer("BAAI/bge-m3")  # Multilingual model

        self.collection_name = collection_name
        self.uri = uri

        self.llm = llm
        self.reranker_model = reranker_model

        # Create collection if it doesn't exist
        self._create_collection()

    def _create_collection(self):
        """Create collection with schema if it doesn't exist."""
        try:
            # Check if collection exists
            if self.client.has_collection(self.collection_name):
                self.logger.info(f"Collection '{self.collection_name}' already exists.")
                return

            # Create collection with schema
            schema = self.client.create_schema(
                auto_id=False,
                enable_dynamic_field=True
            )

            # Add fields
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=255, is_primary=True)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)  # BAAI/bge-m3 has 1024 dimensions
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)

            # Create index parameters
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="FLAT",
                metric_type="COSINE"
            )

            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )

            self.logger.info(f"Created collection '{self.collection_name}' successfully.")

        except Exception as e:
            self.logger.error(f"Failed to create collection: {e}", exc_info=True)
            raise

    def add_documents(self, doc_ids: List[str], contents: List[str], metadatas: List[Dict] = None):
        """Add documents with embeddings and metadata to Milvus."""
        self.logger.info(f"Adding {len(doc_ids)} documents to Milvus...")

        if not doc_ids or not contents:
            self.logger.warning("Empty doc_ids or contents provided. Nothing to add.")
            return

        if len(doc_ids) != len(contents):
            raise ValueError("doc_ids and contents must have the same length")

        # Normalize content to handle Unicode characters
        contents = [unicodedata.normalize("NFKC", content) for content in contents]

        try:
            # Prepare metadata
            if metadatas is None:
                metadatas = [{}] * len(doc_ids)

            # Generate embeddings
            embeddings = self.embedding_model.encode(
                contents,
                normalize_embeddings=True,
                batch_size=16,
                show_progress_bar=False
            ).tolist()

            # Prepare data for insertion
            data = []
            for doc_id, content, embedding, metadata in zip(doc_ids, contents, embeddings, metadatas):
                # Merge metadata with required fields
                entity = {
                    "id": doc_id,
                    "vector": embedding,
                    "text": content,
                    **metadata  # Add dynamic metadata fields
                }
                data.append(entity)

            # Insert data
            res = self.client.insert(
                collection_name=self.collection_name,
                data=data
            )

            self.logger.info(f"Inserted {len(data)} documents into Milvus. Insert count: {res['insert_count']}")

        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}", exc_info=True)
            raise

    def query(self, query_text: str, top_k: int = 3):
        """Query the vector database."""
        self.logger.info(f"Querying Milvus for: {query_text}")

        # Normalize the query text
        query_text = unicodedata.normalize("NFKC", query_text)

        # Generate embedding
        embedding = self.embedding_model.encode(
            query_text,
            normalize_embeddings=True
        ).tolist()

        # Search Milvus
        results = self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            anns_field="vector",
            search_params={"metric_type": "COSINE"},
            limit=top_k,
            output_fields=["text", "encoded_title", "published_date", "source_name", "source_url"]
        )

        # Format results to match expected format
        ids = [hit["id"] for hit in results[0]]
        documents = [hit["entity"]["text"] for hit in results[0]]
        metadatas = [{k: v for k, v in hit["entity"].items() if k not in ["text", "vector"]} for hit in results[0]]
        distances = [hit["distance"] for hit in results[0]]

        formatted_results = {
            "ids": [ids],
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances]
        }

        self.logger.info(f"Query Result length: {len(documents)}")
        return formatted_results

    def doc_exists(self, doc_id: str) -> bool:
        """Check if a document exists in the collection."""
        try:
            # Query for the document by id
            results = self.client.query(
                collection_name=self.collection_name,
                filter=f"id == '{doc_id}'",
                limit=1,
                output_fields=["id"]
            )
            return len(results) > 0
        except Exception as e:
            self.logger.error(f"Failed to check if document exists: {e}")
            return False

    def similarity_search(self, query: str, k: int = 4):
        """
        Perform similarity search and return Document objects compatible with LangChain.

        Args:
            query (str): The query text.
            k (int): Number of top results to return.

        Returns:
            List[Document]: List of Document objects.
        """
        from langchain_core.documents import Document

        results = self.query(query, k)
        documents = []
        for doc_text, metadata in zip(results["documents"][0], results["metadatas"][0]):
            documents.append(Document(page_content=doc_text, metadata=metadata))
        return documents
