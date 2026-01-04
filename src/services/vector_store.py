
import chromadb
import logging
import unicodedata

from abc import ABC, abstractmethod
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.services.llm import LLMService

# Retrieval strategy interface
class RetrievalStrategy(ABC):
    @abstractmethod
    def retrieve(self, collection, query_text, embedding_model, top_k, **kwargs):
        """
        kwargs can include llm, reranker_model, or any other future parameters.
        """
        pass

# Default dense retrieval strategy
class DenseRetrievalStrategy(RetrievalStrategy):
    def retrieve(self, collection, query_text, embedding_model, top_k, **kwargs):
        embedding = embedding_model.encode(query_text, normalize_embeddings=True).tolist()
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        return results

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


# Advance retrieval strategy stub (for future extension)
class AdvanceRetrievalStrategy(RetrievalStrategy):
    def retrieve(self,
                 collection,
                 query_text:str,
                 embedding_model:SentenceTransformer,
                 top_k:int,
                 reranker_model=None,
                 llm:LLMService=None, 
                 **kwargs):
        

        # Step 1: Query expansion using LLM
        # Use a prompt template for expansion
        expansion_prompt = kwargs.get('expansion_prompt',
            """You are an expert information retrieval assistant specialized in news and current events.
               Your task is to expand a user query into multiple high-quality search queries for retrieving 
               relevant news documents.
               Follow these rules strictly:
                - Preserve the original intent of the query
                - Do NOT introduce unrelated topics
                - Include entity expansions (organizations, people, locations)
                - Include paraphrases and alternative phrasings
                - Include relevant abbreviations, do not expand them.
                - Include time-aware variations when applicable
                - Only prodive the expnded query, not any headers or metadata
            \n\nUser Query: {query}\n\nExpanded Queries:"""
        )
        expanded_query = query_text
        if llm is not None:
            try:
                prompt = expansion_prompt.format(query=query_text)
                # LLMService returns a generator, join all tokens to get the expanded query
                expanded_query = "".join(llm.generate_content_stream(prompt)).strip()
                # Use only the first line if LLM returns a long response
                # expanded_query = expanded_query.split("\n")[0].strip() if expanded_query else query_text
            except Exception as e:
                # Fallback to original query if expansion fails
                expanded_query = query_text

        # Step 2: Embed the expanded query
        embedding = embedding_model.encode(expanded_query, normalize_embeddings=True).tolist()

        # Step 3: Perform dense retrieval
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k * 2 if reranker_model is not None else top_k,  # retrieve more for reranking
            include=["documents", "metadatas", "distances"]
        )

        # Step 4: Rerank results using reranker_model (if provided)
        if reranker_model is not None and 'documents' in results:
            ids = results['ids'][0]
            docs = results['documents'][0]
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0]
            # Prepare pairs for reranker: (query, doc)
            pairs = [(expanded_query, doc) for doc in docs]
            try:
                scores = reranker_model.predict(pairs)
                # Sort by reranker score (descending)
                reranked = sorted(zip(scores, ids, docs, metadatas, distances), key=lambda x: x[0], reverse=True)
                # Take top_k
                reranked = reranked[:top_k]
                # Rebuild results dict
                results['ids'][0] = [_id for _, _id, _, _, _ in reranked]
                results['documents'][0] = [doc for _, _, doc, _, _ in reranked]
                if 'metadatas' in results:
                    results['metadatas'][0] = [meta for _, _, _, meta, _ in reranked]
                if 'distances' in results:
                    # Optionally, set reranked distances to None or keep original order
                    results['distances'][0] = [dis for _, _, _, _, dis in reranked]
            except Exception as e:
                # If reranking fails, fallback to dense retrieval order
                pass

        return results
    



# Abstract base class for vector store
class VectorStore(ABC):
    @abstractmethod
    def add_documents(self, doc_ids: list, contents: list, metadatas: list = None):
        pass

    @abstractmethod
    def query(self, query_text: str, top_k: int = 3):
        pass




class ChromaDBInterface(VectorStore):
    def __init__(
        self,
        collection_name: str = "documents",
        vector_db_path: str = "./chroma_db",
        embedding_model=None,
        client=None,
        retrieval_strategy: RetrievalStrategy = None,
        llm=None,
        reranker_model=None
    ):
        self.logger = logging.getLogger(__name__)
        # Allow dependency injection for embedding model and client
        if client is not None:
            self.client = client
        else:
            self.client = chromadb.PersistentClient(path=vector_db_path)

        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = SentenceTransformer("BAAI/bge-m3")  # Multilingual model

        self.collection = self.client.get_or_create_collection(name=collection_name)

        # Allow pluggable retrieval strategy
        if retrieval_strategy is not None:
            self.retrieval_strategy = retrieval_strategy
        else:
            self.retrieval_strategy = DenseRetrievalStrategy()

        self.llm = llm
        self.reranker_model = reranker_model

    def add_documents(self, doc_ids: list, contents: list, metadatas: list = None):
        """Cleans and adds documents with embeddings and metadata to ChromaDB."""
        self.logger.info(f"Adding {len(doc_ids)} documents to vector_db...")

        if not doc_ids or not contents:
            self.logger.warning("Empty doc_ids or contents provided. Nothing to add.")
            return

        # Normalize content to handle Unicode characters
        contents = [unicodedata.normalize("NFKC", content) for content in contents]

        try:
            # Check existing IDs
            existing_ids = set(self.collection.get(ids=doc_ids)["ids"])
            new_doc_pairs = [
                (doc_id, content, metadata)
                for doc_id, content, metadata in zip(doc_ids, contents, metadatas or [{}] * len(doc_ids))
                if doc_id not in existing_ids
            ]

            if not new_doc_pairs:
                self.logger.info("No new documents to add. All documents already exist.")
                return

            new_doc_ids, new_contents, new_metadatas = zip(*new_doc_pairs)

            # Generate embeddings
            embeddings = self.embedding_model.encode(
                new_contents,
                normalize_embeddings=True,
                batch_size=16,
                show_progress_bar=False
            ).tolist()

            # Add documents with metadata
            self.collection.add(
                ids=list(new_doc_ids),
                embeddings=embeddings,
                documents=list(new_contents),
                metadatas=list(new_metadatas)
            )
            self.logger.info(f"Added {len(new_doc_ids)} new documents to vector_db.")

        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}", exc_info=True)

    def query(self, query_text: str, top_k: int = 3):
        """Retrieves top-k similar documents along with metadata using the configured retrieval strategy."""
        self.logger.info(f"Querying vector db for: {query_text}")
        
        # Normalize the query text
        query_text = unicodedata.normalize("NFKC", query_text)

        # Use the retrieval strategy
        results = self.retrieval_strategy.retrieve(
            self.collection,
            query_text,
            self.embedding_model,
            top_k,
            llm=self.llm,
            reranker_model=self.reranker_model
        )

        self.logger.info(f"Query Result: {results}")
        return results

# Example usage
if __name__ == "__main__":
    # You can swap retrieval strategies here
    db = ChromaDBInterface(retrieval_strategy=DenseRetrievalStrategy())
    db.add_documents(["doc1"], ["Sample content for testing."], [{"author": "John Doe", "category": "Test"}])
    results = db.query("testing sample", top_k=2)
    print(results)
