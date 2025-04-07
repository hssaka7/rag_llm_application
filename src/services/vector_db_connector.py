import logging
import chromadb
import unicodedata
from sentence_transformers import SentenceTransformer

class ChromaDBInterface:
    def __init__(self, collection_name: str = "documents", vector_db_path="./chroma_db"):
        self.logger = logging.getLogger(__name__)
        self.client = chromadb.PersistentClient(path=vector_db_path)
        self.model = SentenceTransformer("BAAI/bge-m3")  # Multilingual model
        self.collection = self.client.get_or_create_collection(name=collection_name)

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
            embeddings = self.model.encode(
                new_contents,
                normalize_embeddings=True,
                batch_size=32,
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
        """Retrieves top-k similar documents along with metadata."""
        self.logger.info(f"Querying vector db for: {query_text}")
        
        # Normalize the query text
        query_text = unicodedata.normalize("NFKC", query_text)

        # Get embedding for query
        embedding = self.model.encode(query_text, normalize_embeddings=True).tolist()
        
        # Query the vector db
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        self.logger.info(f"Query Result: {results}")
        return results

# Example usage
if __name__ == "__main__":
    db = ChromaDBInterface()
    db.add_documents(["doc1"], ["Sample content for testing."], [{"author": "John Doe", "category": "Test"}])
    results = db.query("testing sample", top_k=2)
    print(results)
