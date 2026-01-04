
from sentence_transformers import SentenceTransformer

from src.services.llm.base import LLMService
from src.retrieval.base import RetrievalStrategy

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
        self.logger.info(f"Expanded Query: {expanded_query}")
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