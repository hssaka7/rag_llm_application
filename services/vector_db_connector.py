
import logging
import chromadb
from sentence_transformers import SentenceTransformer




class ChromaDBInterface:
    def __init__(self, collection_name: str = "documents", vector_db_path = "./chroma_db"):
        
        self.logger = logging.getLogger(__name__)
        self.client = chromadb.PersistentClient(path=vector_db_path)
        
        self.model = SentenceTransformer("BAAI/bge-m3")  # Multilingual model
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, doc_ids: list, contents: list):
        """Cleans and adds a document with embeddings to ChromaDB."""
        
        self.logger.info(f"Adding {len(doc_ids)} documnets to vector_db ...")

       
        embeddings = self.model.encode(contents, 
                                       normalize_embeddings=True, 
                                       batch_size=32, 
                                       show_progress_bar=False).tolist()
             
        self.collection.add(ids=doc_ids, embeddings=embeddings, documents= contents)
        self.logger.info(f"Added {len(doc_ids)} documnets")


    def query(self, query_text: str, top_k: int = 3):
        """Retrieves top-k similar documents."""
        
        self.logger.info(f"Querying vector db for : {query_text} ")
        embedding = self.model.encode(query_text, normalize_embeddings=True).tolist()
        results = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        
        self.logger.info(f"Queryi Result is  : {results} /n")

        return results

# Example Usage
if __name__ == "__main__":
    chroma_db = ChromaDBInterface()
    
    # # Sample documents from your dataset
    # documents = [
    #     ("6453bcb39d7a959380bdd1be", "काठमाडौं । पछिल्लो सात कारोबार दिन लगातार घटेको नेपाल स्टक एक्सचेञ्ज (नेप्से) परिसूचकमा बुधबार भने दोहोरो अङ्कको वृद्धि देखिएको छ ।"),
    #     ("64757903f6642d53c2fe4872", "सञ्चार तथा सूचना प्रविधिअन्तर्गत ‘आधुनिक जीवनको आधार’ सूचना प्रविधि र सञ्चारको मान्यतालाई आत्मसात गर्दै नागरिकका लागि सूचना प्रविधिको प्रयोगलाई सरल, सहज र पहुँचयोग्य बनाइएको छ।"),
    #     ("6475ce38bd779c1442dc71dc", "नुवाकोट : मोटरसाइकल दुर्घटनामा नुवाकोटको दाङसिङमा युवकको मृत्यु भएको छ। तारकेश्वर गाउँपालिका-१ दाङसिंङका ३२ वर्षीय चतुरमान तामाङको शुक्रबार बेलुकी भएको दुर्घटनामा मृत्यु गएको जिल्ला प्रहरी कार्यालय नुवाकोटले जनाएको छ।")
    # ]
    
    # # Add documents to ChromaDB
    # for doc_id, content in documents:
    #     chroma_db.add_documents([doc_id], [content])
    
    # Query example
    query_result = chroma_db.query("Tell me about the information technology in Nepal", top_k=1)
    
    from  pprint import pprint
    pprint(query_result)
