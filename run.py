
import logging
import logging.config
import math
import os
import pandas as pd
import re
import time
import unicodedata

from bs4 import BeautifulSoup
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

from services.vector_db_connector import ChromaDBInterface
from utils import parse_yaml


class ETL:

    def __init__(self, raw_filepath, vector_db_path):
        self.logger = logging.getLogger(__name__)
        self.raw_filepath = raw_filepath
        self.vector_db_path = vector_db_path
        self.vector_db_conn = ChromaDBInterface(vector_db_path = self.vector_db_path)

        transform_folder = os.path.join('data', 'transform')
        if not os.path.exists(transform_folder):
            os.makedirs(transform_folder)
        
        self.embeddings_path = os.path.join(transform_folder, 'transform_{batch_number}.json')
        
    
    @staticmethod
    def clean_sentence(text):
        # TODO Remove HTML tags
        # text = BeautifulSoup(text, "html.parser").get_text()

        text = unicodedata.normalize("NFKC", text)   # Normalize Unicode characters 
        text = re.sub(r"\s+", " ", text).strip() # remove unwanted spaces
        text = re.sub(r'\[.*?\]', '', text).strip() # Remove square brackets and everything inside them
        text = re.sub(r'\s*\.\s*\.\s*\.\s*$', '', text).strip() # Remove trailing dots with spaces (e.g., " . . .", " . .. ")
        return text
    
    
    @staticmethod
    def extract_raw_data(file_path):
        df = pd.read_json(file_path)
        return df
        
        

    def transform_raw_data(self, df, batch_size = 1000 ):

        df["_id"] = df['_id'].apply(lambda x: x['$oid'])
        df["description"] = df["description"].apply(lambda x: None if x=='' else x)
        # Taking only the news with descriptions.
        df = df[df["description"].notna()]

        df["encoded_description"] = df["description"].apply(ETL.clean_sentence)

        return df[['_id', 'encoded_description']]

   
    @staticmethod
    def create_embeddings(doc_ids: list, contents:list):
        model = SentenceTransformer("BAAI/bge-m3")
        embeddings = model.encode(contents, 
                                normalize_embeddings=True, 
                                batch_size=32, 
                                show_progress_bar=False).tolist()
        
        return embeddings


    def load_data(self, df, batch_size = 100):
        # TODO load the data to vector db
        total_batch = math.ceil(len(df)/ batch_size)
        for start in range(0, len(df), batch_size):
            
            batch_number = start // batch_size + 1
            batch = df.iloc[start : start + batch_size]
            
            start_time = time.time()
            self.logger.info(f"Inserting batch {batch_number} / {total_batch} with {len(batch)} records...")
            
            self.vector_db_conn.add_documents(
                doc_ids=batch["_id"].astype(str).tolist(),
                contents=batch["encoded_description"].astype(str).tolist()
            )

            time_taken = round( (time.time() - start_time), 2 )
            self.logger.info(f"Batch {batch_number} / {batch_size} completed in {time_taken} sec")
            
        self.logger.info("All batch completed")


    def run(self):

        self.logger.info("Running ETL for creating embeddings")
        
        self.logger.info("Extracting raw data ... ")
        raw_df = self.extract_raw_data(self.raw_filepath)
        self.logger.info(f"Raw data extracted. Raw data shape = {raw_df.shape}")
        
        self.logger.info("Trasforming raw data ...")
        transform_df = self.transform_raw_data(raw_df)
        self.logger.info(f"Trasformation completed. Transformed  data shape = {raw_df.shape}")
        
        self.logger.info("Loading raw data ...")
        response = self.load_data(transform_df)
        self.logger.info(f"Load response is {response}")
        





if __name__ == "__main__":
    
    logger_file_path = 'logging_config.yaml'
    raw_file_path = 'data/news.json'
    chroma_db_path = './data/chroma_db'

    # set loggers
    logging.config.dictConfig(parse_yaml(logger_file_path))



    
    etl_runner = ETL(raw_filepath= raw_file_path,
                     vector_db_path=chroma_db_path)
    
    etl_runner.run()


  