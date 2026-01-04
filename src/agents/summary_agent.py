import os
import logging
import logging.config
import pandas as pd
from dotenv import load_dotenv

from src.services.vector_store.base import VectorStore
from src.services.llm.base import LLMService
from src.utils.utils import parse_yaml
from src.agents.rag_chatbot import graph, memory

# Environment setup
load_dotenv()

logging_config_path = os.environ["LOGGER_FILE_PATH"]
logging.config.dictConfig(parse_yaml(logging_config_path))
logger = logging.getLogger(__name__)



class SummaryAgent:
    def __init__(self,
        vector_db:VectorStore,
        llm_service:LLMService,):
        
        self.logger = logger
        
        self.vector_db = vector_db
        self.llm_service = llm_service

        self.chat = graph
        self.chat_memory = memory

        self.prompt_file_map ={
                "Misinformation Detector Prompt": "prompts/misinformation_detector.md",
                "Social Media reporter Prompt": "prompts/social_media_reporter.md",
                "Website Reporter Prompt": "prompts/website_reporter.md",
                "Story Builder Prompt": "prompts/story_builder.md"
        }
        self.prompt_map = {}

    def read_prompt(self, prompt_path:str):
        with open(prompt_path, 'r', encoding='utf-8') as fp:
            base_system_prompt = fp.read()
            return base_system_prompt
    
    def save_prompt(self, prompt_path:str, prompt:str):
        try:
            with open(prompt_path, 'w', encoding='utf-8') as fp:
                fp.writelines(prompt)
            return "✅ System prompt saved successfully."
        except Exception as e:
            logger.error(f"Error saving system prompt: {str(e)}")
            return f"❌ Failed to save system prompt. Error: {str(e)}"
    
    def add_documents(self, doc_id, content, metadata):
        if not doc_id or not content:
            return "❌ Please provide both Document ID and Content."
        
        metadata_dict = {}
        try:
            metadata_dict = eval(metadata)
        except Exception as e:
            return f"❌ Invalid metadata format. Error: {str(e)}"
        
        self.vector_db.add_documents([doc_id], [content], [metadata_dict])
        return f"✅ Document {doc_id} added successfully."

    def query_db(self, query_text, top_k):
        if not query_text:
            return pd.DataFrame(), "❌ Please provide a query."
        
        result = self.vector_db.query(query_text, int(top_k))
        
        doc_ids = result['ids'][0]
        distances = result['distances'][0]
        documents = result['documents'][0]
        metadatas = result['metadatas'][0]
        
        titles = [x.get("encoded_title", "") for x in metadatas]
        published_dates = [x.get("published_date", "") for x in metadatas]
        source_names = [x.get("source_name", "") for x in metadatas]
        source_urls = [x.get("source_url", "") for x in metadatas]

        data = {
            "Document ID": doc_ids,
            "Score": distances,
            "Content": documents,
            "Title": titles,
            "Published Date": published_dates,
            "Sources":source_names,
            "Source URL": source_urls,
        }
        df = pd.DataFrame(data)
        
        return df

    def stream_summary(self, prompt_name,contents_df):
        agent_role = self.read_prompt(self.prompt_file_map[prompt_name])
        if contents_df.empty:
            yield "No documents to summarize."
            return
        
        prompt = f"## Below are the {len(contents_df)} documents to summarize:\n"
        for i, content in enumerate(contents_df.to_dict('records')):
            title = content["Title"]
            published_date = content["Published Date"]
            source_name = content["Sources"]
            source_url = content["Source URL"]
            document = content["Content"]
            prompt += f"{i+1}. "
            prompt += f"Title: {title}\n"
            prompt += f"Content: {document}\n"
            prompt += f"Published Date: {published_date}\n"
            prompt += f"Source Name: {source_name} "
            prompt += f"Source URL: {source_url}\n\n"

        response_stream = self.llm_service.generate_content_stream(
            prompt=prompt,
            system_instruction=agent_role
        )
        summary = ""
        for chunk in response_stream:
            summary += chunk
            yield summary
    
    def stream_chat(self, chat_history):
        config = {"configurable": {"thread_id": "abc123"}}
        user_message = chat_history[-2]["content"]
        agent_response = self.chat.invoke({"messages": [ {"role": "user", "content": user_message } ]},config=config)
        for chunks in agent_response["messages"][-1].content:
            yield chunks
    
    def clear_chat_memory(self):
        self.chat_memory.delete_thread("abc123")