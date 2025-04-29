import gradio as gr
import logging
import logging.config
import os
import pandas as pd

import random
import time

from dotenv import load_dotenv

from agents.rag_chatbot import graph, memory
from services.vector_db_connector import ChromaDBInterface  
from services.llm import GeminiService
from utils.utils import parse_yaml


# Environment setup
load_dotenv()


logging_config_path = os.environ["LOGGER_FILE_PATH"]
logging.config.dictConfig(parse_yaml(logging_config_path))
logger = logging.getLogger(__name__)

class SummaryAgent:
    def __init__(self):
        self.logger = logger
        self.system_prompt_path = os.environ["SYSTEM_PROMPT_PATH"]
        self.llm_api_key = os.environ["GEMINI_API_KEY"]
        self.chroma_db_path = os.environ["CHROMA_DB_PATH"]
       
        self.gemini_service = GeminiService(self.llm_api_key)

        self.chat = graph
        self.chat_memory = memory

        self.vector_db = ChromaDBInterface(vector_db_path=self.chroma_db_path)
        

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

        response_stream = self.gemini_service.generate_content_stream(
            model="gemini-2.0-flash",
            contents=[prompt],
            system_instruction= agent_role,
        )
        
        for chunk in response_stream:
            yield chunk
    
    def stream_chat(self, chat_history):
        
        config = {"configurable": {"thread_id": "abc123"}}
        user_message = chat_history[-2]["content"]
        
        agent_response = self.chat.invoke({"messages": [ {"role": "user", "content": user_message } ]},config=config)
        
        for chunks in agent_response["messages"][-1].content:
            yield chunks
    
    def clear_chat_memory(self):
        self.chat_memory.delete_thread("abc123")

        

summary_agent = SummaryAgent()
prompt_context = {k:{} for k in summary_agent.prompt_file_map.keys()}

with gr.Blocks(gr.themes.Ocean()) as app:
    gr.Markdown("## ChromaDB Document Management UI")
    
    with gr.Tab("Add Document"):
        doc_id = gr.Textbox(label="Document ID")
        content = gr.Textbox(label="Content", lines=5)
        metadata = gr.Textbox(label="Metadata (e.g., {'author': 'John', 'category': 'Tech'})", lines=2)
        add_btn = gr.Button("Add Document")
        add_output = gr.Textbox(label="Status")
        add_btn.click(summary_agent.add_documents, inputs=[doc_id, content, metadata], outputs=add_output)

    with gr.Tab("Prompts"):
        
        for prompt_name, prompt_filepath in summary_agent.prompt_file_map.items():
            with gr.Tab(prompt_name):
                prompt_context[prompt_name]["prompt_input"] = gr.Textbox(
                    label=f"Edit: {prompt_name}",
                    lines=25,
                    value=summary_agent.read_prompt(prompt_filepath),
                    interactive=True
                )
                prompt_context[prompt_name]["prompt_file_location"]=gr.Textbox(
                    label = f"{prompt_name} File path",
                    value = prompt_filepath,
                    interactive=False
                )
                prompt_context[prompt_name]["save_button"] = gr.Button(f"Save {prompt_name}")
                prompt_context[prompt_name]["save_output"] = gr.Textbox(label="Save Status", interactive=False)
                    
                prompt_context[prompt_name]["save_button"].click(
                        summary_agent.save_prompt,
                        inputs= [prompt_context[prompt_name]["prompt_file_location"],
                                prompt_context[prompt_name]["prompt_input"]],
                        outputs=[prompt_context[prompt_name]["save_output"]]
                )

    with gr.Tab("Query Database"):
        query_text = gr.Textbox(label="Query")
        top_k = gr.Number(minimum=1, maximum=100, value=10, label="Top K")
        query_btn = gr.Button("Search")
        
        query_output = gr.DataFrame(label="Query Results")
        query_btn.click(summary_agent.query_db,
                        inputs=[query_text, top_k],
                        outputs=query_output)
        
        prompt_dropdown = gr.Dropdown(
            choices=list(summary_agent.prompt_file_map.keys()),
            label="Select a System Prompt",
            value=list(summary_agent.prompt_file_map.keys())[0],  # default to first
        )
        

        summarize_btn = gr.Button("Summarize Documents")
        summary_output = gr.Textbox(label="Summary", lines=30)

        summarize_btn.click(
            summary_agent.stream_summary,
            inputs=[prompt_dropdown, query_output],
            outputs=summary_output
        )
     
    with gr.Tab("Chat"):
        chatbot = gr.Chatbot(type="messages")
        msg = gr.Textbox()
        clear_button = gr.Button("Clear")

        def user_request(msg, chat_history):
            if msg:

                chat_history.append({"role": "user", "content": msg})
                return "", chat_history
            
        def bot_response(chat_history:list):

            # generate AI response
            bot_message = summary_agent.stream_chat(chat_history)
           
            chat_history.append({"role": "assistant", "content": ""})
            for character in bot_message:
                chat_history[-1]['content'] += character
                
                yield chat_history
            

    msg.submit(user_request, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_response, chatbot, chatbot
    )

    clear_button.click(
        lambda: summary_agent.clear_chat_memory(), None, chatbot, queue=False
    )




app.launch(pwa=True)
