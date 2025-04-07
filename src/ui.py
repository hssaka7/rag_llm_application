import gradio as gr
import logging
import logging.config
import os
import pandas as pd

from dotenv import load_dotenv

from services.vector_db_connector import ChromaDBInterface  
from services.llm import GeminiService
from utils.utils import parse_yaml


# Environment setup
load_dotenv()

api_key = os.environ["GEMINI_API_KEY"]
system_prompt_path = os.environ["SYSTEM_PROMPT_PATH"]
chroma_db_path = os.environ["CHROMA_DB_PATH"]
logging_config_path = os.environ["LOGGER_FILE_PATH"]

logging.config.dictConfig(parse_yaml(logging_config_path))
logger = logging.getLogger(__name__)

gemini_service = GeminiService(api_key)
db = ChromaDBInterface(vector_db_path=chroma_db_path)

# set default base system prompt
base_system_prompt = ""
with open(system_prompt_path, 'r', encoding='utf-8') as fp:
    base_system_prompt = fp.read()


def add_documents(doc_id, content, metadata):
    if not doc_id or not content:
        return "Please provide both Document ID and Content."
    
    metadata_dict = {}
    try:
        metadata_dict = eval(metadata)
    except Exception as e:
        return f"Invalid metadata format. Error: {str(e)}"
    
    db.add_documents([doc_id], [content], [metadata_dict])
    return f"Document {doc_id} added successfully."

def save_system_prompt(system_prompt):
    global base_system_prompt

    try:
        base_system_prompt = system_prompt
        with open(system_prompt_path, 'w', encoding='utf-8') as fp:
            fp.writelines(system_prompt)
        return "✅ System prompt saved successfully."
    except Exception as e:
        logger.error(f"Error saving system prompt: {str(e)}")
        return f"❌ Failed to save system prompt. Error: {str(e)}"



def query_db(query_text, top_k):
    if not query_text:
        return pd.DataFrame(), "Please provide a query."
    
    result = db.query(query_text, int(top_k))
    
    doc_ids = result['ids'][0]
    distances = result['distances'][0]
    documents = result['documents'][0]
    metadatas = result['metadatas'][0]
    
    data = {
        "Document ID": doc_ids,
        "Score": distances,
        "Content": documents,
        "Metadata": [str(metadata) for metadata in metadatas],
    }
    df = pd.DataFrame(data)
    
    return df

def stream_summary(system_instruction, contents_df):
    
    if contents_df.empty:
        yield "No documents to summarize."
        return
    
    docs = contents_df["Content"].tolist()
    prompt = "Summarize the following documents briefly:\n" + "\n\n".join(docs)
    
    response_stream = gemini_service.generate_content_stream(
        model="gemini-2.0-flash",
        contents=[prompt],
        system_instruction= system_instruction,
    )
    
    for chunk in response_stream:
        yield chunk

with gr.Blocks() as app:
    gr.Markdown("## ChromaDB Document Management UI")
    
    with gr.Tab("Add Document"):
        doc_id = gr.Textbox(label="Document ID")
        content = gr.Textbox(label="Content", lines=5)
        metadata = gr.Textbox(label="Metadata (e.g., {'author': 'John', 'category': 'Tech'})", lines=2)
        add_btn = gr.Button("Add Document")
        add_output = gr.Textbox(label="Status")
        add_btn.click(add_documents, inputs=[doc_id, content, metadata], outputs=add_output)

    with gr.Tab("Prompts"):
        
        system_prompt_input = gr.Textbox(label="Edit System Prompt",
                                         lines=25,
                                         value=base_system_prompt,
                                         interactive=True)
        
        save_prompt_button = gr.Button("Save Prompt")
        save_prompt_output = gr.Textbox(label="Save Status", interactive=False)

        save_prompt_button.click(
            fn=save_system_prompt,
            inputs=[system_prompt_input],
            outputs=[save_prompt_output]
        )

        

    with gr.Tab("Query Database"):
        query_text = gr.Textbox(label="Query")
        top_k = gr.Slider(minimum=1, maximum=30, value=5, label="Top K")
        query_btn = gr.Button("Search")
        
        query_output = gr.DataFrame(label="Query Results")
        query_btn.click(query_db,
                        inputs=[query_text, top_k],
                        outputs=query_output)

        summarize_btn = gr.Button("Summarize Documents")
        summary_output = gr.Textbox(label="Summary", lines=30)

        summarize_btn.click(
            stream_summary,
            inputs=[system_prompt_input, query_output],
            outputs=summary_output
        )
    
    

app.launch()
