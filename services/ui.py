import gradio as gr
from vector_db_connector import ChromaDBInterface  # Ensure this file contains your ChromaDBInterface


db = ChromaDBInterface(vector_db_path="./data/chroma_db")

def add_documents(doc_id, content):
    if not doc_id or not content:
        return "Please provide both Document ID and Content."
    db.add_documents([doc_id], [content])
    return f"Document {doc_id} added successfully."

def query_db(query_text, top_k):
    if not query_text:
        return "Please provide a query."
    result = db.query(query_text, int(top_k))
    response = "\n\n".join([f"Doc ID: {doc_id}\nScore: {score}\nContent: {text}"
                                for doc_id, score, text in zip(result['ids'][0], result['distances'][0], result['documents'][0])])
    return response or "No results found."

with gr.Blocks() as app:
    gr.Markdown("## ChromaDB Document Management UI")
    
    with gr.Tab("Add Document"):
        doc_id = gr.Textbox(label="Document ID")
        content = gr.Textbox(label="Content", lines=5)
        add_btn = gr.Button("Add Document")
        add_output = gr.Textbox(label="Status")
        add_btn.click(add_documents, inputs=[doc_id, content], outputs=add_output)
    
    with gr.Tab("Query Database"):
        query_text = gr.Textbox(label="Query")
        top_k = gr.Slider(minimum=1, maximum=10, value=3, label="Top K")
        query_btn = gr.Button("Search")
        query_output = gr.Textbox(label="Results", lines=10)
        query_btn.click(query_db, inputs=[query_text, top_k], outputs=query_output)
    
app.launch()