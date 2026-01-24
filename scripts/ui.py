import gradio as gr

import sys
import os

from sentence_transformers import CrossEncoder

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



from src.agents.summary_agent import SummaryAgent
from src.services.vector_store import get_vector_store
from src.retrieval.advance import AdvanceRetrievalStrategy
from src.services.llm.ollama import OllamaService



vector_sercice = get_vector_store(llm=OllamaService(),
								   reranker_model=CrossEncoder("cross-encoder/stsb-roberta-base"))
llm_servicce = OllamaService()
summary_agent = SummaryAgent(vector_db=vector_sercice,
                             llm_service=llm_servicce)

prompt_context = {k:{} for k in summary_agent.prompt_file_map.keys()}

with gr.Blocks(gr.themes.Ocean()) as app:
	gr.Markdown("## Milvus Document Management UI")

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
		chatbot = gr.Chatbot()
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

