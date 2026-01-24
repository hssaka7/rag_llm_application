


# RAG-based LLM Document Querying System

This project is a Retrieval-Augmented Generation (RAG) system for querying documents using Large Language Models (LLMs). It leverages **Milvus** vector database for efficient document retrieval and integrates with Ollama for advanced local LLM querying and reporting.

---

## Features

- **Document Ingestion & ETL:** Load and preprocess documents into a vector database.
- **RAG Chatbot:** Query documents using LLM-powered natural language interface.
- **Customizable Prompts:** Use prompt templates for different reporting and detection tasks.
- **Modular Design:** Easily extend with new agents, services, or data sources.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/rag_llm_application.git
cd rag_llm_application
```


### 2. Install and Run Ollama

Ollama is required to run local LLMs (e.g., Llama 3, Mistral, etc.).

**Install Ollama:**

- [Ollama Download & Install Guide](https://ollama.com/download)

**Start Ollama:**

```bash
ollama serve
```

**Pull a Model (e.g., Llama 3):**

```bash
ollama pull llama3
```

You can use any supported model (see [Ollama Models](https://ollama.com/library)).

### 3. Install and Run Milvus

Milvus is used as the vector database for document storage and retrieval.

**Option 1: Using Docker (Recommended)**

```bash
# Install Docker if not already installed
# Then run Milvus standalone server
docker run -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest standalone
```

**Option 2: Using Milvus Lite (Local File-based)**

For development, you can use Milvus Lite which stores data locally without a server:

```bash
pip install milvus-lite
```

Then set `MILVUS_URI = './data/milvus.db'` in your `.env` file.

### 4. Create and Activate a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt --no-cache-dir
```

---

## Usage

### 1. Start Milvus (if using Docker)

If you're using Docker for Milvus, make sure the container is running:

```bash
docker run -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest standalone
```

### 2. Run the ETL Process (Load Data)

```bash
python scripts/etl.py
```

### 3. Start the Querying UI

```bash
python scripts/ui.py
```

---

## Project Structure

- `src/` - Main source code
	- `agents/` - RAG chatbot and agent logic
	- `services/` - LLM and **Milvus** vector DB connectors
	- `utils/` - Utility functions
- `scripts/` - ETL pipeline and UI scripts
	- `etl.py` - ETL pipeline for data ingestion
	- `ui.py` - User interface for querying
- `prompts/` - Prompt templates for various tasks
- `configs/` - Configuration files (e.g., logging)
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation

---

## Configuration

- **Ollama Model:** By default, the UI uses the `llama3` model. You can change this in the code or by pulling a different model with Ollama.
- **Milvus Database:** Configure the Milvus connection in `.env`:
  - `MILVUS_URI`: Set to `'http://localhost:19530'` for Docker server or `'./data/milvus.db'` for Milvus Lite
- **Logging:** See `configs/logging_config.yaml` for logging setup.
- **Prompts:** Customize prompt templates in the `prompts/` directory.

---
