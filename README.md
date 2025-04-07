

# RAG based llm document querying system

- open .env and replace the value for GEMINI_API_KEY by actual gemini api key and save the file.


- Install dependencies: 
```
python -m virtualenv .venv
source .venv/bin/activate
pip install -r "requirements.txt" --no-cache-dir
```

- run Vector db ETL process to load the data.

```
python src/etl.py
```

- To run the vector db querying process:

```
python src/ui.py
```