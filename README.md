# GenAI Finance Project

A local-first RAG based assistant for finance.  
This project lets you:

- Ingest financial documents (annual reports, PDFs, research notes)
- Build a FAISS vector index over them
- Query the documents using a local Hugging Face LLM (TinyLlama) through a FastAPI backend
- Optionally schedule automated ingestion with Airflow

---

## Tech Stack

- Python 3.10+
- FastAPI + Uvicorn
- Hugging Face Transformers and Embeddings
- FAISS for vector search
- Airflow for scheduled ingestion
- Local Hugging Face model: TinyLlama/TinyLlama-1.1B-Chat-v1.0

---

## Project Structure

```text
genai_finance_project/
├─ airflow/
│  └─ dags/
│     └─ financial_rag_ingestion.py      # Airflow DAG to automate ingestion and indexing
├─ app/
│  ├─ config.py                          # App configuration (paths, model names, env)
│  ├─ main.py                            # FastAPI entrypoint
│  └─ rag_pipeline.py                    # Core RAG pipeline (retrieve + generate)
├─ data/
│  ├─ raw/                               # Raw documents (PDFs, TXT, etc.)
│  └─ processed/                         # Optional preprocessed / cleaned text
├─ ingestion/
│  └─ preprocess_docs.py                 # Script to clean and chunk documents
├─ vectorstore/
│  ├─ build_index.py                     # Script to build the FAISS index
│  └─ store.py                           # Vector store helper utilities
├─ requirements.txt
├─ .gitattributes
├─ .gitignore
└─ README.md
