import os
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
VECTORSTORE_PATH = DATA_PROCESSED_DIR / "faiss_index"

# Embedding switches
USE_OPENAI_EMBEDDINGS = False  # True = OpenAI, False = HuggingFace

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# LLM switches
USE_OPENAI_LLM = False  # True = ChatOpenAI (needs quota), False = local HF model
LLM_MODEL = "gpt-4.1-mini"  # used only if USE_OPENAI_LLM = True

# Small local chat model (CPU-friendly-ish)
LOCAL_LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
