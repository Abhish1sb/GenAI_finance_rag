from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from app.config import (
    VECTORSTORE_PATH,
    USE_OPENAI_EMBEDDINGS,
    EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_API_KEY,
)


def get_embeddings():
    """
    Dynamically chooses between Hugging Face embeddings (local)
    and OpenAI embeddings (cloud) based on USE_OPENAI_EMBEDDINGS flag.
    """

    if USE_OPENAI_EMBEDDINGS:
        print("🔄 Using OpenAI Embeddings...")
        return OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
        )

    else:
        print("🔄 Using Hugging Face Embeddings...")
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},  # change to "cuda" if needed
        )


def build_faiss_index(docs):
    embeddings = get_embeddings()
    return FAISS.from_documents(docs, embeddings)


def save_faiss_index(vs, path: Path = VECTORSTORE_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(path))


def load_faiss_index(path: Path = VECTORSTORE_PATH):
    embeddings = get_embeddings()
    return FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True
    )
