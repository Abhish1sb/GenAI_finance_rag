from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from app.config import DATA_RAW_DIR


def load_raw_docs():
    docs = []

    # Load text files
    for path in Path(DATA_RAW_DIR).glob("*.txt"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs.append(Document(page_content=text, metadata={"source": path.name}))

    # Load PDF files
    for path in Path(DATA_RAW_DIR).glob("*.pdf"):
        loader = PyPDFLoader(str(path))
        pdf_pages = loader.load()
        for page in pdf_pages:
            # Ensure metadata includes the filename
            page.metadata["source"] = path.name
            docs.append(page)

    return docs


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=400
    )
    return splitter.split_documents(docs)
