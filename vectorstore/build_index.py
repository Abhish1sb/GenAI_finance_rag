from ingestion.preprocess_docs import load_raw_docs, split_docs
from vectorstore.store import build_faiss_index, save_faiss_index


def main():
    docs = load_raw_docs()
    print(f"Loaded {len(docs)} raw docs")
    if not docs:
        print("No documents found in data/raw. Add some .pdf or .txt files.")
        return

    chunks = split_docs(docs)
    print(f"Created {len(chunks)} chunks")
    if not chunks:
        print("No chunks created. Check if documents have readable text.")
        return

    vs = build_faiss_index(chunks)
    save_faiss_index(vs)
    print("FAISS index built and saved successfully.")


if __name__ == "__main__":
    main()
