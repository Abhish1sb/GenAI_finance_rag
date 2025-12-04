from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_pipeline import get_qa_chain

app = FastAPI(title="GenAI Financial Research Assistant")
qa_chain = get_qa_chain()

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask(req: QueryRequest):
    result = qa_chain.invoke({"query": req.query})
    answer = result["result"]
    sources = [
        {"source": d.metadata.get("source"), "snippet": d.page_content[:200]}
        for d in result["source_documents"]
    ]
    return {"answer": answer, "sources": sources}
