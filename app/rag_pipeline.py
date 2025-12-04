from typing import Dict, Any

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from app.config import (
    LLM_MODEL,
    OPENAI_API_KEY,
    USE_OPENAI_LLM,
    LOCAL_LLM_MODEL,
)
from vectorstore.store import load_faiss_index


class LocalChatLLM:
    """
    Simple local chat-style LLM wrapper using Hugging Face transformers.
    Provides an .invoke(prompt_str) method that returns an object with .content,
    similar to ChatOpenAI, so SimpleRAGChain can treat both the same way.
    """

    _generator = None

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device

        if LocalChatLLM._generator is None:
            print(f"🔁 Loading local HF model: {model_name} (this may take some time on first run)...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            LocalChatLLM._generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if device == "cuda" else -1,
            )

        self.generator = LocalChatLLM._generator

    def invoke(self, prompt: str):
        # Basic text-generation call; you can tune max_new_tokens, temperature, etc.
        outputs = self.generator(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.3,
        )
        full_text = outputs[0]["generated_text"]

        # Wrap in a simple object with `.content` so it matches ChatOpenAI usage
        return type("LocalResponse", (object,), {"content": full_text})()


class SimpleRAGChain:
    """
    Minimal RAG-style chain compatible with .invoke(...).

    Steps:
    - retrieves relevant docs
    - builds a prompt with context + question
    - calls the LLM (OpenAI or local HF)
    - returns answer + source_documents
    """

    def __init__(self, llm, retriever, prompt: PromptTemplate):
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Support both {"query": "..."} and {"question": "..."}
        question = inputs.get("query") or inputs.get("question")
        if not question:
            raise ValueError("SimpleRAGChain expects a 'query' or 'question' key.")

        # 1) Retrieve documents (retriever is a Runnable in LC v1)
        docs = self.retriever.invoke(question)

        # 2) Build context string
        context = "\n\n".join(d.page_content for d in docs)

        # 3) Format prompt
        prompt_str = self.prompt.format(context=context, question=question)

        # 4) Call LLM
        response = self.llm.invoke(prompt_str)
        answer = getattr(response, "content", str(response))

        # 5) Return in the same shape as RetrievalQA did
        return {
            "result": answer,
            "source_documents": docs,
        }


def get_llm():
    """
    Returns either:
    - ChatOpenAI (remote, needs OpenAI quota), or
    - LocalChatLLM (Hugging Face transformer, fully local)
    based on USE_OPENAI_LLM flag in config.
    """
    if USE_OPENAI_LLM:
        print("⚙ Using OpenAI Chat LLM...")
        return ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0.1,
        )
    else:
        print("⚙ Using local Hugging Face LLM...")
        return LocalChatLLM(
            model_name=LOCAL_LLM_MODEL,
            device="cpu",  # change to "cuda" if you have a GPU set up
        )


def get_retriever():
    vs = load_faiss_index()
    retriever = vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2},
    )
    return retriever


def get_qa_chain() -> SimpleRAGChain:
    llm = get_llm()
    retriever = get_retriever()

    template = """
You are a financial research assistant. Answer the question using only the provided context.
If the answer is not in the context, say you do not know.

Context:
{context}

Question:
{question}

Answer (be concise, structured, and analytical):
"""
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"],
    )

    return SimpleRAGChain(llm=llm, retriever=retriever, prompt=prompt)
