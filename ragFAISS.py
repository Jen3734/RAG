"""
RAG pipeline: LangChain + FAISS + DeepSeek API.

Run (Windows):
  Double-click run_ragFAISS.bat
  OR:  .\\.venv\\Scripts\\python.exe ragFAISS.py

Do NOT use python3.exe on Windows (often a broken Store stub).

First-time setup:
  python -m venv .venv
  .venv\\Scripts\\pip install -r requirements.txt

Set DEEPSEEK_API_KEY and HF_TOKEN in the configuration section below.
"""

import os
import sys
from pathlib import Path

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import bs4
import faiss

# --- Configuration ---

DEEPSEEK_API_KEY = "sk-b818de2caa644d908f15e51241e79f48"
HF_TOKEN = "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"
DEEPSEEK_MODEL = "deepseek-chat"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
FAISS_INDEX_PATH = "faiss_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 4

DEFAULT_WEB_URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"
USER_AGENT = "ragFAISS/1.0 (local RAG demo)"

os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY
os.environ.setdefault("USER_AGENT", USER_AGENT)


def _log(msg: str) -> None:
    print(msg, flush=True)


def configure_hf_token() -> bool:
    """Authenticate Hugging Face Hub downloads with HF_TOKEN."""
    if not HF_TOKEN or HF_TOKEN == "your-hf-token-here":
        return False

    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

    try:
        from huggingface_hub import login

        login(token=HF_TOKEN, add_to_git_credential=False)
    except ImportError:
        pass

    return True


def build_llm() -> BaseChatModel:
    return init_chat_model(
        DEEPSEEK_MODEL,
        model_provider="deepseek",
        temperature=0.7,
        max_tokens=1024,
    )


def build_embeddings() -> HuggingFaceEmbeddings:
    configure_hf_token()
    model_kwargs = {}
    if HF_TOKEN and HF_TOKEN != "your-hf-token-here":
        model_kwargs["token"] = HF_TOKEN

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
    )


def load_and_split_documents(web_url: str = DEFAULT_WEB_URL):
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=(web_url,),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


def build_vector_store(
    splits,
    embeddings: HuggingFaceEmbeddings | None = None,
) -> FAISS:
    embeddings = embeddings or build_embeddings()
    embedding_dim = len(embeddings.embed_query("hello world"))
    index = faiss.IndexFlatL2(embedding_dim)
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_documents(documents=splits)
    return vector_store


def save_vector_store(vector_store: FAISS, path: str = FAISS_INDEX_PATH) -> None:
    vector_store.save_local(path)


def load_vector_store(
    path: str = FAISS_INDEX_PATH,
    embeddings: HuggingFaceEmbeddings | None = None,
) -> FAISS:
    embeddings = embeddings or build_embeddings()
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def build_rag_chain(vector_store: FAISS, llm: BaseChatModel | None = None):
    llm = llm or build_llm()
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    prompt = ChatPromptTemplate.from_template(
        "Answer the question using only the context below. "
        "If the context does not contain enough information, say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {input}"
    )
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)


def query(rag_chain, question: str) -> str:
    result = rag_chain.invoke({"input": question})
    return result["answer"]


def get_or_build_vector_store(embeddings: HuggingFaceEmbeddings) -> FAISS:
    index_dir = Path(FAISS_INDEX_PATH)
    if index_dir.is_dir() and (index_dir / "index.faiss").exists():
        _log(f"Loading existing FAISS index from {FAISS_INDEX_PATH}...")
        return load_vector_store(FAISS_INDEX_PATH, embeddings)

    _log("Loading and indexing documents (first run may take a few minutes)...")
    splits = load_and_split_documents()
    _log(f"Split into {len(splits)} chunks. Building embeddings...")
    vector_store = build_vector_store(splits, embeddings)
    save_vector_store(vector_store)
    _log(f"Saved FAISS index to {FAISS_INDEX_PATH}.")
    return vector_store


def main():
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your-deepseek-api-key-here":
        _log("Error: set DEEPSEEK_API_KEY in the configuration section.")
        sys.exit(1)

    if HF_TOKEN and HF_TOKEN != "your-hf-token-here":
        _log("Using HF_TOKEN for Hugging Face Hub downloads.")
    else:
        _log("HF_TOKEN not set; using anonymous Hub access (may be slower).")

    _log("Loading embedding model (downloads on first run)...")
    embeddings = build_embeddings()

    vector_store = get_or_build_vector_store(embeddings)

    _log("Connecting to DeepSeek API...")
    rag_chain = build_rag_chain(vector_store, build_llm())

    questions = [
        "What is the standard method for Task Decomposition?",
        "What are common extensions of that method?",
    ]
    for q in questions:
        _log(f"\nQ: {q}")
        _log(f"A: {query(rag_chain, q)}")


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        _log(f"\nMissing dependency: {exc}")
        _log("Install with: .venv\\Scripts\\pip install -r requirements.txt")
        _log("Or run: run_ragFAISS.bat")
        sys.exit(1)
