"""
RAG pipeline: LangChain + FAISS + DeepSeek API.

Run (Windows):
  Double-click run_ragFAISS.bat
  OR:  .\\.venv\\Scripts\\python.exe ragFAISS.py

Do NOT use python3.exe on Windows (often a broken Store stub).

First-time setup:
  python -m venv .venv
  .venv\\Scripts\\pip install -r requirements.txt

API keys are read from HF_auth.log and deepseek_api_auth.log in this folder.
"""

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
HF_AUTH_LOG = SCRIPT_DIR / "HF_auth.log"
DEEPSEEK_AUTH_LOG = SCRIPT_DIR / "deepseek_api_auth.log"

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import faiss

# --- Configuration ---


def _load_secret_from_log(path: Path, name: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"{name} file is empty: {path}")
    if "=" in value and not value.startswith(("hf_", "sk-")):
        value = value.split("=", 1)[1].strip()
    return value


HF_TOKEN = _load_secret_from_log(HF_AUTH_LOG, "HF_TOKEN")
DEEPSEEK_API_KEY = _load_secret_from_log(DEEPSEEK_AUTH_LOG, "DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = "deepseek-chat"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
FAISS_INDEX_PATH = "faiss_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 4

DOCUMENT_FILE_PATH = Path(
    r"C:\Users\jenni\Documents\books\rag"
    r"\Revolution Apex  Revolution CT with Apex edition User Manual "
    r"25MW27_UM_6789300-1EN_2.pdf"
)

os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY


def _log(msg: str) -> None:
    print(msg, flush=True)


def configure_hf_token() -> bool:
    """Authenticate Hugging Face Hub downloads with HF_TOKEN."""
    if not HF_TOKEN:
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
    if HF_TOKEN:
        model_kwargs["token"] = HF_TOKEN

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
    )


def load_documents(file_path: Path | str = DOCUMENT_FILE_PATH):
    """Load a single document from a fixed file path."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Document file not found: {path.resolve()}")

    ext = path.suffix.lower()
    if ext == ".pdf":
        docs = PyPDFLoader(str(path)).load()
    elif ext in {".txt", ".md", ".markdown"}:
        docs = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Use .pdf, .txt, .md, or .markdown."
        )

    for doc in docs:
        doc.metadata["source"] = str(path)

    _log(f"Loaded {len(docs)} page(s) from {path.name}")
    return docs


def load_and_split_documents(file_path: Path | str = DOCUMENT_FILE_PATH):
    docs = load_documents(file_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    splits = splitter.split_documents(docs)
    _log(f"Split into {len(splits)} chunks.")
    return splits


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

    _log(f"Loading and indexing document: {DOCUMENT_FILE_PATH.name}...")
    splits = load_and_split_documents()
    _log("Building embeddings...")
    vector_store = build_vector_store(splits, embeddings)
    save_vector_store(vector_store)
    _log(f"Saved FAISS index to {FAISS_INDEX_PATH}.")
    return vector_store


def main():
    if not DEEPSEEK_API_KEY:
        _log(f"Error: set DEEPSEEK_API_KEY in {DEEPSEEK_AUTH_LOG.name}.")
        sys.exit(1)

    if HF_TOKEN:
        _log(f"Using HF_TOKEN from {HF_AUTH_LOG.name}.")
    else:
        _log("HF_TOKEN not set; using anonymous Hub access (may be slower).")

    _log("Loading embedding model (downloads on first run)...")
    embeddings = build_embeddings()

    vector_store = get_or_build_vector_store(embeddings)

    _log("Connecting to DeepSeek API...")
    rag_chain = build_rag_chain(vector_store, build_llm())

    questions = [
        "Explain what is the Backup Timer Indicator",
        "what to do if a ct scan is aborted?",
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
