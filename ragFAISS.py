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
import uuid
from pathlib import Path

import faiss
import numpy as np

from searchEngine import RagSearchEngine
from tocLoader import TocEntry, TocLoader, format_toc_entries

SCRIPT_DIR = Path(__file__).resolve().parent
HF_AUTH_LOG = SCRIPT_DIR / "HF_auth.log"
DEEPSEEK_AUTH_LOG = SCRIPT_DIR / "deepseek_api_auth.log"

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
FAISS_IVF_NLIST = 110
FAISS_NPROBE = 32
CHUNK_SIZE = 200
CHUNK_OVERLAP = 30
RETRIEVAL_K = 8
TOC_TOPIC_K = 5
TOC_CANDIDATE_POOL = 100
CHUNK_HEAD_CHARS = 150
TOC_TEXT_SCAN_MAX_PAGES = 15
TOC_CONTENTS_KEYWORDS = ("table of contents", "contents", "chapter", "section")

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


def build_faiss_ivf_sq8_index(
    vectors: np.ndarray,
    *,
    nlist: int = FAISS_IVF_NLIST,
    nprobe: int = FAISS_NPROBE,
) -> faiss.Index:
    """Build a FAISS IVF index with 8-bit scalar quantization (IVF + SQ8)."""
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array")
    num_vectors, embedding_dim = vectors.shape
    if num_vectors == 0:
        raise ValueError("Need at least one vector to build the index")

    effective_nlist = min(nlist, num_vectors)
    if effective_nlist < nlist:
        _log(
            f"Clamping IVF nlist from {nlist} to {effective_nlist} "
            f"(only {num_vectors} vectors available)."
        )

    quantizer = faiss.IndexFlatL2(embedding_dim)
    index = faiss.IndexIVFScalarQuantizer(
        quantizer,
        embedding_dim,
        effective_nlist,
        faiss.ScalarQuantizer.QT_8bit,
    )

    _log(f"Training FAISS IVF{effective_nlist},SQ8 index on {num_vectors} vectors...")
    index.train(vectors)
    index.add(vectors)
    index.nprobe = min(nprobe, effective_nlist)
    _log(f"FAISS index ready (nlist={effective_nlist}, nprobe={index.nprobe}).")
    return index


def _configure_faiss_search(index: faiss.Index, nprobe: int = FAISS_NPROBE) -> None:
    """Set nprobe on IVF indexes when loading from disk."""
    if isinstance(index, faiss.IndexIVF):
        index.nprobe = min(nprobe, index.nlist)


def build_vector_store(
    splits,
    embeddings: HuggingFaceEmbeddings | None = None,
) -> FAISS:
    embeddings = embeddings or build_embeddings()
    texts = [doc.page_content for doc in splits]
    _log(f"Embedding {len(texts)} chunks for IVF,SQ8 index...")
    vectors = np.array(embeddings.embed_documents(texts), dtype=np.float32)

    index = build_faiss_ivf_sq8_index(vectors)

    docstore = InMemoryDocstore()
    index_to_docstore_id: dict[int, str] = {}
    for i, doc in enumerate(splits):
        doc_id = str(uuid.uuid4())
        index_to_docstore_id[i] = doc_id
        docstore.add({doc_id: doc})

    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )


def save_vector_store(vector_store: FAISS, path: str = FAISS_INDEX_PATH) -> None:
    vector_store.save_local(path)


def load_vector_store(
    path: str = FAISS_INDEX_PATH,
    embeddings: HuggingFaceEmbeddings | None = None,
) -> FAISS:
    embeddings = embeddings or build_embeddings()
    vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    _configure_faiss_search(vector_store.index)
    return vector_store


def load_document_toc(
    file_path: Path | str = DOCUMENT_FILE_PATH,
    *,
    preview_count: int = 12,
) -> tuple[list[TocEntry], str]:
    """Load PDF table of contents and print a preview."""
    path = Path(file_path)
    if path.suffix.lower() != ".pdf":
        _log("TOC loader skipped (document is not a PDF).")
        return [], "skipped"

    _log(f"Loading table of contents from {path.name}...")
    toc_loader = TocLoader(
        max_text_pages=TOC_TEXT_SCAN_MAX_PAGES,
        contents_keywords=TOC_CONTENTS_KEYWORDS,
        log_fn=_log,
    )
    toc_entries, toc_method = toc_loader.load(path)

    if toc_entries:
        _log(f"Table of contents preview ({toc_method}):")
        _log(format_toc_entries(toc_entries[:preview_count]))
        if len(toc_entries) > preview_count:
            _log(f"... and {len(toc_entries) - preview_count} more entries.")
    else:
        _log("No table of contents entries found.")

    return toc_entries, toc_method


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


def run_question_with_search_trace(
    search_engine: RagSearchEngine,
    question: str,
) -> str:
    """Run one question: print sub-queries, chunk heads, then return the answer."""
    sub_queries, doc_lists, merged_docs = search_engine.retrieve_all(question)
    search_engine.print_search_details(question, sub_queries, doc_lists)
    _log(f"Merged {len(merged_docs)} unique chunk(s) for final answer.")
    answer = search_engine.synthesize_answer(question, sub_queries, merged_docs)
    return answer


def main():
    if not DEEPSEEK_API_KEY:
        _log(f"Error: set DEEPSEEK_API_KEY in {DEEPSEEK_AUTH_LOG.name}.")
        sys.exit(1)

    if HF_TOKEN:
        _log(f"Using HF_TOKEN from {HF_AUTH_LOG.name}.")
    else:
        _log("HF_TOKEN not set; using anonymous Hub access (may be slower).")

    toc_entries, toc_method = load_document_toc()

    _log("Loading embedding model (downloads on first run)...")
    embeddings = build_embeddings()

    vector_store = get_or_build_vector_store(embeddings)

    _log("Connecting to DeepSeek API...")
    llm = build_llm()
    search_engine = RagSearchEngine(
        vector_store,
        llm,
        toc_entries=toc_entries,
        retrieval_k=RETRIEVAL_K,
        topic_k=TOC_TOPIC_K,
        toc_candidate_pool=TOC_CANDIDATE_POOL,
        chunk_head_chars=CHUNK_HEAD_CHARS,
        log_fn=_log,
    )

    questions = [
        "Explain what is the Backup Timer Indicator",
        "what to do if a ct scan is aborted?",
        "what is the difference between a scout and a primary scan?",
        "what is the difference between a GSI scan and a helical scan?",
        "what are the major axial scan rotation frequencies?",
    ]
    for q in questions:
        _log(f"\nQ: {q}")
        _log(f"A: {run_question_with_search_trace(search_engine, q)}")


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        _log(f"\nMissing dependency: {exc}")
        _log("Install with: .venv\\Scripts\\pip install -r requirements.txt")
        _log("Or run: run_ragFAISS.bat")
        sys.exit(1)
