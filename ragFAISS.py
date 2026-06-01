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

import numpy as np

from faissIndex import (
    build_faiss_vector_store,
    load_faiss_vector_store,
    save_faiss_vector_store,
)
from imageEmbedder import ImageEmbedder
from searchEngine import RagSearchEngine
from textEmbedder import TextEmbedder
from tocLoader import TocLoader, format_toc_entries

SCRIPT_DIR = Path(__file__).resolve().parent
HF_AUTH_LOG = SCRIPT_DIR / "HF_auth.log"
DEEPSEEK_AUTH_LOG = SCRIPT_DIR / "deepseek_api_auth.log"

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
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
CLIP_MODEL = "sentence-transformers/clip-ViT-B-32"
FAISS_INDEX_PATH = "faiss_index"
FAISS_IMAGE_INDEX_PATH = "faiss_image_index"
EXTRACTED_IMAGE_DIR = "extracted_images"
TEXT_FAISS_NLIST = 110
TEXT_FAISS_NPROBE = 32
IMAGE_FAISS_NLIST = 10
IMAGE_FAISS_NPROBE = 32
CHUNK_SIZE = 200
CHUNK_OVERLAP = 30
RETRIEVAL_K = 2
IMAGE_RETRIEVAL_K = 4
MIN_IMAGE_SIZE = 64
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


def build_llm() -> BaseChatModel:
    return init_chat_model(
        DEEPSEEK_MODEL,
        model_provider="deepseek",
        temperature=0.7,
        max_tokens=1024,
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
        doc.metadata.setdefault("modality", "text")

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


def get_or_build_text_vector_store(text_embedder: TextEmbedder):
    index_dir = Path(FAISS_INDEX_PATH)
    if index_dir.is_dir() and (index_dir / "index.faiss").exists():
        _log(f"Loading existing text FAISS index from {FAISS_INDEX_PATH}...")
        return load_faiss_vector_store(
            FAISS_INDEX_PATH,
            text_embedder.langchain_embeddings,
            nprobe=TEXT_FAISS_NPROBE,
        )

    _log(f"Loading and indexing document: {DOCUMENT_FILE_PATH.name}...")
    splits = load_and_split_documents()
    texts = [doc.page_content for doc in splits]
    _log(f"Embedding {len(texts)} text chunks...")
    vectors = np.array(text_embedder.embed_documents(texts), dtype=np.float32)

    vector_store = build_faiss_vector_store(
        splits,
        vectors,
        text_embedder.langchain_embeddings,
        nlist=TEXT_FAISS_NLIST,
        nprobe=TEXT_FAISS_NPROBE,
        log_fn=_log,
    )
    save_faiss_vector_store(vector_store, FAISS_INDEX_PATH)
    _log(f"Saved text FAISS index to {FAISS_INDEX_PATH}.")
    return vector_store


def get_or_build_image_vector_store(image_embedder: ImageEmbedder):
    index_dir = Path(FAISS_IMAGE_INDEX_PATH)
    if index_dir.is_dir() and (index_dir / "index.faiss").exists():
        _log(f"Loading existing image FAISS index from {FAISS_IMAGE_INDEX_PATH}...")
        return image_embedder.load_vector_store(
            FAISS_IMAGE_INDEX_PATH,
            nprobe=IMAGE_FAISS_NPROBE,
        )

    if DOCUMENT_FILE_PATH.suffix.lower() != ".pdf":
        _log("Image index skipped (document is not a PDF).")
        return None

    _log(f"Extracting and indexing images from: {DOCUMENT_FILE_PATH.name}...")
    vector_store = image_embedder.build_vector_store(
        nlist=IMAGE_FAISS_NLIST,
        nprobe=IMAGE_FAISS_NPROBE,
    )
    if vector_store is None:
        return None

    image_embedder.save_vector_store(vector_store, FAISS_IMAGE_INDEX_PATH)
    _log(f"Saved image FAISS index to {FAISS_IMAGE_INDEX_PATH}.")
    return vector_store


def load_document_toc(
    file_path: Path | str = DOCUMENT_FILE_PATH,
    *,
    preview_count: int = 12,
):
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


def run_question_with_search_trace(
    search_engine: RagSearchEngine,
    question: str,
) -> str:
    """Run one question: print sub-queries, chunk heads, then return the answer."""
    sub_queries, doc_lists, merged_docs = search_engine.retrieve_all(question)
    search_engine.print_search_details(question, sub_queries, doc_lists)
    _log(f"Merged {len(merged_docs)} unique chunk(s) for final answer.")
    return search_engine.synthesize_answer(question, sub_queries, merged_docs)


def main():
    if not DEEPSEEK_API_KEY:
        _log(f"Error: set DEEPSEEK_API_KEY in {DEEPSEEK_AUTH_LOG.name}.")
        sys.exit(1)

    if HF_TOKEN:
        _log(f"Using HF_TOKEN from {HF_AUTH_LOG.name}.")
    else:
        _log("HF_TOKEN not set; using anonymous Hub access (may be slower).")

    toc_entries, _toc_method = load_document_toc()

    _log("Loading text embedding model...")
    text_embedder = TextEmbedder(model_name=EMBEDDING_MODEL, hf_token=HF_TOKEN)
    text_vector_store = get_or_build_text_vector_store(text_embedder)

    _log("Loading CLIP image embedder...")
    image_embedder = ImageEmbedder(
        DOCUMENT_FILE_PATH,
        clip_model=CLIP_MODEL,
        min_image_size=MIN_IMAGE_SIZE,
        extracted_image_dir=EXTRACTED_IMAGE_DIR,
        log_fn=_log,
    )
    image_vector_store = get_or_build_image_vector_store(image_embedder)

    _log("Connecting to DeepSeek API...")
    llm = build_llm()
    search_engine = RagSearchEngine(
        text_vector_store,
        llm,
        toc_entries=toc_entries,
        image_vector_store=image_vector_store,
        retrieval_k=RETRIEVAL_K,
        image_retrieval_k=IMAGE_RETRIEVAL_K,
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
