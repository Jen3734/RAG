"""Shared FAISS IVF+SQ8 index helpers."""

from __future__ import annotations

import uuid
from collections.abc import Callable

import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def configure_faiss_search(index: faiss.Index, nprobe: int) -> None:
    """Set nprobe on IVF indexes when loading from disk."""
    if isinstance(index, faiss.IndexIVF):
        index.nprobe = min(nprobe, index.nlist)


def build_faiss_ivf_sq8_index(
    vectors: np.ndarray,
    *,
    nlist: int,
    nprobe: int,
    log_fn: Callable[[str], None],
) -> faiss.Index:
    """Build a FAISS IVF index with 8-bit scalar quantization (IVF + SQ8)."""
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array")
    num_vectors, embedding_dim = vectors.shape
    if num_vectors == 0:
        raise ValueError("Need at least one vector to build the index")

    effective_nlist = min(nlist, num_vectors)
    if effective_nlist < nlist:
        log_fn(
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

    log_fn(f"Training FAISS IVF{effective_nlist},SQ8 index on {num_vectors} vectors...")
    index.train(vectors)
    index.add(vectors)
    index.nprobe = min(nprobe, effective_nlist)
    log_fn(f"FAISS index ready (nlist={effective_nlist}, nprobe={index.nprobe}).")
    return index


def build_faiss_vector_store(
    documents: list[Document],
    vectors: np.ndarray,
    embedding_function,
    *,
    nlist: int,
    nprobe: int,
    log_fn: Callable[[str], None],
) -> FAISS:
    """Build a LangChain FAISS store from precomputed vectors and documents."""
    index = build_faiss_ivf_sq8_index(
        vectors,
        nlist=nlist,
        nprobe=nprobe,
        log_fn=log_fn,
    )

    docstore = InMemoryDocstore()
    index_to_docstore_id: dict[int, str] = {}
    for i, doc in enumerate(documents):
        doc_id = str(uuid.uuid4())
        index_to_docstore_id[i] = doc_id
        docstore.add({doc_id: doc})

    return FAISS(
        embedding_function=embedding_function,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )


def load_faiss_vector_store(
    path: str,
    embedding_function,
    *,
    nprobe: int,
) -> FAISS:
    vector_store = FAISS.load_local(
        path,
        embedding_function,
        allow_dangerous_deserialization=True,
    )
    configure_faiss_search(vector_store.index, nprobe)
    return vector_store


def save_faiss_vector_store(vector_store: FAISS, path: str) -> None:
    vector_store.save_local(path)
