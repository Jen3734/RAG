"""
Multi-query RAG search engine with query decomposition, TOC-guided retrieval,
and final answer synthesis.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from tocLoader import TocEntry

DEFAULT_RETRIEVAL_K = 8
DEFAULT_IMAGE_RETRIEVAL_K = 4
DEFAULT_TOC_TOPIC_K = 5
DEFAULT_TOC_CANDIDATE_POOL = 100
DEFAULT_CHUNK_HEAD_CHARS = 150

QUERY_DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You analyze user questions for document search and split them into focused "
            "search queries (one per line).\n"
            "Rules:\n"
            "- If the question contains multiple independent sub-questions, split each into "
            "a separate search query.\n"
            "- If the question contains undefined or unclear terminology, add a separate "
            "sub-query asking what that term means (e.g. 'What is [term]?').\n"
            "- If the question includes an assumption, add a separate sub-query to search "
            "for evidence about that assumption.\n"
            "- Generate sub-queries with different meanings and search angles; do not repeat "
            "or rephrase the same query in different words.\n"
            "- Always include the original question intent as at least one search query.\n"
            "Return ONLY the search queries, one per line, with no numbering or bullets.",
        ),
        ("human", "Question: {query}\n\nSearch queries:"),
    ]
)

FINAL_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user's question using only the retrieved context below. "
            "If the context does not contain enough information, say you don't know. "
            "When multiple sub-questions were searched, address each part clearly.",
        ),
        (
            "human",
            "Original question: {question}\n\n"
            "Sub-queries used for retrieval:\n{sub_queries}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Answer:",
        ),
    ]
)

TOC_SELECT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You match user questions to manual table-of-contents entries. "
            "Pick the entries most likely to contain the answer. "
            "Return ONLY comma-separated line numbers from the list (e.g. 3,7,12). "
            "No explanation.",
        ),
        (
            "human",
            "Question: {query}\n\n"
            "Pick up to {k} most relevant TOC entries:\n"
            "{toc_list}\n\n"
            "Line numbers:",
        ),
    ]
)


def _default_log(msg: str) -> None:
    print(msg, flush=True)


def decompose_query(query: str, llm: BaseChatModel) -> list[str]:
    """
    Decide whether a query should be split into multiple search queries.

    Returns one or more focused queries for retrieval.
    """
    chain = QUERY_DECOMPOSE_PROMPT | llm
    response = chain.invoke({"query": query})
    content = response.content if isinstance(response.content, str) else str(response.content)

    sub_queries = []
    for line in content.splitlines():
        line = re.sub(r"^[\s\-*\d.)]+", "", line.strip())
        if line and len(line) > 2:
            sub_queries.append(line)

    if not sub_queries:
        return [query.strip()]

    seen: set[str] = set()
    unique: list[str] = []
    for q in sub_queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            unique.append(q)

    return unique if len(unique) > 1 else [query.strip()]


def _prefilter_toc_candidates(
    toc_entries: list[TocEntry],
    query: str,
    max_candidates: int = DEFAULT_TOC_CANDIDATE_POOL,
) -> list[tuple[int, TocEntry]]:
    """Narrow TOC to a candidate pool before LLM selection."""
    query_terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
    scored: list[tuple[int, int, TocEntry]] = []

    for idx, entry in enumerate(toc_entries):
        title_lower = entry["title"].lower()
        score = sum(1 for term in query_terms if term in title_lower)
        if score:
            scored.append((score, idx, entry))

    scored.sort(key=lambda item: (-item[0], item[1]))
    candidates = [(idx, entry) for _, idx, entry in scored[:max_candidates]]

    if len(candidates) < max_candidates:
        seen = {idx for idx, _ in candidates}
        for idx, entry in enumerate(toc_entries):
            if idx in seen:
                continue
            if entry["level"] <= 2:
                candidates.append((idx, entry))
            if len(candidates) >= max_candidates:
                break

    return candidates[:max_candidates]


def _parse_toc_line_numbers(response: str, max_index: int) -> list[int]:
    numbers = []
    for part in re.findall(r"\d+", response):
        num = int(part)
        if 1 <= num <= max_index and num not in numbers:
            numbers.append(num)
    return numbers


def select_related_toc_topics(
    toc_entries: list[TocEntry],
    query: str,
    llm: BaseChatModel,
    k: int = DEFAULT_TOC_TOPIC_K,
    *,
    max_candidates: int = DEFAULT_TOC_CANDIDATE_POOL,
    log_fn: Callable[[str], None] = _default_log,
) -> list[TocEntry]:
    """Use the LLM to pick up to k TOC topics most related to the query."""
    if not toc_entries or k <= 0:
        return []

    candidates = _prefilter_toc_candidates(toc_entries, query, max_candidates)
    if not candidates:
        return []

    toc_list = "\n".join(
        f"{line_num}. {entry['title']}"
        + (f" (p. {entry['page']})" if entry["page"] is not None else "")
        for line_num, (_, entry) in enumerate(candidates, start=1)
    )

    chain = TOC_SELECT_PROMPT | llm
    response = chain.invoke({"query": query, "k": k, "toc_list": toc_list})
    content = response.content if isinstance(response.content, str) else str(response.content)

    selected_lines = _parse_toc_line_numbers(content, len(candidates))[:k]
    if not selected_lines:
        log_fn("LLM returned no TOC matches; using top keyword candidates.")
        return [entry for _, entry in candidates[:k]]

    return [candidates[line - 1][1] for line in selected_lines]


def _augment_query_with_toc_topics(query: str, topics: list[TocEntry]) -> str:
    if not topics:
        return query
    hints = "\n".join(
        f"- {entry['title']}"
        + (f" (page {entry['page']})" if entry["page"] is not None else "")
        for entry in topics
    )
    return (
        f"{query}\n\n"
        "Focus retrieval on these manual sections from the table of contents:\n"
        f"{hints}"
    )


def _format_chunk_head(doc: Document, max_chars: int = 150) -> str:
    """Return a one-line preview of a document chunk."""
    text = " ".join(doc.page_content.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _format_documents_for_prompt(docs: list[Document]) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        if doc.metadata.get("modality") == "image":
            image_path = doc.metadata.get("image_path", "unknown")
            blocks.append(
                f"[{i}] Image source: {source}\n"
                f"Image file: {image_path}\n"
                f"{doc.page_content}"
            )
        else:
            blocks.append(f"[{i}] Source: {source}\n{doc.page_content}")
    return "\n\n".join(blocks)


class RagSearchEngine:
    """Multi-query RAG retrieval with TOC guidance and final answer synthesis."""

    def __init__(
        self,
        vector_store: FAISS,
        llm: BaseChatModel,
        *,
        toc_entries: list[TocEntry] | None = None,
        image_vector_store: FAISS | None = None,
        retrieval_k: int = DEFAULT_RETRIEVAL_K,
        image_retrieval_k: int = DEFAULT_IMAGE_RETRIEVAL_K,
        topic_k: int = DEFAULT_TOC_TOPIC_K,
        toc_candidate_pool: int = DEFAULT_TOC_CANDIDATE_POOL,
        chunk_head_chars: int = DEFAULT_CHUNK_HEAD_CHARS,
        log_fn: Callable[[str], None] | None = None,
    ):
        self.vector_store = vector_store
        self.image_vector_store = image_vector_store
        self.llm = llm
        self.toc_entries = toc_entries or []
        self.retrieval_k = retrieval_k
        self.image_retrieval_k = image_retrieval_k
        self.topic_k = topic_k
        self.toc_candidate_pool = toc_candidate_pool
        self.chunk_head_chars = chunk_head_chars
        self._log = log_fn or _default_log

    def decompose_query(self, question: str) -> list[str]:
        return decompose_query(question, self.llm)

    def print_search_details(
        self,
        question: str,
        sub_queries: list[str],
        doc_lists: list[list[Document]],
    ) -> None:
        """Print separated search queries and retrieved chunk heads for one question."""
        self._log(f"\n--- Search details: {question} ---")
        self._log(f"Sub-queries ({len(sub_queries)}):")
        for i, sub_q in enumerate(sub_queries, start=1):
            self._log(f"  [{i}] {sub_q}")

        for sub_q, docs in zip(sub_queries, doc_lists):
            self._log(f"\nRetrieved chunks for: {sub_q}")
            if not docs:
                self._log("  (no chunks retrieved)")
                continue
            for j, doc in enumerate(docs, start=1):
                page = doc.metadata.get("page", doc.metadata.get("page_number", "?"))
                modality = doc.metadata.get("modality", "text")
                head = _format_chunk_head(doc, self.chunk_head_chars)
                if modality == "image":
                    image_path = doc.metadata.get("image_path", "unknown")
                    self._log(f"  image chunk {j} (page {page}): {head} [{image_path}]")
                else:
                    self._log(f"  chunk {j} (page {page}): {head}")

    def retrieve_all(
        self,
        question: str,
    ) -> tuple[list[str], list[list[Document]], list[Document]]:
        """Decompose question, retrieve per sub-query, and merge unique chunks."""
        sub_queries = self.decompose_query(question)
        doc_lists = [self.retrieve(q) for q in sub_queries]
        merged_docs = self._merge_documents(doc_lists)
        return sub_queries, doc_lists, merged_docs

    def _augment_search_query(self, search_query: str) -> str:
        if not self.toc_entries:
            return search_query
        related = select_related_toc_topics(
            self.toc_entries,
            search_query,
            self.llm,
            k=self.topic_k,
            max_candidates=self.toc_candidate_pool,
            log_fn=self._log,
        )
        if not related:
            return search_query
        self._log(f"TOC topics for '{search_query}':")
        for entry in related:
            page = f" (p. {entry['page']})" if entry["page"] is not None else ""
            self._log(f"  - {entry['title']}{page}")
        return _augment_query_with_toc_topics(search_query, related)

    def retrieve(self, search_query: str) -> list[Document]:
        """Retrieve text and image documents for one search query."""
        augmented = self._augment_search_query(search_query)
        text_docs = self.vector_store.similarity_search(augmented, k=self.retrieval_k)

        if self.image_vector_store:
            image_docs = self.image_vector_store.similarity_search(
                augmented,
                k=self.image_retrieval_k,
            )
            return self._merge_documents([text_docs, image_docs])

        return text_docs

    @staticmethod
    def _merge_documents(doc_lists: list[list[Document]]) -> list[Document]:
        merged: list[Document] = []
        seen: set[str] = set()
        for docs in doc_lists:
            for doc in docs:
                if doc.metadata.get("modality") == "image":
                    key = doc.metadata.get("image_path", doc.page_content)
                else:
                    key = doc.page_content.strip()
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)
        return merged

    def synthesize_answer(
        self,
        question: str,
        sub_queries: list[str],
        documents: list[Document],
    ) -> str:
        """Build final answer from merged retrieval results."""
        if not documents:
            return "I don't know."

        context = _format_documents_for_prompt(documents)
        sub_queries_text = "\n".join(f"- {q}" for q in sub_queries)
        chain = FINAL_ANSWER_PROMPT | self.llm
        response = chain.invoke(
            {
                "question": question,
                "sub_queries": sub_queries_text,
                "context": context,
            }
        )
        return response.content if isinstance(response.content, str) else str(response.content)

    def search(self, question: str) -> str:
        """
        Decompose question, retrieve for each sub-query, merge results,
        and synthesize a final answer with the LLM.
        """
        sub_queries, doc_lists, merged_docs = self.retrieve_all(question)
        self.print_search_details(question, sub_queries, doc_lists)
        self._log(f"Merged {len(merged_docs)} unique chunk(s) for final answer.")
        return self.synthesize_answer(question, sub_queries, merged_docs)
