"""
Multi-query RAG search engine with query decomposition, TOC-guided retrieval,
and final answer synthesis.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from imageEmbedder import ClipEmbeddings, FingerPrintEmbedding, IMAGE_MARK_PATTERN
from tocLoader import TocEntry

DEFAULT_RETRIEVAL_K = 8
DEFAULT_IMAGE_RETRIEVAL_K = 4
DEFAULT_TOC_TOPIC_K = 5
DEFAULT_TOC_CANDIDATE_POOL = 100
DEFAULT_CHUNK_HEAD_CHARS = 150
DEFAULT_CLIP_DESCRIPTION_TOP_K = 3

CLIP_VISUAL_CANDIDATES = (
    "a technical diagram from a medical device manual",
    "a screenshot of a control panel or user interface",
    "an indicator light, icon, or status display",
    "a table of technical specifications or parameters",
    "a photograph of medical imaging equipment",
    "a flowchart or procedural illustration",
    "a warning or safety label",
)

QUERY_DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You analyze user questions for document search and split them into focused "
            "search queries (one per line).\n"
            "The manual text contains inline image placeholders written as [IMAGE_MARK N], "
            "where each number links to a CLIP-indexed image (diagrams, UI screenshots, "
            "warning labels, indicator lights, icons, and status signs). Retrieved chunks "
            "may include CLIP image descriptions appended as 'Image #N: ...'.\n"
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
            "- Image-aware retrieval: when the question involves warnings, indicators, signs, "
            "lights, icons, symbols, alarms, status displays, control panels, or other visual "
            "elements, add at least one sub-query that explicitly targets the related "
            "[IMAGE_MARK] illustrations (e.g. 'warning indicator sign icon image for [term]', "
            "'visual display or diagram of [term]', 'control panel indicator for [term]').\n"
            "- Prefer sub-queries that will retrieve chunks containing [IMAGE_MARK N] markers "
            "and CLIP image descriptions when those images would help explain the answer.\n"
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
            "When multiple sub-questions were searched, address each part clearly.\n"
            "The context may include CLIP-analyzed images identified as [Image #N] with "
            "descriptions. When an image helps explain the answer—especially for warnings, "
            "indicators, signs, icons, or control-panel displays—reference it inline using "
            "exactly [Image #N] (image number only, no file path).",
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
            image_number = doc.metadata.get("image_number", "?")
            blocks.append(
                f"[{i}] Image #{image_number} source: {source}\n"
                f"Image file: {image_path}\n"
                f"{doc.page_content}"
            )
        else:
            referenced_images = doc.metadata.get("referenced_images", [])
            image_lines = ""
            if referenced_images:
                image_lines = "\n".join(
                    f"Referenced image #{ref['image_number']}: {ref['image_path']}"
                    for ref in referenced_images
                )
                image_lines = f"\n{image_lines}"
            blocks.append(f"[{i}] Source: {source}\n{doc.page_content}{image_lines}")
    return "\n\n".join(blocks)


def _resolve_image_marks_in_documents(
    docs: list[Document],
    fingerprint_registry: FingerPrintEmbedding | None,
) -> list[Document]:
    """When chunks contain [IMAGE_MARK N], attach the numbered image to results."""
    if not fingerprint_registry:
        return docs

    resolved: list[Document] = []
    seen_image_numbers: set[int] = set()

    for doc in docs:
        resolved.append(doc)
        referenced_images = []
        for match in IMAGE_MARK_PATTERN.finditer(doc.page_content):
            image_number = int(match.group(1))
            image_path = fingerprint_registry.get_image_path(image_number)
            if image_path is None or not image_path.is_file():
                continue
            referenced_images.append(
                {
                    "image_number": image_number,
                    "image_path": str(image_path.resolve()),
                    "fingerprint": fingerprint_registry.get_fingerprint(image_number),
                }
            )
            if image_number in seen_image_numbers:
                continue
            seen_image_numbers.add(image_number)
            resolved.append(
                Document(
                    page_content=(
                        f"[Image #{image_number}] referenced near: "
                        f"{_format_chunk_head(doc, 120)}"
                    ),
                    metadata={
                        "source": doc.metadata.get("source", "unknown"),
                        "modality": "image",
                        "image_number": image_number,
                        "image_path": str(image_path.resolve()),
                        "fingerprint": fingerprint_registry.get_fingerprint(image_number),
                        "page": doc.metadata.get("page"),
                        "page_number": doc.metadata.get("page_number"),
                    },
                )
            )

        if referenced_images:
            doc.metadata["referenced_images"] = referenced_images

    return resolved


def _strip_image_marks(text: str) -> str:
    return IMAGE_MARK_PATTERN.sub("", text).strip()


def _collect_image_paths(documents: list[Document]) -> dict[int, str]:
    """Map image numbers to local image file paths from retrieved documents."""
    image_paths: dict[int, str] = {}
    for doc in documents:
        if doc.metadata.get("modality") == "image":
            image_number = doc.metadata.get("image_number")
            image_path = doc.metadata.get("image_path")
            if image_number is not None and image_path:
                image_paths[int(image_number)] = str(image_path)
    return image_paths


def _collect_image_contexts(
    documents: list[Document],
    question: str,
    sub_queries: list[str],
) -> dict[int, list[str]]:
    """Build CLIP text candidates per image from the question and chunk context."""
    image_paths = _collect_image_paths(documents)
    contexts: dict[int, list[str]] = defaultdict(list)
    base_candidates = [question, *sub_queries, *CLIP_VISUAL_CANDIDATES]

    for image_number in image_paths:
        contexts[image_number].extend(base_candidates)

    for doc in documents:
        chunk_text = _strip_image_marks(doc.page_content)
        if not chunk_text:
            continue

        if doc.metadata.get("modality") == "image":
            image_number = doc.metadata.get("image_number")
            if image_number is not None:
                contexts[int(image_number)].append(chunk_text)
            continue

        referenced_numbers = {
            int(ref["image_number"])
            for ref in doc.metadata.get("referenced_images", [])
        }
        for match in IMAGE_MARK_PATTERN.finditer(doc.page_content):
            referenced_numbers.add(int(match.group(1)))

        for image_number in referenced_numbers:
            if image_number in contexts:
                contexts[image_number].append(chunk_text)

    return contexts


def _describe_retrieved_images(
    documents: list[Document],
    question: str,
    sub_queries: list[str],
    clip_embeddings: ClipEmbeddings | None,
    *,
    top_k: int = DEFAULT_CLIP_DESCRIPTION_TOP_K,
    log_fn: Callable[[str], None] = _default_log,
) -> dict[int, str]:
    """Analyze retrieved images with CLIP and return image-number descriptions."""
    image_paths = _collect_image_paths(documents)
    if not image_paths or clip_embeddings is None:
        return {}

    contexts = _collect_image_contexts(documents, question, sub_queries)
    descriptions: dict[int, str] = {}

    for image_number, image_path in sorted(image_paths.items()):
        path = Path(image_path)
        if not path.is_file():
            continue
        description = clip_embeddings.describe_image(
            path,
            contexts.get(image_number, [question, *sub_queries]),
            top_k=top_k,
        )
        descriptions[image_number] = description
        log_fn(f"CLIP image #{image_number}: {description}")

    return descriptions


def _append_image_descriptions_to_documents(
    documents: list[Document],
    descriptions: dict[int, str],
) -> list[Document]:
    """Attach image-number/description pairs to the end of relevant chunks."""
    if not descriptions:
        return documents

    refined: list[Document] = []
    for doc in documents:
        image_numbers: set[int] = set()
        if doc.metadata.get("modality") == "image":
            image_number = doc.metadata.get("image_number")
            if image_number is not None:
                image_numbers.add(int(image_number))
        for ref in doc.metadata.get("referenced_images", []):
            image_numbers.add(int(ref["image_number"]))
        for match in IMAGE_MARK_PATTERN.finditer(doc.page_content):
            image_numbers.add(int(match.group(1)))

        relevant = [
            (image_number, descriptions[image_number])
            for image_number in sorted(image_numbers)
            if image_number in descriptions
        ]
        if not relevant:
            refined.append(doc)
            continue

        description_lines = "\n".join(
            f"Image #{image_number}: {description}"
            for image_number, description in relevant
        )
        updated = Document(
            page_content=f"{doc.page_content.rstrip()}\n\n{description_lines}",
            metadata=dict(doc.metadata),
        )
        updated.metadata["clip_image_descriptions"] = {
            str(image_number): description for image_number, description in relevant
        }
        refined.append(updated)

    return refined


def _format_answer_image_mark(image_number: int) -> str:
    return f"[Image #{image_number}]"


def _collect_clip_related_image_numbers(documents: list[Document]) -> list[int]:
    """Return sorted image numbers referenced in CLIP-refined chunks."""
    numbers: set[int] = set()
    for doc in documents:
        clip_descriptions = doc.metadata.get("clip_image_descriptions", {})
        numbers.update(int(image_number) for image_number in clip_descriptions)
    return sorted(numbers)


def _insert_related_images_into_answer(answer: str, image_numbers: list[int]) -> str:
    """Append related CLIP image markers to the final answer."""
    if not image_numbers:
        return answer

    missing = [
        image_number
        for image_number in image_numbers
        if _format_answer_image_mark(image_number) not in answer
    ]
    if not missing:
        return answer

    image_lines = "\n".join(_format_answer_image_mark(image_number) for image_number in missing)
    return f"{answer.rstrip()}\n\nRelated images:\n{image_lines}"


class RagSearchEngine:
    """Multi-query RAG retrieval with TOC guidance and final answer synthesis."""

    def __init__(
        self,
        vector_store: FAISS,
        llm: BaseChatModel,
        *,
        toc_entries: list[TocEntry] | None = None,
        image_vector_store: FAISS | None = None,
        fingerprint_registry: FingerPrintEmbedding | None = None,
        clip_embeddings: ClipEmbeddings | None = None,
        retrieval_k: int = DEFAULT_RETRIEVAL_K,
        image_retrieval_k: int = DEFAULT_IMAGE_RETRIEVAL_K,
        topic_k: int = DEFAULT_TOC_TOPIC_K,
        toc_candidate_pool: int = DEFAULT_TOC_CANDIDATE_POOL,
        chunk_head_chars: int = DEFAULT_CHUNK_HEAD_CHARS,
        log_fn: Callable[[str], None] | None = None,
    ):
        self.vector_store = vector_store
        self.image_vector_store = image_vector_store
        self.fingerprint_registry = fingerprint_registry
        self.clip_embeddings = clip_embeddings
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
                    image_number = doc.metadata.get("image_number", "?")
                    self._log(
                        f"  image chunk {j} (page {page}, #{image_number}): "
                        f"{head} [{image_path}]"
                    )
                else:
                    refs = doc.metadata.get("referenced_images", [])
                    ref_note = ""
                    if refs:
                        numbers = ", ".join(f"#{r['image_number']}" for r in refs)
                        ref_note = f" [images: {numbers}]"
                    self._log(f"  chunk {j} (page {page}): {head}{ref_note}")

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
        text_docs = _resolve_image_marks_in_documents(text_docs, self.fingerprint_registry)

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
                elif doc.metadata.get("image_number") is not None:
                    key = f"image:{doc.metadata['image_number']}"
                else:
                    key = doc.page_content.strip()
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)
        return merged

    def refine_answer(
        self,
        question: str,
        sub_queries: list[str],
        documents: list[Document],
    ) -> list[Document]:
        """Analyze retrieved images with CLIP and append image descriptions to chunks."""
        if not documents or self.clip_embeddings is None:
            return documents

        image_paths = _collect_image_paths(documents)
        if not image_paths:
            return documents

        self._log(f"Refining answer context with CLIP for {len(image_paths)} image(s)...")
        descriptions = _describe_retrieved_images(
            documents,
            question,
            sub_queries,
            self.clip_embeddings,
            log_fn=self._log,
        )
        return _append_image_descriptions_to_documents(documents, descriptions)

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
        answer = response.content if isinstance(response.content, str) else str(response.content)
        related_images = _collect_clip_related_image_numbers(documents)
        if related_images:
            self._log(
                "Inserting related CLIP images into answer: "
                + ", ".join(f"#{n}" for n in related_images)
            )
        return _insert_related_images_into_answer(answer, related_images)

    def search(self, question: str) -> str:
        """
        Decompose question, retrieve for each sub-query, merge results,
        and synthesize a final answer with the LLM.
        """
        sub_queries, doc_lists, merged_docs = self.retrieve_all(question)
        self.print_search_details(question, sub_queries, doc_lists)
        self._log(f"Merged {len(merged_docs)} unique chunk(s) for final answer.")
        refined_docs = self.refine_answer(question, sub_queries, merged_docs)
        return self.synthesize_answer(question, sub_queries, refined_docs)
