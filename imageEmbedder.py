"""PDF image extraction, fingerprint embedding, and CLIP embedding."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from PIL import Image

from faissIndex import (
    build_faiss_vector_store,
    load_faiss_vector_store,
    save_faiss_vector_store,
)

DEFAULT_CLIP_MODEL = "sentence-transformers/clip-ViT-B-32"
DEFAULT_MIN_IMAGE_SIZE = 64
DEFAULT_EXTRACTED_IMAGE_DIR = "extracted_images"
DEFAULT_FINGERPRINT_HASH_SIZE = 16
DEFAULT_REFORMATTED_TEXT_DIR = "reformatted_documents"
DEFAULT_FINGERPRINT_REGISTRY = "image_fingerprint_registry.json"
IMAGE_MARK_PATTERN = re.compile(r"\[IMAGE_MARK\s+(\d+)\]")


def _default_log(msg: str) -> None:
    print(msg, flush=True)


def format_image_mark(image_number: int) -> str:
    """Return the inline marker substituted for an image in document text."""
    return f"[IMAGE_MARK {image_number}]"


class FingerPrintEmbedding:
    """Assign image numbers and store perceptual fingerprints for retrieval."""

    def __init__(self, *, hash_size: int = DEFAULT_FINGERPRINT_HASH_SIZE):
        self.hash_size = hash_size
        self._fingerprints: dict[int, str] = {}
        self._image_paths: dict[int, Path] = {}
        self._next_number = 1

    @staticmethod
    def compute_fingerprint(image: Image.Image, hash_size: int = DEFAULT_FINGERPRINT_HASH_SIZE) -> str:
        """Average-hash fingerprint as a hex string."""
        gray = image.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.float32)
        avg = pixels.mean()
        bits = (pixels >= avg).astype(np.uint8).flatten()
        n_bytes = (len(bits) + 7) // 8
        packed = np.zeros(n_bytes, dtype=np.uint8)
        for index, bit in enumerate(bits):
            if bit:
                packed[index // 8] |= 1 << (7 - index % 8)
        return packed.tobytes().hex()

    def register_image(
        self,
        image_path: Path | str,
        *,
        image: Image.Image | None = None,
    ) -> int:
        """Assign the next image number, fingerprint the image, and record its path."""
        path = Path(image_path)
        image_number = self._next_number
        self._next_number += 1

        if image is None:
            with Image.open(path) as opened:
                image = opened.convert("RGB")

        self._fingerprints[image_number] = self.compute_fingerprint(image, self.hash_size)
        self._image_paths[image_number] = path.resolve()
        return image_number

    def get_fingerprint(self, image_number: int) -> str | None:
        return self._fingerprints.get(image_number)

    def get_image_path(self, image_number: int) -> Path | None:
        return self._image_paths.get(image_number)

    @property
    def image_count(self) -> int:
        return len(self._image_paths)

    def save(self, path: Path | str) -> None:
        """Persist image-number registry to JSON."""
        registry_path = Path(path)
        payload = {
            "hash_size": self.hash_size,
            "next_number": self._next_number,
            "fingerprints": {str(k): v for k, v in self._fingerprints.items()},
            "image_paths": {str(k): str(v) for k, v in self._image_paths.items()},
        }
        registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> FingerPrintEmbedding:
        """Load image-number registry from JSON."""
        registry_path = Path(path)
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
        registry = cls(hash_size=payload.get("hash_size", DEFAULT_FINGERPRINT_HASH_SIZE))
        registry._next_number = payload.get("next_number", 1)
        registry._fingerprints = {
            int(k): v for k, v in payload.get("fingerprints", {}).items()
        }
        registry._image_paths = {
            int(k): Path(v) for k, v in payload.get("image_paths", {}).items()
        }
        if registry._fingerprints:
            registry._next_number = max(
                registry._next_number,
                max(registry._fingerprints.keys()) + 1,
            )
        return registry


class ClipEmbeddings(Embeddings):
    """CLIP embeddings for images (documents) and text (queries)."""

    def __init__(self, model_name: str = DEFAULT_CLIP_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Install sentence-transformers for CLIP: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed image file paths with CLIP image encoder."""
        if not texts:
            return []
        images = [Image.open(path).convert("RGB") for path in texts]
        vectors = self._model.encode(images, convert_to_numpy=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a text query with CLIP text encoder."""
        vector = self._model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return vector.tolist()

    def describe_image(
        self,
        image_path: str | Path,
        text_candidates: list[str],
        *,
        top_k: int = 3,
        min_score: float = 0.15,
    ) -> str:
        """Describe an image by ranking text candidates with CLIP similarity."""
        if not text_candidates:
            return "No description available."

        unique_candidates: list[str] = []
        seen: set[str] = set()
        for text in text_candidates:
            cleaned = " ".join(text.split())
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append(cleaned)

        if not unique_candidates:
            return "No description available."

        image = Image.open(image_path).convert("RGB")
        image_vector = np.array(
            self._model.encode(image, convert_to_numpy=True, show_progress_bar=False),
            dtype=np.float32,
        )
        image_norm = np.linalg.norm(image_vector)
        if image_norm == 0:
            return unique_candidates[0]
        image_vector /= image_norm

        text_vectors = np.array(
            self._model.encode(unique_candidates, convert_to_numpy=True, show_progress_bar=False),
            dtype=np.float32,
        )
        text_norms = np.linalg.norm(text_vectors, axis=1, keepdims=True)
        text_norms[text_norms == 0] = 1.0
        text_vectors /= text_norms

        scores = text_vectors @ image_vector
        ranked = sorted(
            zip(unique_candidates, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        selected = [text for text, score in ranked[:top_k] if score >= min_score]
        if not selected:
            selected = [ranked[0][0]]
        return "; ".join(selected)


class ImageEmbedder:
    """Extract images from PDF pages and build a CLIP+FAISS image index."""

    def __init__(
        self,
        pdf_path: Path | str,
        *,
        clip_model: str = DEFAULT_CLIP_MODEL,
        min_image_size: int = DEFAULT_MIN_IMAGE_SIZE,
        extracted_image_dir: str = DEFAULT_EXTRACTED_IMAGE_DIR,
        log_fn: Callable[[str], None] | None = None,
    ):
        self.pdf_path = Path(pdf_path)
        self.clip_model = clip_model
        self.min_image_size = min_image_size
        self.extracted_image_dir = Path(extracted_image_dir)
        self._log = log_fn or _default_log
        self._clip = ClipEmbeddings(clip_model)

    @property
    def clip_embeddings(self) -> ClipEmbeddings:
        return self._clip

    def extract_images(self) -> list[Document]:
        """Extract embedded images from the PDF into extracted_image_dir."""
        try:
            import fitz
        except ImportError as exc:
            raise ImportError("Install pymupdf: pip install pymupdf") from exc

        if not self.pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path.resolve()}")

        self.extracted_image_dir.mkdir(parents=True, exist_ok=True)
        pdf_stem = self.pdf_path.stem.replace(" ", "_")[:80]
        output_dir = self.extracted_image_dir / pdf_stem
        output_dir.mkdir(parents=True, exist_ok=True)

        documents: list[Document] = []
        seen_xrefs: set[int] = set()

        doc = fitz.open(self.pdf_path)
        try:
            for page_number in range(len(doc)):
                page = doc[page_number]
                for image_index, image_info in enumerate(page.get_images(full=True)):
                    xref = image_info[0]
                    if xref in seen_xrefs:
                        continue

                    try:
                        extracted = doc.extract_image(xref)
                    except Exception:
                        continue

                    width = extracted.get("width", 0)
                    height = extracted.get("height", 0)
                    if width < self.min_image_size or height < self.min_image_size:
                        continue

                    image_bytes = extracted["image"]
                    ext = extracted.get("ext", "png")
                    digest = hashlib.md5(image_bytes).hexdigest()[:12]
                    image_path = output_dir / f"page_{page_number + 1}_img_{image_index}.{ext}"
                    if not image_path.exists():
                        image_path.write_bytes(image_bytes)

                    seen_xrefs.add(xref)
                    documents.append(
                        Document(
                            page_content=(
                                f"[Image page {page_number + 1}, "
                                f"size {width}x{height}, file {image_path.name}]"
                            ),
                            metadata={
                                "source": str(self.pdf_path),
                                "modality": "image",
                                "page": page_number,
                                "page_number": page_number + 1,
                                "image_index": image_index,
                                "image_path": str(image_path.resolve()),
                                "width": width,
                                "height": height,
                                "image_xref": xref,
                            },
                        )
                    )
        finally:
            doc.close()

        self._log(f"Extracted {len(documents)} image(s) from {self.pdf_path.name}")
        return documents

    def embed_image_documents(
        self,
        image_documents: list[Document],
    ) -> tuple[list[Document], np.ndarray]:
        """Embed extracted image documents with CLIP."""
        if not image_documents:
            return [], np.array([], dtype=np.float32)

        image_paths = [doc.metadata["image_path"] for doc in image_documents]
        self._log(f"Embedding {len(image_paths)} image(s) with CLIP ({self.clip_model})...")
        vectors = np.array(self._clip.embed_documents(image_paths), dtype=np.float32)
        return image_documents, vectors

    def build_vector_store(
        self,
        image_documents: list[Document] | None = None,
        *,
        nlist: int,
        nprobe: int,
    ):
        if image_documents is None:
            image_documents = self.extract_images()

        if not image_documents:
            self._log("No images extracted; skipping image vector store.")
            return None

        image_documents, vectors = self.embed_image_documents(image_documents)
        return build_faiss_vector_store(
            image_documents,
            vectors,
            self._clip,
            nlist=nlist,
            nprobe=nprobe,
            log_fn=self._log,
        )

    def save_vector_store(self, vector_store, path: str) -> None:
        save_faiss_vector_store(vector_store, path)

    def load_vector_store(self, path: str, *, nprobe: int):
        return load_faiss_vector_store(path, self._clip, nprobe=nprobe)

    def reformat_document_with_image_marks(
        self,
        *,
        reformatted_text_dir: str | Path = DEFAULT_REFORMATTED_TEXT_DIR,
        registry_path: str | Path = DEFAULT_FINGERPRINT_REGISTRY,
        fingerprint_registry: FingerPrintEmbedding | None = None,
    ) -> tuple[list[Document], FingerPrintEmbedding]:
        """
        Read the PDF, replace each image with an [IMAGE_MARK N] marker in page text,
        save numbered image copies, and write the reformatted document as plain text.
        """
        try:
            import fitz
        except ImportError as exc:
            raise ImportError("Install pymupdf: pip install pymupdf") from exc

        if not self.pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path.resolve()}")

        registry = fingerprint_registry or FingerPrintEmbedding()
        reformatted_dir = Path(reformatted_text_dir)
        reformatted_dir.mkdir(parents=True, exist_ok=True)

        pdf_stem = self.pdf_path.stem.replace(" ", "_")[:80]
        output_dir = self.extracted_image_dir / pdf_stem
        output_dir.mkdir(parents=True, exist_ok=True)

        reformatted_text_path = reformatted_dir / f"{pdf_stem}.txt"
        page_documents: list[Document] = []
        full_text_parts: list[str] = []
        xref_to_number: dict[int, int] = {}

        doc = fitz.open(self.pdf_path)
        try:
            for page_number in range(len(doc)):
                page = doc[page_number]
                elements: list[tuple[float, float, str, object]] = []

                for block in page.get_text("blocks"):
                    if block[6] != 0:
                        continue
                    text = block[4].strip()
                    if text:
                        elements.append((block[1], block[0], "text", text))

                for image_index, image_info in enumerate(page.get_images(full=True)):
                    xref = image_info[0]
                    for rect in page.get_image_rects(xref):
                        elements.append((rect.y0, rect.x0, "image", (xref, image_index)))

                elements.sort(key=lambda item: (item[0], item[1]))

                page_parts: list[str] = []
                for _, _, kind, payload in elements:
                    if kind == "text":
                        page_parts.append(str(payload))
                        continue

                    xref, image_index = payload
                    if xref in xref_to_number:
                        page_parts.append(format_image_mark(xref_to_number[xref]))
                        continue

                    try:
                        extracted = doc.extract_image(xref)
                    except Exception:
                        continue

                    width = extracted.get("width", 0)
                    height = extracted.get("height", 0)
                    if width < self.min_image_size or height < self.min_image_size:
                        continue

                    image_bytes = extracted["image"]
                    ext = extracted.get("ext", "png")
                    temp_path = output_dir / f"page_{page_number + 1}_img_{image_index}.{ext}"
                    temp_path.write_bytes(image_bytes)

                    with Image.open(temp_path) as image:
                        image_number = registry.register_image(temp_path, image=image.convert("RGB"))

                    numbered_path = output_dir / f"{image_number}.{ext}"
                    if numbered_path != temp_path:
                        if numbered_path.exists():
                            numbered_path.unlink()
                        temp_path.rename(numbered_path)
                        registry._image_paths[image_number] = numbered_path.resolve()

                    xref_to_number[xref] = image_number
                    page_parts.append(format_image_mark(image_number))

                page_text = "\n".join(part for part in page_parts if part).strip()
                if not page_text:
                    page_text = page.get_text().strip()

                page_header = f"--- Page {page_number + 1} ---"
                full_text_parts.append(f"{page_header}\n{page_text}")

                page_documents.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            "source": str(self.pdf_path),
                            "modality": "text",
                            "page": page_number,
                            "page_number": page_number + 1,
                        },
                    )
                )
        finally:
            doc.close()

        reformatted_text_path.write_text("\n\n".join(full_text_parts), encoding="utf-8")
        registry.save(registry_path)

        self._log(
            f"Reformatted {self.pdf_path.name}: {registry.image_count} image(s) marked, "
            f"text written to {reformatted_text_path.name}"
        )
        return page_documents, registry
