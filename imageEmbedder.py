"""PDF image extraction and CLIP embedding."""

from __future__ import annotations

import hashlib
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


def _default_log(msg: str) -> None:
    print(msg, flush=True)


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
