"""Text embedding via HuggingFace sentence-transformers."""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings


def configure_hf_token(hf_token: str | None) -> bool:
    """Authenticate Hugging Face Hub downloads with HF_TOKEN."""
    if not hf_token:
        return False

    import os

    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

    try:
        from huggingface_hub import login

        login(token=hf_token, add_to_git_credential=False)
    except ImportError:
        pass

    return True


class TextEmbedder:
    """Embed text chunks using a HuggingFace sentence-transformers model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        hf_token: str | None = None,
    ):
        configure_hf_token(hf_token)
        model_kwargs = {}
        if hf_token:
            model_kwargs["token"] = hf_token

        self.model_name = model_name
        self._embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
        )

    @property
    def langchain_embeddings(self) -> HuggingFaceEmbeddings:
        """Underlying LangChain embeddings object (required by FAISS.load_local)."""
        return self._embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)
