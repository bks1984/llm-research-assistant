"""Embedding back-ends exposed through the LangChain `Embeddings` interface."""
from __future__ import annotations

import os

from langchain_core.embeddings import Embeddings

from .config import EMBEDDING_BACKENDS, Settings


class LocalMiniLMEmbeddings(Embeddings):
    """all-MiniLM-L6-v2 run through ONNX Runtime (ships with chromadb).

    No API key and no PyTorch needed. The ~80 MB model is downloaded once on first use and
    cached in ~/.cache/chroma.
    """

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_backend == EMBEDDING_BACKENDS[1]:
        from langchain_openai import OpenAIEmbeddings

        if settings.provider != "OpenAI":
            # Embeddings always need an OpenAI key, even when chat runs on Groq.
            key = os.getenv("OPENAI_API_KEY", "")
            base = os.getenv("OPENAI_BASE_URL") or None
        else:
            key, base = settings.api_key, settings.base_url or None
        if not key:
            raise ValueError("OpenAI embeddings need OPENAI_API_KEY. Use the local model instead.")
        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=key, base_url=base)
    return LocalMiniLMEmbeddings()
