"""In-memory Chroma index over the uploaded papers."""
from __future__ import annotations

import uuid

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class PaperIndex:
    """Holds one Chroma collection plus the page-level docs it was built from."""

    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        # A unique name keeps separate Streamlit sessions from sharing a collection.
        self.store = Chroma(
            collection_name=f"papers-{uuid.uuid4().hex[:8]}",
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self.pages: dict[str, list[Document]] = {}  # file name -> page docs
        self.chunk_counts: dict[str, int] = {}

    @property
    def files(self) -> list[str]:
        return list(self.pages)

    def add(self, file_name: str, pages: list[Document], chunks: list[Document]) -> int:
        if file_name in self.pages or not chunks:
            return 0
        self.store.add_documents(chunks)
        self.pages[file_name] = pages
        self.chunk_counts[file_name] = len(chunks)
        return len(chunks)

    def search(self, query: str, k: int = 5, files: list[str] | None = None) -> list[tuple[Document, float]]:
        """Top-k chunks with a relevance score in [0, 1] (1 = identical)."""
        if not self.pages:
            return []
        flt = None
        if files:
            flt = {"source": files[0]} if len(files) == 1 else {"source": {"$in": files}}
        # Chroma returns cosine *distance* (0 = identical, up to 2 = opposite).
        hits = self.store.similarity_search_with_score(query, k=k, filter=flt)
        return [(doc, min(1.0, max(0.0, 1.0 - dist))) for doc, dist in hits]

    def clear(self) -> None:
        self.store.delete_collection()
        self.pages.clear()
        self.chunk_counts.clear()
