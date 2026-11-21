"""Retrieval-augmented answering (the notebook's `generate_rag_response`)."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from langchain_core.documents import Document

from .config import OUT_OF_SCOPE_REPLY, Settings
from .llm import LLMClient
from .prompts import BASE_SYSTEM_MESSAGE, QNA_SYSTEM_MESSAGE, QNA_USER_TEMPLATE
from .vector_store import PaperIndex


@dataclass
class Retrieval:
    question: str
    hits: list[tuple[Document, float]] = field(default_factory=list)

    @property
    def contexts(self) -> list[str]:
        return [doc.page_content for doc, _ in self.hits]

    @property
    def top_score(self) -> float:
        return max((s for _, s in self.hits), default=0.0)

    def format_context(self) -> str:
        """Label every excerpt so the model can cite it."""
        blocks = []
        for i, (doc, _) in enumerate(self.hits, start=1):
            label = f"[Source {i} | {doc.metadata.get('source', '?')} | page {doc.metadata.get('page', '?')}]"
            blocks.append(f"{label}\n{doc.page_content}")
        return "\n\n---\n\n".join(blocks)


def retrieve(index: PaperIndex, question: str, k: int, files: list[str] | None = None) -> Retrieval:
    return Retrieval(question=question, hits=index.search(question, k=k, files=files))


def build_user_message(retrieval: Retrieval) -> str:
    return QNA_USER_TEMPLATE.format(context=retrieval.format_context(), question=retrieval.question)


def answer(llm: LLMClient, retrieval: Retrieval) -> str:
    if not retrieval.hits:
        return OUT_OF_SCOPE_REPLY
    return llm.complete(build_user_message(retrieval), QNA_SYSTEM_MESSAGE)


def answer_stream(llm: LLMClient, retrieval: Retrieval) -> Iterator[str]:
    if not retrieval.hits:
        yield OUT_OF_SCOPE_REPLY
        return
    yield from llm.stream(build_user_message(retrieval), QNA_SYSTEM_MESSAGE)


def base_answer(llm: LLMClient, question: str) -> str:
    """Answer with no retrieved context, for the base-vs-RAG comparison."""
    return llm.complete(question, BASE_SYSTEM_MESSAGE)


def make_llm(settings: Settings) -> LLMClient:
    return LLMClient(settings)
