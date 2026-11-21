"""Offline tests: no API key or network needed."""
from __future__ import annotations

import pymupdf as fitz
import pytest
from langchain_core.documents import Document

from research_assistant.document_loader import file_hash, load_pdf_bytes, split_documents
from research_assistant.evaluation import parse_score
from research_assistant.insights import opening_text, parse_json_object
from research_assistant.llm import strip_reasoning
from research_assistant.rag import Retrieval, build_user_message


def make_pdf(pages: list[str]) -> bytes:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 550, 800), text, fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


def test_load_pdf_keeps_page_numbers():
    data = make_pdf(["Few-shot learning page one.", "", "Prompt tuning page three."])
    docs = load_pdf_bytes(data, "paper.pdf")
    assert [d.metadata["page"] for d in docs] == [1, 3]  # blank page skipped
    assert all(d.metadata["source"] == "paper.pdf" for d in docs)
    assert docs[0].metadata["total_pages"] == 3


def test_split_respects_token_budget():
    long = Document(page_content="prompt engineering " * 2000, metadata={"source": "x.pdf", "page": 1})
    chunks = split_documents([long], chunk_size=128, chunk_overlap=16)
    assert len(chunks) > 10
    assert all(c.metadata["source"] == "x.pdf" for c in chunks)
    assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))


def test_file_hash_is_stable():
    assert file_hash(b"abc") == file_hash(b"abc") != file_hash(b"abd")


@pytest.mark.parametrize(
    "text,key,expected",
    [
        ("steps...\n{groundedness_score: 4}", "groundedness_score", 4),
        ('reasoning\n{"relevance_score": 5}', "relevance_score", 5),
        ("score was 2 earlier\nFinal: {'groundedness_score': 3}  ", "groundedness_score", 3),
        ("no score here", "relevance_score", None),
        ("Overall rating\n5", "relevance_score", 5),
    ],
)
def test_parse_score(text, key, expected):
    assert parse_score(text, key) == expected


def test_parse_json_object_handles_fences_and_chatter():
    assert parse_json_object('```json\n{"title": "GPT-3"}\n```') == {"title": "GPT-3"}
    assert parse_json_object('Here you go: {"a": 1} hope it helps') == {"a": 1}


def test_strip_reasoning():
    assert strip_reasoning("<think>hmm</think>\nAnswer: 42") == "Answer: 42"


def test_opening_text_truncates():
    pages = [Document(page_content="x" * 100) for _ in range(5)]
    assert len(opening_text(pages, max_chars=250).replace("\n", "")) == 250


def test_prompt_labels_sources():
    hits = [(Document(page_content="ctx", metadata={"source": "a.pdf", "page": 2}), 0.8)]
    msg = build_user_message(Retrieval(question="What?", hits=hits))
    assert "[Source 1 | a.pdf | page 2]" in msg and "###Question\nWhat?" in msg
