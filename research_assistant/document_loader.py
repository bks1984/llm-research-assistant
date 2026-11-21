"""Load PDFs into page-level Documents and split them into token-sized chunks."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pymupdf as fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def load_pdf_bytes(data: bytes, file_name: str) -> list[Document]:
    """One Document per page, with file name, page number and PDF metadata attached."""
    docs: list[Document] = []
    with fitz.open(stream=data, filetype="pdf") as pdf:
        meta = pdf.metadata or {}
        for i, page in enumerate(pdf):
            text = page.get_text("text").strip()
            if not text:
                continue  # skip blank / image-only pages
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_name,
                        "page": i + 1,
                        "total_pages": pdf.page_count,
                        "pdf_title": meta.get("title") or "",
                        "pdf_author": meta.get("author") or "",
                    },
                )
            )
    return docs


def load_pdf_path(path: str | Path) -> list[Document]:
    path = Path(path)
    return load_pdf_bytes(path.read_bytes(), path.name)


def split_documents(docs: list[Document], chunk_size: int = 512, chunk_overlap: int = 64) -> list[Document]:
    """Token-based splitting with tiktoken's cl100k_base, as in the notebook."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
    return chunks
