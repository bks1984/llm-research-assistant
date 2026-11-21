"""Per-paper metadata + summary extraction, and cross-paper trend analysis."""
from __future__ import annotations

import json
import re

from langchain_core.documents import Document

from .llm import LLMClient
from .prompts import PAPER_INSIGHT_SYSTEM, PAPER_INSIGHT_USER, TRENDS_SYSTEM

# ~4 chars per token; keep the input well inside small-context models.
MAX_INSIGHT_CHARS = 14_000


def opening_text(pages: list[Document], max_chars: int = MAX_INSIGHT_CHARS) -> str:
    """Title, abstract and introduction live on the first pages, so read from the start."""
    out, total = [], 0
    for doc in pages:
        chunk = doc.page_content[: max_chars - total]
        out.append(chunk)
        total += len(chunk)
        if total >= max_chars:
            break
    return "\n\n".join(out)


def parse_json_object(text: str) -> dict:
    """Parse the first JSON object in `text`, tolerating ```json fences and chatter."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def extract_paper_insight(llm: LLMClient, file_name: str, pages: list[Document]) -> dict:
    user = PAPER_INSIGHT_USER.format(file_name=file_name, text=opening_text(pages))
    try:
        raw = llm.complete(user, PAPER_INSIGHT_SYSTEM, temperature=0, max_tokens=1200, json_mode=True)
    except Exception:
        # Some models/endpoints reject response_format; retry as plain text.
        raw = llm.complete(user, PAPER_INSIGHT_SYSTEM, temperature=0, max_tokens=1200)
    data = parse_json_object(raw)
    data["file"] = file_name
    data["pages"] = pages[-1].metadata.get("total_pages", len(pages)) if pages else 0
    return data


def analyse_trends(llm: LLMClient, insights: list[dict]) -> str:
    slim = [
        {k: d.get(k) for k in ("title", "year", "keywords", "summary", "key_findings", "limitations")}
        for d in insights
    ]
    return llm.complete(json.dumps(slim, indent=2), TRENDS_SYSTEM, temperature=0.2, max_tokens=1200)
