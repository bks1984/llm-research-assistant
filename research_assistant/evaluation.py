"""LLM-as-a-judge scoring of groundedness and relevance (1-5), as in the notebook.

The notebook read the score with `text[-2]`, which breaks as soon as the model adds a
trailing space or a two-digit number. `parse_score` uses a regex instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import LLMClient
from .prompts import GROUNDEDNESS_RATER_SYSTEM, JUDGE_USER_TEMPLATE, RELEVANCE_RATER_SYSTEM


def parse_score(text: str, key: str) -> int | None:
    """Return the last `key: N` score (1-5) in the judge's output, or None."""
    matches = re.findall(rf"['\"]?{key}['\"]?\s*[:=]\s*['\"]?([1-5])\b", text or "")
    if matches:
        return int(matches[-1])
    # Fallback: a lone digit 1-5 in the last line.
    last = (text or "").strip().splitlines()[-1:] or [""]
    lone = re.findall(r"\b([1-5])\b", last[0])
    return int(lone[-1]) if lone else None


@dataclass
class JudgeResult:
    groundedness: int | None
    relevance: int | None
    groundedness_reasoning: str
    relevance_reasoning: str


def judge(llm: LLMClient, question: str, contexts: list[str], answer: str) -> JudgeResult:
    user = JUDGE_USER_TEMPLATE.format(
        question=question, context="\n\n---\n\n".join(contexts), answer=answer
    )
    model = llm.settings.judge_model
    g_text = llm.complete(user, GROUNDEDNESS_RATER_SYSTEM, model=model, temperature=0, max_tokens=700)
    r_text = llm.complete(user, RELEVANCE_RATER_SYSTEM, model=model, temperature=0, max_tokens=700)
    return JudgeResult(
        groundedness=parse_score(g_text, "groundedness_score"),
        relevance=parse_score(r_text, "relevance_score"),
        groundedness_reasoning=g_text,
        relevance_reasoning=r_text,
    )
