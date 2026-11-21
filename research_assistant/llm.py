"""Thin wrapper around the OpenAI SDK that also talks to Groq (OpenAI-compatible API)."""
from __future__ import annotations

import re
from collections.abc import Iterator

from openai import OpenAI

from .config import Settings

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks that some reasoning models emit."""
    return _THINK_RE.sub("", text or "").strip()


class LLMClient:
    """One object per settings combination; cheap to create."""

    def __init__(self, settings: Settings):
        if not settings.api_key:
            raise ValueError(
                f"No API key for {settings.provider}. Add it in the sidebar or in your .env file."
            )
        self.settings = settings
        kwargs = {"api_key": settings.api_key}
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        self.client = OpenAI(**kwargs)

    def _params(self, model: str | None, temperature: float | None, max_tokens: int | None) -> dict:
        s = self.settings
        params = {
            "model": model or s.model,
            "temperature": s.temperature if temperature is None else temperature,
            "top_p": s.top_p,
            "max_tokens": max_tokens or s.max_tokens,
        }
        if s.reasoning_effort:
            params["extra_body"] = {"reasoning_effort": s.reasoning_effort}
        return params

    @staticmethod
    def _messages(system: str | None, user: str) -> list[dict]:
        msgs = [{"role": "system", "content": system}] if system else []
        msgs.append({"role": "user", "content": user})
        return msgs

    def complete(
        self,
        user: str,
        system: str | None = None,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        params = self._params(model, temperature, max_tokens)
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(messages=self._messages(system, user), **params)
        return strip_reasoning(resp.choices[0].message.content or "")

    def stream(
        self,
        user: str,
        system: str | None = None,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Yield text deltas, hiding any <think> block from the reader."""
        params = self._params(model, temperature, max_tokens)
        stream = self.client.chat.completions.create(
            messages=self._messages(system, user), stream=True, **params
        )
        buffer, in_think = "", False
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            if not delta:
                continue
            buffer += delta
            # Hold text back while a <think> block is open.
            while True:
                if in_think:
                    end = buffer.lower().find("</think>")
                    if end == -1:
                        break
                    buffer, in_think = buffer[end + len("</think>"):].lstrip(), False
                else:
                    start = buffer.lower().find("<think>")
                    if start == -1:
                        # Keep a short tail in case "<think>" is split across chunks.
                        safe = len(buffer) - len("<think>")
                        if safe > 0:
                            yield buffer[:safe]
                            buffer = buffer[safe:]
                        break
                    if start:
                        yield buffer[:start]
                    buffer, in_think = buffer[start + len("<think>"):], True
        if buffer and not in_think:
            yield buffer
