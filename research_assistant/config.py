"""Central settings for the research assistant.

Every value can be overridden with an environment variable (or a `.env` file in the
project root), so the same code runs locally, in Docker and on Hugging Face Spaces.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

PAPERS_DIR = ROOT / "data" / "papers"

# Provider presets. Groq exposes an OpenAI-compatible endpoint, so one client serves both.
PROVIDERS: dict[str, dict[str, str]] = {
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        "default_model": "qwen/qwen3.8-27b",
    },
    "OpenAI": {
        "base_url": "",  # empty -> SDK default (or OPENAI_BASE_URL if set)
        "key_env": "OPENAI_API_KEY",
        "model_env": "OPENAI_MODEL",
        "default_model": "gpt-4o-mini",
    },
}

# Embedding back-ends. "Local" needs no API key (ONNX all-MiniLM-L6-v2 bundled with chromadb).
EMBEDDING_BACKENDS = ("Local (all-MiniLM-L6-v2)", "OpenAI (text-embedding-3-small)")

# The three evaluation questions used in the notebook.
SAMPLE_QUESTIONS = [
    "What is the difference between zero-shot and few-shot prompting in language models?",
    "How does the performance of in-context learning change based on the number of "
    "examples provided and the nature of the task?",
    "What are the limitations of discrete prompts, and how to overcome them?",
]

OUT_OF_SCOPE_REPLY = "Sorry, this is out of my knowledge base."


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def default_provider() -> str:
    """Pick the provider whose API key is present; Groq wins a tie."""
    explicit = os.getenv("LLM_PROVIDER")
    if explicit in PROVIDERS:
        return explicit
    for name, preset in PROVIDERS.items():
        if os.getenv(preset["key_env"]):
            return name
    return "Groq"


@dataclass
class Settings:
    """Runtime knobs. The Streamlit sidebar edits a copy of this per session."""

    provider: str = field(default_factory=default_provider)
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    reasoning_effort: str = field(default_factory=lambda: os.getenv("REASONING_EFFORT", ""))

    embedding_backend: str = EMBEDDING_BACKENDS[0]

    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 512))       # tokens
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 64))  # tokens
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))

    temperature: float = field(default_factory=lambda: _env_float("TEMPERATURE", 0.3))
    top_p: float = 0.95
    max_tokens: int = field(default_factory=lambda: _env_int("MAX_TOKENS", 700))

    judge_model: str = ""  # empty -> same as `model`

    def __post_init__(self) -> None:
        self.resolve()

    def resolve(self) -> "Settings":
        """Fill blank fields from the provider preset and environment."""
        preset = PROVIDERS.get(self.provider, PROVIDERS["Groq"])
        if not self.model:
            self.model = os.getenv(preset["model_env"], preset["default_model"])
        if not self.api_key:
            self.api_key = os.getenv(preset["key_env"], "")
        if not self.base_url:
            if self.provider == "OpenAI":
                self.base_url = os.getenv("OPENAI_BASE_URL", "")
            else:
                self.base_url = preset["base_url"]
        if not self.judge_model:
            self.judge_model = os.getenv("JUDGE_MODEL", "") or self.model
        return self
