"""Download the four prompt-engineering papers used in the case study from arXiv.

Usage:  python scripts/download_papers.py
Saves the PDFs to data/papers/. The app's sidebar can then load them in one click.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "papers"

PAPERS = {
    "2005.14165": "language_models_are_few_shot_learners.pdf",          # Brown et al., GPT-3
    "2104.08691": "power_of_scale_prompt_tuning.pdf",                   # Lester et al.
    "2010.15980": "autoprompt.pdf",                                     # Shin et al.
    "2102.07350": "prompt_programming_beyond_few_shot.pdf",             # Reynolds & McDonell
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for arxiv_id, name in PAPERS.items():
        dest = OUT / name
        if dest.exists():
            print(f"✓ {name} already present")
            continue
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        print(f"↓ {url} -> {dest.relative_to(ROOT)}")
        req = urllib.request.Request(url, headers={"User-Agent": "research-assistant-demo/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        if not data.startswith(b"%PDF"):
            print(f"  ✗ {arxiv_id} did not return a PDF", file=sys.stderr)
            continue
        dest.write_bytes(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
