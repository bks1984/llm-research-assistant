"""Run the notebook's base-vs-RAG experiment from the command line.

Usage:
    python scripts/evaluate_cli.py                      # LLM-as-a-judge only
    python scripts/evaluate_cli.py --ragas              # also RAGAS (pip install -r requirements-eval.txt)
    python scripts/evaluate_cli.py --papers path/to/pdfs --out results.csv

Reads PDFs from data/papers/ by default (see scripts/download_papers.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_assistant.config import PAPERS_DIR, SAMPLE_QUESTIONS, Settings  # noqa: E402
from research_assistant.document_loader import load_pdf_path, split_documents  # noqa: E402
from research_assistant.embeddings import get_embeddings  # noqa: E402
from research_assistant.evaluation import judge  # noqa: E402
from research_assistant.llm import LLMClient  # noqa: E402
from research_assistant.rag import answer, base_answer, retrieve  # noqa: E402
from research_assistant.vector_store import PaperIndex  # noqa: E402


def build_index(settings: Settings, folder: Path) -> PaperIndex:
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs in {folder}. Run: python scripts/download_papers.py")
    index = PaperIndex(get_embeddings(settings))
    for pdf in pdfs:
        pages = load_pdf_path(pdf)
        n = index.add(pdf.name, pages, split_documents(pages, settings.chunk_size, settings.chunk_overlap))
        print(f"indexed {pdf.name}: {len(pages)} pages, {n} chunks")
    return index


def run_ragas(rows: list[dict], settings: Settings) -> pd.DataFrame:
    """Faithfulness, answer relevancy and context precision (no reference answers needed)."""
    from datasets import Dataset
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.metrics import AnswerRelevancy, Faithfulness, LLMContextPrecisionWithoutReference

    llm = ChatOpenAI(
        model=settings.judge_model, api_key=settings.api_key, base_url=settings.base_url or None, temperature=0
    )
    emb = get_embeddings(settings)
    frames = []
    for method in ("Base (no RAG)", "RAG"):
        sub = [r for r in rows if r["method"] == method]
        ds = Dataset.from_dict(
            {
                "question": [r["question"] for r in sub],
                "answer": [r["answer"] for r in sub],
                "contexts": [r["contexts"] for r in sub],
            }
        )
        res = evaluate(
            ds,
            metrics=[AnswerRelevancy(), LLMContextPrecisionWithoutReference(), Faithfulness()],
            llm=llm,
            embeddings=emb,
        ).to_pandas()
        res["method"] = method
        frames.append(res)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--papers", type=Path, default=PAPERS_DIR)
    ap.add_argument("--out", type=Path, default=Path("evaluation_results.csv"))
    ap.add_argument("--ragas", action="store_true", help="also compute RAGAS metrics")
    args = ap.parse_args()

    settings = Settings()
    print(f"Provider: {settings.provider} · model: {settings.model} · judge: {settings.judge_model}")
    index = build_index(settings, args.papers)
    llm = LLMClient(settings)

    rows = []
    for q in SAMPLE_QUESTIONS:
        retrieval = retrieve(index, q, settings.top_k)
        for method, ans in (("Base (no RAG)", base_answer(llm, q)), ("RAG", answer(llm, retrieval))):
            res = judge(llm, q, retrieval.contexts, ans)
            rows.append(
                {
                    "question": q,
                    "method": method,
                    "answer": ans,
                    "contexts": retrieval.contexts,
                    "groundedness": res.groundedness,
                    "relevance": res.relevance,
                }
            )
            print(f"[{method:13}] G={res.groundedness} R={res.relevance} · {q[:60]}")

    df = pd.DataFrame(rows)
    print("\nAverage LLM-judge scores (1-5):")
    print(df.groupby("method")[["groundedness", "relevance"]].mean().round(2))

    if args.ragas:
        rg = run_ragas(rows, settings)
        print("\nAverage RAGAS scores (0-1):")
        print(rg.groupby("method").mean(numeric_only=True).round(3))
        rg.drop(columns=["retrieved_contexts", "contexts"], errors="ignore").to_csv(
            args.out.with_name(args.out.stem + "_ragas.csv"), index=False
        )

    df.drop(columns=["contexts"]).to_csv(args.out, index=False)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
