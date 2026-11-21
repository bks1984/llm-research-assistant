"""Streamlit UI for the LLM-powered research assistant.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import json
import time

import pandas as pd
import streamlit as st

from research_assistant import config
from research_assistant.config import EMBEDDING_BACKENDS, PROVIDERS, SAMPLE_QUESTIONS, Settings
from research_assistant.document_loader import file_hash, load_pdf_bytes, split_documents
from research_assistant.embeddings import get_embeddings
from research_assistant.evaluation import judge
from research_assistant.insights import analyse_trends, extract_paper_insight
from research_assistant.llm import LLMClient
from research_assistant.rag import answer, answer_stream, base_answer, retrieve
from research_assistant.vector_store import PaperIndex

st.set_page_config(page_title="LLM Research Assistant", page_icon="📚", layout="wide")


# ----------------------------------------------------------------------------- state
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("raw_files", {})       # file name -> pdf bytes (kept so the index can be rebuilt)
    ss.setdefault("index", None)         # PaperIndex
    ss.setdefault("index_key", None)     # (embedding backend, chunk size, overlap)
    ss.setdefault("messages", [])        # chat history
    ss.setdefault("insights", {})        # file name -> insight dict
    ss.setdefault("trends", "")
    ss.setdefault("eval_df", None)


@st.cache_resource(show_spinner="Loading embedding model…")
def cached_embeddings(backend: str, provider: str, api_key: str, base_url: str):
    s = Settings(provider=provider, api_key=api_key, base_url=base_url, embedding_backend=backend)
    return get_embeddings(s)


# ----------------------------------------------------------------------------- sidebar
def sidebar() -> Settings:
    with st.sidebar:
        st.header("⚙️ Settings")

        provider = st.selectbox(
            "LLM provider", list(PROVIDERS), index=list(PROVIDERS).index(config.default_provider())
        )
        defaults = Settings(provider=provider)
        model = st.text_input("Chat model", value=defaults.model, help="Any chat model your provider serves.")
        key_env = PROVIDERS[provider]["key_env"]
        api_key = st.text_input(
            f"{provider} API key",
            type="password",
            placeholder=f"Loaded from {key_env}" if defaults.api_key else f"Paste key or set {key_env} in .env",
        )

        with st.expander("Retrieval", expanded=False):
            embedding_backend = st.selectbox("Embeddings", EMBEDDING_BACKENDS)
            chunk_size = st.slider("Chunk size (tokens)", 128, 1024, defaults.chunk_size, 64)
            chunk_overlap = st.slider("Chunk overlap (tokens)", 0, 256, defaults.chunk_overlap, 16)
            top_k = st.slider("Chunks retrieved (k)", 1, 12, defaults.top_k)

        with st.expander("Generation", expanded=False):
            temperature = st.slider("Temperature", 0.0, 1.0, defaults.temperature, 0.05)
            max_tokens = st.slider("Max tokens", 128, 2048, defaults.max_tokens, 64)
            judge_model = st.text_input("Judge model (evaluation)", value=defaults.judge_model)

        settings = Settings(
            provider=provider,
            model=model.strip(),
            api_key=api_key.strip(),
            embedding_backend=embedding_backend,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            temperature=temperature,
            max_tokens=max_tokens,
            judge_model=judge_model.strip(),
        )

        st.divider()
        st.header("📄 Papers")
        uploads = st.file_uploader("Upload research PDFs", type="pdf", accept_multiple_files=True)
        for up in uploads or []:
            st.session_state.raw_files.setdefault(up.name, up.getvalue())

        local = sorted(config.PAPERS_DIR.glob("*.pdf")) if config.PAPERS_DIR.exists() else []
        if local and st.button(f"Load {len(local)} PDF(s) from data/papers", use_container_width=True):
            for p in local:
                st.session_state.raw_files.setdefault(p.name, p.read_bytes())

        if st.session_state.raw_files and st.button("🗑️ Clear all papers", use_container_width=True):
            if st.session_state.index:
                st.session_state.index.clear()
            for k in ("raw_files", "index", "index_key", "messages", "insights", "trends", "eval_df"):
                del st.session_state[k]
            st.rerun()

    return settings


# ----------------------------------------------------------------------------- indexing
def sync_index(settings: Settings) -> PaperIndex | None:
    """Make sure every uploaded PDF is chunked, embedded and searchable."""
    ss = st.session_state
    if not ss.raw_files:
        return None

    key = (settings.embedding_backend, settings.chunk_size, settings.chunk_overlap)
    if ss.index is None or ss.index_key != key:
        if ss.index is not None:
            ss.index.clear()
        try:
            emb = cached_embeddings(settings.embedding_backend, settings.provider, settings.api_key, settings.base_url)
        except Exception as exc:
            st.error(f"Could not load embeddings: {exc}")
            return None
        ss.index, ss.index_key = PaperIndex(emb), key

    index: PaperIndex = ss.index
    pending = [n for n in ss.raw_files if n not in index.pages]
    if pending:
        bar = st.sidebar.progress(0.0, text="Indexing…")
        seen_hashes = {file_hash(ss.raw_files[n]) for n in index.pages}
        for i, name in enumerate(pending, start=1):
            bar.progress((i - 1) / len(pending), text=f"Indexing {name}")
            data = ss.raw_files[name]
            if file_hash(data) in seen_hashes:
                st.sidebar.info(f"Skipped duplicate: {name}")
                index.pages[name] = []
                continue
            pages = load_pdf_bytes(data, name)
            if not pages:
                st.sidebar.warning(f"No extractable text in {name} (scanned PDF?).")
                index.pages[name] = []
                continue
            chunks = split_documents(pages, settings.chunk_size, settings.chunk_overlap)
            index.add(name, pages, chunks)
            seen_hashes.add(file_hash(data))
        bar.empty()

    with st.sidebar:
        rows = [
            {"Paper": n, "Pages": len(p), "Chunks": index.chunk_counts.get(n, 0)}
            for n, p in index.pages.items()
            if p
        ]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    return index


def get_llm(settings: Settings) -> LLMClient | None:
    try:
        return LLMClient(settings)
    except ValueError as exc:
        st.warning(str(exc))
        return None


def show_sources(hits) -> None:
    with st.expander(f"📎 Sources ({len(hits)} chunks)"):
        for i, (doc, score) in enumerate(hits, start=1):
            st.markdown(
                f"**Source {i}** · `{doc.metadata.get('source')}` · page {doc.metadata.get('page')} "
                f"· relevance {score:.2f}"
            )
            st.caption(doc.page_content[:700] + ("…" if len(doc.page_content) > 700 else ""))


# ----------------------------------------------------------------------------- tabs
def tab_ask(settings: Settings, index: PaperIndex | None) -> None:
    ss = st.session_state
    if index is None or not index.chunk_counts:
        st.info("Upload one or more research PDFs in the sidebar to start asking questions.")
        return

    c1, c2 = st.columns([3, 1])
    with c1:
        scope = st.multiselect("Search in", list(index.chunk_counts), placeholder="All papers")
    with c2:
        compare = st.toggle("Compare with no-RAG answer", help="Also ask the LLM without any retrieved context.")

    st.caption("Try: " + " · ".join(f"*{q}*" for q in SAMPLE_QUESTIONS[:2]))

    history = st.container()  # keeps the conversation above the input box
    for msg in ss.messages:
        with history.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("base"):
                with st.expander("Answer without RAG (LLM only)"):
                    st.markdown(msg["base"])
            if msg.get("hits"):
                show_sources(msg["hits"])

    question = st.chat_input("Ask a question about your papers…")
    if not question:
        return
    llm = get_llm(settings)
    if llm is None:
        return

    ss.messages.append({"role": "user", "content": question})
    with history.chat_message("user"):
        st.markdown(question)

    with history.chat_message("assistant"):
        retrieval = retrieve(index, question, settings.top_k, scope or None)
        t0 = time.perf_counter()
        try:
            text = st.write_stream(answer_stream(llm, retrieval))
        except Exception as exc:
            text = f"Sorry, I encountered an error:\n\n```\n{exc}\n```"
            st.error(text)
        st.caption(f"⏱️ {time.perf_counter() - t0:.1f}s · top relevance {retrieval.top_score:.2f}")
        base = None
        if compare:
            with st.spinner("Asking the LLM without context…"):
                try:
                    base = base_answer(llm, question)
                except Exception as exc:
                    base = f"Error: {exc}"
            with st.expander("Answer without RAG (LLM only)", expanded=True):
                st.markdown(base)
        show_sources(retrieval.hits)

    ss.messages.append({"role": "assistant", "content": text, "hits": retrieval.hits, "base": base})


def tab_insights(settings: Settings, index: PaperIndex | None) -> None:
    ss = st.session_state
    if index is None or not index.chunk_counts:
        st.info("Upload papers first.")
        return

    st.write(
        "Extract the title, authors, year, keywords, a summary and key findings from each paper, "
        "then look for common themes and trends across the collection."
    )
    todo = [f for f in index.files if index.pages.get(f) and f not in ss.insights]
    c1, c2, c3 = st.columns(3)
    run = c1.button(f"🔍 Extract insights ({len(todo)} new)", disabled=not todo, use_container_width=True)
    trends = c2.button("📈 Analyse trends", disabled=len(ss.insights) < 1, use_container_width=True)
    if c3.button("↺ Reset insights", disabled=not ss.insights, use_container_width=True):
        ss.insights, ss.trends = {}, ""
        st.rerun()

    if run or trends:
        llm = get_llm(settings)
        if llm is None:
            return
        if run:
            bar, failed = st.progress(0.0), False
            for i, name in enumerate(todo, start=1):
                bar.progress((i - 1) / len(todo), text=f"Reading {name}")
                try:
                    ss.insights[name] = extract_paper_insight(llm, name, index.pages[name])
                except Exception as exc:
                    failed = True
                    st.error(f"{name}: {exc}")
            bar.empty()
            if not failed and not trends:
                st.rerun()  # refresh button labels/states
        if trends:
            with st.spinner("Comparing papers…"):
                try:
                    ss.trends = analyse_trends(llm, list(ss.insights.values()))
                except Exception as exc:
                    st.error(str(exc))

    if not ss.insights:
        return

    table = pd.DataFrame(
        [
            {
                "File": d.get("file"),
                "Title": d.get("title"),
                "Authors": ", ".join(d.get("authors") or []),
                "Year": d.get("year"),
                "Venue / ID": d.get("venue_or_id"),
                "Keywords": ", ".join(d.get("keywords") or []),
            }
            for d in ss.insights.values()
        ]
    )
    st.subheader("Metadata")
    st.dataframe(table, hide_index=True, use_container_width=True)

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Metadata CSV", table.to_csv(index=False), "paper_metadata.csv", "text/csv")
    d2.download_button(
        "⬇️ Full insights JSON", json.dumps(list(ss.insights.values()), indent=2), "paper_insights.json", "application/json"
    )

    st.subheader("Summaries")
    for d in ss.insights.values():
        with st.expander(f"**{d.get('title') or d.get('file')}**"):
            st.markdown(d.get("summary") or "_No summary_")
            if d.get("key_findings"):
                st.markdown("**Key findings**\n" + "\n".join(f"- {x}" for x in d["key_findings"]))
            if d.get("limitations"):
                st.markdown("**Limitations**\n" + "\n".join(f"- {x}" for x in d["limitations"]))

    if ss.trends:
        st.subheader("Trends across the collection")
        st.markdown(ss.trends)


def tab_evaluate(settings: Settings, index: PaperIndex | None) -> None:
    ss = st.session_state
    st.write(
        "Reproduces the notebook's experiment: each question is answered **without RAG** (LLM only) "
        "and **with RAG**, then an LLM judge scores both answers 1–5 for **groundedness** (is it "
        "supported by the retrieved context?) and **relevance** (does it address the question?)."
    )
    if index is None or not index.chunk_counts:
        st.info("Upload papers first.")
        return

    questions_text = st.text_area("Questions (one per line)", "\n".join(SAMPLE_QUESTIONS), height=120)
    questions = [q.strip() for q in questions_text.splitlines() if q.strip()]
    st.caption(f"{len(questions)} question(s) × 2 answers × 2 judge calls = {len(questions) * 6} LLM calls")

    if st.button("▶️ Run evaluation", disabled=not questions):
        llm = get_llm(settings)
        if llm is None:
            return
        rows = []
        bar = st.progress(0.0)
        for i, q in enumerate(questions, start=1):
            bar.progress((i - 1) / len(questions), text=f"Q{i}: {q[:60]}…")
            try:
                retrieval = retrieve(index, q, settings.top_k)
                answers = {"Base (no RAG)": base_answer(llm, q), "RAG": answer(llm, retrieval)}
                for method, ans in answers.items():
                    res = judge(llm, q, retrieval.contexts, ans)
                    rows.append(
                        {
                            "question": q,
                            "method": method,
                            "answer": ans,
                            "groundedness": res.groundedness,
                            "relevance": res.relevance,
                            "groundedness_reasoning": res.groundedness_reasoning,
                            "relevance_reasoning": res.relevance_reasoning,
                        }
                    )
            except Exception as exc:
                st.error(f"Q{i} failed: {exc}")
        bar.empty()
        ss.eval_df = pd.DataFrame(rows) if rows else None

    df = ss.eval_df
    if df is None or df.empty:
        return

    avg = df.groupby("method")[["groundedness", "relevance"]].mean().round(2)
    st.subheader("Average scores (1–5)")
    m = st.columns(len(avg) * 2)
    for j, (method, row) in enumerate(avg.iterrows()):
        m[2 * j].metric(f"{method} · groundedness", row["groundedness"])
        m[2 * j + 1].metric(f"{method} · relevance", row["relevance"])
    st.bar_chart(avg.T, stack=False)

    st.subheader("Per question")
    st.dataframe(
        df[["question", "method", "groundedness", "relevance", "answer"]],
        hide_index=True,
        use_container_width=True,
    )
    with st.expander("Judge reasoning"):
        for _, r in df.iterrows():
            st.markdown(f"**{r['method']}** — {r['question']}")
            st.text(r["groundedness_reasoning"][-1500:])
            st.text(r["relevance_reasoning"][-1500:])
            st.divider()
    st.download_button("⬇️ Results CSV", df.to_csv(index=False), "evaluation_results.csv", "text/csv")


def tab_about() -> None:
    st.markdown(
        """
### How it works
1. **Load**: each PDF is read page by page with PyMuPDF. File name and page number are kept as metadata.
2. **Chunk**: pages are split into ~512-token chunks (tiktoken `cl100k_base`) with a small overlap.
3. **Embed and index**: chunks are embedded (local all-MiniLM-L6-v2 or OpenAI) and stored in an in-memory Chroma collection.
4. **Retrieve**: the question is embedded and the top-k most similar chunks are fetched (cosine similarity).
5. **Generate**: the chunks, labelled with source and page, go into a grounded prompt. The LLM must answer only from them and cite its sources.
6. **Evaluate**: an LLM judge scores groundedness and relevance, comparing RAG with a plain LLM answer.

Your papers stay in this browser session. Nothing is written to disk.
"""
    )


# ----------------------------------------------------------------------------- main
def main() -> None:
    init_state()
    settings = sidebar()
    index = sync_index(settings)

    st.title("📚 LLM-Powered Research Assistant")
    st.caption(
        f"RAG over your research papers · {settings.provider} `{settings.model}` · "
        f"{settings.embedding_backend.split(' ')[0]} embeddings · k={settings.top_k}"
    )

    t_ask, t_ins, t_eval, t_about = st.tabs(["💬 Ask", "📑 Paper insights", "📊 Evaluate", "ℹ️ About"])
    with t_ask:
        tab_ask(settings, index)
    with t_ins:
        tab_insights(settings, index)
    with t_eval:
        tab_evaluate(settings, index)
    with t_about:
        tab_about()


main()
