# 📚 LLM-Powered Research Assistant

A Streamlit app that lets research teams **upload long scientific PDFs and ask questions about them in plain English**.
Answers come from Retrieval-Augmented Generation (RAG): they are grounded in the papers and cite the file and page they came from.
The app also extracts each paper's **metadata and summary**, finds **trends across the collection**, and **evaluates** RAG against a plain LLM with an LLM-as-a-judge.

See [docs/EXPLANATION.md](docs/EXPLANATION.md) for a walkthrough of the code and how each notebook step maps to it.

---

## The problem

R&D teams can't keep up with the volume of new publications. Reading, summarising and pulling insights out of long papers by hand is slow and delays decisions.
The goal is a prototype that:

* answers questions by retrieving the relevant passages from long research papers,
* accepts natural-language queries, so nobody has to read the whole document first,
* simplifies literature review by extracting summaries, metadata and emerging trends.

The reference dataset is four prompt-engineering papers:

| # | Paper | Authors | arXiv |
|---|---|---|---|
| 1 | Language Models are Few-Shot Learners | Brown et al. (OpenAI) | [2005.14165](https://arxiv.org/abs/2005.14165) |
| 2 | The Power of Scale for Parameter-Efficient Prompt Tuning | Lester, Al-Rfou, Constant (Google) | [2104.08691](https://arxiv.org/abs/2104.08691) |
| 3 | AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts | Shin et al. | [2010.15980](https://arxiv.org/abs/2010.15980) |
| 4 | Prompt Programming for Large Language Models: Beyond the Few-Shot Paradigm | Reynolds & McDonell | [2102.07350](https://arxiv.org/abs/2102.07350) |

The app works with **any** text-based PDFs, not only these four.

## Features

| Tab | What it does |
|---|---|
| 💬 **Ask** | Chat with your papers. Answers stream in and cite `[Source n]`. An expandable panel shows each retrieved chunk with its file, page and relevance score. You can limit the search to some papers, or switch on **Compare with no-RAG answer** to see what the LLM says without context. |
| 📑 **Paper insights** | For each paper, pulls out the title, authors, year, venue/arXiv ID, keywords, a summary, key findings and limitations. **Analyse trends** writes a literature-review brief covering common themes, differences, emerging trends and gaps. Results download as CSV/JSON. |
| 📊 **Evaluate** | The notebook's experiment, built in. Each question gets a base (no-RAG) answer and a RAG answer. An LLM judge scores both 1–5 for **groundedness** and **relevance**. You get averages, a chart, the judge's reasoning and a CSV. |
| ℹ️ **About** | How the pipeline works. |

Other details:

* **Two LLM providers**: Groq (free tier) or OpenAI. Groq is called through its OpenAI-compatible endpoint, so one client serves both. You can type in any model name.
* **No-key embeddings**: the default embedding model is all-MiniLM-L6-v2 running on ONNX (bundled with ChromaDB). It needs no API key and no PyTorch. OpenAI `text-embedding-3-small` is optional.
* **Tunable retrieval**: chunk size, overlap, *k*, temperature and max tokens are sidebar sliders. If you change the embedding or chunk settings, the index is rebuilt automatically.
* **Private by design**: uploaded papers stay in the browser session's memory and are never written to disk.

## Architecture

```
          ┌──────────────┐   pages + metadata   ┌───────────────┐  ~512-token chunks  ┌──────────────────┐
PDFs ───▶ │ PyMuPDF load │ ───────────────────▶ │ tiktoken split│ ──────────────────▶ │ Embeddings       │
          └──────────────┘                      └───────────────┘                     │ MiniLM / OpenAI  │
                                                                                      └────────┬─────────┘
                                                                                               ▼
question ──▶ embed ──▶ Chroma similarity search (top-k, cosine) ◀──────────────────── Chroma (in-memory)
                              │
                              ▼  labelled excerpts  [Source i | file | page]
                     ┌─────────────────────┐
                     │ Grounded prompt     │──▶ LLM (Groq / OpenAI) ──▶ answer + citations
                     └─────────────────────┘
                              │
                              ▼
                   LLM-as-a-judge (groundedness, relevance) · optional RAGAS
```

## Project structure

```
llm-research-assistant/
├── app.py                         # Streamlit UI (sidebar, Ask / Insights / Evaluate / About tabs)
├── research_assistant/
│   ├── config.py                  # Settings dataclass, provider presets, env loading
│   ├── prompts.py                 # System/user prompts for Q&A, insights and the judge
│   ├── llm.py                     # OpenAI-SDK client for Groq/OpenAI, streaming, <think> stripping
│   ├── embeddings.py              # Local ONNX MiniLM or OpenAI embeddings
│   ├── document_loader.py         # PDF → page Documents → token chunks
│   ├── vector_store.py            # PaperIndex: Chroma collection + per-paper bookkeeping
│   ├── rag.py                     # retrieve → build prompt → answer (and the no-RAG baseline)
│   ├── insights.py                # metadata/summary extraction and cross-paper trends
│   └── evaluation.py              # LLM-as-a-judge with robust score parsing
├── scripts/
│   ├── download_papers.py         # fetch the 4 case-study papers from arXiv into data/papers/
│   └── evaluate_cli.py            # command-line base-vs-RAG evaluation (+ optional RAGAS)
├── tests/test_core.py             # offline unit tests (no API key needed)
├── docs/EXPLANATION.md            # code walkthrough and notebook → app mapping
├── data/papers/                   # put PDFs here (git-ignored)
├── .streamlit/config.toml         # upload limit, theme
├── .vscode/                       # interpreter, run/debug configs
├── Dockerfile                     # local Docker or Hugging Face Spaces
├── requirements.txt               # runtime dependencies
├── requirements-dev.txt           # + pytest, ragas, datasets
└── .env.example                   # copy to .env and add your key
```

## Quick start

### 1. Clone and install

```powershell
git clone https://github.com/<your-username>/llm-research-assistant.git
cd llm-research-assistant
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

(With [uv](https://docs.astral.sh/uv/): `uv venv .venv` then `uv pip install -r requirements.txt`.)

### 2. Add an API key

```powershell
copy .env.example .env            # macOS/Linux: cp .env.example .env
```

Fill in `GROQ_API_KEY` (free at <https://console.groq.com/keys>) **or** `OPENAI_API_KEY`.
You can also paste a key into the sidebar at runtime.

### 3. (Optional) Get the case-study papers

```powershell
python scripts/download_papers.py
```

### 4. Run

```powershell
streamlit run app.py
```

Open <http://localhost:8501>, upload PDFs (or click **Load PDFs from data/papers**) and ask a question.
The first run downloads the ~80 MB embedding model once.

In **VS Code**, open the folder, choose the `.venv` interpreter, and press **F5** with the *Streamlit: app.py* configuration.

## Evaluation

**In the app**: open the 📊 Evaluate tab and click *Run evaluation*. It uses the notebook's three questions by default.

**From the command line**:

```powershell
python scripts/evaluate_cli.py                 # LLM-as-a-judge, writes evaluation_results.csv
pip install -r requirements-dev.txt
python scripts/evaluate_cli.py --ragas         # + RAGAS faithfulness / answer relevancy / context precision
```

What to expect (the same pattern the notebook found):

* **Groundedness** is much higher with RAG. The base model writes plausible but generic or unsupported statements, while RAG answers stick to the retrieved passages.
* **Relevance** is high for both. A strong LLM can address the question either way, but only RAG answers can be traced back to the source.

## Tests

```powershell
pytest -q
```

The tests cover PDF loading, page metadata, token chunking, judge-score parsing, JSON extraction and prompt building. They run offline.

## Deploy

**Docker**

```powershell
docker build -t research-assistant .
docker run -p 8501:8501 --env-file .env research-assistant
```

**Hugging Face Spaces**

1. Create a new Space and choose **Docker** as the SDK.
2. Push this repository to the Space (or upload `app.py`, `research_assistant/`, `requirements.txt`, `Dockerfile` and `.streamlit/`).
3. Under *Settings → Variables and secrets*, add `GROQ_API_KEY` or `OPENAI_API_KEY` as a secret.
4. The Space builds and serves the app on port 8501.

**Streamlit Community Cloud**: point it at `app.py` and add the key under *Secrets*.

## Configuration

Every setting can be changed in the sidebar or in `.env`:

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` / `OPENAI_API_KEY` | – | Provider key. The provider whose key is set is chosen by default. |
| `LLM_PROVIDER` | auto | Force `Groq` or `OpenAI`. |
| `GROQ_MODEL` / `OPENAI_MODEL` | `qwen/qwen3.8-27b` / `gpt-4o-mini` | Chat model. |
| `JUDGE_MODEL` | chat model | Model used to grade answers. A stronger one is better. |
| `REASONING_EFFORT` | – | Passed to reasoning models (e.g. `none` for Qwen3 on Groq). |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 512 / 64 | Chunk size and overlap, in tokens. |
| `TOP_K` | 5 | Number of chunks retrieved per question. |
| `TEMPERATURE` / `MAX_TOKENS` | 0.3 / 700 | Generation settings. |

## Limitations and next steps

* Scanned, image-only PDFs have no text layer. Add OCR (e.g. Tesseract) to support them.
* Tables, equations and figures lose their structure when converted to plain text.
* The index is in memory per session. For a shared, growing library, use a persistent Chroma path or a hosted vector DB.
* Possible improvements: hybrid BM25 + vector search, a cross-encoder re-ranker, and per-section chunking (Abstract / Method / Results).

## Acknowledgements

The case study comes from the UT Austin McCombs *Generative AI Applications* programme. The papers belong to their authors.
