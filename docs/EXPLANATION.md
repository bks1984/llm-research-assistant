# How the Research Assistant Works

This document follows a question through the code, explains the design choices, and maps each step of the original case-study notebook to where it now lives.

---

## 1. Concepts in one minute

**Why not just ask the LLM?** An LLM answers from what it learned in training. It can't see your PDFs, it may be out of date, and when it doesn't know something it often makes up an answer that sounds right. In the notebook, base-prompt answers scored low on *groundedness* for exactly this reason.

**Retrieval-Augmented Generation (RAG)** fixes this by doing a search first:

1. Cut the papers into small passages (*chunks*).
2. Turn each chunk into a vector (*embedding*) that captures its meaning.
3. When a question comes in, embed it too and fetch the chunks whose vectors are closest (*semantic search*).
4. Give those chunks to the LLM with strict instructions: *answer only from this context, and cite it*.

The LLM now acts as a reader and summariser of the evidence, not as an oracle.

---

## 2. A question's journey through the code

### Step 1: Load the PDFs (`research_assistant/document_loader.py`)

```python
with fitz.open(stream=data, filetype="pdf") as pdf:
    for i, page in enumerate(pdf):
        Document(page_content=page.get_text("text"), metadata={"source": file_name, "page": i + 1, ...})
```

* **PyMuPDF** (`fitz`) is fast and handles two-column academic layouts well. The notebook also used it, through LangChain's `PyMuPDFLoader`.
* Each page becomes one `Document` with **file name and page number** attached. These are what make citations like `prompt_tuning.pdf, page 4` possible.
* Blank or image-only pages are skipped, so empty chunks never reach the index.
* Files arrive as bytes from Streamlit's uploader, so nothing is written to a temporary file. (The notebook's app wrote temp files.)

### Step 2: Chunk (`split_documents`)

```python
RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name="cl100k_base", chunk_size=512, chunk_overlap=64)
```

* Chunks are measured in **tokens**, not characters (same as the notebook). 512 tokens is about one or two dense paragraphs: enough context to be meaningful, small enough to be precise.
* The **recursive** splitter tries paragraph breaks first, then lines, then sentences, then words, so chunks rarely cut a sentence in half.
* **Overlap (64 tokens)** is new. The notebook used none. Overlap stops an idea that straddles a chunk boundary from being lost.

### Step 3: Embed and index (`embeddings.py`, `vector_store.py`)

* **Default embeddings:** `all-MiniLM-L6-v2` through ChromaDB's built-in ONNX runtime. The notebook offered it as the open-source alternative to OpenAI embeddings. Running it on ONNX means **no API key, no PyTorch download, and it runs on CPU**, which matters on free hosting.
* **Optional:** OpenAI `text-embedding-3-small`, if you have an OpenAI key.
* `PaperIndex` wraps a LangChain `Chroma` store configured for **cosine** distance. Each session gets a uniquely named collection, so two browser tabs never see each other's papers.
* It also keeps the page-level documents for each paper. The Insights tab needs a paper's opening pages, not random chunks.

### Step 4: Retrieve (`rag.py → retrieve`)

```python
index.search(question, k=5, files=[...])   # similarity_search_with_relevance_scores
```

* Returns the top-*k* chunks **with a relevance score** (0–1). The UI shows the score, so you can tell whether the papers really cover the question.
* An optional **metadata filter** limits the search to selected papers (Chroma `where={"source": {"$in": [...]}}`).

### Step 5: Build the prompt and generate (`prompts.py`, `rag.py`, `llm.py`)

Each chunk is labelled before it goes into the prompt:

```
[Source 2 | power_of_scale_prompt_tuning.pdf | page 4]
<chunk text>
---
[Source 3 | autoprompt.pdf | page 2]
...
```

The system prompt is the notebook's `qna_system_message`, extended to require citations by label:

* use **only** the context,
* cite sources,
* reply exactly *"Sorry, this is out of my knowledge base."* when the context doesn't contain the answer,
* never invent numbers, authors or titles.

`LLMClient` (in `llm.py`) uses the **OpenAI Python SDK** for both providers. Groq exposes an OpenAI-compatible endpoint (`https://api.groq.com/openai/v1`), so switching provider only changes `base_url` and the key. It also:

* **streams** tokens to the UI (`st.write_stream`),
* **removes `<think>…</think>`** blocks that reasoning models (e.g. Qwen3) emit, including when the tag is split across stream chunks,
* passes `reasoning_effort` through when it is configured.

### Step 6: Show it (`app.py`)

The answer streams into the chat. Below it, a **Sources** expander lists every retrieved chunk with its file, page and score. With **Compare with no-RAG answer** switched on, the same question also goes to the LLM with no context, which makes the hallucination problem easy to see side by side.

---

## 3. Paper insights (`insights.py`)

This covers the *"extract summaries, metadata, and emerging trends"* part of the business brief, which the notebook described but did not implement.

1. **Per paper**: the first ~14,000 characters of the paper (title, authors, abstract, introduction) go to the LLM with a prompt asking for **JSON only**: `title, authors, year, venue_or_id, keywords, summary, key_findings, limitations`. JSON mode (`response_format={"type": "json_object"}`) is tried first. If the endpoint rejects it, the call is retried as plain text. `parse_json_object` accepts ```json fences or extra text around the object.
2. **Across papers**: the structured summaries are sent back to the LLM to write a short literature-review brief with *common themes, how approaches differ, emerging trends, open questions*.

Working from structured summaries instead of raw text keeps the trend prompt small, and keeps every claim traceable to one paper's summary.

---

## 4. Evaluation (`evaluation.py`, Evaluate tab, `scripts/evaluate_cli.py`)

### LLM-as-a-judge (from the notebook)

For each question:

1. Retrieve the top-*k* context.
2. Produce a **base answer** (LLM only) and a **RAG answer**.
3. Ask a judge LLM to rate each answer **1–5** on:
   * **Groundedness**: is the answer derived only from the context?
   * **Relevance**: does it address the main aspects of the question?

**One improvement over the notebook:** the notebook read the score with `response[-2]`, i.e. the second-to-last character. That breaks as soon as the judge adds a trailing newline, a space or quotes. `parse_score` uses a regex for `groundedness_score: N` and falls back to a lone digit on the last line. The unit tests cover these cases.

### RAGAS (optional)

`python scripts/evaluate_cli.py --ragas` runs the notebook's three reference-free RAGAS metrics, scored 0–1:

| Metric | Question it answers |
|---|---|
| Faithfulness | Are the answer's claims supported by the retrieved contexts? |
| Answer Relevancy | How well does the answer address the question? |
| LLM Context Precision (w/o reference) | Were the retrieved chunks actually useful for the answer? |

RAGAS is in `requirements-dev.txt`, not `requirements.txt`, because it pulls in heavy dependencies that the deployed app doesn't need.

---

## 5. Notebook → project mapping

| Notebook section | Where it lives now | What changed |
|---|---|---|
| Install & import libraries | `requirements.txt`, `requirements-dev.txt` | Current package versions; LangChain only where it helps (splitter, Chroma wrapper). |
| Read `config.json`, set env vars | `config.py` (`.env` + `Settings`) | `.env` / env vars / sidebar input; Groq **or** OpenAI. |
| `response()`: plain LLM call | `rag.base_answer`, `llm.LLMClient.complete` | Used as the baseline in Compare mode and in the Evaluate tab. |
| Unzip & `PyMuPDFLoader` | `document_loader.load_pdf_bytes` | Reads uploaded bytes directly and records file and page metadata. |
| `RecursiveCharacterTextSplitter.from_tiktoken_encoder` | `document_loader.split_documents` | Adds overlap and a chunk ID. |
| `OpenAIEmbeddings` / SentenceTransformer option | `embeddings.py` | Local ONNX MiniLM by default; OpenAI optional. |
| `Chroma.from_documents`, `as_retriever(k=5)` | `vector_store.PaperIndex` | Cosine space, relevance scores, per-paper filter, per-session collection. |
| `qna_system_message`, `qna_user_message_template` | `prompts.py` | Source labels and citation instructions. |
| `generate_rag_response()` | `rag.retrieve` + `rag.answer` / `answer_stream` | Streaming; empty context returns the out-of-scope reply. |
| Mistral / HF local model option | Groq provider | Serves open-weight models (Qwen, Llama…) without a GPU. |
| LLM-as-a-judge prompts & function | `prompts.py`, `evaluation.judge` | Fixed score parsing; judge reasoning shown in the UI. |
| Base vs RAG DataFrames & averages | Evaluate tab, `evaluate_cli.py` | Metrics, bar chart, CSV export. |
| RAGAS evaluation | `evaluate_cli.py --ragas` | Same three metrics. |
| Streamlit app + Dockerfile + HF Spaces | `app.py`, `Dockerfile`, README *Deploy* | Multi-tab UI, non-root Docker user as HF Spaces requires. |
| "Summaries, metadata, emerging trends" (business brief) | `insights.py`, Paper insights tab | **New**: implemented, not just described. |

---

## 6. Design decisions and trade-offs

* **OpenAI SDK instead of LangChain chat models.** One small, well-understood client covers both providers, gives direct control over streaming, and avoids version churn in LangChain integration packages.
* **In-memory index.** Uploaded papers are private to each session and nothing sensitive is left on disk. The trade-off is re-indexing after a restart, which takes seconds for a handful of papers.
* **Session-level re-indexing.** The raw PDF bytes are kept in session state. Changing chunk size or the embedding model rebuilds the index without re-uploading.
* **Duplicate detection.** Files are hashed (SHA-256), so uploading the same PDF under a different name doesn't double its weight in retrieval.
* **Temperature 0.3 for answers, 0 for judging and extraction.** Answers can read naturally, while scores and JSON stay deterministic.
