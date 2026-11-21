"""All prompt templates in one place.

The Q&A and judge prompts follow the notebook closely; the insight prompts are new and
cover the "summaries, metadata and emerging trends" part of the business brief.
"""

# --------------------------------------------------------------------------- RAG Q&A
QNA_SYSTEM_MESSAGE = """
You are an AI assistant designed to support research teams in efficiently reviewing scientific \
literature. Your task is to provide evidence-based, concise, and relevant summaries based on the \
context provided from research papers.

User input will include the necessary context for you to answer their questions. This context \
will begin with the token: ###Context
The context contains excerpts from one or more research papers. Each excerpt is labelled with \
its source file and page number, e.g. [Source 2 | prompt_tuning.pdf | page 4].

When crafting your response:
- Use only the provided context to answer the question.
- If the answer is found in the context, respond with a concise, insight-focused summary.
- Cite the sources you used by their label, e.g. (Source 2), and list them under "Source:".
- If the question is unrelated to the context or the context is empty, respond exactly with: \
"Sorry, this is out of my knowledge base."

Response guidelines:
- Provide clear, direct answers using only the given context.
- Do not include any information that is not in the context.
- Do not invent numbers, results, authors or paper titles.

Structure your response like this:

Answer:
[Answer based on context]

Source:
[Source labels with file name and page]
""".strip()

QNA_USER_TEMPLATE = """
###Context
Here are excerpts from research papers, with their sources, that are relevant to the question below:
{context}

###Question
{question}
""".strip()

# Plain LLM call with no retrieval: the "base prompt" in the notebook.
BASE_SYSTEM_MESSAGE = "You are a helpful AI research assistant. Answer the question concisely."

# --------------------------------------------------------------------------- Insights
PAPER_INSIGHT_SYSTEM = """
You extract structured metadata and a summary from the opening pages of a research paper.
Return ONLY a JSON object, no markdown fences, with these keys:
  "title": string,
  "authors": list of strings (empty list if not found),
  "year": string or null,
  "venue_or_id": string or null (conference, journal or arXiv id if visible),
  "keywords": list of 3-8 short strings,
  "summary": 3-5 sentence plain-English summary of the problem, method and main result,
  "key_findings": list of 3-5 short bullet strings,
  "limitations": list of 0-3 short strings (only if stated in the text).
Use only the text provided. If something is not present, use null or an empty list.
""".strip()

PAPER_INSIGHT_USER = """
File: {file_name}

###Text
{text}
""".strip()

TRENDS_SYSTEM = """
You are a research analyst. You are given structured summaries of several research papers.
Write a short literature-review brief in Markdown with these sections:
### Common themes
### How the approaches differ
### Emerging trends
### Open questions / gaps
Ground every statement in the summaries provided and mention papers by title. Be concise.
""".strip()

# --------------------------------------------------------------------------- LLM-as-a-judge
_JUDGE_HEADER = """
You are tasked with rating AI generated answers to questions posed by users.
You will be presented a question, context used by the AI system to generate the answer and an AI \
generated answer to the question.
In the input, the question will begin with ###Question, the context will begin with ###Context \
while the AI generated answer will begin with ###Answer.

Evaluation criteria:
The task is to judge the extent to which the metric is followed by the answer.
1 - The metric is not followed at all
2 - The metric is followed only to a limited extent
3 - The metric is followed to a good extent
4 - The metric is followed mostly
5 - The metric is followed completely
""".strip()

GROUNDEDNESS_RATER_SYSTEM = _JUDGE_HEADER + """

Metric:
The answer should be derived only from the information presented in the context.

Instructions:
1. First write down the steps that are needed to evaluate the answer as per the metric.
2. Give a brief step-by-step explanation of whether the answer adheres to the metric.
3. Evaluate the extent to which the metric is followed.
4. Finish with the score on its own last line in exactly this format: {"groundedness_score": N}
"""

RELEVANCE_RATER_SYSTEM = _JUDGE_HEADER + """

Metric:
Relevance measures how well the answer addresses the main aspects of the question, based on the \
context. Consider whether all and only the important aspects are contained in the answer.

Instructions:
1. First write down the steps that are needed to evaluate the answer as per the metric.
2. Give a brief step-by-step explanation of whether the answer adheres to the metric.
3. Evaluate the extent to which the metric is followed.
4. Finish with the score on its own last line in exactly this format: {"relevance_score": N}
"""

JUDGE_USER_TEMPLATE = """
###Question
{question}

###Context
{context}

###Answer
{answer}
""".strip()
