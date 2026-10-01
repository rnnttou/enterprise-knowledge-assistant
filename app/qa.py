"""Step 4: answer questions with citations, grounded in retrieved chunks.

The chat model only ever sees the top-k chunks returned by search(), never
the whole corpus, and is instructed to answer from them alone and cite the
excerpt number(s) it used. That keeps answers checkable against a page and
makes "I don't know" the expected response when nothing relevant is found.

The main entry point is answer(), which returns a plain dict:
    {"answer": str, "citations": [{marker, source, page, snippet}], "refused": bool}

Usage:
    python -m app.qa "How many vacation days do employees get?"
    python -m app.qa "salary bands" --groups hr
"""

import argparse
import re
import time
from dataclasses import dataclass, field

from google import genai
from google.genai import errors, types

from app.config import CHAT_MODEL, GEMINI_API_KEY, MAX_DISTANCE
from app.vector_store import get_collection, hybrid_search, search

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using ONLY the numbered "
    "excerpts below. Do not use outside knowledge. After each fact, cite the "
    "excerpt number(s) it came from in square brackets, e.g. [1] or [2][3]. "
    "If the excerpts don't contain the answer, say so instead of guessing. "
    "You may use the previous conversation to understand follow-up questions "
    "(e.g. what 'it' or 'they' refers to), but every fact must still come "
    "from the numbered excerpts, not from the previous conversation."
)

MAX_RETRIES = 6
SNIPPET_CHARS = 200   # how much of each cited chunk to show the user
NO_RESULTS_MESSAGE = "I couldn't find anything relevant in the indexed documents."
REFUSAL_MESSAGE = "I couldn't find this in the documents."

# How many previous turns feed into retrieval for a follow-up question, e.g.
# "What about for part-time staff?" needs the prior question to search well.
HISTORY_TURNS_FOR_RETRIEVAL = 2

# Matches citation markers like "[3]" or "[1, 2]" in the model's answer.
CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class Answer:
    text: str
    sources: list[dict] = field(default_factory=list)


def build_context(hits: list[dict]) -> str:
    """Numbered excerpts the model can cite, e.g. "[1] handbook.pdf, p. 3: ..."."""
    return "\n\n".join(
        f"[{i}] {hit['source']}, p. {hit['page']}: {hit['text']}"
        for i, hit in enumerate(hits, start=1)
    )


def build_history(history: list[dict] | None) -> str:
    """Prior Q&A turns, formatted so the model can resolve follow-up questions."""
    if not history:
        return ""
    turns = "\n\n".join(f"Q: {turn['question']}\nA: {turn['answer']}" for turn in history)
    return f"Previous conversation:\n{turns}\n\n"


def contextualize_query(question: str, history: list[dict] | None) -> str:
    """Fold recent questions into the retrieval query for follow-ups.

    A question like "What about for part-time staff?" shares almost no words
    with the relevant chunk on its own; prepending the prior question(s)
    gives hybrid_search something to match against.
    """
    if not history:
        return question
    prior_questions = [turn["question"] for turn in history[-HISTORY_TURNS_FOR_RETRIEVAL:]]
    return "\n".join([*prior_questions, question])


def clean_citations(text: str, num_sources: int) -> tuple[str, list[int]]:
    """Drop citation markers that don't point at a chunk we actually sent.

    The model sometimes invents a marker like [7] when only 5 excerpts exist.
    We remove those so every citation the user sees is real, and return the
    valid marker numbers in order, e.g. ("... [1][3]", [1, 3]).
    """
    used: list[int] = []

    def fix(match: re.Match) -> str:
        numbers = [int(n) for n in match.group(1).split(",")]
        valid = [n for n in numbers if 1 <= n <= num_sources]
        for n in valid:
            if n not in used:
                used.append(n)
        # "[1, 9]" becomes "[1]"; "[9]" disappears entirely.
        return "".join(f"[{n}]" for n in valid)

    cleaned = CITATION_PATTERN.sub(fix, text)
    # Tidy the gaps left by removed markers: "days ." -> "days." and double spaces.
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip(), used


class GeminiAnswerer:
    def __init__(self, api_key: str = GEMINI_API_KEY, model: str = CHAT_MODEL):
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def answer(self, question: str, hits: list[dict], history: list[dict] | None = None) -> str:
        if not hits:
            return NO_RESULTS_MESSAGE
        prompt = f"{build_history(history)}{build_context(hits)}\n\nQuestion: {question}"
        config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
        response = self._call_with_retry(prompt, config)
        return response.text

    def _call_with_retry(self, prompt: str, config: types.GenerateContentConfig):
        for attempt in range(MAX_RETRIES):
            try:
                return self.client.models.generate_content(model=self.model, contents=prompt, config=config)
            except (errors.ClientError, errors.ServerError) as exc:
                # 429 = free-tier rate limit, 5xx = temporary server trouble. Anything else is a real error.
                retryable = exc.code == 429 or exc.code >= 500
                if not retryable or attempt == MAX_RETRIES - 1:
                    raise
                wait = min(2 ** attempt * 5, 60)
                print(f"  Gemini returned {exc.code}, retrying in {wait}s...")
                time.sleep(wait)


def ask(question: str, groups: list[str] | None = None, k: int = 5, *, collection=None, embedder=None, answerer=None) -> Answer:
    from app.embeddings import GeminiEmbedder

    collection = collection or get_collection()
    embedder = embedder or GeminiEmbedder()
    answerer = answerer or GeminiAnswerer()

    hits = search(collection, embedder, question, groups, k)
    text = answerer.answer(question, hits)
    return Answer(text=text, sources=hits)


def answer(
    question: str,
    groups: list[str] | None = None,
    k: int = 5,
    *,
    history: list[dict] | None = None,
    collection=None,
    embedder=None,
    answerer=None,
) -> dict:
    """Answer a question from the documents the user's groups may read.

    Returns {"answer": str, "citations": [...], "refused": bool}. Each citation is
    {"marker": 1, "source": "employee-handbook.pdf", "page": 18, "snippet": "..."}.
    `history` is the conversation so far, as [{"question": ..., "answer": ...}, ...]
    oldest first; pass it so a follow-up like "What about managers?" can be
    retrieved and answered in context. The keyword arguments let tests pass
    fakes instead of Gemini and ChromaDB.
    """
    from app.embeddings import GeminiEmbedder

    collection = collection or get_collection()
    embedder = embedder or GeminiEmbedder()

    # 1. Retrieve the top-k chunks, blending vector and keyword search (Step 7)
    #    over a query that folds in recent questions (Step 8) for follow-ups,
    #    already filtered by the user's access groups.
    retrieval_query = contextualize_query(question, history)
    hits = hybrid_search(collection, embedder, retrieval_query, groups, k)

    # 2. If even the closest chunk is far away, the documents probably don't
    #    cover this question. Refuse without spending a chat-model call.
    if not hits or hits[0]["distance"] > MAX_DISTANCE:
        return {"answer": REFUSAL_MESSAGE, "citations": [], "refused": True}

    # 3. Ask the model, giving it only these numbered chunks plus any history.
    answerer = answerer or GeminiAnswerer()
    raw_text = answerer.answer(question, hits, history)

    # 4. Remove any [n] that doesn't match a chunk we sent.
    text, markers = clean_citations(raw_text, len(hits))

    # 5. Turn each valid marker back into its source, page and a short snippet.
    citations = [
        {
            "marker": n,
            "source": hits[n - 1]["source"],
            "page": hits[n - 1]["page"],
            "snippet": hits[n - 1]["text"][:SNIPPET_CHARS],
        }
        for n in sorted(markers)
    ]

    # An answer with no valid citation isn't grounded in the documents
    # (usually the model saying "the excerpts don't say"), so flag it as refused.
    return {"answer": text, "citations": citations, "refused": not citations}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about the indexed documents.")
    parser.add_argument("question")
    parser.add_argument("--groups", default="", help="comma-separated groups, e.g. hr,finance")
    parser.add_argument("-k", type=int, default=5, help="number of chunks to retrieve")
    args = parser.parse_args()

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    result = answer(args.question, groups, args.k)

    print(result["answer"])
    if result["citations"]:
        print("\nSources:")
        for c in result["citations"]:
            snippet = c["snippet"].replace("\n", " ")
            print(f"  [{c['marker']}] {c['source']}, p. {c['page']}: {snippet}...")


if __name__ == "__main__":
    main()
