"""Step 4: answer questions with citations, grounded in retrieved chunks.

The chat model only ever sees the top-k chunks returned by search(), never
the whole corpus, and is instructed to answer from them alone and cite the
excerpt number(s) it used. That keeps answers checkable against a page and
makes "I don't know" the expected response when nothing relevant is found.

Usage:
    python -m app.qa "How many vacation days do employees get?"
    python -m app.qa "salary bands" --groups hr
"""

import argparse
import time
from dataclasses import dataclass, field

from google import genai
from google.genai import errors, types

from app.config import CHAT_MODEL, GEMINI_API_KEY
from app.vector_store import get_collection, search

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using only the numbered "
    "excerpts below. Cite the excerpt number(s) you used in square brackets, "
    "e.g. [1] or [2][3], right after each fact they support. If the excerpts "
    "don't contain the answer, say so instead of guessing."
)

MAX_RETRIES = 6
NO_RESULTS_MESSAGE = "I couldn't find anything relevant in the indexed documents."


@dataclass
class Answer:
    text: str
    sources: list[dict] = field(default_factory=list)


def build_context(hits: list[dict]) -> str:
    """Numbered excerpts the model can cite by index, e.g. "[1] Handbook, p. 3"."""
    return "\n\n".join(
        f"[{i}] {hit['title']}, p. {hit['page']}\n{hit['text']}"
        for i, hit in enumerate(hits, start=1)
    )


class GeminiAnswerer:
    def __init__(self, api_key: str = GEMINI_API_KEY, model: str = CHAT_MODEL):
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def answer(self, question: str, hits: list[dict]) -> str:
        if not hits:
            return NO_RESULTS_MESSAGE
        prompt = f"{build_context(hits)}\n\nQuestion: {question}"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about the indexed documents.")
    parser.add_argument("question")
    parser.add_argument("--groups", default="", help="comma-separated groups, e.g. hr,finance")
    parser.add_argument("-k", type=int, default=5, help="number of chunks to retrieve")
    args = parser.parse_args()

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    result = ask(args.question, groups, args.k)

    print(result.text)
    if result.sources:
        print("\nSources:")
        for i, hit in enumerate(result.sources, start=1):
            print(f"  [{i}] {hit['title']}, p. {hit['page']} ({hit['source']})")


if __name__ == "__main__":
    main()
