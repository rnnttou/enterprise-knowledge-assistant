"""Step 3a: turn text into vectors with the Gemini embedding API.

Chunks and questions are embedded with different task types, which Gemini
uses to place questions close to the passages that answer them.
"""

import math
import time

from google import genai
from google.genai import errors, types

from app.config import EMBED_BATCH_SIZE, EMBEDDING_DIM, EMBEDDING_MODEL, GEMINI_API_KEY

MAX_RETRIES = 6


def _normalize(vector: list[float]) -> list[float]:
    # Gemini only normalizes full-size (3072) vectors; do it ourselves for smaller ones.
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class GeminiEmbedder:
    def __init__(
        self,
        api_key: str = GEMINI_API_KEY,
        model: str = EMBEDDING_MODEL,
        dim: int = EMBEDDING_DIM,
        batch_size: int = EMBED_BATCH_SIZE,
    ):
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.dim = dim
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            response = self._call_with_retry(batch, config)
            vectors.extend(_normalize(e.values) for e in response.embeddings)
        return vectors

    def _call_with_retry(self, batch: list[str], config: types.EmbedContentConfig):
        for attempt in range(MAX_RETRIES):
            try:
                return self.client.models.embed_content(model=self.model, contents=batch, config=config)
            except (errors.ClientError, errors.ServerError) as exc:
                # 429 = free-tier rate limit, 5xx = temporary server trouble. Anything else is a real error.
                retryable = exc.code == 429 or exc.code >= 500
                if not retryable or attempt == MAX_RETRIES - 1:
                    raise
                wait = min(2 ** attempt * 5, 60)
                print(f"  Gemini returned {exc.code}, retrying in {wait}s...")
                time.sleep(wait)
