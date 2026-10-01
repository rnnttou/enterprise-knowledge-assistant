"""Step 5: FastAPI backend exposing the question-answering pipeline over HTTP.

Run with:
    uvicorn app.api:app --reload
"""

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.qa import answer as answer_question
from app.vector_store import get_collection, index_documents

app = FastAPI(title="Enterprise Knowledge Assistant", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    groups: list[str] = Field(default_factory=list)
    k: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    marker: int      # the [n] used in the answer text
    source: str      # file path inside data/, e.g. "employee-handbook.pdf"
    page: int
    snippet: str     # start of the cited chunk, so the user can check it


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    refused: bool    # True when the documents don't contain the answer


class IndexResponse(BaseModel):
    indexed: list[str]
    skipped: list[str]
    removed: list[str]
    empty: list[str]
    chunks_added: int


# Indirection so tests can swap in fakes with app.dependency_overrides,
# instead of hitting Gemini and a real ChromaDB on every request.
def get_ask():
    return answer_question


def get_indexer():
    def run_indexing() -> dict:
        from app.embeddings import GeminiEmbedder

        collection = get_collection()
        stats = index_documents(collection, GeminiEmbedder())
        return vars(stats)

    return run_indexing


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, ask_fn=Depends(get_ask)) -> AskResponse:
    try:
        result = ask_fn(request.question, request.groups, request.k)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    # answer() already returns {"answer", "citations", "refused"}, matching AskResponse.
    return AskResponse(**result)


@app.post("/index", response_model=IndexResponse)
def reindex(indexer=Depends(get_indexer)) -> IndexResponse:
    try:
        stats = indexer()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return IndexResponse(**stats)
