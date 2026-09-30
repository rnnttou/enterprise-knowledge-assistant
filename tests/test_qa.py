"""QA tests with fake embedder and chat model, so they run offline and cost nothing."""

import hashlib

import chromadb
import pytest

from app.qa import NO_RESULTS_MESSAGE, Answer, build_context, ask
from app.vector_store import get_collection, index_documents
from tests.test_vector_store import write_pdf


class FakeEmbedder:
    model = "fake-embedder"

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.lower().encode()).digest()
        return [b / 255 for b in digest[:8]]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


class FakeAnswerer:
    def __init__(self):
        self.calls = []

    def answer(self, question: str, hits: list[dict]) -> str:
        self.calls.append((question, hits))
        if not hits:
            return NO_RESULTS_MESSAGE
        return f"Answer using [1] from {hits[0]['title']}."


@pytest.fixture
def collection():
    client = chromadb.EphemeralClient()
    col = get_collection(client, model="fake-embedder", dim=8)
    yield col
    client.delete_collection(col.name)


@pytest.fixture
def indexed(collection, tmp_path):
    write_pdf(tmp_path / "handbook.pdf", ["Holiday policy page one"])
    index_documents(collection, FakeEmbedder(), tmp_path)
    return collection


def test_build_context_numbers_excerpts_with_title_and_page():
    hits = [
        {"title": "Handbook", "page": 3, "text": "Employees get 25 days."},
        {"title": "Handbook", "page": 4, "text": "Unused days expire in March."},
    ]
    context = build_context(hits)
    assert "[1] Handbook, p. 3\nEmployees get 25 days." in context
    assert "[2] Handbook, p. 4\nUnused days expire in March." in context


def test_build_context_of_no_hits_is_empty():
    assert build_context([]) == ""


def test_ask_returns_answer_and_sources(indexed):
    answerer = FakeAnswerer()
    result = ask(
        "How many vacation days?",
        collection=indexed,
        embedder=FakeEmbedder(),
        answerer=answerer,
    )

    assert isinstance(result, Answer)
    assert "handbook" in result.text
    assert len(result.sources) == 1
    assert result.sources[0]["source"] == "handbook.pdf"
    assert answerer.calls[0][0] == "How many vacation days?"


def test_ask_respects_group_filter(collection, tmp_path):
    write_pdf(tmp_path / "handbook.pdf", ["Holiday policy page one"])
    write_pdf(tmp_path / "hr" / "salaries.pdf", ["Salary bands are confidential"])
    index_documents(collection, FakeEmbedder(), tmp_path)

    public_result = ask("salaries", groups=[], collection=collection, embedder=FakeEmbedder(), answerer=FakeAnswerer())
    hr_result = ask("salaries", groups=["hr"], collection=collection, embedder=FakeEmbedder(), answerer=FakeAnswerer())

    assert {h["source"] for h in public_result.sources} == {"handbook.pdf"}
    assert {h["source"] for h in hr_result.sources} == {"handbook.pdf", "hr/salaries.pdf"}


def test_answerer_returns_fixed_message_when_nothing_found():
    class EmptyAnswerer(FakeAnswerer):
        pass

    answerer = EmptyAnswerer()
    assert answerer.answer("anything", []) == NO_RESULTS_MESSAGE
