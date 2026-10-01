"""API tests with fake dependencies, so they run offline and cost nothing."""

import pytest
from fastapi.testclient import TestClient

from app.api import app, get_ask, get_indexer
from app.qa import Answer


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_answer_and_sources(client):
    def fake_ask(question, groups, k):
        assert question == "How many vacation days?"
        assert groups == ["hr"]
        assert k == 3
        return Answer(
            text="Employees get 25 days [1].",
            sources=[{"title": "Handbook", "source": "handbook.pdf", "page": 18, "distance": 0.12}],
        )

    app.dependency_overrides[get_ask] = lambda: fake_ask

    response = client.post("/ask", json={"question": "How many vacation days?", "groups": ["hr"], "k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Employees get 25 days [1]."
    assert body["sources"] == [{"title": "Handbook", "source": "handbook.pdf", "page": 18, "distance": 0.12}]


def test_ask_defaults_groups_and_k(client):
    captured = {}

    def fake_ask(question, groups, k):
        captured["groups"] = groups
        captured["k"] = k
        return Answer(text="ok", sources=[])

    app.dependency_overrides[get_ask] = lambda: fake_ask

    response = client.post("/ask", json={"question": "anything"})

    assert response.status_code == 200
    assert captured == {"groups": [], "k": 5}


def test_ask_rejects_empty_question(client):
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_surfaces_missing_api_key_as_500(client):
    def fake_ask(question, groups, k):
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")

    app.dependency_overrides[get_ask] = lambda: fake_ask

    response = client.post("/ask", json={"question": "anything"})

    assert response.status_code == 500
    assert "GEMINI_API_KEY" in response.json()["detail"]


def test_reindex_returns_stats(client):
    app.dependency_overrides[get_indexer] = lambda: lambda: {
        "indexed": ["handbook.pdf"],
        "skipped": [],
        "removed": [],
        "empty": [],
        "chunks_added": 4,
    }

    response = client.post("/index")

    assert response.status_code == 200
    assert response.json() == {
        "indexed": ["handbook.pdf"],
        "skipped": [],
        "removed": [],
        "empty": [],
        "chunks_added": 4,
    }
