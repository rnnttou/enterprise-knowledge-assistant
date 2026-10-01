"""API tests with fake dependencies, so they run offline and cost nothing."""

import pytest
from fastapi.testclient import TestClient

from app.api import app, conversations, get_ask, get_indexer


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()
    conversations.clear()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_answer_and_sources(client):
    def fake_ask(question, groups, k, *, history=None):
        assert question == "How many vacation days?"
        assert groups == ["hr"]
        assert k == 3
        return {
            "answer": "Employees get 25 days [1].",
            "citations": [{"marker": 1, "source": "handbook.pdf", "page": 18, "snippet": "Employees get 25 days."}],
            "refused": False,
        }

    app.dependency_overrides[get_ask] = lambda: fake_ask

    response = client.post("/ask", json={"question": "How many vacation days?", "groups": ["hr"], "k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Employees get 25 days [1]."
    assert body["citations"] == [{"marker": 1, "source": "handbook.pdf", "page": 18, "snippet": "Employees get 25 days."}]
    assert body["refused"] is False
    assert body["conversation_id"]


def test_ask_passes_refusal_through(client):
    app.dependency_overrides[get_ask] = lambda: lambda question, groups, k, *, history=None: {
        "answer": "I couldn't find this in the documents.",
        "citations": [],
        "refused": True,
    }

    response = client.post("/ask", json={"question": "chocolate cake recipe?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "I couldn't find this in the documents."
    assert body["citations"] == []
    assert body["refused"] is True


def test_ask_defaults_groups_and_k(client):
    captured = {}

    def fake_ask(question, groups, k, *, history=None):
        captured["groups"] = groups
        captured["k"] = k
        return {"answer": "ok", "citations": [], "refused": False}

    app.dependency_overrides[get_ask] = lambda: fake_ask

    response = client.post("/ask", json={"question": "anything"})

    assert response.status_code == 200
    assert captured == {"groups": [], "k": 5}


def test_ask_rejects_empty_question(client):
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_surfaces_missing_api_key_as_500(client):
    def fake_ask(question, groups, k, *, history=None):
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")

    app.dependency_overrides[get_ask] = lambda: fake_ask

    response = client.post("/ask", json={"question": "anything"})

    assert response.status_code == 500
    assert "GEMINI_API_KEY" in response.json()["detail"]


def test_ask_without_conversation_id_starts_a_new_conversation(client):
    app.dependency_overrides[get_ask] = lambda: lambda question, groups, k, *, history=None: {
        "answer": "ok",
        "citations": [],
        "refused": False,
    }

    response = client.post("/ask", json={"question": "anything"})

    assert response.status_code == 200
    assert len(response.json()["conversation_id"]) > 0


def test_ask_reuses_conversation_id_and_passes_history(client):
    captured_history = []

    def fake_ask(question, groups, k, *, history=None):
        captured_history.append(history)
        return {"answer": f"answer to {question}", "citations": [], "refused": False}

    app.dependency_overrides[get_ask] = lambda: fake_ask

    first = client.post("/ask", json={"question": "first question"})
    conversation_id = first.json()["conversation_id"]

    second = client.post("/ask", json={"question": "second question", "conversation_id": conversation_id})

    assert second.json()["conversation_id"] == conversation_id
    assert captured_history[0] == []
    assert captured_history[1] == [{"question": "first question", "answer": "answer to first question"}]


def test_ask_caps_stored_history_length(client):
    app.dependency_overrides[get_ask] = lambda: lambda question, groups, k, *, history=None: {
        "answer": "ok",
        "citations": [],
        "refused": False,
    }

    from app.api import MAX_HISTORY_TURNS, conversations

    conversation_id = None
    for i in range(MAX_HISTORY_TURNS + 3):
        response = client.post("/ask", json={"question": f"question {i}", "conversation_id": conversation_id})
        conversation_id = response.json()["conversation_id"]

    assert len(conversations[conversation_id]) == MAX_HISTORY_TURNS


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
