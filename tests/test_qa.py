"""QA tests with fake embedder and chat model, so they run offline and cost nothing."""

import hashlib

import chromadb
import pytest

from app import qa
from app.qa import (
    NO_RESULTS_MESSAGE,
    REFUSAL_MESSAGE,
    Answer,
    GeminiAnswerer,
    answer,
    ask,
    build_context,
    build_history,
    clean_citations,
    contextualize_query,
)
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


def test_build_context_numbers_excerpts_with_file_and_page():
    hits = [
        {"source": "handbook.pdf", "page": 3, "text": "Employees get 25 days."},
        {"source": "handbook.pdf", "page": 4, "text": "Unused days expire in March."},
    ]
    context = build_context(hits)
    assert "[1] handbook.pdf, p. 3: Employees get 25 days." in context
    assert "[2] handbook.pdf, p. 4: Unused days expire in March." in context


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


# --- answer(): the Step 4 entry point -------------------------------------
# These tests mock the Gemini chat call (no network, no cost) and replace
# search() with fixed hits so we control how close the best match is.

HANDBOOK_HITS = [
    {"source": "employee-handbook.pdf", "page": 18, "distance": 0.30, "text": "Full-time staff accrue 25 vacation days per year."},
    {"source": "employee-handbook.pdf", "page": 21, "distance": 0.33, "text": "The College closes for winter break."},
]


class FakeGeminiClient:
    """Stands in for genai.Client: records the prompt and returns canned text."""

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts = []
        self.models = self   # so client.models.generate_content(...) works

    def generate_content(self, model, contents, config):
        self.prompts.append(contents)

        class Response:
            text = self.reply

        return Response()


def gemini_with_reply(reply: str) -> GeminiAnswerer:
    answerer = GeminiAnswerer(api_key="fake-key")
    answerer.client = FakeGeminiClient(reply)
    return answerer


@pytest.fixture
def fixed_hits(monkeypatch):
    """Make hybrid_search() return whatever hits the test puts in this list."""
    hits = []
    monkeypatch.setattr(qa, "hybrid_search", lambda collection, embedder, question, groups, k: hits)
    return hits


def run_answer(question, answerer, history=None):
    # collection/embedder are never used because hybrid_search() is patched.
    return answer(question, collection=object(), embedder=FakeEmbedder(), answerer=answerer, history=history)


def test_answer_with_good_match_returns_cited_answer(fixed_hits):
    fixed_hits.extend(HANDBOOK_HITS)
    gemini = gemini_with_reply("Full-time staff get 25 vacation days per year [1].")

    result = run_answer("How many vacation days?", gemini)

    assert result == {
        "answer": "Full-time staff get 25 vacation days per year [1].",
        "citations": [
            {
                "marker": 1,
                "source": "employee-handbook.pdf",
                "page": 18,
                "snippet": "Full-time staff accrue 25 vacation days per year.",
            }
        ],
        "refused": False,
    }
    # The model saw numbered excerpts with file and page.
    prompt = gemini.client.prompts[0]
    assert "[1] employee-handbook.pdf, p. 18: Full-time staff accrue" in prompt
    assert "Question: How many vacation days?" in prompt


def test_answer_refuses_when_best_match_is_weak(fixed_hits):
    fixed_hits.append({"source": "employee-handbook.pdf", "page": 44, "distance": 0.60, "text": "Unrelated page."})
    gemini = gemini_with_reply("should never be used")

    result = run_answer("What is the recipe for chocolate cake?", gemini)

    assert result == {"answer": REFUSAL_MESSAGE, "citations": [], "refused": True}
    assert gemini.client.prompts == []   # no chat-model call was spent


def test_answer_refuses_when_search_finds_nothing(fixed_hits):
    result = run_answer("anything", gemini_with_reply("unused"))
    assert result["refused"] is True


def test_answer_drops_citation_markers_that_dont_match_a_chunk(fixed_hits):
    fixed_hits.extend(HANDBOOK_HITS)   # only [1] and [2] exist
    gemini = gemini_with_reply("Staff get 25 days [1][7]. The College closes in winter [2, 9]. Pay is monthly [5].")

    result = run_answer("Tell me about time off", gemini)

    assert result["answer"] == "Staff get 25 days [1]. The College closes in winter [2]. Pay is monthly."
    assert [c["marker"] for c in result["citations"]] == [1, 2]
    assert result["refused"] is False


def test_answer_without_valid_citations_counts_as_refused(fixed_hits):
    fixed_hits.extend(HANDBOOK_HITS)
    gemini = gemini_with_reply("The excerpts don't say anything about parking.")

    result = run_answer("Where do I park?", gemini)

    assert result["citations"] == []
    assert result["refused"] is True


def test_clean_citations_keeps_order_and_removes_duplicates():
    text, markers = clean_citations("A [2]. B [1][2]. C [3].", num_sources=2)
    assert text == "A [2]. B [1][2]. C."
    assert markers == [2, 1]


# --- Step 8: conversation memory -------------------------------------------

HISTORY = [
    {"question": "How many vacation days do employees get?", "answer": "25 days [1]."},
    {"question": "Do unused days roll over?", "answer": "No, they expire in March [2]."},
]


def test_contextualize_query_without_history_returns_question_unchanged():
    assert contextualize_query("What about managers?", None) == "What about managers?"
    assert contextualize_query("What about managers?", []) == "What about managers?"


def test_contextualize_query_prepends_recent_questions():
    query = contextualize_query("What about managers?", HISTORY)
    assert query == (
        "How many vacation days do employees get?\n"
        "Do unused days roll over?\n"
        "What about managers?"
    )


def test_contextualize_query_keeps_only_the_most_recent_turns():
    long_history = [{"question": f"question {i}", "answer": "a"} for i in range(5)]
    query = contextualize_query("latest", long_history)
    assert query == "question 3\nquestion 4\nlatest"


def test_build_history_of_no_history_is_empty():
    assert build_history(None) == ""
    assert build_history([]) == ""


def test_build_history_formats_turns_as_qa_pairs():
    formatted = build_history(HISTORY)
    assert "Q: How many vacation days do employees get?" in formatted
    assert "A: 25 days [1]." in formatted
    assert "Q: Do unused days roll over?" in formatted


def test_answer_sends_history_to_the_model(fixed_hits):
    fixed_hits.extend(HANDBOOK_HITS)
    gemini = gemini_with_reply("Managers also get 25 days [1].")

    result = run_answer("What about managers?", gemini, history=HISTORY)

    assert result["refused"] is False
    prompt = gemini.client.prompts[0]
    assert "Previous conversation" in prompt
    assert "Do unused days roll over?" in prompt
    assert "Question: What about managers?" in prompt


def test_answer_uses_history_to_retrieve_for_a_followup(monkeypatch):
    captured = {}

    def fake_hybrid_search(collection, embedder, question, groups, k):
        captured["question"] = question
        return list(HANDBOOK_HITS)

    monkeypatch.setattr(qa, "hybrid_search", fake_hybrid_search)
    gemini = gemini_with_reply("Managers also get 25 days [1].")

    run_answer("What about managers?", gemini, history=HISTORY)

    assert captured["question"] == contextualize_query("What about managers?", HISTORY)


def test_answer_without_history_behaves_as_before(fixed_hits):
    fixed_hits.extend(HANDBOOK_HITS)
    gemini = gemini_with_reply("Full-time staff get 25 vacation days per year [1].")

    result = run_answer("How many vacation days?", gemini)

    assert result["refused"] is False
    assert "Previous conversation" not in gemini.client.prompts[0]
