"""Tests for the evaluation summary math (Step 10). No network, no ChromaDB."""

from app.evaluate import summarize


def test_summarize_computes_hit_rate():
    results = [
        {"hit": True, "latency": 0.1},
        {"hit": True, "latency": 0.2},
        {"hit": False, "latency": 0.3},
        {"hit": False, "latency": 0.4},
    ]
    summary = summarize(results)
    assert summary["questions"] == 4
    assert summary["hit_rate"] == 0.5


def test_summarize_computes_average_latency():
    results = [{"hit": True, "latency": 0.1}, {"hit": True, "latency": 0.3}]
    summary = summarize(results)
    assert summary["avg_latency"] == 0.2


def test_summarize_p95_is_near_the_slowest_result():
    results = [{"hit": True, "latency": i / 100} for i in range(1, 21)]  # 0.01 .. 0.20
    summary = summarize(results)
    assert summary["p95_latency"] == 0.20


def test_summarize_of_no_results_does_not_crash():
    summary = summarize([])
    assert summary == {"questions": 0, "hit_rate": 0.0, "avg_latency": 0.0, "p95_latency": 0.0}
