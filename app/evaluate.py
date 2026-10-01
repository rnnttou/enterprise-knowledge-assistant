"""Step 10: evaluate retrieval hit rate and latency on a fixed set of questions.

By default this only exercises hybrid_search(): for each question, does the
expected source document appear in the top-k results? That's cheap (no chat
model call) and safe to re-run after any retrieval change.

Pass --full to run the complete answer() pipeline instead (uses your Gemini
quota and is much slower), which also reports the refusal rate.

Usage:
    python -m app.evaluate
    python -m app.evaluate -k 10
    python -m app.evaluate --full
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from app.config import PROJECT_ROOT
from app.vector_store import get_collection, hybrid_search

EVAL_QUESTIONS_PATH = PROJECT_ROOT / "tests" / "eval_questions.json"


def load_questions(path: Path = EVAL_QUESTIONS_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_retrieval(questions: list[dict], collection, embedder, k: int) -> list[dict]:
    results = []
    for item in questions:
        start = time.perf_counter()
        hits = hybrid_search(collection, embedder, item["question"], item.get("groups", []), k)
        latency = time.perf_counter() - start
        hit = any(h["source"] == item["expected_source"] for h in hits)
        results.append({**item, "hit": hit, "latency": latency})
    return results


def evaluate_full(questions: list[dict], k: int) -> list[dict]:
    from app.qa import answer

    results = []
    for item in questions:
        start = time.perf_counter()
        result = answer(item["question"], item.get("groups", []), k)
        latency = time.perf_counter() - start
        hit = any(c["source"] == item["expected_source"] for c in result["citations"])
        results.append({**item, "hit": hit, "refused": result["refused"], "latency": latency})
    return results


def summarize(results: list[dict]) -> dict:
    latencies = sorted(r["latency"] for r in results)
    hits = sum(1 for r in results if r["hit"])
    # A simple nearest-rank percentile: good enough for a few dozen samples.
    p95_index = min(int(len(latencies) * 0.95), len(latencies) - 1)
    return {
        "questions": len(results),
        "hit_rate": hits / len(results) if results else 0.0,
        "avg_latency": statistics.mean(latencies) if latencies else 0.0,
        "p95_latency": latencies[p95_index] if latencies else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval hit rate and latency.")
    parser.add_argument("--full", action="store_true", help="also run the chat model (uses API quota) and report refusals")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--questions", type=Path, default=EVAL_QUESTIONS_PATH)
    args = parser.parse_args()

    questions = load_questions(args.questions)

    if args.full:
        results = evaluate_full(questions, args.k)
    else:
        from app.embeddings import GeminiEmbedder

        results = evaluate_retrieval(questions, get_collection(), GeminiEmbedder(), args.k)

    summary = summarize(results)
    print(
        f"{summary['questions']} questions, hit rate {summary['hit_rate']:.1%}, "
        f"avg latency {summary['avg_latency'] * 1000:.0f}ms, p95 {summary['p95_latency'] * 1000:.0f}ms"
    )

    misses = [r for r in results if not r["hit"]]
    if misses:
        print(f"\nMissed ({len(misses)}):")
        for r in misses:
            print(f"  - {r['question']!r} (expected {r['expected_source']})")

    if args.full:
        refused = sum(1 for r in results if r["refused"])
        print(f"Refused: {refused}/{len(results)}")


if __name__ == "__main__":
    main()
