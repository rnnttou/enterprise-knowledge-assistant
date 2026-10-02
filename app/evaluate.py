"""Step 10: evaluate retrieval quality and refusal behavior.

Ground truth: each answerable question in tests/eval_questions.json carries
an expected_source and expected_page, found by locating the exact passage
that answers it in the source PDF (see git history for the derivation). A
hit counts a chunk whose page is within PAGE_TOLERANCE of the expected page,
since chunking can place an answer's start just before or after a page
boundary. The question set also includes questions no document can answer
(marked "unanswerable": true), used to measure whether the pipeline refuses
instead of guessing.

By default this only exercises retrieval (search / bm25_search /
hybrid_search): cheap (only the query is embedded, no chat model call) and
safe to re-run after any retrieval change. It reports, per method:
    hit@1   - is the expected page the single top result?
    hit@5   - does the expected page appear anywhere in the top 5?
    MRR     - mean reciprocal rank of the expected page (0 if never found)
and separately, whether hybrid_search's distance-based refusal gate (the
same one app.qa.answer() uses) would correctly refuse each unanswerable
question, and would NOT wrongly refuse an answerable one.

Pass --full to run the complete answer() pipeline instead (uses your Gemini
quota and is much slower): reports hit rate and refusal from the real
answers, including actual chat-model refusals.

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

from app.config import MAX_DISTANCE, PROJECT_ROOT
from app.vector_store import bm25_search, get_collection, hybrid_search, search

EVAL_QUESTIONS_PATH = PROJECT_ROOT / "tests" / "eval_questions.json"

# A chunk's page counts as a hit if it's within this many pages of the
# expected one, absorbing chunking imprecision around page boundaries.
PAGE_TOLERANCE = 1


def load_questions(path: Path = EVAL_QUESTIONS_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def split_questions(questions: list[dict]) -> tuple[list[dict], list[dict]]:
    """(answerable, unanswerable), by the presence of "unanswerable": true."""
    answerable = [q for q in questions if not q.get("unanswerable")]
    unanswerable = [q for q in questions if q.get("unanswerable")]
    return answerable, unanswerable


def is_page_hit(hit: dict, item: dict, tolerance: int = PAGE_TOLERANCE) -> bool:
    return hit["source"] == item["expected_source"] and abs(hit["page"] - item["expected_page"]) <= tolerance


def rank_of_first_hit(hits: list[dict], item: dict) -> int | None:
    """1-based rank of the first hit matching item's expected page, or None."""
    for rank, hit in enumerate(hits, start=1):
        if is_page_hit(hit, item):
            return rank
    return None


def evaluate_retrieval(questions: list[dict], method_fn, k: int) -> list[dict]:
    """Run one retrieval method over each answerable question.

    method_fn(question, groups, k) -> list[hit], already bound to a
    collection (and embedder, for the vector-based methods).
    """
    results = []
    for item in questions:
        start = time.perf_counter()
        hits = method_fn(item["question"], item.get("groups", []), k)
        latency = time.perf_counter() - start
        rank = rank_of_first_hit(hits, item)
        results.append({
            **item,
            "hit": rank is not None,
            "hit_at_1": rank == 1,
            "reciprocal_rank": 1 / rank if rank else 0.0,
            "latency": latency,
        })
    return results


def evaluate_full(questions: list[dict], k: int) -> list[dict]:
    from app.qa import answer

    results = []
    for item in questions:
        start = time.perf_counter()
        result = answer(item["question"], item.get("groups", []), k)
        latency = time.perf_counter() - start
        rank = None
        for i, c in enumerate(result["citations"], start=1):
            if c["source"] == item.get("expected_source") and abs(c["page"] - item.get("expected_page", -999)) <= PAGE_TOLERANCE:
                rank = i
                break
        results.append({
            **item,
            "hit": rank is not None,
            "hit_at_1": rank == 1,
            "reciprocal_rank": 1 / rank if rank else 0.0,
            "refused": result["refused"],
            "latency": latency,
        })
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


def summarize_ranked(results: list[dict]) -> dict:
    n = len(results)
    return {
        "questions": n,
        "hit@1": sum(1 for r in results if r["hit_at_1"]) / n if n else 0.0,
        "hit@5": sum(1 for r in results if r["hit"]) / n if n else 0.0,
        "mrr": sum(r["reciprocal_rank"] for r in results) / n if n else 0.0,
        "avg_latency": statistics.mean(r["latency"] for r in results) if n else 0.0,
    }


def compare_methods(collection, embedder, questions: list[dict], k: int = 5) -> dict[str, dict]:
    """Retrieval quality of vector-only, BM25-only, and hybrid search, side by side.

    This is what justifies hybrid retrieval rather than just asserting it:
    if hybrid's hit@5/MRR isn't clearly at least as good as both individual
    methods, blending them wasn't worth the extra code.
    """
    methods = {
        "vector": lambda q, groups, k: search(collection, embedder, q, groups, k),
        "bm25": lambda q, groups, k: bm25_search(collection, q, groups, k),
        "hybrid": lambda q, groups, k: hybrid_search(collection, embedder, q, groups, k),
    }
    return {name: summarize_ranked(evaluate_retrieval(questions, fn, k)) for name, fn in methods.items()}


def evaluate_refusal(collection, embedder, questions: list[dict], k: int = 5) -> dict:
    """Would hybrid_search's distance gate (same one app.qa.answer() uses)
    refuse each question, without spending a chat-model call?

    Reports refusal accuracy separately for unanswerable questions (should
    refuse) and answerable ones (should NOT refuse) -- a method that refuses
    everything would ace the first number and fail the second.
    """
    answerable, unanswerable = split_questions(questions)

    def would_refuse(item: dict) -> bool:
        hits = hybrid_search(collection, embedder, item["question"], item.get("groups", []), k)
        return not hits or hits[0]["best_vector_distance"] > MAX_DISTANCE

    correctly_refused = [would_refuse(item) for item in unanswerable]
    wrongly_refused = [would_refuse(item) for item in answerable]
    return {
        "unanswerable_questions": len(unanswerable),
        "correct_refusal_rate": sum(correctly_refused) / len(unanswerable) if unanswerable else 0.0,
        "answerable_questions": len(answerable),
        "false_refusal_rate": sum(wrongly_refused) / len(answerable) if answerable else 0.0,
    }


def print_comparison_table(comparison: dict[str, dict]) -> None:
    print(f"{'method':<8} {'hit@1':>7} {'hit@5':>7} {'mrr':>7} {'avg latency':>13}")
    for name, s in comparison.items():
        print(f"{name:<8} {s['hit@1']:>6.1%} {s['hit@5']:>6.1%} {s['mrr']:>7.3f} {s['avg_latency'] * 1000:>10.0f}ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality and refusal behavior.")
    parser.add_argument("--full", action="store_true", help="also run the chat model (uses API quota) and report refusals")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--questions", type=Path, default=EVAL_QUESTIONS_PATH)
    args = parser.parse_args()

    from app.embeddings import GeminiEmbedder

    questions = load_questions(args.questions)
    answerable, unanswerable = split_questions(questions)
    collection = get_collection()
    embedder = GeminiEmbedder()

    if args.full:
        results = evaluate_full(answerable, args.k)
        summary = summarize_ranked(results)
        print(
            f"{summary['questions']} answerable questions, hit@1 {summary['hit@1']:.1%}, "
            f"hit@5 {summary['hit@5']:.1%}, MRR {summary['mrr']:.3f}, "
            f"avg latency {summary['avg_latency'] * 1000:.0f}ms"
        )
        misses = [r for r in results if not r["hit"]]
        if misses:
            print(f"\nMissed ({len(misses)}):")
            for r in misses:
                print(f"  - {r['question']!r} (expected {r['expected_source']} p.{r['expected_page']})")

        refused = sum(1 for r in results if r["refused"])
        print(f"\nWrongly refused answerable questions: {refused}/{len(results)}")

        unanswerable_results = evaluate_full(unanswerable, args.k)
        correctly_refused = sum(1 for r in unanswerable_results if r["refused"])
        print(f"Correctly refused unanswerable questions: {correctly_refused}/{len(unanswerable_results)}")
        return

    print(f"Retrieval comparison over {len(answerable)} answerable questions (k={args.k}):\n")
    comparison = compare_methods(collection, embedder, answerable, args.k)
    print_comparison_table(comparison)

    print(f"\nRefusal check over {len(unanswerable)} unanswerable questions (distance gate, no chat-model call):")
    refusal = evaluate_refusal(collection, embedder, questions, args.k)
    print(f"  Correctly refused: {refusal['correct_refusal_rate']:.1%} ({refusal['unanswerable_questions']} questions)")
    print(f"  Wrongly refused answerable questions: {refusal['false_refusal_rate']:.1%} ({refusal['answerable_questions']} questions)")


if __name__ == "__main__":
    main()
