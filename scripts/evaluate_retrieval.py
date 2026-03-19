import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import json
import time
from collections import defaultdict
from backend.services.agentic_retrieval import run_agentic_retrieval


# CONSTANTS

EVALUATION_FILE: str = "data/evaluation.json"
SEPARATOR: str = "─" * 60


# DATA LOADING
def load_evaluation_data() -> list[dict]:
    """
    Loads and returns the evaluation dataset from JSON.
    """
    try:
        with open(EVALUATION_FILE, "r", encoding="utf-8") as f:
            data: list[dict] = json.load(f)
        print(f"Loaded {len(data)} test cases from {EVALUATION_FILE}")
        return data
    except FileNotFoundError:
        print(f"ERROR: Evaluation file not found: {EVALUATION_FILE}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse {EVALUATION_FILE}: {e}")
        sys.exit(1)


# METRIC HELPERS
def find_hit_rank(results: list[dict], expected_text: str) -> int | None:
    """
    Finds the rank (1-indexed) of the first result containing expected_text.
    """
    for rank, result in enumerate(results, start=1):
 
        chunk_text: str = result.get("text", "").lower()
 
        if expected_text.lower() in chunk_text:
            # Check if the expected text appears ANYWHERE in the chunk.
            return rank

    return None

def reciprocal_rank(rank: int | None) -> float:
    """
    Computes the reciprocal rank for one query.
 
    MRR formula component: 1 / rank if hit, 0 if miss.
    """
    if rank is None:
        return 0.0
    return 1.0 / rank


def evaluate(workspace: str, top_k: int = 3) -> dict:
    """
    Runs the full evaluation pipeline for a given workspace.
    """
    dataset: list[dict] = load_evaluation_data()

    workspace_cases: list[dict] = [
        case for case in dataset
        if case["workspace"] == workspace
    ]
    if not workspace_cases:
        print(f"No test cases found for workspace '{workspace}'.")
        print(f"Available workspaces in dataset: "
              f"{list(set(c['workspace'] for c in dataset))}")
        sys.exit(1)
 
    total: int = len(workspace_cases)
    print(f"\nEvaluating {total} test cases for workspace='{workspace}' | top_k={top_k}")
    print(SEPARATOR)
    # Per-query tracking
    hits: int = 0
    # Count of queries where the expected chunk was found (Recall numerator).
 
    reciprocal_ranks: list[float] = []
    # One reciprocal rank value per query (for MRR calculation).
 
    latencies: list[float] = []
    # Per-query latency in milliseconds.
 
    per_type_results: dict = defaultdict(lambda: {"hits": 0, "total": 0, "rr_sum": 0.0})
    # Groups results by query_type.
    
    # Evaluate each test case
    for case in workspace_cases:
        query_id: int = case["id"]
        query: str = case["query"]
        expected_text: str = case["expected_text"]
        query_type: str = case.get("query_type", "unknown")
 
        print(f"\n[{query_id:02d}] [{query_type.upper():9s}] {query}")
        
        # Run retrieval and measure latency
        query_start: float = time.time()
 
        results: list[dict] = run_agentic_retrieval(
            query=query,
            workspace=workspace,
        )
        # run_agentic_retrieval() runs the full agentic pipeline:
        # plan → sub-queries → search() per sub-query → deduplicate.
 
        query_latency_ms: float = (time.time() - query_start) * 1000
        latencies.append(query_latency_ms)
 
        # Check for hit and compute rank
        hit_rank: int | None = find_hit_rank(results, expected_text)
        rr: float = reciprocal_rank(hit_rank)
 
        reciprocal_ranks.append(rr)
 
        # Update per-type counters
        per_type_results[query_type]["total"] += 1
        per_type_results[query_type]["rr_sum"] += rr
 
        if hit_rank is not None:
            hits += 1
            per_type_results[query_type]["hits"] += 1
            print(f"  ✅ HIT at rank {hit_rank} | latency={query_latency_ms:.0f}ms")
        else:
            print(f"  ❌ MISS | latency={query_latency_ms:.0f}ms")
            print(f"     Expected: '{expected_text}'")
            # Print what we were looking for on a miss — useful for debugging.
            if results:
                first_source = results[0].get("source", "unknown")
                print(f"     Got:      {len(results)} results, first from '{first_source}'")
 
    # Compute aggregate metrics
    recall_at_k: float = hits / total if total > 0 else 0.0
    # Recall@k = hits / total queries
    # How many queries had their expected chunk in the top-k results.
 
    mrr: float = sum(reciprocal_ranks) / total if total > 0 else 0.0
    # MRR = mean of all reciprocal ranks
    # sum(reciprocal_ranks) / total
    # A miss contributes 0.0, a rank-1 hit contributes 1.0.
 
    avg_latency_ms: float = sum(latencies) / len(latencies) if latencies else 0.0
    min_latency_ms: float = min(latencies) if latencies else 0.0
    max_latency_ms: float = max(latencies) if latencies else 0.0
 
   
    print(f"\n{SEPARATOR}")
    print(f"EVALUATION RESULTS — workspace='{workspace}' | top_k={top_k}")
    print(SEPARATOR)
    print(f"Total queries:   {total}")
    print(f"Hits:            {hits}")
    print(f"Misses:          {total - hits}")
    print(SEPARATOR)
    print(f"Recall@{top_k}:        {recall_at_k:.3f}  ({hits}/{total})")
    print(f"MRR:             {mrr:.3f}")
    # MRR interpretation:
    #   0.9+ excellent — correct chunk almost always at rank 1
    #   0.7-0.9 good  — correct chunk usually in top 2
    #   0.5-0.7 fair  — correct chunk found but sometimes buried
    #   < 0.5   poor  — retrieval needs improvement
    print(SEPARATOR)
    print(f"Avg latency:     {avg_latency_ms:.0f}ms")
    print(f"Min latency:     {min_latency_ms:.0f}ms")
    print(f"Max latency:     {max_latency_ms:.0f}ms")
    print(SEPARATOR)
 
    # Per-Type Breakdown
    print("PER QUERY TYPE BREAKDOWN:")
    print()
 
    for query_type, type_data in sorted(per_type_results.items()):
        # sorted() iterates types in alphabetical order for consistent output.
        type_total: int = type_data["total"]
        type_hits: int = type_data["hits"]
        type_recall: float = type_hits / type_total if type_total > 0 else 0.0
        type_mrr: float = type_data["rr_sum"] / type_total if type_total > 0 else 0.0
 
        print(f"  {query_type.upper():10s} | "
              f"Recall@{top_k}: {type_recall:.2f} ({type_hits}/{type_total}) | "
              f"MRR: {type_mrr:.3f}")
 
    print(SEPARATOR)
 
    # Return metrics dict 
    metrics: dict = {
        "workspace":       workspace,
        "top_k":           top_k,
        "total_queries":   total,
        "hits":            hits,
        "misses":          total - hits,
        f"recall_at_{top_k}": round(recall_at_k, 3),
        "mrr":             round(mrr, 3),
        "avg_latency_ms":  round(avg_latency_ms, 1),
        "min_latency_ms":  round(min_latency_ms, 1),
        "max_latency_ms":  round(max_latency_ms, 1),
        "per_type": {
            qt: {
                "hits":    d["hits"],
                "total":   d["total"],
                f"recall_at_{top_k}": round(d["hits"] / d["total"], 3) if d["total"] > 0 else 0.0,
                "mrr":     round(d["rr_sum"] / d["total"], 3) if d["total"] > 0 else 0.0,
            }
            for qt, d in per_type_results.items()
        }
    }
 
    return metrics

# ARGUMENT PARSING + ENTRY POINT

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval pipeline against ground-truth test cases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # RawDescriptionHelpFormatter preserves newlines in the description.
        epilog=(
            "Examples:\n"
            "  python scripts/evaluate_retrieval.py --workspace got\n"
            "  python scripts/evaluate_retrieval.py --workspace dexter --top_k 5\n"
        ),
    )
 
    parser.add_argument(
        "--workspace",
        type=str,
        required=True,
        # required=True — this argument MUST be provided.
        # Without it, argparse prints a usage error and exits.
        help="Workspace to evaluate (e.g. got, dexter)",
    )
 
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        # default=3 — if --top_k is not provided, args.top_k = 3.
        help="Number of results to retrieve per query (default: 3)",
    )
 
    return parser.parse_args()
 
 
if __name__ == "__main__":
 
    args = parse_args()
    results = evaluate(workspace=args.workspace, top_k=args.top_k)
 
    # Optionally save results to JSON for CI/CD pipelines
    output_path = f"data/eval_results_{args.workspace}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")