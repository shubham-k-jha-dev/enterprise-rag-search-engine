import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import time
import urllib.request
import urllib.error
import urllib.parse


# CONSTANTS

EVALUATION_FILE: str = "data/evaluation.json"
SEPARATOR: str = "─" * 60

UNFAITHFUL_PHRASES: tuple[str, ...] = (
    "i don't have",
    "i don't know",
    "i'm not sure",
    "my knowledge",
    "as of my",
    "i cannot find",
    "not mentioned in",
    "not provided in",
    "outside my knowledge",
)


# HTTP HELPER

def call_ask_endpoint(
    query: str,
    workspace: str,
    base_url: str,
    api_key: str,
    conversation_id: str = "eval-session",
) -> tuple[str, float]:
    """
    Calls the /ask endpoint and collects the full streamed response.
    """
    encoded_query = urllib.parse.quote(query)
    encoded_workspace = urllib.parse.quote(workspace)
    encoded_conv_id = urllib.parse.quote(conversation_id)

    url = (
        f"{base_url}/ask"
        f"?query={encoded_query}"
        f"&workspace={encoded_workspace}"
        f"&conversation_id={encoded_conv_id}"
    )

    req = urllib.request.Request(
        url,
        headers={"X-API-Key": api_key},
        method="GET",
    )

    start: float = time.time()

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            raw_bytes: bytes = response.read()
            answer: str = raw_bytes.decode("utf-8")

    except urllib.error.HTTPError as e:
        answer = f"ERROR: HTTP {e.code} — {e.reason}"
    except urllib.error.URLError as e:
        answer = f"ERROR: Cannot connect to {base_url} — {e.reason}"
    except Exception as e:
        answer = f"ERROR: {str(e)}"

    latency_ms: float = (time.time() - start) * 1000
    return answer, latency_ms


# ANSWER ANALYSIS HELPERS

def check_answer_hit(answer: str, expected_text: str) -> bool:
    """
    Checks if the answer contains the expected text (case-insensitive).
    """
    return expected_text.lower() in answer.lower()


def check_answer_faithfulness(answer: str) -> bool:
    """
    Returns True if the answer appears faithful to the retrieved context.
    Returns False if it contains phrases indicating refusal or fallback.
    """
    answer_lower: str = answer.lower()
    for phrase in UNFAITHFUL_PHRASES:
        if phrase in answer_lower:
            return False
    return True


# MAIN EVALUATION FUNCTION

def evaluate_answers(
    workspace: str,
    base_url: str,
    api_key: str,
) -> dict:
    """
    Evaluates answer quality for all test cases in the given workspace.
    """
    try:
        with open(EVALUATION_FILE, "r", encoding="utf-8") as f:
            dataset: list[dict] = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {EVALUATION_FILE} not found.")
        sys.exit(1)

    workspace_cases: list[dict] = [
        c for c in dataset if c["workspace"] == workspace
    ]

    if not workspace_cases:
        print(f"No test cases for workspace '{workspace}'.")
        sys.exit(1)

    total: int = len(workspace_cases)
    print(f"\nEvaluating ANSWER QUALITY for {total} cases | workspace='{workspace}'")
    print(f"Server: {base_url}")
    print(SEPARATOR)

    # Per-query tracking
    answer_hits: int = 0
    faithful_count: int = 0
    latencies: list[float] = []
    answer_lengths: list[int] = []

    for case in workspace_cases:
        query_id: int = case["id"]
        query: str = case["query"]
        expected_text: str = case["expected_text"]
        query_type: str = case.get("query_type", "unknown")

        print(f"\n[{query_id:02d}] [{query_type.upper():9s}] {query}")

        answer, latency_ms = call_ask_endpoint(
            query=query,
            workspace=workspace,
            base_url=base_url,
            api_key=api_key,
        )
        latencies.append(latency_ms)

        if answer.startswith("ERROR:"):
            print(f"  ⚠️  {answer}")
            print(f"     Latency: {latency_ms:.0f}ms")
            continue

        answer_length: int = len(answer)
        answer_lengths.append(answer_length)

        hit: bool = check_answer_hit(answer, expected_text)
        faithful: bool = check_answer_faithfulness(answer)

        if hit:
            answer_hits += 1
        if faithful:
            faithful_count += 1

        hit_symbol: str = "✅" if hit else "❌"
        faith_symbol: str = "✅" if faithful else "⚠️ "

        print(f"  {hit_symbol} Answer hit: {hit} | "
              f"{faith_symbol} Faithful: {faithful} | "
              f"Length: {answer_length} chars | "
              f"Latency: {latency_ms:.0f}ms")

        if not hit:
            print(f"     Expected: '{expected_text}'")
            preview: str = answer[:200].replace("\n", " ")
            print(f"     Answer preview: '{preview}...'")

    # Compute aggregate metrics
    evaluated: int = len(answer_lengths)

    answer_hit_rate: float = answer_hits / total if total > 0 else 0.0
    faithfulness_rate: float = faithful_count / evaluated if evaluated > 0 else 0.0
    avg_length: float = sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0.0
    avg_latency: float = sum(latencies) / len(latencies) if latencies else 0.0

    # Print report
    print(f"\n{SEPARATOR}")
    print(f"ANSWER EVALUATION RESULTS — workspace='{workspace}'")
    print(SEPARATOR)
    print(f"Total queries:       {total}")
    print(f"Successfully evaluated: {evaluated}")
    print(SEPARATOR)
    print(f"Answer Hit Rate:     {answer_hit_rate:.3f}  ({answer_hits}/{total})")
    print(f"Faithfulness Rate:   {faithfulness_rate:.3f}  ({faithful_count}/{evaluated})")
    print(f"Avg Answer Length:   {avg_length:.0f} chars")
    print(f"Avg Latency:         {avg_latency:.0f}ms")
    print(SEPARATOR)

    metrics: dict = {
        "workspace":          workspace,
        "total_queries":      total,
        "evaluated":          evaluated,
        "answer_hit_rate":    round(answer_hit_rate, 3),
        "faithfulness_rate":  round(faithfulness_rate, 3),
        "avg_answer_length":  round(avg_length, 1),
        "avg_latency_ms":     round(avg_latency, 1),
    }

    return metrics


# ARGUMENT PARSING + ENTRY POINT

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate LLM answer quality against ground-truth test cases.",
        epilog=(
            "Examples:\n"
            "  python scripts/evaluate_answers.py --workspace got\n"
            "  python scripts/evaluate_answers.py --workspace dexter "
            "--base_url http://localhost:8000\n"
        ),
    )
    parser.add_argument(
        "--workspace",
        type=str,
        required=True,
        help="Workspace to evaluate (e.g. got, dexter)",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8000",
        help="FastAPI server base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.getenv("API_KEY", "Shubh@2005"),
        help="API key for authentication (default: reads API_KEY env var)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("⚠️  NOTE: The FastAPI server must be running before this script.")
    print(f"    Connecting to: {args.base_url}")

    metrics = evaluate_answers(
        workspace=args.workspace,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    output_path = f"data/answer_eval_{args.workspace}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {output_path}")
