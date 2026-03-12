import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.services.agentic_retrieval import run_agentic_retrieval
import argparse

EVALUATION_FILE = "data/evaluation.json"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=str,
        required=True,
        help="Workspace to evaluate"
    )
    return parser.parse_args()


def load_evaluation_data():
    with open(EVALUATION_FILE, "r") as f:
        data = json.load(f)
    return data


def evaluate(workspace):  # workspace passed in as argument
    dataset = load_evaluation_data()

    total_queries = len(dataset)
    hits = 0

    for item in dataset:
        query = item["query"]
        expected_text = item["expected_text"]

        print(f"\nEVALUATING QUERY: {query}")

        results = run_agentic_retrieval(query=query, workspace=workspace)

        found = False

        for r in results:
            text = r.get("text", "").lower()
            if expected_text.lower() in text:
                found = True
                break

        # outside the for loop — check after all results are scanned
        if found:
            hits += 1
            print("HIT")
        else:
            print("MISS")

    recall = hits / total_queries

    print("\nEvaluation Results")
    print(f"Queries tested: {total_queries}")
    print(f"Hits: {hits}")
    print(f"Recall@k: {recall:.2f}")


if __name__ == "__main__":
    args = parse_args()      # parse args
    evaluate(args.workspace) # pass workspace into evaluate