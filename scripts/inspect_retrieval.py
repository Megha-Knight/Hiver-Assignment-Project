"""CLI utility to inspect historical conversation retrieval matches and evidence.

Usage:
    python scripts/inspect_retrieval.py --query "Where is my package tracking says delivered"
    python scripts/inspect_retrieval.py --query "My Kindle screen is frozen" --intent TECHNICAL_AND_DIGITAL_SUPPORT
"""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import PATHS
from src.retrieval.retriever import HistoricalRetriever
from src.utils.logger import get_logger

logger = get_logger("inspect_retrieval")


def main():
    parser = argparse.ArgumentParser(description="Query historical retrieval index.")
    parser.add_argument("--query", type=str, required=True, help="Customer query text")
    parser.add_argument("--intent", type=str, default=None, help="Optional intent filter")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top matches")
    args = parser.parse_args()

    retriever = HistoricalRetriever(
        index_npz_path=PATHS.RETRIEVAL_INDEX_NPZ,
        meta_json_path=PATHS.RETRIEVAL_INDEX_META,
    )

    print("\n" + "=" * 75)
    print(f"QUERY: \"{args.query}\"")
    if args.intent:
        print(f"INTENT FILTER: {args.intent}")
    print("=" * 75)

    results = retriever.query(
        customer_message=args.query,
        intent_filter=args.intent,
        top_k=args.top_k,
    )

    if not results:
        print("No matches found.")
        return

    for rank, res in enumerate(results, 1):
        print(f"\n[RANK {rank}] Score: {res.similarity_score:.4f} | ID: {res.retrieval_id}")
        print(f"  Conversation ID:  {res.conversation_id}")
        print(f"  Outcome Evidence: {res.outcome_evidence_type} (Status: {res.resolution_status})")
        print(f"  Customer Problem: \"{res.customer_problem_summary}\"")
        print(f"  Support Evidence: \"{res.support_response_evidence.replace(chr(10), ' ')}\"")
        print(f"  Source Tweets:    {res.source_tweet_ids}")
        print("-" * 75)


if __name__ == "__main__":
    main()
