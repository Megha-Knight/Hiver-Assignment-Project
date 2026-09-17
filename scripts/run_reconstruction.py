"""CLI Runner for Conversation Reconstruction Prototype.

Reconstructs multi-turn conversational threads for candidate brands,
validates graph consistency, verifies cycle prevention and deduplication,
and outputs formatted sample threads for manual inspection.
"""

import argparse
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AUDIT_CONFIG, PATHS
from src.data.reconstruct import ConversationReconstructor
from src.utils.logger import get_logger
import pandas as pd

logger = get_logger("run_reconstruction")


def reconstruct_brand_threads(csv_path: Path, brand: str = "AmazonHelp", max_rows: int = None):
    """Reconstructs threads for a specific brand from twcs.csv."""
    logger.info(f"Reconstructing conversation threads for brand: {brand}")
    reconstructor = ConversationReconstructor(brand=brand)

    # Pass 1: find brand tweets and references
    brand_tweets = set()
    referenced_ids = set()

    pass1_cols = ["tweet_id", "author_id", "response_tweet_id", "in_response_to_tweet_id"]
    for chunk in pd.read_csv(csv_path, usecols=pass1_cols, chunksize=250_000, nrows=max_rows):
        sub = chunk[chunk["author_id"] == brand]
        if len(sub) == 0:
            continue
        brand_tweets.update(sub["tweet_id"].astype("int64"))
        parents = sub["in_response_to_tweet_id"].dropna().astype("int64")
        referenced_ids.update(parents)
        for resps in sub["response_tweet_id"].dropna():
            for r in str(resps).split(","):
                r = r.strip()
                if r.isdigit():
                    referenced_ids.add(int(r))

    target_ids = brand_tweets | referenced_ids
    logger.info(f"Identified {len(brand_tweets):,} brand tweets and {len(referenced_ids):,} referenced customer tweets.")

    # Pass 2: load target rows into reconstructor
    for chunk in pd.read_csv(csv_path, chunksize=250_000, nrows=max_rows):
        filtered = chunk[chunk["tweet_id"].isin(target_ids)]
        if len(filtered) == 0:
            continue
        for row in filtered.itertuples(index=False):
            tid = int(row.tweet_id)
            parent_val = row.in_response_to_tweet_id
            parent_id = int(parent_val) if pd.notna(parent_val) and str(parent_val).strip() != "" else None
            resps = []
            resp_val = row.response_tweet_id
            if pd.notna(resp_val) and str(resp_val).strip() != "":
                for r in str(resp_val).split(","):
                    r = r.strip()
                    if r.isdigit():
                        resps.append(int(r))

            reconstructor.add_tweet(
                tweet_id=tid,
                author_id=str(row.author_id),
                inbound=bool(row.inbound),
                created_at=str(row.created_at),
                text=str(row.text) if pd.notna(row.text) else "",
                parent_id=parent_id,
                response_ids=resps,
            )

    conversations = reconstructor.reconstruct_all()
    return conversations


def main():
    parser = argparse.ArgumentParser(description="Run conversation reconstruction prototype.")
    parser.add_argument("--brand", type=str, default="AmazonHelp", help="Target brand name")
    parser.add_argument("--dataset", type=Path, default=PATHS.get_raw_dataset_path(), help="Dataset path")
    parser.add_argument("--sample-output", type=Path, default=PATHS.RESULTS_DIR / "prototype_reconstruction.json")
    args = parser.parse_args()

    t0 = time.time()
    conversations = reconstruct_brand_threads(csv_path=args.dataset, brand=args.brand)

    # Verification checks
    total_convs = len(conversations)
    broken_convs = sum(1 for c in conversations if c.is_broken)
    cycle_convs = sum(1 for c in conversations if c.has_cycle)
    deep_convs = sum(1 for c in conversations if c.turn_count >= 3)

    logger.info(f"Reconstruction summary for {args.brand}:")
    logger.info(f"  Total conversations: {total_convs:,}")
    logger.info(f"  Conversations >= 3 turns: {deep_convs:,} ({deep_convs/total_convs*100:.1f}%)")
    logger.info(f"  Broken links detected: {broken_convs:,} ({broken_convs/total_convs*100:.1f}%)")
    logger.info(f"  Cycles detected defensively: {cycle_convs}")

    # Export top 10 sample conversations
    sample_convs = [c.to_dict() for c in sorted(conversations, key=lambda x: x.turn_count, reverse=True)[:10]]
    args.sample_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.sample_output, "w", encoding="utf-8") as f:
        json.dump(sample_convs, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved sample reconstructed conversations to: {args.sample_output}")
    logger.info(f"Reconstruction finished in {time.time() - t0:.2f} seconds.")


if __name__ == "__main__":
    main()
