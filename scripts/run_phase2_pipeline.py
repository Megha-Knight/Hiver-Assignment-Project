"""Phase 2 Orchestration Pipeline Runner for AmazonHelp.

Executes the complete pipeline:
1. Graph-based conversation reconstruction from twcs.csv
2. Participant role normalization (customer, support) and validation
3. Multi-part support response grouping with audit trail
4. Language detection and confidence scoring via langdetect
5. Tier classification and streaming JSONL exports
6. Exploratory dialogue analysis and matplotlib plot generation
7. Intent discovery via SentenceTransformer embeddings and clustering
8. Generation of all Phase 2 documentation and reports
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

from src.config import AUDIT_CONFIG, PATHS, PIPELINE_CONFIG
from src.data.analysis import run_conversation_analysis
from src.data.intent_discovery import run_intent_clustering
from src.data.preprocess import ConversationPreprocessor, ProcessedConversation
from src.data.reconstruct import ConversationReconstructor
from src.utils.logger import get_logger

logger = get_logger("phase2_pipeline")


def reconstruct_conversations(csv_path: Path, brand: str = "AmazonHelp", max_rows: int = None):
    """Reconstructs threads for AmazonHelp from raw CSV using 2-pass streaming."""
    logger.info(f"Pass 1: Identifying tweets and references for brand: {brand}")
    brand_tweets: Set[int] = set()
    referenced_ids: Set[int] = set()

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
    logger.info(
        f"Identified {len(brand_tweets):,} {brand} tweets and {len(referenced_ids):,} referenced customer tweets."
    )

    logger.info("Pass 2: Loading target tweet rows into ConversationReconstructor...")
    reconstructor = ConversationReconstructor(brand=brand)
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

    logger.info("Traversing conversation graph components (BFS)...")
    conversations = reconstructor.reconstruct_all()
    return conversations


def generate_language_report(stats: Dict, output_path: Path):
    """Writes docs/language_filter_report.md."""
    lang_stats = stats["language_filtering_statistics"]
    conf_dist = lang_stats["confidence_distribution"]
    top_non_en = lang_stats["top_detected_non_english_languages"]

    content = f"""# Language Filtering & Verification Report

> **Dataset**: `AmazonHelp` Twitter Conversations  
> **Tool**: `langdetect` (Deterministic Seed: `{PIPELINE_CONFIG.SEED}`)  
> **Configured English Confidence Threshold**: `{PIPELINE_CONFIG.ENGLISH_CONFIDENCE_THRESHOLD}`

---

## 1. Summary of Language Filtering Results

| Metric | Conversation Count | Percentage (%) |
| :--- | :---: | :---: |
| **Total Reconstructed Conversations** | **{stats['total_reconstructed_conversations']:,}** | **100.0%** |
| **English Conversations (`is_english=True`)** | **{lang_stats['english_count']:,}** | **{lang_stats['english_pct']:.2f}%** |
| **Non-English Conversations (`prob >= 0.80`)** | **{lang_stats['non_english_count']:,}** | **{lang_stats['non_english_count']/stats['total_reconstructed_conversations']*100:.2f}%** |
| **Uncertain Conversations (`prob < 0.80` / short / ambiguous)** | **{lang_stats['uncertain_count']:,}** | **{lang_stats['uncertain_count']/stats['total_reconstructed_conversations']*100:.2f}%** |

---

## 2. Confidence Score Distribution (All Conversations)

| Confidence Interval | Conversation Count | Share (%) | Description |
| :--- | :---: | :---: | :--- |
| **0.90 – 1.00** | {conf_dist['0.90_1.00']:,} | {conf_dist['0.90_1.00']/stats['total_reconstructed_conversations']*100:.2f}% | High-certainty single-language prediction |
| **0.80 – 0.90** | {conf_dist['0.80_0.90']:,} | {conf_dist['0.80_0.90']/stats['total_reconstructed_conversations']*100:.2f}% | Acceptable certainty above threshold |
| **0.70 – 0.80** | {conf_dist['0.70_0.80']:,} | {conf_dist['0.70_0.80']/stats['total_reconstructed_conversations']*100:.2f}% | Sub-threshold; sent to uncertain bucket |
| **Below 0.70 / Failed** | {conf_dist['below_0.70']:,} | {conf_dist['below_0.70']/stats['total_reconstructed_conversations']*100:.2f}% | Ultra-short queries, handles, or emojis |

---

## 3. Top Detected Non-English Languages

Amazon operates international storefronts (.de, .es, .fr, .in, .co.jp). When non-English inquiries reach `@AmazonHelp`, they are detected and partitioned cleanly:

| Detected ISO Code | Conversation Count | Storefront Domain Context |
| :---: | :---: | :--- |
"""
    for code, count in top_non_en.items():
        content += f"| `{code}` | {count:,} | International customer query |\n"

    content += """
---

## 4. Destination File Mapping

- **English Benchmark Candidates**: `data/processed/amazonhelp_english.jsonl`
- **Non-English & Uncertain Holdout**: `data/processed/amazonhelp_non_english_or_uncertain.jsonl`

No data was silently discarded. All uncertain and non-English dialogues are preserved with their confidence metrics for future multilingual expansion.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Saved language report to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 2 AmazonHelp preprocessing pipeline.")
    parser.add_argument("--brand", type=str, default="AmazonHelp", help="Target brand")
    parser.add_argument("--dataset", type=Path, default=PATHS.get_raw_dataset_path(), help="twcs.csv path")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit rows for rapid testing")
    args = parser.parse_args()

    t_start = time.time()
    logger.info(f"Starting Phase 2 Pipeline for {args.brand} on dataset {args.dataset}...")

    # 1. Reconstruction
    raw_convs = reconstruct_conversations(csv_path=args.dataset, brand=args.brand, max_rows=args.max_rows)
    logger.info(f"Reconstructed {len(raw_convs):,} raw conversation threads.")

    # 2. Preprocessing & Tiering
    preprocessor = ConversationPreprocessor(
        brand=args.brand,
        english_threshold=PIPELINE_CONFIG.ENGLISH_CONFIDENCE_THRESHOLD,
        max_multipart_gap=PIPELINE_CONFIG.MULTI_PART_MAX_GAP_SECONDS,
    )

    logger.info("Preprocessing, validating, and streaming dataset tiers...")
    PATHS.ensure_directories()

    # Open streaming file handles
    f_english = open(PATHS.AMAZONHELP_ENGLISH_JSONL, "w", encoding="utf-8")
    f_non_english = open(PATHS.AMAZONHELP_NON_ENGLISH_JSONL, "w", encoding="utf-8")
    f_exploration = open(PATHS.AMAZONHELP_EXPLORATION_JSONL, "w", encoding="utf-8")
    f_retrieval = open(PATHS.AMAZONHELP_RETRIEVAL_JSONL, "w", encoding="utf-8")
    f_pristine = open(PATHS.AMAZONHELP_PRISTINE_JSONL, "w", encoding="utf-8")

    total_convs = len(raw_convs)
    total_tweets = 0
    valid_conv_count = 0
    invalid_conv_count = 0

    val_rejection_counts = Counter()
    consecutive_dup_count = 0
    impossible_role_count = 0
    missing_text_count = 0
    dup_tweet_id_count = 0
    timestamp_order_count = 0
    broken_link_count = 0
    cycle_count = 0

    english_count = 0
    non_english_count = 0
    uncertain_count = 0
    non_english_langs = Counter()
    conf_bins = {"0.90_1.00": 0, "0.80_0.90": 0, "0.70_0.80": 0, "below_0.70": 0}

    tier_counts = {"exploration": 0, "retrieval_candidate": 0, "pristine_candidate": 0}
    multipart_grouped_turns_total = 0
    convs_with_multipart_grouping = 0
    resolution_counts = Counter()

    processed_pristine_sample: List[ProcessedConversation] = []
    processed_all_sample: List[ProcessedConversation] = []

    for raw_conv in tqdm(raw_convs, desc="Processing conversations"):
        processed = preprocessor.process_conversation(raw_conv)
        total_tweets += len(raw_conv.turns)

        # Validation stats
        v = processed.validation
        if v.is_valid:
            valid_conv_count += 1
        else:
            invalid_conv_count += 1

        if v.has_consecutive_duplicates:
            consecutive_dup_count += 1
        if v.has_impossible_role_sequence:
            impossible_role_count += 1
        if v.has_missing_text:
            missing_text_count += 1
        if v.has_duplicate_tweet_ids:
            dup_tweet_id_count += 1
        if v.has_timestamp_order_error:
            timestamp_order_count += 1
        if processed.is_broken:
            broken_link_count += 1
        if processed.has_cycle:
            cycle_count += 1

        for r in v.rejection_reasons:
            val_rejection_counts[r.split("_")[0]] += 1

        # Multi-part grouping stats
        grouped_turns_in_conv = sum(1 for t in processed.normalized_turns if t.get("is_grouped", False))
        if grouped_turns_in_conv > 0:
            convs_with_multipart_grouping += 1
            multipart_grouped_turns_total += grouped_turns_in_conv

        # Language stats
        conf = processed.language_confidence
        if conf >= 0.90:
            conf_bins["0.90_1.00"] += 1
        elif conf >= 0.80:
            conf_bins["0.80_0.90"] += 1
        elif conf >= 0.70:
            conf_bins["0.70_0.80"] += 1
        else:
            conf_bins["below_0.70"] += 1

        if processed.is_english and not processed.is_uncertain_language:
            english_count += 1
        elif not processed.is_english and not processed.is_uncertain_language:
            non_english_count += 1
            non_english_langs[processed.language] += 1
        else:
            uncertain_count += 1

        # Resolution stats
        resolution_counts[processed.resolution_status] += 1

        # Tier streaming
        conv_json = json.dumps(processed.to_dict(), ensure_ascii=False) + "\n"

        if processed.is_english and not processed.is_uncertain_language:
            f_english.write(conv_json)
        else:
            f_non_english.write(conv_json)

        if "exploration" in processed.dataset_tiers:
            tier_counts["exploration"] += 1
            f_exploration.write(conv_json)

        if "retrieval_candidate" in processed.dataset_tiers:
            tier_counts["retrieval_candidate"] += 1
            f_retrieval.write(conv_json)

        if "pristine_candidate" in processed.dataset_tiers:
            tier_counts["pristine_candidate"] += 1
            f_pristine.write(conv_json)
            if len(processed_pristine_sample) < 15_000:
                processed_pristine_sample.append(processed)

        if len(processed_all_sample) < 25_000 and processed.is_english:
            processed_all_sample.append(processed)

    # Close file handles
    f_english.close()
    f_non_english.close()
    f_exploration.close()
    f_retrieval.close()
    f_pristine.close()

    # Compile preprocessing statistics JSON
    stats = {
        "brand": args.brand,
        "total_reconstructed_conversations": total_convs,
        "total_tweets_in_conversations": total_tweets,
        "validation_statistics": {
            "valid_conversations_count": valid_conv_count,
            "invalid_conversations_count": invalid_conv_count,
            "consecutive_duplicate_text_count": consecutive_dup_count,
            "impossible_role_sequence_count": impossible_role_count,
            "missing_text_count": missing_text_count,
            "duplicate_tweet_id_count": dup_tweet_id_count,
            "timestamp_order_error_count": timestamp_order_count,
            "broken_link_count": broken_link_count,
            "cycle_count": cycle_count,
            "rejection_reasons_summary": dict(val_rejection_counts.most_common(10)),
        },
        "language_filtering_statistics": {
            "english_count": english_count,
            "non_english_count": non_english_count,
            "uncertain_count": uncertain_count,
            "english_pct": round(english_count / total_convs * 100, 2) if total_convs else 0.0,
            "confidence_distribution": conf_bins,
            "top_detected_non_english_languages": dict(non_english_langs.most_common(8)),
        },
        "tier_counts": {
            "exploration_dataset_count": tier_counts["exploration"],
            "retrieval_candidates_count": tier_counts["retrieval_candidate"],
            "pristine_candidates_count": tier_counts["pristine_candidate"],
        },
        "multipart_grouping_statistics": {
            "total_grouped_turns": multipart_grouped_turns_total,
            "conversations_with_grouped_turns": convs_with_multipart_grouping,
            "pct_conversations_with_grouping": round(convs_with_multipart_grouping / total_convs * 100, 2)
            if total_convs
            else 0.0,
        },
        "resolution_status_statistics": dict(resolution_counts.most_common()),
    }

    with open(PATHS.PREPROCESSING_STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved preprocessing statistics to {PATHS.PREPROCESSING_STATS_JSON}")

    # Generate language filter report
    generate_language_report(stats, PATHS.LANGUAGE_FILTER_REPORT_MD)

    # 3. Exploratory Analysis & Plots
    analysis_pool = processed_pristine_sample if processed_pristine_sample else processed_all_sample
    logger.info(f"Running conversation analysis on sample of {len(analysis_pool):,} conversations...")
    analysis_results = run_conversation_analysis(analysis_pool, output_dir=PATHS.ANALYSIS_DIR)

    # 4. Intent Clustering & Artifact Generation
    logger.info("Executing intent clustering and generating discovery artifact...")
    run_intent_clustering(
        conversations=analysis_pool,
        k_clusters=PIPELINE_CONFIG.INTENT_CLUSTERS_K,
        sample_size=PIPELINE_CONFIG.INTENT_SAMPLE_SIZE,
        output_markdown_path=PATHS.INTENT_DISCOVERY_EXAMPLES_MD,
    )

    elapsed = time.time() - t_start
    logger.info(f"Phase 2 pipeline completed successfully in {elapsed:.2f} seconds ({elapsed/60:.1f} minutes).")


if __name__ == "__main__":
    main()
