"""Validation script for Phase 2 outputs and constraints.

Verifies:
1. All required output files exist and are non-empty
2. Role normalization across all turns
3. Chronological timestamp ordering
4. Uniqueness of tweet IDs within every conversation
5. Language filtering consistency
6. 20 random conversation inspections
"""

import json
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, PIPELINE_CONFIG
from src.utils.logger import get_logger

logger = get_logger("verify_phase2")


def main():
    logger.info("Starting comprehensive Phase 2 verification...")

    required_files = [
        PATHS.AMAZONHELP_EXPLORATION_JSONL,
        PATHS.AMAZONHELP_RETRIEVAL_JSONL,
        PATHS.AMAZONHELP_PRISTINE_JSONL,
        PATHS.AMAZONHELP_ENGLISH_JSONL,
        PATHS.AMAZONHELP_NON_ENGLISH_JSONL,
        PATHS.PREPROCESSING_STATS_JSON,
        PATHS.ANALYSIS_DIR / "conv_length_distribution.png",
        PATHS.ANALYSIS_DIR / "message_length_distribution.png",
        PATHS.ANALYSIS_DIR / "turn_alternation_distribution.png",
        PATHS.ANALYSIS_DIR / "resolution_status_distribution.png",
        PATHS.ANALYSIS_DIR / "escalation_and_handoff_frequencies.png",
        PATHS.LANGUAGE_FILTER_REPORT_MD,
        PATHS.INTENT_DISCOVERY_EXAMPLES_MD,
        PATHS.INTENT_TAXONOMY_PROPOSAL_MD,
        PATHS.CONVERSATION_STATE_PROPOSAL_MD,
        PATHS.AGENT_ACTION_PROPOSAL_MD,
        PATHS.AUTONOMY_POLICY_MD,
        PATHS.EVALUATION_SPLIT_STRATEGY_MD,
    ]

    missing_files = []
    for p in required_files:
        if not p.exists() or p.stat().st_size == 0:
            missing_files.append(p)
        else:
            logger.info(f"Verified artifact exists: {p.relative_to(PROJECT_ROOT)} ({p.stat().st_size:,} bytes)")

    if missing_files:
        logger.error(f"Missing or empty artifacts: {missing_files}")
        sys.exit(1)

    # Verify line counts
    def count_jsonl_lines(path: Path) -> int:
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for _ in f:
                count += 1
        return count

    counts = {
        "exploration": count_jsonl_lines(PATHS.AMAZONHELP_EXPLORATION_JSONL),
        "retrieval": count_jsonl_lines(PATHS.AMAZONHELP_RETRIEVAL_JSONL),
        "pristine": count_jsonl_lines(PATHS.AMAZONHELP_PRISTINE_JSONL),
        "english": count_jsonl_lines(PATHS.AMAZONHELP_ENGLISH_JSONL),
        "non_english_or_uncertain": count_jsonl_lines(PATHS.AMAZONHELP_NON_ENGLISH_JSONL),
    }

    print("\n" + "=" * 60)
    print("PHASE 2 DATASET TIER COUNTS:")
    print("=" * 60)
    for tier, cnt in counts.items():
        print(f"  {tier:26s}: {cnt:,} conversations")
    print("=" * 60 + "\n")

    # Load stats JSON
    with open(PATHS.PREPROCESSING_STATS_JSON, "r", encoding="utf-8") as f:
        stats = json.load(f)

    assert counts["exploration"] == stats["tier_counts"]["exploration_dataset_count"]
    assert counts["pristine"] == stats["tier_counts"]["pristine_candidates_count"]
    assert counts["english"] == stats["language_filtering_statistics"]["english_count"]
    logger.info("Dataset line counts match preprocessing_statistics.json exactly.")

    # Randomly inspect 20 pristine conversations
    random.seed(PIPELINE_CONFIG.SEED)
    pristine_convs = []
    with open(PATHS.AMAZONHELP_PRISTINE_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            pristine_convs.append(json.loads(line))

    sample_20 = random.sample(pristine_convs, 20)
    logger.info(f"Randomly inspecting 20 conversations from pristine tier...")

    for idx, conv in enumerate(sample_20, 1):
        # 1. Role normalization check
        roles = [t["role"] for t in conv["normalized_turns"]]
        for r in roles:
            assert r in ["customer", "support"], f"Invalid role {r} in {conv['conversation_id']}"

        # 2. Chronological order check
        timestamps = [t["timestamp"] for t in conv["normalized_turns"]]
        for i in range(len(timestamps) - 1):
            assert timestamps[i] <= timestamps[i + 1], f"Timestamp ordering error in {conv['conversation_id']}"

        # 3. Duplicate tweet ID check
        all_tweet_ids = []
        for t in conv["normalized_turns"]:
            all_tweet_ids.extend(t["tweet_ids"])
        assert len(all_tweet_ids) == len(set(all_tweet_ids)), f"Duplicate tweet ID in {conv['conversation_id']}"

        # 4. Text validity
        for t in conv["normalized_turns"]:
            assert len(t["text"].strip()) > 0, f"Empty text in {conv['conversation_id']}"

        # 5. Allowed resolution status
        assert conv["resolution_status"] in [
            "UNKNOWN",
            "APPARENTLY_RESOLVED",
            "UNRESOLVED",
            "HANDOFF_OR_DM",
            "INSUFFICIENT_EVIDENCE",
        ], f"Invalid resolution status in {conv['conversation_id']}"

        print(
            f"[{idx:02d}/20] Verified {conv['conversation_id']} | Turns: {len(conv['normalized_turns'])} | "
            f"Lang: {conv['language']} ({conv['language_confidence']:.2f}) | Status: {conv['resolution_status']} | PASS"
        )

    logger.info("All 20 random inspections PASSED all structural, role, and ordering assertions.")
    logger.info("Phase 2 Verification completed successfully!")


if __name__ == "__main__":
    main()
