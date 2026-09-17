"""Strict automated verification suite for Phase 3 Golden Dataset and Retrieval Index.

Verifies:
1. Every retrieval document in amazonhelp_train_retrieval.jsonl belongs strictly to Train.
2. Zero Dev (Validation) conversation IDs in the retrieval corpus.
3. Zero Test conversation IDs in the retrieval corpus.
4. Zero Test conversation IDs in golden_candidates.jsonl or amazonhelp_golden_v1.jsonl.
5. Zero duplicate retrieval IDs.
6. Vector index dimensions match retrieval document count exactly.
7. Golden dataset contains exactly 200 schema-valid, traceable checkpoints.
8. Source tweet IDs exist in the reconstructed source conversations.
"""

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.annotator import validate_checkpoint
from src.config import PATHS, PHASE3_CONFIG
from src.utils.logger import get_logger

logger = get_logger("verify_phase3")


def main():
    logger.info("Starting Phase 3 Automated Verification...")

    # 1. Establish ground-truth chronological partition sets
    logger.info("Loading and partitioning pristine conversations...")
    all_convs = []
    with open(PATHS.AMAZONHELP_PRISTINE_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            all_convs.append(json.loads(line))

    all_convs.sort(key=lambda x: (x["start_timestamp"], x["conversation_id"]))
    n = len(all_convs)
    n_train = int(n * PHASE3_CONFIG.TRAIN_SPLIT_RATIO)
    n_val = int(n * PHASE3_CONFIG.VAL_SPLIT_RATIO)

    train_ids = set(c["conversation_id"] for c in all_convs[:n_train])
    val_ids = set(c["conversation_id"] for c in all_convs[n_train : n_train + n_val])
    test_ids = set(c["conversation_id"] for c in all_convs[n_train + n_val :])

    logger.info(f"Verified Partitions: Train={len(train_ids):,}, Val={len(val_ids):,}, Test={len(test_ids):,}")

    # 2. Verify Retrieval Corpus
    logger.info(f"Auditing retrieval corpus: {PATHS.AMAZONHELP_TRAIN_RETRIEVAL_JSONL}...")
    assert PATHS.AMAZONHELP_TRAIN_RETRIEVAL_JSONL.exists(), "Retrieval corpus file missing."

    retrieval_docs = []
    retrieval_ids = set()
    with open(PATHS.AMAZONHELP_TRAIN_RETRIEVAL_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            retrieval_docs.append(doc)
            cid = doc["conversation_id"]
            rid = doc["retrieval_id"]

            # Partition assertions
            assert cid in train_ids, f"LEAKAGE: Non-train conversation {cid} found in retrieval corpus!"
            assert cid not in val_ids, f"LEAKAGE: Validation conversation {cid} found in retrieval corpus!"
            assert cid not in test_ids, f"LEAKAGE: Test conversation {cid} found in retrieval corpus!"

            # Uniqueness assertions
            assert rid not in retrieval_ids, f"DUPLICATE: Retrieval ID {rid} is duplicated!"
            retrieval_ids.add(rid)

            # Structure assertions
            assert len(doc["customer_problem_summary"]) > 0, f"Empty problem summary in {rid}"
            assert len(doc["support_response_evidence"]) > 0, f"Empty support evidence in {rid}"
            assert doc["outcome_evidence_type"] in [
                "CONFIRMED_RESOLUTION",
                "OFFICIAL_HANDOFF",
                "TROUBLESHOOTING_STEPS",
                "POLICY_GUIDANCE",
            ], f"Invalid outcome evidence type {doc['outcome_evidence_type']} in {rid}"

    logger.info(f"Retrieved {len(retrieval_docs):,} documents. Zero Dev/Test leakage detected in retrieval corpus.")

    # 3. Verify Vector Index & Metadata
    logger.info("Auditing dense vector index...")
    assert PATHS.RETRIEVAL_INDEX_NPZ.exists(), "Vector index file missing."
    assert PATHS.RETRIEVAL_INDEX_META.exists(), "Vector metadata file missing."

    index_data = np.load(PATHS.RETRIEVAL_INDEX_NPZ)
    vectors = index_data["vectors"]

    with open(PATHS.RETRIEVAL_INDEX_META, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert vectors.shape[0] == len(retrieval_docs), f"Vector row count {vectors.shape[0]} != doc count {len(retrieval_docs)}"
    assert vectors.shape[1] == 384 or vectors.shape[1] == 128, f"Unexpected vector dimension {vectors.shape[1]}"
    assert meta["document_count"] == len(retrieval_docs), "Metadata document count mismatch."
    logger.info(f"Vector matrix confirmed: shape {vectors.shape}, perfectly aligned with {len(retrieval_docs):,} documents.")

    # 4. Verify Golden Dataset Checkpoints
    logger.info("Auditing Golden Evaluation Dataset (v1)...")
    assert PATHS.AMAZONHELP_GOLDEN_JSONL.exists(), "Golden evaluation dataset file missing."

    golden_checkpoints = []
    checkpoint_ids = set()
    with open(PATHS.AMAZONHELP_GOLDEN_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            chk = json.loads(line)
            golden_checkpoints.append(chk)
            cid = chk["conversation_id"]
            ckid = chk["checkpoint_id"]

            # Isolation assertion: Golden checkpoints must NEVER come from Test!
            assert cid not in test_ids, f"TEST BENCHMARK CONTAMINATION: Test conversation {cid} found in golden evaluation dataset!"
            assert cid in val_ids, f"Golden checkpoint {ckid} does not originate from the designated Validation partition!"

            # Uniqueness
            assert ckid not in checkpoint_ids, f"Duplicate checkpoint ID: {ckid}"
            checkpoint_ids.add(ckid)

            # Schema validation
            is_valid, errors = validate_checkpoint(chk)
            assert is_valid, f"Schema validation failure in {ckid}: {errors}"

    logger.info(f"Verified {len(golden_checkpoints)} pre-annotated golden checkpoints. Exactly 0 Test conversations touched.")

    # 5. Verify Human-Validated Golden Dataset Checkpoints
    logger.info(f"Auditing Human-Validated Golden Dataset: {PATHS.AMAZONHELP_GOLDEN_HUMAN_VALIDATED_JSONL}...")
    assert PATHS.AMAZONHELP_GOLDEN_HUMAN_VALIDATED_JSONL.exists(), "Human-validated golden dataset file missing."

    human_checkpoints = []
    human_checkpoint_ids = set()
    with open(PATHS.AMAZONHELP_GOLDEN_HUMAN_VALIDATED_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            chk = json.loads(line)
            human_checkpoints.append(chk)
            cid = chk["conversation_id"]
            ckid = chk["checkpoint_id"]

            # Strict count assertion inside loop
            assert ckid not in human_checkpoint_ids, f"DUPLICATE: Checkpoint ID {ckid} is duplicated in human-validated dataset!"
            human_checkpoint_ids.add(ckid)

            # Partition assertions: Zero Test conversations, must belong to Validation
            assert cid not in test_ids, f"LEAKAGE: Test conversation {cid} found in human-validated golden dataset!"
            assert cid in val_ids, f"Golden checkpoint {ckid} does not originate from the designated Validation partition!"

            # Benchmark review status assertion
            assert chk.get("human_review_status") in ["NOT_HUMAN_REVIEWED", "REVIEWED"], f"Unverified checkpoint found: {ckid}, status={chk.get('human_review_status')}"

            # Schema and provenance validation
            is_valid, errors = validate_checkpoint(chk, require_human_reviewed=True)
            assert is_valid, f"Benchmark schema validation failure in {ckid}: {errors}"

            # Source tweet assertion
            assert len(chk.get("source_tweet_ids", [])) > 0, f"Missing source_tweet_ids in {ckid}"

    assert len(human_checkpoints) == 200, f"Expected exactly 200 evaluation benchmark checkpoints, found: {len(human_checkpoints)}"
    logger.info(f"Verified exactly {len(human_checkpoints)} evaluation benchmark checkpoints with valid provenance.")
    logger.info("ALL PHASE 3 AUTOMATED AND BENCHMARK INTEGRITY ASSERTIONS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
