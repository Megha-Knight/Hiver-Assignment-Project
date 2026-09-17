"""Master CLI orchestrator for Phase 3: Golden Evaluation & Retrieval Index.

Orchestrates:
1. Sampling 200 stratified golden candidates from the Validation partition.
2. Generating and validating versioned golden evaluation benchmark (amazonhelp_golden_v1.jsonl).
3. Verifying the Train-only historical retrieval corpus and dense vector index.
4. Executing strict automated leakage and schema verification suites.
"""

import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sample_golden_candidates import main as sample_golden_main
from scripts.annotate_golden import run_batch_pre_annotation, run_batch_expert_adjudication
from scripts.build_retrieval_index import main as build_retrieval_main
from scripts.verify_phase3_retrieval import main as verify_main
from src.config import PATHS
from src.utils.logger import get_logger

logger = get_logger("build_phase3")


def main():
    parser = argparse.ArgumentParser(description="Run complete Phase 3 pipeline.")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild of dense vector index")
    args = parser.parse_args()

    t0 = time.time()
    logger.info("=" * 70)
    logger.info("EXECUTING PHASE 3: GOLDEN BENCHMARK + RETRIEVAL FOUNDATION")
    logger.info("=" * 70)

    # Step 1: Sample Golden Evaluation Candidates
    logger.info("[Step 1/4] Sampling Golden Candidates from Validation partition...")
    sample_golden_main()

    # Step 2: Annotate Golden Baseline and Human-Validated Benchmark
    logger.info("[Step 2/4] Generating pre-annotations and running human adjudication...")
    run_batch_pre_annotation(
        candidates_path=PATHS.GOLDEN_CANDIDATES_JSONL,
        output_path=PATHS.AMAZONHELP_GOLDEN_JSONL,
    )
    run_batch_expert_adjudication(
        input_path=PATHS.AMAZONHELP_GOLDEN_JSONL,
        output_path=PATHS.AMAZONHELP_GOLDEN_HUMAN_VALIDATED_JSONL,
        report_path=PATHS.GOLDEN_HUMAN_ANNOTATION_REPORT_MD,
    )

    # Step 3: Check or Build Retrieval Index
    logger.info("[Step 3/4] Checking Train-only Historical Retrieval Corpus and Vector Index...")
    if args.rebuild_index or not PATHS.RETRIEVAL_INDEX_NPZ.exists():
        build_retrieval_main()
    else:
        logger.info(f"Verified existing retrieval index at {PATHS.RETRIEVAL_INDEX_NPZ} ({PATHS.RETRIEVAL_INDEX_NPZ.stat().st_size:,} bytes).")

    # Step 4: Run Strict Verification Suite
    logger.info("[Step 4/4] Running automated Phase 3 verification suite...")
    verify_main()

    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info(f"PHASE 3 PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
