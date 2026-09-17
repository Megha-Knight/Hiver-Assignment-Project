"""Builds the Train-only historical retrieval corpus and vector index.

Workflow:
1. Strict partition filtering: Ingests ONLY the 42,909 Train conversations.
2. Filters out non-actionable complaints, drop-offs, and boilerplate.
3. Formats multi-turn retrieval documents with explicit evidence typing.
4. Generates data/retrieval/amazonhelp_train_retrieval.jsonl.
5. Encodes documents into normalized dense vectors using local embedding engine.
6. Exports vector matrix to data/indexes/retrieval_index.npz and metadata to retrieval_index_meta.json.
7. Saves comprehensive metrics to results/phase3_statistics.json and results/retrieval_build_report.md.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, PHASE3_CONFIG
from src.retrieval.corpus_builder import build_train_retrieval_corpus, generate_retrieval_report
from src.retrieval.embeddings import get_embedder
from src.utils.logger import get_logger

logger = get_logger("build_retrieval_index")


def main():
    parser = argparse.ArgumentParser(description="Build historical retrieval corpus and vector index.")
    parser.add_argument("--pristine", type=Path, default=PATHS.AMAZONHELP_PRISTINE_JSONL)
    parser.add_argument("--output-corpus", type=Path, default=PATHS.AMAZONHELP_TRAIN_RETRIEVAL_JSONL)
    parser.add_argument("--output-index", type=Path, default=PATHS.RETRIEVAL_INDEX_NPZ)
    parser.add_argument("--output-meta", type=Path, default=PATHS.RETRIEVAL_INDEX_META)
    args = parser.parse_args()

    t0 = time.time()
    logger.info("Starting Phase 3 Retrieval Index Construction...")
    PATHS.ensure_directories()

    # 1. Build and filter Train-only corpus
    retrieval_docs, stats = build_train_retrieval_corpus(
        pristine_path=args.pristine,
        output_path=args.output_corpus,
    )
    logger.info(f"Retained {len(retrieval_docs):,} vetted retrieval documents from {stats['total_train_conversations']:,} Train conversations.")

    # 2. Generate Markdown build report
    generate_retrieval_report(stats, PATHS.RETRIEVAL_BUILD_REPORT_MD)

    # 3. Vector Embedding & Indexing
    logger.info(f"Initializing embedding engine ({PHASE3_CONFIG.EMBEDDING_MODEL_NAME})...")
    embedder = get_embedder(model_name=PHASE3_CONFIG.EMBEDDING_MODEL_NAME)

    # Texts to embed: combine problem summary + support guidance
    corpus_texts = []
    for doc in retrieval_docs:
        combined = f"{doc['customer_problem_summary']} Support: {doc['support_response_evidence']}"
        corpus_texts.append(combined)

    logger.info(f"Encoding {len(corpus_texts):,} documents into dense vector space...")
    vectors = embedder.embed_texts(corpus_texts)
    logger.info(f"Vector matrix created with shape {vectors.shape} and dtype {vectors.dtype}.")

    # 4. Save Index & Metadata
    args.output_index.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_index, vectors=vectors)
    logger.info(f"Saved compressed vector index to: {args.output_index}")

    metadata_payload = {
        "index_version": "v1.0_train_only",
        "embedder_model": embedder.model_name,
        "vector_dimension": embedder.dimension,
        "document_count": len(retrieval_docs),
        "source_corpus_file": str(args.output_corpus.relative_to(PROJECT_ROOT)),
        "documents": retrieval_docs,
    }

    with open(args.output_meta, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)
    logger.info(f"Saved index metadata to: {args.output_meta}")

    # 5. Compile and export Phase 3 statistics JSON
    phase3_stats = {
        "phase": 3,
        "retrieval_corpus_statistics": stats,
        "vector_index_statistics": {
            "model_name": embedder.model_name,
            "dimension": embedder.dimension,
            "indexed_documents": len(retrieval_docs),
            "index_file_size_bytes": args.output_index.stat().st_size,
        },
        "build_duration_seconds": round(time.time() - t0, 2),
    }

    with open(PATHS.PHASE3_STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(phase3_stats, f, indent=2)
    logger.info(f"Saved Phase 3 statistics to: {PATHS.PHASE3_STATS_JSON}")

    elapsed = time.time() - t0
    logger.info(f"Retrieval index construction completed successfully in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
