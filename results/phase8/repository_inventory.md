# Phase 8: Repository Artifact & Data Inventory

**Date**: 2026-09-16  
**Auditor**: Senior Evaluation & Release Engineer  
**Status**: CLEAN & ISOLATED

---

## 1. Directory Structure & Footprint Breakdown

The repository is partitioned into modular, isolated directories with strict boundaries separating source code, runtime assets, frozen benchmarks, and git-ignored raw data.

| Directory | Purpose | File Count | Size (Approx) | Tracked in Git |
| :--- | :--- | :---: | :---: | :---: |
| **`src/`** | Production agent, policy, retrieval, evaluation, state modules | 45 files | ~390 KB | **YES** |
| **`tests/`** | Smoke tests, Phase 7C/7D unittests (28 total tests) | 3 files | ~30 KB | **YES** |
| **`scripts/`** | Governance verification scripts (Phase 7A–8) & tools | 36 files | ~425 KB | **YES** |
| **`docs/`** | Architecture specs, decision log, demo script, reports | 20+ files | ~250 KB | **YES** |
| **`results/`** | Official experimental metrics, audits, telemetry, Phase 4–8 reports | 54 files | ~1.2 MB | **YES** |
| **`data/golden/`** | 200 human-validated Dev checkpoints & splits | 4 files | ~1.16 MB | **YES** |
| **`data/retrieval/`**| 5,502 Train-only historical support dialogues | 1 file | ~9.5 MB | **YES** |
| **`data/indexes/`** | Pre-computed MiniLM vector index (`retrieval_index.npz`) | 2 files | ~18.6 MB | **YES** |
| **`models/baselines/`**| Trained TF-IDF + Logistic Regression weights (`tfidf_logreg_intent.joblib`)| 1 file | ~1.7 MB | **YES** |
| **`data/evaluation/`**| $N=40$ human review packet & completed blinded human reviews | 2 files | ~223 KB | **YES** |
| **`data/processed/`**| Large intermediate conversation files (85k dialogues) | 5 files | ~930 MB | **NO (Ignored)** |
| **Root Files** | `README.md`, `cli.py`, `requirements.txt`, `.gitignore`, `.env.example` | 5 files | ~30 KB | **YES** |

---

## 2. Core Runtime Assets (Required for Evaluation)

The following assets are committed and tracked in git, enabling immediate reproduction without training or downloading vector indexes:

1. **Golden Evaluation Benchmark**:
   - `data/golden/amazonhelp_golden_v1_human_validated.jsonl` ($N=200$ Dev checkpoints; 49 Easy, 91 Medium, 60 Hard).
2. **Train-Only Retrieval Corpus**:
   - `data/retrieval/amazonhelp_train_retrieval_corpus.jsonl` (5,502 verified AmazonHelp historical resolution dialogues; zero Dev/Test leakage).
3. **Pre-computed Retrieval Vector Index**:
   - `data/indexes/retrieval_index.npz` (5,502 $\times$ 384 embeddings pre-indexed via `all-MiniLM-L6-v2`).
4. **Baseline Model Weights**:
   - `models/baselines/tfidf_logreg_intent.joblib` (Pre-trained Phase 4 TF-IDF + Logistic Regression model; 87.50% Intent Accuracy).
5. **Blinded Human Review Records**:
   - `data/evaluation/human_review_packet_n40.jsonl` (Blinded review packet, zero gold labels).
   - `data/evaluation/human_reviews_n40.jsonl` (40 completed human evaluation records).

---

## 3. Data Isolation & Leakage Boundary

- **Train Set**: 5,502 dialogues utilized exclusively for retrieval corpus and TF-IDF training.
- **Validation (Dev) Set**: 200 human-validated checkpoints used for model development, threshold tuning, and Phase 7C/7D response evaluation.
- **Test Set**: 5,365 chronologically final conversations remain completely pristine, untouched, and unindexed.

---

## 4. Git Cleanliness & Exclusion Rules

The [`.gitignore`](.gitignore) explicitly excludes:
- `data/processed/*.jsonl` (930 MB intermediate dataset reconstruction).
- `data/raw/*.csv` (Raw Kaggle 800MB CSV).
- Python virtualenvs (`.venv/`, `venv/`).
- Bytecode caches (`__pycache__/`, `*.pyc`).
- Transient execution logs (`*.log`, `results/*.tmp`).

---

## 5. Summary

The repository package is compact, clean, self-contained (~32 MB total tracked assets), and fully reproducible out of the box.
