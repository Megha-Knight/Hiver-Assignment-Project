# Phase 7A — Repository Cleanliness & Submission Size Audit

**Audit Date:** 2026-09-16  
**Auditor:** Automated Take-Home Evaluation Hardening Agent  
**Status:** PLAN PROPOSED (Zero Files Deleted in Phase 7A per Safety Governance)

---

## 1. Executive Summary

This audit evaluates the repository storage footprint, categorizes all 198 files across the repository tree, and defines a surgical cleanup plan to optimize repository size for external evaluation.

* **Current Total Repository Footprint:** **940.72 MB** across 198 files.
* **Primary Size Driver:** Pre-processed intermediate data files in `data/processed/` account for **909.15 MB (96.6%)** of total repository disk space.
* **Target Submission Size (Post-Cleanup / Optimized):** **~31.5 MB**, representing a **96.6% reduction** without loss of any evaluation capability or verification repeatability.

Per Rule 3, **no files have been deleted during Phase 7A**. This cleanup plan categorizes every file into **KEEP**, **ARCHIVE**, or **REMOVE** for action in Phase 7B.

---

## 2. Storage Breakdown by Category

| Category | File Count | Current Size | Target Disposition | Post-Cleanup Size |
| :--- | :--- | :--- | :--- | :--- |
| **GENERATED_ARTIFACT (Intermediate Data)** | 36 files | 909.15 MB | **ARCHIVE / EXCLUDE** | 0.00 MB |
| **INDEX (Precomputed Vector Embeddings)** | 2 files | 18.23 MB | **KEEP** | 18.23 MB |
| **DATA (Train Retrieval Corpus)** | 1 file | 9.35 MB | **KEEP** | 9.35 MB |
| **MODEL (Baseline Joblib Weights)** | 1 file | 1.66 MB | **KEEP** | 1.66 MB |
| **EVALUATION (Golden Human Benchmark)** | 3 files | 0.96 MB | **KEEP** | 0.96 MB |
| **TEMPORARY (`__pycache__`, `.pyc`)** | 53 files | 0.60 MB | **REMOVE** | 0.00 MB |
| **SCRIPT (Verification & Run Harnesses)** | 26 files | 0.31 MB | **KEEP** | 0.31 MB |
| **CORE_RUNTIME (`src/` modules)** | 37 files | 0.25 MB | **KEEP** | 0.25 MB |
| **DOCUMENTATION (`docs/`, `results/`)** | 20 files | 0.17 MB | **KEEP** | 0.17 MB |
| **EXPERIMENT (Analysis tables)** | 7 files | 0.04 MB | **KEEP** | 0.04 MB |
| **TOTAL** | **198 files** | **940.72 MB** | **Action in Phase 7B** | **~31.5 MB** |

---

## 3. Surgical Disposition Plan

### 3.1 KEEP (Essential for Immediate Evaluation & Verification)
These files MUST remain committed in the final repository so any evaluator can immediately clone and run the agent and tests without downloading external files:

1. **Production Code (`src/`):** All 37 source files (data pipeline, policy engines, retrieval engine, LLM client, schemas, safety validator, agent).
2. **Authoritative Golden Benchmark (`data/golden/`):**
   * `amazonhelp_golden_v1_human_validated.jsonl` (200 checkpoints, 0.44 MB)
   * `human_validation_log.jsonl` (Audit trail of human decisions, 0.45 MB)
   * `golden_candidates.jsonl` (Pre-label candidates, 0.07 MB)
3. **Train-Only Retrieval Corpus (`data/retrieval/`):**
   * `amazonhelp_train_retrieval.jsonl` (5,502 dialogues, 9.35 MB)
4. **Pre-computed Retrieval Index (`data/indexes/`):**
   * `retrieval_index.npz` (MiniLM dense embeddings, 17.34 MB)
   * `retrieval_index_meta.json` (Index lookup metadata, 0.89 MB)
5. **Trained Baseline Model (`models/baselines/`):**
   * `tfidf_logreg_intent.joblib` (1.66 MB)
6. **Documentation & Reports (`docs/`, `results/`):** All Markdown reports, architecture specs, and JSON evaluation metrics.
7. **Verification & Execution Scripts (`scripts/`):** All `verify_*.py`, `run_*.py`, and `compare_*.py` tools.
8. **Configuration & Dependencies:** `requirements.txt`, `.env.example`, `.gitignore`, `README.md`.

---

### 3.2 ARCHIVE / EXCLUDE (Generated Intermediate Files: 909 MB)
These files are reproducible intermediate outputs of the raw data preparation pipeline (`run_phase2_pipeline.py`). They are **not** needed to run or evaluate the support agent, as the final clean corpus (`amazonhelp_train_retrieval.jsonl`) and index are already packaged:

1. `data/processed/amazonhelp_english.jsonl` (374.2 MB) — All 63,493 English conversations.
2. `data/processed/amazonhelp_retrieval_candidates.jsonl` (232.1 MB) — Unfiltered retrieval candidates.
3. `data/processed/amazonhelp_pristine_candidates.jsonl` (223.4 MB) — Partition staging pool.
4. `data/processed/amazonhelp_exploration.jsonl` (79.4 MB) — Phase 2 exploratory sample.
5. `data/processed/amazonhelp_non_english_or_uncertain.jsonl` (0.05 MB) — Filtered non-English.
6. Temporary cluster analysis JSONs in `results/analysis/` (0.01 MB).

**Action for Phase 7B:** Add `data/processed/*.jsonl` to `.gitignore` so they are not included in the git repository submission, and document the reproduction command in `README.md`.

---

### 3.3 REMOVE (Cache & Build Residue: 0.60 MB)
These files are bytecode caches and ephemeral runtime files that should never be committed to git:

1. `**/__pycache__/*.pyc` (53 files across `src/` and `scripts/`, ~0.60 MB)
2. Operating system metadata (`.DS_Store`, `Thumbs.db` if present)
3. Editor-specific local states (`.vscode/` if present)

**Action for Phase 7B:** Delete `__pycache__` directories and confirm `.gitignore` includes `**/__pycache__/` and `*.pyc`.

---

## 4. Verification Check

This cleanup plan will be validated by Check 21 of `scripts/verify_phase7a.py`.
