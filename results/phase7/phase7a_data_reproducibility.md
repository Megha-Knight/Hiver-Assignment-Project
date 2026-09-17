# Phase 7A — Data Reproducibility & Pipeline Lineage Audit

**Audit Date:** 2026-09-16  
**Auditor:** Automated Take-Home Evaluation Hardening Agent  
**Status:** PASS (Fully Documented, Deterministic, & Leakage-Free)

---

## 1. Executive Summary

This document details the exact lineage, transformation stages, partition controls, and data reproduction instructions for the AmazonHelp Customer Support AI Agent.

To ensure zero friction for external evaluators, the repository ships with all essential processed evaluation data:
1. **Human-Validated Golden Benchmark** (200 checkpoints, Dev partition only)
2. **Historical Retrieval Corpus** (5,502 dialogues, Train partition only)
3. **Pre-computed Dense Vector Index** (MiniLM-L6-v2 embeddings, 18.2 MB)
4. **Trained Baseline Model** (`tfidf_logreg_intent.joblib`, 1.66 MB)

An evaluator does **not** need to re-download the multi-gigabyte Kaggle Twitter dataset to run, verify, or evaluate the agent. However, the complete deterministic reconstruction pipeline is fully documented and provided should end-to-end reproduction from raw CSV be desired.

---

## 2. Dataset Lineage & Transformation Pipeline

The data flows deterministically through the following pipeline:

```
Kaggle "Customer Support on Twitter" (twcs.csv, ~2.8M tweets)
  │
  ▼ [scripts/run_reconstruction.py]
Brand Ingestion & Multi-Turn Conversation Reconstruction
  │ Filter: @AmazonHelp (85,087 conversations reconstructed)
  ▼
Language & Quality Filtering [scripts/run_phase2_pipeline.py]
  │ langdetect (confidence >= 0.80) -> 63,493 English conversations
  ▼
Deterministic 3-Way Partitioning (Seed = 42)
  ├── Train Split:      80% (42,909 conversations)
  ├── Validation Split: 10% ( 5,363 conversations)
  └── Test Split:       10% ( 5,365 conversations) [HELD-OUT / UNTOUCHED]
  │
  ├──► Train Partition ONLY:
  │      └── Filter: Multi-turn (depth >= 2), clean dialogue
  │      └── Output: data/retrieval/amazonhelp_train_retrieval.jsonl (5,502 dialogues)
  │      └── Index:  data/indexes/retrieval_index.npz (all-MiniLM-L6-v2)
  │
  └──► Validation Partition ONLY:
         └── Sample: Stratified candidate sampling (200 checkpoints)
         └── Pre-label: Rule-based annotator (Phase 3 candidate)
         └── Human Adjudication: 100% human reviewed & validated
         └── Output: data/golden/amazonhelp_golden_v1_human_validated.jsonl (200 checkpoints)
```

---

## 3. Partitioning & Strict Anti-Leakage Guarantees

| Data Artifact | Partition Source | Conversation Count | Checkpoint Count | Leakage Check |
| :--- | :--- | :--- | :--- | :--- |
| `amazonhelp_train_retrieval.jsonl` | **Train (80%)** | 5,502 | N/A | **0 Dev / 0 Test overlap** |
| `retrieval_index.npz` | **Train (80%)** | 5,502 | N/A | **0 Dev / 0 Test overlap** |
| `amazonhelp_golden_v1_human_validated.jsonl` | **Validation / Dev (10%)** | 196 (unique) | 200 | **0 Train / 0 Test overlap** |
| **Test Partition** | **Test (10%)** | 5,365 | N/A | **FROZEN / NEVER ACCESSED** |

### Verified Leakage Invariants:
1. **Corpus Isolation:** The retrieval engine indexes **only** Train partition conversations. No Dev or Test conversations exist in `retrieval_index.npz`.
2. **Benchmark Isolation:** The 200 human-validated evaluation checkpoints are drawn strictly from the Validation/Dev set.
3. **Pristine Test Set:** The 5,365 Test set conversations were never loaded, fitted, or indexed during baseline training, retrieval index construction, prompt engineering, or human validation.

---

## 4. Ground Truth Human Validation

The 200 evaluation checkpoints in `data/golden/amazonhelp_golden_v1_human_validated.jsonl` represent gold ground-truth established through expert human adjudication:
- **Pre-annotation:** Automated rule-based pre-labeling.
- **Human Review:** 100% of the 200 checkpoints were individually evaluated by a human adjudicator inspecting multi-turn dialogue history, current customer text, ambiguity cues, and risk triggers.
- **Corrections Applied:** 17 labels were explicitly corrected during adjudication (detailed in `results/golden_human_annotation_report.md` and logged in `data/golden/human_validation_log.jsonl`).
- **Final Benchmark:** Every checkpoint contains validated fields: `intent`, `state`, `action`, `escalate`, `escalation_reason`, and `difficulty` (49 Easy, 91 Medium, 60 Hard).

---

## 5. Instructions for End-to-End Reproduction from Raw Data

Should an evaluator wish to rebuild the pipeline from scratch:

### Step 1: Download Raw Kaggle Dataset
Obtain `twcs.csv` from Kaggle (ThoughtVector Customer Support on Twitter) and place it at:
`data/raw/twcs.csv` (or in the parent workspace directory `../twcs/twcs.csv`).

### Step 2: Reconstruct Conversations
```bash
python scripts/run_reconstruction.py
```
*Outputs reconstructed conversations for candidate brands and validates conversation trees.*

### Step 3: Run Phase 2 Ingestion, Language Filtering & Stratification
```bash
python scripts/run_phase2_pipeline.py
```
*Applies language detection, produces `amazonhelp_english.jsonl`, generates partitions, and samples candidate pools.*

### Step 4: Build Dense Retrieval Index
```bash
python scripts/build_retrieval_index.py
```
*Encodes 5,502 Train conversations using SentenceTransformer `all-MiniLM-L6-v2` into normalized embeddings in `data/indexes/`.*

### Step 5: Verify Golden Benchmark Integrity
```bash
python scripts/verify_phase3_retrieval.py
```
*Confirms exact 200 golden checkpoints, 5,502 retrieval documents, and 0% cross-partition leakage.*
