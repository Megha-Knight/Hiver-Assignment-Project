# Phase 7A — Frozen Experimental Metric Consistency Audit

**Audit Date:** 2026-09-16  
**Auditor:** Automated Take-Home Evaluation Hardening Agent  
**Status:** AUDIT COMPLETE — DISCREPANCIES CATALOGED FOR RECTIFICATION

---

## 1. Executive Summary

This audit performs an exhaustive cross-file inspection of all quantitative metrics across Phase 4, Phase 5, Phase 6B, and Phase 6C. 

Per Phase 7A instructions, experimental source files and baseline metrics are strictly frozen; contradictions are transparently surfaced, analyzed, and cataloged with specific proposed corrections for subsequent submission hardening.

---

## 2. Authoritative Frozen Benchmark Reference Values

The following benchmark metrics are authoritative and frozen:

### Phase 4 Deterministic Baseline (200 Golden Checkpoints)
* **Intent Accuracy:** 87.50%
* **Intent Macro-F1:** 83.08%
* **State Accuracy:** 89.50%
* **State Macro-F1:** 48.39%
* **Action Accuracy:** 88.50%
* **Action Macro-F1:** 80.17%
* **Escalation Precision:** 78.57%
* **Escalation Recall:** 24.44%
* **Escalation F1:** 37.29%
* **False Auto-Handle Rate (FAHR):** 75.56%
* **Overall Exact Match:** 69.00%
* **Hard Exact Match:** 33.33%

### Phase 5 Retrieval-Only Augmentation (Primary K=5)
*(Source of Truth: `results/phase5_retrieval_metrics.json` & `results/phase5_retrieval_report.md`)*
* **Intent Accuracy:** 84.50%
* **Intent Macro-F1:** 79.94% (rounded from 79.93%)
* **State Accuracy:** 88.50%
* **State Macro-F1:** 47.97%
* **Action Accuracy:** 86.00%
* **Action Macro-F1:** 77.39%
* **Escalation Precision:** 73.33%
* **Escalation Recall:** 24.44%
* **Escalation F1:** 36.67%
* **False Auto-Handle Rate (FAHR):** 75.56%
* **Overall Exact Match:** 64.50%
* **Hard Exact Match:** 31.67%

### Phase 6B LLM-Only Controlled Agent (Zero Retrieval, llama3.2:1b)
*(Source of Truth: `results/phase6/phase6b_metrics.json`)*
* **Intent Accuracy:** 60.00%
* **Intent Macro-F1:** 42.34%
* **State Accuracy:** 57.50%
* **State Macro-F1:** 19.99%
* **Action Accuracy:** 25.50%
* **Action Macro-F1:** 9.27%
* **Escalation Precision:** 66.67%
* **Escalation Recall:** 13.33%
* **Escalation F1:** 22.22%
* **False Auto-Handle Rate:** 86.67%
* **Overall Exact Match:** 19.50%
* **Hard Exact Match:** 0.00%
* **Unsupported Action Rate:** 0.00%
* **Safety Violations:** 0

### Phase 6C Tri-Layer Architecture (LLM + Retrieval + Policy, llama3.2:1b, K=5)
*(Source of Truth: `results/phase6/phase6c_metrics.json`)*
* **Intent Accuracy:** 86.00%
* **Intent Macro-F1:** 80.73%
* **State Accuracy:** 77.50%
* **State Macro-F1:** 40.04%
* **Action Accuracy:** 67.50%
* **Action Macro-F1:** 48.73%
* **Escalation Precision:** 45.45%
* **Escalation Recall:** 33.33%
* **Escalation F1:** 38.46%
* **False Auto-Handle Rate:** 66.67%
* **Overall Exact Match:** 38.50%
* **Hard Exact Match:** 21.67%
* **Unsupported Action Rate:** 0.00%
* **Safety Violations:** 0
* **Retrieval Utility Breakdown:** Helped: 21/200, Harmed: 12/200, Neutral: 167/200

---

## 3. Detailed Cross-Report Discrepancy Registry

### Item 1: Phase 5 Reference Values in `scripts/compare_phase4_phase6.py`
* **File:** `scripts/compare_phase4_phase6.py` (lines 50–64)
* **Observed Hardcoded Value:**
  ```python
  p5 = {
      "intent_acc": 0.8450,
      "intent_f1": 0.8012,
      "state_acc": 0.8950,
      "state_f1": 0.4839,
      "action_acc": 0.8700,
      "action_f1": 0.7840,
      "esc_prec": 0.7500,
      "esc_rec": 0.2667,
      "esc_f1": 0.3934,
      "fahr": 0.7333,
      "exact_match": 0.6650,
      "hard_exact": 0.3000,
  }
  ```
* **Expected Frozen Value (from `phase5_retrieval_metrics.json`):**
  * `intent_f1`: **0.7994** (79.93%)
  * `state_acc`: **0.8850**
  * `state_f1`: **0.4797**
  * `action_acc`: **0.8600**
  * `action_f1`: **0.7739**
  * `esc_prec`: **0.7333**
  * `esc_rec`: **0.2444**
  * `esc_f1`: **0.3667**
  * `fahr`: **0.7556**
  * `exact_match`: **0.6450**
  * `hard_exact`: **0.3167**
* **Severity:** Medium (Reporting Discrepancy in comparison script)
* **Root Cause:** A preliminary experimental draft of Phase 5 K=5 was hardcoded into the comparative script rather than parsing directly from `results/phase5_retrieval_metrics.json`.
* **Proposed Correction for Phase 7B:** Update `scripts/compare_phase4_phase6.py` to dynamically load `PATHS.PHASE5_RETRIEVAL_METRICS_JSON["k_sweep"]["5"]` ensuring 100% single-source-of-truth alignment.

---

### Item 2: Phase 5 Column in `results/phase6/phase6c_report.md` Comparison Table
* **File:** `results/phase6/phase6c_report.md` (Comparative Matrix Table)
* **Observed Value:** Displays the preliminary Phase 5 values from `compare_phase4_phase6.py` (e.g., State Acc 89.50%, Action Acc 87.00%, Exact Match 66.50%).
* **Expected Frozen Value:** State Acc 88.50%, Action Acc 86.00%, Exact Match 64.50%.
* **Severity:** Medium (Documentary Inconsistency)
* **Root Cause:** Table was populated directly from the console output of `compare_phase4_phase6.py`.
* **Proposed Correction for Phase 7B:** Re-align the Phase 5 column in the final summary report with the authoritative frozen values from `results/phase5_retrieval_report.md`.

---

### Item 3: Phase 6C End-to-End Latency Claim
* **Files:** `results/phase6/phase6c_report.md`, `results/phase6/phase6c_metrics.json`
* **Reported Value:** Mean: 73.82 ms, Median: 71.60 ms, P95: 98.40 ms
* **Evaluation Context:** 
  * Retrieval latency: 22.45 ms (MiniLM on CPU)
  * Evidence building & prompt construction: ~2.1 ms
  * LLM generation latency: ~48.5 ms (measured via `MockOllamaClient` deterministic offline simulation)
* **Audit Finding:** Real end-to-end CPU generation with `llama3.2:1b` running locally on an Intel/AMD CPU requires approximately **1,500 ms to 3,500 ms** per dialogue turn. The 73.82 ms figure accurately measures the *pipeline framework latency + dense retrieval latency under mock LLM execution*, but does **not** represent live Ollama token generation time.
* **Severity:** High (Scientific Honesty & Evaluation Rigor)
* **Audit Determination:** **CLAIM REQUIRES CORRECTION**
* **Proposed Correction for Phase 7B / README:** Explicitly document the distinction between:
  1. *Harness framework latency* (retrieval + deterministic policy + evidence builder = ~25–30 ms)
  2. *Live local CPU LLM generation latency* (~1.5–3.5 s per turn with `llama3.2:1b`).
  Never present the mock simulation latency as live neural token generation time.

---

## 4. Verification Check Status

The automated verification suite (`scripts/verify_phase7a.py`) incorporates a check that validates all authoritative frozen metrics and verifies that discrepancies are transparently flagged rather than hidden.
