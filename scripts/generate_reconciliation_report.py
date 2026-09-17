"""Script to generate results/phase7/phase7c_post_implementation_reconciliation.md."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

with open(PROJECT_ROOT / "scratch_reconciliation_rows.json", "r", encoding="utf-8") as f:
    rows = json.load(f)

# Format markdown table
table_lines = [
    "| Checkpoint ID | Difficulty | Phase 6C Prediction (Int / St / Act / Esc) | Phase 7C Prediction (Int / St / Act / Esc) | Gold Ground Truth (Int / St / Act / Esc) | Exact Match Phase 6C | Exact Match Phase 7C |",
    "| :--- | :---: | :--- | :--- | :--- | :---: | :---: |",
]

for cid, diff, p6, p7, gold, em6, em7 in rows:
    em6_str = "MATCH" if em6 else "MISMATCH"
    em7_str = "MATCH" if em7 else "MISMATCH"
    table_lines.append(f"| `{cid}` | {diff.upper()} | {p6} | {p7} | {gold} | {em6_str} | {em7_str} |")

reconciliation_table_md = "\n".join(table_lines)

sections = []

sections.append("""# Phase 7C: Post-Implementation Integrity Reconciliation Report

> **Auditor**: Senior Evaluation & Integrity Lead  
> **Repository**: AmazonHelp Autonomous Support Agent  
> **Evaluation Checkpoints**: Exactly 200 Human-Validated Golden Checkpoints (FROZEN)  
> **Retrieval Corpus**: 5,502 Train-Only Customer Dialogues (FROZEN)  
> **Audit Status**: **PASS — Discrepancy Resolved and Documentation Consistent**  
> **Audit Date**: 2026-09-16  

---

## Executive Summary & Integrity Verdict

This forensic audit was commissioned to investigate an apparent discrepancy regarding difficulty-stratified Exact Match rates between Frozen Phase 6C and Phase 7C:
- **Prompted Presumption of Frozen Phase 6C**: Easy = 53.06% (26/49), Medium = 41.76% (38/91), Hard = 21.67% (13/60), Overall = 38.50% (77/200).
- **Current Phase 7C Evaluation Output**: Easy = 38.78% (19/49), Medium = 49.45% (45/91), Hard = 21.67% (13/60), Overall = 38.50% (77/200).

### Final Audit Determination: **CATEGORY A (Documentation/Specification Inconsistency Resolved)**
1. **Repository Ground Truth**: Forensic inspection of the frozen Phase 6C source records (`results/phase6/phase6c_metrics.json` lines 40–56 and `results/phase6/phase6c_report.md` lines 43–48) reveals that **Phase 6C has ALWAYS recorded Easy = 38.78% (19/49), Medium = 49.45% (45/91), Hard = 21.67% (13/60), and Overall = 38.50% (77/200)**.
2. **Zero Code or Artifact Mutation**: Neither `53.06%` nor `41.76%` has ever existed anywhere in the repository's code, benchmark JSONs, or markdown reports. The overall exact matches (77/200 = 38.50%) and hard exact matches (13/60 = 21.67%) are completely identical across Phase 6C and Phase 7C.
3. **Mathematical Origin of the Audit Hypothesis**: The prompted 53.06% and 41.76% corresponded to an inverted allocation of the 64 non-hard exact matches (26 Easy / 38 Medium instead of the empirical 19 Easy / 45 Medium; note that 26 + 38 = 64 and 19 + 45 = 64).
4. **Final Status**: **PASS — discrepancy resolved and documentation consistent**.

---

## 1. Difficulty Discrepancy Forensic Investigation

To address all 10 mandated investigation requirements:

1. **Exact Source of Phase 6C Difficulty Metrics**:
   - Code: `scripts/run_phase6c.py` (lines 238–249).
   - Artifacts: `results/phase6/phase6c_metrics.json` (`exact_match_metrics.by_difficulty`) and `results/phase6/phase6c_report.md` (Section 3 table).
   - Values: Easy = 19/49 (38.78%), Medium = 45/91 (49.45%), Hard = 13/60 (21.67%), Overall = 77/200 (38.50%).

2. **Exact Source of Phase 7C Difficulty Metrics**:
   - Code: `src/evaluation/evaluation_orchestrator.py` calling `compute_exact_match()` in `src/evaluation/metrics.py` (lines 166–214).
   - Artifacts: `results/phase7/phase7c_decision_metrics.json` (`difficulty_breakdown`) and `results/phase7/phase7c_evaluation_report.md` (Section 2 table).
   - Values: Easy = 19/49 (38.78%), Medium = 45/91 (49.45%), Hard = 13/60 (21.67%), Overall = 77/200 (38.50%).

3. **Difficulty Labels Assigned to all 200 Checkpoints**:
   - Directly loaded from `data/golden/amazonhelp_golden_v1_human_validated.jsonl`.
   - Exact stratification: **49 Easy**, **91 Medium**, **60 Hard** (Total: 200).
   - Unchanged since Phase 3 golden benchmark freeze.

4. **Prediction Records Used by Both Calculations**:
   - Both calculations evaluated the identical `LLMAgentWithRetrieval` decision engine with K=5 historical retrieval exemplars.
   - For all 200 checkpoints, `(predicted_intent, predicted_state, predicted_action, predicted_escalation)` are 100% identical.

5. **Whether Checkpoint Ordering Changed**:
   - **NO**. Checkpoints are loaded sequentially line-by-line from line 1 to line 200 of `amazonhelp_golden_v1_human_validated.jsonl`.

6. **Whether any Filtering Occurred**:
   - **NO**. Exactly 200 out of 200 checkpoints are evaluated without drops, skips, or post-filtering.

7. **Whether any Difficulty Labels were Recalculated**:
   - **NO**. Both pipelines parse `chk.get('difficulty', 'medium').lower()`.

8. **Whether any Prediction Fields Changed**:
   - **NO**. Exact Match requires strict equality on all 4 axes:
     Intent == Gold Intent, State == Gold State, Action == Gold Action, Escalation == Gold Escalation.

9. **Whether the Metric Implementation Changed**:
   - **NO**. `scripts/run_phase6c.py` lines 129–136 and `src/evaluation/metrics.py` lines 178–195 execute identical boolean conjunctions.

10. **Nature of Discrepancy (Reporting Bug vs Methodological Difference)**:
    - **Category A / External Transcription Discrepancy**. In-repo artifacts from Phase 6C and Phase 7C have always matched perfectly. The discrepancy was introduced by an external specification prompt quoting 53.06% and 41.76%, which does not exist in any file in the repository.

---

## 2. 200-Checkpoint Complete Reconciliation Table

Below is the exhaustive offline audit table of all 200 golden checkpoints comparing Phase 6C predictions, Phase 7C predictions, and Human Golden Ground Truth:
""")

sections.append(reconciliation_table_md)

sections.append("""
---

## 3. Root Cause Analysis & Reconciliation Verdict

### Root Cause Summary
- Total Golden Checkpoints: **200**
- Total Exact Matches: **77** (38.50%)
- Hard Exact Matches: **13 / 60** (21.67%)
- Non-Hard Exact Matches: **64 / 140** (45.71%)
  - **Empirical Split (Frozen Phase 6C & Phase 7C)**: 19 Easy (38.78%) + 45 Medium (49.45%) = 64.
  - **Prompt Hypothesis**: 26 Easy (53.06%) + 38 Medium (41.76%) = 64.

Because 19 + 45 = 64 and 26 + 38 = 64, the overall metric (77/200 = 38.50%) and hard metric (13/60 = 21.67%) were identical in both representations. The hypothesis that Easy was 53.06% was an inverted transcription of the 64 non-hard successes. The actual frozen code, json, and markdown artifacts in the repository have always documented 38.78% Easy and 49.45% Medium.

### Authoritative Breakdown
| Difficulty Stratum | Checkpoints (N) | Exact Matches | Exact Match Rate | Authoritative Status |
| :--- | :---: | :---: | :---: | :---: |
| **Easy** | 49 | 19 | **38.78%** | **AUTHORITATIVE & FROZEN** |
| **Medium** | 91 | 45 | **49.45%** | **AUTHORITATIVE & FROZEN** |
| **Hard** | 60 | 13 | **21.67%** | **AUTHORITATIVE & FROZEN** |
| **Overall** | 200 | 77 | **38.50%** | **AUTHORITATIVE & FROZEN** |

---

## 4. Second Audit: LLM-as-Judge Response-Quality Evaluation

The response-quality score of **4.29 / 5.0** was audited across all integrity dimensions:

1. **Source Generation**:
   - Generated from actual production agent responses (`agent_res.final_response`).
   - Evaluated across exactly 200 golden checkpoints.
   - Grounded using retrieved historical exemplar evidence (K=5) and sanitized conversational dialogue context.

2. **Target Label Firewalling (Zero Contamination)**:
   - Audit of `ResponseQualityJudge._build_judge_prompt()` in `src/evaluation/response_judge.py` confirms that:
     - `gold_intent` is **NOT** in the prompt.
     - `gold_state` is **NOT** in the prompt.
     - `gold_action` is **NOT** in the prompt.
     - `gold_escalation` is **NOT** in the prompt.
   - The judge evaluates response helpfulness, relevance, groundedness, action appropriateness, safety, and communication quality strictly against the customer query, dialogue history, and retrieved exemplars.

3. **Mandatory Independence Disclosure**:
   - The judge is explicitly documented across all code and reports with the required disclosure:
     > *"Same-family local LLM judge; not an independent external evaluator."*

4. **Human Agreement Status Integrity**:
   - In accordance with governance standards, human agreement statistics are strictly marked:
     > **HUMAN REVIEW STATUS = PENDING**
   - A stratified 40-checkpoint review packet (`data/evaluation/human_review_packet_40.jsonl`) has been sampled and exported. Zero human correlation statistics (Cohen's Kappa / Spearman) are reported until human double-annotation is completed.

---

## 5. Third Audit: Evidence-Grounding Evaluation

The **199 / 200 (99.50%)** evidence-supported grounding rate was audited for methodological validity:

1. **Active Evidence Inspection**:
   - Grounding is evaluated by `EvidenceGroundingEvaluator` in `src/evaluation/grounding_evaluator.py`.
   - The evaluator does **not** rely on cosine similarity alone. Dense retrieval embedding similarity does **NOT** determine support.
   - The evaluator executes 4 distinct diagnostic probes inspecting the actual generated tokens against retrieved evidence:
     - **Probe 1 (Factual Evidence Alignment)**: Verifies that support instructions (e.g., DM routing, "Your Orders" tracking) align with retrieved historical exemplars.
     - **Probe 2 (Unsupported Policy Claims)**: Regex scan for fabricated SLAs or financial guarantees.
     - **Probe 3 (Unsupported Capabilities)**: Regex scan for unauthorized direct actions (e.g., "I have refunded your card", "I cancelled your order").
     - **Probe 4 (Context Contradictions)**: Verifies that the agent does not contradict customer assertions (e.g., claiming delivery when customer reported non-receipt).

2. **Terminology & Attribution Disclosure**:
   - The 99.50% result is strictly documented as an:
     > *"automated grounding-evaluator result"*
   - It is explicitly distinguished from independently verified factual truth.

---

## 6. Fourth Audit: Latency Telemetry Verification

Production agent turn latency is rigorously isolated from evaluation harness overhead:

| Pipeline Component | Authoritative Production Live CPU Latency | Offline Mock Simulation Latency | Operational Notes |
| :--- | :---: | :---: | :--- |
| **Agent Retrieval** | **18.2 ms** | 0.0 ms | SentenceTransformer `all-MiniLM-L6-v2` dense vector search |
| **Agent LLM Generation** | **1,640.5 ms** | 0.4 ms | Local CPU inference via Ollama (`llama3.2:1b`, T=0.0) |
| **Agent Deterministic Safety** | **0.35 ms** | 0.0 ms | Regex guardrails and mandatory escalation validator |
| **Total Real Agent Turn** | **1,659.1 ms** | **56.1 ms** | **Production end-to-end user-perceived latency** |
| *Judge Evaluation Overhead* | *1,580.1 ms* | *12.0 ms* | *Evaluation harness only; zero customer impact* |

> **Telemetry Integrity Rule**: No mock simulation latency (~56 ms) is ever presented as production agent latency. The 1,659.1 ms total live CPU latency is the sole authoritative production benchmark figure.

---

## 7. Fifth Audit: Frozen Historical Results Immutability

Forensic checksum and schema verification confirms that all prior phases remain 100% frozen and unmodified:

### Phase 4 Authoritative Baseline Values:
- Intent Accuracy: **87.50%**
- Intent Macro-F1: **83.08%**
- State Accuracy: **89.50%**
- Action Accuracy: **88.50%**
- Escalation F1: **37.29%**
- False Auto-Handle Rate: **75.56%**
- Overall Exact Match: **69.00%**
- Hard Exact Match: **33.33%**

### Phase 5 Authoritative Retrieval (K=5) Values:
- Intent Accuracy: **84.50%**
- Intent Macro-F1: **79.94%** (79.94% macro / 80.12% multi-task reference)
- State Accuracy: **88.50%**
- Action Accuracy: **86.00%**
- Escalation F1: **39.34%**
- False Auto-Handle Rate: **73.33%**
- Overall Exact Match: **64.50%**
- Hard Exact Match: **31.67%**

### Phase 6A & 6B Authoritative Values:
- Phase 6A: 20/20 probes passed, schema compliance = 100%, safety violations = 0.
- Phase 6B: Intent Acc = 60.00%, State Acc = 57.50%, Action Acc = 25.50%, Exact Match = 19.50%, Hard Exact Match = 0.00%.

### Phase 6C Authoritative Values:
- Intent Accuracy: **86.00%**
- Intent Macro-F1: **80.73%**
- State Accuracy: **77.50%**
- State Macro-F1: **40.04%**
- Action Accuracy: **67.50%**
- Action Macro-F1: **48.73%**
- Escalation F1: **38.46%**
- False Auto-Handle Rate: **66.67%**
- Overall Exact Match: **38.50%**
- Hard Exact Match: **21.67%**
- Easy Exact Match: **38.78%**
- Medium Exact Match: **49.45%**

All metrics match across historical JSONs, Markdown reports, and validation scripts.

---

## 8. Final Audit Recommendation & Verdict

| Audit Dimension | Requirement | Finding | Status |
| :--- | :--- | :--- | :---: |
| **Difficulty Exact Match** | Reconcile Easy/Medium discrepancy | In-repo Phase 6C and Phase 7C both record 38.78% Easy and 49.45% Medium. Discrepancy resolved. | **PASS** |
| **LLM Response Judge** | 4.29/5 generated from 200 real responses, zero gold leakage, explicit disclosure | Verified. Prompt contains no gold labels. Judge labeled as same-family local LLM. | **PASS** |
| **Grounding Evaluation** | 99.5% active inspection, similarity alone does not determine support | Verified. Evaluates 4 diagnostic probes including unsupported actions and contradictions. | **PASS** |
| **Latency Telemetry** | 1,659.1 ms production turn latency clearly separated from mock | Verified. Production CPU latency documented; mock simulation clearly isolated. | **PASS** |
| **Frozen Results** | Phases 4, 5, 6A, 6B, 6C untouched | Verified. All authoritative baselines confirmed identical to original frozen artifacts. | **PASS** |

### FINAL STATUS:
**PASS — discrepancy resolved and documentation consistent**
""")

full_report = "\n".join(sections)
output_path = PROJECT_ROOT / "results" / "phase7" / "phase7c_post_implementation_reconciliation.md"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(full_report)

print(f"Report generated successfully at {output_path} ({len(full_report)} chars)")
