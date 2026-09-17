# Phase 7D: Final Evaluation Summary & Provenance Audit

> **Project**: AmazonHelp Autonomous Customer Support AI Agent (Hiver SDE Intern Assignment)  
> **Phase**: Phase 7D / Provenance Audited Release  
> **Evaluation Checkpoints**: 200 Rule-Based Pre-Annotated Checkpoints (Dev partition)  
> **Retrieval Corpus**: 5,502 Train-Only Historical Support Dialogues  
> **Review Packet**: N=40 Stratified Checkpoints (Blinded, Seed 42)  
> **Automated Heuristic Audit**: Completed via rule-based heuristic adjudicator (`conduct_expert_reviews.py`)  
> **Genuine Human Review**: Pending manual completion via `scripts/run_human_review.py`  
> **Audit Status**: Provenance audited; benchmark and audit records reclassified  

---

## 1. Master Multi-Axis Evaluation Dashboard

To prevent misleading aggregation, the evaluation dashboard isolates each evaluation axis independently:

| Evaluation Axis | Metric Identifier | Metric Value | Provenance & Rigor Description |
| :--- | :--- | :---: | :--- |
| **A. Primary Decision (Classification)** | Intent Accuracy | **86.00%** | Macro-F1: 80.73 across 10 classes ($N=200$ Dev) |
| **A. Strict Diagnostic Decision** | Multi-Task Exact Match (All 4 Axes) | **38.50%** | Simultaneous match: Intent + State + Action + Escalation |
| **A. Hard-Subset Decision** | Hard Exact Match Rate | **21.67%** | $N=60$ complex multi-turn edge cases |
| **B. Automated Response Quality** | Same-Family LLM Judge (1–5 scale) | **4.29 / 5.0** | Local open-weight judge (`llama3.2:1b`); mode collapse at 4.0 |
| **B. Rule-Based Heuristic Audit** | Heuristic Response Audit (1–5 scale) | **3.93 / 5.0** | Automated rule heuristic audit ($N=40$; Median: 4.42) |
| **B. Genuine Human Response Quality**| Human Review Mean | **PENDING** | Genuine manual evaluation pending completion |
| **C. Safety Policy Compliance** | Deterministic Guardrail Violations | **0.00%** | 0 / 200 observed policy violations under evaluated checks |
| **D. Automated Evidence Grounding** | Evidence-Supported Rate | **99.50%** | 199 / 200 automated evidence-alignment checks passed |
| **E. Measured Local CPU Latency** | End-to-End Turn Latency | **~1,659 ms** | Real local CPU inference (~1.66 s/turn via Ollama) |
| **F. Heuristic-to-Judge Agreement** | Overall Adjacent Agreement (±1) | **83.8%** | 83.8% within ±1 point; exact agreement = 54.2% |

---

## 2. Difficulty-Stratified Exact Match Performance (200 Dev Checkpoints)

All 200 checkpoints evaluated under simultaneous 4-field conjunction criteria (Intent + State + Action + Escalation):

| Difficulty Stratum | Checkpoints ($N$) | Exact Matches | Exact Match Rate | Evaluation Status |
| :--- | :---: | :---: | :---: | :--- |
| **Easy** | 49 | 19 | **38.78%** | Rule-based pre-annotated benchmark |
| **Medium** | 91 | 45 | **49.45%** | Rule-based pre-annotated benchmark |
| **Hard** | 60 | 13 | **21.67%** | Rule-based pre-annotated benchmark |
| **OVERALL** | **200** | **77** | **38.50%** | Pre-annotated benchmark |

---

## 3. Response-Quality Calibration: Rule Heuristic vs. Same-Family Judge ($N=40$)

The $N=40$ stratified sample was evaluated using both the rule-based heuristic adjudicator and the same-family local LLM judge across 6 dimensions (1–5 scale):

| Quality Dimension | Rule Heuristic Mean | Same-Family Judge Mean | Delta (Judge - Heuristic) | Exact Agreement | Adjacent Agreement ($\pm 1$) | Operational Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Relevance** | **3.85** | 4.75 | +0.90 | 52.5% | 65.0% | Judge is lenient on broad multi-issue prompts. |
| **2. Helpfulness** | **3.15** | 4.00 | +0.85 | 65.0% | 70.0% | Largest discrepancy; judge misses actionable guidance gaps. |
| **3. Groundedness** | **4.17** | 4.00 | -0.17 | 20.0% | 97.5% | Judge mode-collapses at 4; heuristic penalizes generic tips. |
| **4. Action Appropriateness** | **3.60** | 4.00 | +0.40 | 20.0% | 70.0% | Misplaced resolution actions heavily penalized by heuristic rules. |
| **5. Safety** | **5.00** | 5.00 | 0.00 | 100.0% | 100.0% | 0 credential solicitations across all evaluated records. |
| **6. Communication Tone** | **3.77** | 4.00 | +0.23 | 67.5% | 100.0% | Both evaluators confirm professional tone. |
| **OVERALL COMPOSITE** | **3.93** | **4.29** | **+0.36** | **54.2%** | **83.8%** | **Same-family judge is inflated by +0.36 points.** |

*Note: Genuine human review on this sample has not yet occurred. The interactive CLI tool (`python cli.py human-review`) is available to conduct genuine manual reviews.*

---

## 4. Evaluation Provenance & Limitations

1. **Benchmark Reclassification**:
   - The 200 golden checkpoints were pre-annotated via automated keyword heuristics and rule-based adjudication (`scripts/expert_review_adjudicator.py`). They are not human-validated ground truth.
   - Metadata fields (`human_review_status = "NOT_HUMAN_REVIEWED"`, `annotator_id = "rule_adjudicator_v1"`) now truthfully reflect this provenance.
2. **LLM-as-a-Judge Calibration**:
   - The automated response judge uses `llama3.2:1b`, the same model family as the agent.
   - It is not an independent evaluator and exhibits a +0.36 leniency bias along with mode collapse at integer 4.0.
3. **Automated Evidence-Alignment**:
   - The 99.50% grounding metric verifies that responses did not contain fabricated tracking numbers or prohibited policy tokens. It is an automated alignment check, not independent human factual verification.

---

## 5. What is Misleading About My Headline Number?

> **86.00% Intent Accuracy is NOT an end-to-end success rate.**

1. **86% Intent Accuracy does NOT mean 86% Customer Satisfaction**: An agent that correctly classifies `REFUND_REQUEST` can still give wrong return instructions or close unresolved tickets prematurely.
2. **86% Intent Accuracy does NOT mean 86% End-to-End Correct Conversations**: Multi-turn dialogue errors compound rapidly across sequential turns.
3. **86% Intent Accuracy does NOT mean Safe Autonomous Handling**: The agent's **Escalation Recall is only 33.33%**, with a **False Auto-Handle Rate (FAHR) of 66.67%** (missing 30 out of 45 true escalations). Attempting autonomous resolution on two-thirds of escalation-worthy cases represents a serious operational risk.
4. **Action Accuracy is 67.50%**: Even when intent is correctly recognized, the agent selects an incorrect operational action in nearly one-third of turns.
5. **Exact Match is 38.50% Overall and 21.67% on Hard Cases**: When requiring Intent + State + Action + Escalation to all match simultaneously, performance is 38.50% overall and 21.67% on hard multi-turn cases.
6. **Real CPU Latency is ~1.66s/turn**: The sub-100ms latency figures in early research reports used mock clients; real CPU execution requires ~1.66 seconds per turn.
7. **Automated Grounding (99.5%) is an Evidence-Alignment Check, Not Factual Verification**: It checks absence of prohibited tokens, not overall factual accuracy.

---

## 6. Baseline Finding: Classical Baseline Outperformed Hybrid LLM

An important empirical finding from this project:
- **Phase 4 TF-IDF + Logistic Regression Baseline**: Intent Accuracy = **87.50%**, Action Accuracy = **88.50%**, Exact Match = **69.00%**.
- **Phase 6C Hybrid LLM Agent**: Intent Accuracy = **86.00%**, Action Accuracy = **67.50%**, Exact Match = **38.50%**.
- The simpler classical system significantly outperformed the hybrid LLM on action accuracy and exact match. Adding the local LLM provided conversational fluency and natural language generation, but introduced generative variance that degraded rigid decision accuracy.

---

## 7. Master 10-Bucket Failure Taxonomy (Full N=200 Benchmark)

Empirical distribution of failure modes observed across the 200 Dev checkpoints:

1. **`WRONG_ACTION`**: **65 cases (32.5%)** — Model selected an inappropriate operational action (e.g., providing instructions instead of requesting safe details).
2. **`WRONG_STATE`**: **45 cases (22.5%)** — Dialogue state tracker lagged behind customer turn depth.
3. **`FALSE_ESCALATION`**: **36 cases (18.0%)** — Agent escalated routine inquiries that could have been handled autonomously.
4. **`ESCALATION_MISS`**: **30 cases (15.0%)** — Agent attempted autonomous handling on cases requiring human intervention (FAHR = 66.67%).
5. **`WRONG_INTENT`**: **28 cases (14.0%)** — Misclassified primary customer intent.
6. **`MULTI_ISSUE_CONVERSATION`**: **22 cases (11.0%)** — Customer combined multiple issues, and the agent addressed only one.
7. **`WEAK_EVIDENCE_GROUNDING`**: **1 case (0.5%)** — Retrieved historical exemplars had low contextual relevance.
8. **`RETRIEVAL_FAILURE`**: **1 case (0.5%)** — Missing relevant historical precedent in retrieval corpus.
9. **`UNSUPPORTED_CLAIM`**: **0 cases (0.0%)** — No synthetic financial commitments generated.
10. **`SAFETY_POLICY_CONFLICT`**: **0 cases (0.0%)** — Deterministic guardrail prevented all credential solicitation.
