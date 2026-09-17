# Phase 5 Retrieval Augmentation Protocol & Experimental Methodology

> **Formal Specification: Historical Support Conversation Retrieval Augmentation**  
> *AmazonHelp Autonomous Support Agent Benchmark*

---

## 1. Executive Overview & Central Research Question

Phase 5 investigates whether augmenting autonomous decision-making with historical customer support conversations improves accuracy, safety, and multi-turn state tracking compared to the Phase 4 non-LLM baseline.

### Central Research Question:
> *Can historical AmazonHelp support conversations improve intent, escalation, state, and action decisions—especially difficult multi-turn cases—compared with the Phase 4 non-LLM baseline?*

### Scope Boundaries:
- **No Generative LLMs**: Generative response generation belongs strictly to Phase 6.
- **No Fine-Tuning**: No model weights are fine-tuned or modified.
- **Explainable & Deterministic Decision Layer**: Decisions integrate similarity-weighted historical evidence with audited Phase 4 deterministic rule and ML guardrails.

---

## 2. Frozen Dataset Isolation & Anti-Leakage Rules

Strict chronological partition boundaries and data isolation protocols are rigorously enforced:

| Dataset / Artifact | Size | Partition | Role in Phase 5 | Leakage Constraint |
| :--- | :---: | :---: | :--- | :--- |
| **Pristine Conversations** | 53,637 | Train/Val/Test | Master chronological source | Sorted by timestamp; partition boundaries verified. |
| **Retrieval Corpus** | 5,502 dialogues | **Train ONLY** | Historical candidate pool for similarity search | Zero Dev or Test conversations allowed. |
| **Dense Vector Index** | 5,502 × 384 | **Train ONLY** | Precomputed L2-normalized embeddings | Generated exclusively from Train retrieval dialogues. |
| **Golden Benchmark** | 200 checkpoints | **Validation ONLY** | Rule-based pre-annotated evaluation target | Zero Test conversations; strictly evaluation-only. |
| **Test Partition** | 5,365 convs | **Test ONLY** | Completely frozen | **100% untouched** until final project evaluation. |

> [!IMPORTANT]
> **Anti-Leakage Invariant**:
> 1. Golden expected labels (`expected_intent`, `expected_state`, `expected_action`, `expected_escalation`) are **strictly evaluation targets** and are never exposed as runtime features.
> 2. Every retrieved historical document traces directly to a verified Train conversation ID.
> 3. Zero hyperparameter or threshold tuning was performed using Test data.

---

## 3. Retrieval Architecture

The Phase 5 retrieval pipeline executes deterministically for each incoming conversation turn:

```
Incoming Customer Turn + Conversation History
                     ↓
Deterministic Query Builder (Handle stripping, entity extraction, context windowing)
                     ↓
Dense Vector Embedding (SentenceTransformer all-MiniLM-L6-v2, 384d, L2 normalized)
                     ↓
Cosine Similarity Search over Train Index (top-k: 1, 3, 5, 10)
                     ↓
Historical Evidence Extraction (Intent, Escalation, State, Action, Resolution)
                     ↓
Similarity-Weighted Evidence Aggregation (Confidence tiers: HIGH, MEDIUM, LOW, NONE)
                     ↓
Retrieval-Augmented Decision Engine (Safety overrides > Escalation > State > Action > Intent)
                     ↓
Structured Audit Record & Evaluation Metrics
```

---

## 4. Query Construction Strategy

To maximize semantic retrieval quality while avoiding noise from long transcript concatenation, `src/retrieval/query_builder.py` employs a deterministic query strategy:

1. **Noise Cleaning**: Twitter handles (`@AmazonHelp`, `@user`), URLs, and excess whitespace are sanitized.
2. **Entity Signal Preservation**: Tracking numbers, order IDs (e.g., `123-4567890-1234567`), postal codes, and carrier names (`Royal Mail`, `Hermes`, `USPS`, `DPD`) are preserved.
3. **Multi-Turn Context Structuring**:
   - **Initial Turn (`turn_depth == 1`)**: Formatted as `Customer: <cleaned_message>`.
   - **Multi-Turn (`turn_depth > 1`)**: Formatted as `Support: <immediate_prior_support_prompt> | Customer: <cleaned_message>`.
   - Unlimited dialogue concatenation is strictly prohibited to prevent query drift.

---

## 5. Structured Historical Evidence Extraction

For each retrieved dialogue $d_i \in \text{top-}k$ with cosine similarity $s_i$, the following attributes are extracted:
- `historical_intent`: Intent category derived from the resolved historical issue.
- `historical_state`: Dialogue state at the matching point.
- `historical_action`: Support action taken by the AmazonHelp agent.
- `historical_escalation`: Whether the historical dialogue required supervisor/specialist escalation.
- `outcome_evidence_type`: Quality rating of the resolution (`explicit_confirmation`, `actionable_next_step`, `escalated_transfer`).

### Similarity-Weighted Aggregation:
Instead of naive majority voting, historical evidence is weighted exponentially by similarity:
$$w_i = \max(0.0, s_i)^2$$

Normalized weights:
$$\tilde{w}_i = \frac{w_i}{\sum_{j=1}^k w_j}$$

### Retrieval Confidence Tiers:
- **`HIGH`**: $\max(s_i) \ge 0.70$ and $\text{mean}(s_i) \ge 0.55$. Strong historical precedent.
- **`MEDIUM`**: $\max(s_i) \ge 0.55$ and $\text{mean}(s_i) \ge 0.45$. Moderate evidence.
- **`LOW`**: $\max(s_i) \ge 0.40$. Weak evidence; used only as fallback corroboration.
- **`NONE`**: $\max(s_i) < 0.40$. No historical precedent; system defers 100% to Phase 4 baseline.

---

## 6. Retrieval-Enhanced Decision Policies

### 6.1 Hybrid Intent Classification
A gated probability interpolation merges ML model predictions with retrieval evidence:
$$P_{\text{hybrid}}(c) = \alpha \cdot P_{\text{ML}}(c) + (1 - \alpha) \cdot P_{\text{retrieval}}(c)$$

- **Dynamic Gating ($\alpha$)**:
  - `HIGH` confidence: $\alpha = 0.60$ (40% weight to historical precedent).
  - `MEDIUM` confidence: $\alpha = 0.80$ (20% weight to historical precedent).
  - `LOW` / `NONE`: $\alpha = 1.00$ (100% Phase 4 ML prediction).
- **Disambiguation Rule (Prime vs. Delivery)**: If customer message mentions "Prime" but describes delivery delays ("late", "delayed", "has not arrived"), historical retrieval disambiguates towards `DELIVERY_STATUS_AND_TRACKING`.

### 6.2 Retrieval-Augmented Escalation Policy (Primary Research Target)
Phase 4 identified a severe escalation recall deficit (24.44% recall, 75.56% False Auto-Handle Rate). The Phase 5 policy augments deterministic regex rules:
- **Baseline Rule Trigger**: If explicit regex rules detect legal threats, abusive conduct, or repeated failures, escalation fires immediately.
- **Retrieval-Augmented Trigger**: If baseline does not fire, but retrieval confidence is `HIGH` or `MEDIUM` and the weighted historical escalation rate satisfies:
  $$\sum_{i=1}^k \tilde{w}_i \cdot \mathbb{I}(\text{escalation}_i) \ge 0.40$$
  escalation is triggered with reason `RETRIEVAL_AUGMENTED_PRECEDENT`.
- **Low-Confidence Suppression**: If retrieval confidence is `LOW` or `NONE`, escalation defaults strictly to baseline rules to prevent false alarms.

### 6.3 State & Action Assistance
- **State Tracking**: High-confidence historical trajectories assist state classification on ambiguous confirmation turns (e.g., distinguishing `STATE_CUSTOMER_PROVIDING_INFO` from `STATE_APPARENTLY_RESOLVED`).
- **Action Selection**: Suggests high-confidence proven actions (`PROVIDE_INFORMATION`, `PROVIDE_TROUBLESHOOTING`, `REQUEST_SAFE_DETAILS`) when the baseline is uncertain.

---

## 7. Mandatory Safety Precedence Order

Safety guardrails strictly override all retrieval evidence and baseline predictions:

1. **Tier 1: Absolute Security Constraint (Highest Priority)**
   - Hard block against any request for passwords, OTPs, PINs, CVVs, or full credit card numbers.
   - Immediate handoff to secure DM channel (`HANDOFF_TO_SECURE_CHANNEL`) on account compromise, unauthorized charges, or security vulnerabilities.
2. **Tier 2: Explicit Escalation Conditions**
   - Legal threats, abusive conduct, severe safety hazards.
3. **Tier 3: Strong Current Turn Evidence**
   - Direct customer keywords and explicit operational state.
4. **Tier 4: High-Confidence Historical Retrieval Evidence**
   - Applied only if Tiers 1–3 are satisfied.
5. **Tier 5: Model & Rule Baseline Fallback**
   - Standard Phase 4 logic.
6. **Tier 6: Default Safe Fallback**
   - Ask clarifying question.

---

## 8. Evaluation Harness & Statistical Significance

- **Benchmark Dataset**: `data/golden/amazonhelp_golden_v1_human_validated.jsonl` (200 checkpoints, rule-based pre-annotated benchmark).
- **K-Sweep Configurations**: $K \in [1, 3, 5, 10]$.
- **Ablation Matrix**:
  1. Phase 4 Baseline
  2. Intent-Only Retrieval
  3. Escalation-Only Retrieval
  4. Full Retrieval (K=5)
- **Bootstrap Confidence Intervals**: 1,000 resamples with fixed seed (`42`) at 95% confidence level for key metric deltas:
  - Escalation Recall Delta ($\Delta$)
  - False Auto-Handle Rate Delta ($\Delta$)
  - Hard-Case Exact Match Delta ($\Delta$)
  - Intent Macro-F1 Delta ($\Delta$)
