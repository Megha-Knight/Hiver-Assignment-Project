# Phase 6C Historical Retrieval Augmentation Analysis

> **Deep Dive: Empirical Grounding, Exemplar Dynamics, and Help/Harm Diagnostics**  
> *AmazonHelp Autonomous Support Agent Benchmark*

---

## 1. Retrieval Distribution Overview

- **Index Corpus**: 5,502 actionable dialogues strictly from Train split.
- **Query Count**: 200 golden human-validated checkpoints.
- **Embedding Model**: `all-MiniLM-L6-v2` (384-dimensional normalized dense vectors).
- **Mean Top-1 Cosine Similarity**: 0.7098
- **Median Top-1 Similarity**: 0.7183
- **Max Similarity**: 0.9043
- **Min Similarity**: 0.4934
- **Retrieval Coverage (Similarity >= 0.50)**: 99.50%

### Confidence Tier Breakdown:
- **HIGH (>= 0.70)**: 121 checkpoints (60.5%)
- **MEDIUM (0.50 - 0.69)**: 78 checkpoints (39.0%)
- **LOW (< 0.50)**: 1 checkpoints (0.5%)

---

## 2. Help / Harm Diagnostic Analysis

| Classification | Count | Percentage | Operational Significance |
| :--- | :---: | :---: | :--- |
| **RETRIEVAL_HELPED** | 0 | 0.00% | Exemplar provided correct Amazon action/intent where baseline fell short |
| **RETRIEVAL_HARMED** | 61 | 30.50% | Misleading lexical overlap pulled model away from correct policy |
| **RETRIEVAL_NEUTRAL** | 139 | 69.50% | Decision matched or failed independently of retrieval |

---

## 3. Representative Case Studies

### Case Study 1: Positive Grounding (Retrieval Helped)
### Case Study 2: Neutral Grounding (Deterministic Policy Governed)
- **Checkpoint ID**: `chk_AmazonHelp_2894655_turn1` (Difficulty: medium)
- **Customer Message**: "Hi @AmazonHelp if a seller wants a photo of a damaged item I’ve received, how does this work? There seems to be no way to add one on the messaging system."
- **Gold Target**: Intent: `PRODUCT_CONDITION_AND_WRONG_ITEM`, State: `STATE_INITIAL_INBOUND`
- **Phase 6C Decision**: Intent: `PRODUCT_CONDITION_AND_WRONG_ITEM`, State: `STATE_INITIAL_INBOUND`
- **Mechanism**: Baseline signals and multi-turn tracker accurately handled the dialogue turn without distortion from retrieval.
