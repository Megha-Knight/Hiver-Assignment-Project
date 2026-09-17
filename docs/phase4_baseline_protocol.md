# Phase 4 Baseline Protocol & Experimental Methodology

> **Formal Specification: Non-LLM Intent & Escalation Baselines**  
> *AmazonHelp Autonomous Support Agent Benchmark*

---

## 1. Overview & Purpose

Phase 4 establishes empirical, reproducible non-LLM baselines for:
1. **Intent Classification**: Trivial Majority Baseline vs. TF-IDF + Logistic Regression.
2. **Escalation Detection**: Conservative Always-Escalate vs. Never-Escalate Baselines.

These non-LLM baselines serve as the scientific anchor for subsequent evaluation of retrieval-augmented prompting (Phase 5) and local open-weight language models (Phase 6).

---

## 2. Dataset Isolation & Partition Boundaries

Strict chronological partition boundaries are maintained:

| Partition | Conversations | Chronological Window (UTC) | Role in Phase 4 |
| :--- | :---: | :---: | :--- |
| **Train** | 42,909 | 2015-12-23 to 2017-11-24 | Sole training corpus for TF-IDF vectorizer and Logistic Regression. |
| **Validation (Dev)** | 5,363 | 2017-11-24 to 2017-11-29 | Source of the 200 pre-annotated evaluation checkpoints. |
| **Test** | 5,365 | 2017-11-29 to 2017-12-03 | **100% FROZEN & UNTOUCHED**. Reserved for final benchmark. |

> [!IMPORTANT]
> - Zero Test conversation IDs were exposed during vocabulary extraction, TF-IDF fitting, or hyperparameter selection.
> - The 200 rule-based pre-annotated evaluation checkpoints originate strictly from Validation and were never included in the training matrix.

---

## 3. Trivial Baseline Specification

### 3.1 Majority Intent Baseline
- **Definition**: Predicts the single most frequent intent observed in training data for every incoming message.
- **Operational Majority Class**: `DELIVERY_STATUS_AND_TRACKING` (representing 24.5% of the pre-annotated benchmark and the primary specific operational complaint volume).
- **Behavior**: Macro-F1 is severely penalized due to 0.0 recall on the remaining 9 intent classes.

### 3.2 Conservative Escalation Baseline
- **Never Escalate Mode (`default_escalate=False`)**: Assumes all inquiries can be auto-handled. Maximizes autonomous volume, but exposes 100% of genuine escalations to safety failure.
- **Always Escalate Mode (`default_escalate=True`)**: Ultra-conservative policy escalating every single turn to human agents. Guarantees 0% safety failure, but provides 0% autonomy.

---

## 4. Simple ML Baseline: TF-IDF + Logistic Regression

### 4.1 Input Representation
To capture dialogue progression without leaking future turns, the input representation incorporates immediate prior support context:
- **Turn 1 (Inbound)**: `"Customer: <customer_message>"`
- **Turn > 1 (Multi-Turn)**: `"Support Context: <last_support_message> | Customer: <customer_message>"`

### 4.2 Vectorization & Hyperparameters
- **Vectorizer**: `TfidfVectorizer`
  - `ngram_range`: `(1, 2)` (unigrams and bigrams)
  - `max_features`: `15,000`
  - `sublinear_tf`: `True`
  - `stop_words`: `english`
  - `token_pattern`: `(?u)\b\w+\b`
- **Classifier**: `LogisticRegression`
  - `C`: `1.0`
  - `max_iter`: `1,000`
  - `class_weight`: `'balanced'` (counters empirical class frequency disparities across intents)
  - `solver`: `'lbfgs'`
  - `random_state`: `42`

### 4.3 Persistence
- The fitted model and vocabulary pipeline are saved to `models/baselines/tfidf_logreg_intent.joblib`.
