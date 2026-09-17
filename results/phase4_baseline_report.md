# Phase 4 Intent & Escalation Baseline Benchmark Report

> **Non-LLM Baseline Benchmark on 200 Human-Validated Golden Checkpoints**  
> *AmazonHelp Autonomous Support Agent Benchmark*

---

## 1. Executive Summary & Overview

Phase 4 establishes empirical, reproducible non-LLM baselines to serve as the scientific comparison anchor for subsequent retrieval-augmented and local LLM agents (Phases 5 & 6).

### Benchmark Architecture:
- **Evaluation Ground Truth**: Exactly 200 hand-validated decision checkpoints (`data/golden/amazonhelp_golden_v1_human_validated.jsonl`) sampled strictly from the **Validation (Dev) partition**.
- **Zero Leakage**: Test partition (5,365 conversations) remains 100% untouched. Model fitting was conducted strictly on the **Train partition (42,909 conversations)**.
- **Audit Timestamp**: `2026-09-15 15:59:21 UTC`

---

## 2. Intent Classification: Trivial vs. Simple ML Baseline Comparison

| Metric | Trivial Baseline (Majority Class) | ML Baseline (TF-IDF + Logistic Regression) | Delta (Absolute Improvement) |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 24.50% | 87.50% | **+63.00%** |
| **Macro-Averaged F1** | 3.94% | 83.08% | **+79.14%** |
| **Weighted-Averaged F1** | 9.64% | 88.52% | **+78.88%** |

> [!NOTE]
> **Trivial Baseline Majority Class**: Predicted `DELIVERY_STATUS_AND_TRACKING` across all 200 checkpoints, achieving 24.5% accuracy corresponding to the ground-truth prevalence of delivery inquiries in the validation sample.

---

## 3. Detailed ML Baseline Intent Performance (Per-Class Breakdown)

| Intent Class | Support | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| `DELIVERY_STATUS_AND_TRACKING` | 49 | 95.5% | 85.7% | **90.3%** |
| `RETURN_REFUND_AND_REPLACEMENT` | 34 | 94.1% | 94.1% | **94.1%** |
| `CANCELLATION_AND_ORDER_MODIFICATION` | 12 | 100.0% | 91.7% | **95.7%** |
| `PAYMENT_BILLING_AND_PROMOTIONS` | 17 | 88.9% | 94.1% | **91.4%** |
| `ACCOUNT_ACCESS_AND_SECURITY` | 21 | 94.4% | 81.0% | **87.2%** |
| `PRIME_MEMBERSHIP_AND_BENEFITS` | 11 | 52.9% | 81.8% | **64.3%** |
| `PRODUCT_CONDITION_AND_WRONG_ITEM` | 25 | 88.9% | 96.0% | **92.3%** |
| `TECHNICAL_AND_DIGITAL_SUPPORT` | 17 | 100.0% | 82.3% | **90.3%** |
| `POLICY_AND_GENERAL_INQUIRIES` | 10 | 100.0% | 70.0% | **82.3%** |
| `OTHER_OR_UNCLEAR` | 4 | 30.0% | 75.0% | **42.9%** |

---

## 4. Confidence Distribution & Uncertainty Analysis

- **Mean Prediction Confidence**: 0.8385
- **Median Prediction Confidence**: 0.9315
- **Min / Max Confidence**: 0.3135 / 1.0000
- **Low Confidence (< 0.40) Sample Count**: 7 checkpoints

### Representative Low-Confidence Checkpoints (Challenging Ambiguities):

- **Checkpoint**: `chk_AmazonHelp_2899622_turn2` (Confidence: `0.31`)
  *Customer Message*: *"@115850 @AmazonHelp @4030 They can,'t assist.. keep the amount. Let me aware the people or tell them about your fraud policy. You are just cheating the people and use their money. 9929029996 This is my contact no. Let me assist if u can."*
  *Expected Intent*: `POLICY_AND_GENERAL_INQUIRIES` | *Predicted*: `POLICY_AND_GENERAL_INQUIRIES`

- **Checkpoint**: `chk_AmazonHelp_2850664_turn2` (Confidence: `0.32`)
  *Customer Message*: *"@AmazonHelp It's everything I try and order! Why am I charged for prime when you can't fulfil your part of agreement? Should report you to trading standards! All this money all year for piss poor service!"*
  *Expected Intent*: `PRIME_MEMBERSHIP_AND_BENEFITS` | *Predicted*: `OTHER_OR_UNCLEAR`

- **Checkpoint**: `chk_AmazonHelp_2909234_turn5` (Confidence: `0.35`)
  *Customer Message*: *"@AmazonHelp My wife has just gotten off the phone with a rep that was nice enough to call her back. She also explained to them how egregious the delivery diver's behavior has been for the whole neighborhood. I am beyond shocked at how many packages were just lazily dropped at the wrong place"*
  *Expected Intent*: `DELIVERY_STATUS_AND_TRACKING` | *Predicted*: `DELIVERY_STATUS_AND_TRACKING`

- **Checkpoint**: `chk_AmazonHelp_2910334_turn1` (Confidence: `0.33`)
  *Customer Message*: *"@AmazonHelp I pay for Prime and I was charged for this item a week ago. Screenshot shows the last update on the tracker. I spoke with a support person yesterday - they said it would arrive by today. I'm very disappointed. Is there a point in Prime anymore? https://t.co/0wSQTQJr8f"*
  *Expected Intent*: `PAYMENT_BILLING_AND_PROMOTIONS` | *Predicted*: `PRIME_MEMBERSHIP_AND_BENEFITS`

- **Checkpoint**: `chk_AmazonHelp_2238658_turn1` (Confidence: `0.35`)
  *Customer Message*: *"@115850 @119625 what is this? I am not able to cast any video using my android device and Google Chromecast? Your cc team is saying that you guys are upgrading your application. I think you guys are cheating and trying to sell your fire TV deliberately."*
  *Expected Intent*: `TECHNICAL_AND_DIGITAL_SUPPORT` | *Predicted*: `PRIME_MEMBERSHIP_AND_BENEFITS`

---

## 5. Performance Stratified by Difficulty Tier

| Difficulty Tier | Checkpoints | Accuracy | Macro-F1 |
| :--- | :---: | :---: | :---: |
| `EASY` | 49 | 89.8% | 82.4% |
| `MEDIUM` | 91 | 87.9% | 76.7% |
| `HARD` | 60 | 85.0% | 65.2% |

---

## 6. Escalation Baselines: Autonomy vs. Safety Trade-Off

| Policy Mode | Precision | Recall | F1-Score | False Auto-Handle Rate (FAHR) | Operating Characteristic |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Never Escalate** (Default Auto-Handle) | 0.0% | 0.0% | 0.0% | **100.0%** | Maximizes autonomous volume, but exposes 100.0% of genuine escalations to dangerous auto-handling. |
| **Always Escalate** (Ultra-Conservative) | 22.5% | 100.0% | 36.7% | **0.0%** | Eliminates safety risk (0% FAHR), but destroys agent utility (0% autonomy). |

> [!IMPORTANT]
> **False Auto-Handle Rate (FAHR)**: Defined as $\frac{\text{FN}}{\text{TP} + \text{FN}}$. In customer support AI, FAHR measures safety failure—the rate at which dangerous turns (fraud, double charges, regulatory threats) are erroneously handled by an automated bot without human oversight.

---

## 7. Model Persistence & Provenance

- **Fitted Model Artifact**: `models/baselines/tfidf_logreg_intent.joblib`
- **Input Representation**: Immediate prior support utterance prepended to incoming customer message.
- **Reproducibility Guarantee**: Deterministic seed (`42`), zero access to Validation/Test during vectorizer fitting.
