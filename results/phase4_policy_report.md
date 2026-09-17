# Phase 4 Deterministic Policy & Unified Decision Engine Report

> **Evaluation of Deterministic Escalation, State Tracking, and Action Selection**  
> *AmazonHelp Autonomous Support Agent Benchmark*

---

## 1. Executive Summary

The Phase 4 policy suite connects rule-based escalation, conversation state tracking, and action selection into a deterministic, auditable decision engine.

### Key High-Level Performance Metrics:
- **Escalation Precision / Recall / F1**: 78.6% / 24.4% / **37.3%**
- **False Auto-Handle Rate (FAHR)**: **75.6%** (Down from 100.0% in naive baseline)
- **Conversation State Tracking Accuracy**: **89.5%** (Macro-F1: 48.4%)
- **Action Policy Accuracy**: **88.5%** (Macro-F1: 80.2%)
- **Complete Multi-Task Exact Match**: **69.0%** across all 4 decision axes

---

## 2. Escalation Policy Evaluation

| Metric | Value | Description |
| :--- | :---: | :--- |
| **True Positives (TP)** | 11 | Correctly identified escalations (fraud, threats, payment disputes) |
| **False Positives (FP)** | 3 | Non-escalated turns unnecessarily escalated |
| **False Negatives (FN)** | 34 | Escalations missed by deterministic rules |
| **True Negatives (TN)** | 152 | Correctly auto-handled routine inquiries |
| **Precision** | 78.57% | Reliability when escalation flag is triggered |
| **Recall** | 24.44% | Coverage of risky situations |
| **Escalation F1-Score** | **37.29%** | Harmonic balance between safety and autonomy |
| **False Auto-Handle Rate** | **75.56%** | Risk metric (target < 10%) |

---

## 3. Conversation State Tracking Performance

- **Overall State Accuracy**: 89.50%
- **Macro-Averaged F1**: 48.39%

| Dialogue State | Support | Per-State Accuracy |
| :--- | :---: | :---: |
| `STATE_INITIAL_INBOUND` | 122 | **99.2%** |
| `STATE_CUSTOMER_PROVIDING_INFO` | 40 | **100.0%** |
| `STATE_TROUBLESHOOTING_ACTIVE` | 19 | **47.4%** |
| `STATE_CUSTOMER_ESCALATION` | 14 | **35.7%** |
| `STATE_APPARENTLY_RESOLVED` | 5 | **80.0%** |

---

## 4. Action Policy Performance

- **Overall Action Accuracy**: 88.50%
- **Macro-Averaged F1**: 80.17%

| Agent Action | Support | Per-Action Accuracy |
| :--- | :---: | :---: |
| `PROVIDE_INFORMATION` | 10 | **70.0%** |
| `ASK_CLARIFICATION` | 13 | **92.3%** |
| `REQUEST_SAFE_DETAILS` | 41 | **95.1%** |
| `PROVIDE_TROUBLESHOOTING` | 16 | **87.5%** |
| `OFFER_NEXT_STEP` | 8 | **87.5%** |
| `HANDOFF_TO_SECURE_CHANNEL` | 93 | **95.7%** |
| `EMPATHIZE_AND_DEESCALATE` | 14 | **35.7%** |
| `CONFIRM_RESOLUTION` | 5 | **80.0%** |

---

## 5. End-to-End Decision Exact-Match Analysis

An exact match requires that **all 4 decision dimensions** (`intent`, `state`, `action`, `escalation`) simultaneously match the human ground truth.

- **Overall Exact Match**: 138 / 200 (**69.0%**)
- Intent Match Rate: 87.5%
- State Match Rate: 89.5%
- Action Match Rate: 88.5%
- Escalation Match Rate: 81.5%

### Exact Match Stratified by Difficulty:

| Difficulty Tier | Total | Exact Matches | Exact Match Rate |
| :--- | :---: | :---: | :---: |
| `EASY` | 49 | 44 | **89.8%** |
| `MEDIUM` | 91 | 74 | **81.3%** |
| `HARD` | 60 | 20 | **33.3%** |

---

## 6. Safety Guardrail & Credential Verification

- **Secrets Solicitation Test**: 100% PASS. Zero attempts to solicit passwords, PINs, OTPs, CVVs, or full credit card credentials.
- **Controlled Vocabulary Integrity**: 100% PASS. All predicted intents, states, actions, and escalation reasons belong strictly to approved controlled vocabularies.
- **Auditability**: Every decision outputs complete transition trace and justification evidence.
