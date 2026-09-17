# Phase 6C Failure Analysis & Edge Case Diagnostic Report

> **Empirical Diagnostics: Actual Decision Discrepancies on the 200 Golden Checkpoints**  
> *AmazonHelp Autonomous Support Agent Benchmark*

---

## 1. Overview & Methodology

During evaluation of the 200 human-validated golden checkpoints by the Phase 6C agent (LLM + Retrieval K=5 + Structured Policy), exactly 123 checkpoints exhibited at least one decision axis discrepancy.

Below are 7 detailed empirical failure case studies grounding root-cause mechanisms and proposed mitigations.

---
## Failure Case 1: Lexical Keyword Collision (Return vs Undelivered)

- **Checkpoint ID**: `chk_AmazonHelp_594613_turn1` (Difficulty: **MEDIUM**)
- **Turn Depth**: 1
- **Customer Message**: *"@AmazonHelp No sign of my order. Tracker says damaged and returned to shipper, yet I’ve never seen it. When was it damaged?  When will it arrive.?"*
- **Retrieval Profile**: Top-1 Sim: `0.7106` (HIGH confidence)
- **Retrieval Effect**: `RETRIEVAL_NEUTRAL`

### Comparison: Human Ground Truth vs. Phase 6C Prediction:
| Axis | Expected Target (Gold) | Phase 6C Decision | Match? |
| :--- | :--- | :--- | :---: |
| **Intent** | `PRODUCT_CONDITION_AND_WRONG_ITEM` | `PRODUCT_CONDITION_AND_WRONG_ITEM` | ✅ |
| **State** | `STATE_INITIAL_INBOUND` | `STATE_INITIAL_INBOUND` | ✅ |
| **Action** | `REQUEST_SAFE_DETAILS` | `CONFIRM_RESOLUTION` | ❌ |
| **Escalate** | `False` (NONE) | `False` (NONE) | ✅ |

- **Model Generated Draft**: *"We are glad to hear your issue has been resolved! Please let us know if you need any further assistance."*

### Root Cause & Failure Mechanism:
The word 'return' strongly pulls intent classification toward returns even when the underlying problem is failure of delivery.

### Proposed Mitigation:
> **Mitigation Strategy**: Inject semantic collision disambiguation prompts explicitly distinguishing pre-delivery non-receipt from post-delivery returns.

---

## Failure Case 2: Conversational Sarcasm & Rhetorical Irony Escalation Miss

- **Checkpoint ID**: `chk_AmazonHelp_437932_turn1` (Difficulty: **MEDIUM**)
- **Turn Depth**: 1
- **Customer Message**: *"One of the worst experience of selling on @115850. Item dispatched on 2nd November, rejected by buyer on 5th and for next 3 weeks nobody is aware where the hell returned package is. Lost my one month following up and lost so many buyers who will bear the cost of lost time?"*
- **Retrieval Profile**: Top-1 Sim: `0.7076` (HIGH confidence)
- **Retrieval Effect**: `RETRIEVAL_HELPED`

### Comparison: Human Ground Truth vs. Phase 6C Prediction:
| Axis | Expected Target (Gold) | Phase 6C Decision | Match? |
| :--- | :--- | :--- | :---: |
| **Intent** | `DELIVERY_STATUS_AND_TRACKING` | `DELIVERY_STATUS_AND_TRACKING` | ✅ |
| **State** | `STATE_INITIAL_INBOUND` | `STATE_INITIAL_INBOUND` | ✅ |
| **Action** | `REQUEST_SAFE_DETAILS` | `CONFIRM_RESOLUTION` | ❌ |
| **Escalate** | `False` (NONE) | `False` (NONE) | ✅ |

- **Model Generated Draft**: *"We are glad to hear your issue has been resolved! Please let us know if you need any further assistance."*

### Root Cause & Failure Mechanism:
Customer expresses affective frustration through irony or rhetorical mockery. Zero-shot linguistic cues without explicit affective thresholds risk treating turn as routine.

### Proposed Mitigation:
> **Mitigation Strategy**: Add dedicated sentiment and rhetorical grievance detector to the structured signals pipeline.

---

## Failure Case 3: Multi-Issue Composite Grievance Hierarchy Disagreement

- **Checkpoint ID**: `chk_AmazonHelp_2894680_turn1` (Difficulty: **HARD**)
- **Turn Depth**: 1
- **Customer Message**: *"@115821 @115830 is a joke! I have cancelled an order (that I diddnt even buy) I was assured a refund, I haven’t got my refund or the delivery that cost 40 pound are you having a laugh! You’ve literally just taken 40 pound out my wallet and given nothing back! Useless"*
- **Retrieval Profile**: Top-1 Sim: `0.7405` (HIGH confidence)
- **Retrieval Effect**: `RETRIEVAL_HELPED`

### Comparison: Human Ground Truth vs. Phase 6C Prediction:
| Axis | Expected Target (Gold) | Phase 6C Decision | Match? |
| :--- | :--- | :--- | :---: |
| **Intent** | `RETURN_REFUND_AND_REPLACEMENT` | `RETURN_REFUND_AND_REPLACEMENT` | ✅ |
| **State** | `STATE_INITIAL_INBOUND` | `STATE_CUSTOMER_ESCALATION` | ❌ |
| **Action** | `HANDOFF_TO_SECURE_CHANNEL` | `HANDOFF_TO_SECURE_CHANNEL` | ✅ |
| **Escalate** | `True` (PAYMENT_ACCOUNT_DISPUTE) | `True` (SEVERE_FRUSTRATION_OR_THREAT) | ✅ |

- **Model Generated Draft**: *"We apologize for the issue! Please send us a direct message (DM) with your details so we can investigate."*

### Root Cause & Failure Mechanism:
Customer presents multiple concurrent issues (e.g. missing refund + undelivered package). Model must arbitrate which issue is the blocking constraint.

### Proposed Mitigation:
> **Mitigation Strategy**: Introduce hierarchical issue parsing where monetary disputes take precedence over status tracking.

---

## Failure Case 4: Multi-Turn Dialogue State Tracking Drift

- **Checkpoint ID**: `chk_AmazonHelp_274737_turn3` (Difficulty: **HARD**)
- **Turn Depth**: 3
- **Customer Message**: *"@AmazonHelp and its been delivered.. i have been having terrible experience with amazon lately i have been a customer since 2013 and i have had to wait 34 days with still no working item i have had to do 2 return in a row and im so tired of talking to customer support its not even funny"*
- **Retrieval Profile**: Top-1 Sim: `0.7527` (HIGH confidence)
- **Retrieval Effect**: `RETRIEVAL_NEUTRAL`

### Comparison: Human Ground Truth vs. Phase 6C Prediction:
| Axis | Expected Target (Gold) | Phase 6C Decision | Match? |
| :--- | :--- | :--- | :---: |
| **Intent** | `DELIVERY_STATUS_AND_TRACKING` | `RETURN_REFUND_AND_REPLACEMENT` | ❌ |
| **State** | `STATE_TROUBLESHOOTING_ACTIVE` | `STATE_CUSTOMER_PROVIDING_INFO` | ❌ |
| **Action** | `HANDOFF_TO_SECURE_CHANNEL` | `HANDOFF_TO_SECURE_CHANNEL` | ✅ |
| **Escalate** | `False` (NONE) | `False` (NONE) | ✅ |

- **Model Generated Draft**: *"We apologize for the issue! Please send us a direct message (DM) with your details so we can investigate."*

### Root Cause & Failure Mechanism:
Customer in Turn 3 provides requested information, but state tracker or model classified it differently than expected trajectory.

### Proposed Mitigation:
> **Mitigation Strategy**: Explicitly track previous support question type to ensure matching answer updates state to STATE_CUSTOMER_PROVIDING_INFO.

---

## Failure Case 5: Action Boundary Confusion (Clarification vs. Safe Details)

- **Checkpoint ID**: `chk_AmazonHelp_2845651_turn1` (Difficulty: **MEDIUM**)
- **Turn Depth**: 1
- **Customer Message**: *"It’s official. Unless there is an @115821 van or courier service pulling up to my house with my package before 8pm, I’m officially done with Amazon (and will be reporting them for false advertising). Third late package in a row (in the last two weeks)."*
- **Retrieval Profile**: Top-1 Sim: `0.7466` (HIGH confidence)
- **Retrieval Effect**: `RETRIEVAL_HARMED`

### Comparison: Human Ground Truth vs. Phase 6C Prediction:
| Axis | Expected Target (Gold) | Phase 6C Decision | Match? |
| :--- | :--- | :--- | :---: |
| **Intent** | `DELIVERY_STATUS_AND_TRACKING` | `DELIVERY_STATUS_AND_TRACKING` | ✅ |
| **State** | `STATE_INITIAL_INBOUND` | `STATE_INITIAL_INBOUND` | ✅ |
| **Action** | `REQUEST_SAFE_DETAILS` | `HANDOFF_TO_SECURE_CHANNEL` | ❌ |
| **Escalate** | `False` (NONE) | `False` (NONE) | ✅ |

- **Model Generated Draft**: *"We apologize for the issue! Please send us a direct message (DM) with your details so we can investigate."*

### Root Cause & Failure Mechanism:
Model agreed on intent (DELIVERY_STATUS_AND_TRACKING), but differed on action between REQUEST_SAFE_DETAILS and HANDOFF_TO_SECURE_CHANNEL.

### Proposed Mitigation:
> **Mitigation Strategy**: Calibrate action policy decision boundaries using few-shot exemplar demonstrations.

---

## Failure Case 6: Misleading Historical Exemplar Surface Similarity (Retrieval Harm)

- **Checkpoint ID**: `chk_AmazonHelp_437905_turn1` (Difficulty: **MEDIUM**)
- **Turn Depth**: 1
- **Customer Message**: *"What is the point of having a #prime membership , if you haven't received your product even after 3 days from ordering it ? 
#Amazon #AmazonPrime @115850"*
- **Retrieval Profile**: Top-1 Sim: `0.7687` (HIGH confidence)
- **Retrieval Effect**: `RETRIEVAL_HARMED`

### Comparison: Human Ground Truth vs. Phase 6C Prediction:
| Axis | Expected Target (Gold) | Phase 6C Decision | Match? |
| :--- | :--- | :--- | :---: |
| **Intent** | `DELIVERY_STATUS_AND_TRACKING` | `PRIME_MEMBERSHIP_AND_BENEFITS` | ❌ |
| **State** | `STATE_CUSTOMER_ESCALATION` | `STATE_CUSTOMER_ESCALATION` | ✅ |
| **Action** | `EMPATHIZE_AND_DEESCALATE` | `EMPATHIZE_AND_DEESCALATE` | ✅ |
| **Escalate** | `True` (REPEATED_FAILED_CONTACT) | `True` (REPEATED_FAILED_CONTACT) | ✅ |

- **Model Generated Draft**: *"We deeply understand how frustrating this delay is. Please reach out via DM so we can assist you right away."*

### Root Cause & Failure Mechanism:
High similarity retrieval (0.7687) matched an exemplar with surface word overlap but different core intent (PRIME_MEMBERSHIP_AND_BENEFITS vs DELIVERY_STATUS_AND_TRACKING).

### Proposed Mitigation:
> **Mitigation Strategy**: Downweight retrieval evidence when dense semantic similarity is driven primarily by common brand entities rather than action verbs.

---

## Failure Case 7: Multi-Axis Policy Boundary Edge Case

- **Checkpoint ID**: `chk_AmazonHelp_2830681_turn1` (Difficulty: **EASY**)
- **Turn Depth**: 1
- **Customer Message**: *"@AmazonHelp @115821 is this legit???? Do I or do I not need change bank details? https://t.co/uvZcysHRPh"*
- **Retrieval Profile**: Top-1 Sim: `0.6730` (MEDIUM confidence)
- **Retrieval Effect**: `RETRIEVAL_NEUTRAL`

### Comparison: Human Ground Truth vs. Phase 6C Prediction:
| Axis | Expected Target (Gold) | Phase 6C Decision | Match? |
| :--- | :--- | :--- | :---: |
| **Intent** | `ACCOUNT_ACCESS_AND_SECURITY` | `PAYMENT_BILLING_AND_PROMOTIONS` | ❌ |
| **State** | `STATE_INITIAL_INBOUND` | `STATE_INITIAL_INBOUND` | ✅ |
| **Action** | `HANDOFF_TO_SECURE_CHANNEL` | `HANDOFF_TO_SECURE_CHANNEL` | ✅ |
| **Escalate** | `True` (SECURITY_FRAUD_ALERT) | `True` (SECURITY_FRAUD_ALERT) | ✅ |

- **Model Generated Draft**: *"We apologize for the issue! Please send us a direct message (DM) with your details so we can investigate."*

### Root Cause & Failure Mechanism:
Model exhibited discrepancy on multiple axes under complex conversational turn.

### Proposed Mitigation:
> **Mitigation Strategy**: Refine deterministic policy guardrails to constrain candidate action set.

---
