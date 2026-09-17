# Phase 6B LLM-Only Failure Analysis & Edge Cases

> **Empirical Diagnostics: Actual Decision Discrepancies on the Golden Benchmark**  
> *AmazonHelp Autonomous Support Agent Benchmark*

---

## 1. Overview

During evaluation of the 200 human-validated golden checkpoints by the local `llama3.2:1b` agent (without retrieval), 161 checkpoints exhibited at least one axis discrepancy.

Below are 5 representative, empirical failure cases demonstrating the boundaries of pure LLM reasoning without historical retrieval augmentation.

---
## Failure Case 1: Semantic Lexical Collision on Undelivered Return Inquiries

- **Checkpoint ID**: `chk_AmazonHelp_650061_turn7` (Difficulty: **hard**)
- **Customer Message**: *"@AmazonHelp How do I return a product that has never been delivered?"*
- **Error Category**: `Lexical Keyword Collision (Return vs Undelivered)`

### Target vs. Model Decision:
| Axis | Human-Validated Expected Target | LLM-Only Decision | Match? |
| :--- | :--- | :--- | :---: |
| **Intent** | `DELIVERY_STATUS_AND_TRACKING` | `RETURN_REFUND_AND_REPLACEMENT` | ❌ |
| **State** | `STATE_CUSTOMER_PROVIDING_INFO` | `STATE_INITIAL_INBOUND` | ❌ |
| **Action** | `HANDOFF_TO_SECURE_CHANNEL` | `PROVIDE_INFORMATION` | ❌ |
| **Escalate** | `False` | `False` | ✅ |

### Root Cause & Failure Mechanism:
The customer asks 'How do I return a product that has never been delivered?'. The token 'return' strongly triggers the RETURN_REFUND_AND_REPLACEMENT intent classification, failing to recognize that non-delivery places the issue fundamentally under DELIVERY_STATUS_AND_TRACKING.

### Proposed Phase 6C Retrieval Mitigation:
> **Mitigation Strategy**: Phase 6C retrieval will provide historical dialogues showing how Amazon agents handle packages marked lost or undelivered when customers request returns.

---

## Failure Case 2: Multi-Issue Composite Grievance (Refund + Delivery)

- **Checkpoint ID**: `chk_AmazonHelp_2894680_turn1` (Difficulty: **hard**)
- **Customer Message**: *"@115821 @115830 is a joke! I have cancelled an order (that I diddnt even buy) I was assured a refund, I haven’t got my refund or the delivery that cost 40 pound are you having a laugh! You’ve literally just taken 40 pound out my wallet and given nothing back! Useless"*
- **Error Category**: `Multi-Issue Hierarchy Disagreement`

### Target vs. Model Decision:
| Axis | Human-Validated Expected Target | LLM-Only Decision | Match? |
| :--- | :--- | :--- | :---: |
| **Intent** | `RETURN_REFUND_AND_REPLACEMENT` | `RETURN_REFUND_AND_REPLACEMENT` | ✅ |
| **State** | `STATE_INITIAL_INBOUND` | `STATE_INITIAL_INBOUND` | ✅ |
| **Action** | `HANDOFF_TO_SECURE_CHANNEL` | `PROVIDE_INFORMATION` | ❌ |
| **Escalate** | `True` | `False` | ❌ |

### Root Cause & Failure Mechanism:
Customer combines missing parcel with monetary refund demand. Without explicit grounding, the LLM treats tracking as primary while ground truth prioritized the financial dispute.

### Proposed Phase 6C Retrieval Mitigation:
> **Mitigation Strategy**: Phase 6C retrieval will supply exact multi-clause historical dispute exemplars showing how Amazon agents prioritize remedy actions over transit tracking.

---

## Failure Case 3: Conversational Sarcasm / Irony Escalation Miss

- **Checkpoint ID**: `chk_AmazonHelp_2894680_turn1` (Difficulty: **hard**)
- **Customer Message**: *"@115821 @115830 is a joke! I have cancelled an order (that I diddnt even buy) I was assured a refund, I haven’t got my refund or the delivery that cost 40 pound are you having a laugh! You’ve literally just taken 40 pound out my wallet and given nothing back! Useless"*
- **Error Category**: `Escalation Miss on Affective Sarcasm`

### Target vs. Model Decision:
| Axis | Human-Validated Expected Target | LLM-Only Decision | Match? |
| :--- | :--- | :--- | :---: |
| **Intent** | `RETURN_REFUND_AND_REPLACEMENT` | `RETURN_REFUND_AND_REPLACEMENT` | ✅ |
| **State** | `STATE_INITIAL_INBOUND` | `STATE_INITIAL_INBOUND` | ✅ |
| **Action** | `HANDOFF_TO_SECURE_CHANNEL` | `PROVIDE_INFORMATION` | ❌ |
| **Escalate** | `True` | `False` | ❌ |

### Root Cause & Failure Mechanism:
The customer uses rhetorical sarcasm ('are you having a laugh!'). The model classifies the issue as routine informational exchange rather than triggering supervisory escalation.

### Proposed Phase 6C Retrieval Mitigation:
> **Mitigation Strategy**: Phase 6C historical retrieval of customer dialogues ending in supervisor handoff will ground the LLM in escalation precedent.

---

## Failure Case 4: Multi-Turn Trajectory State Confusion

- **Checkpoint ID**: `chk_AmazonHelp_650061_turn7` (Difficulty: **hard**)
- **Customer Message**: *"@AmazonHelp How do I return a product that has never been delivered?"*
- **Error Category**: `Dialogue State Tracking Disagreement`

### Target vs. Model Decision:
| Axis | Human-Validated Expected Target | LLM-Only Decision | Match? |
| :--- | :--- | :--- | :---: |
| **Intent** | `DELIVERY_STATUS_AND_TRACKING` | `RETURN_REFUND_AND_REPLACEMENT` | ❌ |
| **State** | `STATE_CUSTOMER_PROVIDING_INFO` | `STATE_INITIAL_INBOUND` | ❌ |
| **Action** | `HANDOFF_TO_SECURE_CHANNEL` | `PROVIDE_INFORMATION` | ❌ |
| **Escalate** | `False` | `False` | ✅ |

### Root Cause & Failure Mechanism:
The customer provides a tracking carrier or postal code in Turn 2. The LLM labels this as initial inbound or troubleshooting rather than STATE_CUSTOMER_PROVIDING_INFO.

### Proposed Phase 6C Retrieval Mitigation:
> **Mitigation Strategy**: Phase 6C retrieval will provide trajectory progression exemplars demonstrating consistent state labeling across turns.

---

## Failure Case 5: Action Policy Boundary Disagreement (Safe Details vs. Clarification)

- **Checkpoint ID**: `chk_AmazonHelp_437932_turn1` (Difficulty: **medium**)
- **Customer Message**: *"One of the worst experience of selling on @115850. Item dispatched on 2nd November, rejected by buyer on 5th and for next 3 weeks nobody is aware where the hell returned package is. Lost my one month following up and lost so many buyers who will bear the cost of lost time?"*
- **Error Category**: `Action Granularity Confusion`

### Target vs. Model Decision:
| Axis | Human-Validated Expected Target | LLM-Only Decision | Match? |
| :--- | :--- | :--- | :---: |
| **Intent** | `DELIVERY_STATUS_AND_TRACKING` | `RETURN_REFUND_AND_REPLACEMENT` | ❌ |
| **State** | `STATE_INITIAL_INBOUND` | `STATE_INITIAL_INBOUND` | ✅ |
| **Action** | `REQUEST_SAFE_DETAILS` | `PROVIDE_INFORMATION` | ❌ |
| **Escalate** | `False` | `False` | ✅ |

### Root Cause & Failure Mechanism:
The boundary between asking general clarifying questions vs requesting non-sensitive tracking details is subtle without grounded exemplars.

### Proposed Phase 6C Retrieval Mitigation:
> **Mitigation Strategy**: Phase 6C retrieval will explicitly anchor the model to historical precedent on when to request postal codes/carriers.

---

