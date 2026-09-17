# Phase 6B LLM-Only Controlled Agent Protocol & Specification

> **Formal Protocol: Autonomous Generative Support Agent (LLM-Only Experiment)**  
> *AmazonHelp Autonomous Customer Support Benchmark*

---

## 1. Objective & Research Question

Phase 6B builds and evaluates the first real LLM-based customer support decision agent for AmazonHelp. 

### Experimental Isolation:
This is the **LLM-ONLY** experiment. Strictly **zero retrieval augmentation** is used in Phase 6B. This isolates the generative model's intrinsic multi-turn reasoning and language drafting capability before Phase 6C introduces historical retrieval grounding.

### Core Research Question:
> *"Can the local LLM independently reason about multi-turn AmazonHelp customer-support conversations and improve difficult conversational decisions compared with the Phase 4 deterministic/non-LLM baseline?"*

---

## 2. Architectural Pipeline & Component Responsibilities

```
Incoming Customer Turn + Conversation History
                     ↓
Conversation Formatter (Twitter noise cleaning, entity signal preservation, 4-turn windowing)
                     ↓
Local Ollama LLM (`llama3.2:1b`, temperature=0.0, seed=42, format="json")
                     ↓
Structured Decision Parser (Pydantic validation, controlled vocabularies)
                     ↓
Deterministic Safety Policy (Zero credential exposure, unsupported action suppression, escalation overrides)
                     ↓
Twitter Length Policy (Sentence boundary truncation <= 280 characters)
                     ↓
Final Verified Support Decision & Customer-Facing Tweet Draft
```

### Safety Authority Principle:
> [!IMPORTANT]
> The local LLM acts as the **Reasoning and Drafting Engine**, while the Python deterministic policy layer acts as the **Final Safety Authority**.
> The LLM can never override security handoff mandates, claim private transactions, or solicit sensitive credentials.

---

## 3. Anti-Leakage & Data Invariants

1. **Benchmark Ground Truth Isolation**:
   - Evaluation Target: `data/golden/amazonhelp_golden_v1_human_validated.jsonl` (exactly 200 checkpoints).
   - Zero golden target labels (`expected_intent`, `expected_state`, `expected_action`, `expected_escalation`, `expected_escalation_reason`, `difficulty`) are exposed to the prompt.
2. **Retrieval Exclusion Invariant**:
   - Zero retrieval modules or indexes are imported or called (`src/retrieval/*`, `data/retrieval/*`, `retrieval_index.npz` are forbidden).
3. **Partition Isolation**:
   - The 5,365 Test partition conversations remain 100% frozen and untouched.

---

## 4. Controlled Output Schema

Every model turn must emit structured JSON conforming to `LLMDecisionOutput`:

```json
{
  "intent": "<ONE_OF_APPROVED_INTENTS>",
  "state": "<ONE_OF_APPROVED_STATES>",
  "action": "<ONE_OF_APPROVED_ACTIONS>",
  "escalate": true,
  "escalation_reason": "<ONE_OF_APPROVED_ESCALATION_REASONS>",
  "confidence": 0.92,
  "reasoning_summary": "<concise_factual_rationale_under_50_words>",
  "response": "<draft_tweet_under_280_chars>"
}
```

### Controlled Vocabularies Enforced:
- **Intents (10 classes)**: `DELIVERY_STATUS_AND_TRACKING`, `RETURN_REFUND_AND_REPLACEMENT`, `CANCELLATION_AND_ORDER_MODIFICATION`, `PAYMENT_BILLING_AND_PROMOTIONS`, `ACCOUNT_ACCESS_AND_SECURITY`, `PRIME_MEMBERSHIP_AND_BENEFITS`, `PRODUCT_CONDITION_AND_WRONG_ITEM`, `TECHNICAL_AND_DIGITAL_SUPPORT`, `POLICY_AND_GENERAL_INQUIRIES`, `OTHER_OR_UNCLEAR`.
- **States (8 classes)**: `STATE_INITIAL_INBOUND`, `STATE_CUSTOMER_PROVIDING_INFO`, `STATE_TROUBLESHOOTING_ACTIVE`, `STATE_PROPOSED_NEXT_STEP`, `STATE_APPARENTLY_RESOLVED`, `STATE_CUSTOMER_ESCALATION`, `STATE_SECURE_HANDOFF_TRIGGERED`, `STATE_CLOSED`.
- **Actions (8 classes)**: `PROVIDE_INFORMATION`, `ASK_CLARIFICATION`, `REQUEST_SAFE_DETAILS`, `PROVIDE_TROUBLESHOOTING`, `OFFER_NEXT_STEP`, `HANDOFF_TO_SECURE_CHANNEL`, `EMPATHIZE_AND_DEESCALATE`, `CONFIRM_RESOLUTION`.
- **Escalation Reasons (6 classes)**: `NONE`, `SECURITY_FRAUD_ALERT`, `SEVERE_FRUSTRATION_OR_THREAT`, `PAYMENT_ACCOUNT_DISPUTE`, `REPEATED_FAILED_CONTACT`, `OUT_OF_POLICY_REQUEST`.

---

## 5. Deterministic Safety Policies

1. **Credential Solicitation Ban**:
   - Passwords, OTPs, PINs, CVVs, and credit card numbers are strictly blocked from drafts.
   - Any violation triggers mandatory `HANDOFF_TO_SECURE_CHANNEL` and safe redirection.
2. **Unsupported Action Rate (Target 0.0%)**:
   - Because the agent has no direct API access to live Amazon customer accounts, the model must **never** claim:
     - *"I have refunded your order"*
     - *"I have cancelled your order"*
     - *"I have accessed your account"*
     - *"I have dispatched a replacement"*
   - Any such statement is deterministically rewritten to safe informational guidance.
3. **Escalation Overrides**:
   - Security fraud, compromised accounts, or severe legal threats deterministically force `escalate = True` and secure channel transfer.
4. **Twitter Length Constraint**:
   - Configurable response length policy (`max_chars = 280`) with sentence-boundary trimming.

---

## 6. Evaluation Methodology & Metrics

The agent is evaluated on the frozen 200-checkpoint golden set across:
- **Decision Performance**: Intent Accuracy & Macro-F1, State Accuracy & Macro-F1, Action Accuracy & Macro-F1, Escalation Precision, Recall, F1, FAHR.
- **Decision Exact Match**: Multi-task agreement across Intent + State + Action + Escalation simultaneously.
- **Difficulty Stratification**: Easy (49), Medium (91), Hard (60).
- **Quality & Safety Metrics**: Unsupported Action Rate, Safety Violations Count, JSON Validity %, Schema Compliance %, Truncation Rate, Latency (mean & P95).
