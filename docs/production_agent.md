# AmazonHelp Production Customer Support AI Agent

**Evaluator & Production Architecture Guide**  
**Phase:** 7B (Production Agent + Evaluator Demo)  
**Selected Local Model:** `llama3.2:1b` (via local Ollama daemon)  
**Dense Retrieval Embedder:** `all-MiniLM-L6-v2` (SentenceTransformer, 384 dimensions)  
**Historical Retrieval Corpus:** 5,502 Train-only multi-turn support dialogues (K=5)  
**Evaluation Benchmark:** 200 rule-based pre-annotated Dev-only checkpoints  

---

## 1. System Overview

The AmazonHelp Autonomous Support Agent is a hybrid AI system designed for multi-turn customer support on Twitter. It combines:
1. **Deterministic Structured Understanding:** Fast baseline intent classification and multi-turn state tracking.
2. **Historical Dense Retrieval:** Augmenting decisions with 5,502 actionable, high-quality historical customer service conversations from the Train partition.
3. **Local Neural Generation:** Constrained structured decision-making using the open-weight `llama3.2:1b` model.
4. **Deterministic Post-Generation Safety Layer:** An authoritative guardrail outside the LLM that intercepts sensitive credential requests, rewrites fabricated transaction claims, and enforces mandatory safety escalations.

```
Customer Conversation Turn
         │
         ▼
Conversation Manager (Session State & Memory)
         │
         ▼
Structured Signals & Phase 4 Policies
         │
         ├─────────────────────────────────────────┐
         ▼                                         ▼
Deterministic Policy Engine              Historical Retrieval (K=5)
(Intent, State, Escalation)              (Train-Only Corpus, 5,502 docs)
         │                                         │
         └────────────────────┬────────────────────┘
                              ▼
                       Evidence Builder
                              │
                              ▼
                          Local LLM
                  (llama3.2:1b via Ollama)
                              │
                              ▼
                     Structured Decision
                              │
                              ▼
                 Deterministic Safety Layer
                 (Credential Guardrails & Policy)
                              │
                              ▼
                    Final Action & Response
```

---

## 2. Taxonomy and Vocabularies

### 2.1 The 10 Controlled Support Intents
1. `DELIVERY_STATUS_AND_TRACKING`: Order tracking, delayed transit, carrier updates.
2. `RETURN_REFUND_AND_REPLACEMENT`: Returns, drop-offs, refund timelines, replacements.
3. `PRODUCT_CONDITION_AND_WRONG_ITEM`: Damaged packages, missing pieces, wrong items.
4. `PAYMENT_BILLING_AND_PROMOTIONS`: Double billing, card authorization, promotional codes.
5. `ACCOUNT_ACCESS_AND_SECURITY`: Password reset, hacked accounts, 2FA/OTP issues.
6. `PRIME_MEMBERSHIP_AND_BENEFITS`: Prime charges, annual renewal, video/shipping perks.
7. `TECHNICAL_AND_DIGITAL_SUPPORT`: Kindle, Fire TV, app errors, digital streaming bugs.
8. `CANCELLATION_AND_ORDER_MODIFICATION`: Modifying delivery address or canceling orders.
9. `POLICY_AND_GENERAL_INQUIRIES`: Warranty, international shipping, store policies.
10. `OTHER_OR_UNCLEAR`: Vague greetings or unclassifiable inquiries.

### 2.2 The 8 Dialogue States
1. `STATE_INITIAL_INBOUND`: Opening customer grievance.
2. `STATE_CLARIFICATION_REQUESTED`: Agent prompted customer for necessary ambiguity resolution.
3. `STATE_CUSTOMER_PROVIDING_INFO`: Customer supplied order ID, carrier name, or postcode.
4. `STATE_TROUBLESHOOTING_ACTIVE`: Active technical diagnostic steps in progress.
5. `STATE_SECURE_HANDOFF_TRIGGERED`: Customer routed to private direct message (DM).
6. `STATE_CUSTOMER_ESCALATION`: Frustrated customer or high-risk safety event.
7. `STATE_APPARENTLY_RESOLVED`: Issue addressed and acknowledged.
8. `STATE_ABANDONED_OR_CLOSED`: Dialogue ended or timed out.

### 2.3 The 8 Support Actions
1. `PROVIDE_INFORMATION`: Share policy, tracking URL, or self-service guidance.
2. `ASK_CLARIFICATION`: Query missing details on ambiguous requests.
3. `REQUEST_SAFE_DETAILS`: Request safe non-sensitive identifiers (postcode, carrier name).
4. `PROVIDE_TROUBLESHOOTING`: Step-by-step resolution steps for digital/app issues.
5. `OFFER_NEXT_STEP`: Actionable operational direction.
6. `HANDOFF_TO_SECURE_CHANNEL`: Direct customer to private DM or verified support link.
7. `EMPATHIZE_AND_DEESCALATE`: Acknowledge severe frustration and defuse tension.
8. `CONFIRM_RESOLUTION`: Acknowledge completion and close turn.

---

## 3. Historical Retrieval Augmentation (K=5)

* **Corpus Isolation:** The retrieval corpus contains **5,502 dialogues** drawn **strictly from the Train partition**. No Validation (Dev) or Test conversations were ever indexed.
* **Embedding Model:** Dense 384-dimensional embeddings generated with `sentence-transformers/all-MiniLM-L6-v2`.
* **Search Mechanics:** Cosine similarity via L2-normalized vector dot product over `data/indexes/retrieval_index.npz`.
* **Confidence Tiers:**
  * **HIGH:** Similarity $\ge 0.70$
  * **MEDIUM:** $0.50 \le \text{Similarity} < 0.70$
  * **LOW:** $\text{Similarity} < 0.50$
* **Zero Leakage Invariant:** The retrieval engine provides only historical customer text, resolution summaries, and derived support replies. It never receives or exposes gold target labels.

---

## 4. Deterministic Safety Architecture

The generative LLM is **never** the final authority on safety or transactions. The post-generation `DeterministicSafetyValidator` strictly enforces:

1. **Authentication Credential Suppression:** Hard blocks any solicitation of passwords, passcodes, one-time passwords (OTP), PIN numbers, CVV/CVC codes, or full credit card numbers. If detected, the response is rewritten and forced to a secure channel handoff.
2. **Fabricated Action Claims:** Rewrites false claims where the model pretends to directly execute database actions:
   * `"I have refunded your order"` $\rightarrow$ Rewritten to safe self-service steps under 'Your Orders'.
   * `"I have cancelled your item"` $\rightarrow$ Rewritten to cancellation instructions.
   * `"I logged into your account"` $\rightarrow$ Rewritten to safe informational guidance.
3. **Mandatory Safety Escalations:** Enforces immediate escalation to human specialists on:
   * Account compromise or fraud alerts (`SECURITY_FRAUD_ALERT`)
   * Payment disputes or unauthorized card charges (`PAYMENT_ACCOUNT_DISPUTE`)
   * Severe customer frustration, legal threats, or ombudsman mentions (`SEVERE_FRUSTRATION_OR_THREAT`)

---

## 5. Local Ollama Runtime & Latency Truth

### 5.1 Local LLM Setup
The production agent runs locally via Ollama:
```bash
# 1. Start Ollama service
ollama serve

# 2. Pull official 1B open-weight model
ollama pull llama3.2:1b
```

### 5.2 Latency Truth & Scientific Integrity
> **Phase 6C's 73.82 ms latency figure used `MockOllamaClient` and should not be interpreted as real end-to-end local LLM latency.**

* **Framework Overhead:** MiniLM dense retrieval + policy rules + safety validation requires **~25 ms**.
* **Real Local CPU Generation:** Running `llama3.2:1b` locally on modern multi-core CPU requires **~1.5 to 3.5 seconds per turn**, depending on prompt token length and generated output tokens.
* **Offline Simulation Mode (`--mock`):** Allows rapid testing and verification in offline environments without a running Ollama daemon.

---

## 6. Command-Line Operations

The unified CLI (`cli.py`) operates from the repository root:

### 6.1 Interactive Multi-Turn Chat
```bash
# Run with live local Ollama
python cli.py chat

# Run in offline simulation mode
python cli.py chat --mock
```

### 6.2 Deterministic Demo Scenarios
```bash
# Run all 8 demo scenarios
python cli.py demo --mock

# Run a specific scenario (e.g. account security)
python cli.py demo --scenario 05 --mock
```

### 6.3 Golden Benchmark Evaluation
```bash
# Evaluate a subset of 20 checkpoints
python cli.py evaluate --limit 20 --mock

# Evaluate all 200 rule-based pre-annotated checkpoints
python cli.py evaluate --mock
```

### 6.4 Latency Benchmark
```bash
# Measure real CPU Ollama generation latency breakdown
python cli.py benchmark --iterations 5
```

### 6.5 System Verification
```bash
# Run automated verification suites (Phase 7A and Phase 7B)
python cli.py verify
```

### 6.6 Production Smoke Tests
```bash
# Run all 12 production unit and integration smoke tests
python -m unittest tests/test_production_smoke.py
```
