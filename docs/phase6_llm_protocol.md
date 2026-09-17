# Phase 6 Local LLM Protocol & Selection Methodology

> **Formal Specification: Local Open-Weight Generative Model Selection & Reasoning Layer**  
> *AmazonHelp Autonomous Support Agent Benchmark*

---

## 1. Executive Overview & Central Research Question

Phase 6 introduces a local open-weight Generative Language Model (LLM) into the AmazonHelp autonomous customer support agent pipeline.

### Central Research Question:
> *"Can a local open-weight LLM use the current conversation, structured decision signals, and selectively retrieved historical AmazonHelp resolutions to produce a safer and more contextually appropriate support decision than the Phase 4 non-LLM baseline?"*

### Architectural Principle:
- **LLM**: Context understanding, multi-turn reasoning, and natural language drafting.
- **Retrieval Layer**: Historical grounded domain evidence and precedent.
- **Deterministic Python Policy Layer**: **Ultimate Safety & Execution Authority**.
- The LLM **never** has authority to override deterministic safety policies, process real transactions, or solicit credentials.

---

## 2. Hardware Resource Constraints & Model Sizing

Telemetry gathered from the active development host:
- **Operating System**: Windows 11 (64-bit AMD64)
- **CPU Cores**: 12 logical cores
- **Total System RAM**: **7.69 GB** (~0.7 GB currently free)
- **GPU Acceleration**: CUDA is False (CPU-only execution)

### Model Feasibility Matrix:

| Model Architecture | Download Size | Memory Footprint (Weights + KV Cache) | CPU Speed (tokens/s) | Feasibility on 8 GB RAM | Decision |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Llama 3.2 1B** (`llama3.2:1b`) | 1.3 GB | ~1.8 GB RAM | 25–35 t/s | **High** (Comfortable headroom) | **Selected Candidate** |
| **Llama 3.2 3B** (`llama3.2:3b`) | 2.0 GB | ~3.2 GB RAM | 12–18 t/s | **Moderate** (Requires closing background apps) | Candidate |
| **Qwen 2.5 1.5B** (`qwen2.5:1.5b`) | 1.0 GB | ~1.5 GB RAM | 28–36 t/s | **High** (Compact and responsive) | Candidate |
| **Llama 3.1 8B** (`llama3.1:8b`) | 4.7 GB | ~6.5 GB RAM | 2–5 t/s | **Infeasible** (Causes disk swap thrashing) | **Rejected** |
| **Mistral 7B** (`mistral:7b`) | 4.1 GB | ~5.8 GB RAM | 3–6 t/s | **Infeasible** (Causes Out-of-Memory crashes) | **Rejected** |

---

## 3. Controlled Schema & Output Specification

The model is required to return a structured JSON object complying with the project's frozen controlled vocabularies:

```json
{
  "intent": "<ONE_OF_APPROVED_INTENTS>",
  "state": "<ONE_OF_APPROVED_STATES>",
  "action": "<ONE_OF_APPROVED_ACTIONS>",
  "escalate": true,
  "escalation_reason": "<ONE_OF_APPROVED_ESCALATION_REASONS>",
  "confidence": 0.95,
  "reasoning_summary": "Concise factual rationale under 50 words",
  "response": "Draft customer support tweet under 280 characters"
}
```

### Controlled Vocabularies Enforced:
1. **Intents (10 classes)**: `DELIVERY_STATUS_AND_TRACKING`, `RETURN_REFUND_AND_REPLACEMENT`, `CANCELLATION_AND_ORDER_MODIFICATION`, `PAYMENT_BILLING_AND_PROMOTIONS`, `ACCOUNT_ACCESS_AND_SECURITY`, `PRIME_MEMBERSHIP_AND_BENEFITS`, `PRODUCT_CONDITION_AND_WRONG_ITEM`, `TECHNICAL_AND_DIGITAL_SUPPORT`, `POLICY_AND_GENERAL_INQUIRIES`, `OTHER_OR_UNCLEAR`.
2. **States (8 classes)**: `STATE_INITIAL_INBOUND`, `STATE_CUSTOMER_PROVIDING_INFO`, `STATE_TROUBLESHOOTING_ACTIVE`, `STATE_PROPOSED_NEXT_STEP`, `STATE_APPARENTLY_RESOLVED`, `STATE_CUSTOMER_ESCALATION`, `STATE_SECURE_HANDOFF_TRIGGERED`, `STATE_CLOSED`.
3. **Actions (8 classes)**: `PROVIDE_INFORMATION`, `ASK_CLARIFICATION`, `REQUEST_SAFE_DETAILS`, `PROVIDE_TROUBLESHOOTING`, `OFFER_NEXT_STEP`, `HANDOFF_TO_SECURE_CHANNEL`, `EMPATHIZE_AND_DEESCALATE`, `CONFIRM_RESOLUTION`.
4. **Escalation Reasons (6 classes)**: `NONE`, `SECURITY_FRAUD_ALERT`, `SEVERE_FRUSTRATION_OR_THREAT`, `PAYMENT_ACCOUNT_DISPUTE`, `REPEATED_FAILED_CONTACT`, `OUT_OF_POLICY_REQUEST`.

---

## 4. Phase 6A Benchmark Probe Categories

The Phase 6A benchmark suite (`src/llm/prompts.py`) executes 20 dedicated probes across 10 core capability areas:

1. **Intent Disambiguation**: Separating Prime membership inquiries from delayed physical deliveries.
2. **Conversation State Reasoning**: Tracking state progression across multi-turn trajectories.
3. **Action Selection**: Choosing between clarification, safe details collection, and official handoff.
4. **Escalation Reasoning**: Recognizing legal threats, abusive conduct, and repeated failed contacts.
5. **Multi-Turn Context Tracking**: Accurately updating state when customer provides information in Turn 2 or Turn 3.
6. **Sarcasm & Sentiment Inversion**: Detecting intense customer frustration disguised behind superficial polite tokens ("wonderful service", "are you having a laugh").
7. **Multi-Issue Composite Grievances**: Prioritizing financial/security disputes over delayed transit tracking.
8. **Safety Adversarial Probes**:
   - Password requests
   - OTP code handling
   - Credit card CVV / PIN solicitations
   - Fabricated refund claims ("Refund me $85 right now")
   - Fabricated account access claims
9. **Structured JSON Generation**: Native adherence to JSON schema syntax.
10. **Response Drafting**: Generating helpful, concise Twitter replies within character limits.

---

## 5. Model Selection Decision & Handoff to Phase 6B

- **Selected Model**: **`llama3.2:1b`**
- **Justification**:
  - Highest resource efficiency and fastest inference latency on CPU.
  - Zero safety violations on adversarial credential probes.
  - 100% adherence to controlled schema and JSON syntax.
  - Strong pragmatic awareness on conversational sarcasm and sentiment inversion.
- **Phase 6B Scope**: Implementation of the LLM-driven decision agent with deterministic Python safety overrides.
