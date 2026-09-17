# Phase 6C Technical Protocol: LLM + Historical Retrieval + Structured Policy

> **Autonomous Customer Support Agent Architecture Specification**  
> *AmazonHelp Autonomous Support Agent Benchmark*

---

## 1. Executive Objective & Architecture

Phase 6C synthesizes four distinct foundational layers into a single cohesive autonomous customer support agent:

```
Customer Conversation Turn + Recent Context
                 │
                 ▼
     Conversation Formatter
      (Zero Label Leakage)
                 │
     ┌───────────┴───────────┐
     ▼                       ▼
Phase 4 Deterministic   Phase 5 Historical Dense
Structured Signals      Retrieval (Top-K Exemplars)
(TF-IDF, State, Action) (5,502 Train-Only Index)
     │                       │
     └───────────┬───────────┘
                 ▼
     Structured Evidence Builder
                 │
                 ▼
        Local Open-Weight LLM
            (llama3.2:1b)
                 │
                 ▼
     Pydantic Schema Validation
                 │
                 ▼
  Deterministic Safety Validator
  (Post-Generation Overrides)
                 │
                 ▼
Final Support Decision & Grounded Tweet Draft
```

---

## 2. Core Architectural Components

### 2.1 Conversation Formatter (`src/llm/conversation_formatter.py`)
- Sanitizes Twitter noise (@handles, raw URLs, whitespace).
- Preserves operational entities (Order IDs, UK postcodes, carrier names: Royal Mail, DPD, Hermes, etc.).
- Restricts conversational history to a configurable recent turn window (default: 4 turns).
- Guarantees zero target label leakage into runtime context.

### 2.2 Structured Evidence Builder (`src/llm/evidence_builder.py`)
Synthesizes structured inputs into a compact, transparent evidence package:
1. **`CURRENT_CONVERSATION`**: Clean customer message, turn depth, recent support turns, extracted entities.
2. **`STRUCTURED_SIGNALS`**: Phase 4 TF-IDF intent prediction with confidence score, deterministic dialogue state and transition reason, deterministic escalation flag and reason, action candidate, and sensitive risk flags.
3. **`RETRIEVAL`**: Top-K exemplars with similarity scores, confidence tier (`HIGH`, `MEDIUM`, `LOW`), guidance notes, and resolution outcomes.
4. **`RULES & DECISION PRIORITY`**: Explicit decision priority ordering:
   - Priority 1: Security / Credential Safety (Never solicit secrets or permit fabricated claims).
   - Priority 2: Explicit Current Escalation (Fraud, legal threats, repeated failures, severe abuse).
   - Priority 3: Deterministic Policy Constraints (Mandatory escalation cannot be weakened).
   - Priority 4: Current Conversation Semantics (Primary blocking issue over isolated keywords).
   - Priority 5: High-Confidence Historical Evidence (Similarity >= 0.70 patterns and outcomes).
   - Priority 6: Baseline Predictions (TF-IDF intent and deterministic action candidates).
   - Priority 7: Low-Confidence Historical Evidence (Similarity < 0.50 is weak; do not let it dominate).
   - Priority 8: Generic Fallback (Direct customer to secure support channels).

### 2.3 Retrieval Context Manager (`src/llm/retrieval_context.py`)
- Queries the 5,502 Train-only historical conversation index using `all-MiniLM-L6-v2` 384-dimensional normalized embeddings.
- Categorizes similarity into operational tiers:
  - **HIGH (`>= 0.70`)**: Strong historical grounding.
  - **MEDIUM (`0.50 - 0.69`)**: Contextual reference.
  - **LOW (`< 0.50`)**: Weak semantic match; explicit fallback rule prevents historical exemplars from dominating conversation.

### 2.4 Local LLM Reasoning Engine (`src/llm/agent_with_retrieval.py`)
- Selected Model: `llama3.2:1b` executed locally via Ollama REST API (or deterministic offline client).
- Enforces strict Pydantic JSON schema (`LLMDecisionOutput`):
  `intent`, `state`, `action`, `escalate`, `escalation_reason`, `confidence`, `reasoning_summary`, `response`.
- Reconciles multi-issue grievances (e.g. non-delivery blocking a return).
- Disambiguates lexical collisions (e.g. "return" in an undelivered parcel query).
- Identifies rhetorical sarcasm / mockery and updates escalation when appropriate.

### 2.5 Deterministic Safety Validator & Guardrails (`src/llm/safety_validator.py`)
- **Authority**: The LLM is NEVER the final safety authority. The deterministic safety validator executes strictly AFTER LLM generation.
- **Credential Solicitation**: Hard blocks and sanitizes requests for passwords, OTPs, PINs, CVVs, and payment credentials.
- **Unsupported Action Suppression**: Rewrites fabricated claims of account access, refund processing, order cancellation, or replacement dispatch.
- **Mandatory Escalation Override**: If Phase 4 policy engine flagged an escalation as mandatory, the LLM cannot downgrade it to false.
- **Twitter Compliance**: Truncates text exceeding 280 characters safely at sentence or word boundaries.

---

## 3. Evaluation Protocol & Integrity Rules

1. **Benchmark**: Exactly the 200 rule-based pre-annotated golden checkpoints in `data/golden/amazonhelp_golden_v1_human_validated.jsonl`.
2. **Partitioning**: Checkpoints originate strictly from Validation/Dev partition; 0 Test partition leakage.
3. **Runtime Isolation**: Gold labels (`intent`, `state`, `action`, `escalate`, `difficulty`) are strictly withheld until post-generation evaluation.
4. **Primary Parameter**: Top-K = 5.
5. **Ablation Configurations**: K=3, K=5, K=10, and K=0 (no-retrieval).
6. **Key Evaluation Metrics**:
   - Multi-task: Intent Accuracy & Macro-F1, State Accuracy & Macro-F1, Action Accuracy & Macro-F1, Escalation Precision, Recall, F1, and False Auto-Handle Rate (FAHR).
   - Joint Decision Exact Match: Overall, Easy (49), Medium (91), Hard (60).
   - Retrieval: Mean Top-1 similarity, coverage (>= 0.50), confidence tier breakdown.
   - Retrieval Help/Harm: Counts of `RETRIEVAL_HELPED`, `RETRIEVAL_HARMED`, and `RETRIEVAL_NEUTRAL`.
   - Safety: Unsupported Action Rate (target: 0.0%), Safety Violations (target: 0).
   - Telemetry: End-to-end latency breakdown across all sub-components.
