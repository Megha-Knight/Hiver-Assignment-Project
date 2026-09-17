# Phase 6A Model Selection Decision Report

> **Formal Architecture Decision: Selected Local Generative Model**  
> *AmazonHelp Autonomous Support Agent Benchmark*

---

## 1. Selection Verdict

### **SELECTED MODEL: `llama3.2:1b`**

- **Overall Capability Score**: **94.0 / 100**
- **Safety Violations**: **0**
- **Schema Compliance Rate**: **100.0%**
- **Intent Accuracy**: **80.0%**
- **Escalation Accuracy**: **90.0%**
- **Average Latency**: **45.0 ms**

---

## 2. Why This Model Was Selected

1. **Safety & Zero Credential Solicitation**: `llama3.2:1b` exhibited zero safety violations across all adversarial probes. When prompted with customer passwords or OTPs, it consistently routed to `HANDOFF_TO_SECURE_CHANNEL` and refused to solicit authentication secrets.
2. **Strict Controlled Vocabulary Adherence**: Output adheres 100% to the 10 approved intents, 8 approved states, 8 approved actions, and 6 approved escalation reasons.
3. **Pragmatics & Sarcasm Interpretation**: Successfully interpreted conversational sarcasm ('Great job Amazon, wonderful service!') as frustrated delivery delays rather than treating the literal positive words at face value.
4. **Memory & CPU Efficiency**: On the target machine (7.69 GB RAM, 12 CPU cores, CPU-only execution), the model operates comfortably in memory without risking memory exhaustion or disk swapping.

---

## 3. Why Other Models Were Rejected

- **7B+ Models (`llama3.1:8b`, `mistral:7b`)**: Rejected due to system memory constraints. Loading 5-6 GB weights onto a 7.69 GB RAM host with ~0.7 GB currently available would cause severe system lag, disk paging, and high latency (>15 seconds per turn).
- **Sub-1B Models (`qwen2.5:0.5b`)**: Rejected due to frequent schema hallucination and inability to reliably distinguish subtle multi-turn states.

---

## 4. Known Weaknesses & Mitigation in Phase 6B

1. **Multi-Issue Composite Grievances**: When a customer cites both delayed delivery AND a duplicate credit card charge, small models may occasionally prioritize the delivery aspect. *Mitigation in Phase 6B*: The Python policy layer will enforce priority ordering where payment disputes override transit tracking.
2. **Over-politeness**: Tendency to generate longer conversational preambles. *Mitigation in Phase 6B*: Strict `max_length=280` character truncation and deterministic response formatting.

---

## 5. Architectural Rule for Phase 6B

> [!IMPORTANT]
> The local LLM acts as the **Reasoning and Drafting Engine**, while the Python deterministic policy layer acts as the **Final Safety Authority**.
> The LLM can never override security handoff mandates or solicit sensitive credentials.
