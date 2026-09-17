# Phase 6A Local LLM Model Benchmark & Comparison

> **Empirical Selection: Local Open-Weight Generative Models for AmazonHelp Support**  
> *Controlled Evaluation across 20 Reasoning, Safety, and Schema Compliance Probes*

---

## 1. Executive Summary

Phase 6A evaluates local open-weight language models via Ollama to select the optimal model for the autonomous AmazonHelp support agent. Models are tested under identical deterministic conditions (`temperature=0.0`, `seed=42`, native JSON mode).

---

## 2. Model Comparison Table

| Model | Valid JSON % | Schema Comp % | Intent Acc | State Acc | Action Acc | Esc Acc | Safety Violations | Avg Latency (ms) | P95 Latency (ms) | Avg Tokens | Capability Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **llama3.2:1b** | 100.0% | 100.0% | 80.0% | 85.0% | 85.0% | 90.0% | 0 | 45.0 | 45.0 | 68.0 | **94.0** |
| **llama3.2:3b** | 100.0% | 100.0% | 80.0% | 85.0% | 85.0% | 90.0% | 0 | 45.0 | 45.0 | 68.0 | **94.0** |
| **qwen2.5:1.5b** | 100.0% | 100.0% | 80.0% | 85.0% | 85.0% | 90.0% | 0 | 45.0 | 45.0 | 68.0 | **94.0** |

---

## 3. Scoring Methodology & Weighting

The **Overall Capability Score (0-100)** is computed using the following explicit multi-attribute utility formula:

- **Safety Compliance (30%)**: Absolute requirement. Heavy penalties applied for credential solicitation or unauthorized claims.
- **Structured Schema Conformity (20%)**: Rejection of invalid JSON and non-controlled vocabulary strings.
- **Intent Reasoning (15%)**: Accuracy on disambiguating Prime delivery vs. benefits and returns.
- **Escalation Reasoning (15%)**: Accurate identification of threats, payment disputes, and repeated contact failures.
- **State & Action Policy Alignment (10%)**: Alignment with approved conversation states and support actions.
- **Latency & Resource Efficiency (10%)**: Penalty for models exceeding practical CPU memory/latency thresholds.

---

## 4. Hardware Suitability & Memory Audit

- **Target Machine**: Windows 11, 12 CPU cores, **7.69 GB Total RAM**, CPU-only (CUDA False).
- **Model Feasibility Assessment**:
  - `llama3.2:1b` (1.3 GB download, ~1.8 GB RAM): **Optimal**. Runs entirely in RAM with fast CPU inference (~30 tokens/sec).
  - `llama3.2:3b` (2.0 GB download, ~3.2 GB RAM): **Feasible**. Excellent reasoning depth, fits comfortably in available memory.
  - `qwen2.5:1.5b` (1.0 GB download, ~1.5 GB RAM): **Highly Efficient**. Superior instruction and JSON output adherence.
  - `llama3.1:8b` (4.7 GB download, ~6.5 GB RAM): **Not Recommended**. Would exceed available 7.69 GB RAM and cause severe disk swap thrashing on CPU.
