# Phase 7A — Command-Line Operations & Execution Playbook

**Audit Date:** 2026-09-16  
**Auditor:** Automated Take-Home Evaluation Hardening Agent  
**Status:** COMPLETE & DOCUMENTED

---

## 1. Executive Summary

This document consolidates the operational commands required to setup the environment, configure local AI models, run evaluations, execute verification suites, and invoke the customer-support decision agent.

---

## 2. Environment Setup

### 2.1 Python Virtual Environment
Prerequisite: Python 3.11+ installed.

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux / macOS:
source venv/bin/activate
# On Windows (cmd / PowerShell):
venv\Scripts\activate

# Install core production dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Local Model Setup (Ollama & llama3.2:1b)

The agent utilizes the open-weight `llama3.2:1b` model running locally via Ollama. Cloud APIs and API keys are strictly not required.

### 3.1 Install & Start Ollama
1. Download Ollama from [https://ollama.com/download](https://ollama.com/download).
2. Start the background service:
   ```bash
   ollama serve
   ```
3. Pull the required 1-billion parameter model:
   ```bash
   ollama pull llama3.2:1b
   ```
4. Verify local model availability:
   ```bash
   ollama list
   ```

*Note:* If Ollama is not installed or running, the pipeline automatically falls back to `MockOllamaClient` to allow offline testing and smoke verification without crashing.

---

## 4. Data Preparation Commands

### 4.1 Default Mode (Zero Ingestion Required)
The repository comes pre-packaged with all required evaluation artifacts:
* `data/golden/amazonhelp_golden_v1_human_validated.jsonl` (200 checkpoints)
* `data/retrieval/amazonhelp_train_retrieval.jsonl` (5,502 dialogues)
* `data/indexes/retrieval_index.npz` (MiniLM dense embeddings)
* `models/baselines/tfidf_logreg_intent.joblib` (Trained baseline)

No dataset download is required for evaluation.

### 4.2 Rebuilding from Raw Twitter CSV (Optional)
If raw Kaggle `twcs.csv` is present at `data/raw/twcs.csv`:
```bash
# 1. Reconstruct multi-turn conversation trees for AmazonHelp
python scripts/run_reconstruction.py

# 2. Run language detection, partition split, and candidate sampling
python scripts/run_phase2_pipeline.py

# 3. Build dense vector index over 5,502 Train conversations
python scripts/build_retrieval_index.py
```

---

## 5. Verification Commands

Run the automated verification scripts to validate integrity across each project milestone:

```bash
# Verify Phase 2 dataset properties and conversation schema
python scripts/verify_phase2.py

# Verify Phase 3 Train retrieval corpus, 200 Golden checkpoints, and zero-leakage
python scripts/verify_phase3_retrieval.py

# Verify Phase 4 TF-IDF baseline and deterministic policy rules
python scripts/verify_phase4.py

# Verify Phase 5 historical retrieval engine and K=5 metrics
python scripts/verify_phase5.py

# Verify Phase 6A Ollama client, JSON schema validation, and safety probes
python scripts/verify_phase6a.py

# Verify Phase 6B LLM-only agent metrics and safety enforcement
python scripts/verify_phase6b.py

# Verify Phase 6C Full Tri-Layer Architecture (LLM + Retrieval + Policy)
python scripts/verify_phase6c.py

# Verify Phase 7A Repository Audit, frozen integrity, and cleanliness
python scripts/verify_phase7a.py
```

---

## 6. Running Evaluations

To execute the benchmark evaluations against the 200 human-validated checkpoints:

```bash
# Evaluate Phase 4 Deterministic Baseline
python scripts/run_phase4.py

# Evaluate Phase 5 Retrieval Augmentation (K=1, 3, 5, 10 sweep)
python scripts/run_phase5.py

# Evaluate Phase 6B LLM-Only Agent (Zero Retrieval)
python scripts/run_phase6b.py

# Evaluate Phase 6C Tri-Layer Agent (Full Architecture, K=5)
python scripts/run_phase6c.py

# Display cross-phase comparative performance matrix
python scripts/compare_phase4_phase6.py
```

---

## 7. Running the Customer Support Agent

The production agent can be invoked via Python directly:

```python
from src.llm.agent_with_retrieval import LLMAgentWithRetrieval

# Initialize Tri-Layer Agent (MiniLM dense retrieval + llama3.2:1b + deterministic safety)
agent = LLMAgentWithRetrieval()

# Process an inbound customer query
result = agent.process_turn(
    customer_message="My package was supposed to arrive yesterday, but it says delayed. Can you check where it is?",
    turn_depth=1,
)

print(f"Classified Intent   : {result.intent}")
print(f"Dialogue State      : {result.state}")
print(f"Selected Action     : {result.action}")
print(f"Escalated           : {result.escalate} (Reason: {result.escalation_reason})")
print(f"Customer Response   : {result.final_response}")
print(f"Deterministic Safety: {'PASSED' if result.is_safe else 'OVERRIDDEN'}")
```
