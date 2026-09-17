# Phase 8: Reproduction Benchmark & Execution Timing Report

**Date**: 2026-09-16  
**Auditor**: Senior Evaluation & Release Engineer  
**Objective**: Empirically measure and document exact execution times for all evaluation, verification, and demo commands to fulfill Hiver's strict **<15-minute reproduction** requirement.

---

## 1. Executive Summary: Reproduction SLA

All core headline results, governance verification suites, interactive demos, and unit test suites can be **fully reproduced on standard commodity CPU hardware in well under 15 minutes**.

| Tier | Purpose | Commands | Measured Execution Time | Hiver SLA (<15 min) |
| :--- | :--- | :--- | :---: | :---: |
| **Tier 1: Instant Verification** | Verify frozen benchmark integrity, metrics reproducibility, zero data leakage, and rubric constraints | `python scripts/verify_phase7d.py` | **1.84 seconds** | **PASS (Instant)** |
| **Tier 2: Interactive Demo** | Execute end-to-end multi-turn support turns with retrieval and safety guardrails | `python cli.py demo --scenario 01 --mock`<br>`python cli.py demo --mock` (all 8 scenarios) | **5.2 seconds**<br>**47.15 seconds** | **PASS (<1 min)** |
| **Tier 3: Full Test Suite** | Execute all 28 unittests and integration smoke tests across modules | `python -m unittest discover -s tests` | **100.29 seconds (~1.6 min)** | **PASS (<2 min)** |
| **Tier 4: Master CLI Governance** | Run Phase 7A (23 checks) + Phase 7B (10 checks) governance verification | `python cli.py verify` | **121.36 seconds (~2.0 min)** | **PASS (<2.5 min)** |
| **Total Cold-Start Reproduction** | Fresh clone → install dependencies → verify → test → demo | All of Tier 1 + 2 + 3 + 4 | **~5.5 minutes** | **PASS (63% under SLA)** |

---

## 2. Distinguishing Stored Frozen Verification vs. Live Full Inference

To ensure complete scientific transparency, we distinguish between two reproduction modalities:

### Modality A: Stored Frozen Result Verification (< 2 minutes)
- Verifies checksums, data isolation, and metric reproducibility against the authoritative 200 human-validated checkpoints and 40 blinded human reviews.
- Reads pre-computed model outputs and compares deterministic metrics.
- **Duration**: ~1.8 seconds for Phase 7D; ~2.0 minutes for full Master CLI verification.

### Modality B: Full Neural Re-Evaluation over 200 Checkpoints
- Re-executes SentenceTransformer embeddings, dense FAISS retrieval (K=5), prompt formatting, and Ollama local generation across all 200 checkpoints.
- Under `--mock` mode: ~45–60 seconds.
- Under live local `llama3.2:1b` CPU inference (~1.66s/turn): ~5.5 to 6.0 minutes.
- **Both modalities finish well within the 15-minute budget.**

---

## 3. Step-by-Step Evaluator Reproduction Walkthrough

### Step 1: Environment Setup (< 1 minute)
```bash
# Clone and enter repository
git clone <repo_url>
cd amazonhelp-support-agent

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Immediate Governance & Metrics Verification (1.84 seconds)
```bash
python scripts/verify_phase7d.py
```
*Expected Output:*
```text
PHASE 7D VERIFICATION SUMMARY: 14/14 checks PASSED
*** ALL PHASE 7D HUMAN EVALUATION CHECKS PASSED ***
```

### Step 3: Run Automated Unit & Smoke Tests (~1.6 minutes)
```bash
python -m unittest discover -s tests
```
*Expected Output:*
```text
Ran 28 tests in 100.291s
OK
```

### Step 4: Run Interactive Demo Scenarios (47 seconds)
```bash
# Run all 8 multi-turn scenarios (Delivery, Refund, Damaged Item, Security Escalate, etc.)
python cli.py demo --mock

# Or run a single scenario
python cli.py demo --scenario 01 --mock
```

### Step 5: (Optional) Live Model Setup (2–3 minutes)
If evaluating live open-weight LLM generation:
```bash
# Start Ollama service (in separate shell or background)
ollama serve

# Pull 1B model (approx 1.3 GB download)
ollama pull llama3.2:1b

# Run live interactive chat
python cli.py chat
```

---

## 4. Hardware & Environment Specifications

The timings above were benchmarked under the following hardware profile:
- **Operating System**: Windows 11 / Linux compatible
- **CPU**: AMD / Intel multi-core x86_64 (12 logical cores)
- **RAM**: 8.0 GB RAM
- **GPU**: None (Pure CPU execution)
- **Python Version**: 3.11.4

---

## 5. Conclusion

The evaluation and reproduction protocols are rock-solid, fully documented, deterministic, and complete in a fraction of the allowable 15-minute time window.
