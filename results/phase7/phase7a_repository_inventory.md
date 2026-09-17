# Phase 7A Repository Inventory & File Categorization

> **Comprehensive Structural Audit: Complete File Inventory & Submission Eligibility**  
> *AmazonHelp Autonomous Support Agent Benchmark*

---

## 1. Executive Summary & Inventory Overview

The repository consists of **198 total files** across source code, benchmark datasets, dense retrieval vector indexes, trained models, evaluation scripts, and historical phase reports.

Total repository footprint on disk is **~940.71 MB**, heavily dominated by Phase 2 preprocessed intermediate exploration JSONL files (~909 MB).

### Category Distribution:

| Category | File Count | Disk Footprint | Submission Requirement | Action in Cleanup Plan |
| :--- | :---: | :---: | :---: | :--- |
| **`CORE_RUNTIME`** | 37 | 0.25 MB | **MANDATORY** | KEEP (Core agent runtime, safety validator, policy engine) |
| **`DATA`** | 1 | 9.35 MB | **MANDATORY** | KEEP (`amazonhelp_train_retrieval.jsonl` Train-only corpus) |
| **`EVALUATION`** | 3 | 0.96 MB | **MANDATORY** | KEEP (`amazonhelp_golden_v1_human_validated.jsonl`) |
| **`INDEX`** | 2 | 18.23 MB | **MANDATORY** | KEEP (Dense MiniLM index vectors & metadata) |
| **`MODEL`** | 1 | 1.66 MB | **MANDATORY** | KEEP (`tfidf_logreg_intent.joblib` intent model) |
| **`DOCUMENTATION`** | 20 | 0.17 MB | **MANDATORY** | KEEP (Technical protocols, taxonomy specifications) |
| **`SCRIPT`** | 26 | 0.31 MB | **MANDATORY** | KEEP (Reproducibility & verification runners) |
| **`EXPERIMENT`** | 7 | 0.04 MB | **MANDATORY** | KEEP (Baseline classifiers & evaluation utilities) |
| **`GENERATED_ARTIFACT`** | 35 | 909.14 MB | **SELECTIVE** | KEEP benchmark reports/metrics; ARCHIVE huge processed intermediate datasets |
| **`TEMPORARY`** | 53 | 0.60 MB | **EXCLUDED** | REMOVE (`__pycache__`, `.pyc` bytecode caches) |
| **`TEST`** | 0 | 0.00 MB | **OPTIONAL** | Verification scripts currently serve as test suite |

---

## 2. Detailed Component Inventory

### 2.1 Core Runtime (`src/`)

| File Path | Category | Purpose | Required for Submission? | Safe to Remove? | Reason / Rationale |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `src/config.py` | `CORE_RUNTIME` | Centralized path and parameter configuration | **YES** | NO | Single source of truth for paths, seeds, and constants |
| `src/utils/logger.py` | `CORE_RUNTIME` | Standardized application logger | **YES** | NO | Provides structured execution logging |
| `src/llm/conversation_formatter.py` | `CORE_RUNTIME` | Twitter noise sanitizer & entity cue extractor | **YES** | NO | Sanitizes handles/URLs with zero label leakage |
| `src/llm/model_client.py` | `CORE_RUNTIME` | Local Ollama REST client & offline test harness | **YES** | NO | Model execution engine for `llama3.2:1b` |
| `src/llm/schemas.py` | `CORE_RUNTIME` | Pydantic validation schemas (`LLMDecisionOutput`) | **YES** | NO | Enforces strict JSON structure and controlled vocabularies |
| `src/llm/safety_validator.py` | `CORE_RUNTIME` | Deterministic post-generation safety overrides | **YES** | NO | Blocks credentials, suppresses fabricated transactions |
| `src/llm/evidence_builder.py` | `CORE_RUNTIME` | Structured evidence package synthesizer | **YES** | NO | Assembles conversation, Phase 4 signals, and retrieval |
| `src/llm/retrieval_context.py` | `CORE_RUNTIME` | Dense retrieval context & confidence tiering | **YES** | NO | Queries MiniLM index, tags HIGH/MEDIUM/LOW confidence |
| `src/llm/agent.py` | `CORE_RUNTIME` | Phase 6B LLM-only agent implementation | **YES** | NO | Preserves Phase 6B zero-retrieval comparative harness |
| `src/llm/agent_with_retrieval.py` | `CORE_RUNTIME` | Phase 6C full support agent with retrieval | **YES** | NO | Final production agent implementation |
| `src/llm/prompts.py` | `CORE_RUNTIME` | Prompt templates & capability probes | **YES** | NO | System prompt and benchmark probes |
| `src/policy/action_policy.py` | `CORE_RUNTIME` | Deterministic action selection policy | **YES** | NO | Phase 4 deterministic action engine |
| `src/policy/escalation_policy.py` | `CORE_RUNTIME` | Deterministic escalation rule engine | **YES** | NO | Mandatory safety triggers (fraud, legal, abuse) |
| `src/policy/unified_decision_engine.py`| `CORE_RUNTIME` | Orchestrates Phase 4 deterministic decisions | **YES** | NO | Baseline decision pipeline |
| `src/state/state_tracker.py` | `CORE_RUNTIME` | Deterministic multi-turn state tracker | **YES** | NO | Tracks dialogue state trajectory |
| `src/retrieval/retriever.py` | `CORE_RUNTIME` | Dense vector similarity search engine | **YES** | NO | Cosine similarity query engine over index |
| `src/retrieval/evidence.py` | `CORE_RUNTIME` | Historical exemplar evidence extraction | **YES** | NO | Extracts structured outcomes and actions from exemplars |
| `src/retrieval/embeddings.py` | `CORE_RUNTIME` | SentenceTransformer embedding interface | **YES** | NO | Vector encoder for queries and text |
| `src/retrieval/query_builder.py` | `CORE_RUNTIME` | Contextual query formulation | **YES** | NO | Builds search queries from conversation turns |
| `src/retrieval/retrieval_policy.py` | `CORE_RUNTIME` | Phase 5 retrieval-augmented policy rules | **YES** | NO | Preserves Phase 5 experimental comparative code |

### 2.2 Datasets & Benchmark Foundations (`data/`)

| File Path | Category | Purpose | Required for Submission? | Safe to Remove? | Reason / Rationale |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `data/golden/amazonhelp_golden_v1_human_validated.jsonl` | `EVALUATION` | Authoritative 200-checkpoint golden evaluation set | **YES** | NO | 100% human-validated benchmark from Validation split |
| `data/golden/amazonhelp_golden_v1.jsonl` | `EVALUATION` | Pre-annotation rule-based golden candidates | **YES** | NO | Audit provenance: pre-annotation baseline before human review |
| `data/golden/golden_candidates.jsonl` | `EVALUATION` | Stratified candidate pool (300 candidates) | OPTIONAL | NO | Historical provenance of golden set selection |
| `data/retrieval/amazonhelp_train_retrieval.jsonl` | `DATA` | 5,502 actionable dialogues (Train partition only) | **YES** | NO | Retrieval corpus required for Phase 5 and Phase 6C |
| `data/indexes/retrieval_index.npz` | `INDEX` | 5,502 normalized 384-d MiniLM vector embeddings | **YES** | NO | Pre-computed dense vector matrix for fast retrieval |
| `data/indexes/retrieval_index_meta.json` | `INDEX` | Metadata mapping document attributes to vectors | **YES** | NO | Pre-computed metadata index (avoids rebuilding at runtime) |
| `data/processed/amazonhelp_english.jsonl` | `GENERATED_ARTIFACT`| 63,493 English AmazonHelp conversations (231 MB) | NO | YES | Intermediate artifact; can be regenerated from raw data |
| `data/processed/amazonhelp_exploration.jsonl`| `GENERATED_ARTIFACT`| Exploration split of conversations (231 MB) | NO | YES | Intermediate exploration artifact |
| `data/processed/amazonhelp_pristine_candidates.jsonl`| `GENERATED_ARTIFACT`| Pristine filtered candidate conversations (187 MB) | NO | YES | Intermediate artifact |
| `data/processed/amazonhelp_retrieval_candidates.jsonl`| `GENERATED_ARTIFACT`| Filtered candidate pool for retrieval (231 MB) | NO | YES | Intermediate artifact |
| `data/processed/amazonhelp_non_english_or_uncertain.jsonl`| `GENERATED_ARTIFACT`| Excluded low-confidence non-English turns (71 MB) | NO | YES | Audit trail artifact |

### 2.3 Models (`models/`)

| File Path | Category | Purpose | Required for Submission? | Safe to Remove? | Reason / Rationale |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `models/baselines/tfidf_logreg_intent.joblib` | `MODEL` | Trained TF-IDF vectorizer + LogisticRegression model (1.66 MB) | **YES** | NO | Phase 4 baseline model required for runtime structured signals |

### 2.4 Verification & Benchmark Scripts (`scripts/`)

| File Path | Category | Purpose | Required for Submission? | Safe to Remove? | Reason / Rationale |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `scripts/run_phase6c.py` | `SCRIPT` | Master Phase 6C benchmark runner | **YES** | NO | Reproduces Phase 6C metrics, report, and retrieval analysis |
| `scripts/verify_phase6c.py` | `SCRIPT` | 27-point automated verification suite for Phase 6C | **YES** | NO | Primary automated acceptance test suite for Phase 6C |
| `scripts/analyze_phase6c_failures.py` | `SCRIPT` | Empirical failure analyzer for Phase 6C | **YES** | NO | Extracts and documents 7 empirical failure case studies |
| `scripts/compare_phase4_phase6.py` | `SCRIPT` | Cross-phase comparison inspector | **YES** | NO | Prints standardized multi-phase comparison table |
| `scripts/run_phase6b.py` | `SCRIPT` | Master Phase 6B benchmark runner | **YES** | NO | Reproduces Phase 6B LLM-only benchmark |
| `scripts/verify_phase6b.py` | `SCRIPT` | 16-point automated verification suite for Phase 6B | **YES** | NO | Acceptance test suite for Phase 6B |
| `scripts/run_phase5.py` | `SCRIPT` | Phase 5 retrieval benchmark runner | **YES** | NO | Reproduces Phase 5 retrieval-only benchmark |
| `scripts/verify_phase5.py` | `SCRIPT` | Automated verification suite for Phase 5 | **YES** | NO | Acceptance test suite for Phase 5 |
| `scripts/run_phase4.py` | `SCRIPT` | Phase 4 baseline benchmark runner | **YES** | NO | Reproduces Phase 4 baseline metrics |
| `scripts/verify_phase4.py` | `SCRIPT` | Automated verification suite for Phase 4 | **YES** | NO | Acceptance test suite for Phase 4 |
| `scripts/annotate_golden.py` | `SCRIPT` | Golden human review CLI workflow tool | **YES** | NO | Audit provenance tool used for Phase 3 human validation |
| `scripts/build_retrieval_index.py` | `SCRIPT` | Vector index builder for Train corpus | **YES** | NO | Allows rebuilding dense index from Train corpus |
| `scripts/sample_golden_candidates.py` | `SCRIPT` | Golden dataset sampling algorithm | **YES** | NO | Provenance tool for Golden dataset construction |

### 2.5 Documentation (`docs/`)

| File Path | Category | Purpose | Required for Submission? | Safe to Remove? | Reason / Rationale |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `docs/phase6c_protocol.md` | `DOCUMENTATION` | Technical protocol for Phase 6C | **YES** | NO | Documents final 5-layer architecture and decision rules |
| `docs/phase6b_llm_agent_protocol.md` | `DOCUMENTATION` | Technical protocol for Phase 6B | **YES** | NO | Documents LLM-only experiment boundary |
| `docs/phase5_retrieval_protocol.md` | `DOCUMENTATION` | Technical protocol for Phase 5 | **YES** | NO | Documents dense retrieval index & evaluation design |
| `docs/phase4_baseline_protocol.md` | `DOCUMENTATION` | Technical protocol for Phase 4 | **YES** | NO | Documents non-LLM baseline formulation |
| `docs/phase4_decision_policy.md` | `DOCUMENTATION` | Deterministic action & escalation rules | **YES** | NO | Complete policy engine specification |
| `docs/golden_dataset_protocol.md` | `DOCUMENTATION` | Golden evaluation benchmark specification | **YES** | NO | Documents checkpoint construction & sampling criteria |
| `docs/intent_taxonomy_proposal.md` | `DOCUMENTATION` | 10-class intent taxonomy specification | **YES** | NO | Formal definitions and boundaries for all 10 intents |
| `docs/conversation_state_proposal.md`| `DOCUMENTATION` | 8-state conversation tracking specification | **YES** | NO | Formal state machine transition rules |
| `docs/agent_action_proposal.md` | `DOCUMENTATION` | 8-action vocabulary specification | **YES** | NO | Controlled action definitions and usage constraints |
| `docs/autonomy_policy.md` | `DOCUMENTATION` | Safety and autonomy boundary specification | **YES** | NO | Mandatory boundaries (no fake refunds, no credentials) |
| `docs/brand_selection_report.md` | `DOCUMENTATION` | Phase 2 multi-brand audit report | **YES** | NO | Justifies selecting AmazonHelp over candidate brands |

### 2.6 Results & Reports (`results/`)

| File Path | Category | Purpose | Required for Submission? | Safe to Remove? | Reason / Rationale |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `results/phase6/phase6c_metrics.json` | `GENERATED_ARTIFACT`| Phase 6C benchmark metrics payload | **YES** | NO | Structured JSON metrics for Phase 6C |
| `results/phase6/phase6c_report.md` | `GENERATED_ARTIFACT`| Phase 6C comprehensive evaluation report | **YES** | NO | Headline benchmark report for Phase 6C |
| `results/phase6/phase6c_failure_analysis.md` | `GENERATED_ARTIFACT`| Phase 6C empirical failure diagnostics | **YES** | NO | 7 detailed empirical failure case studies |
| `results/phase6/phase6c_retrieval_analysis.md` | `GENERATED_ARTIFACT`| Phase 6C dense retrieval audit & help/harm | **YES** | NO | Analysis of retrieval distribution and effect |
| `results/phase6/phase6b_metrics.json` | `GENERATED_ARTIFACT`| Phase 6B benchmark metrics payload | **YES** | NO | Structured JSON metrics for Phase 6B |
| `results/phase6/phase6b_report.md` | `GENERATED_ARTIFACT`| Phase 6B benchmark evaluation report | **YES** | NO | Comparative report for LLM-only experiment |
| `results/phase6/phase6b_failure_analysis.md` | `GENERATED_ARTIFACT`| Phase 6B failure diagnostics | **YES** | NO | 5 detailed empirical failure case studies for 6B |
| `results/phase5_retrieval_metrics.json` | `GENERATED_ARTIFACT`| Phase 5 retrieval metrics payload | **YES** | NO | Structured JSON metrics for Phase 5 |
| `results/phase5_retrieval_report.md` | `GENERATED_ARTIFACT`| Phase 5 retrieval benchmark report | **YES** | NO | Frozen comparative report for Phase 5 |
| `results/phase4_baseline_report.md` | `GENERATED_ARTIFACT`| Phase 4 baseline evaluation report | **YES** | NO | Frozen comparative report for Phase 4 |
| `results/golden_human_annotation_report.md`| `GENERATED_ARTIFACT`| Phase 3 human golden validation report | **YES** | NO | Documents 100% human review of the 200 checkpoints |

### 2.7 Temporary & Intermediate Files (`__pycache__/`, etc.)

| Directory / Pattern | Category | Purpose | Required for Submission? | Safe to Remove? | Reason / Rationale |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `**/__pycache__/*` (53 files) | `TEMPORARY` | Python compiled bytecode (`.pyc`) | **NO** | **YES** | Ephemeral bytecode caches; auto-regenerated by Python |
| `results/prototype_reconstruction.json` | `GENERATED_ARTIFACT` | Phase 2 prototype sample (373 KB) | NO | YES | Ephemeral prototype reconstruction test |
| `results/reconstructed_conversations_sample.json`| `GENERATED_ARTIFACT`| Phase 2 sample (51 KB) | NO | YES | Ephemeral prototype test sample |
