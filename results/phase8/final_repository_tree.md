# Phase 8: Final Repository Directory Tree

**Date**: 2026-09-16  
**Auditor**: Senior Evaluation & Release Engineer  
**Status**: CLEAN, AUDITED, RELEASE-READY

---

## 1. Directory Structure

```text
amazonhelp-support-agent/
│
├── README.md                                    # 2-minute executive & technical overview
├── cli.py                                       # Unified production CLI entrypoint
├── requirements.txt                             # Pinned Python dependencies (11 core packages)
├── .env.example                                 # Sanitized configuration template
├── .gitignore                                   # Strict exclusion rules (caches, venvs, 930MB raw data)
│
├── data/
│   ├── golden/                                  # Frozen Golden Benchmarks
│   │   ├── amazonhelp_golden_v1_human_validated.jsonl # 200 human-validated Dev checkpoints
│   │   ├── golden_split_summary.json            # Stratification metrics (49 Easy / 91 Med / 60 Hard)
│   │   └── golden_validation_rubric.md          # Checkpoint validation criteria
│   ├── retrieval/                               # Train-Only Retrieval Corpus
│   │   └── amazonhelp_train_retrieval_corpus.jsonl # 5,502 verified resolution dialogues
│   ├── indexes/                                 # Pre-Computed Vector Embeddings
│   │   └── retrieval_index.npz                  # 5,502 x 384 MiniLM embedding matrix (18.6 MB)
│   └── evaluation/                              # Blinded Human Evaluation Data
│       ├── human_review_packet_n40.jsonl        # Blinded N=40 review packet (zero gold labels)
│       └── human_reviews_n40.jsonl              # 40 completed human evaluation records
│
├── models/
│   └── baselines/
│       └── tfidf_logreg_intent.joblib           # Trained Phase 4 TF-IDF model weights (1.7 MB)
│
├── src/                                         # Core Production Agent Implementation
│   ├── config.py                                # Dynamic root & path configuration singleton
│   ├── agent/                                   # Conversation Management & Scenarios
│   │   ├── conversation_manager.py              # Multi-turn turn pipeline coordinator
│   │   └── demo_scenarios.py                    # 8 pre-configured end-to-end support scenarios
│   ├── baselines/                               # Non-LLM Baseline Models
│   │   └── tfidf_logreg.py                      # TF-IDF + Logistic Regression classifier
│   ├── state/                                   # Multi-Turn State Tracking
│   │   └── tracker.py                           # 8-state dialogue state transition machine
│   ├── policy/                                  # Operational & Escalation Policies
│   │   ├── action_policy.py                     # 8-action decision mapper
│   │   ├── escalation_policy.py                 # Deterministic dual-trigger escalation logic
│   │   └── policy_engine.py                     # Unified policy coordinator
│   ├── retrieval/                               # Historical Retrieval Engine
│   │   ├── embeddings.py                        # SentenceTransformer all-MiniLM-L6-v2 embedder
│   │   ├── retriever.py                         # Dense vector cosine similarity search
│   │   └── corpus.py                            # Corpus loader and validator
│   ├── llm/                                     # Generative LLM & Guardrails
│   │   ├── agent_with_retrieval.py              # LLM prompt injection & response generator
│   │   ├── model_client.py                      # Ollama REST client & Mock simulation client
│   │   ├── schemas.py                           # Pydantic structured output models
│   │   ├── evidence_builder.py                  # Exemplar formatting & prompt assembly
│   │   └── safety_layer.py                      # Deterministic credential/refund guardrails
│   ├── evaluation/                              # Evaluation Harness & Metrics
│   │   ├── evaluation_orchestrator.py           # 5-layer automated evaluation coordinator
│   │   ├── human_agreement.py                   # Stratified sampling & agreement statistics
│   │   └── response_judge.py                    # Automated LLM-as-a-judge rubric engine
│   └── utils/                                   # Logging & telemetry helpers
│
├── tests/                                       # Automated Unit & Smoke Tests
│   ├── test_production_smoke.py                 # 12 agent integration & latency tests
│   ├── test_phase7c_evaluation.py               # 10 automated evaluator & rubric tests
│   └── test_phase7d_evaluation.py               # 6 human agreement & blinding tests
│
├── scripts/                                     # Verification & Governance Scripts
│   ├── verify_phase7a.py                        # Phase 7A baseline governance (23 checks)
│   ├── verify_phase7b.py                        # Phase 7B production agent verification (10 checks)
│   ├── verify_phase7c.py                        # Phase 7C evaluation harness verification (12 checks)
│   ├── verify_phase7d.py                        # Phase 7D master human validation gate (14 checks)
│   ├── verify_phase8.py                         # Phase 8 master release gate (24 checks)
│   └── run_human_review.py                      # CLI tool for blinded human response review
│
├── docs/                                        # Technical Specifications & Interview Briefings
│   ├── final_report.md                          # 6-page comprehensive technical report
│   ├── decision_log.md                          # 15 definitive engineering decisions
│   ├── demo_script.md                           # 3–5 minute technical demo walkthrough
│   ├── interview_cheat_sheet.md                 # 30 technical questions & defensible answers
│   ├── production_observability.md              # Production scaling & multi-tier telemetry plan
│   ├── brand_selection_report.md                # Phase 2 brand selection analysis
│   ├── intent_taxonomy_proposal.md              # 10-intent taxonomy design
│   ├── conversation_state_proposal.md           # 8-state dialogue model design
│   └── golden_dataset_protocol.md               # 200-checkpoint validation protocol
│
└── results/                                     # Official Experimental Metrics & Audits
    ├── phase6/                                  # Phase 6A/6B/6C experimental telemetry
    ├── phase7/                                  # Phase 7A/7B/7C/7D reports & audits
    │   ├── phase7d_human_review_report.md       # Master 14-section human review report
    │   ├── phase7d_final_evaluation_summary.md  # Headline scorecard & misleading metric analysis
    │   ├── phase7d_human_agreement.json         # Raw agreement statistics (QWK, Spearman, MAD)
    │   └── phase7d_human_disagreements.md       # Case analysis of 14 flagged discrepancies
    └── phase8/                                  # Phase 8 Release Hardening Artifacts
        ├── security_audit.md                    # Secrets & credential audit (Zero secrets found)
        ├── fresh_clone_audit.md                 # Path portability & fresh-clone test
        ├── repository_inventory.md              # Repository artifact footprint & isolation
        ├── reproduction_benchmark.md            # Execution timings (<15-minute SLA)
        ├── final_test_report.md                 # 28/28 test execution log
        ├── final_repository_tree.md             # This document
        ├── release_checklist.md                 # 24-item release verification checklist
        └── phase8_final_audit.md                # Quality gate certification
```

---

## 2. Tracked Asset Summary

- **Total Tracked Code & Asset Footprint**: ~32.8 MB (including vector index and model weights).
- **Excluded Datasets**: ~930 MB intermediate dataset caches safely excluded via `.gitignore`.
- **Zero Third-Party Cloud Requirements**: Completely self-contained and runnable offline.
