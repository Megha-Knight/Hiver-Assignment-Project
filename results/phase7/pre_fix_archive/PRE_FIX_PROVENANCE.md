# Pre-Fix Evaluation Artifacts Provenance Archive

**Archived At (UTC)**: 2026-09-17T08:52:14.326971+00:00  
**Preserved Files**:
- `phase7c_decision_metrics.json`
- `phase7c_response_quality.json`
- `phase7c_grounding_metrics.json`
- `phase7c_safety_metrics.json`
- `phase7c_failure_analysis.json`
- `phase7c_evaluation_report.md`
- `human_review_packet_n40.jsonl`

---

## Provenance Distinction

### PRE-FIX (Archived in this directory):
- **Generation Phase**: Phase 7C initial run (pre-correction).
- **Retrieval Schema**: Contained mismatched keys (`customer_text`, `support_reply`, `similarity` vs canonical `customer_problem_summary`, `support_response`, `similarity_score`).
- **Grounding Evaluator State**: Evaluator looked up `support_reply` (empty), defaulting missing evidence to `PARTIAL` and reporting an invalid `99.50%` grounding rate.
- **Mock LLM Behavior**: Overwrote active customer complaint actions with retrieved historical actions (`action = top_ret_act`) whenever similarity >= 0.70, forcing `CONFIRM_RESOLUTION` ("glad your issue has been resolved") onto unresolved complaints.
- **Validity**: **NOT VALID** for headline performance or production claims. Preserved strictly for forensic auditability and regression tracking.

### POST-FIX (Active files in `results/phase7/` and `data/evaluation/`):
- **Generation Phase**: Controlled post-fix re-evaluation.
- **Retrieval Schema**: Unified canonical schema with backward-compatible legacy aliasing.
- **Grounding Evaluator State**: Evaluates canonical `support_response`; strictly marks empty/missing evidence as `UNSUPPORTED` / not grounded; enforces capability checks.
- **Mock LLM Behavior**: Context-specific, policy-bounded deterministic response generation without historical action overrides.
- **Human Benchmark Integrity**: `data/evaluation/genuine_human_reviews_n40.jsonl` preserved untouched.
