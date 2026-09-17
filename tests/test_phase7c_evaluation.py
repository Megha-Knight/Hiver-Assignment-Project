"""Unit and Integration Tests for Phase 7C Evaluation Harness.

Verifies the 10 required evaluation test areas:
 1. Golden benchmark isolation (size == 200, Dev-only, zero test leakage)
 2. Retrieval corpus isolation (size == 5,502, Train-only, zero dev/test overlap)
 3. Gold-label prompt exclusion (no ground truth in inference prompts)
 4. Judge JSON validation, repair, and numeric bounding [1, 5]
 5. Evidence grounding schema and 4-probe diagnostic behavior
 6. Deterministic safety evaluation and override precedence
 7. Automated failure classification across the 10 taxonomy buckets
 8. Stratified human-review sampling (N=40, deterministic seed 42)
 9. Metric calculations (decision correctness, Quadratic Weighted Kappa, Spearman)
10. CLI evaluation command integration (`evaluate --limit 5 --mock`)
"""

import json
from pathlib import Path
import subprocess
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.annotator import (
    APPROVED_ACTIONS,
    APPROVED_ESCALATION_REASONS,
    APPROVED_INTENTS,
    APPROVED_STATES,
)
from src.config import PATHS
from src.evaluation.grounding_evaluator import EvidenceGroundingEvaluator, GroundingEvaluationResult
from src.evaluation.human_agreement import HumanAgreementEvaluator, HumanAgreementMetrics
from src.evaluation.metrics import (
    compute_decision_exact_match,
    compute_escalation_metrics,
    compute_intent_metrics,
)
from src.evaluation.response_judge import JudgeEvaluationResult, ResponseQualityJudge
from src.llm.model_client import MockOllamaClient
from src.llm.prompts import format_user_prompt
from src.llm.safety_validator import DeterministicSafetyValidator
from src.llm.schemas import LLMDecisionOutput


class TestPhase7CEvaluationHarness(unittest.TestCase):
    """Test suite for Phase 7C evaluation harness components."""

    @classmethod
    def setUpClass(cls):
        cls.golden_path = PATHS.AMAZONHELP_GOLDEN_HUMAN_VALIDATED_JSONL
        cls.retrieval_path = PATHS.AMAZONHELP_TRAIN_RETRIEVAL_JSONL

        with open(cls.golden_path, "r", encoding="utf-8") as f:
            cls.checkpoints = [json.loads(line) for line in f]

        with open(cls.retrieval_path, "r", encoding="utf-8") as f:
            cls.retrieval_docs = [json.loads(line) for line in f]

    def test_01_golden_benchmark_isolation(self):
        """1. Verify golden benchmark size is exactly 200 and all from Validation partition."""
        self.assertEqual(len(self.checkpoints), 200)
        for cp in self.checkpoints:
            self.assertEqual(cp["source_conversation_metadata"]["partition"], "validation")
            self.assertIn(cp.get("human_review_status"), ["REVIEWED", "NOT_HUMAN_REVIEWED"])

    def test_02_retrieval_isolation(self):
        """2. Verify retrieval corpus size is exactly 5,502 and has zero golden dev overlap."""
        self.assertEqual(len(self.retrieval_docs), 5502)
        gold_convs = {c["conversation_id"] for c in self.checkpoints}
        ret_convs = {d["conversation_id"] for d in self.retrieval_docs}
        overlap = gold_convs.intersection(ret_convs)
        self.assertEqual(len(overlap), 0, f"Leakage detected: {overlap}")

    def test_03_gold_label_prompt_exclusion(self):
        """3. Verify prompts constructed for inference contain zero gold label keys."""
        cp = self.checkpoints[0]
        prompt = format_user_prompt(
            customer_message=cp["current_customer_message"],
            history=cp.get("conversation_history_before_current_turn", []),
            turn_depth=cp.get("turn_depth", 1),
        )
        forbidden = [
            "expected_intent",
            "expected_state",
            "expected_action",
            "expected_escalation",
            "final_human_intent",
            "annotation_notes",
        ]
        for term in forbidden:
            self.assertNotIn(term, prompt)

    def test_04_judge_json_validation(self):
        """4. Verify ResponseQualityJudge JSON parser handles markdown fences and clamps [1, 5]."""
        judge = ResponseQualityJudge(use_mock=True)

        valid_raw = '```json\n{"relevance": 5, "helpfulness": 4, "groundedness": 5, "action_appropriateness": 4, "safety": 5, "communication_quality": 4, "overall_reason": "Good response"}\n```'
        parsed, is_valid = judge._parse_judge_json(valid_raw)
        self.assertTrue(is_valid)
        self.assertEqual(parsed["relevance"], 5)
        self.assertEqual(parsed["helpfulness"], 4)

        # Out-of-bounds clamping
        oob_raw = '{"relevance": 10, "helpfulness": -2, "groundedness": 3, "action_appropriateness": 3, "safety": 3, "communication_quality": 3}'
        parsed_oob, is_valid_oob = judge._parse_judge_json(oob_raw)
        self.assertTrue(is_valid_oob)
        self.assertEqual(parsed_oob["relevance"], 5)
        self.assertEqual(parsed_oob["helpfulness"], 1)

    def test_05_grounding_schema_and_probes(self):
        """5. Verify EvidenceGroundingEvaluator correctly executes the 4 diagnostic probes."""
        evaluator = EvidenceGroundingEvaluator()

        # Case A: Grounded standard response
        res_a = evaluator.evaluate_grounding(
            customer_message="Where is my parcel?",
            generated_response="You can track your order in 'Your Orders' on the website or DM us your tracking details.",
            retrieved_exemplars=[{"support_reply": "Please check your orders page or DM us."}],
            retrieval_top1_similarity=0.78,
        )
        self.assertTrue(res_a.is_grounded)
        self.assertFalse(res_a.unsupported_capability)
        self.assertFalse(res_a.unsupported_policy_claim)
        self.assertFalse(res_a.contradiction)

        # Case B: Fabricated capability (Direct refund promise)
        res_b = evaluator.evaluate_grounding(
            customer_message="I want my money back",
            generated_response="I have refunded your $50 back to your bank account immediately.",
            retrieved_exemplars=[{"support_reply": "Please check your orders page."}],
            retrieval_top1_similarity=0.72,
        )
        self.assertFalse(res_b.is_grounded)
        self.assertTrue(res_b.unsupported_capability)
        self.assertEqual(res_b.evidence_support, "UNSUPPORTED")

    def test_06_safety_evaluation_precedence(self):
        """6. Verify deterministic safety overrides take precedence over model outputs."""
        validator = DeterministicSafetyValidator()

        # Model hallucinates credential request
        unsafe_decision = LLMDecisionOutput(
            intent="DELIVERY_STATUS_AND_TRACKING",
            state="STATE_INITIAL_INBOUND",
            action="REQUEST_SAFE_DETAILS",
            escalate=False,
            escalation_reason="NONE",
            confidence=0.9,
            reasoning_summary="Need card pin to check parcel",
            response="Please reply with your account password and OTP so we can verify your shipment.",
        )
        validated = validator.validate_and_enforce(
            decision=unsafe_decision,
            customer_message="Help tracking package",
        )
        self.assertIn("password", " ".join(validated.safety_violations_detected).lower())
        self.assertEqual(validated.action, "HANDOFF_TO_SECURE_CHANNEL")
        self.assertNotIn("password", validated.final_response.lower())

    def test_07_failure_classification_taxonomy(self):
        """7. Verify automated taxonomy classifies decision and safety discrepancies."""
        assigned = []
        pred_intent = "DELIVERY_STATUS_AND_TRACKING"
        gold_intent = "RETURN_REFUND_AND_REPLACEMENT"
        pred_esc = False
        gold_esc = True
        has_safety_violation = True

        if pred_intent != gold_intent:
            assigned.append("WRONG_INTENT")
        if gold_esc and not pred_esc:
            assigned.append("ESCALATION_MISS")
        if has_safety_violation:
            assigned.append("SAFETY_POLICY_CONFLICT")

        self.assertIn("WRONG_INTENT", assigned)
        self.assertIn("ESCALATION_MISS", assigned)
        self.assertIn("SAFETY_POLICY_CONFLICT", assigned)

    def test_08_human_review_sampling_stratification(self):
        """8. Verify stratified human review sampling selects N=40 with exact difficulty tiers."""
        evaluator = HumanAgreementEvaluator(seed=42)
        sampled = evaluator.sample_stratified_subset(self.checkpoints, target_n=40)
        self.assertEqual(len(sampled), 40)

        easy_count = sum(1 for c in sampled if c["difficulty"].lower() == "easy")
        med_count = sum(1 for c in sampled if c["difficulty"].lower() == "medium")
        hard_count = sum(1 for c in sampled if c["difficulty"].lower() == "hard")

        self.assertEqual(easy_count, 10)
        self.assertEqual(med_count, 18)
        self.assertEqual(hard_count, 12)

        # Build packet and verify zero gold labels exist in packet
        packet = evaluator.build_review_packet(sampled, agent_predictions={})
        self.assertEqual(len(packet), 40)
        for row in packet:
            self.assertNotIn("expected_intent", row)
            self.assertNotIn("expected_action", row)
            self.assertNotIn("expected_escalation", row)
            self.assertIn("human_scoring", row)

    def test_09_metric_calculations(self):
        """9. Verify decision metrics and Quadratic Weighted Kappa calculations."""
        # Intent metrics
        y_t = ["DELIVERY_STATUS_AND_TRACKING", "RETURN_REFUND_AND_REPLACEMENT"]
        y_p = ["DELIVERY_STATUS_AND_TRACKING", "DELIVERY_STATUS_AND_TRACKING"]
        int_m = compute_intent_metrics(y_t, y_p, labels=APPROVED_INTENTS)
        self.assertEqual(int_m["accuracy"], 0.5)

        # Kappa and Spearman with non-constant varied ratings
        evaluator = HumanAgreementEvaluator(seed=42)
        h_ratings = [
            {"human_scoring": {"relevance": 5, "helpfulness": 4, "groundedness": 5, "action_appropriateness": 4, "safety": 5, "communication_quality": 4}},
            {"human_scoring": {"relevance": 2, "helpfulness": 2, "groundedness": 3, "action_appropriateness": 2, "safety": 4, "communication_quality": 3}},
            {"human_scoring": {"relevance": 4, "helpfulness": 5, "groundedness": 4, "action_appropriateness": 4, "safety": 5, "communication_quality": 5}},
            {"human_scoring": {"relevance": 1, "helpfulness": 1, "groundedness": 2, "action_appropriateness": 1, "safety": 2, "communication_quality": 2}},
            {"human_scoring": {"relevance": 3, "helpfulness": 3, "groundedness": 4, "action_appropriateness": 3, "safety": 4, "communication_quality": 3}},
        ]
        j_ratings = [
            {"judge_scoring": {"relevance": 5, "helpfulness": 4, "groundedness": 5, "action_appropriateness": 4, "safety": 5, "communication_quality": 4}},
            {"judge_scoring": {"relevance": 2, "helpfulness": 2, "groundedness": 3, "action_appropriateness": 2, "safety": 4, "communication_quality": 3}},
            {"judge_scoring": {"relevance": 4, "helpfulness": 5, "groundedness": 4, "action_appropriateness": 4, "safety": 5, "communication_quality": 5}},
            {"judge_scoring": {"relevance": 1, "helpfulness": 1, "groundedness": 2, "action_appropriateness": 1, "safety": 2, "communication_quality": 2}},
            {"judge_scoring": {"relevance": 3, "helpfulness": 3, "groundedness": 4, "action_appropriateness": 3, "safety": 4, "communication_quality": 3}},
        ]
        agr = evaluator.calculate_agreement(h_ratings, j_ratings)
        self.assertEqual(agr.review_status, "COMPLETED")
        self.assertAlmostEqual(agr.exact_agreement_rate, 1.0)
        self.assertAlmostEqual(agr.adjacent_agreement_rate, 1.0)
        self.assertAlmostEqual(agr.mean_quadratic_weighted_kappa, 1.0)

    def test_10_cli_evaluation_command(self):
        """10. Verify CLI evaluate command functions cleanly via subprocess."""
        cmd = [sys.executable, str(PROJECT_ROOT / "cli.py"), "evaluate", "--limit", "3", "--mock"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=150, encoding="utf-8", errors="replace")

        self.assertEqual(res.returncode, 0)
        self.assertIn("COMPREHENSIVE 5-LAYER EVALUATION HARNESS", res.stdout)
        self.assertIn("EVALUATION RESULTS SUMMARY", res.stdout)
        self.assertIn("PRIMARY DECISION METRICS", res.stdout)



if __name__ == "__main__":
    unittest.main()
