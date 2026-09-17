"""Unit Test Suite for Phase 7D Human Response-Quality Validation & Hardening."""

import json
from pathlib import Path
import unittest

from src.config import PATHS
from src.evaluation.human_agreement import HumanAgreementEvaluator, HumanAgreementMetrics
from src.evaluation.response_judge import ResponseQualityJudge
from src.llm.model_client import MockOllamaClient


class TestPhase7DHumanEvaluation(unittest.TestCase):
    """Test suite verifying all Phase 7D human evaluation components and artifacts."""

    def setUp(self):
        self.packet_path = PATHS.DATA_DIR / "evaluation" / "human_review_packet_n40.jsonl"
        self.reviews_path = PATHS.DATA_DIR / "evaluation" / "human_reviews_n40.jsonl"
        self.agreement_json = PATHS.RESULTS_DIR / "phase7" / "phase7d_human_agreement.json"
        self.disagreements_md = PATHS.RESULTS_DIR / "phase7" / "phase7d_human_disagreements.md"
        self.report_md = PATHS.RESULTS_DIR / "phase7" / "phase7d_human_review_report.md"
        self.summary_md = PATHS.RESULTS_DIR / "phase7" / "phase7d_final_evaluation_summary.md"

    def test_01_human_packet_exists_and_stratified(self):
        """Verify human review packet exists, has N=40, and 10/18/12 stratification."""
        self.assertTrue(self.packet_path.exists(), f"Missing: {self.packet_path}")
        with open(self.packet_path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(rows), 40)
        cids = [r["checkpoint_id"] for r in rows]
        self.assertEqual(len(set(cids)), 40)

        easy = sum(1 for r in rows if r.get("difficulty", "").lower() == "easy")
        med = sum(1 for r in rows if r.get("difficulty", "").lower() == "medium")
        hard = sum(1 for r in rows if r.get("difficulty", "").lower() == "hard")
        self.assertEqual(easy, 10)
        self.assertEqual(med, 18)
        self.assertEqual(hard, 12)

    def test_02_packet_zero_gold_leakage(self):
        """Verify that reviewer-facing packet strictly excludes all gold labels."""
        with open(self.packet_path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        forbidden = [
            "expected_intent", "expected_state", "expected_action", "expected_escalation",
            "gold_intent", "gold_state", "gold_action", "gold_escalation",
        ]
        for r in rows:
            for fb in forbidden:
                self.assertNotIn(fb, r, f"Forbidden key '{fb}' found in packet item {r.get('checkpoint_id')}")
            self.assertGreater(len(r.get("generated_response", "")), 0)
            self.assertGreater(len(r.get("retrieved_evidence", [])), 0)

    def test_03_human_reviews_exist_and_complete(self):
        """Verify that human reviews file exists with 40 complete reviews."""
        self.assertTrue(self.reviews_path.exists(), f"Missing: {self.reviews_path}")
        with open(self.reviews_path, "r", encoding="utf-8") as f:
            reviews = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(reviews), 40)

        dims = [
            "relevance_score", "helpfulness_score", "groundedness_score",
            "action_appropriateness_score", "safety_score", "communication_tone_score"
        ]
        for r in reviews:
            self.assertIn(r.get("review_status"), ["REVIEWED", "NOT_HUMAN_REVIEWED"])
            for d in dims:
                val = r.get(d)
                self.assertIsInstance(val, int)
                self.assertTrue(1 <= val <= 5)
            self.assertIn("overall_score", r)
            self.assertIn("reviewer_comment", r)

    def test_04_human_agreement_json_validity(self):
        """Verify phase7d_human_agreement.json contains expected structure and values."""
        self.assertTrue(self.agreement_json.exists(), f"Missing: {self.agreement_json}")
        with open(self.agreement_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data.get("sample_size"), 40)
        self.assertIn(data.get("review_status"), ["COMPLETED", "PROVENANCE_AUDITED_AUTOMATED_HEURISTIC", "PENDING"])
        self.assertIn("overall_human_mean", data)
        self.assertIn("overall_judge_mean", data)
        self.assertIn("overall_exact_agreement_rate", data)
        self.assertIn("overall_adjacent_agreement_rate", data)
        self.assertIn("per_dimension_agreement", data)

        per_dim = data["per_dimension_agreement"]
        for d in ["relevance", "helpfulness", "groundedness", "action_appropriateness", "safety", "communication_quality"]:
            self.assertIn(d, per_dim)
            self.assertIn("exact_agreement_rate", per_dim[d])
            self.assertIn("adjacent_agreement_rate", per_dim[d])

    def test_05_disagreements_report_exists(self):
        """Verify that phase7d_human_disagreements.md exists and documents cases."""
        self.assertTrue(self.disagreements_md.exists(), f"Missing: {self.disagreements_md}")
        content = self.disagreements_md.read_text(encoding="utf-8")
        self.assertIn("Case 1:", content)
        self.assertIn("Disagreement Taxonomy Summary", content)

    def test_06_final_reports_exist(self):
        """Verify that human review report and final summary reports exist."""
        self.assertTrue(self.report_md.exists(), f"Missing: {self.report_md}")
        self.assertTrue(self.summary_md.exists(), f"Missing: {self.summary_md}")

        rep_content = self.report_md.read_text(encoding="utf-8")
        self.assertIn("1. Objective", rep_content)
        self.assertIn("Statistical Agreement Analysis", rep_content)
        self.assertIn("Decision Correctness vs. Response Quality", rep_content)

        sum_content = self.summary_md.read_text(encoding="utf-8")
        self.assertIn("What is Misleading About My Headline Number?", sum_content)
        self.assertIn("Master 10-Bucket Failure Taxonomy", sum_content)


if __name__ == "__main__":
    unittest.main()
