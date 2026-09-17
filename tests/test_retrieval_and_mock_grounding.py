"""Regression Tests for Canonical Exemplar Schema, Grounding Evaluator, and Mock Client.

Verifies:
1. Canonical exemplar serialization (RetrievedExemplar.to_dict).
2. Legacy exemplar compatibility (RetrievalContext.from_dict).
3. Human-review evidence display helper formatting.
4. Grounding evaluator with valid evidence returns grounded.
5. Grounding evaluator with missing/empty evidence returns ungrounded.
6. Mock client does not override current action with retrieved historical action.
7. Unresolved delivery complaint does not produce CONFIRM_RESOLUTION.
8. Refund complaint does not claim refund completion.
9. Security complaint respects security policy.
10. Technical complaint does not receive unrelated delivery instructions.
"""

from pathlib import Path
import sys
import unittest
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.grounding_evaluator import EvidenceGroundingEvaluator
from src.llm.model_client import MockOllamaClient
from src.llm.retrieval_context import RetrievedExemplar
from src.agent.conversation_manager import RetrievedExemplarSummary


class TestRetrievalAndMockGroundingRegression(unittest.TestCase):
    """Regression test suite for retrieval schema alignment and grounded mock generation."""

    def test_01_canonical_exemplar_serialization(self):
        """1. Verify RetrievedExemplar.to_dict contains both canonical fields and legacy aliases."""
        ex = RetrievedExemplar(
            retrieval_id="ret_test_01",
            conversation_id="conv_test_01",
            similarity_score=0.825,
            confidence_tier="HIGH",
            derived_intent="DELIVERY_STATUS_AND_TRACKING",
            derived_state="STATE_INITIAL_INBOUND",
            derived_action="REQUEST_SAFE_DETAILS",
            historical_escalation=False,
            historical_escalation_reason="NONE",
            resolution_status="APPARENTLY_RESOLVED",
            customer_problem_summary="Where is my parcel?",
            support_response="Please reply with your postal code so we can track it.",
        )
        d = ex.to_dict()
        # Canonical fields
        self.assertIn("similarity_score", d)
        self.assertIn("customer_problem_summary", d)
        self.assertIn("support_response", d)
        self.assertEqual(d["similarity_score"], 0.825)
        self.assertEqual(d["customer_problem_summary"], "Where is my parcel?")
        self.assertEqual(d["support_response"], "Please reply with your postal code so we can track it.")
        # Legacy aliases
        self.assertIn("similarity", d)
        self.assertIn("customer_text", d)
        self.assertIn("support_reply", d)
        self.assertEqual(d["similarity"], 0.825)
        self.assertEqual(d["customer_text"], "Where is my parcel?")
        self.assertEqual(d["support_reply"], "Please reply with your postal code so we can track it.")

    def test_02_legacy_exemplar_compatibility(self):
        """2. Verify RetrievedExemplar.from_dict parses legacy dictionary format."""
        legacy_dict = {
            "retrieval_id": "ret_legacy_02",
            "conversation_id": "conv_legacy_02",
            "similarity": 0.74,
            "derived_intent": "RETURN_REFUND_AND_REPLACEMENT",
            "customer_text": "Need refund for damaged shoes",
            "support_reply": "You can start a return via Your Orders.",
        }
        ex = RetrievedExemplar.from_dict(legacy_dict)
        self.assertEqual(ex.similarity_score, 0.74)
        self.assertEqual(ex.customer_problem_summary, "Need refund for damaged shoes")
        self.assertEqual(ex.support_response, "You can start a return via Your Orders.")

    def test_03_exemplar_summary_compatibility(self):
        """3. Verify RetrievedExemplarSummary exposes canonical properties and to_dict aliases."""
        summary = RetrievedExemplarSummary(
            similarity=0.79,
            intent="DELIVERY_STATUS_AND_TRACKING",
            action="REQUEST_SAFE_DETAILS",
            customer_text="Package delayed",
            support_reply="Checking carrier updates",
        )
        self.assertEqual(summary.similarity_score, 0.79)
        self.assertEqual(summary.customer_problem_summary, "Package delayed")
        self.assertEqual(summary.support_response, "Checking carrier updates")
        d = summary.to_dict()
        self.assertEqual(d["similarity_score"], 0.79)
        self.assertEqual(d["customer_problem_summary"], "Package delayed")
        self.assertEqual(d["support_response"], "Checking carrier updates")

    def test_04_grounding_evaluator_with_valid_evidence(self):
        """4. Verify grounding evaluator recognizes groundedness when matching canonical evidence exists."""
        evaluator = EvidenceGroundingEvaluator()
        res = evaluator.evaluate_grounding(
            customer_message="Where is my tracking ID?",
            generated_response="You can check your tracking details directly under 'Your Orders' or send us a DM.",
            retrieved_exemplars=[
                {
                    "support_response": "Track your parcel under Your Orders on the website or DM us.",
                    "similarity_score": 0.78,
                }
            ],
            retrieval_top1_similarity=0.78,
        )
        self.assertTrue(res.is_grounded)
        self.assertEqual(res.evidence_support, "SUPPORTED")

    def test_05_grounding_evaluator_with_missing_evidence(self):
        """5. Verify grounding evaluator marks empty/missing evidence as UNSUPPORTED and ungrounded."""
        evaluator = EvidenceGroundingEvaluator()
        # Case A: empty exemplars list
        res_empty = evaluator.evaluate_grounding(
            customer_message="Where is my parcel?",
            generated_response="Please send us a DM with your details so we can check.",
            retrieved_exemplars=[],
            retrieval_top1_similarity=0.0,
        )
        self.assertFalse(res_empty.is_grounded)
        self.assertEqual(res_empty.evidence_support, "UNSUPPORTED")

        # Case B: exemplars present but support_response is empty string
        res_blank = evaluator.evaluate_grounding(
            customer_message="Where is my parcel?",
            generated_response="Please send us a DM with your details so we can check.",
            retrieved_exemplars=[{"support_response": "", "similarity_score": 0.75}],
            retrieval_top1_similarity=0.75,
        )
        self.assertFalse(res_blank.is_grounded)
        self.assertEqual(res_blank.evidence_support, "UNSUPPORTED")

    def test_06_mock_client_does_not_override_action_with_retrieved_action(self):
        """6. Verify MockOllamaClient does NOT overwrite current action candidate with historical action."""
        client = MockOllamaClient()
        signals = {
            "baseline_intent": "DELIVERY_STATUS_AND_TRACKING",
            "deterministic_state": "STATE_INITIAL_INBOUND",
            "deterministic_action_candidate": "REQUEST_SAFE_DETAILS",
            "deterministic_escalation": False,
            "deterministic_escalation_reason": "NONE",
            "retrieval_confidence": "HIGH",
            "top_1_similarity": 0.85,
            "top_retrieved_action": "CONFIRM_RESOLUTION",  # Historical action is CONFIRM_RESOLUTION
        }
        res = client.generate_decision(
            model_name="llama3.2:1b",
            customer_message="My package has not arrived and tracking stopped updating.",
            turn_depth=1,
            evidence_signals=signals,
        )
        self.assertTrue(res.is_schema_compliant)
        decision = res.decision
        self.assertIsNotNone(decision)
        # Action must remain REQUEST_SAFE_DETAILS, NOT CONFIRM_RESOLUTION!
        self.assertEqual(decision.action, "REQUEST_SAFE_DETAILS")
        self.assertNotEqual(decision.action, "CONFIRM_RESOLUTION")

    def test_07_unresolved_delivery_complaint_does_not_produce_confirm_resolution(self):
        """7. Verify unresolved delivery complaint never produces 'We are glad to hear your issue has been resolved!'."""
        client = MockOllamaClient()
        signals = {
            "baseline_intent": "DELIVERY_STATUS_AND_TRACKING",
            "deterministic_state": "STATE_INITIAL_INBOUND",
            "deterministic_action_candidate": "REQUEST_SAFE_DETAILS",
            "deterministic_escalation": False,
            "deterministic_escalation_reason": "NONE",
            "retrieval_confidence": "HIGH",
            "top_1_similarity": 0.82,
            "top_retrieved_action": "CONFIRM_RESOLUTION",
        }
        res = client.generate_decision(
            model_name="llama3.2:1b",
            customer_message="Does Prime no longer offer next day delivery? What I ordered yesterday has not arrived.",
            turn_depth=1,
            evidence_signals=signals,
        )
        decision = res.decision
        self.assertNotIn("glad your issue has been resolved", decision.response.lower())
        self.assertIn("delivery", decision.response.lower())

    def test_08_refund_complaint_does_not_claim_refund_completion(self):
        """8. Verify refund complaint does not fabricate that a refund was already processed."""
        client = MockOllamaClient()
        signals = {
            "baseline_intent": "RETURN_REFUND_AND_REPLACEMENT",
            "deterministic_state": "STATE_INITIAL_INBOUND",
            "deterministic_action_candidate": "PROVIDE_INFORMATION",
            "deterministic_escalation": False,
            "deterministic_escalation_reason": "NONE",
            "retrieval_confidence": "HIGH",
            "top_1_similarity": 0.77,
            "top_retrieved_action": "CONFIRM_RESOLUTION",
        }
        res = client.generate_decision(
            model_name="llama3.2:1b",
            customer_message="I haven't gotten my money back for the cancelled order.",
            turn_depth=1,
            evidence_signals=signals,
        )
        decision = res.decision
        resp_lower = decision.response.lower()
        self.assertNotIn("i refunded", resp_lower)
        self.assertNotIn("i have refunded", resp_lower)
        self.assertNotIn("refund has been processed", resp_lower)
        self.assertNotIn("glad your issue has been resolved", resp_lower)

    def test_09_security_complaint_respects_security_policy(self):
        """9. Verify security/account compromise complaints mandate secure DM handoff."""
        client = MockOllamaClient()
        signals = {
            "baseline_intent": "ACCOUNT_ACCESS_AND_SECURITY",
            "deterministic_state": "STATE_CUSTOMER_ESCALATION",
            "deterministic_action_candidate": "HANDOFF_TO_SECURE_CHANNEL",
            "deterministic_escalation": True,
            "deterministic_escalation_reason": "SECURITY_FRAUD_ALERT",
            "retrieval_confidence": "HIGH",
            "top_1_similarity": 0.79,
            "top_retrieved_action": "CONFIRM_RESOLUTION",
        }
        res = client.generate_decision(
            model_name="llama3.2:1b",
            customer_message="Someone hacked my account and changed my password! Is this legit?",
            turn_depth=1,
            evidence_signals=signals,
        )
        decision = res.decision
        self.assertEqual(decision.intent, "ACCOUNT_ACCESS_AND_SECURITY")
        self.assertEqual(decision.action, "HANDOFF_TO_SECURE_CHANNEL")
        self.assertTrue(decision.escalate)
        self.assertIn("dm", decision.response.lower())
        self.assertIn("security", decision.response.lower())

    def test_10_technical_complaint_does_not_receive_unrelated_instructions(self):
        """10. Verify Kindle hardware issues receive device reboot guidance rather than browser cache."""
        client = MockOllamaClient()
        signals = {
            "baseline_intent": "TECHNICAL_AND_DIGITAL_SUPPORT",
            "deterministic_state": "STATE_INITIAL_INBOUND",
            "deterministic_action_candidate": "PROVIDE_TROUBLESHOOTING",
            "deterministic_escalation": False,
            "deterministic_escalation_reason": "NONE",
            "retrieval_confidence": "HIGH",
            "top_1_similarity": 0.76,
            "top_retrieved_action": "PROVIDE_TROUBLESHOOTING",
        }
        res = client.generate_decision(
            model_name="llama3.2:1b",
            customer_message="My Kindle screen is frozen and won't turn on even when charged.",
            turn_depth=1,
            evidence_signals=signals,
        )
        decision = res.decision
        resp_lower = decision.response.lower()
        self.assertIn("kindle", resp_lower)
        self.assertIn("power button", resp_lower)
        self.assertNotIn("browser cache", resp_lower)


if __name__ == "__main__":
    unittest.main()
