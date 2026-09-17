"""Production Smoke Tests for AmazonHelp Customer Support AI Agent.

Verifies the 12 production criteria:
 1. CLI starts and returns help without error
 2. Chat session initializes cleanly
 3. Historical Train-only retrieval functions (K=5, similarity scores)
 4. Ollama unavailable behavior fails safely with actionable guidance
 5. Structured LLM JSON output schema parses and validates
 6. Deterministic safety validator suppresses credentials (password, OTP, PIN, CVV)
 7. Account/security compromise forces deterministic escalation
 8. Payment dispute / unauthorized transaction forces escalation
 9. Multi-turn dialogue state persists across sequential turns
10. No gold labels enter runtime inference code
11. Zero cross-partition leakage in the retrieval corpus
12. Project functions from root with dynamic relative paths
"""

import json
from pathlib import Path
import re
import subprocess
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.conversation_manager import ConversationManager
from src.annotation.annotator import (
    APPROVED_ACTIONS,
    APPROVED_ESCALATION_REASONS,
    APPROVED_INTENTS,
    APPROVED_STATES,
)
from src.config import PATHS
from src.llm.model_client import MockOllamaClient, OllamaClient
from src.llm.safety_validator import DeterministicSafetyValidator
from src.llm.schemas import LLMDecisionOutput, parse_and_validate_llm_output
from src.retrieval.retriever import HistoricalRetriever


class TestProductionSmoke(unittest.TestCase):
    """Production test suite for external evaluator verification."""

    def test_01_cli_starts_and_help_works(self):
        """1. Verify CLI starts and returns comprehensive help without crash."""
        res = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "cli.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("AmazonHelp Autonomous Customer Support AI Agent", res.stdout)
        self.assertIn("chat", res.stdout)
        self.assertIn("demo", res.stdout)
        self.assertIn("evaluate", res.stdout)
        self.assertIn("benchmark", res.stdout)
        self.assertIn("verify", res.stdout)

    def test_02_chat_initialization(self):
        """2. Verify ConversationManager initializes with clean defaults."""
        mgr = ConversationManager(use_mock_llm=True)
        self.assertTrue(mgr.conversation_id.startswith("conv_"))
        self.assertEqual(len(mgr.turn_history), 0)
        self.assertEqual(len(mgr.turn_records), 0)
        self.assertEqual(mgr.current_state, "STATE_INITIAL_INBOUND")
        self.assertFalse(mgr.escalation_status)

    def test_03_retrieval_engine_works(self):
        """3. Verify Train-only historical retrieval produces top-5 matches."""
        retriever = HistoricalRetriever()
        results = retriever.query("Where is my package? Tracking is delayed.", top_k=5)
        self.assertEqual(len(results), 5)
        for r in results:
            self.assertGreaterEqual(r.similarity_score, 0.0)
            self.assertLessEqual(r.similarity_score, 1.0)
            self.assertTrue(bool(r.derived_intent))
            self.assertTrue(bool(r.conversation_id))
            # Prove no golden benchmark IDs
            self.assertFalse(r.conversation_id.startswith("chk_"))

    def test_04_ollama_unavailable_graceful_behavior(self):
        """4. Verify that offline Ollama raises clear guidance rather than crashing with unhandled exception."""
        # Force non-mock mode when Ollama is offline
        if not OllamaClient().is_available():
            mgr = ConversationManager(use_mock_llm=False)
            with self.assertRaises(RuntimeError) as ctx:
                mgr.process_turn("Hello, where is my order?")
            self.assertIn("Local LLM is unavailable", str(ctx.exception))
            self.assertIn("ollama serve", str(ctx.exception))
            self.assertIn("--mock", str(ctx.exception))

    def test_05_structured_llm_response_parses(self):
        """5. Verify Pydantic schema validation over structured JSON output."""
        valid_json = json.dumps({
            "intent": "DELIVERY_STATUS_AND_TRACKING",
            "state": "STATE_INITIAL_INBOUND",
            "action": "PROVIDE_INFORMATION",
            "escalate": False,
            "escalation_reason": "NONE",
            "confidence": 0.95,
            "reasoning_summary": "Customer tracking delayed package.",
            "response": "You can track your package directly on the Amazon app.",
        })
        val = parse_and_validate_llm_output(valid_json)
        self.assertTrue(val.is_valid_json)
        self.assertTrue(val.is_schema_compliant)
        self.assertIsNotNone(val.decision)
        self.assertEqual(val.decision.intent, "DELIVERY_STATUS_AND_TRACKING")

    def test_06_safety_validator_blocks_credentials(self):
        """6. Verify deterministic safety validator suppresses sensitive authentication credentials."""
        validator = DeterministicSafetyValidator()
        unsafe_decision = LLMDecisionOutput(
            intent="ACCOUNT_ACCESS_AND_SECURITY",
            state="STATE_INITIAL_INBOUND",
            action="REQUEST_SAFE_DETAILS",
            escalate=False,
            escalation_reason="NONE",
            confidence=0.9,
            reasoning_summary="Draft asking for verification.",
            response="Please send us your password and your OTP code so we can verify your identity.",
        )
        validated = validator.validate_and_enforce(
            decision=unsafe_decision,
            customer_message="I cannot log into my account.",
            turn_depth=1,
        )
        self.assertGreater(len(validated.safety_violations_detected), 0)
        self.assertIn("solicited or exposed sensitive credentials", validated.safety_violations_detected[0])
        self.assertNotIn("password", validated.final_response.lower())
        self.assertNotIn("otp", validated.final_response.lower())
        self.assertEqual(validated.action, "HANDOFF_TO_SECURE_CHANNEL")

    def test_07_account_security_escalates(self):
        """7. Verify account compromise forces deterministic escalation."""
        validator = DeterministicSafetyValidator()
        decision = LLMDecisionOutput(
            intent="ACCOUNT_ACCESS_AND_SECURITY",
            state="STATE_INITIAL_INBOUND",
            action="PROVIDE_INFORMATION",
            escalate=False,
            escalation_reason="NONE",
            confidence=0.8,
            reasoning_summary="Inbound account alert.",
            response="Please check your settings on the website.",
        )
        validated = validator.validate_and_enforce(
            decision=decision,
            customer_message="Someone hacked my account and bought a laptop with my card! Help fraud!",
            turn_depth=1,
        )
        self.assertTrue(validated.escalate)
        self.assertEqual(validated.escalation_reason, "SECURITY_FRAUD_ALERT")
        self.assertEqual(validated.state, "STATE_CUSTOMER_ESCALATION")
        self.assertEqual(validated.action, "HANDOFF_TO_SECURE_CHANNEL")

    def test_08_payment_fraud_dispute_escalates(self):
        """8. Verify unauthorized payment dispute forces escalation."""
        validator = DeterministicSafetyValidator()
        decision = LLMDecisionOutput(
            intent="PAYMENT_BILLING_AND_PROMOTIONS",
            state="STATE_INITIAL_INBOUND",
            action="PROVIDE_INFORMATION",
            escalate=False,
            escalation_reason="NONE",
            confidence=0.8,
            reasoning_summary="Billing question.",
            response="You can check your payment transactions under Your Orders.",
        )
        validated = validator.validate_and_enforce(
            decision=decision,
            customer_message="I see an unauthorized fraudulent charge of $450 on my credit card statement!",
            turn_depth=1,
        )
        self.assertTrue(validated.escalate)
        self.assertEqual(validated.escalation_reason, "PAYMENT_ACCOUNT_DISPUTE")
        self.assertEqual(validated.action, "HANDOFF_TO_SECURE_CHANNEL")

    def test_09_multi_turn_state_persists(self):
        """9. Verify conversation state evolves across sequential turns."""
        mgr = ConversationManager(use_mock_llm=True)
        # Turn 1
        rec1 = mgr.process_turn("Where is my order? It is late.")
        self.assertEqual(rec1.turn_number, 1)
        self.assertEqual(mgr.current_intent, "DELIVERY_STATUS_AND_TRACKING")
        self.assertEqual(len(mgr.turn_history), 2)  # 1 customer + 1 agent

        # Turn 2: Customer provides tracking details
        rec2 = mgr.process_turn("My postcode is SW1A 1AA and tracking is 202-1234567-8901234")
        self.assertEqual(rec2.turn_number, 2)
        self.assertEqual(mgr.current_state, "STATE_CUSTOMER_PROVIDING_INFO")
        self.assertEqual(len(mgr.turn_history), 4)
        self.assertEqual(len(mgr.turn_records), 2)

    def test_10_no_gold_labels_in_runtime(self):
        """10. Verify no gold ground-truth fields are accessed by runtime inference."""
        leakage_patterns = [
            re.compile(r'\bgold_intent\b'),
            re.compile(r'\bgold_state\b'),
            re.compile(r'\bgold_action\b'),
            re.compile(r'\bgold_escalation\b'),
            re.compile(r'\bhuman_label\b'),
            re.compile(r'\bground_truth\b'),
            re.compile(r'\btarget_label\b'),
        ]
        scanned_files = [
            PROJECT_ROOT / "src" / "agent" / "conversation_manager.py",
            PROJECT_ROOT / "src" / "llm" / "agent_with_retrieval.py",
            PROJECT_ROOT / "src" / "llm" / "evidence_builder.py",
            PROJECT_ROOT / "src" / "llm" / "conversation_formatter.py",
            PROJECT_ROOT / "src" / "policy" / "action_policy.py",
            PROJECT_ROOT / "src" / "policy" / "escalation_policy.py",
            PROJECT_ROOT / "src" / "state" / "state_tracker.py",
        ]
        violations = []
        for sf in scanned_files:
            if sf.exists():
                text = sf.read_text(encoding="utf-8")
                for pat in leakage_patterns:
                    if pat.search(text):
                        violations.append(f"{sf.name}: {pat.pattern}")
        self.assertEqual(len(violations), 0, f"Runtime label leakage detected: {violations}")

    def test_11_no_dev_test_retrieval_leakage(self):
        """11. Verify zero overlap between Train retrieval corpus and Dev golden benchmark."""
        golden_conv_ids = set()
        with open(PATHS.AMAZONHELP_GOLDEN_HUMAN_VALIDATED_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                golden_conv_ids.add(json.loads(line)["conversation_id"])

        retrieval_conv_ids = set()
        with open(PATHS.AMAZONHELP_TRAIN_RETRIEVAL_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                retrieval_conv_ids.add(json.loads(line)["conversation_id"])

        overlap = golden_conv_ids.intersection(retrieval_conv_ids)
        self.assertEqual(len(overlap), 0, f"Retrieval corpus contains Dev conversations: {len(overlap)}")

    def test_12_project_root_relative_paths(self):
        """12. Verify all core asset paths resolve cleanly relative to project root."""
        self.assertTrue(PATHS.AMAZONHELP_GOLDEN_HUMAN_VALIDATED_JSONL.exists())
        self.assertTrue(PATHS.AMAZONHELP_TRAIN_RETRIEVAL_JSONL.exists())
        self.assertTrue(PATHS.RETRIEVAL_INDEX_NPZ.exists())
        self.assertTrue(PATHS.RETRIEVAL_INDEX_META.exists())
        self.assertTrue(PATHS.TFIDF_LOGREG_MODEL_PATH.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
