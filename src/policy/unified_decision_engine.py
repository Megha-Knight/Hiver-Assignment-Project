"""Unified Deterministic Decision Engine.

Orchestrates:
Conversation Checkpoint
  -> Intent Classifier (TF-IDF + LogReg or Baseline)
  -> Deterministic Escalation Policy
  -> Deterministic Conversation State Tracker
  -> Deterministic Action Policy
  -> Structured Decision Output
"""

from typing import Any, Dict, List, Optional

from src.policy.action_policy import DeterministicActionPolicy
from src.policy.escalation_policy import DeterministicEscalationPolicy
from src.state.state_tracker import ConversationStateTracker
from src.utils.logger import get_logger

logger = get_logger("unified_decision_engine")


class UnifiedDeterministicDecisionEngine:
    """Pipelines intent, state, escalation, and action into a structured decision."""

    def __init__(
        self,
        intent_classifier: Any,
        escalation_policy: Optional[DeterministicEscalationPolicy] = None,
        state_tracker: Optional[ConversationStateTracker] = None,
        action_policy: Optional[DeterministicActionPolicy] = None,
    ):
        self.intent_classifier = intent_classifier
        self.escalation_policy = escalation_policy or DeterministicEscalationPolicy()
        self.state_tracker = state_tracker or ConversationStateTracker()
        self.action_policy = action_policy or DeterministicActionPolicy()

    def process_checkpoint(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Processes a golden checkpoint and outputs a complete structured decision."""
        msg = checkpoint["current_customer_message"]
        turn_depth = checkpoint.get("turn_depth", 1)
        history = checkpoint.get("conversation_history_before_current_turn", [])

        # 1. Intent Classification
        if hasattr(self.intent_classifier, "predict_with_confidence"):
            intent_res = self.intent_classifier.predict_with_confidence(msg, history)
            intent = intent_res["intent"]
            intent_conf = intent_res["confidence"]
        elif hasattr(self.intent_classifier, "predict_one"):
            intent = self.intent_classifier.predict_one(msg, history)
            intent_conf = 1.0
        else:
            intent = "OTHER_OR_UNCLEAR"
            intent_conf = 0.5

        # 2. Escalation Policy Evaluation
        esc_decision = self.escalation_policy.evaluate(
            customer_message=msg,
            turn_depth=turn_depth,
            history=history,
            intent=intent,
        )

        # 3. Conversation State Tracking
        state, state_transition = self.state_tracker.determine_state(
            customer_message=msg,
            turn_depth=turn_depth,
            history=history,
            escalation_decision=esc_decision,
            intent=intent,
        )

        # 4. Action Policy Selection
        action_res = self.action_policy.select_action(
            intent=intent,
            state=state,
            escalation_decision=esc_decision,
            customer_message=msg,
            turn_depth=turn_depth,
            history=history,
        )

        decision_evidence = [
            f"Intent: {intent} (conf: {intent_conf:.2f})",
            f"State: {state} (transition: {state_transition.get('transition_reason')})",
            f"Escalation: {esc_decision['escalation']} ({esc_decision['reason']})",
            f"Action: {action_res['action']} ({action_res['reasoning']})",
        ]
        decision_evidence.extend(esc_decision.get("evidence", []))

        return {
            "checkpoint_id": checkpoint.get("checkpoint_id"),
            "intent": intent,
            "intent_confidence": intent_conf,
            "state": state,
            "action": action_res["action"],
            "escalation": esc_decision["escalation"],
            "escalation_reason": esc_decision["reason"],
            "state_transition": state_transition,
            "action_reasoning": action_res["reasoning"],
            "decision_evidence": decision_evidence,
            "is_safe": action_res.get("is_safe", True),
        }
