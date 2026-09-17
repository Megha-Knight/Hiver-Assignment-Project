"""Retrieval-Augmented Decision Policy Engine for AmazonHelp Agent.

Integrates:
1. Hybrid Intent Classification (TF-IDF LogReg + Similarity-Weighted Retrieval)
2. Retrieval-Augmented Escalation Policy (Recovering Missed Escalations)
3. State-Aware Retrieval Assistance
4. Action Policy Assistance with Strict Safety Precedence Overrides
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.annotation.annotator import (
    APPROVED_ACTIONS,
    APPROVED_ESCALATION_REASONS,
    APPROVED_INTENTS,
    APPROVED_STATES,
)
from src.baselines.tfidf_logreg import TfidfLogRegIntentClassifier
from src.policy.action_policy import DeterministicActionPolicy
from src.policy.escalation_policy import DeterministicEscalationPolicy
from src.retrieval.evidence import AggregatedRetrievalEvidence
from src.state.state_tracker import ConversationStateTracker
from src.utils.logger import get_logger

logger = get_logger("retrieval_policy")


class RetrievalAugmentedDecisionEngine:
    """Combines Phase 4 deterministic policies with structured historical retrieval evidence."""

    def __init__(
        self,
        intent_classifier: TfidfLogRegIntentClassifier,
        escalation_policy: Optional[DeterministicEscalationPolicy] = None,
        state_tracker: Optional[ConversationStateTracker] = None,
        action_policy: Optional[DeterministicActionPolicy] = None,
        escalation_retrieval_threshold: float = 0.40,
    ):
        self.intent_classifier = intent_classifier
        self.escalation_policy = escalation_policy or DeterministicEscalationPolicy()
        self.state_tracker = state_tracker or ConversationStateTracker()
        self.action_policy = action_policy or DeterministicActionPolicy()
        self.escalation_retrieval_threshold = escalation_retrieval_threshold

    def determine_hybrid_intent(
        self,
        customer_message: str,
        history: Optional[List[Dict[str, Any]]],
        evidence: AggregatedRetrievalEvidence,
        alpha_override: Optional[float] = None,
    ) -> Tuple[str, float, Dict[str, float]]:
        """Combines ML predicted probabilities with similarity-weighted retrieval distribution.

        Gating formula:
            P_hybrid(c) = alpha * P_ML(c) + (1 - alpha) * P_retrieval(c)

        Dynamic alpha depends on retrieval confidence:
        - HIGH   : alpha = 0.60 (40% weight to high-similarity retrieval)
        - MEDIUM : alpha = 0.80 (20% weight to retrieval)
        - LOW    : alpha = 0.95 (5% weight to retrieval)
        - NONE   : alpha = 1.00 (100% ML baseline)
        """
        ml_pred = self.intent_classifier.predict_with_confidence(customer_message, history)
        ml_probas = ml_pred["distribution"]
        ret_dist = evidence.weighted_intent_distribution

        if alpha_override is not None:
            alpha = alpha_override
        else:
            conf = evidence.retrieval_confidence
            if conf == "HIGH":
                alpha = 0.60
            elif conf == "MEDIUM":
                alpha = 0.80
            elif conf == "LOW":
                alpha = 0.95
            else:
                alpha = 1.0

        hybrid_dist = {}
        for intent in APPROVED_INTENTS:
            p_ml = ml_probas.get(intent, 0.0)
            p_ret = ret_dist.get(intent, 0.0)
            hybrid_dist[intent] = round(alpha * p_ml + (1.0 - alpha) * p_ret, 4)

        # Disambiguation heuristic: Prime Delivery Complaints
        msg_lower = customer_message.lower()
        if "prime" in msg_lower and any(k in msg_lower for k in ["deliver", "delay", "order", "arrive", "not received", "days"]):
            if "cancel prime" not in msg_lower and "membership fee" not in msg_lower:
                # If retrieval shows strong delivery evidence, favor delivery
                if ret_dist.get("DELIVERY_STATUS_AND_TRACKING", 0.0) >= 0.20:
                    hybrid_dist["DELIVERY_STATUS_AND_TRACKING"] += 0.25

        # Normalization & Top Selection
        total_p = sum(hybrid_dist.values()) or 1.0
        normalized = {k: round(v / total_p, 4) for k, v in hybrid_dist.items()}
        top_intent = max(normalized.items(), key=lambda x: x[1])[0]
        top_confidence = normalized[top_intent]

        return top_intent, top_confidence, normalized

    def determine_retrieval_augmented_escalation(
        self,
        customer_message: str,
        turn_depth: int,
        history: Optional[List[Dict[str, Any]]],
        intent: str,
        evidence: AggregatedRetrievalEvidence,
    ) -> Dict[str, Any]:
        """Evaluates escalation using Phase 4 deterministic rules augmented with retrieval consensus.

        Strategy:
        1. Rule-based triggers fire with high priority (preserving high precision).
        2. If rule does NOT fire, evaluate retrieval evidence:
           If retrieval_confidence in ["HIGH", "MEDIUM"] AND weighted_escalation_rate >= threshold:
           Trigger recovered escalation.
        """
        base_decision = self.escalation_policy.evaluate(
            customer_message=customer_message,
            turn_depth=turn_depth,
            history=history,
            intent=intent,
        )

        # If base rule already triggers, accept it
        if base_decision["escalation"]:
            base_decision["evidence"].append("Phase 4 deterministic rule trigger")
            base_decision["source"] = "rule_policy"
            return base_decision

        # Check for retrieval-augmented escalation recovery
        conf = evidence.retrieval_confidence
        esc_rate = evidence.weighted_escalation_rate

        if conf in ["HIGH", "MEDIUM"] and esc_rate >= self.escalation_retrieval_threshold:
            # Check whether retrieved reason is meaningful
            ret_reason = evidence.top_escalation_reason
            if ret_reason in ["PAYMENT_ACCOUNT_DISPUTE", "REPEATED_FAILED_CONTACT", "SEVERE_FRUSTRATION_OR_THREAT", "SECURITY_FRAUD_ALERT"]:
                recovery_evidence = [
                    f"Retrieval-recovered escalation (confidence: {conf}, sim-weighted rate: {esc_rate:.2f})",
                    f"Retrieved escalation precedent: {ret_reason}",
                ]
                return {
                    "escalation": True,
                    "reason": ret_reason,
                    "confidence": round(float(evidence.mean_similarity), 4),
                    "evidence": recovery_evidence,
                    "source": "retrieval_recovered",
                }

        # Keep non-escalated decision
        base_decision["source"] = "rule_policy_clean"
        return base_decision

    def process_checkpoint_augmented(
        self,
        checkpoint: Dict[str, Any],
        evidence: AggregatedRetrievalEvidence,
        mode: str = "full",  # "full", "intent_only", "escalation_only", "baseline"
    ) -> Dict[str, Any]:
        """Runs the complete retrieval-augmented decision pipeline for a golden checkpoint.

        Modes:
        - "baseline": Pure Phase 4 decision.
        - "intent_only": Hybrid intent + Phase 4 escalation/state/action.
        - "escalation_only": Phase 4 intent + retrieval-augmented escalation.
        - "full": Hybrid intent + retrieval-augmented escalation + assisted state/action with safety overrides.
        """
        msg = checkpoint["current_customer_message"]
        turn_depth = checkpoint.get("turn_depth", 1)
        history = checkpoint.get("conversation_history_before_current_turn", [])

        # 1. Intent Decision
        if mode in ["full", "intent_only"]:
            intent, intent_conf, intent_dist = self.determine_hybrid_intent(msg, history, evidence)
        else:
            ml_pred = self.intent_classifier.predict_with_confidence(msg, history)
            intent = ml_pred["intent"]
            intent_conf = ml_pred["confidence"]
            intent_dist = ml_pred["distribution"]

        # 2. Escalation Decision
        if mode in ["full", "escalation_only"]:
            esc_decision = self.determine_retrieval_augmented_escalation(msg, turn_depth, history, intent, evidence)
        else:
            esc_decision = self.escalation_policy.evaluate(msg, turn_depth, history, intent)
            esc_decision["source"] = "baseline_rule"

        # 3. Conversation State Decision (Assisted by evidence)
        state, state_transition = self.state_tracker.determine_state(
            customer_message=msg,
            turn_depth=turn_depth,
            history=history,
            escalation_decision=esc_decision,
            intent=intent,
        )

        # Retrieval assistance on state if ambiguous
        if mode == "full" and evidence.retrieval_confidence == "HIGH":
            # If historical cases universally resolved and customer expresses thanks
            if evidence.top_retrieved_state == "STATE_APPARENTLY_RESOLVED" and "thank" in msg.lower():
                state = "STATE_APPARENTLY_RESOLVED"
            elif esc_decision["escalation"] and state != "STATE_CUSTOMER_ESCALATION":
                state = "STATE_CUSTOMER_ESCALATION"

        # 4. Action Policy Decision with Safety Overrides (Part I)
        action_res = self.action_policy.select_action(
            intent=intent,
            state=state,
            escalation_decision=esc_decision,
            customer_message=msg,
            turn_depth=turn_depth,
            history=history,
        )

        # Retrieval assistance on action (only if safe and high-confidence)
        if mode == "full" and evidence.retrieval_confidence == "HIGH" and not esc_decision["escalation"]:
            ret_action = evidence.top_retrieved_action
            if ret_action in ["PROVIDE_INFORMATION", "PROVIDE_TROUBLESHOOTING", "REQUEST_SAFE_DETAILS"]:
                if action_res["action"] == "ASK_CLARIFICATION":
                    # Retrieval clarifies ambiguous action to specific actionable guidance
                    action_res["action"] = ret_action
                    action_res["reasoning"] += f" (Upgraded via high-confidence historical precedent: {ret_action})"

        decision_evidence = [
            f"Intent: {intent} (conf: {intent_conf:.2f}, mode: {mode})",
            f"State: {state} (transition: {state_transition.get('transition_reason')})",
            f"Escalation: {esc_decision['escalation']} ({esc_decision['reason']}, source: {esc_decision.get('source')})",
            f"Action: {action_res['action']} ({action_res['reasoning']})",
            f"Retrieval: max_sim={evidence.max_similarity:.2f}, conf={evidence.retrieval_confidence}, top_k={evidence.top_k}",
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
            "decision_evidence": decision_evidence,
            "is_safe": action_res.get("is_safe", True),
            "retrieval_confidence": evidence.retrieval_confidence,
            "max_similarity": evidence.max_similarity,
            "mean_similarity": evidence.mean_similarity,
        }
