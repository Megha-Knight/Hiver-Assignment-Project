"""Deterministic Multi-Turn Conversation State Tracker.

Implements the 8-state dialogue state machine for the AmazonHelp support agent.
Processes multi-turn conversation context and records auditable transition traces:
{
    "current_state": str,
    "trigger_evidence": str,
    "next_state": str,
    "transition_reason": str
}
"""

from typing import Any, Dict, List, Optional, Tuple
import re

from src.annotation.annotator import APPROVED_STATES
from src.data.preprocess import RE_RESOLVED_SIGNALS, RE_UNRESOLVED_SIGNALS
from src.utils.logger import get_logger

logger = get_logger("state_tracker")


class ConversationStateTracker:
    """Tracks and updates conversation state across multi-turn support dialogues."""

    def __init__(self):
        self.approved_states = set(APPROVED_STATES)

    def determine_state(
        self,
        customer_message: str,
        turn_depth: int,
        history: Optional[List[Dict[str, Any]]] = None,
        escalation_decision: Optional[Dict[str, Any]] = None,
        intent: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Determines dialogue state following customer utterance and outputs transition audit.

        Returns:
            (next_state, transition_record)
        """
        msg = customer_message.strip()
        msg_lower = msg.lower()
        history = history or []

        # 1. Turn 1 (Initial Inbound)
        if turn_depth <= 1 or not history:
            current_state = "STATE_INITIAL_INBOUND"

            # Check if initial message is already an escalation
            if escalation_decision and escalation_decision.get("escalation"):
                reason = escalation_decision.get("reason")
                if reason in ["SEVERE_FRUSTRATION_OR_THREAT", "REPEATED_FAILED_CONTACT"]:
                    next_state = "STATE_CUSTOMER_ESCALATION"
                    transition = {
                        "current_state": current_state,
                        "trigger_evidence": f"Initial customer inbound exhibits escalation ({reason})",
                        "next_state": next_state,
                        "transition_reason": "Customer initiates conversation with extreme hostility or repeated failure",
                    }
                    return next_state, transition

            next_state = "STATE_INITIAL_INBOUND"
            transition = {
                "current_state": "NONE",
                "trigger_evidence": "First customer message in session",
                "next_state": next_state,
                "transition_reason": "Initiation of customer support inquiry",
            }
            return next_state, transition

        # 2. Turn > 1 (Multi-Turn Context)
        # Infer prior dialogue state from last support message
        last_support = ""
        for h in reversed(history):
            if h.get("role") == "support":
                last_support = h.get("text", "")
                break
        last_support_lower = last_support.lower()

        # Determine implied state before this turn
        if "dm" in last_support_lower or "http" in last_support_lower:
            implied_prior_state = "STATE_SECURE_HANDOFF_TRIGGERED"
        elif "?" in last_support:
            implied_prior_state = "STATE_CLARIFICATION_REQUESTED"
        elif any(k in last_support_lower for k in ["restart", "unplug", "settings", "try", "steps"]):
            implied_prior_state = "STATE_TROUBLESHOOTING_ACTIVE"
        else:
            implied_prior_state = "STATE_TROUBLESHOOTING_ACTIVE"

        # Check for customer-declared resolution
        is_resolved = bool(RE_RESOLVED_SIGNALS.search(msg_lower) and not RE_UNRESOLVED_SIGNALS.search(msg_lower))
        if any(w in msg_lower for w in ["thanks for the speedy fix", "already delivered today", "got the replacement copy", "issue is resolved", "thanks all sorted"]):
            is_resolved = True

        if is_resolved:
            next_state = "STATE_APPARENTLY_RESOLVED"
            transition = {
                "current_state": implied_prior_state,
                "trigger_evidence": f"Customer indicates resolution: '{msg[:60]}...'",
                "next_state": next_state,
                "transition_reason": "Customer explicitly acknowledges receipt, fix, or satisfactory outcome",
            }
            return next_state, transition

        # Check for customer escalation in later turns
        if escalation_decision and escalation_decision.get("escalation"):
            reason = escalation_decision.get("reason")
            if reason in ["SEVERE_FRUSTRATION_OR_THREAT", "REPEATED_FAILED_CONTACT"]:
                next_state = "STATE_CUSTOMER_ESCALATION"
                transition = {
                    "current_state": implied_prior_state,
                    "trigger_evidence": f"Customer escalation detected in turn {turn_depth}: '{reason}'",
                    "next_state": next_state,
                    "transition_reason": "Customer rejects ongoing handling with escalation/threat signals",
                }
                return next_state, transition

        # Check if customer is providing requested information
        if implied_prior_state in ["STATE_CLARIFICATION_REQUESTED", "STATE_SECURE_HANDOFF_TRIGGERED"] or "?" in last_support or "let us know" in last_support_lower or "order" in last_support_lower:
            next_state = "STATE_CUSTOMER_PROVIDING_INFO"
            transition = {
                "current_state": implied_prior_state,
                "trigger_evidence": f"Customer replies to prior support inquiry: '{last_support[:50]}...'",
                "next_state": next_state,
                "transition_reason": "Customer provides answers or context requested by support agent",
            }
            return next_state, transition

        # Default multi-turn active troubleshooting
        next_state = "STATE_TROUBLESHOOTING_ACTIVE"
        transition = {
            "current_state": implied_prior_state,
            "trigger_evidence": f"Continuing multi-turn exchange at depth {turn_depth}",
            "next_state": next_state,
            "transition_reason": "Multi-turn diagnostic or instructional progression",
        }
        return next_state, transition
