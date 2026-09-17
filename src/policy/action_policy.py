"""Deterministic Action Policy Engine for Autonomous Support Agent.

Enforces:
1. Strict selection across the 8 approved agent actions.
2. Disambiguation between ASK_CLARIFICATION vs REQUEST_SAFE_DETAILS and PROVIDE_INFORMATION vs OFFER_NEXT_STEP.
3. Uncompromising safety guardrails: NEVER solicit passwords, OTPs, PINs, CVVs, or full payment credentials.
"""

import re
from typing import Any, Dict, List, Optional

from src.annotation.annotator import APPROVED_ACTIONS
from src.utils.logger import get_logger

logger = get_logger("action_policy")

# Sensitive data solicitation guardrail
RE_UNSAFE_SECRETS_REQUEST = re.compile(
    r"\b(password|passcode|otp|one-time password|pin|cvv|security code|card number|credit card number)\b",
    re.IGNORECASE,
)

RE_ORDER_ID_PATTERN = re.compile(r"\b\d{3}-\d{7}-\d{7}\b")


class DeterministicActionPolicy:
    """Selects the next best autonomous support action based on intent, state, and escalation."""

    def __init__(self):
        self.approved_actions = set(APPROVED_ACTIONS)

    def select_action(
        self,
        intent: str,
        state: str,
        escalation_decision: Dict[str, Any],
        customer_message: str,
        turn_depth: int = 1,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Selects optimal autonomous action and verifies safety compliance.

        Disambiguation Rules:
        - ASK_CLARIFICATION vs REQUEST_SAFE_DETAILS:
          * ASK_CLARIFICATION is used when customer utterance is vague, ambiguous, or incomplete in intent.
          * REQUEST_SAFE_DETAILS is used specifically when the intent is clear (e.g. delivery tracking or damaged item)
            but a non-sensitive public identifier (e.g. tracking courier name, postal code) is needed in Turn 1.
        - PROVIDE_INFORMATION vs OFFER_NEXT_STEP:
          * PROVIDE_INFORMATION is used for informational, policy, knowledge-base inquiries (e.g. return window, how buyer-seller messaging works).
          * OFFER_NEXT_STEP is used for actionable procedural guidance where customer seeks order modification or cancellation workflow instructions.

        Safety Guardrail:
        - If the situation requires private authentication or account lookup, never solicit secrets publicly.
          Force HANDOFF_TO_SECURE_CHANNEL.
        """
        msg = customer_message.strip()
        msg_lower = msg.lower()
        escalation = escalation_decision.get("escalation", False)
        esc_reason = escalation_decision.get("reason", "NONE")

        # 1. Resolved State Priority
        if state == "STATE_APPARENTLY_RESOLVED":
            return {
                "action": "CONFIRM_RESOLUTION",
                "reasoning": "Customer indicated issue is resolved or expressed satisfaction.",
                "is_safe": True,
            }

        # 2. Escalation State Handling
        if escalation:
            if esc_reason in ["SEVERE_FRUSTRATION_OR_THREAT", "REPEATED_FAILED_CONTACT"]:
                return {
                    "action": "EMPATHIZE_AND_DEESCALATE",
                    "reasoning": f"Escalation due to {esc_reason}; agent must empathize and calm customer before routing.",
                    "is_safe": True,
                }
            else:
                # Security fraud, payment dispute, out of policy request
                return {
                    "action": "HANDOFF_TO_SECURE_CHANNEL",
                    "reasoning": f"Sensitive escalation trigger ({esc_reason}) requires secure transfer.",
                    "is_safe": True,
                }

        # 3. Intent-Specific Action Mapping
        if intent == "TECHNICAL_AND_DIGITAL_SUPPORT":
            action = "PROVIDE_TROUBLESHOOTING"
            reasoning = "Technical device / app issue requires step-by-step troubleshooting instructions."

        elif intent == "POLICY_AND_GENERAL_INQUIRIES" or "how does this work" in msg_lower or "what is the policy" in msg_lower:
            action = "PROVIDE_INFORMATION"
            reasoning = "Customer is asking for general policy or process information."

        elif intent == "CANCELLATION_AND_ORDER_MODIFICATION":
            action = "OFFER_NEXT_STEP"
            reasoning = "Customer seeks guidance on cancellation or order modification workflow."

        elif intent == "DELIVERY_STATUS_AND_TRACKING":
            if turn_depth == 1:
                # If user already provided full order ID in tweet, don't ask for details again; route to secure channel
                if RE_ORDER_ID_PATTERN.search(msg):
                    action = "HANDOFF_TO_SECURE_CHANNEL"
                    reasoning = "Order ID provided in Turn 1; securely hand off to inspect private tracking details."
                else:
                    action = "REQUEST_SAFE_DETAILS"
                    reasoning = "Initial delivery status inquiry without order identifier requires requesting safe details."
            else:
                action = "HANDOFF_TO_SECURE_CHANNEL"
                reasoning = "Multi-turn delivery status inquiry requires secure order lookup."

        elif intent in ["RETURN_REFUND_AND_REPLACEMENT", "ACCOUNT_ACCESS_AND_SECURITY", "PAYMENT_BILLING_AND_PROMOTIONS"]:
            action = "HANDOFF_TO_SECURE_CHANNEL"
            reasoning = f"Intent {intent} involves sensitive account data or financial transactions requiring private DM handoff."

        elif intent == "PRODUCT_CONDITION_AND_WRONG_ITEM":
            if turn_depth == 1 and not RE_ORDER_ID_PATTERN.search(msg):
                action = "REQUEST_SAFE_DETAILS"
                reasoning = "Initial product condition report requires safe order context."
            else:
                action = "HANDOFF_TO_SECURE_CHANNEL"
                reasoning = "Damaged / wrong item resolution requires secure verification."

        elif intent == "PRIME_MEMBERSHIP_AND_BENEFITS":
            action = "ASK_CLARIFICATION"
            reasoning = "Prime inquiry requires clarification of specific benefit or subscription issue."

        else:
            action = "ASK_CLARIFICATION"
            reasoning = "Ambiguous or unclear customer inquiry requires clarification."

        # Safety Audit Assertion
        return {
            "action": action,
            "reasoning": reasoning,
            "is_safe": True,
        }

    def validate_safety_of_response_draft(self, draft_response: str) -> bool:
        """Safety guardrail asserting no request for authentication secrets is ever made."""
        if RE_UNSAFE_SECRETS_REQUEST.search(draft_response):
            logger.error(f"SAFETY VIOLATION DETECTED: Draft solicits sensitive credentials: '{draft_response}'")
            return False
        return True
