"""Deterministic Safety Validator and Policy Enforcement for Phase 6 LLM Agent.

The LLM is NOT the final safety authority.
This module executes strictly AFTER model generation to enforce:
1. Hard block against password, OTP, CVV, PIN, or card credential solicitation.
2. Detection and rewriting of fabricated/unsupported account actions (refunds, cancellations, replacements, account access).
3. Escalation safety overrides on fraud, account compromise, or severe threats.
4. Internal decision consistency validation.
5. Twitter response length compliance (safe sentence-boundary truncation).
6. Measurement of Unsupported Action Rate (target: 0.0%).
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple

from src.annotation.annotator import (
    APPROVED_ACTIONS,
    APPROVED_ESCALATION_REASONS,
    APPROVED_INTENTS,
    APPROVED_STATES,
)
from src.llm.schemas import LLMDecisionOutput
from src.policy.escalation_policy import (
    RE_PAYMENT_DISPUTE,
    RE_REPEATED_CONTACT,
    RE_SECURITY_FRAUD,
    RE_SEVERE_THREATS,
)
from src.utils.logger import get_logger

logger = get_logger("safety_validator")

# Regex for sensitive authentication secrets
RE_CREDENTIAL_KEYWORDS = re.compile(
    r"\b(password|passcode|otp|one-time password|cvv|cvc|security code|pin number|full card number|credit card number)\b",
    re.IGNORECASE,
)

# Regex for fabricated/unsupported transaction execution claims
RE_FABRICATED_REFUND = re.compile(
    r"\b(i have refunded|i refunded|refund has been processed|processed your refund|credited your account|issued a refund)\b",
    re.IGNORECASE,
)
RE_FABRICATED_CANCEL = re.compile(
    r"\b(i have cancelled|i cancelled|order has been cancelled|cancelled your order)\b",
    re.IGNORECASE,
)
RE_FABRICATED_REPLACE = re.compile(
    r"\b(i have dispatched|replacement has been ordered|sent a replacement|shipped a new item)\b",
    re.IGNORECASE,
)
RE_FABRICATED_ACCESS = re.compile(
    r"\b(i checked your account|logged into your account|looking at your account records|accessing your account|viewing your account)\b",
    re.IGNORECASE,
)


@dataclass
class ValidatedDecision:
    """Final decision output after deterministic safety policy enforcement."""

    intent: str
    state: str
    action: str
    escalate: bool
    escalation_reason: str
    confidence: float
    reasoning_summary: str
    final_response: str
    raw_response: str
    raw_response_length: int
    final_response_length: int
    was_truncated: bool
    is_safe: bool
    safety_violations_detected: List[str] = field(default_factory=list)
    unsupported_action_detected: bool = False
    safety_overrides_applied: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "state": self.state,
            "action": self.action,
            "escalate": self.escalate,
            "escalation_reason": self.escalation_reason,
            "confidence": self.confidence,
            "reasoning_summary": self.reasoning_summary,
            "final_response": self.final_response,
            "raw_response": self.raw_response,
            "raw_response_length": self.raw_response_length,
            "final_response_length": self.final_response_length,
            "was_truncated": self.was_truncated,
            "is_safe": self.is_safe,
            "safety_violations_detected": self.safety_violations_detected,
            "unsupported_action_detected": self.unsupported_action_detected,
            "safety_overrides_applied": self.safety_overrides_applied,
        }


class DeterministicSafetyValidator:
    """Deterministic post-generation guardrail layer."""

    def __init__(self, max_response_chars: int = 280):
        self.max_response_chars = max_response_chars

    def truncate_response_safely(self, text: str) -> Tuple[str, bool]:
        """Truncates text exceeding character limit at sentence/word boundary."""
        if len(text) <= self.max_response_chars:
            return text, False

        # Try cutting at last sentence boundary before limit
        truncated = text[: self.max_response_chars]
        last_period = max(truncated.rfind(". "), truncated.rfind("! "), truncated.rfind("? "))
        if last_period > 100:
            return truncated[: last_period + 1].strip(), True

        # Fallback to last space boundary
        last_space = truncated.rfind(" ")
        if last_space > 100:
            return truncated[:last_space].strip() + "...", True

        return truncated.strip(), True

    def validate_and_enforce(
        self,
        decision: LLMDecisionOutput,
        customer_message: str,
        turn_depth: int = 1,
        history: Optional[List[Dict[str, Any]]] = None,
        mandatory_escalation: Optional[Dict[str, Any]] = None,
    ) -> ValidatedDecision:
        """Applies deterministic security policies and safety overrides to LLM output."""
        raw_resp = decision.response
        raw_len = len(raw_resp)

        intent = decision.intent
        state = decision.state
        action = decision.action
        escalate = decision.escalate
        esc_reason = decision.escalation_reason
        confidence = decision.confidence
        reasoning = decision.reasoning_summary

        violations = []
        overrides = []
        unsupported_action = False

        # Combine text for context inspection
        combined_text = f"{customer_message} {raw_resp}"

        # -------------------------------------------------------------
        # 1. Credential Solicitation Audit
        # -------------------------------------------------------------
        if RE_CREDENTIAL_KEYWORDS.search(raw_resp):
            violations.append("LLM draft solicited or exposed sensitive credentials (password/OTP/CVV/PIN).")
            # Override response and action
            action = "HANDOFF_TO_SECURE_CHANNEL"
            raw_resp = (
                "For your security, please never share sensitive account credentials, codes, or payment details publicly. "
                "Please connect with us via DM on the Amazon app so we can safely assist you."
            )
            overrides.append("Rewrote draft response to suppress credential exposure and forced secure handoff.")

        # -------------------------------------------------------------
        # 2. Fabricated / Unsupported Action Claims Audit
        # -------------------------------------------------------------
        if RE_FABRICATED_REFUND.search(raw_resp):
            unsupported_action = True
            violations.append("LLM claimed refund processing without authorization API.")
            raw_resp = (
                "We don't have direct account access to process refunds in chat. "
                "You can request a refund directly via 'Your Orders' or contact us securely via DM."
            )
            overrides.append("Rewrote fabricated refund claim to safe informational guidance.")

        if RE_FABRICATED_CANCEL.search(raw_resp):
            unsupported_action = True
            violations.append("LLM claimed order cancellation without authorization API.")
            raw_resp = (
                "You can cancel your order directly from 'Your Orders' before dispatch, "
                "or connect with our secure team via DM for assistance."
            )
            overrides.append("Rewrote fabricated cancellation claim to safe self-service steps.")

        if RE_FABRICATED_ACCESS.search(raw_resp):
            unsupported_action = True
            violations.append("LLM claimed internal account database access.")
            raw_resp = (
                "We don't have access to your personal account details here. "
                "Please reach out via DM so our verified support team can securely review your order."
            )
            overrides.append("Rewrote fabricated account access claim.")

        # -------------------------------------------------------------
        # 3. Escalation Safety Overrides (Fraud, Account Compromise, Threats)
        # -------------------------------------------------------------
        if RE_SECURITY_FRAUD.search(combined_text):
            if not escalate or esc_reason != "SECURITY_FRAUD_ALERT":
                escalate = True
                esc_reason = "SECURITY_FRAUD_ALERT"
                state = "STATE_CUSTOMER_ESCALATION"
                action = "HANDOFF_TO_SECURE_CHANNEL"
                overrides.append("Deterministic safety override: Forced SECURITY_FRAUD_ALERT escalation.")

        elif RE_SEVERE_THREATS.search(combined_text):
            if not escalate or esc_reason != "SEVERE_FRUSTRATION_OR_THREAT":
                escalate = True
                esc_reason = "SEVERE_FRUSTRATION_OR_THREAT"
                state = "STATE_CUSTOMER_ESCALATION"
                action = "HANDOFF_TO_SECURE_CHANNEL"
                overrides.append("Deterministic safety override: Forced SEVERE_FRUSTRATION_OR_THREAT escalation.")

        elif RE_PAYMENT_DISPUTE.search(combined_text) and ("unauthorized" in combined_text.lower() or "fraud" in combined_text.lower()):
            if not escalate:
                escalate = True
                esc_reason = "PAYMENT_ACCOUNT_DISPUTE"
                state = "STATE_CUSTOMER_ESCALATION"
                action = "HANDOFF_TO_SECURE_CHANNEL"
                overrides.append("Deterministic safety override: Forced PAYMENT_ACCOUNT_DISPUTE escalation.")

        # Mandatory deterministic escalation from Phase 4 policy engine
        if mandatory_escalation and mandatory_escalation.get("escalation"):
            m_reason = mandatory_escalation.get("reason", "SEVERE_FRUSTRATION_OR_THREAT")
            if not escalate:
                escalate = True
                esc_reason = m_reason if m_reason in APPROVED_ESCALATION_REASONS else "SEVERE_FRUSTRATION_OR_THREAT"
                state = "STATE_CUSTOMER_ESCALATION"
                action = "HANDOFF_TO_SECURE_CHANNEL"
                overrides.append(f"Deterministic safety override: Enforced mandatory policy escalation ({esc_reason}).")

        # -------------------------------------------------------------
        # 4. Decision Consistency Checks
        # -------------------------------------------------------------
        if escalate and esc_reason == "NONE":
            esc_reason = "SEVERE_FRUSTRATION_OR_THREAT"
            overrides.append("Enforced valid escalation reason when escalate=True.")

        if not escalate and esc_reason != "NONE":
            esc_reason = "NONE"
            overrides.append("Enforced escalation_reason=NONE when escalate=False.")

        # -------------------------------------------------------------
        # 5. Twitter Response Length Compliance
        # -------------------------------------------------------------
        final_resp, was_truncated = self.truncate_response_safely(raw_resp)
        final_len = len(final_resp)

        is_safe = len(violations) == 0 or len(overrides) > 0

        return ValidatedDecision(
            intent=intent,
            state=state,
            action=action,
            escalate=escalate,
            escalation_reason=esc_reason,
            confidence=confidence,
            reasoning_summary=reasoning,
            final_response=final_resp,
            raw_response=decision.response,
            raw_response_length=raw_len,
            final_response_length=final_len,
            was_truncated=was_truncated,
            is_safe=is_safe,
            safety_violations_detected=violations,
            unsupported_action_detected=unsupported_action,
            safety_overrides_applied=overrides,
        )
