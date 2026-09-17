"""Deterministic Rule-Based Escalation Policy Engine.

Evaluates customer utterance and dialogue context for explicit safety,
regulatory, financial, and repeated-contact risk triggers.

Produces auditable structured decisions:
{
    "escalation": bool,
    "reason": str,
    "confidence": float,
    "evidence": List[str]
}
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from src.annotation.annotator import APPROVED_ESCALATION_REASONS
from src.data.preprocess import (
    RE_ESCALATION_PHRASES,
    RE_REPEATED_CONTACT_SIGNALS,
    RE_UNRESOLVED_SIGNALS,
)
from src.utils.logger import get_logger

logger = get_logger("escalation_policy")

# Regex patterns for high-risk categories
RE_SECURITY_FRAUD = re.compile(
    r"\b(phishing|is this legit|fake email|scam|scammers|suspicious email|compromised|hacked|stolen account|"
    r"unauthorized login|otp|2fa|verification code|locked out of my account|account takeover)\b",
    re.IGNORECASE,
)

RE_PAYMENT_DISPUTE = re.compile(
    r"\b(charged (?:twice|three|3|multiple times)|double charge|unauthorized charge|declined my bank|"
    r"charged me for something i cancelled|refund my amount|not giving clear statement of my refund|"
    r"fraudulent charge|bank balance|bank account deducted|money taken without)\b",
    re.IGNORECASE,
)

RE_SEVERE_THREATS = re.compile(
    r"\b(trading standards|better business bureau|bbb|lawyer|attorney|police|court|sue you|legal action|"
    r"piss poor|bullshit|scamming customers|cheating customers|worst service ever|disgusting service|"
    r"unacceptable service)\b",
    re.IGNORECASE,
)

RE_REPEATED_CONTACT = re.compile(
    r"\b(after phone call|called twice|called (?:3|4|5|several) times|already (?:called|emailed|chatted|told)|"
    r"3rd day waiting|even after 3 days|3 days from ordering|no use complaining through their customer call center|"
    r"still no answer|no response for days|waiting since yesterday|how many times do i have to)\b",
    re.IGNORECASE,
)

RE_OUT_OF_POLICY = re.compile(
    r"\b(waive the fee|make an exception|demand immediate compensation|override your policy)\b",
    re.IGNORECASE,
)


class DeterministicEscalationPolicy:
    """Evaluates multi-turn context and customer text against deterministic escalation rules."""

    def __init__(self):
        self.reasons_vocab = APPROVED_ESCALATION_REASONS

    def evaluate(
        self,
        customer_message: str,
        turn_depth: int = 1,
        history: Optional[List[Dict[str, Any]]] = None,
        intent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluates a customer utterance and returns a structured escalation decision.

        Precedence Ordering:
        1. SECURITY_FRAUD_ALERT: immediate account risk overrides all other triggers.
        2. SEVERE_FRUSTRATION_OR_THREAT: legal/regulatory threats mandate human intervention.
        3. PAYMENT_ACCOUNT_DISPUTE: unauthorized financial transactions require dispute handling.
        4. REPEATED_FAILED_CONTACT: multi-turn / multi-channel contact failures.
        5. OUT_OF_POLICY_REQUEST: custom manual discretion requested.
        """
        msg = customer_message.strip()
        msg_lower = msg.lower()
        evidence: List[str] = []

        # 1. Check Security Fraud Alert
        sec_match = RE_SECURITY_FRAUD.search(msg)
        if sec_match:
            evidence.append(f"Security trigger match: '{sec_match.group(0)}'")
            return {
                "escalation": True,
                "reason": "SECURITY_FRAUD_ALERT",
                "confidence": 0.98,
                "evidence": evidence,
            }

        # 2. Check Severe Frustration or Threats (Legal, Regulatory, Hostility)
        threat_match = RE_SEVERE_THREATS.search(msg)
        if threat_match:
            evidence.append(f"Severe hostility/threat match: '{threat_match.group(0)}'")
            return {
                "escalation": True,
                "reason": "SEVERE_FRUSTRATION_OR_THREAT",
                "confidence": 0.95,
                "evidence": evidence,
            }

        # Also check general escalation regex from Phase 2
        esc_general_match = RE_ESCALATION_PHRASES.search(msg_lower)
        if esc_general_match and any(k in msg_lower for k in ["sue", "lawyer", "police", "legal", "trading standards", "court"]):
            evidence.append(f"Regulatory/legal phrase match: '{esc_general_match.group(0)}'")
            return {
                "escalation": True,
                "reason": "SEVERE_FRUSTRATION_OR_THREAT",
                "confidence": 0.92,
                "evidence": evidence,
            }

        # 3. Check Payment Account Dispute
        pay_match = RE_PAYMENT_DISPUTE.search(msg)
        if pay_match:
            evidence.append(f"Payment dispute match: '{pay_match.group(0)}'")
            return {
                "escalation": True,
                "reason": "PAYMENT_ACCOUNT_DISPUTE",
                "confidence": 0.94,
                "evidence": evidence,
            }

        # 4. Check Repeated Failed Contact
        rep_match = RE_REPEATED_CONTACT.search(msg)
        if rep_match:
            evidence.append(f"Repeated contact failure match: '{rep_match.group(0)}'")
            return {
                "escalation": True,
                "reason": "REPEATED_FAILED_CONTACT",
                "confidence": 0.91,
                "evidence": evidence,
            }

        rep_general = RE_REPEATED_CONTACT_SIGNALS.search(msg_lower)
        if rep_general and turn_depth >= 2:
            evidence.append(f"Multi-turn repeated contact signal: '{rep_general.group(0)}'")
            return {
                "escalation": True,
                "reason": "REPEATED_FAILED_CONTACT",
                "confidence": 0.88,
                "evidence": evidence,
            }

        # 5. Check Out of Policy Request
        oop_match = RE_OUT_OF_POLICY.search(msg)
        if oop_match:
            evidence.append(f"Out of policy match: '{oop_match.group(0)}'")
            return {
                "escalation": True,
                "reason": "OUT_OF_POLICY_REQUEST",
                "confidence": 0.85,
                "evidence": evidence,
            }

        # Default: No Escalation
        return {
            "escalation": False,
            "reason": "NONE",
            "confidence": 0.95,
            "evidence": ["no_escalation_triggers_detected"],
        }
