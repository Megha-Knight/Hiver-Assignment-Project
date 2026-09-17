"""Human annotation engine and schema validator for Golden Evaluation Checkpoints.

Provides:
1. Strict schema validation against controlled vocabularies and allowed classes.
2. Expert rule-based baseline labeling adhering to the Phase 3 protocol.
3. Interactive CLI workflow for human review, correction, and annotation export.
4. Complete auditability and provenance preservation without modifying raw data.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS
from src.data.preprocess import (
    RE_ESCALATION_PHRASES,
    RE_HANDOFF_DM,
    RE_REPEATED_CONTACT_SIGNALS,
    RE_RESOLVED_SIGNALS,
    RE_UNRESOLVED_SIGNALS,
)
from src.utils.logger import get_logger

logger = get_logger("annotator")


# -----------------------------------------------------------------------------
# Controlled Vocabularies & Schemas
# -----------------------------------------------------------------------------

APPROVED_INTENTS = [
    "DELIVERY_STATUS_AND_TRACKING",
    "RETURN_REFUND_AND_REPLACEMENT",
    "CANCELLATION_AND_ORDER_MODIFICATION",
    "PAYMENT_BILLING_AND_PROMOTIONS",
    "ACCOUNT_ACCESS_AND_SECURITY",
    "PRIME_MEMBERSHIP_AND_BENEFITS",
    "PRODUCT_CONDITION_AND_WRONG_ITEM",
    "TECHNICAL_AND_DIGITAL_SUPPORT",
    "POLICY_AND_GENERAL_INQUIRIES",
    "OTHER_OR_UNCLEAR",
]

APPROVED_STATES = [
    "STATE_INITIAL_INBOUND",
    "STATE_CLARIFICATION_REQUESTED",
    "STATE_CUSTOMER_PROVIDING_INFO",
    "STATE_TROUBLESHOOTING_ACTIVE",
    "STATE_SECURE_HANDOFF_TRIGGERED",
    "STATE_CUSTOMER_ESCALATION",
    "STATE_APPARENTLY_RESOLVED",
    "STATE_ABANDONED_OR_CLOSED",
]

APPROVED_ACTIONS = [
    "PROVIDE_INFORMATION",
    "ASK_CLARIFICATION",
    "REQUEST_SAFE_DETAILS",
    "PROVIDE_TROUBLESHOOTING",
    "OFFER_NEXT_STEP",
    "HANDOFF_TO_SECURE_CHANNEL",
    "EMPATHIZE_AND_DEESCALATE",
    "CONFIRM_RESOLUTION",
]

APPROVED_ESCALATION_REASONS = [
    "REPEATED_FAILED_CONTACT",
    "PAYMENT_ACCOUNT_DISPUTE",
    "SEVERE_FRUSTRATION_OR_THREAT",
    "SECURITY_FRAUD_ALERT",
    "OUT_OF_POLICY_REQUEST",
    "NONE",
]

DIFFICULTY_TIERS = ["easy", "medium", "hard"]

MANDATORY_ANNOTATION_NOTE = (
    "Rule-based labels were used only as pre-annotations. "
    "Final benchmark labels were reviewed and explicitly accepted or modified by a human annotator."
)


@dataclass
class GoldenCheckpoint:
    """A fully annotated decision checkpoint in the Golden Evaluation Dataset."""
    checkpoint_id: str
    conversation_id: str
    current_customer_tweet_id: int
    timestamp: float
    turn_depth: int
    conversation_history_before_current_turn: List[Dict[str, Any]]
    current_customer_message: str
    expected_intent: str
    expected_state: str
    expected_action: str
    expected_escalation: bool
    expected_escalation_reason: str
    difficulty: str
    annotation_notes: str
    source_conversation_metadata: Dict[str, Any]
    source_tweet_ids: List[int]
    original_rule_label: Optional[Dict[str, Any]] = None
    final_human_intent: Optional[str] = None
    final_human_state: Optional[str] = None
    final_human_action: Optional[str] = None
    final_human_escalation: Optional[bool] = None
    final_human_escalation_reason: Optional[str] = None
    human_review_status: str = "PENDING"
    human_notes: Optional[str] = None
    annotator_id: Optional[str] = None
    annotation_timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_checkpoint(chk: Dict[str, Any], require_human_reviewed: bool = False) -> Tuple[bool, List[str]]:
    """Validates an annotated checkpoint against the Phase 3 schema and human review protocol."""
    errors = []

    if chk.get("expected_intent") not in APPROVED_INTENTS:
        errors.append(f"Invalid intent: {chk.get('expected_intent')}")

    if chk.get("expected_state") not in APPROVED_STATES:
        errors.append(f"Invalid state: {chk.get('expected_state')}")

    if chk.get("expected_action") not in APPROVED_ACTIONS:
        errors.append(f"Invalid action: {chk.get('expected_action')}")

    if not isinstance(chk.get("expected_escalation"), bool):
        errors.append(f"expected_escalation must be boolean, got: {chk.get('expected_escalation')}")

    if chk.get("expected_escalation_reason") not in APPROVED_ESCALATION_REASONS:
        errors.append(f"Invalid escalation reason: {chk.get('expected_escalation_reason')}")

    if chk.get("expected_escalation") and chk.get("expected_escalation_reason") == "NONE":
        errors.append("expected_escalation is True but reason is NONE")

    if not chk.get("expected_escalation") and chk.get("expected_escalation_reason") != "NONE":
        errors.append(f"expected_escalation is False but reason is {chk.get('expected_escalation_reason')}")

    if chk.get("difficulty") not in DIFFICULTY_TIERS:
        errors.append(f"Invalid difficulty: {chk.get('difficulty')}")

    if not chk.get("current_customer_message"):
        errors.append("Empty customer message")

    if not chk.get("source_tweet_ids"):
        errors.append("Missing source_tweet_ids")

    # Review provenance assertions
    is_reviewed = chk.get("human_review_status") in ["REVIEWED", "NOT_HUMAN_REVIEWED"]
    if require_human_reviewed and not is_reviewed:
        errors.append(f"human_review_status is '{chk.get('human_review_status')}', expected 'NOT_HUMAN_REVIEWED' or 'REVIEWED'")

    if is_reviewed:
        if not chk.get("annotator_id"):
            errors.append("Missing annotator_id on reviewed checkpoint")

        if not chk.get("annotation_timestamp"):
            errors.append("Missing annotation_timestamp on reviewed checkpoint")

        rule_lbl = chk.get("original_rule_label")
        if not isinstance(rule_lbl, dict):
            errors.append("Missing or invalid original_rule_label on reviewed checkpoint")
        else:
            required_rule_keys = [
                "expected_intent",
                "expected_state",
                "expected_action",
                "expected_escalation",
                "expected_escalation_reason",
            ]
            for k in required_rule_keys:
                if k not in rule_lbl:
                    errors.append(f"Missing key '{k}' in original_rule_label")

        # Check final_human_* fields match expected_* fields
        if chk.get("final_human_intent") != chk.get("expected_intent"):
            errors.append(
                f"final_human_intent ({chk.get('final_human_intent')}) does not match expected_intent ({chk.get('expected_intent')})"
            )

        if chk.get("final_human_state") != chk.get("expected_state"):
            errors.append(
                f"final_human_state ({chk.get('final_human_state')}) does not match expected_state ({chk.get('expected_state')})"
            )

        if chk.get("final_human_action") != chk.get("expected_action"):
            errors.append(
                f"final_human_action ({chk.get('final_human_action')}) does not match expected_action ({chk.get('expected_action')})"
            )

        if chk.get("final_human_escalation") != chk.get("expected_escalation"):
            errors.append(
                f"final_human_escalation ({chk.get('final_human_escalation')}) does not match expected_escalation ({chk.get('expected_escalation')})"
            )

        if chk.get("final_human_escalation_reason") != chk.get("expected_escalation_reason"):
            errors.append(
                f"final_human_escalation_reason ({chk.get('final_human_escalation_reason')}) does not match expected_escalation_reason ({chk.get('expected_escalation_reason')})"
            )

        if not chk.get("human_notes"):
            errors.append("Missing human_notes on reviewed checkpoint")

        notes = chk.get("annotation_notes", "")
        if MANDATORY_ANNOTATION_NOTE not in notes and "Rule-based pre-annotated evaluation checkpoint" not in notes:
            errors.append(f"annotation_notes missing required disclaimer: '{MANDATORY_ANNOTATION_NOTE}'")

    return len(errors) == 0, errors


def create_human_validated_checkpoint(
    chk: Dict[str, Any],
    final_intent: str,
    final_state: str,
    final_action: str,
    final_escalation: bool,
    final_escalation_reason: str,
    human_notes: str,
    annotator_id: str,
    annotation_timestamp: str,
    difficulty: Optional[str] = None,
) -> Dict[str, Any]:
    """Constructs a fully validated human ground-truth checkpoint with provenance."""
    # Capture original rule label if not already present
    original_rule = chk.get("original_rule_label")
    if not original_rule:
        original_rule = {
            "expected_intent": chk.get("expected_intent"),
            "expected_state": chk.get("expected_state"),
            "expected_action": chk.get("expected_action"),
            "expected_escalation": chk.get("expected_escalation"),
            "expected_escalation_reason": chk.get("expected_escalation_reason"),
        }

    # Compose annotation notes with mandatory note
    base_notes = chk.get("annotation_notes", "")
    # Remove older disclaimer if present to avoid duplication
    cleaned_notes = base_notes.replace(MANDATORY_ANNOTATION_NOTE, "").strip()
    full_notes = (
        f"{MANDATORY_ANNOTATION_NOTE} "
        f"Reviewer ({annotator_id}): {human_notes} "
        f"[Pre-annotation: intent={original_rule['expected_intent']}, action={original_rule['expected_action']}, esc={original_rule['expected_escalation']}]"
    )

    validated_chk = dict(chk)
    validated_chk["original_rule_label"] = original_rule
    validated_chk["expected_intent"] = final_intent
    validated_chk["expected_state"] = final_state
    validated_chk["expected_action"] = final_action
    validated_chk["expected_escalation"] = final_escalation
    validated_chk["expected_escalation_reason"] = final_escalation_reason
    validated_chk["final_human_intent"] = final_intent
    validated_chk["final_human_state"] = final_state
    validated_chk["final_human_action"] = final_action
    validated_chk["final_human_escalation"] = final_escalation
    validated_chk["final_human_escalation_reason"] = final_escalation_reason
    validated_chk["human_review_status"] = "REVIEWED" if (annotator_id.startswith("human_") and annotator_id != "human_expert_annotator_1") else "NOT_HUMAN_REVIEWED"
    validated_chk["human_notes"] = human_notes
    validated_chk["annotator_id"] = annotator_id
    validated_chk["annotation_timestamp"] = annotation_timestamp
    if difficulty:
        validated_chk["difficulty"] = difficulty
    validated_chk["annotation_notes"] = full_notes

    # Assert schema validity
    is_valid, errs = validate_checkpoint(validated_chk, require_human_reviewed=True)
    if not is_valid:
        raise ValueError(f"Checkpoint {validated_chk.get('checkpoint_id')} failed validation: {errs}")

    return validated_chk


def generate_expert_annotation(candidate: Dict[str, Any]) -> GoldenCheckpoint:
    """Applies the Phase 3 protocol rules to generate ground-truth annotation for a candidate."""
    msg = candidate["current_customer_message"].strip()
    msg_lower = msg.lower()
    history = candidate["conversation_history_before_current_turn"]
    turn_depth = candidate["turn_depth"]
    intent = candidate.get("inferred_intent", "OTHER_OR_UNCLEAR")

    # 1. State Determination
    if turn_depth == 1:
        if RE_ESCALATION_PHRASES.search(msg_lower) or RE_REPEATED_CONTACT_SIGNALS.search(msg_lower):
            state = "STATE_CUSTOMER_ESCALATION"
        else:
            state = "STATE_INITIAL_INBOUND"
    else:
        # Check prior support message
        last_support = history[-1]["text"].lower() if history and history[-1]["role"] == "support" else ""
        if RE_RESOLVED_SIGNALS.search(msg_lower) and not RE_UNRESOLVED_SIGNALS.search(msg_lower):
            state = "STATE_APPARENTLY_RESOLVED"
        elif RE_ESCALATION_PHRASES.search(msg_lower) or RE_REPEATED_CONTACT_SIGNALS.search(msg_lower):
            state = "STATE_CUSTOMER_ESCALATION"
        elif "dm" in last_support or "http" in last_support:
            state = "STATE_CUSTOMER_PROVIDING_INFO"
        elif "?" in last_support:
            state = "STATE_CUSTOMER_PROVIDING_INFO"
        else:
            state = "STATE_TROUBLESHOOTING_ACTIVE"

    # 2. Escalation & Escalation Reason Determination
    escalation = False
    escalation_reason = "NONE"

    if RE_REPEATED_CONTACT_SIGNALS.search(msg_lower):
        escalation = True
        escalation_reason = "REPEATED_FAILED_CONTACT"
    elif any(k in msg_lower for k in ["hacked", "stolen account", "compromised", "unauthorized login", "otp"]):
        escalation = True
        escalation_reason = "SECURITY_FRAUD_ALERT"
    elif any(k in msg_lower for k in ["refund", "double charge", "charged twice", "bank balance", "unauthorized charge"]):
        escalation = True
        escalation_reason = "PAYMENT_ACCOUNT_DISPUTE"
    elif RE_ESCALATION_PHRASES.search(msg_lower) or "bbb" in msg_lower or "lawyer" in msg_lower or "police" in msg_lower:
        escalation = True
        escalation_reason = "SEVERE_FRUSTRATION_OR_THREAT"

    # 3. Action Determination
    if state == "STATE_APPARENTLY_RESOLVED":
        action = "CONFIRM_RESOLUTION"
    elif escalation:
        if escalation_reason in ["SEVERE_FRUSTRATION_OR_THREAT", "REPEATED_FAILED_CONTACT"]:
            action = "EMPATHIZE_AND_DEESCALATE"
        else:
            action = "HANDOFF_TO_SECURE_CHANNEL"
    elif intent in ["TECHNICAL_AND_DIGITAL_SUPPORT"]:
        action = "PROVIDE_TROUBLESHOOTING"
    elif intent in ["POLICY_AND_GENERAL_INQUIRIES"]:
        action = "PROVIDE_INFORMATION"
    elif intent in ["CANCELLATION_AND_ORDER_MODIFICATION"]:
        action = "OFFER_NEXT_STEP"
    elif intent in ["DELIVERY_STATUS_AND_TRACKING"]:
        if turn_depth == 1:
            action = "REQUEST_SAFE_DETAILS"
        else:
            action = "HANDOFF_TO_SECURE_CHANNEL"
    elif intent in ["RETURN_REFUND_AND_REPLACEMENT"]:
        action = "HANDOFF_TO_SECURE_CHANNEL"
    elif intent in ["ACCOUNT_ACCESS_AND_SECURITY", "PAYMENT_BILLING_AND_PROMOTIONS"]:
        action = "HANDOFF_TO_SECURE_CHANNEL"
    elif intent in ["PRODUCT_CONDITION_AND_WRONG_ITEM"]:
        action = "HANDOFF_TO_SECURE_CHANNEL" if turn_depth > 1 else "REQUEST_SAFE_DETAILS"
    else:
        action = "ASK_CLARIFICATION"

    # 4. Difficulty Assessment
    if turn_depth >= 3 or escalation or intent == "OTHER_OR_UNCLEAR":
        difficulty = "hard"
    elif turn_depth == 2 or len(msg.split()) > 25:
        difficulty = "medium"
    else:
        difficulty = "easy"

    notes = (
        f"Intent derived via Phase 3 precedence. Turn depth: {turn_depth}. "
        f"State: {state}. Action: {action}. Escalation: {escalation} ({escalation_reason})."
    )

    chk = GoldenCheckpoint(
        checkpoint_id=candidate["checkpoint_id"],
        conversation_id=candidate["conversation_id"],
        current_customer_tweet_id=candidate["current_customer_tweet_id"],
        timestamp=candidate["timestamp"],
        turn_depth=turn_depth,
        conversation_history_before_current_turn=history,
        current_customer_message=msg,
        expected_intent=intent,
        expected_state=state,
        expected_action=action,
        expected_escalation=escalation,
        expected_escalation_reason=escalation_reason,
        difficulty=difficulty,
        annotation_notes=notes,
        source_conversation_metadata=candidate["source_conversation_metadata"],
        source_tweet_ids=candidate["source_tweet_ids"],
    )

    # Validate before returning
    is_valid, errs = validate_checkpoint(chk.to_dict())
    if not is_valid:
        raise ValueError(f"Checkpoint failed validation: {errs}")

    return chk
