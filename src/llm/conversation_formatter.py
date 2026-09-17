"""Context-aware conversation formatter for Phase 6 LLM customer-support agent.

Features:
1. Twitter noise sanitization (removes @handles, t.co URLs, excess whitespace).
2. Entity signal preservation (order IDs, postcodes, carrier names, error codes).
3. Windowed multi-turn context (configurable recent turn window, default 4 turns).
4. Zero target label or evaluation metadata leakage.
"""

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional

# Regex patterns for cleaning Twitter noise
RE_TWITTER_HANDLE = re.compile(r"@[A-Za-z0-9_]+", re.UNICODE)
RE_URL = re.compile(r"https?://\S+|www\.\S+", re.UNICODE)
RE_EXCESS_WHITESPACE = re.compile(r"\s+", re.UNICODE)

# Entity preservation patterns (for signal verification)
RE_ORDER_ID = re.compile(r"\b\d{3}-\d{7}-\d{7}\b")
RE_POSTCODE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.IGNORECASE)


def clean_text_content(text: str) -> str:
    """Sanitizes raw tweet text while preserving alphanumeric tokens and entity cues."""
    if not text:
        return ""
    # Strip user handles and URLs
    cleaned = RE_TWITTER_HANDLE.sub("", text)
    cleaned = RE_URL.sub("", cleaned)
    # Collapse whitespace
    cleaned = RE_EXCESS_WHITESPACE.sub(" ", cleaned).strip()
    return cleaned


@dataclass
class FormattedConversation:
    """Structured container for formatted dialogue ready for model prompt."""

    current_customer_message: str
    cleaned_customer_message: str
    turn_depth: int
    history_turns_included: int
    formatted_prompt_context: str
    detected_entities: List[str]


class ConversationFormatter:
    """Formats multi-turn support conversations into concise, clean prompt contexts."""

    def __init__(self, max_history_turns: int = 4):
        self.max_history_turns = max_history_turns

    def format_turn(
        self,
        customer_message: str,
        history: Optional[List[Dict[str, Any]]] = None,
        turn_depth: int = 1,
    ) -> FormattedConversation:
        """Formats conversation context for LLM input without leaking future or evaluation data."""
        cleaned_msg = clean_text_content(customer_message)
        history = history or []

        # Extract entities for audit
        entities = []
        if RE_ORDER_ID.search(customer_message):
            entities.append("ORDER_ID")
        if RE_POSTCODE.search(customer_message):
            entities.append("POSTCODE")
        for carrier in ["royal mail", "hermes", "dpd", "usps", "ups", "fedex", "dhl", "yodel"]:
            if carrier in customer_message.lower():
                entities.append(f"CARRIER_{carrier.upper()}")

        # Select window of recent turns
        recent_history = history[-self.max_history_turns :] if history else []

        lines = []
        if recent_history:
            lines.append("### CONVERSATION HISTORY:")
            for turn in recent_history:
                raw_role = turn.get("role", "")
                role_label = "Customer" if raw_role == "customer" else "AmazonHelp"
                turn_text = clean_text_content(turn.get("text", turn.get("content", "")))
                if turn_text:
                    lines.append(f"{role_label}: {turn_text}")
            lines.append("")

        lines.append(f"### CURRENT TURN (Turn {turn_depth}):")
        lines.append(f"Customer: {cleaned_msg}")
        lines.append("")
        lines.append("Analyze the conversation and output your structured JSON decision object:")

        formatted_context = "\n".join(lines)

        return FormattedConversation(
            current_customer_message=customer_message,
            cleaned_customer_message=cleaned_msg,
            turn_depth=turn_depth,
            history_turns_included=len(recent_history),
            formatted_prompt_context=formatted_context,
            detected_entities=entities,
        )
