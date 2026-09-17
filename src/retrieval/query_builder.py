"""Deterministic Retrieval Query Construction Engine for AmazonHelp Conversations.

Implements structured, context-aware query building prioritizing:
1. Current customer message (core intent and grievance)
2. Relevant recent support context (avoiding noise from early turns)
3. Extracted entity cues (order numbers, couriers, tracking signals)
4. Context typing (single-turn, clarification, troubleshooting, escalation)
"""

from dataclasses import asdict, dataclass
import re
from typing import Any, Dict, List, Optional

RE_USER_HANDLES = re.compile(r"@[A-Za-z0-9_]+", re.IGNORECASE)
RE_ORDER_ID = re.compile(r"\b\d{3}-\d{7}-\d{7}\b")
RE_COURIERS = re.compile(
    r"\b(hermes|dpd|royal mail|ats|dtdc|fedex|ups|usps|yodel|amazon logistics)\b",
    re.IGNORECASE,
)
RE_URLS = re.compile(r"https?://\S+")


@dataclass
class RetrievalQuery:
    """Encapsulates deterministic query text and structured provenance metadata."""
    query_text: str
    cleaned_customer_message: str
    prior_support_context: Optional[str]
    detected_entities: List[str]
    turn_depth: int
    context_type: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def clean_tweet_text(text: str) -> str:
    """Removes Twitter user handles and URLs to maximize embedding semantic signal."""
    cleaned = RE_USER_HANDLES.sub("", text)
    cleaned = RE_URLS.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def extract_entities(text: str) -> List[str]:
    """Extracts explicit order IDs and carrier mentions."""
    entities = []
    order_matches = RE_ORDER_ID.findall(text)
    for om in order_matches:
        entities.append(f"order_id:{om}")

    courier_matches = RE_COURIERS.findall(text)
    for cm in courier_matches:
        entities.append(f"courier:{cm.lower()}")

    return entities


def build_retrieval_query(
    customer_message: str,
    history: Optional[List[Dict[str, Any]]] = None,
    turn_depth: int = 1,
) -> RetrievalQuery:
    """Constructs a deterministic query representation from current message and dialogue history."""
    raw_msg = customer_message.strip()
    cleaned_msg = clean_tweet_text(raw_msg)
    entities = extract_entities(raw_msg)
    history = history or []

    # 1. Single-Turn Query
    if turn_depth <= 1 or not history:
        query_text = cleaned_msg
        context_type = "single_turn"
        prior_support_context = None

        return RetrievalQuery(
            query_text=query_text,
            cleaned_customer_message=cleaned_msg,
            prior_support_context=prior_support_context,
            detected_entities=entities,
            turn_depth=turn_depth,
            context_type=context_type,
        )

    # 2. Multi-Turn Query: Extract immediate prior support utterance
    last_support_raw = ""
    for h in reversed(history):
        if h.get("role") == "support":
            last_support_raw = h.get("text", "")
            break

    cleaned_support = clean_tweet_text(last_support_raw)
    prior_support_context = cleaned_support if cleaned_support else None

    # Determine Context Type
    last_support_lower = last_support_raw.lower()
    msg_lower = raw_msg.lower()

    if any(k in msg_lower for k in ["lawyer", "police", "trading standards", "court", "piss poor", "sue", "unacceptable"]):
        context_type = "escalation"
    elif "?" in last_support_raw or "let us know" in last_support_lower:
        context_type = "clarification"
    elif any(k in last_support_lower for k in ["restart", "unplug", "settings", "firmware", "app"]):
        context_type = "troubleshooting"
    else:
        context_type = "followup"

    # Construct weighted multi-turn representation
    # Customer message takes priority, contextualized with what support asked
    if cleaned_support:
        # Keep support context concise (up to 120 chars) to prevent query drift
        support_snippet = cleaned_support[:120].strip()
        query_text = f"Support: {support_snippet} | Customer: {cleaned_msg}"
    else:
        query_text = cleaned_msg

    return RetrievalQuery(
        query_text=query_text,
        cleaned_customer_message=cleaned_msg,
        prior_support_context=prior_support_context,
        detected_entities=entities,
        turn_depth=turn_depth,
        context_type=context_type,
    )
