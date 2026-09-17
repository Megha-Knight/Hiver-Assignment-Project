"""Production preprocessing and analysis pipeline for AmazonHelp conversations.

Implements:
1. Participant role normalization (customer, support) with validation against inbound field
2. Structural validation checks (duplicates, missing text, ordering, impossible sequences)
3. Multi-part support response grouping with audit trail
4. Language detection and confidence scoring via langdetect
5. Conservative resolution status heuristics (UNKNOWN, APPARENTLY_RESOLVED, UNRESOLVED, HANDOFF_OR_DM, INSUFFICIENT_EVIDENCE)
6. Dataset tier assignment (Exploration, Retrieval Candidates, Pristine Evaluation Candidates)
"""

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import langdetect
from langdetect import DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

from src.config import PIPELINE_CONFIG
from src.data.reconstruct import ConversationTurn, ReconstructedConversation
from src.utils.logger import get_logger

# Enforce deterministic language detection across runs
DetectorFactory.seed = PIPELINE_CONFIG.SEED

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Heuristic Patterns
# -----------------------------------------------------------------------------

RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
RE_MENTION = re.compile(r"@\w+")
RE_WHITESPACE = re.compile(r"\s+")

# Multi-part indicators (e.g., "1/2", "(1/2)", "part 1/2", "1 of 2")
RE_MULTIPART_INDICATOR = re.compile(
    r"(?:\b|\()([1-9])\s*[/of]\s*([1-9])(?:\b|\))|\bpart\s*([1-9])\b",
    re.IGNORECASE,
)

# Hand-off / DM routing cues in support messages
RE_HANDOFF_DM = re.compile(
    r"(?:\b(?:dm|direct\s*message|p\.?m\.?|private\s*message)\b|"
    r"amzn\.to/(?:dm|help|contact|directmessage|mail)|"
    r"reach\s*out\s*(?:to\s*us\s*)?(?:via|in|through)\s*(?:dm|direct\s*message)|"
    r"send\s*(?:us\s*)?(?:a\s*)?(?:dm|direct\s*message)|"
    r"contact\s*(?:our\s*)?(?:support|team|chat|phone|customer\s*service)|"
    r"click\s*(?:the\s*link|here)\s*to\s*(?:connect|reach|chat|contact))",
    re.IGNORECASE,
)

# Apparent customer resolution / satisfaction signals
RE_RESOLVED_SIGNALS = re.compile(
    r"\b(?:thank\s*you|thanks|thx|appreciate\s*(?:it|your\s*help)|"
    r"that\s*worked|worked\s*great|fixed\s*now|sorted\s*now|resolved\s*now|"
    r"got\s*it\s*working|all\s*set|good\s*now|helped\s*a\s*lot|perfect\s*now)\b",
    re.IGNORECASE,
)

# Customer unaddressed complaint / frustration signals
RE_UNRESOLVED_SIGNALS = re.compile(
    r"\b(?:still\s*(?:waiting|not\s*working|haven't\s*received|broken|delayed|no\s*refund|no\s*response)|"
    r"worst\s*(?:service|experience)|useless|ridiculous|unacceptable|unhelpful|nobody\s*helped|"
    r"fraud|scam|terrible|horrible|waste\s*of\s*time|never\s*buying\s*again)\b",
    re.IGNORECASE,
)

# Escalation / supervisor phrases
RE_ESCALATION_PHRASES = re.compile(
    r"\b(?:manager|supervisor|escalat(?:e|ed|ion)|legal\s*action|lawyer|lawsuit|"
    r"fraud|police|file\s*a\s*complaint|better\s*business\s*bureau|bbb|cancel\s*(?:prime|account|order))\b",
    re.IGNORECASE,
)

# Repeated-contact signals
RE_REPEATED_CONTACT_SIGNALS = re.compile(
    r"\b(?:already\s*(?:called|contacted|spoken|dm(?:'d|ed)?|emailed)|"
    r"second\s*time|third\s*time|fourth\s*time|multiple\s*times|again\s*and\s*again|"
    r"spoke\s*to\s*(?:someone|an?\s*agent|support)|called\s*(?:yesterday|earlier|support)|"
    r"still\s*(?:waiting|pending)|no\s*one\s*(?:called|replied|helped))\b",
    re.IGNORECASE,
)


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------

@dataclass
class NormalizedTurn:
    """A logical turn in a conversation, potentially merging multi-part messages."""
    turn_index: int
    role: str  # 'customer' or 'support'
    text: str
    tweet_ids: List[int]
    created_at_first: str
    timestamp_first: float
    timestamp_last: float
    is_grouped: bool = False
    grouped_message_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "role": self.role,
            "text": self.text,
            "tweet_ids": self.tweet_ids,
            "created_at": self.created_at_first,
            "timestamp": self.timestamp_first,
            "timestamp_last": self.timestamp_last,
            "is_grouped": self.is_grouped,
            "grouped_message_count": self.grouped_message_count,
        }


@dataclass
class ConversationValidation:
    """Results of defensive validation checks on a reconstructed thread."""
    is_valid: bool
    has_consecutive_duplicates: bool = False
    has_impossible_role_sequence: bool = False
    has_missing_text: bool = False
    has_duplicate_tweet_ids: bool = False
    has_timestamp_order_error: bool = False
    rejection_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessedConversation:
    """Production representation of a processed conversation thread."""
    conversation_id: str
    brand: str
    original_tweet_ids: List[int]
    start_timestamp: float
    end_timestamp: float
    duration_seconds: float
    is_broken: bool
    broken_reasons: List[str]
    has_cycle: bool
    language: str
    language_confidence: float
    is_english: bool
    is_uncertain_language: bool
    resolution_status: str  # UNKNOWN, APPARENTLY_RESOLVED, UNRESOLVED, HANDOFF_OR_DM, INSUFFICIENT_EVIDENCE
    resolution_evidence: str
    validation: ConversationValidation
    dataset_tiers: List[str]  # e.g., ["exploration", "retrieval", "pristine"]
    original_turns: List[Dict[str, Any]]
    normalized_turns: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "brand": self.brand,
            "original_tweet_ids": self.original_tweet_ids,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "duration_seconds": round(self.duration_seconds, 1),
            "is_broken": self.is_broken,
            "broken_reasons": self.broken_reasons,
            "has_cycle": self.has_cycle,
            "language": self.language,
            "language_confidence": round(self.language_confidence, 4),
            "is_english": self.is_english,
            "is_uncertain_language": self.is_uncertain_language,
            "resolution_status": self.resolution_status,
            "resolution_evidence": self.resolution_evidence,
            "validation": self.validation.to_dict(),
            "dataset_tiers": self.dataset_tiers,
            "original_turns": self.original_turns,
            "normalized_turns": self.normalized_turns,
        }


# -----------------------------------------------------------------------------
# Normalization & Validation Functions
# -----------------------------------------------------------------------------

def clean_text_for_langdetect(text: str) -> str:
    """Strips URLs, user mentions, and normalizes whitespace for reliable language detection."""
    text = RE_URL.sub("", text)
    text = RE_MENTION.sub("", text)
    text = RE_WHITESPACE.sub(" ", text).strip()
    return text


def detect_language(text: str, english_threshold: float = 0.80) -> Tuple[str, float, bool, bool]:
    """Detects text language using langdetect.

    Returns:
        (detected_language, confidence, is_english, is_uncertain)
    """
    cleaned = clean_text_for_langdetect(text)
    # If text is too short or lacks alphabetical words, mark as uncertain
    if len(re.sub(r"[^a-zA-Z]", "", cleaned)) < 10:
        return "uncertain", 0.0, False, True

    try:
        predictions = langdetect.detect_langs(cleaned)
        if not predictions:
            return "uncertain", 0.0, False, True

        top_pred = predictions[0]
        lang = top_pred.lang
        prob = top_pred.prob

        if lang == "en" and prob >= english_threshold:
            return "en", prob, True, False
        elif lang != "en" and prob >= english_threshold:
            return lang, prob, False, False
        else:
            return lang, prob, False, True
    except LangDetectException:
        return "uncertain", 0.0, False, True
    except Exception:
        return "uncertain", 0.0, False, True


def validate_conversation_structure(
    turns: List[ConversationTurn],
    brand: str = "AmazonHelp",
) -> ConversationValidation:
    """Executes defensive validation checks on a reconstructed conversation thread."""
    rejection_reasons = []
    has_consecutive_duplicates = False
    has_impossible_role_sequence = False
    has_missing_text = False
    has_duplicate_tweet_ids = False
    has_timestamp_order_error = False

    if not turns:
        rejection_reasons.append("empty_turn_list")
        return ConversationValidation(
            is_valid=False,
            has_impossible_role_sequence=True,
            rejection_reasons=rejection_reasons,
        )

    # 1. Check duplicate tweet IDs
    seen_tweet_ids = set()
    for t in turns:
        if t.tweet_id in seen_tweet_ids:
            has_duplicate_tweet_ids = True
            rejection_reasons.append(f"duplicate_tweet_id_{t.tweet_id}")
        seen_tweet_ids.add(t.tweet_id)

    # 2. Check missing text
    for t in turns:
        if not t.text or not str(t.text).strip():
            has_missing_text = True
            rejection_reasons.append(f"missing_text_in_turn_{t.turn_index}")

    # 3. Check chronological timestamp ordering
    for i in range(len(turns) - 1):
        if turns[i + 1].timestamp < turns[i].timestamp:
            has_timestamp_order_error = True
            rejection_reasons.append(
                f"timestamp_order_violation_{turns[i].turn_index}_vs_{turns[i+1].turn_index}"
            )

    # 4. Check consecutive duplicate turns (identical text from same author)
    for i in range(len(turns) - 1):
        t1, t2 = turns[i], turns[i + 1]
        if t1.role == t2.role:
            c1 = clean_text_for_langdetect(t1.text).lower()
            c2 = clean_text_for_langdetect(t2.text).lower()
            if c1 and c1 == c2:
                has_consecutive_duplicates = True
                rejection_reasons.append(f"consecutive_duplicate_text_turns_{t1.turn_index}_{t2.turn_index}")

    # 5. Check impossible role assignment
    # Support role must come from the target brand or author acting as brand
    for t in turns:
        if t.role == "support" and t.author_id != brand and not t.author_id.startswith(brand):
            # Outbound turn from non-brand author
            has_impossible_role_sequence = True
            rejection_reasons.append(f"invalid_support_author_{t.author_id}")

    is_valid = (
        not has_duplicate_tweet_ids
        and not has_missing_text
        and not has_timestamp_order_error
        and not has_impossible_role_sequence
    )

    return ConversationValidation(
        is_valid=is_valid,
        has_consecutive_duplicates=has_consecutive_duplicates,
        has_impossible_role_sequence=has_impossible_role_sequence,
        has_missing_text=has_missing_text,
        has_duplicate_tweet_ids=has_duplicate_tweet_ids,
        has_timestamp_order_error=has_timestamp_order_error,
        rejection_reasons=rejection_reasons,
    )


def group_multipart_support_turns(
    turns: List[ConversationTurn],
    max_gap_seconds: float = 300.0,
) -> List[NormalizedTurn]:
    """Conservatively groups consecutive support messages into a single logical turn.

    Conditions for grouping:
    1. Both turns are sent by 'support'
    2. No customer message exists between them
    3. The time delta is <= max_gap_seconds OR explicit multi-part indicator (e.g. 1/2) matches
    """
    if not turns:
        return []

    normalized: List[NormalizedTurn] = []
    i = 0
    curr_index = 1

    while i < len(turns):
        curr_turn = turns[i]
        
        # If customer, keep as individual turn
        if curr_turn.role == "customer":
            normalized.append(
                NormalizedTurn(
                    turn_index=curr_index,
                    role="customer",
                    text=curr_turn.text,
                    tweet_ids=[curr_turn.tweet_id],
                    created_at_first=curr_turn.created_at,
                    timestamp_first=curr_turn.timestamp,
                    timestamp_last=curr_turn.timestamp,
                    is_grouped=False,
                    grouped_message_count=1,
                )
            )
            curr_index += 1
            i += 1
            continue

        # If support, check for consecutive support messages
        grouped_texts = [curr_turn.text]
        grouped_ids = [curr_turn.tweet_id]
        first_created_at = curr_turn.created_at
        first_ts = curr_turn.timestamp
        last_ts = curr_turn.timestamp

        j = i + 1
        while j < len(turns) and turns[j].role == "support":
            next_turn = turns[j]
            gap = max(0.0, next_turn.timestamp - last_ts)

            # Check if explicit multipart marker exists or within short time gap
            has_multipart_marker = bool(
                RE_MULTIPART_INDICATOR.search(curr_turn.text)
                or RE_MULTIPART_INDICATOR.search(next_turn.text)
            )

            if gap <= max_gap_seconds or has_multipart_marker:
                grouped_texts.append(next_turn.text)
                grouped_ids.append(next_turn.tweet_id)
                last_ts = next_turn.timestamp
                j += 1
            else:
                # Exceeded time gap and not multipart -> separate message sequence
                break

        is_grouped = len(grouped_ids) > 1
        combined_text = "\n".join(grouped_texts)

        normalized.append(
            NormalizedTurn(
                turn_index=curr_index,
                role="support",
                text=combined_text,
                tweet_ids=grouped_ids,
                created_at_first=first_created_at,
                timestamp_first=first_ts,
                timestamp_last=last_ts,
                is_grouped=is_grouped,
                grouped_message_count=len(grouped_ids),
            )
        )
        curr_index += 1
        i = j

    return normalized


def determine_resolution_status(
    normalized_turns: List[NormalizedTurn],
) -> Tuple[str, str]:
    """Assigns conservative resolution status based on observable conversation evidence.

    Returns:
        (resolution_status, evidence_rule)
    """
    if len(normalized_turns) < 2:
        return "INSUFFICIENT_EVIDENCE", "conversation_has_less_than_two_turns"

    customer_turns = [t for t in normalized_turns if t.role == "customer"]
    support_turns = [t for t in normalized_turns if t.role == "support"]

    if not customer_turns or not support_turns:
        return "INSUFFICIENT_EVIDENCE", "missing_either_customer_or_support_turn"

    last_turn = normalized_turns[-1]
    last_support_turn = support_turns[-1]
    last_customer_turn = customer_turns[-1]

    # Rule 1: DM / External Handoff
    # If the last support message (or the conversation end) explicitly routes to DM / Help Portal
    if RE_HANDOFF_DM.search(last_support_turn.text):
        if last_turn.role == "support" or (
            last_turn.role == "customer"
            and not RE_RESOLVED_SIGNALS.search(last_turn.text)
            and len(last_turn.text.split()) <= 8
        ):
            return "HANDOFF_OR_DM", "support_directed_to_dm_or_external_channel"

    # Rule 2: Apparently Resolved
    # The conversation concludes with a clear customer satisfaction / acknowledgment signal
    if last_turn.role == "customer":
        if RE_RESOLVED_SIGNALS.search(last_turn.text) and not RE_UNRESOLVED_SIGNALS.search(last_turn.text):
            return "APPARENTLY_RESOLVED", "customer_expressed_gratitude_or_resolution"

    # Also if second to last was customer satisfaction and last was support polite sign-off
    if len(normalized_turns) >= 3 and last_turn.role == "support":
        prev_customer = normalized_turns[-2]
        if prev_customer.role == "customer" and RE_RESOLVED_SIGNALS.search(prev_customer.text):
            if not RE_UNRESOLVED_SIGNALS.search(prev_customer.text):
                return "APPARENTLY_RESOLVED", "customer_acknowledged_resolution_before_signoff"

    # Rule 3: Unresolved
    # Conversation terminates with customer expressing frustration or unresolved issue with no reply
    if last_turn.role == "customer":
        if RE_UNRESOLVED_SIGNALS.search(last_turn.text):
            return "UNRESOLVED", "customer_expressed_unresolved_frustration_at_close"
        # Customer asked a question or stated an issue, and support never replied
        return "UNRESOLVED", "unanswered_terminal_customer_turn"

    # Rule 4: Unknown
    # Terminal support message without DM link (e.g. asking for order number, and customer dropped off)
    return "UNKNOWN", "interaction_ended_without_explicit_resolution_or_dm_cue"


# -----------------------------------------------------------------------------
# Main Preprocessor Class
# -----------------------------------------------------------------------------

class ConversationPreprocessor:
    """Preprocesses reconstructed conversation threads into production tiers."""

    def __init__(
        self,
        brand: str = "AmazonHelp",
        english_threshold: float = PIPELINE_CONFIG.ENGLISH_CONFIDENCE_THRESHOLD,
        max_multipart_gap: float = PIPELINE_CONFIG.MULTI_PART_MAX_GAP_SECONDS,
    ):
        self.brand = brand
        self.english_threshold = english_threshold
        self.max_multipart_gap = max_multipart_gap

    def process_conversation(
        self,
        raw_conv: ReconstructedConversation,
    ) -> ProcessedConversation:
        """Processes a single reconstructed conversation with validation and normalization."""
        # 1. Structural validation
        val = validate_conversation_structure(raw_conv.turns, brand=self.brand)

        # 2. Multi-part support grouping
        normalized = group_multipart_support_turns(
            raw_conv.turns,
            max_gap_seconds=self.max_multipart_gap,
        )

        # 3. Language detection: combine customer text (where issues are articulated)
        customer_texts = [t.text for t in raw_conv.turns if t.role == "customer"]
        combined_text = " ".join(customer_texts) if customer_texts else " ".join(t.text for t in raw_conv.turns)

        lang, conf, is_en, is_unc = detect_language(combined_text, self.english_threshold)

        # 4. Determine resolution status
        res_status, res_evidence = determine_resolution_status(normalized)

        # 5. Classify into dataset tiers
        dataset_tiers = []

        # Tier 1: Exploration Dataset (all valid English conversations, including broken ones)
        if is_en and not is_unc:
            dataset_tiers.append("exploration")

        # Tier 2: Retrieval Candidate Dataset
        # Requirements: English, turn_count >= 2, at least 1 customer and 1 support turn
        has_cust = any(t.role == "customer" for t in normalized)
        has_supp = any(t.role == "support" for t in normalized)
        has_actionable_resolution = res_status in [
            "APPARENTLY_RESOLVED",
            "HANDOFF_OR_DM",
            "UNKNOWN",
            "UNRESOLVED",
        ]

        if is_en and not is_unc and len(normalized) >= PIPELINE_CONFIG.MIN_RETRIEVAL_TURNS and has_cust and has_supp:
            # If broken, allow only if it contains usable resolution evidence
            if not raw_conv.is_broken or has_actionable_resolution:
                dataset_tiers.append("retrieval_candidate")

        # Tier 3: Pristine Evaluation Candidate Dataset
        # Requirements: English, unbroken, no structural validation errors, valid timestamps, >=1 cust, >=1 supp
        if (
            is_en
            and not is_unc
            and not raw_conv.is_broken
            and val.is_valid
            and has_cust
            and has_supp
            and not raw_conv.has_cycle
        ):
            dataset_tiers.append("pristine_candidate")

        # Format original turns for output
        original_turns_dict = [t.to_dict() for t in raw_conv.turns]
        normalized_turns_dict = [t.to_dict() for t in normalized]

        start_ts = raw_conv.turns[0].timestamp if raw_conv.turns else 0.0
        end_ts = raw_conv.turns[-1].timestamp if raw_conv.turns else 0.0
        orig_tweet_ids = [t.tweet_id for t in raw_conv.turns]

        return ProcessedConversation(
            conversation_id=raw_conv.conversation_id,
            brand=self.brand,
            original_tweet_ids=orig_tweet_ids,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            duration_seconds=raw_conv.duration_seconds,
            is_broken=raw_conv.is_broken,
            broken_reasons=raw_conv.broken_reasons,
            has_cycle=raw_conv.has_cycle,
            language=lang,
            language_confidence=conf,
            is_english=is_en,
            is_uncertain_language=is_unc,
            resolution_status=res_status,
            resolution_evidence=res_evidence,
            validation=val,
            dataset_tiers=dataset_tiers,
            original_turns=original_turns_dict,
            normalized_turns=normalized_turns_dict,
        )
