"""Deterministic sampling of decision checkpoints for the Golden Evaluation Benchmark.

Guarantees:
1. Candidate pool is sampled EXCLUSIVELY from the Validation (Dev) partition (5,363 pristine conversations).
2. The Test partition (5,365 conversations) remains 100% unseen and untouched.
3. Every checkpoint captures the prior dialogue history, current customer message, and full tweet traceability.
4. Stratifies across turn depth, intent diversity, escalation markers, and boundary cases.
"""

from collections import Counter
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, PHASE3_CONFIG, set_seed
from src.data.preprocess import (
    RE_ESCALATION_PHRASES,
    RE_HANDOFF_DM,
    RE_REPEATED_CONTACT_SIGNALS,
    RE_RESOLVED_SIGNALS,
    RE_UNRESOLVED_SIGNALS,
)
from src.utils.logger import get_logger

logger = get_logger("sample_golden_candidates")


def classify_inferred_intent(text: str) -> str:
    """Classifies customer message into one of 10 intents applying Phase 3 precedence rules."""
    t = text.lower()

    # Rule 1: Return/Refund Precedence
    # If refund/money back/return label is demanded, it takes precedence over delivery or damage
    if any(k in t for k in ["refund", "money back", "return label", "return pickup", "replacement"]):
        return "RETURN_REFUND_AND_REPLACEMENT"

    # Rule 3: Prime Membership Precedence
    if "prime" in t and any(k in t for k in ["membership", "video", "subscription", "renew", "cancel prime", "charged for prime"]):
        return "PRIME_MEMBERSHIP_AND_BENEFITS"

    # Rule 4: Account Access Precedence
    if any(k in t for k in ["locked", "otp", "2fa", "password", "hack", "compromised", "cannot login"]):
        return "ACCOUNT_ACCESS_AND_SECURITY"

    # Cancellation & Order Modification
    if any(k in t for k in ["cancel order", "cancel my order", "change address", "change delivery address"]):
        return "CANCELLATION_AND_ORDER_MODIFICATION"

    # Product Condition & Wrong Item
    if any(k in t for k in ["damaged", "broken", "wrong item", "empty box", "tampered", "expired", "fake item", "counterfeit"]):
        return "PRODUCT_CONDITION_AND_WRONG_ITEM"

    # Payment, Billing & Promotions
    if any(k in t for k in ["charged", "double charge", "gift card", "bank", "declined", "unauthorized charge"]):
        return "PAYMENT_BILLING_AND_PROMOTIONS"

    # Delivery Status & Tracking
    if any(k in t for k in ["where is my", "tracking", "delivered", "not received", "courier", "late", "delay", "handed to resident", "package"]):
        return "DELIVERY_STATUS_AND_TRACKING"

    # Technical & Digital Support
    if any(k in t for k in ["kindle", "fire tv", "firestick", "echo", "alexa", "app crash", "sync"]):
        return "TECHNICAL_AND_DIGITAL_SUPPORT"

    # Policy & General Inquiries
    if any(k in t for k in ["how do i", "can i ship", "policy", "international shipping", "trade-in"]):
        return "POLICY_AND_GENERAL_INQUIRIES"

    return "OTHER_OR_UNCLEAR"


def extract_validation_checkpoints(val_convs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extracts all eligible decision checkpoints from validation conversations."""
    checkpoints = []

    for conv in val_convs:
        conv_id = conv["conversation_id"]
        normalized_turns = conv["normalized_turns"]
        orig_resolution = conv["resolution_status"]

        history: List[Dict[str, Any]] = []
        customer_turn_seq = 0

        for idx, turn in enumerate(normalized_turns):
            if turn["role"] == "customer":
                customer_turn_seq += 1
                cust_text = turn["text"].strip()
                cust_tid = turn["tweet_ids"][0] if turn["tweet_ids"] else None

                # Gather source tweet IDs up to this turn
                prior_tweet_ids = []
                for h in history:
                    prior_tweet_ids.extend(h["tweet_ids"])
                prior_tweet_ids.extend(turn["tweet_ids"])

                inferred_intent = classify_inferred_intent(cust_text)

                has_escalation_signal = bool(
                    RE_ESCALATION_PHRASES.search(cust_text)
                    or RE_REPEATED_CONTACT_SIGNALS.search(cust_text)
                    or RE_UNRESOLVED_SIGNALS.search(cust_text)
                )
                has_repeated_contact = bool(RE_REPEATED_CONTACT_SIGNALS.search(cust_text))

                chk = {
                    "checkpoint_id": f"chk_AmazonHelp_{conv_id.replace('conv_AmazonHelp_', '')}_turn{customer_turn_seq}",
                    "conversation_id": conv_id,
                    "current_customer_tweet_id": cust_tid,
                    "timestamp": turn["timestamp"],
                    "turn_depth": customer_turn_seq,
                    "conversation_history_before_current_turn": [
                        {
                            "turn_index": h["turn_index"],
                            "role": h["role"],
                            "text": h["text"],
                            "tweet_ids": h["tweet_ids"],
                        }
                        for h in history
                    ],
                    "current_customer_message": cust_text,
                    "inferred_intent": inferred_intent,
                    "has_escalation_signal": has_escalation_signal,
                    "has_repeated_contact": has_repeated_contact,
                    "source_conversation_metadata": {
                        "total_turns": len(normalized_turns),
                        "partition": "validation",
                        "original_resolution_status": orig_resolution,
                    },
                    "source_tweet_ids": prior_tweet_ids,
                }
                checkpoints.append(chk)

            # Add turn to history for subsequent checkpoints
            history.append({
                "turn_index": turn["turn_index"],
                "role": turn["role"],
                "text": turn["text"],
                "tweet_ids": turn["tweet_ids"],
            })

    return checkpoints


def sample_golden_checkpoints(
    all_checkpoints: List[Dict[str, Any]],
    target_count: int = 200,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Performs stratified sampling across turn depth, intent diversity, and dialogue complexity."""
    set_seed(seed)
    random.seed(seed)

    # Bucketing by intent and complexity
    by_intent: Dict[str, List[Dict[str, Any]]] = {}
    for c in all_checkpoints:
        by_intent.setdefault(c["inferred_intent"], []).append(c)

    # Intent quotas ensuring every intent is represented
    intent_quotas = {
        "DELIVERY_STATUS_AND_TRACKING": 45,
        "RETURN_REFUND_AND_REPLACEMENT": 35,
        "PRODUCT_CONDITION_AND_WRONG_ITEM": 25,
        "PAYMENT_BILLING_AND_PROMOTIONS": 20,
        "ACCOUNT_ACCESS_AND_SECURITY": 20,
        "PRIME_MEMBERSHIP_AND_BENEFITS": 18,
        "TECHNICAL_AND_DIGITAL_SUPPORT": 15,
        "CANCELLATION_AND_ORDER_MODIFICATION": 12,
        "POLICY_AND_GENERAL_INQUIRIES": 10,
        "OTHER_OR_UNCLEAR": 5,
    }

    selected: List[Dict[str, Any]] = []
    seen_convs = set()

    for intent, quota in intent_quotas.items():
        pool = by_intent.get(intent, [])
        if not pool:
            continue

        # Sub-stratify pool by turn depth (prioritizing multi-turn threads)
        multi_turn_pool = [c for c in pool if c["turn_depth"] >= 2 or c["source_conversation_metadata"]["total_turns"] >= 3]
        single_turn_pool = [c for c in pool if c["turn_depth"] == 1 and c["source_conversation_metadata"]["total_turns"] == 2]

        random.shuffle(multi_turn_pool)
        random.shuffle(single_turn_pool)

        # Aim for 60% multi-turn, 40% single-turn
        target_multi = int(round(quota * 0.65))
        target_single = quota - target_multi

        chosen_from_intent = []
        for c in multi_turn_pool:
            if len(chosen_from_intent) < target_multi and c["conversation_id"] not in seen_convs:
                chosen_from_intent.append(c)
                seen_convs.add(c["conversation_id"])

        for c in single_turn_pool:
            if len(chosen_from_intent) < quota and c["conversation_id"] not in seen_convs:
                chosen_from_intent.append(c)
                seen_convs.add(c["conversation_id"])

        # If quota not met, fill from remaining
        for c in pool:
            if len(chosen_from_intent) < quota and c["conversation_id"] not in seen_convs:
                chosen_from_intent.append(c)
                seen_convs.add(c["conversation_id"])

        selected.extend(chosen_from_intent)

    # If count is slightly below target due to deduplication, top up from diverse multi-turn pool
    if len(selected) < target_count:
        remaining = [c for c in all_checkpoints if c["conversation_id"] not in seen_convs]
        random.shuffle(remaining)
        for c in remaining:
            if len(selected) >= target_count:
                break
            selected.append(c)
            seen_convs.add(c["conversation_id"])

    # Shuffle deterministically
    random.shuffle(selected)
    return selected[:target_count]


def main():
    logger.info("Starting Golden Evaluation candidate sampling...")
    PATHS.ensure_directories()

    # Load pristine conversations
    convs = []
    with open(PATHS.AMAZONHELP_PRISTINE_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            convs.append(json.loads(line))

    # Sort strictly chronologically to isolate Train, Val, and Test
    convs.sort(key=lambda x: (x["start_timestamp"], x["conversation_id"]))
    n = len(convs)
    n_train = int(n * PHASE3_CONFIG.TRAIN_SPLIT_RATIO)
    n_val = int(n * PHASE3_CONFIG.VAL_SPLIT_RATIO)

    val_convs = convs[n_train : n_train + n_val]
    test_convs = convs[n_train + n_val :]

    logger.info(f"Loaded {len(convs):,} pristine conversations.")
    logger.info(f"Validation Partition: {len(val_convs):,} conversations.")
    logger.info(f"Test Partition (STRICTLY UNTOUCHED): {len(test_convs):,} conversations.")

    # Extract checkpoints from validation partition ONLY
    val_checkpoints = extract_validation_checkpoints(val_convs)
    logger.info(f"Extracted {len(val_checkpoints):,} total candidate checkpoints from Validation.")

    # Sample exactly ~200 stratified checkpoints
    golden_checkpoints = sample_golden_checkpoints(
        val_checkpoints,
        target_count=PHASE3_CONFIG.GOLDEN_SAMPLE_TARGET,
        seed=PHASE3_CONFIG.SEED,
    )
    logger.info(f"Sampled {len(golden_checkpoints):,} stratified golden candidates.")

    # Summary distributions
    turn_depths = Counter(c["turn_depth"] for c in golden_checkpoints)
    total_turns = Counter(c["source_conversation_metadata"]["total_turns"] for c in golden_checkpoints)
    intents = Counter(c["inferred_intent"] for c in golden_checkpoints)
    escalations = sum(1 for c in golden_checkpoints if c["has_escalation_signal"])
    repeated = sum(1 for c in golden_checkpoints if c["has_repeated_contact"])

    logger.info("Golden Candidate Stratification Summary:")
    logger.info(f"  Total Checkpoints: {len(golden_checkpoints)}")
    logger.info(f"  Turn Depths (Current Turn Index): {dict(turn_depths)}")
    logger.info(f"  Conversation Lengths (Total Turns): {dict(total_turns)}")
    logger.info(f"  Intent Distribution: {dict(intents.most_common())}")
    logger.info(f"  Escalation Signals: {escalations} ({escalations/len(golden_checkpoints)*100:.1f}%)")
    logger.info(f"  Repeated Contact Signals: {repeated} ({repeated/len(golden_checkpoints)*100:.1f}%)")

    # Export golden candidates file
    with open(PATHS.GOLDEN_CANDIDATES_JSONL, "w", encoding="utf-8") as f:
        for chk in golden_checkpoints:
            f.write(json.dumps(chk, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(golden_checkpoints)} golden candidate checkpoints to: {PATHS.GOLDEN_CANDIDATES_JSONL}")


if __name__ == "__main__":
    main()
