"""Historical Evidence Extraction and Aggregation Engine.

Extracts structured evidence from retrieved historical dialogues:
- historical_intent
- historical_state
- historical_action
- historical_outcome
- historical_escalation
- similarity_score

Aggregates top-K evidence into similarity-weighted signals:
- weighted_intent_distribution
- weighted_escalation_rate
- weighted_action_distribution
- retrieval_confidence ("HIGH", "MEDIUM", "LOW", "NONE")
"""

from dataclasses import asdict, dataclass
import re
from typing import Any, Dict, List, Optional

import numpy as np

from scripts.sample_golden_candidates import classify_inferred_intent
from src.annotation.annotator import (
    APPROVED_ACTIONS,
    APPROVED_INTENTS,
    APPROVED_STATES,
)
from src.policy.escalation_policy import (
    RE_PAYMENT_DISPUTE,
    RE_REPEATED_CONTACT,
    RE_SECURITY_FRAUD,
    RE_SEVERE_THREATS,
)
from src.retrieval.retriever import RetrievalResult


@dataclass
class RetrievedDialogueEvidence:
    """Structured evidence extracted from an individual historical conversation."""
    retrieval_id: str
    conversation_id: str
    similarity_score: float
    historical_intent: str
    historical_state: str
    historical_action: str
    historical_outcome: str
    historical_escalation: bool
    historical_escalation_reason: str
    customer_problem_summary: str
    support_response_evidence: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AggregatedRetrievalEvidence:
    """Aggregated similarity-weighted evidence across top-K retrieved conversations."""
    query: str
    top_k: int
    max_similarity: float
    mean_similarity: float
    retrieval_confidence: str  # HIGH, MEDIUM, LOW, NONE
    weighted_intent_distribution: Dict[str, float]
    top_retrieved_intent: str
    top_intent_weight: float
    weighted_escalation_rate: float
    escalation_supported: bool
    top_escalation_reason: str
    weighted_action_distribution: Dict[str, float]
    top_retrieved_action: str
    weighted_state_distribution: Dict[str, float]
    top_retrieved_state: str
    evidence_items: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_dialogue_evidence(result: RetrievalResult) -> RetrievedDialogueEvidence:
    """Extracts structured intent, state, action, and escalation from a retrieved match."""
    sim = result.similarity_score
    prob = result.customer_problem_summary.strip()
    full_text = result.conversation_text
    outcome_type = result.outcome_evidence_type
    res_status = result.resolution_status

    # 1. Intent Extraction
    intent = classify_inferred_intent(prob)
    if intent == "OTHER_OR_UNCLEAR" and full_text:
        intent = classify_inferred_intent(full_text[:300])

    # 2. Escalation & Reason Extraction
    text_to_check = f"{prob} {full_text[:400]}"
    escalation = False
    esc_reason = "NONE"

    if RE_SECURITY_FRAUD.search(text_to_check):
        escalation = True
        esc_reason = "SECURITY_FRAUD_ALERT"
    elif RE_SEVERE_THREATS.search(text_to_check):
        escalation = True
        esc_reason = "SEVERE_FRUSTRATION_OR_THREAT"
    elif RE_PAYMENT_DISPUTE.search(text_to_check):
        escalation = True
        esc_reason = "PAYMENT_ACCOUNT_DISPUTE"
    elif RE_REPEATED_CONTACT.search(text_to_check):
        escalation = True
        esc_reason = "REPEATED_FAILED_CONTACT"

    # 3. Action Extraction from Outcome & Support Evidence
    if outcome_type == "OFFICIAL_HANDOFF":
        action = "HANDOFF_TO_SECURE_CHANNEL"
    elif outcome_type == "CONFIRMED_RESOLUTION":
        action = "CONFIRM_RESOLUTION"
    elif outcome_type == "TROUBLESHOOTING_STEPS":
        action = "PROVIDE_TROUBLESHOOTING"
    elif outcome_type == "POLICY_GUIDANCE":
        action = "PROVIDE_INFORMATION"
    else:
        action = "HANDOFF_TO_SECURE_CHANNEL"

    # 4. State Extraction
    if outcome_type == "CONFIRMED_RESOLUTION" or res_status == "APPARENTLY_RESOLVED":
        state = "STATE_APPARENTLY_RESOLVED"
    elif outcome_type == "OFFICIAL_HANDOFF":
        state = "STATE_SECURE_HANDOFF_TRIGGERED"
    elif outcome_type == "TROUBLESHOOTING_STEPS":
        state = "STATE_TROUBLESHOOTING_ACTIVE"
    elif escalation:
        state = "STATE_CUSTOMER_ESCALATION"
    else:
        state = "STATE_CUSTOMER_PROVIDING_INFO"

    return RetrievedDialogueEvidence(
        retrieval_id=result.retrieval_id,
        conversation_id=result.conversation_id,
        similarity_score=sim,
        historical_intent=intent,
        historical_state=state,
        historical_action=action,
        historical_outcome=outcome_type,
        historical_escalation=escalation,
        historical_escalation_reason=esc_reason,
        customer_problem_summary=prob,
        support_response_evidence=result.support_response_evidence[:150],
    )


def determine_retrieval_confidence(max_sim: float) -> str:
    """Assigns controlled retrieval confidence tier based on top-1 similarity score.

    Tiers:
    - HIGH   : max_similarity >= 0.70 (strong semantic and contextual alignment)
    - MEDIUM : 0.55 <= max_similarity < 0.70 (meaningful topic overlap)
    - LOW    : 0.40 <= max_similarity < 0.55 (weak lexical / marginal relevance)
    - NONE   : max_similarity < 0.40 (no reliable historical precedent)
    """
    if max_sim >= 0.70:
        return "HIGH"
    elif max_sim >= 0.55:
        return "MEDIUM"
    elif max_sim >= 0.40:
        return "LOW"
    return "NONE"


def aggregate_retrieval_evidence(
    query_text: str,
    results: List[RetrievalResult],
    top_k: int = 5,
) -> AggregatedRetrievalEvidence:
    """Aggregates top-K retrieved matches into similarity-weighted decision signals."""
    if not results:
        return AggregatedRetrievalEvidence(
            query=query_text,
            top_k=top_k,
            max_similarity=0.0,
            mean_similarity=0.0,
            retrieval_confidence="NONE",
            weighted_intent_distribution={intent: 0.0 for intent in APPROVED_INTENTS},
            top_retrieved_intent="OTHER_OR_UNCLEAR",
            top_intent_weight=0.0,
            weighted_escalation_rate=0.0,
            escalation_supported=False,
            top_escalation_reason="NONE",
            weighted_action_distribution={act: 0.0 for act in APPROVED_ACTIONS},
            top_retrieved_action="ASK_CLARIFICATION",
            weighted_state_distribution={st: 0.0 for st in APPROVED_STATES},
            top_retrieved_state="STATE_INITIAL_INBOUND",
            evidence_items=[],
        )

    # Consider only top_k results
    selected = results[:top_k]
    evidence_items = [extract_dialogue_evidence(r) for r in selected]

    sims = [e.similarity_score for e in evidence_items]
    max_sim = float(np.max(sims))
    mean_sim = float(np.mean(sims))
    retrieval_conf = determine_retrieval_confidence(max_sim)

    # Shift similarity scores to positive weights (e.g. min 0.01)
    weights = [max(0.01, s) for s in sims]
    total_weight = sum(weights)

    # 1. Similarity-Weighted Intent Distribution
    intent_weights = {intent: 0.0 for intent in APPROVED_INTENTS}
    for e, w in zip(evidence_items, weights):
        if e.historical_intent in intent_weights:
            intent_weights[e.historical_intent] += w
    weighted_intent_dist = {k: round(v / total_weight, 4) for k, v in intent_weights.items()}
    top_intent = max(weighted_intent_dist.items(), key=lambda x: x[1])[0]
    top_intent_weight = weighted_intent_dist[top_intent]

    # 2. Similarity-Weighted Escalation Rate
    esc_weight = sum(w for e, w in zip(evidence_items, weights) if e.historical_escalation)
    weighted_esc_rate = round(esc_weight / total_weight, 4)
    esc_supported = weighted_esc_rate >= 0.40  # Support escalation if >=40% weighted consensus

    # Determine dominant escalation reason among escalated examples
    esc_reasons = [e.historical_escalation_reason for e in evidence_items if e.historical_escalation]
    top_esc_reason = max(set(esc_reasons), key=esc_reasons.count) if esc_reasons else "NONE"

    # 3. Similarity-Weighted Action Distribution
    action_weights = {act: 0.0 for act in APPROVED_ACTIONS}
    for e, w in zip(evidence_items, weights):
        if e.historical_action in action_weights:
            action_weights[e.historical_action] += w
    weighted_act_dist = {k: round(v / total_weight, 4) for k, v in action_weights.items()}
    top_action = max(weighted_act_dist.items(), key=lambda x: x[1])[0]

    # 4. Similarity-Weighted State Distribution
    state_weights = {st: 0.0 for st in APPROVED_STATES}
    for e, w in zip(evidence_items, weights):
        if e.historical_state in state_weights:
            state_weights[e.historical_state] += w
    weighted_st_dist = {k: round(v / total_weight, 4) for k, v in state_weights.items()}
    top_state = max(weighted_st_dist.items(), key=lambda x: x[1])[0]

    return AggregatedRetrievalEvidence(
        query=query_text,
        top_k=top_k,
        max_similarity=round(max_sim, 4),
        mean_similarity=round(mean_sim, 4),
        retrieval_confidence=retrieval_conf,
        weighted_intent_distribution=weighted_intent_dist,
        top_retrieved_intent=top_intent,
        top_intent_weight=top_intent_weight,
        weighted_escalation_rate=weighted_esc_rate,
        escalation_supported=esc_supported,
        top_escalation_reason=top_esc_reason,
        weighted_action_distribution=weighted_act_dist,
        top_retrieved_action=top_action,
        weighted_state_distribution=weighted_st_dist,
        top_retrieved_state=top_state,
        evidence_items=[e.to_dict() for e in evidence_items],
    )
