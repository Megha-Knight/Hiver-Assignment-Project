"""Retrieval Context Manager for Phase 6C.

Provides structured retrieval context and evidence formatting:
1. Dense retrieval over the 5,502 Train-only historical conversations.
2. Structured metadata extraction for each retrieved exemplar.
3. Similarity tier classification:
   - HIGH: similarity >= 0.70
   - MEDIUM: 0.50 <= similarity < 0.70
   - LOW: similarity < 0.50
4. Formats compact, transparent exemplar blocks for the evidence builder.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from src.retrieval.evidence import extract_dialogue_evidence
from src.retrieval.retriever import HistoricalRetriever, RetrievalResult
from src.utils.logger import get_logger

logger = get_logger("retrieval_context")


@dataclass
class RetrievedExemplar:
    """Structured historical exemplar for LLM context."""

    retrieval_id: str
    conversation_id: str
    similarity_score: float
    confidence_tier: str  # HIGH, MEDIUM, LOW
    derived_intent: str
    derived_state: str
    derived_action: str
    historical_escalation: bool
    historical_escalation_reason: str
    resolution_status: str
    customer_problem_summary: str
    support_response: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Canonical fields are already in asdict: similarity_score, customer_problem_summary, support_response
        # Provide legacy aliases for backwards compatibility
        d["similarity"] = self.similarity_score
        d["customer_text"] = self.customer_problem_summary
        d["support_reply"] = self.support_response
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RetrievedExemplar":
        """Instantiates RetrievedExemplar resolving canonical fields with legacy fallbacks."""
        sim = d.get("similarity_score")
        if sim is None:
            sim = d.get("similarity", 0.0)
        prob = d.get("customer_problem_summary") or d.get("customer_text", "")
        resp = d.get("support_response") or d.get("support_reply", "")
        return cls(
            retrieval_id=d.get("retrieval_id", "unknown"),
            conversation_id=d.get("conversation_id", "unknown"),
            similarity_score=float(sim),
            confidence_tier=d.get("confidence_tier", "LOW"),
            derived_intent=d.get("derived_intent", "OTHER_OR_UNCLEAR"),
            derived_state=d.get("derived_state", "STATE_INITIAL_INBOUND"),
            derived_action=d.get("derived_action", "PROVIDE_INFORMATION"),
            historical_escalation=bool(d.get("historical_escalation", False)),
            historical_escalation_reason=d.get("historical_escalation_reason", "NONE"),
            resolution_status=d.get("resolution_status", "UNKNOWN"),
            customer_problem_summary=prob,
            support_response=resp,
        )


@dataclass
class RetrievalContextPackage:
    """Complete retrieval package with similarity metrics and structured exemplars."""

    top_k: int
    top_1_similarity: float
    max_similarity: float
    mean_similarity: float
    overall_confidence: str  # HIGH, MEDIUM, LOW
    exemplars: List[RetrievedExemplar]
    guidance_note: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["exemplars"] = [e.to_dict() for e in self.exemplars]
        return d


class RetrievalContextManager:
    """Coordinates retrieval execution and structured exemplar packaging."""

    def __init__(self, retriever: Optional[HistoricalRetriever] = None, default_k: int = 5):
        self.retriever = retriever or HistoricalRetriever()
        self.default_k = default_k

    def retrieve_context(
        self,
        customer_message: str,
        conversation_context: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> RetrievalContextPackage:
        """Retrieves top-K exemplars and constructs a structured retrieval package."""
        k = top_k or self.default_k

        # Query the historical index
        results: List[RetrievalResult] = self.retriever.query(
            customer_message=customer_message,
            conversation_context=conversation_context,
            top_k=k,
        )

        if not results:
            return RetrievalContextPackage(
                top_k=k,
                top_1_similarity=0.0,
                max_similarity=0.0,
                mean_similarity=0.0,
                overall_confidence="LOW",
                exemplars=[],
                guidance_note="Zero historical exemplars retrieved. Rely solely on current conversation and deterministic policy.",
            )

        exemplars: List[RetrievedExemplar] = []
        similarities = [r.similarity_score for r in results]
        top_1 = round(similarities[0], 4)
        max_sim = round(max(similarities), 4)
        mean_sim = round(float(sum(similarities)) / len(similarities), 4)

        # Classify overall confidence
        if top_1 >= 0.70:
            overall_conf = "HIGH"
            guidance = "HIGH confidence historical matches available. Historical resolution patterns are highly relevant but must not override safety rules."
        elif top_1 >= 0.50:
            overall_conf = "MEDIUM"
            guidance = "MEDIUM confidence historical matches. Use as domain reference, prioritizing the current conversation."
        else:
            overall_conf = "LOW"
            guidance = "LOW confidence historical matches (similarity < 0.50). Historical evidence is WEAK. Do NOT let historical examples dominate; rely primarily on current conversation and deterministic policy."

        for r in results:
            ev = extract_dialogue_evidence(r)
            tier = "HIGH" if r.similarity_score >= 0.70 else ("MEDIUM" if r.similarity_score >= 0.50 else "LOW")
            exemplars.append(
                RetrievedExemplar(
                    retrieval_id=r.retrieval_id,
                    conversation_id=r.conversation_id,
                    similarity_score=r.similarity_score,
                    confidence_tier=tier,
                    derived_intent=ev.historical_intent,
                    derived_state=ev.historical_state,
                    derived_action=ev.historical_action,
                    historical_escalation=ev.historical_escalation,
                    historical_escalation_reason=ev.historical_escalation_reason,
                    resolution_status=r.resolution_status,
                    customer_problem_summary=r.customer_problem_summary,
                    support_response=r.support_response_evidence,
                )
            )

        return RetrievalContextPackage(
            top_k=len(exemplars),
            top_1_similarity=top_1,
            max_similarity=max_sim,
            mean_similarity=mean_sim,
            overall_confidence=overall_conf,
            exemplars=exemplars,
            guidance_note=guidance,
        )
