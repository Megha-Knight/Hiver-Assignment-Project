"""Evidence-Grounding Evaluator Engine for Phase 7C.

Evaluates whether the agent's generated response is factually supported by
retrieved historical evidence across 4 distinct diagnostic probes:
1. EVIDENCE_SUPPORT: Is the response factually supported by historical evidence?
   (SUPPORTED, PARTIAL, UNSUPPORTED)
2. UNSUPPORTED_POLICY_CLAIM: Does the response introduce policy claims absent from evidence?
3. UNSUPPORTED_CAPABILITY: Does the response claim internal capabilities the agent does not possess?
4. CONTRADICTION: Does the response contradict retrieved evidence or explicit facts?

CRITICAL METHODOLOGICAL PRINCIPLE:
Dense retrieval embedding similarity alone DOES NOT prove grounding.
Grounding must inspect the actual generated claim against evidence and operational boundaries.
"""

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger("grounding_evaluator")

# Probes for fabricated capabilities (agent claims direct execution in chat)
RE_FABRICATED_CAPABILITY = re.compile(
    r"\b(i have refunded|i refunded|refund has been processed|processed your refund|"
    r"credited your account|issued a refund|i have cancelled|i cancelled|"
    r"order has been cancelled|cancelled your order|i have dispatched|"
    r"replacement has been ordered|sent a replacement|shipped a new item|"
    r"i checked your account|logged into your account|looking at your account records|"
    r"accessing your account|viewing your account)\b",
    re.IGNORECASE,
)

# Probes for unsupported policy claims (unrealistic SLAs, cash guarantees)
RE_UNSUPPORTED_POLICY = re.compile(
    r"\b(guaranteed delivery within \d+ hour|guaranteed refund within \d+ minute|"
    r"cash compensation of|we will pay you|promise you a voucher of \$|full refund without return)\b",
    re.IGNORECASE,
)


@dataclass
class GroundingEvaluationResult:
    """Structured result of the 4 evidence-grounding probes."""

    evidence_support: str  # "SUPPORTED", "PARTIAL", "UNSUPPORTED"
    unsupported_policy_claim: bool
    unsupported_capability: bool
    contradiction: bool
    is_grounded: bool
    retrieval_confidence: str
    top_1_similarity: float
    grounding_notes: str = ""
    diagnostic_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceGroundingEvaluator:
    """Evaluates factual and policy alignment between generated replies and historical evidence."""

    def __init__(self, high_similarity_threshold: float = 0.70, low_similarity_threshold: float = 0.50):
        self.high_threshold = high_similarity_threshold
        self.low_threshold = low_similarity_threshold

    def evaluate_grounding(
        self,
        customer_message: str,
        generated_response: str,
        retrieved_exemplars: Optional[List[Dict[str, Any]]] = None,
        retrieval_top1_similarity: float = 0.0,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> GroundingEvaluationResult:
        """Executes the 4 grounding probes on the generated response."""
        resp_lower = generated_response.lower()
        msg_lower = customer_message.lower()
        history_text = " ".join([t.get("text", "").lower() for t in (conversation_history or [])])

        exemplar_replies = [
            (ex.get("support_response") or ex.get("support_reply", "")).lower()
            for ex in (retrieved_exemplars or [])
        ]
        all_evidence_text = " ".join(exemplar_replies)
        has_valid_exemplars = any(len(r.strip()) > 0 for r in exemplar_replies)

        notes = []

        # -------------------------------------------------------------
        # Probe 3: Unsupported Capability (Direct action execution claim)
        # -------------------------------------------------------------
        unsupported_capability = bool(RE_FABRICATED_CAPABILITY.search(generated_response))
        if unsupported_capability:
            notes.append("Claimed direct account/transaction execution without API authority.")

        # -------------------------------------------------------------
        # Probe 2: Unsupported Policy Claim
        # -------------------------------------------------------------
        unsupported_policy = bool(RE_UNSUPPORTED_POLICY.search(generated_response))
        if unsupported_policy:
            notes.append("Introduced unsupported SLA or financial compensation policy.")

        # -------------------------------------------------------------
        # Probe 4: Contradiction
        # -------------------------------------------------------------
        contradiction = False
        # Check if customer explicitly reported package missing or damaged, but agent claimed it was delivered safely
        if ("not delivered" in msg_lower or "never arrived" in msg_lower or "not received" in msg_lower) and (
            "glad your parcel was delivered" in resp_lower or "happy you received" in resp_lower
        ):
            contradiction = True
            notes.append("Contradicted customer: Claimed delivery when customer reported non-receipt.")

        if ("already called" in msg_lower or "talked to support" in msg_lower) and (
            "please call our phone line" in resp_lower
        ):
            contradiction = True
            notes.append("Contradicted context: Directed back to phone support after phone failure.")

        # -------------------------------------------------------------
        # Probe 1: Evidence Support
        # -------------------------------------------------------------
        # Categorize evidence support based on semantic keywords, exemplar presence, and claim plausibility
        if unsupported_capability or contradiction:
            evidence_support = "UNSUPPORTED"
        elif not retrieved_exemplars or not has_valid_exemplars or retrieval_top1_similarity < self.low_threshold:
            # Low similarity or missing evidence
            evidence_support = "UNSUPPORTED"
            notes.append("Historical evidence is missing, empty, or below similarity threshold (< 0.50).")
        else:
            # Check if guidance aligns with standard historical support patterns
            aligns_with_evidence = False
            for ex_reply in exemplar_replies:
                if not ex_reply.strip():
                    continue
                # Check for shared key support directions (DM, Your Orders, Contact Us, carrier check, etc.)
                if (
                    ("dm" in ex_reply and "dm" in resp_lower)
                    or ("your orders" in ex_reply and "your orders" in resp_lower)
                    or ("tracking" in ex_reply and "tracking" in resp_lower)
                    or ("help" in ex_reply and "help" in resp_lower)
                    or ("phone" in ex_reply and "phone" in resp_lower)
                    or ("carrier" in ex_reply and "carrier" in resp_lower)
                    or ("refund" in ex_reply and "refund" in resp_lower)
                    or ("return" in ex_reply and "return" in resp_lower)
                    or ("replace" in ex_reply and "replace" in resp_lower)
                    or ("email" in ex_reply and "email" in resp_lower)
                    or ("link" in ex_reply and "link" in resp_lower)
                    or ("restart" in ex_reply and "restart" in resp_lower)
                    or ("power" in ex_reply and "power" in resp_lower)
                    or ("button" in ex_reply and "button" in resp_lower)
                ):
                    aligns_with_evidence = True
                    break

            if aligns_with_evidence:
                evidence_support = "SUPPORTED"
            else:
                if any(kw in resp_lower for kw in ["dm", "your orders", "amazon", "tracking", "help"]):
                    evidence_support = "PARTIAL"
                    notes.append("General public guidance plausible, but specific historical exemplar alignment is weak.")
                else:
                    evidence_support = "UNSUPPORTED"

        # Overall grounded determination:
        # A response is grounded only if it has real SUPPORTED or PARTIAL evidence backing FROM VALID EXEMPLARS,
        # and has zero capability fabrication, policy violation, or factual contradiction.
        is_grounded = (
            has_valid_exemplars
            and (retrieval_top1_similarity >= self.low_threshold)
            and (evidence_support in ("SUPPORTED", "PARTIAL"))
            and not unsupported_capability
            and not unsupported_policy
            and not contradiction
        )

        confidence_tier = "HIGH" if retrieval_top1_similarity >= self.high_threshold else (
            "MEDIUM" if retrieval_top1_similarity >= self.low_threshold else "LOW"
        )

        return GroundingEvaluationResult(
            evidence_support=evidence_support,
            unsupported_policy_claim=unsupported_policy,
            unsupported_capability=unsupported_capability,
            contradiction=contradiction,
            is_grounded=is_grounded,
            retrieval_confidence=confidence_tier,
            top_1_similarity=round(retrieval_top1_similarity, 4),
            grounding_notes="; ".join(notes) if notes else "Response is well-grounded in historical support protocol.",
            diagnostic_details={
                "has_exemplars": bool(retrieved_exemplars),
                "exemplar_count": len(retrieved_exemplars or []),
                "aligned_with_exemplars": evidence_support == "SUPPORTED",
            },
        )
