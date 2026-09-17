"""Phase 6C Customer Support Decision Agent with Historical Retrieval.

Integrates:
1. Multi-turn conversation formatter (zero label leakage).
2. Phase 4 structured deterministic signals (TF-IDF intent, state tracker, escalation, action policy).
3. Phase 5 Train-only historical retrieval (Top-K exemplars, similarity scores, confidence tier).
4. Local LLM (llama3.2:1b) reasoning over structured evidence.
5. Deterministic post-generation safety validator and policy guardrails.
6. Comprehensive latency telemetry and execution traceability.
"""

from dataclasses import asdict, dataclass
import time
from typing import Any, Dict, List, Optional

from src.llm.evidence_builder import EvidenceBuilder, EvidencePackage
from src.llm.model_client import MockOllamaClient, OllamaClient
from src.llm.safety_validator import DeterministicSafetyValidator, ValidatedDecision
from src.llm.schemas import ValidationResult
from src.utils.logger import get_logger

logger = get_logger("agent_with_retrieval")


@dataclass
class AgentWithRetrievalExecutionResult:
    """Complete execution record of a Phase 6C decision turn."""

    checkpoint_id: Optional[str]
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
    unsupported_action_detected: bool
    safety_violations_detected: List[str]
    safety_overrides_applied: List[str]
    is_valid_json: bool
    is_schema_compliant: bool
    retries_used: int
    top_k_used: int
    retrieval_top_1_similarity: float
    retrieval_mean_similarity: float
    retrieval_confidence: str
    baseline_intent: str
    baseline_state: str
    baseline_action: str
    baseline_escalate: bool
    latency_breakdown_ms: Dict[str, float]
    total_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LLMAgentWithRetrieval:
    """Phase 6C customer support agent combining conversation, retrieval, and LLM reasoning."""

    def __init__(
        self,
        model_name: str = "llama3.2:1b",
        client: Optional[Any] = None,
        evidence_builder: Optional[EvidenceBuilder] = None,
        safety_validator: Optional[DeterministicSafetyValidator] = None,
        default_k: int = 5,
        max_retries: int = 1,
    ):
        self.model_name = model_name
        self.client = client or (
            OllamaClient() if OllamaClient().is_available() else MockOllamaClient()
        )
        self.evidence_builder = evidence_builder or EvidenceBuilder()
        self.safety_validator = safety_validator or DeterministicSafetyValidator(max_response_chars=280)
        self.default_k = default_k
        self.max_retries = max_retries

    def process_turn(
        self,
        customer_message: str,
        turn_depth: int = 1,
        history: Optional[List[Dict[str, Any]]] = None,
        checkpoint_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> AgentWithRetrievalExecutionResult:
        """Executes the complete Phase 6C pipeline: retrieval -> structured evidence -> LLM -> safety validation."""
        t_start = time.perf_counter()
        k = top_k if top_k is not None else self.default_k

        # 1. Build structured evidence package (retrieval + Phase 4 signals + conversation)
        t_ev_0 = time.perf_counter()
        evidence: EvidencePackage = self.evidence_builder.build_evidence(
            customer_message=customer_message,
            turn_depth=turn_depth,
            history=history,
            checkpoint_id=checkpoint_id,
            top_k=k,
        )
        self.last_evidence = evidence
        t_ev_end = time.perf_counter()
        evidence_latency_ms = (t_ev_end - t_ev_0) * 1000.0

        # 2. Format custom prompt
        t_prompt_0 = time.perf_counter()
        prompt = self.evidence_builder.format_prompt_from_evidence(evidence)
        t_prompt_end = time.perf_counter()
        prompt_latency_ms = (t_prompt_end - t_prompt_0) * 1000.0

        # 3. Extract evidence signals for model guidance
        evidence_signals = {
            "baseline_intent": evidence.structured_signals.baseline_intent,
            "deterministic_state": evidence.structured_signals.deterministic_state,
            "deterministic_action_candidate": evidence.structured_signals.deterministic_action_candidate,
            "deterministic_escalation": evidence.structured_signals.deterministic_escalation,
            "deterministic_escalation_reason": evidence.structured_signals.deterministic_escalation_reason,
            "retrieval_confidence": evidence.retrieval_package.overall_confidence,
            "top_1_similarity": evidence.retrieval_package.top_1_similarity,
            "mean_similarity": evidence.retrieval_package.mean_similarity,
            "top_retrieved_action": (
                evidence.retrieval_package.exemplars[0].derived_action
                if evidence.retrieval_package.exemplars
                else None
            ),
        }

        # 4. Invoke LLM client
        t_llm_0 = time.perf_counter()
        val_res: ValidationResult = self.client.generate_decision(
            model_name=self.model_name,
            customer_message=customer_message,
            history=history,
            turn_depth=turn_depth,
            retrieval_exemplars=[e.to_dict() for e in evidence.retrieval_package.exemplars],
            max_retries=self.max_retries,
            custom_prompt=prompt,
            evidence_signals=evidence_signals,
        )
        t_llm_end = time.perf_counter()
        llm_latency_ms = (t_llm_end - t_llm_0) * 1000.0

        # 5. Deterministic Safety Validation and Policy Overrides
        t_val_0 = time.perf_counter()
        decision_obj = val_res.decision
        if decision_obj is None:
            # Fallback to structured baseline signals if model output was unparsable
            decision_obj = LLMDecisionOutput(
                intent=evidence.structured_signals.baseline_intent,
                state=evidence.structured_signals.deterministic_state,
                action=evidence.structured_signals.deterministic_action_candidate,
                escalate=evidence.structured_signals.deterministic_escalation,
                escalation_reason=evidence.structured_signals.deterministic_escalation_reason,
                confidence=0.5,
                reasoning_summary="Fallback to deterministic baseline after unparsable model output.",
                response="Please reach out to us via direct message (DM) with your details so we can assist.",
            )

        mandatory_esc = {
            "escalation": evidence.structured_signals.deterministic_escalation,
            "reason": evidence.structured_signals.deterministic_escalation_reason,
        }
        validated: ValidatedDecision = self.safety_validator.validate_and_enforce(
            decision=decision_obj,
            customer_message=customer_message,
            turn_depth=turn_depth,
            history=history,
            mandatory_escalation=mandatory_esc,
        )
        t_val_end = time.perf_counter()
        validation_latency_ms = (t_val_end - t_val_0) * 1000.0

        total_latency_ms = (time.perf_counter() - t_start) * 1000.0

        latency_breakdown = {
            "evidence_and_retrieval_ms": round(evidence_latency_ms, 2),
            "prompt_formatting_ms": round(prompt_latency_ms, 2),
            "llm_generation_ms": round(llm_latency_ms, 2),
            "safety_validation_ms": round(validation_latency_ms, 2),
        }

        return AgentWithRetrievalExecutionResult(
            checkpoint_id=checkpoint_id,
            intent=validated.intent,
            state=validated.state,
            action=validated.action,
            escalate=validated.escalate,
            escalation_reason=validated.escalation_reason,
            confidence=validated.confidence,
            reasoning_summary=validated.reasoning_summary,
            final_response=validated.final_response,
            raw_response=validated.raw_response,
            raw_response_length=validated.raw_response_length,
            final_response_length=validated.final_response_length,
            was_truncated=validated.was_truncated,
            is_safe=validated.is_safe,
            unsupported_action_detected=validated.unsupported_action_detected,
            safety_violations_detected=validated.safety_violations_detected,
            safety_overrides_applied=validated.safety_overrides_applied,
            is_valid_json=val_res.is_valid_json,
            is_schema_compliant=val_res.is_schema_compliant,
            retries_used=val_res.retries_used,
            top_k_used=k,
            retrieval_top_1_similarity=evidence.retrieval_package.top_1_similarity,
            retrieval_mean_similarity=evidence.retrieval_package.mean_similarity,
            retrieval_confidence=evidence.retrieval_package.overall_confidence,
            baseline_intent=evidence.structured_signals.baseline_intent,
            baseline_state=evidence.structured_signals.deterministic_state,
            baseline_action=evidence.structured_signals.deterministic_action_candidate,
            baseline_escalate=evidence.structured_signals.deterministic_escalation,
            latency_breakdown_ms=latency_breakdown,
            total_latency_ms=round(total_latency_ms, 2),
        )

    def process_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        top_k: Optional[int] = None,
    ) -> AgentWithRetrievalExecutionResult:
        """Processes a golden benchmark checkpoint without accessing expected_* target fields."""
        msg = checkpoint["current_customer_message"]
        depth = checkpoint.get("turn_depth", 1)
        history = checkpoint.get("conversation_history_before_current_turn", [])
        cid = checkpoint.get("checkpoint_id")
        return self.process_turn(
            customer_message=msg,
            turn_depth=depth,
            history=history,
            checkpoint_id=cid,
            top_k=top_k,
        )
