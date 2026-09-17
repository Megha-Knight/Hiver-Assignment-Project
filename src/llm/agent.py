"""LLM-Driven Customer Support Decision Agent for Phase 6B.

Architecture:
Customer Turn + Windowed History
       ↓
Conversation Formatter
       ↓
Local Ollama LLM (llama3.2:1b)
       ↓
Structured Output Parser (Pydantic)
       ↓
Deterministic Safety Validator
       ↓
Final Verified Support Decision & Tweet Draft

IMPORTANT:
- ZERO RETRIEVAL MODULES IMPORTED OR CALLED.
- ZERO TARGET LABELS OR BENCHMARK METADATA EXPOSED AT RUNTIME.
- DETERMINISTIC POST-GENERATION SAFETY OVERRIDES ENFORCED.
"""

from dataclasses import asdict, dataclass
import json
import time
from typing import Any, Dict, List, Optional

from src.annotation.annotator import (
    APPROVED_ACTIONS,
    APPROVED_ESCALATION_REASONS,
    APPROVED_INTENTS,
    APPROVED_STATES,
)
from src.llm.conversation_formatter import ConversationFormatter, FormattedConversation
from src.llm.model_client import MockOllamaClient, OllamaClient
from src.llm.safety_validator import DeterministicSafetyValidator, ValidatedDecision
from src.llm.schemas import LLMDecisionOutput, ValidationResult
from src.utils.logger import get_logger

logger = get_logger("llm_agent")


@dataclass
class AgentExecutionResult:
    """Complete execution record of an LLM support decision turn."""

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
    generation_latency_ms: float
    output_tokens: int
    detected_entities: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LLMCustomerSupportAgent:
    """Core autonomous LLM support agent with deterministic safety validation."""

    def __init__(
        self,
        model_name: str = "llama3.2:1b",
        client: Optional[Any] = None,
        formatter: Optional[ConversationFormatter] = None,
        safety_validator: Optional[DeterministicSafetyValidator] = None,
        max_retries: int = 1,
    ):
        self.model_name = model_name
        self.client = client or (
            OllamaClient() if OllamaClient().is_available() else MockOllamaClient()
        )
        self.formatter = formatter or ConversationFormatter(max_history_turns=4)
        self.safety_validator = safety_validator or DeterministicSafetyValidator(max_response_chars=280)
        self.max_retries = max_retries

    def process_turn(
        self,
        customer_message: str,
        turn_depth: int = 1,
        history: Optional[List[Dict[str, Any]]] = None,
        checkpoint_id: Optional[str] = None,
    ) -> AgentExecutionResult:
        """Executes the full LLM inference, parsing, and safety validation pipeline for a turn."""
        t0 = time.perf_counter()

        # 1. Format conversation context (zero label leakage)
        fmt: FormattedConversation = self.formatter.format_turn(
            customer_message=customer_message,
            history=history,
            turn_depth=turn_depth,
        )

        # 2. Invoke local LLM client
        val_res: ValidationResult = self.client.generate_decision(
            model_name=self.model_name,
            customer_message=customer_message,
            history=history,
            turn_depth=turn_depth,
            retrieval_exemplars=None,  # STRICTLY ZERO RETRIEVAL IN PHASE 6B
            max_retries=self.max_retries,
        )

        total_latency = (time.perf_counter() - t0) * 1000.0

        # 3. Fallback handling if raw output completely failed schema validation
        if not val_res.is_schema_compliant or val_res.decision is None:
            # Deterministic safe fallback
            fallback_decision = LLMDecisionOutput(
                intent="OTHER_OR_UNCLEAR",
                state="STATE_INITIAL_INBOUND" if turn_depth == 1 else "STATE_CUSTOMER_PROVIDING_INFO",
                action="ASK_CLARIFICATION",
                escalate=False,
                escalation_reason="NONE",
                confidence=0.50,
                reasoning_summary="Schema validation failure fallback.",
                response="Could you please provide a few more details so we can assist you with your Amazon order?",
            )
            val_decision = fallback_decision
        else:
            val_decision = val_res.decision

        # 4. Apply deterministic post-generation safety overrides
        safety_eval: ValidatedDecision = self.safety_validator.validate_and_enforce(
            decision=val_decision,
            customer_message=customer_message,
            turn_depth=turn_depth,
            history=history,
        )

        return AgentExecutionResult(
            checkpoint_id=checkpoint_id,
            intent=safety_eval.intent,
            state=safety_eval.state,
            action=safety_eval.action,
            escalate=safety_eval.escalate,
            escalation_reason=safety_eval.escalation_reason,
            confidence=safety_eval.confidence,
            reasoning_summary=safety_eval.reasoning_summary,
            final_response=safety_eval.final_response,
            raw_response=safety_eval.raw_response,
            raw_response_length=safety_eval.raw_response_length,
            final_response_length=safety_eval.final_response_length,
            was_truncated=safety_eval.was_truncated,
            is_safe=safety_eval.is_safe,
            unsupported_action_detected=safety_eval.unsupported_action_detected,
            safety_violations_detected=safety_eval.safety_violations_detected,
            safety_overrides_applied=safety_eval.safety_overrides_applied,
            is_valid_json=val_res.is_valid_json,
            is_schema_compliant=val_res.is_schema_compliant,
            retries_used=val_res.retries_used,
            generation_latency_ms=round(total_latency, 2),
            output_tokens=val_res.output_tokens,
            detected_entities=fmt.detected_entities,
        )

    def process_checkpoint(self, checkpoint: Dict[str, Any]) -> AgentExecutionResult:
        """Processes a golden benchmark checkpoint without accessing expected_* fields."""
        msg = checkpoint["current_customer_message"]
        depth = checkpoint.get("turn_depth", 1)
        history = checkpoint.get("conversation_history_before_current_turn", [])
        cid = checkpoint.get("checkpoint_id")
        return self.process_turn(
            customer_message=msg,
            turn_depth=depth,
            history=history,
            checkpoint_id=cid,
        )
