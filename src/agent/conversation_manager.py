"""Production Conversation Manager for AmazonHelp Customer Support AI Agent.

Maintains multi-turn state persistence, conversation history, structured understanding,
historical retrieval evidence (K=5), LLM structured decision-making, and deterministic
post-generation safety validation.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import uuid

from src.annotation.annotator import (
    APPROVED_ACTIONS,
    APPROVED_ESCALATION_REASONS,
    APPROVED_INTENTS,
    APPROVED_STATES,
)
from src.llm.agent_with_retrieval import (
    AgentWithRetrievalExecutionResult,
    LLMAgentWithRetrieval,
)
from src.llm.model_client import MockOllamaClient, OllamaClient
from src.llm.safety_validator import DeterministicSafetyValidator
from src.utils.logger import get_logger

logger = get_logger("conversation_manager")


@dataclass
class RetrievedExemplarSummary:
    """Compact summary of a historical retrieval exemplar suitable for evidence display."""

    similarity: float
    intent: str
    action: str
    customer_text: str
    support_reply: str

    @property
    def similarity_score(self) -> float:
        return self.similarity

    @property
    def customer_problem_summary(self) -> str:
        return self.customer_text

    @property
    def support_response(self) -> str:
        return self.support_reply

    def to_dict(self) -> Dict[str, Any]:
        return {
            "similarity": self.similarity,
            "similarity_score": self.similarity,
            "intent": self.intent,
            "action": self.action,
            "customer_text": self.customer_text,
            "customer_problem_summary": self.customer_text,
            "support_reply": self.support_reply,
            "support_response": self.support_reply,
        }


@dataclass
class TurnDecisionRecord:
    """Comprehensive record of a single conversation turn."""

    turn_number: int
    timestamp: str
    customer_message: str
    intent: str
    state: str
    action: str
    escalate: bool
    escalation_reason: str
    confidence: float
    final_response: str
    retrieval_exemplars: List[RetrievedExemplarSummary] = field(default_factory=list)
    retrieval_confidence: str = "LOW"
    retrieval_top1_similarity: float = 0.0
    reasoning_summary: str = ""
    is_safe: bool = True
    safety_violations: List[str] = field(default_factory=list)
    safety_overrides: List[str] = field(default_factory=list)
    latency_breakdown_ms: Dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    is_mock: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_number": self.turn_number,
            "timestamp": self.timestamp,
            "customer_message": self.customer_message,
            "intent": self.intent,
            "state": self.state,
            "action": self.action,
            "escalate": self.escalate,
            "escalation_reason": self.escalation_reason,
            "confidence": self.confidence,
            "final_response": self.final_response,
            "retrieval_confidence": self.retrieval_confidence,
            "retrieval_top1_similarity": self.retrieval_top1_similarity,
            "reasoning_summary": self.reasoning_summary,
            "is_safe": self.is_safe,
            "safety_violations": self.safety_violations,
            "safety_overrides": self.safety_overrides,
            "latency_breakdown_ms": self.latency_breakdown_ms,
            "total_latency_ms": self.total_latency_ms,
            "is_mock": self.is_mock,
        }


class ConversationManager:
    """Manages an active multi-turn support conversation with persistence and evidence logging."""

    def __init__(
        self,
        conversation_id: Optional[str] = None,
        use_mock_llm: bool = False,
        model_name: str = "llama3.2:1b",
        default_k: int = 5,
    ):
        self.conversation_id: str = conversation_id or f"conv_{uuid.uuid4().hex[:8]}"
        self.use_mock_llm: bool = use_mock_llm
        self.model_name: str = model_name
        self.default_k: int = default_k

        # Multi-turn state tracking
        self.turn_history: List[Dict[str, Any]] = []
        self.turn_records: List[TurnDecisionRecord] = []
        self.current_intent: str = "DELIVERY_STATUS_AND_TRACKING"
        self.current_state: str = "STATE_INITIAL_INBOUND"
        self.last_action: str = "PROVIDE_INFORMATION"
        self.escalation_status: bool = False
        self.escalation_reason: str = "NONE"
        self.last_retrieved_evidence: List[RetrievedExemplarSummary] = []

        # Client resolution
        if self.use_mock_llm:
            self.client = MockOllamaClient(simulated_model_name=self.model_name)
            self.is_real_ollama = False
        else:
            real_client = OllamaClient()
            if real_client.is_available():
                self.client = real_client
                self.is_real_ollama = True
            else:
                self.client = None
                self.is_real_ollama = False

        # Initialize underlying Tri-layer agent
        if self.client is not None:
            self.agent = LLMAgentWithRetrieval(
                model_name=self.model_name,
                client=self.client,
                default_k=self.default_k,
            )
        else:
            self.agent = None

    def is_ollama_ready(self) -> bool:
        """Returns True if a real local Ollama daemon is reachable."""
        return OllamaClient().is_available()

    def process_turn(self, customer_message: str) -> TurnDecisionRecord:
        """Processes a single customer turn within the ongoing multi-turn dialogue."""
        if not customer_message or not customer_message.strip():
            raise ValueError("Customer message cannot be empty.")

        turn_num = len(self.turn_records) + 1

        # Check Ollama availability if mock mode is not selected
        if self.client is None and not self.use_mock_llm:
            raise RuntimeError(
                "Local LLM is unavailable. Please start Ollama with 'ollama serve' "
                "and ensure 'llama3.2:1b' is installed via 'ollama pull llama3.2:1b'. "
                "Alternatively, run with --mock for offline simulation mode."
            )

        # Build conversation history in the expected format for evidence builder
        history_tuples = []
        for t in self.turn_history:
            history_tuples.append({"role": t["role"], "text": t["text"]})

        # Process turn through Tri-layer agent
        raw_result: AgentWithRetrievalExecutionResult = self.agent.process_turn(
            customer_message=customer_message,
            turn_depth=turn_num,
            history=history_tuples,
            checkpoint_id=f"{self.conversation_id}_turn{turn_num}",
            top_k=self.default_k,
        )

        # Update state trackers
        self.current_intent = raw_result.intent
        self.current_state = raw_result.state
        self.last_action = raw_result.action
        self.escalation_status = raw_result.escalate
        self.escalation_reason = raw_result.escalation_reason

        # Extract exemplar summaries from the evidence package
        exemplar_summaries: List[RetrievedExemplarSummary] = []
        if hasattr(self.agent, "last_evidence") and self.agent.last_evidence is not None:
            raw_ex = self.agent.last_evidence.retrieval_package.exemplars
            for ex in raw_ex:
                prob = getattr(ex, "customer_problem_summary", None) or getattr(ex, "customer_text", "")
                resp = getattr(ex, "support_response", None) or getattr(ex, "support_reply", "")
                sim = getattr(ex, "similarity_score", None)
                if sim is None:
                    sim = getattr(ex, "similarity", 0.0)
                exemplar_summaries.append(
                    RetrievedExemplarSummary(
                        similarity=round(float(sim), 4),
                        intent=getattr(ex, "derived_intent", None) or getattr(ex, "intent", "OTHER_OR_UNCLEAR"),
                        action=getattr(ex, "derived_action", None) or getattr(ex, "action", "PROVIDE_INFORMATION"),
                        customer_text=prob[:120] + ("..." if len(prob) > 120 else ""),
                        support_reply=resp[:120] + ("..." if len(resp) > 120 else ""),
                    )
                )

        self.last_retrieved_evidence = exemplar_summaries

        record = TurnDecisionRecord(
            turn_number=turn_num,
            timestamp=datetime.now(timezone.utc).isoformat(),
            customer_message=customer_message,
            intent=raw_result.intent,
            state=raw_result.state,
            action=raw_result.action,
            escalate=raw_result.escalate,
            escalation_reason=raw_result.escalation_reason,
            confidence=raw_result.confidence,
            final_response=raw_result.final_response,
            retrieval_exemplars=exemplar_summaries,
            retrieval_confidence=raw_result.retrieval_confidence,
            retrieval_top1_similarity=raw_result.retrieval_top_1_similarity,
            reasoning_summary=raw_result.reasoning_summary,
            is_safe=raw_result.is_safe,
            safety_violations=raw_result.safety_violations_detected,
            safety_overrides=raw_result.safety_overrides_applied,
            latency_breakdown_ms=raw_result.latency_breakdown_ms,
            total_latency_ms=raw_result.total_latency_ms,
            is_mock=self.use_mock_llm,
        )

        # Append to persistent history
        self.turn_history.append({"turn": turn_num, "role": "customer", "text": customer_message})
        self.turn_history.append({"turn": turn_num, "role": "agent", "text": raw_result.final_response})
        self.turn_records.append(record)

        # Safe structured logging (no credentials)
        logger.info(
            f"Turn {turn_num} [{self.conversation_id}] | "
            f"Intent={raw_result.intent} | State={raw_result.state} | Action={raw_result.action} | "
            f"Escalate={raw_result.escalate} ({raw_result.escalation_reason}) | "
            f"Ret_Sim={raw_result.retrieval_top_1_similarity:.3f} | Latency={raw_result.total_latency_ms:.1f}ms | "
            f"Safe={raw_result.is_safe}"
        )

        return record

    def reset(self) -> None:
        """Clears memory and starts a new conversation session."""
        self.conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
        self.turn_history.clear()
        self.turn_records.clear()
        self.current_intent = "DELIVERY_STATUS_AND_TRACKING"
        self.current_state = "STATE_INITIAL_INBOUND"
        self.last_action = "PROVIDE_INFORMATION"
        self.escalation_status = False
        self.escalation_reason = "NONE"
        self.last_retrieved_evidence.clear()
