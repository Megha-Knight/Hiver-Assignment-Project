"""Structured Evidence Builder for Phase 6C Support Agent.

Synthesizes:
1. Current conversation history and sanitized customer message.
2. Phase 4 structured deterministic signals (TF-IDF intent, state tracker, escalation policy, action policy).
3. Phase 5 historical retrieval evidence (Top-K exemplars, similarity scores, confidence tiers, resolution patterns).
4. Deterministic decision priority rules and safety constraints.

Ensures:
- ZERO golden target labels leaked into prompt.
- Transparent, compact representation structured for llama3.2:1b reasoning.
"""

from dataclasses import asdict, dataclass
import re
from typing import Any, Dict, List, Optional

from src.annotation.annotator import (
    APPROVED_ACTIONS,
    APPROVED_ESCALATION_REASONS,
    APPROVED_INTENTS,
    APPROVED_STATES,
)
from src.baselines.tfidf_logreg import TfidfLogRegIntentClassifier
from src.config import PATHS
from src.llm.conversation_formatter import ConversationFormatter, FormattedConversation
from src.llm.retrieval_context import RetrievalContextManager, RetrievalContextPackage
from src.policy.action_policy import DeterministicActionPolicy
from src.policy.escalation_policy import DeterministicEscalationPolicy
from src.policy.unified_decision_engine import UnifiedDeterministicDecisionEngine
from src.state.state_tracker import ConversationStateTracker
from src.utils.logger import get_logger

logger = get_logger("evidence_builder")


@dataclass
class StructuredSignals:
    """Deterministic signals derived from Phase 4 baselines and policy engines."""

    baseline_intent: str
    baseline_intent_confidence: float
    deterministic_state: str
    state_transition_reason: str
    deterministic_escalation: bool
    deterministic_escalation_reason: str
    deterministic_action_candidate: str
    deterministic_action_reasoning: str
    sensitive_risk_detected: bool
    risk_type: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidencePackage:
    """Consolidated evidence package for the Phase 6C decision agent."""

    checkpoint_id: Optional[str]
    formatted_conversation: FormattedConversation
    structured_signals: StructuredSignals
    retrieval_package: RetrievalContextPackage
    decision_priority: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "formatted_conversation": {
                "current_customer_message": self.formatted_conversation.current_customer_message,
                "cleaned_customer_message": self.formatted_conversation.cleaned_customer_message,
                "turn_depth": self.formatted_conversation.turn_depth,
                "history_turns_included": self.formatted_conversation.history_turns_included,
                "detected_entities": self.formatted_conversation.detected_entities,
            },
            "structured_signals": self.structured_signals.to_dict(),
            "retrieval_package": self.retrieval_package.to_dict(),
            "decision_priority": self.decision_priority,
        }


class EvidenceBuilder:
    """Builds structured evidence packages combining conversation, policy signals, and retrieval."""

    def __init__(
        self,
        intent_classifier: Optional[TfidfLogRegIntentClassifier] = None,
        state_tracker: Optional[ConversationStateTracker] = None,
        escalation_policy: Optional[DeterministicEscalationPolicy] = None,
        action_policy: Optional[DeterministicActionPolicy] = None,
        retrieval_manager: Optional[RetrievalContextManager] = None,
        formatter: Optional[ConversationFormatter] = None,
    ):
        # Load or initialize Phase 4 components
        if intent_classifier is not None:
            self.intent_classifier = intent_classifier
        elif PATHS.TFIDF_LOGREG_MODEL_PATH.exists():
            self.intent_classifier = TfidfLogRegIntentClassifier.load(PATHS.TFIDF_LOGREG_MODEL_PATH)
        else:
            self.intent_classifier = None

        self.state_tracker = state_tracker or ConversationStateTracker()
        self.escalation_policy = escalation_policy or DeterministicEscalationPolicy()
        self.action_policy = action_policy or DeterministicActionPolicy()

        self.decision_engine = UnifiedDeterministicDecisionEngine(
            intent_classifier=self.intent_classifier,
            escalation_policy=self.escalation_policy,
            state_tracker=self.state_tracker,
            action_policy=self.action_policy,
        )

        self.retrieval_manager = retrieval_manager or RetrievalContextManager()
        self.formatter = formatter or ConversationFormatter(max_history_turns=4)

        self.decision_priority = [
            "1. SECURITY / CREDENTIAL SAFETY: Never solicit secrets or allow unsafe claims.",
            "2. EXPLICIT CURRENT ESCALATION: Fraud, legal threats, repeated failures, severe abuse.",
            "3. DETERMINISTIC POLICY CONSTRAINTS: Never override mandatory escalation to false.",
            "4. CURRENT CONVERSATION SEMANTICS: Primary blocking issue over isolated keywords.",
            "5. HIGH-CONFIDENCE HISTORICAL EVIDENCE: Similarity >= 0.70 patterns and outcomes.",
            "6. BASELINE PREDICTIONS: TF-IDF intent and deterministic action candidates.",
            "7. LOW-CONFIDENCE HISTORICAL EVIDENCE: Similarity < 0.50 is weak; do not let it dominate.",
            "8. GENERIC FALLBACK: Direct customer to secure support channels.",
        ]

    def build_evidence(
        self,
        customer_message: str,
        turn_depth: int = 1,
        history: Optional[List[Dict[str, Any]]] = None,
        checkpoint_id: Optional[str] = None,
        top_k: int = 5,
    ) -> EvidencePackage:
        """Constructs complete evidence package for a conversation turn without label leakage."""
        # 1. Format conversation history
        fmt = self.formatter.format_turn(
            customer_message=customer_message,
            history=history,
            turn_depth=turn_depth,
        )

        # 2. Derive Phase 4 deterministic signals
        mock_checkpoint = {
            "checkpoint_id": checkpoint_id or "chk_runtime",
            "current_customer_message": customer_message,
            "turn_depth": turn_depth,
            "conversation_history_before_current_turn": history or [],
        }
        phase4_decision = self.decision_engine.process_checkpoint(mock_checkpoint)

        # Detect sensitive risks in message
        msg_lower = customer_message.lower()
        risk_detected = False
        risk_type = "NONE"
        if re.search(r"\b(password|passcode|otp|cvv|pin|credit card|card details)\b", msg_lower):
            risk_detected = True
            risk_type = "CREDENTIAL_EXPOSURE_RISK"
        elif re.search(r"\b(hacked|unauthorized|stolen card|fraud)\b", msg_lower):
            risk_detected = True
            risk_type = "ACCOUNT_COMPROMISE_RISK"

        signals = StructuredSignals(
            baseline_intent=phase4_decision["intent"],
            baseline_intent_confidence=round(phase4_decision.get("intent_confidence", 0.0), 4),
            deterministic_state=phase4_decision["state"],
            state_transition_reason=phase4_decision.get("state_transition", {}).get("transition_reason", "baseline"),
            deterministic_escalation=phase4_decision["escalation"],
            deterministic_escalation_reason=phase4_decision["escalation_reason"],
            deterministic_action_candidate=phase4_decision["action"],
            deterministic_action_reasoning=phase4_decision.get("action_reasoning", "baseline"),
            sensitive_risk_detected=risk_detected,
            risk_type=risk_type,
        )

        # 3. Retrieve historical exemplars (Train-only)
        conv_context = None
        if history:
            conv_context = " ".join(
                t.get("text", t.get("content", "")) for t in history[-2:] if t.get("role") == "customer"
            )

        retrieval_pkg = self.retrieval_manager.retrieve_context(
            customer_message=customer_message,
            conversation_context=conv_context,
            top_k=top_k,
        )

        return EvidencePackage(
            checkpoint_id=checkpoint_id,
            formatted_conversation=fmt,
            structured_signals=signals,
            retrieval_package=retrieval_pkg,
            decision_priority=self.decision_priority,
        )

    def format_prompt_from_evidence(self, evidence: EvidencePackage) -> str:
        """Formats evidence package into a compact, well-structured user prompt for the LLM."""
        lines = []

        # Section 1: Current Conversation Context
        fmt = evidence.formatted_conversation
        lines.append("### 1. CURRENT CONVERSATION:")
        lines.append(f"Turn Depth: {fmt.turn_depth}")
        lines.append(f"Customer Message: \"{fmt.cleaned_customer_message}\"")
        if fmt.detected_entities:
            lines.append(f"Detected Entities: {', '.join(fmt.detected_entities)}")
        lines.append("")

        # Section 2: Phase 4 Structured Baseline Signals
        sig = evidence.structured_signals
        lines.append("### 2. PHASE 4 STRUCTURED BASELINE SIGNALS:")
        lines.append(f"- Baseline Intent: {sig.baseline_intent} (confidence: {sig.baseline_intent_confidence:.2f})")
        lines.append(f"- Deterministic State: {sig.deterministic_state} (reason: {sig.state_transition_reason})")
        lines.append(f"- Deterministic Escalation: {sig.deterministic_escalation} (reason: {sig.deterministic_escalation_reason})")
        lines.append(f"- Action Candidate: {sig.deterministic_action_candidate} (reasoning: {sig.deterministic_action_reasoning})")
        if sig.sensitive_risk_detected:
            lines.append(f"- SENSITIVE RISK DETECTED: {sig.risk_type} -> MANDATORY SECURE HANDOFF")
        lines.append("")

        # Section 3: Phase 5 Historical Retrieval Evidence
        ret = evidence.retrieval_package
        lines.append("### 3. HISTORICAL RETRIEVAL EVIDENCE (Train Partition Only):")
        lines.append(f"- Top-K: {ret.top_k} | Top-1 Similarity: {ret.top_1_similarity:.4f} | Mean Similarity: {ret.mean_similarity:.4f}")
        lines.append(f"- Overall Retrieval Confidence: {ret.overall_confidence}")
        lines.append(f"- Guidance Note: {ret.guidance_note}")
        lines.append("Historical Exemplars:")
        for idx, ex in enumerate(ret.exemplars, 1):
            lines.append(f"  Exemplar {idx} (Sim: {ex.similarity_score:.4f}, Tier: {ex.confidence_tier}):")
            lines.append(f"    Customer Issue: \"{ex.customer_problem_summary[:140]}\"")
            lines.append(f"    Historical Intent: {ex.derived_intent} | Action: {ex.derived_action} | Escalate: {ex.historical_escalation}")
            lines.append(f"    Support Response: \"{ex.support_response[:140]}\"")
        lines.append("")

        # Section 4: Decision Priority & Reasoning Rules
        lines.append("### 4. DECISION PRIORITY & REASONING RULES:")
        for rule in evidence.decision_priority:
            lines.append(f"- {rule}")
        lines.append("")
        lines.append("CRITICAL INSTRUCTIONS:")
        lines.append("1. Multi-issue grievances: identify the primary blocking issue (e.g. non-delivery blocks returns).")
        lines.append("2. Sarcasm / frustration ('are you having a laugh', 'useless', 'joke'): recognize customer grievance and escalate if unresolved.")
        lines.append("3. Dialogue State: If customer is providing info in Turn 2/3 (tracking ID, carrier, address), state is STATE_CUSTOMER_PROVIDING_INFO.")
        lines.append("4. Never claim you processed a refund, cancelled an order, or checked account records.")
        lines.append("5. If retrieval confidence is LOW (< 0.50), do NOT let exemplars override the conversation.")
        lines.append("")
        lines.append("Output your final decision in strict JSON matching the schema:")

        return "\n".join(lines)
