"""Ollama API client for local open-weight LLM execution and benchmarking.

Features:
1. Native communication with local Ollama daemon (default: http://localhost:11434).
2. Deterministic inference configuration (temperature=0.0, seed=42).
3. Native JSON mode enforcement (`format="json"`).
4. Detailed latency telemetry (loading duration, generation latency, tokens, tokens/sec).
5. Robust retry mechanism on malformed JSON or transient connection errors.
6. Offline test client for testing pipeline integrity when Ollama daemon is not running.
"""

from dataclasses import dataclass
import json
import time
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request

from src.annotation.annotator import (
    APPROVED_ACTIONS,
    APPROVED_ESCALATION_REASONS,
    APPROVED_INTENTS,
    APPROVED_STATES,
)
from src.llm.prompts import SYSTEM_PROMPT, format_user_prompt
from src.llm.schemas import ValidationResult, parse_and_validate_llm_output
from src.utils.logger import get_logger

logger = get_logger("model_client")


@dataclass
class OllamaModelInfo:
    """Metadata describing a locally available Ollama model."""

    name: str
    tag: str
    size_bytes: int
    modified_at: str
    digest: str


class OllamaClient:
    """Client for local Ollama HTTP REST API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 45,
        default_seed: int = 42,
        default_temperature: float = 0.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.default_seed = default_seed
        self.default_temperature = default_temperature

    def is_available(self) -> bool:
        """Checks if local Ollama daemon is reachable."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2) as response:
                return response.status == 200
        except Exception:
            return False

    def list_models(self) -> List[OllamaModelInfo]:
        """Lists all locally installed models in Ollama."""
        url = f"{self.base_url}/api/tags"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
                models = []
                for m in payload.get("models", []):
                    name_parts = m.get("name", "").split(":")
                    name = name_parts[0]
                    tag = name_parts[1] if len(name_parts) > 1 else "latest"
                    models.append(
                        OllamaModelInfo(
                            name=name,
                            tag=tag,
                            size_bytes=m.get("size", 0),
                            modified_at=m.get("modified_at", ""),
                            digest=m.get("digest", ""),
                        )
                    )
                return models
        except Exception as e:
            logger.warning(f"Failed to query Ollama models at {url}: {e}")
            return []

    def generate_decision(
        self,
        model_name: str,
        customer_message: str,
        history: Optional[List[Dict[str, Any]]] = None,
        turn_depth: int = 1,
        retrieval_exemplars: Optional[List[Dict[str, Any]]] = None,
        max_retries: int = 1,
        custom_prompt: Optional[str] = None,
        evidence_signals: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Invokes Ollama model with structured JSON enforcement and latency measurement."""
        if custom_prompt is not None:
            prompt = custom_prompt
        else:
            prompt = format_user_prompt(
                customer_message=customer_message,
                history=history,
                turn_depth=turn_depth,
                retrieval_exemplars=retrieval_exemplars,
            )

        request_body = {
            "model": model_name,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.default_temperature,
                "seed": self.default_seed,
                "num_predict": 300,
            },
        }

        url = f"{self.base_url}/api/generate"
        json_bytes = json.dumps(request_body).encode("utf-8")

        for attempt in range(max_retries + 1):
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(
                    url,
                    data=json_bytes,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                    res_raw = response.read().decode("utf-8")
                    total_latency_ms = (time.perf_counter() - t0) * 1000.0
                    res_payload = json.loads(res_raw)

                    raw_text = res_payload.get("response", "").strip()
                    tokens = res_payload.get("eval_count", 0)

                    val_result = parse_and_validate_llm_output(raw_text)
                    val_result.generation_latency_ms = total_latency_ms
                    val_result.output_tokens = tokens
                    val_result.retries_used = attempt

                    if val_result.is_schema_compliant:
                        return val_result
                    else:
                        logger.warning(
                            f"Model {model_name} attempt {attempt} schema error: {val_result.validation_error}"
                        )
                        if attempt == max_retries:
                            return val_result
            except Exception as exc:
                total_latency_ms = (time.perf_counter() - t0) * 1000.0
                logger.error(f"Ollama API request failed on attempt {attempt}: {exc}")
                if attempt == max_retries:
                    return ValidationResult(
                        raw_text="",
                        is_valid_json=False,
                        is_schema_compliant=False,
                        validation_error=f"ConnectionOrTimeoutError: {str(exc)}",
                        retries_used=attempt,
                        generation_latency_ms=total_latency_ms,
                    )
            time.sleep(0.5)

        return ValidationResult(
            raw_text="",
            is_valid_json=False,
            is_schema_compliant=False,
            validation_error="Max retries exceeded without valid output.",
        )


def _build_grounded_mock_response(
    intent: str,
    state: str,
    action: str,
    escalate: bool,
    msg_lower: str,
) -> str:
    """Constructs a context-specific, policy-compliant response for simulated execution.

    Constraints:
    - Never overrides current action with historical resolution.
    - Never claims a refund, replacement, or delivery occurred.
    - Matches action and intent specifically.
    """
    if action == "CONFIRM_RESOLUTION":
        # Only confirm resolution if customer explicitly expressed resolution or state is resolved
        if state == "STATE_APPARENTLY_RESOLVED" or any(w in msg_lower for w in ["thank", "resolved", "solved", "all good", "received"]):
            return "We are glad to hear your issue has been resolved! Please let us know if you need any further assistance."
        return "We are keeping track of this for you. Please let us know if you need any further assistance with your order."

    if action == "HANDOFF_TO_SECURE_CHANNEL" or escalate:
        if intent == "ACCOUNT_ACCESS_AND_SECURITY":
            return "For your account security, please never share passwords publicly. Please send us a direct message (DM) with your email address so we can secure your account."
        if intent == "PAYMENT_BILLING_AND_PROMOTIONS":
            return "We apologize for the billing issue. To review your account and transactions securely without exposing details publicly, please send us a direct message (DM)."
        if "never been delivered" in msg_lower or "not received" in msg_lower:
            return "We apologize that your package has not arrived. Please reach out to us via direct message (DM) with your order number so we can investigate with the carrier."
        return "We apologize for the frustration. Please send us a direct message (DM) with your order details so our team can investigate this for you."

    if action == "REQUEST_SAFE_DETAILS":
        if intent == "DELIVERY_STATUS_AND_TRACKING":
            return "We would be glad to check your delivery status! Could you please reply with your postal code or carrier name so we can look into tracking?"
        if intent == "PRODUCT_CONDITION_AND_WRONG_ITEM":
            return "We are so sorry your item arrived in that condition. Could you please send us a direct message (DM) with your order ID and a description of the issue?"
        return "We would like to check this for you. Could you please confirm your order number or postal code so we can pull up your delivery details?"

    if action == "PROVIDE_INFORMATION":
        if intent == "DELIVERY_STATUS_AND_TRACKING":
            return "You can track the transit progress and estimated delivery window for your package directly under 'Your Orders' on the Amazon website."
        if intent == "RETURN_REFUND_AND_REPLACEMENT":
            return "You can initiate a return or track your refund timeline via 'Your Orders'. Once processed, refunds generally appear in 3-5 business days depending on your bank."
        if intent == "PRIME_MEMBERSHIP_AND_BENEFITS":
            return "Prime delivery estimates and membership options can be reviewed directly in your Amazon Prime account settings."
        if intent == "PAYMENT_BILLING_AND_PROMOTIONS":
            return "You can review all recent transactions, invoices, and payment methods securely under 'Your Payments' in your account."
        return "You can view detailed policy guidance and self-service management options directly through 'Your Orders' on the Amazon website."

    if action == "EMPATHIZE_AND_DEESCALATE":
        return "We completely understand how frustrating this delay has been and we truly apologize. Please reach out via direct message (DM) so we can look into resolution options for you."

    if action == "PROVIDE_TROUBLESHOOTING":
        if "kindle" in msg_lower or "screen" in msg_lower or "device" in msg_lower:
            return "For Kindle device issues, try pressing and holding the power button for 40 seconds to perform a full reboot. Please let us know if the screen remains unresponsive."
        if "prime video" in msg_lower or "stream" in msg_lower or "app" in msg_lower:
            return "Please try clearing the app cache, restarting your device, or verifying your internet connection to resolve the playback issue."
        return "Please try restarting your device or application to see if that resolves the technical issue. Let us know if the trouble persists."

    if action == "ASK_CLARIFICATION":
        return "Could you please provide a few more details about the issue you are experiencing so we can guide you to the correct solution?"

    if action == "CLOSE_INTERACTION":
        return "Thank you for contacting Amazon Help. Please feel free to reach back out if you need assistance in the future!"

    return "We apologize for the issue! Please send us a direct message (DM) with your details so we can investigate."


class MockOllamaClient:
    """Deterministic offline client simulating candidate models for tests and validation."""

    def __init__(self, simulated_model_name: str = "llama3.2:1b"):
        self.simulated_model_name = simulated_model_name

    def is_available(self) -> bool:
        return True

    def list_models(self) -> List[OllamaModelInfo]:
        return [
            OllamaModelInfo(
                name="llama3.2",
                tag="1b",
                size_bytes=1321200000,
                modified_at="2026-09-15T12:00:00Z",
                digest="sha256:llama3.2-1b-mock",
            ),
            OllamaModelInfo(
                name="llama3.2",
                tag="3b",
                size_bytes=2040000000,
                modified_at="2026-09-15T12:00:00Z",
                digest="sha256:llama3.2-3b-mock",
            ),
            OllamaModelInfo(
                name="qwen2.5",
                tag="1.5b",
                size_bytes=980000000,
                modified_at="2026-09-15T12:00:00Z",
                digest="sha256:qwen2.5-1.5b-mock",
            ),
        ]

    def generate_decision(
        self,
        model_name: str,
        customer_message: str,
        history: Optional[List[Dict[str, Any]]] = None,
        turn_depth: int = 1,
        retrieval_exemplars: Optional[List[Dict[str, Any]]] = None,
        max_retries: int = 1,
        custom_prompt: Optional[str] = None,
        evidence_signals: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Simulates realistic high-quality deterministic LLM structured output."""
        t0 = time.perf_counter()
        msg_lower = customer_message.lower()

        if evidence_signals is not None:
            # Reconcile Phase 4 baseline signals with retrieval and multi-turn context
            intent = evidence_signals.get("baseline_intent", "DELIVERY_STATUS_AND_TRACKING")
            state = evidence_signals.get("deterministic_state", "STATE_INITIAL_INBOUND")
            action = evidence_signals.get("deterministic_action_candidate", "PROVIDE_INFORMATION")
            escalate = evidence_signals.get("deterministic_escalation", False)
            esc_reason = evidence_signals.get("deterministic_escalation_reason", "NONE")
            confidence = 0.90

            # 1. Lexical collision: "How do I return a product that has never been delivered?"
            if "never been delivered" in msg_lower or ("not received" in msg_lower and "return" in msg_lower):
                intent = "DELIVERY_STATUS_AND_TRACKING"
                action = "HANDOFF_TO_SECURE_CHANNEL"

            # 2. Sarcasm / rhetorical frustration:
            if any(term in msg_lower for term in ["laugh", "joke", "useless", "ridiculous", "disgrace", "scam"]):
                escalate = True
                esc_reason = "SEVERE_FRUSTRATION_OR_THREAT"
                state = "STATE_CUSTOMER_ESCALATION"
                action = "HANDOFF_TO_SECURE_CHANNEL"

            # 3. Multi-issue grievance:
            if "refund" in msg_lower and ("delivery" in msg_lower or "dispatch" in msg_lower):
                intent = "RETURN_REFUND_AND_REPLACEMENT"
                if "cancel" in msg_lower:
                    action = "HANDOFF_TO_SECURE_CHANNEL"

            # 4. Turn 2/3 info provision:
            if turn_depth > 1 and any(term in msg_lower for term in ["postcode", "royal mail", "dpd", "hermes", "tracking", "order id", "card", "name", "street"]):
                state = "STATE_CUSTOMER_PROVIDING_INFO"
                if action == "PROVIDE_INFORMATION":
                    action = "REQUEST_SAFE_DETAILS"

            # 5. Security & Account issues:
            if any(term in msg_lower for term in ["password", "hacked", "unauthorized", "stolen", "fraud"]):
                intent = "ACCOUNT_ACCESS_AND_SECURITY"
                escalate = True
                esc_reason = "SECURITY_FRAUD_ALERT"
                state = "STATE_CUSTOMER_ESCALATION"
                action = "HANDOFF_TO_SECURE_CHANNEL"

            # 6. Historical Retrieval Signals:
            ret_conf = evidence_signals.get("retrieval_confidence", "LOW")
            # NOTE: Historical action must NEVER override the current action.
            # Current action is determined by current conversation state and deterministic policy.

            # Build context-specific grounded response aligned with intent, state, action, and policy
            response = _build_grounded_mock_response(
                intent=intent,
                state=state,
                action=action,
                escalate=escalate,
                msg_lower=msg_lower,
            )

            payload = {
                "intent": intent,
                "state": state,
                "action": action,
                "escalate": escalate,
                "escalation_reason": esc_reason,
                "confidence": confidence,
                "reasoning_summary": f"Reconciled baseline ({intent}), retrieval ({ret_conf}), and dialogue turn {turn_depth}.",
                "response": response,
            }

            elapsed_ms = (time.perf_counter() - t0) * 1000.0 + 45.0
            val = parse_and_validate_llm_output(json.dumps(payload))
            val.generation_latency_ms = round(elapsed_ms, 2)
            val.output_tokens = 72
            return val

        # Deterministic simulation matching candidate capability
        intent = "DELIVERY_STATUS_AND_TRACKING"
        state = "STATE_INITIAL_INBOUND"
        action = "REQUEST_SAFE_DETAILS"
        escalate = False
        esc_reason = "NONE"

        if "password" in msg_lower or "unauthorized" in msg_lower or "hacked" in msg_lower:
            intent = "ACCOUNT_ACCESS_AND_SECURITY"
            action = "HANDOFF_TO_SECURE_CHANNEL"
            if "unauthorized" in msg_lower or "hacked" in msg_lower:
                state = "STATE_CUSTOMER_ESCALATION"
                escalate = True
                esc_reason = "SECURITY_FRAUD_ALERT"
        elif "refund" in msg_lower or "return" in msg_lower or "shoes" in msg_lower:
            intent = "RETURN_REFUND_AND_REPLACEMENT"
            action = "PROVIDE_INFORMATION"
        elif "prime video" in msg_lower or "buffering" in msg_lower or "error" in msg_lower:
            intent = "TECHNICAL_AND_DIGITAL_SUPPORT"
            state = "STATE_TROUBLESHOOTING_ACTIVE"
            action = "PROVIDE_TROUBLESHOOTING"
        elif "charged" in msg_lower or "card" in msg_lower or "visa" in msg_lower:
            intent = "PAYMENT_BILLING_AND_PROMOTIONS"
            if "lawyer" in msg_lower or "fraud" in msg_lower:
                state = "STATE_CUSTOMER_ESCALATION"
                action = "HANDOFF_TO_SECURE_CHANNEL"
                escalate = True
                esc_reason = "SEVERE_FRUSTRATION_OR_THREAT"
            elif "twice" in msg_lower:
                action = "HANDOFF_TO_SECURE_CHANNEL"
                escalate = True
                esc_reason = "PAYMENT_ACCOUNT_DISPUTE"
            else:
                action = "PROVIDE_INFORMATION"
        elif "found it" in msg_lower or "neighbor" in msg_lower:
            intent = "DELIVERY_STATUS_AND_TRACKING"
            state = "STATE_APPARENTLY_RESOLVED"
            action = "CONFIRM_RESOLUTION"
        elif "postcode" in msg_lower or "royal mail" in msg_lower or "name" in msg_lower and turn_depth > 1:
            state = "STATE_CUSTOMER_PROVIDING_INFO"
            if "wrong" in msg_lower or "kettle" in msg_lower or "echo" in msg_lower:
                intent = "PRODUCT_CONDITION_AND_WRONG_ITEM"
                action = "HANDOFF_TO_SECURE_CHANNEL"
            else:
                intent = "DELIVERY_STATUS_AND_TRACKING"
                action = "PROVIDE_INFORMATION"
        elif "shattered" in msg_lower or "broken" in msg_lower or "damaged" in msg_lower:
            intent = "PRODUCT_CONDITION_AND_WRONG_ITEM"
            action = "REQUEST_SAFE_DETAILS"
        elif "5th time" in msg_lower or "supervisor" in msg_lower:
            state = "STATE_CUSTOMER_ESCALATION"
            action = "HANDOFF_TO_SECURE_CHANNEL"
            escalate = True
            esc_reason = "REPEATED_FAILED_CONTACT"
        elif "great job" in msg_lower or "champions" in msg_lower:
            action = "EMPATHIZE_AND_DEESCALATE"
        elif "holiday return" in msg_lower or "window" in msg_lower:
            intent = "POLICY_AND_GENERAL_INQUIRIES"
            action = "PROVIDE_INFORMATION"
        elif "cash refund" in msg_lower and "6 months" in msg_lower:
            intent = "RETURN_REFUND_AND_REPLACEMENT"
            state = "STATE_CUSTOMER_ESCALATION"
            action = "HANDOFF_TO_SECURE_CHANNEL"
            escalate = True
            esc_reason = "OUT_OF_POLICY_REQUEST"
        elif "cancel" in msg_lower:
            intent = "CANCELLATION_AND_ORDER_MODIFICATION"
            state = "STATE_CUSTOMER_PROVIDING_INFO" if turn_depth > 1 else "STATE_INITIAL_INBOUND"
            action = "PROVIDE_INFORMATION"

        # Construct context-specific grounded response
        response = _build_grounded_mock_response(
            intent=intent,
            state=state,
            action=action,
            escalate=escalate,
            msg_lower=msg_lower,
        )

        payload = {
            "intent": intent,
            "state": state,
            "action": action,
            "escalate": escalate,
            "escalation_reason": esc_reason,
            "confidence": 0.92,
            "reasoning_summary": f"Classified customer turn based on operational context ({intent}, {action}).",
            "response": response,
        }

        # Simulated latency: ~35-80ms for mock execution
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 + 45.0
        val = parse_and_validate_llm_output(json.dumps(payload))
        val.generation_latency_ms = round(elapsed_ms, 2)
        val.output_tokens = 68
        return val
