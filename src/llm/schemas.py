"""Pydantic schemas and validation models for Phase 6 LLM structured output.

Enforces:
1. Controlled Intent Taxonomy (10 classes from APPROVED_INTENTS)
2. Controlled State Vocabulary (8 classes from APPROVED_STATES)
3. Controlled Action Vocabulary (8 classes from APPROVED_ACTIONS)
4. Controlled Escalation Reasons (6 classes from APPROVED_ESCALATION_REASONS)
5. Strict schema conformity and JSON structure validation.
"""

from dataclasses import asdict, dataclass
import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from src.annotation.annotator import (
    APPROVED_ACTIONS,
    APPROVED_ESCALATION_REASONS,
    APPROVED_INTENTS,
    APPROVED_STATES,
)


class LLMDecisionOutput(BaseModel):
    """Structured decision output produced by the local LLM."""

    intent: str = Field(
        ...,
        description="Controlled intent category from APPROVED_INTENTS",
    )
    state: str = Field(
        ...,
        description="Controlled conversation state from APPROVED_STATES",
    )
    action: str = Field(
        ...,
        description="Controlled support action from APPROVED_ACTIONS",
    )
    escalate: bool = Field(
        ...,
        description="Whether this conversation turn warrants supervisor or specialist escalation",
    )
    escalation_reason: str = Field(
        ...,
        description="Controlled reason from APPROVED_ESCALATION_REASONS",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model self-estimated confidence in [0.0, 1.0]",
    )
    reasoning_summary: str = Field(
        ...,
        min_length=5,
        max_length=400,
        description="Concise factual rationale for the decision (no hidden scratchpads)",
    )
    response: str = Field(
        ...,
        min_length=5,
        max_length=600,
        description="Draft customer-facing Twitter response adhering to AmazonHelp guidelines",
    )

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, v: str) -> str:
        clean_v = v.strip().upper()
        if clean_v not in APPROVED_INTENTS:
            raise ValueError(f"Invalid intent '{v}'. Must be one of: {APPROVED_INTENTS}")
        return clean_v

    @field_validator("state")
    @classmethod
    def validate_state(cls, v: str) -> str:
        clean_v = v.strip().upper()
        if clean_v not in APPROVED_STATES:
            raise ValueError(f"Invalid state '{v}'. Must be one of: {APPROVED_STATES}")
        return clean_v

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        clean_v = v.strip().upper()
        if clean_v not in APPROVED_ACTIONS:
            raise ValueError(f"Invalid action '{v}'. Must be one of: {APPROVED_ACTIONS}")
        return clean_v

    @field_validator("escalation_reason")
    @classmethod
    def validate_escalation_reason(cls, v: str) -> str:
        clean_v = v.strip().upper()
        if clean_v not in APPROVED_ESCALATION_REASONS:
            raise ValueError(
                f"Invalid escalation_reason '{v}'. Must be one of: {APPROVED_ESCALATION_REASONS}"
            )
        return clean_v


@dataclass
class ValidationResult:
    """Detailed record of raw LLM output parsing and validation."""

    raw_text: str
    is_valid_json: bool
    is_schema_compliant: bool
    parsed_json: Optional[Dict[str, Any]] = None
    decision: Optional[LLMDecisionOutput] = None
    validation_error: Optional[str] = None
    retries_used: int = 0
    generation_latency_ms: float = 0.0
    output_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.decision:
            data["decision"] = self.decision.model_dump()
        return data


def parse_and_validate_llm_output(raw_text: str) -> ValidationResult:
    """Parses raw model output, strips markdown json fences, and validates schema."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Step 1: JSON Parsing
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as err:
        # Try extracting innermost JSON object if conversational text surrounds it
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return ValidationResult(
                    raw_text=raw_text,
                    is_valid_json=False,
                    is_schema_compliant=False,
                    validation_error=f"JSONDecodeError: {str(err)}",
                )
        else:
            return ValidationResult(
                raw_text=raw_text,
                is_valid_json=False,
                is_schema_compliant=False,
                validation_error=f"JSONDecodeError: {str(err)}",
            )

    # Step 2: Schema Conformity
    try:
        decision = LLMDecisionOutput.model_validate(data)
        return ValidationResult(
            raw_text=raw_text,
            is_valid_json=True,
            is_schema_compliant=True,
            parsed_json=data,
            decision=decision,
            validation_error=None,
        )
    except Exception as schema_err:
        return ValidationResult(
            raw_text=raw_text,
            is_valid_json=True,
            is_schema_compliant=False,
            parsed_json=data,
            validation_error=f"SchemaValidationError: {str(schema_err)}",
        )
