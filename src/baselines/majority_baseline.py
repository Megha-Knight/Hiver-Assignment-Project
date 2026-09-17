"""Trivial and Conservative Baseline Estimators for Intent and Escalation.

Provides:
1. MajorityIntentBaseline: Predicts the empirical majority class from the Train partition.
2. ConservativeEscalationBaseline: Explicit baseline policy for safety and escalation decisions.
"""

from collections import Counter
from typing import Any, Dict, List, Optional
import numpy as np


class MajorityIntentBaseline:
    """Trivial baseline predicting the majority intent observed in training data."""

    def __init__(self, majority_intent: str = "DELIVERY_STATUS_AND_TRACKING"):
        self.majority_intent = majority_intent

    def fit_from_counts(self, train_intent_counts: Dict[str, int]):
        """Sets majority intent dynamically from Train intent frequencies."""
        if train_intent_counts:
            # Pick highest frequency intent
            self.majority_intent = max(train_intent_counts.items(), key=lambda x: x[1])[0]

    def predict_one(self, text: str, context: Optional[List[Dict[str, Any]]] = None) -> str:
        return self.majority_intent

    def predict(self, texts: List[str], contexts: Optional[List[List[Dict[str, Any]]]] = None) -> List[str]:
        return [self.majority_intent for _ in texts]

    def predict_proba(self, texts: List[str], intent_classes: List[str]) -> np.ndarray:
        """Returns 1.0 for majority class and 0.0 for others."""
        probas = np.zeros((len(texts), len(intent_classes)), dtype=np.float32)
        if self.majority_intent in intent_classes:
            idx = intent_classes.index(self.majority_intent)
            probas[:, idx] = 1.0
        return probas


class ConservativeEscalationBaseline:
    """Explicit escalation baseline documenting the autonomy vs. safety trade-off.

    Modes:
    1. NEVER_ESCALATE (default_escalate=False): Assumes autonomous resolution for all turns.
       - Demonstrates minimum human handoff rate, but maximizes False Auto-Handle Rate (FAHR).
    2. ALWAYS_ESCALATE (default_escalate=True): Ultra-conservative policy escalating 100% of turns.
       - Guarantees 0% FAHR (maximum safety), but yields 0% autonomous resolution.
    """

    def __init__(self, default_escalate: bool = False, default_reason: str = "NONE"):
        self.default_escalate = default_escalate
        self.default_reason = default_reason if default_escalate else "NONE"

    def predict_one(self, text: str, context: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        return {
            "escalation": self.default_escalate,
            "reason": self.default_reason,
            "confidence": 1.0,
            "evidence": ["baseline_static_decision"],
        }

    def predict(self, texts: List[str], contexts: Optional[List[List[Dict[str, Any]]]] = None) -> List[Dict[str, Any]]:
        return [self.predict_one(t) for t in texts]
