"""Comprehensive Evaluator Engine for Phase 4 Baselines and Deterministic Policies.

Evaluates against the 200 human-validated golden checkpoints:
1. Trivial Baseline (Majority intent + Conservative escalation).
2. ML Baseline (TF-IDF + Logistic Regression).
3. Deterministic Escalation Policy.
4. Deterministic State Tracker.
5. Deterministic Action Policy.
6. Unified Decision Exact-Match Pipeline.
"""

from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.annotation.annotator import (
    APPROVED_ACTIONS,
    APPROVED_ESCALATION_REASONS,
    APPROVED_INTENTS,
    APPROVED_STATES,
)
from src.baselines.majority_baseline import (
    ConservativeEscalationBaseline,
    MajorityIntentBaseline,
)
from src.baselines.tfidf_logreg import TfidfLogRegIntentClassifier, format_input_text
from src.evaluation.metrics import (
    compute_decision_exact_match,
    compute_escalation_metrics,
    compute_intent_metrics,
)
from src.policy.action_policy import DeterministicActionPolicy
from src.policy.escalation_policy import DeterministicEscalationPolicy
from src.policy.unified_decision_engine import UnifiedDeterministicDecisionEngine
from src.state.state_tracker import ConversationStateTracker
from src.utils.logger import get_logger

logger = get_logger("baseline_evaluator")


class BaselineEvaluator:
    """Orchestrates comprehensive evaluation of all Phase 4 baselines and policies."""

    def __init__(self, golden_checkpoints_path: Path):
        self.golden_path = golden_checkpoints_path
        self.checkpoints = self._load_checkpoints()

    def _load_checkpoints(self) -> List[Dict[str, Any]]:
        if not self.golden_path.exists():
            raise FileNotFoundError(f"Golden evaluation dataset not found at {self.golden_path}")
        items = []
        with open(self.golden_path, "r", encoding="utf-8") as f:
            for line in f:
                items.append(json.loads(line))
        logger.info(f"Loaded {len(items)} human-validated golden checkpoints from {self.golden_path.name}")
        return items

    def evaluate_trivial_baseline(
        self, majority_intent: str = "DELIVERY_STATUS_AND_TRACKING"
    ) -> Dict[str, Any]:
        """Evaluates Majority Intent Baseline and Conservative Escalation Baseline."""
        intent_baseline = MajorityIntentBaseline(majority_intent=majority_intent)
        esc_baseline_safe = ConservativeEscalationBaseline(default_escalate=True)
        esc_baseline_auto = ConservativeEscalationBaseline(default_escalate=False)

        y_true_intent = [c["expected_intent"] for c in self.checkpoints]
        y_pred_intent = [intent_baseline.predict_one(c["current_customer_message"]) for c in self.checkpoints]

        intent_metrics = compute_intent_metrics(y_true_intent, y_pred_intent, labels=APPROVED_INTENTS)

        y_true_esc = [c["expected_escalation"] for c in self.checkpoints]
        y_pred_esc_auto = [esc_baseline_auto.predict_one(c["current_customer_message"])["escalation"] for c in self.checkpoints]
        y_pred_esc_safe = [esc_baseline_safe.predict_one(c["current_customer_message"])["escalation"] for c in self.checkpoints]

        esc_metrics_auto = compute_escalation_metrics(y_true_esc, y_pred_esc_auto)
        esc_metrics_safe = compute_escalation_metrics(y_true_esc, y_pred_esc_safe)

        return {
            "majority_intent_predicted": majority_intent,
            "intent_metrics": intent_metrics,
            "escalation_never_escalate": esc_metrics_auto,
            "escalation_always_escalate": esc_metrics_safe,
        }

    def evaluate_ml_baseline(self, classifier: TfidfLogRegIntentClassifier) -> Dict[str, Any]:
        """Evaluates TF-IDF + Logistic Regression Intent Classifier."""
        formatted_inputs = [
            format_input_text(c["current_customer_message"], c.get("conversation_history_before_current_turn", []))
            for c in self.checkpoints
        ]
        y_true = [c["expected_intent"] for c in self.checkpoints]
        y_pred = classifier.predict(formatted_inputs)
        probas = classifier.predict_proba(formatted_inputs)

        intent_metrics = compute_intent_metrics(y_true, y_pred, labels=APPROVED_INTENTS)

        # Confidence analysis
        confidences = np.max(probas, axis=1).tolist()
        mean_conf = float(np.mean(confidences))
        median_conf = float(np.median(confidences))
        min_conf = float(np.min(confidences))
        max_conf = float(np.max(confidences))

        # Identify low confidence examples (< 0.40)
        low_confidence_examples = []
        for c, pred, conf in zip(self.checkpoints, y_pred, confidences):
            if conf < 0.40:
                low_confidence_examples.append({
                    "checkpoint_id": c["checkpoint_id"],
                    "customer_message": c["current_customer_message"],
                    "expected_intent": c["expected_intent"],
                    "predicted_intent": pred,
                    "confidence": round(conf, 4),
                })

        # Performance by difficulty tier
        perf_by_diff = {}
        for diff in ["easy", "medium", "hard"]:
            indices = [i for i, c in enumerate(self.checkpoints) if c.get("difficulty") == diff]
            if indices:
                sub_true = [y_true[i] for i in indices]
                sub_pred = [y_pred[i] for i in indices]
                acc = float(accuracy_score(sub_true, sub_pred))
                f1 = float(f1_score(sub_true, sub_pred, labels=APPROVED_INTENTS, average="macro", zero_division=0))
                perf_by_diff[diff] = {
                    "count": len(indices),
                    "accuracy": round(acc, 4),
                    "macro_f1": round(f1, 4),
                }

        return {
            "intent_metrics": intent_metrics,
            "confidence_distribution": {
                "mean": round(mean_conf, 4),
                "median": round(median_conf, 4),
                "min": round(min_conf, 4),
                "max": round(max_conf, 4),
            },
            "low_confidence_count": len(low_confidence_examples),
            "low_confidence_examples": low_confidence_examples[:5],
            "performance_by_difficulty": perf_by_diff,
        }

    def evaluate_deterministic_policies(
        self,
        intent_classifier: Any,
        escalation_policy: Optional[DeterministicEscalationPolicy] = None,
        state_tracker: Optional[ConversationStateTracker] = None,
        action_policy: Optional[DeterministicActionPolicy] = None,
    ) -> Dict[str, Any]:
        """Evaluates the full deterministic policy suite and exact match agreement."""
        engine = UnifiedDeterministicDecisionEngine(
            intent_classifier=intent_classifier,
            escalation_policy=escalation_policy,
            state_tracker=state_tracker,
            action_policy=action_policy,
        )

        decisions = []
        for c in self.checkpoints:
            dec = engine.process_checkpoint(c)
            decisions.append(dec)

        # 1. Escalation Evaluation
        y_true_esc = [c["expected_escalation"] for c in self.checkpoints]
        y_pred_esc = [d["escalation"] for d in decisions]
        esc_metrics = compute_escalation_metrics(y_true_esc, y_pred_esc)

        # 2. State Evaluation
        y_true_state = [c["expected_state"] for c in self.checkpoints]
        y_pred_state = [d["state"] for d in decisions]
        state_acc = float(accuracy_score(y_true_state, y_pred_state))
        state_f1 = float(f1_score(y_true_state, y_pred_state, labels=APPROVED_STATES, average="macro", zero_division=0))

        per_state = {}
        for st in APPROVED_STATES:
            st_indices = [i for i, s in enumerate(y_true_state) if s == st]
            if st_indices:
                st_acc = sum(1 for i in st_indices if y_pred_state[i] == st) / len(st_indices)
                per_state[st] = {
                    "support": len(st_indices),
                    "accuracy": round(float(st_acc), 4),
                }

        # 3. Action Evaluation
        y_true_action = [c["expected_action"] for c in self.checkpoints]
        y_pred_action = [d["action"] for d in decisions]
        action_acc = float(accuracy_score(y_true_action, y_pred_action))
        action_macro_f1 = float(f1_score(y_true_action, y_pred_action, labels=APPROVED_ACTIONS, average="macro", zero_division=0))

        per_action = {}
        for act in APPROVED_ACTIONS:
            act_indices = [i for i, a in enumerate(y_true_action) if a == act]
            if act_indices:
                act_acc = sum(1 for i in act_indices if y_pred_action[i] == act) / len(act_indices)
                per_action[act] = {
                    "support": len(act_indices),
                    "accuracy": round(float(act_acc), 4),
                }

        # 4. Decision Exact Match
        exact_match_metrics = compute_decision_exact_match(self.checkpoints, decisions)

        return {
            "escalation_metrics": esc_metrics,
            "state_metrics": {
                "accuracy": round(state_acc, 4),
                "macro_f1": round(state_f1, 4),
                "per_state": per_state,
            },
            "action_metrics": {
                "accuracy": round(action_acc, 4),
                "macro_f1": round(action_macro_f1, 4),
                "per_action": per_action,
            },
            "exact_match_metrics": exact_match_metrics,
            "decisions_sample": decisions[:3],
        }
