"""Comprehensive Evaluation and Ablation Framework for Phase 5 Retrieval Augmentation.

Implements:
1. K-sweep evaluation across K in [1, 3, 5, 10].
2. Modality ablations (Baseline, Intent-Only, Escalation-Only, Full Retrieval).
3. Retrieval quality distribution (top-1, mean similarity, confidence breakdown).
4. Difficulty tier stratification (Easy, Medium, Hard).
5. Bootstrap confidence intervals (1,000 resamples) for key metric deltas.
"""

from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.annotation.annotator import (
    APPROVED_ACTIONS,
    APPROVED_INTENTS,
    APPROVED_STATES,
)
from src.evaluation.metrics import (
    compute_decision_exact_match,
    compute_escalation_metrics,
    compute_intent_metrics,
)
from src.retrieval.evidence import (
    AggregatedRetrievalEvidence,
    aggregate_retrieval_evidence,
)
from src.retrieval.query_builder import build_retrieval_query
from src.retrieval.retrieval_policy import RetrievalAugmentedDecisionEngine
from src.retrieval.retriever import HistoricalRetriever
from src.utils.logger import get_logger

logger = get_logger("retrieval_evaluator")


def compute_bootstrap_confidence_interval(
    baseline_scores: List[float],
    augmented_scores: List[float],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, float]:
    """Computes empirical bootstrap confidence interval for the delta (augmented - baseline)."""
    assert len(baseline_scores) == len(augmented_scores), "Mismatched score arrays"
    rng = np.random.default_rng(random_seed)
    n = len(baseline_scores)
    deltas = []

    b_arr = np.array(baseline_scores)
    a_arr = np.array(augmented_scores)

    for _ in range(n_resamples):
        indices = rng.choice(n, size=n, replace=True)
        b_sample_mean = np.mean(b_arr[indices])
        a_sample_mean = np.mean(a_arr[indices])
        deltas.append(a_sample_mean - b_sample_mean)

    alpha = 1.0 - confidence_level
    low_pct = (alpha / 2.0) * 100.0
    high_pct = (1.0 - alpha / 2.0) * 100.0

    ci_low = float(np.percentile(deltas, low_pct))
    ci_high = float(np.percentile(deltas, high_pct))
    observed_delta = float(np.mean(a_arr) - np.mean(b_arr))

    return {
        "observed_delta": round(observed_delta, 4),
        "ci_lower": round(ci_low, 4),
        "ci_upper": round(ci_high, 4),
        "confidence_level": confidence_level,
        "n_resamples": n_resamples,
    }


class RetrievalEvaluator:
    """Orchestrates comprehensive Phase 5 retrieval benchmarking and ablations."""

    def __init__(
        self,
        golden_checkpoints_path: Path,
        retriever: HistoricalRetriever,
        decision_engine: RetrievalAugmentedDecisionEngine,
    ):
        self.golden_path = golden_checkpoints_path
        self.retriever = retriever
        self.decision_engine = decision_engine
        self.checkpoints = self._load_checkpoints()

    def _load_checkpoints(self) -> List[Dict[str, Any]]:
        items = []
        with open(self.golden_path, "r", encoding="utf-8") as f:
            for line in f:
                items.append(json.loads(line))
        logger.info(f"Loaded {len(items)} golden checkpoints for Phase 5 retrieval evaluation.")
        return items

    def precompute_retrievals(self, max_k: int = 10) -> List[Tuple[Dict[str, Any], Any, List[Any]]]:
        """Precomputes dense vector retrieval matches for all 200 checkpoints."""
        logger.info(f"Executing dense retrieval for {len(self.checkpoints)} checkpoints (max_k={max_k})...")
        precomputed = []
        for c in self.checkpoints:
            msg = c["current_customer_message"]
            depth = c.get("turn_depth", 1)
            history = c.get("conversation_history_before_current_turn", [])

            q = build_retrieval_query(msg, history, depth)
            matches = self.retriever.query(q.query_text, top_k=max_k)
            precomputed.append((c, q, matches))

        logger.info(f"Successfully retrieved candidates for all {len(precomputed)} checkpoints.")
        return precomputed

    def analyze_retrieval_quality(self, precomputed: List[Tuple[Dict[str, Any], Any, List[Any]]]) -> Dict[str, Any]:
        """Analyzes top-1 and top-5 similarity distributions and coverage across checkpoints."""
        top1_sims = []
        top5_means = []
        conf_counts = Counter()

        for c, q, matches in precomputed:
            if matches:
                top1_sims.append(matches[0].similarity_score)
                top5_slice = matches[:5]
                top5_means.append(float(np.mean([m.similarity_score for m in top5_slice])))
                evidence = aggregate_retrieval_evidence(q.query_text, matches, top_k=5)
                conf_counts[evidence.retrieval_confidence] += 1
            else:
                top1_sims.append(0.0)
                top5_means.append(0.0)
                conf_counts["NONE"] += 1

        total = len(precomputed)
        low_sim_count = sum(1 for s in top1_sims if s < 0.50)

        return {
            "total_queries": total,
            "average_top1_similarity": round(float(np.mean(top1_sims)), 4),
            "median_top1_similarity": round(float(np.median(top1_sims)), 4),
            "min_top1_similarity": round(float(np.min(top1_sims)), 4),
            "max_top1_similarity": round(float(np.max(top1_sims)), 4),
            "average_top5_similarity": round(float(np.mean(top5_means)), 4),
            "low_similarity_count": low_sim_count,
            "low_similarity_rate": round(low_sim_count / total, 4),
            "retrieval_coverage_rate": round((total - conf_counts.get("NONE", 0)) / total, 4),
            "confidence_tier_counts": dict(conf_counts),
        }

    def evaluate_configuration(
        self,
        precomputed: List[Tuple[Dict[str, Any], Any, List[Any]]],
        k: int = 5,
        mode: str = "full",
    ) -> Dict[str, Any]:
        """Evaluates decisions for a specific K and ablation mode."""
        decisions = []
        evidence_list = []

        for c, q, matches in precomputed:
            evidence = aggregate_retrieval_evidence(q.query_text, matches, top_k=k)
            dec = self.decision_engine.process_checkpoint_augmented(c, evidence, mode=mode)
            decisions.append(dec)
            evidence_list.append(evidence)

        # 1. Intent Metrics
        y_true_intent = [c["expected_intent"] for c in self.checkpoints]
        y_pred_intent = [d["intent"] for d in decisions]
        intent_metrics = compute_intent_metrics(y_true_intent, y_pred_intent, labels=APPROVED_INTENTS)

        # 2. Escalation Metrics
        y_true_esc = [c["expected_escalation"] for c in self.checkpoints]
        y_pred_esc = [d["escalation"] for d in decisions]
        esc_metrics = compute_escalation_metrics(y_true_esc, y_pred_esc)

        # False escalation rate = FP / (FP + TN)
        fp = esc_metrics["false_positives"]
        tn = esc_metrics["true_negatives"]
        esc_metrics["false_escalation_rate"] = round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0

        # Count recovered escalations
        recovered_count = sum(1 for c, d in zip(self.checkpoints, decisions) if c["expected_escalation"] and d["escalation"] and "retrieval_recovered" in d.get("decision_evidence", [""])[2])
        esc_metrics["recovered_escalations_count"] = recovered_count

        # 3. State Metrics
        y_true_state = [c["expected_state"] for c in self.checkpoints]
        y_pred_state = [d["state"] for d in decisions]
        state_acc = float(accuracy_score(y_true_state, y_pred_state))
        state_f1 = float(f1_score(y_true_state, y_pred_state, labels=APPROVED_STATES, average="macro", zero_division=0))

        # 4. Action Metrics
        y_true_action = [c["expected_action"] for c in self.checkpoints]
        y_pred_action = [d["action"] for d in decisions]
        action_acc = float(accuracy_score(y_true_action, y_pred_action))
        action_f1 = float(f1_score(y_true_action, y_pred_action, labels=APPROVED_ACTIONS, average="macro", zero_division=0))

        # 5. Exact Match Metrics
        exact_metrics = compute_decision_exact_match(self.checkpoints, decisions)

        return {
            "k": k,
            "mode": mode,
            "intent_metrics": intent_metrics,
            "escalation_metrics": esc_metrics,
            "state_metrics": {
                "accuracy": round(state_acc, 4),
                "macro_f1": round(state_f1, 4),
            },
            "action_metrics": {
                "accuracy": round(action_acc, 4),
                "macro_f1": round(action_f1, 4),
            },
            "exact_match_metrics": exact_metrics,
            "decisions": decisions,
        }
