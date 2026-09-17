"""Evaluation metrics framework for historical conversation retrieval and dialogue decisions.

Implements:
1. Retrieval Ranking Metrics: Recall@K, Mean Reciprocal Rank (MRR), and nDCG@K.
2. Dialogue Decision Metrics: Intent Accuracy, State Accuracy, Action Precision/Recall/F1, and Escalation F1.
"""

from typing import Any, Dict, List, Optional, Set, Union
import numpy as np


def compute_recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 5) -> float:
    """Computes Recall@K: Proportion of relevant documents retrieved in top-K."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / min(len(relevant_ids), k)


def compute_mrr(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 10) -> float:
    """Computes Mean Reciprocal Rank (MRR): 1 / rank of first relevant match."""
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(retrieved_ids: List[str], relevance_scores: Dict[str, float], k: int = 5) -> float:
    """Computes Normalized Discounted Cumulative Gain (nDCG@K)."""
    top_k = retrieved_ids[:k]
    dcg = 0.0
    for rank, doc_id in enumerate(top_k, start=1):
        rel = relevance_scores.get(doc_id, 0.0)
        dcg += (2.0 ** rel - 1.0) / np.log2(rank + 1.0)

    # Ideal DCG
    ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = 0.0
    for rank, rel in enumerate(ideal_rels, start=1):
        idcg += (2.0 ** rel - 1.0) / np.log2(rank + 1.0)

    return (dcg / idcg) if idcg > 0.0 else 0.0


from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support


def evaluate_decision_predictions(
    ground_truth: List[Dict[str, Union[str, bool]]],
    predictions: List[Dict[str, Union[str, bool]]],
) -> Dict[str, float]:
    """Computes multi-task decision accuracy across intent, state, action, and escalation."""
    assert len(ground_truth) == len(predictions), "Mismatched evaluation lengths."
    total = len(ground_truth)
    if total == 0:
        return {}

    intent_correct = sum(1 for gt, pr in zip(ground_truth, predictions) if gt["expected_intent"] == pr["predicted_intent"])
    state_correct = sum(1 for gt, pr in zip(ground_truth, predictions) if gt["expected_state"] == pr["predicted_state"])
    action_correct = sum(1 for gt, pr in zip(ground_truth, predictions) if gt["expected_action"] == pr["predicted_action"])

    # Escalation metrics
    tp = sum(1 for gt, pr in zip(ground_truth, predictions) if gt["expected_escalation"] and pr["predicted_escalation"])
    fp = sum(1 for gt, pr in zip(ground_truth, predictions) if not gt["expected_escalation"] and pr["predicted_escalation"])
    fn = sum(1 for gt, pr in zip(ground_truth, predictions) if gt["expected_escalation"] and not pr["predicted_escalation"])
    tn = sum(1 for gt, pr in zip(ground_truth, predictions) if not gt["expected_escalation"] and not pr["predicted_escalation"])

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    fahr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "intent_accuracy": round(intent_correct / total, 4),
        "state_accuracy": round(state_correct / total, 4),
        "action_accuracy": round(action_correct / total, 4),
        "escalation_precision": round(prec, 4),
        "escalation_recall": round(rec, 4),
        "escalation_f1": round(f1, 4),
        "false_auto_handle_rate": round(fahr, 4),
    }


def compute_intent_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Computes comprehensive multi-class metrics for Intent Classification."""
    from src.annotation.annotator import APPROVED_INTENTS
    all_labels = labels or APPROVED_INTENTS

    total = len(y_true)
    acc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / total if total > 0 else 0.0
    macro_f1 = f1_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=all_labels, average="weighted", zero_division=0)

    p_per, r_per, f1_per, s_per = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, zero_division=0
    )

    per_intent = {}
    for lbl, p, r, f, s in zip(all_labels, p_per, r_per, f1_per, s_per):
        per_intent[lbl] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "support": int(s),
        }

    cm = confusion_matrix(y_true, y_pred, labels=all_labels).tolist()

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_intent": per_intent,
        "confusion_matrix": cm,
        "labels": all_labels,
    }


def compute_escalation_metrics(
    y_true: List[bool],
    y_pred: List[bool],
) -> Dict[str, float]:
    """Computes precision, recall, F1, and False Auto-Handle Rate (FAHR) for Escalation."""
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and yp)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt and not yp)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and not yp)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    fahr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "false_auto_handle_rate": round(fahr, 4),
    }


def compute_decision_exact_match(
    eval_checkpoints: List[Dict[str, Any]],
    predicted_decisions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Computes exact-match agreement across intent, state, action, and escalation."""
    total = len(eval_checkpoints)
    if total == 0:
        return {}

    intent_matches = 0
    state_matches = 0
    action_matches = 0
    esc_matches = 0
    exact_all_matches = 0

    by_difficulty: Dict[str, Dict[str, int]] = {
        "easy": {"total": 0, "exact": 0},
        "medium": {"total": 0, "exact": 0},
        "hard": {"total": 0, "exact": 0},
    }

    for gt, pr in zip(eval_checkpoints, predicted_decisions):
        diff = gt.get("difficulty", "medium")
        if diff not in by_difficulty:
            by_difficulty[diff] = {"total": 0, "exact": 0}
        by_difficulty[diff]["total"] += 1

        m_intent = gt["expected_intent"] == pr["intent"]
        m_state = gt["expected_state"] == pr["state"]
        m_action = gt["expected_action"] == pr["action"]
        m_esc = gt["expected_escalation"] == pr["escalation"]

        if m_intent:
            intent_matches += 1
        if m_state:
            state_matches += 1
        if m_action:
            action_matches += 1
        if m_esc:
            esc_matches += 1

        if m_intent and m_state and m_action and m_esc:
            exact_all_matches += 1
            by_difficulty[diff]["exact"] += 1

    diff_breakdown = {}
    for d, counts in by_difficulty.items():
        rate = round(counts["exact"] / counts["total"], 4) if counts["total"] > 0 else 0.0
        diff_breakdown[d] = {
            "total": counts["total"],
            "exact_matches": counts["exact"],
            "exact_match_rate": rate,
        }

    return {
        "total_checkpoints": total,
        "intent_match_rate": round(intent_matches / total, 4),
        "state_match_rate": round(state_matches / total, 4),
        "action_match_rate": round(action_matches / total, 4),
        "escalation_match_rate": round(esc_matches / total, 4),
        "exact_match_all_rate": round(exact_all_matches / total, 4),
        "exact_matches_count": exact_all_matches,
        "by_difficulty": diff_breakdown,
    }
