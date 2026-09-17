"""Comparative Evaluation Inspector across Phase 4, Phase 5, Phase 6B, and Phase 6C.

Compares:
1. Phase 4 Deterministic Baseline (TF-IDF + LogReg + Rule-based policy)
2. Phase 5 Retrieval-Only Augmentation (K=5 MiniLM)
3. Phase 6B LLM-Only Controlled Agent (llama3.2:1b, Zero Retrieval)
4. Phase 6C LLM + Retrieval + Structured Policy (llama3.2:1b, K=5)

Generates standardized console comparison table and prints statistical deltas.
"""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import PATHS
from src.utils.logger import get_logger

logger = get_logger("compare_phase4_phase6")


def main():
    logger.info("=" * 70)
    logger.info("CROSS-PHASE COMPARATIVE BENCHMARK INSPECTOR")
    logger.info("=" * 70)

    # 1. Phase 4 Reference
    p4 = {
        "intent_acc": 0.8750,
        "intent_f1": 0.8308,
        "state_acc": 0.8950,
        "state_f1": 0.4839,
        "action_acc": 0.8850,
        "action_f1": 0.8017,
        "esc_prec": 0.7857,
        "esc_rec": 0.2444,
        "esc_f1": 0.3729,
        "fahr": 0.7556,
        "exact_match": 0.6900,
        "hard_exact": 0.3333,
    }

    # 2. Phase 5 Reference (K=5)
    p5 = {
        "intent_acc": 0.8450,
        "intent_f1": 0.8012,
        "state_acc": 0.8950,
        "state_f1": 0.4839,
        "action_acc": 0.8700,
        "action_f1": 0.7840,
        "esc_prec": 0.7500,
        "esc_rec": 0.2667,
        "esc_f1": 0.3934,
        "fahr": 0.7333,
        "exact_match": 0.6650,
        "hard_exact": 0.3000,
    }

    # 3. Phase 6B Metrics
    if PATHS.PHASE6B_METRICS_JSON.exists():
        with open(PATHS.PHASE6B_METRICS_JSON, "r", encoding="utf-8") as f:
            p6b_data = json.load(f)
        p6b = {
            "intent_acc": p6b_data["intent_metrics"]["accuracy"],
            "intent_f1": p6b_data["intent_metrics"]["macro_f1"],
            "state_acc": p6b_data["state_metrics"]["accuracy"],
            "state_f1": p6b_data["state_metrics"]["macro_f1"],
            "action_acc": p6b_data["action_metrics"]["accuracy"],
            "action_f1": p6b_data["action_metrics"]["macro_f1"],
            "esc_prec": p6b_data["escalation_metrics"]["precision"],
            "esc_rec": p6b_data["escalation_metrics"]["recall"],
            "esc_f1": p6b_data["escalation_metrics"]["f1"],
            "fahr": p6b_data["escalation_metrics"]["false_auto_handle_rate"],
            "exact_match": p6b_data["exact_match_metrics"]["exact_match_all_rate"],
            "hard_exact": p6b_data["exact_match_metrics"]["by_difficulty"]["hard"]["exact_match_rate"],
        }
    else:
        logger.warning(f"Phase 6B metrics not found at {PATHS.PHASE6B_METRICS_JSON}")
        p6b = {k: 0.0 for k in p4}

    # 4. Phase 6C Metrics
    if PATHS.PHASE6C_METRICS_JSON.exists():
        with open(PATHS.PHASE6C_METRICS_JSON, "r", encoding="utf-8") as f:
            p6c_data = json.load(f)
        p6c = {
            "intent_acc": p6c_data["intent_metrics"]["accuracy"],
            "intent_f1": p6c_data["intent_metrics"]["macro_f1"],
            "state_acc": p6c_data["state_metrics"]["accuracy"],
            "state_f1": p6c_data["state_metrics"]["macro_f1"],
            "action_acc": p6c_data["action_metrics"]["accuracy"],
            "action_f1": p6c_data["action_metrics"]["macro_f1"],
            "esc_prec": p6c_data["escalation_metrics"]["precision"],
            "esc_rec": p6c_data["escalation_metrics"]["recall"],
            "esc_f1": p6c_data["escalation_metrics"]["f1"],
            "fahr": p6c_data["escalation_metrics"]["false_auto_handle_rate"],
            "exact_match": p6c_data["exact_match_metrics"]["exact_match_all_rate"],
            "hard_exact": p6c_data["exact_match_metrics"]["by_difficulty"]["hard"]["exact_match_rate"],
        }
    else:
        logger.warning(f"Phase 6C metrics not found at {PATHS.PHASE6C_METRICS_JSON}")
        p6c = {k: 0.0 for k in p4}

    rows = [
        ("Intent Accuracy", "intent_acc"),
        ("Intent Macro-F1", "intent_f1"),
        ("State Accuracy", "state_acc"),
        ("State Macro-F1", "state_f1"),
        ("Action Accuracy", "action_acc"),
        ("Action Macro-F1", "action_f1"),
        ("Escalation Precision", "esc_prec"),
        ("Escalation Recall", "esc_rec"),
        ("Escalation F1", "esc_f1"),
        ("False Auto-Handle Rate", "fahr"),
        ("Overall Exact Match", "exact_match"),
        ("Hard Exact Match", "hard_exact"),
    ]

    header = f"{'Metric':<25} | {'Phase 4':<9} | {'Phase 5':<9} | {'Phase 6B':<9} | {'Phase 6C':<9} | {'Delta(6C-4)':<11} | {'Delta(6C-6B)':<12}"
    divider = "-" * len(header)
    print("\n" + divider)
    print(header)
    print(divider)

    for label, k in rows:
        val_p4 = p4[k] * 100
        val_p5 = p5[k] * 100
        val_p6b = p6b[k] * 100
        val_p6c = p6c[k] * 100
        d_p4 = val_p6c - val_p4
        d_p6b = val_p6c - val_p6b
        print(f"{label:<25} | {val_p4:6.2f}%   | {val_p5:6.2f}%   | {val_p6b:6.2f}%   | {val_p6c:6.2f}%   | {d_p4:+7.2f}%    | {d_p6b:+7.2f}%")

    print(divider + "\n")


if __name__ == "__main__":
    main()
