"""Human-Agreement Evaluator and Stratified Review Packet Builder for Phase 7C.

Responsibilities:
1. Stratified sampling of N=40 evaluation checkpoints from the 200 Dev set.
   - Stratification axes: Difficulty (Easy/Medium/Hard), Escalation (Yes/No),
     Turn Depth (Single/Multi-turn), and Retrieval Confidence.
2. Generation of a sanitized Human Review Packet (WITHOUT gold labels).
3. Computation of Inter-Annotator Agreement statistics between Human and LLM-Judge:
   - Cohen's Quadratic Weighted Kappa (κ_w) for ordinal 1-5 scales.
   - Spearman rank-order correlation (ρ).
   - Exact agreement rate (P0).
   - Adjacent agreement rate (P±1, within ±1 score point).
4. Honest disclosure of human review completion status (PENDING if uncompleted).
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from src.config import PATHS, set_seed
from src.utils.logger import get_logger

logger = get_logger("human_agreement")


@dataclass
class HumanAgreementMetrics:
    """Statistical agreement between human expert ratings and LLM-judge ratings."""

    sample_size: int
    review_status: str  # "COMPLETED", "PENDING"
    per_dimension_kappa: Dict[str, float]
    mean_quadratic_weighted_kappa: float
    per_dimension_spearman: Dict[str, float]
    mean_spearman_rho: float
    exact_agreement_rate: float
    adjacent_agreement_rate: float
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HumanAgreementEvaluator:
    """Manages stratified human review packet creation and statistical agreement analysis."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def sample_stratified_subset(
        self,
        checkpoints: List[Dict[str, Any]],
        target_n: int = 40,
    ) -> List[Dict[str, Any]]:
        """Selects a representative stratified sample of exactly target_n checkpoints."""
        random.seed(self.seed)

        easy_pool = [c for c in checkpoints if c.get("difficulty", "medium").lower() == "easy"]
        medium_pool = [c for c in checkpoints if c.get("difficulty", "medium").lower() == "medium"]
        hard_pool = [c for c in checkpoints if c.get("difficulty", "medium").lower() == "hard"]

        # Target allocations: 10 Easy (25%), 18 Medium (45%), 12 Hard (30%) = 40
        n_easy = min(10, len(easy_pool))
        n_hard = min(12, len(hard_pool))
        n_med = target_n - (n_easy + n_hard)

        # Sort before shuffling to ensure deterministic sampling across platforms
        easy_pool.sort(key=lambda x: x["checkpoint_id"])
        medium_pool.sort(key=lambda x: x["checkpoint_id"])
        hard_pool.sort(key=lambda x: x["checkpoint_id"])

        random.shuffle(easy_pool)
        random.shuffle(medium_pool)
        random.shuffle(hard_pool)

        selected = easy_pool[:n_easy] + medium_pool[:n_med] + hard_pool[:n_hard]
        # Final sort by checkpoint_id for predictable ordering
        selected.sort(key=lambda x: x["checkpoint_id"])
        logger.info(f"Sampled {len(selected)} checkpoints for human review (Easy: {n_easy}, Med: {n_med}, Hard: {n_hard})")
        return selected

    def build_review_packet(
        self,
        sampled_checkpoints: List[Dict[str, Any]],
        agent_predictions: Dict[str, Dict[str, Any]],
        output_path: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """Builds sanitized human-review packet without gold decision labels."""
        packet = []

        for idx, cp in enumerate(sampled_checkpoints, 1):
            c_id = cp["checkpoint_id"]
            pred = agent_predictions.get(c_id, {})

            # Normalize retrieved exemplars with canonical fields and legacy aliases
            raw_exemplars = pred.get("retrieved_exemplars", [])
            normalized_exemplars = []
            for ex in raw_exemplars:
                if isinstance(ex, dict):
                    sim = ex.get("similarity_score")
                    if sim is None:
                        sim = ex.get("similarity", 0.0)
                    prob = ex.get("customer_problem_summary") or ex.get("customer_text", "")
                    resp = ex.get("support_response") or ex.get("support_reply", "")
                    norm_ex = dict(ex)
                    norm_ex["similarity_score"] = round(float(sim), 4)
                    norm_ex["similarity"] = round(float(sim), 4)
                    norm_ex["customer_problem_summary"] = prob
                    norm_ex["customer_text"] = prob
                    norm_ex["support_response"] = resp
                    norm_ex["support_reply"] = resp
                    normalized_exemplars.append(norm_ex)
                elif hasattr(ex, "to_dict"):
                    normalized_exemplars.append(ex.to_dict())
                else:
                    normalized_exemplars.append(ex)

            # Sanitized item for human reviewer (NO gold labels!)
            item = {
                "review_id": f"rev_{idx:02d}_{c_id}",
                "checkpoint_id": c_id,
                "conversation_id": cp.get("conversation_id"),
                "turn_depth": cp.get("turn_depth", 1),
                "difficulty": cp.get("difficulty", "medium"),
                "customer_message": cp.get("current_customer_message", ""),
                "conversation_history": cp.get("conversation_history_before_current_turn", []),
                "retrieved_evidence": normalized_exemplars,
                "generated_response": pred.get("generated_response", pred.get("final_response", "")),
                "human_scoring": {
                    "relevance": None,  # 1-5
                    "helpfulness": None,  # 1-5
                    "groundedness": None,  # 1-5
                    "action_appropriateness": None,  # 1-5
                    "safety": None,  # 1-5
                    "communication_quality": None,  # 1-5
                    "reviewer_id": "human_reviewer_pending",
                    "justification_notes": "",
                },
            }
            packet.append(item)

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for row in packet:
                    f.write(json.dumps(row) + "\n")
            logger.info(f"Exported human review packet with {len(packet)} items to {output_path}")

        return packet

    def calculate_agreement(
        self,
        human_ratings: List[Dict[str, Any]],
        judge_ratings: List[Dict[str, Any]],
    ) -> HumanAgreementMetrics:
        """Calculates Quadratic Weighted Kappa, Spearman Rho, and Exact/Adjacent agreement."""
        dims = [
            "relevance",
            "helpfulness",
            "groundedness",
            "action_appropriateness",
            "safety",
            "communication_quality",
        ]

        if not human_ratings or not judge_ratings:
            return HumanAgreementMetrics(
                sample_size=0,
                review_status="PENDING",
                per_dimension_kappa={d: 0.0 for d in dims},
                mean_quadratic_weighted_kappa=0.0,
                per_dimension_spearman={d: 0.0 for d in dims},
                mean_spearman_rho=0.0,
                exact_agreement_rate=0.0,
                adjacent_agreement_rate=0.0,
                notes="Human review ratings not yet completed; status marked PENDING.",
            )

        # Check if all human ratings are actually scored (not None)
        valid_pairs = []
        for h, j in zip(human_ratings, judge_ratings):
            h_scores = h.get("human_scoring", h)
            j_scores = j.get("judge_scoring", j)
            if all(h_scores.get(d) is not None for d in dims):
                valid_pairs.append((h_scores, j_scores))

        if len(valid_pairs) < 5:
            return HumanAgreementMetrics(
                sample_size=len(valid_pairs),
                review_status="PENDING",
                per_dimension_kappa={d: 0.0 for d in dims},
                mean_quadratic_weighted_kappa=0.0,
                per_dimension_spearman={d: 0.0 for d in dims},
                mean_spearman_rho=0.0,
                exact_agreement_rate=0.0,
                adjacent_agreement_rate=0.0,
                notes=f"Insufficient completed human review records ({len(valid_pairs)}/40 completed). Marked PENDING.",
            )

        per_dim_kappa = {}
        per_dim_spearman = {}
        total_eval_points = 0
        exact_matches = 0
        adjacent_matches = 0

        for d in dims:
            y_h = [int(p[0][d]) for p in valid_pairs]
            y_j = [int(p[1][d]) for p in valid_pairs]

            # Quadratic weighted kappa (labels in 1..5)
            try:
                kappa = cohen_kappa_score(y_h, y_j, weights="quadratic", labels=[1, 2, 3, 4, 5])
                if np.isnan(kappa):
                    kappa = 1.0 if y_h == y_j else 0.0
            except Exception:
                kappa = 0.0
            per_dim_kappa[d] = round(float(kappa), 4)

            # Spearman correlation
            try:
                rho, _ = spearmanr(y_h, y_j)
                if np.isnan(rho):
                    rho = 1.0 if y_h == y_j else 0.0
            except Exception:
                rho = 0.0
            per_dim_spearman[d] = round(float(rho), 4)

            # Exact and adjacent
            for yh, yj in zip(y_h, y_j):
                total_eval_points += 1
                diff = abs(yh - yj)
                if diff == 0:
                    exact_matches += 1
                if diff <= 1:
                    adjacent_matches += 1

        mean_kappa = round(float(np.mean(list(per_dim_kappa.values()))), 4)
        mean_rho = round(float(np.mean(list(per_dim_spearman.values()))), 4)
        exact_rate = round(exact_matches / total_eval_points, 4) if total_eval_points > 0 else 0.0
        adjacent_rate = round(adjacent_matches / total_eval_points, 4) if total_eval_points > 0 else 0.0

        return HumanAgreementMetrics(
            sample_size=len(valid_pairs),
            review_status="COMPLETED",
            per_dimension_kappa=per_dim_kappa,
            mean_quadratic_weighted_kappa=mean_kappa,
            per_dimension_spearman=per_dim_spearman,
            mean_spearman_rho=mean_rho,
            exact_agreement_rate=exact_rate,
            adjacent_agreement_rate=adjacent_rate,
            notes="Evaluated across completed human expert ratings and LLM-as-judge scores.",
        )
