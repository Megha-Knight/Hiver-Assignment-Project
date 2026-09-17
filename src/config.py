"""Central configuration module for the AmazonHelp Support Agent.

Manages reproducible random seeds, dataset paths, model hyperparameters,
and artifact output directories.
"""

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds for Python standard library and NumPy."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@dataclass(frozen=True)
class PathConfig:
    """Project directory and file paths."""

    # Project directory roots
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    WORKSPACE_ROOT: Path = PROJECT_ROOT.parent

    # Data paths
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    GOLDEN_DATA_DIR: Path = DATA_DIR / "golden"

    # Raw Kaggle TWCS dataset location (checks workspace first, then project raw dir)
    WORKSPACE_RAW_CSV: Path = WORKSPACE_ROOT / "twcs" / "twcs.csv"
    PROJECT_RAW_CSV: Path = RAW_DATA_DIR / "twcs.csv"
    SAMPLE_CSV: Path = WORKSPACE_ROOT / "sample.csv"

    # Results & documentation
    RESULTS_DIR: Path = PROJECT_ROOT / "results"
    DOCS_DIR: Path = PROJECT_ROOT / "docs"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"

    # Specific experiment artifacts
    BRAND_COMPARISON_CSV: Path = RESULTS_DIR / "brand_comparison.csv"
    BRAND_SELECTION_REPORT_MD: Path = DOCS_DIR / "brand_selection_report.md"
    RECONSTRUCTED_SAMPLE_JSON: Path = RESULTS_DIR / "reconstructed_conversations_sample.json"

    # Phase 2 Dataset & Analysis Artifacts
    AMAZONHELP_ENGLISH_JSONL: Path = PROCESSED_DATA_DIR / "amazonhelp_english.jsonl"
    AMAZONHELP_NON_ENGLISH_JSONL: Path = PROCESSED_DATA_DIR / "amazonhelp_non_english_or_uncertain.jsonl"
    AMAZONHELP_EXPLORATION_JSONL: Path = PROCESSED_DATA_DIR / "amazonhelp_exploration.jsonl"
    AMAZONHELP_RETRIEVAL_JSONL: Path = PROCESSED_DATA_DIR / "amazonhelp_retrieval_candidates.jsonl"
    AMAZONHELP_PRISTINE_JSONL: Path = PROCESSED_DATA_DIR / "amazonhelp_pristine_candidates.jsonl"
    PREPROCESSING_STATS_JSON: Path = RESULTS_DIR / "preprocessing_statistics.json"
    ANALYSIS_DIR: Path = RESULTS_DIR / "analysis"

    # Phase 2 Documentation Artifacts
    LANGUAGE_FILTER_REPORT_MD: Path = DOCS_DIR / "language_filter_report.md"
    INTENT_DISCOVERY_EXAMPLES_MD: Path = DOCS_DIR / "intent_discovery_examples.md"
    INTENT_TAXONOMY_PROPOSAL_MD: Path = DOCS_DIR / "intent_taxonomy_proposal.md"
    CONVERSATION_STATE_PROPOSAL_MD: Path = DOCS_DIR / "conversation_state_proposal.md"
    AGENT_ACTION_PROPOSAL_MD: Path = DOCS_DIR / "agent_action_proposal.md"
    AUTONOMY_POLICY_MD: Path = DOCS_DIR / "autonomy_policy.md"
    EVALUATION_SPLIT_STRATEGY_MD: Path = DOCS_DIR / "evaluation_split_strategy.md"

    # Phase 3 Dataset & Retrieval Artifacts
    RETRIEVAL_DATA_DIR: Path = DATA_DIR / "retrieval"
    INDEXES_DATA_DIR: Path = DATA_DIR / "indexes"
    AMAZONHELP_TRAIN_RETRIEVAL_JSONL: Path = RETRIEVAL_DATA_DIR / "amazonhelp_train_retrieval.jsonl"
    GOLDEN_CANDIDATES_JSONL: Path = GOLDEN_DATA_DIR / "golden_candidates.jsonl"
    AMAZONHELP_GOLDEN_JSONL: Path = GOLDEN_DATA_DIR / "amazonhelp_golden_v1.jsonl"
    AMAZONHELP_GOLDEN_HUMAN_VALIDATED_JSONL: Path = GOLDEN_DATA_DIR / "amazonhelp_golden_v1_human_validated.jsonl"
    RETRIEVAL_INDEX_NPZ: Path = INDEXES_DATA_DIR / "retrieval_index.npz"
    RETRIEVAL_INDEX_META: Path = INDEXES_DATA_DIR / "retrieval_index_meta.json"
    PHASE3_STATS_JSON: Path = RESULTS_DIR / "phase3_statistics.json"
    RETRIEVAL_BUILD_REPORT_MD: Path = RESULTS_DIR / "retrieval_build_report.md"
    GOLDEN_HUMAN_ANNOTATION_REPORT_MD: Path = RESULTS_DIR / "golden_human_annotation_report.md"

    # Phase 3 Documentation Artifacts
    GOLDEN_DATASET_PROTOCOL_MD: Path = DOCS_DIR / "golden_dataset_protocol.md"
    RETRIEVAL_DATASET_SPEC_MD: Path = DOCS_DIR / "retrieval_dataset_spec.md"
    PHASE3_STATE_MACHINE_REVIEW_MD: Path = DOCS_DIR / "phase3_state_machine_review.md"
    PHASE3_ACTION_SPACE_REVIEW_MD: Path = DOCS_DIR / "phase3_action_space_review.md"
    PHASE3_LEAKAGE_REPORT_MD: Path = DOCS_DIR / "phase3_leakage_report.md"

    # Phase 4 Baselines, Policies & Models Artifacts
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    BASELINES_MODELS_DIR: Path = MODELS_DIR / "baselines"
    RESULTS_BASELINES_DIR: Path = RESULTS_DIR / "baselines"
    TFIDF_LOGREG_MODEL_PATH: Path = BASELINES_MODELS_DIR / "tfidf_logreg_intent.joblib"
    PHASE4_BASELINE_REPORT_MD: Path = RESULTS_DIR / "phase4_baseline_report.md"
    PHASE4_POLICY_REPORT_MD: Path = RESULTS_DIR / "phase4_policy_report.md"
    PHASE4_BASELINE_PROTOCOL_MD: Path = DOCS_DIR / "phase4_baseline_protocol.md"
    PHASE4_DECISION_POLICY_MD: Path = DOCS_DIR / "phase4_decision_policy.md"

    # Phase 5 Retrieval Augmentation Artifacts
    PHASE5_RETRIEVAL_REPORT_MD: Path = RESULTS_DIR / "phase5_retrieval_report.md"
    PHASE5_FAILURE_ANALYSIS_MD: Path = RESULTS_DIR / "phase5_failure_analysis.md"
    PHASE5_RETRIEVAL_METRICS_JSON: Path = RESULTS_DIR / "phase5_retrieval_metrics.json"
    PHASE5_RETRIEVAL_PROTOCOL_MD: Path = DOCS_DIR / "phase5_retrieval_protocol.md"

    # Phase 6 Local LLM Artifacts
    RESULTS_PHASE6_DIR: Path = RESULTS_DIR / "phase6"
    MODEL_BENCHMARK_JSON: Path = RESULTS_PHASE6_DIR / "model_benchmark.json"
    MODEL_COMPARISON_MD: Path = RESULTS_PHASE6_DIR / "model_comparison.md"
    MODEL_SELECTION_REPORT_MD: Path = RESULTS_PHASE6_DIR / "model_selection_report.md"
    PHASE6_LLM_PROTOCOL_MD: Path = DOCS_DIR / "phase6_llm_protocol.md"

    # Phase 6B LLM-Only Controlled Agent Artifacts
    PHASE6B_METRICS_JSON: Path = RESULTS_PHASE6_DIR / "phase6b_metrics.json"
    PHASE6B_REPORT_MD: Path = RESULTS_PHASE6_DIR / "phase6b_report.md"
    PHASE6B_FAILURE_ANALYSIS_MD: Path = RESULTS_PHASE6_DIR / "phase6b_failure_analysis.md"
    PHASE6B_LLM_AGENT_PROTOCOL_MD: Path = DOCS_DIR / "phase6b_llm_agent_protocol.md"

    # Phase 6C LLM + Retrieval + Structured Policy Artifacts
    PHASE6C_METRICS_JSON: Path = RESULTS_PHASE6_DIR / "phase6c_metrics.json"
    PHASE6C_REPORT_MD: Path = RESULTS_PHASE6_DIR / "phase6c_report.md"
    PHASE6C_FAILURE_ANALYSIS_MD: Path = RESULTS_PHASE6_DIR / "phase6c_failure_analysis.md"
    PHASE6C_RETRIEVAL_ANALYSIS_MD: Path = RESULTS_PHASE6_DIR / "phase6c_retrieval_analysis.md"
    PHASE6C_PROTOCOL_MD: Path = DOCS_DIR / "phase6c_protocol.md"

    def get_raw_dataset_path(self) -> Path:
        """Returns the valid raw dataset path without modifying raw data."""
        if self.WORKSPACE_RAW_CSV.exists():
            return self.WORKSPACE_RAW_CSV
        if self.PROJECT_RAW_CSV.exists():
            return self.PROJECT_RAW_CSV
        if self.SAMPLE_CSV.exists():
            return self.SAMPLE_CSV
        raise FileNotFoundError(
            f"Raw dataset not found at {self.WORKSPACE_RAW_CSV} or {self.PROJECT_RAW_CSV}"
        )

    def ensure_directories(self) -> None:
        """Creates required directories if they do not exist."""
        for directory in [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.GOLDEN_DATA_DIR,
            self.RETRIEVAL_DATA_DIR,
            self.INDEXES_DATA_DIR,
            self.RESULTS_DIR,
            self.RESULTS_BASELINES_DIR,
            self.RESULTS_PHASE6_DIR,
            self.DOCS_DIR,
            self.LOGS_DIR,
            self.ANALYSIS_DIR,
            self.MODELS_DIR,
            self.BASELINES_MODELS_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ModelConfig:
    """Agent and embedding model configuration for take-home assignment."""

    DEFAULT_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")
    DEFAULT_MODEL: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    MAX_TOKENS: int = 1024
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


@dataclass(frozen=True)
class AuditConfig:
    """Configuration for dataset audit and conversation reconstruction."""

    SEED: int = int(os.getenv("RANDOM_SEED", "42"))
    CHUNK_SIZE: int = 100_000
    CANDIDATE_BRANDS: List[str] = field(
        default_factory=lambda: [
            "AmazonHelp",
            "AppleSupport",
            "Uber_Support",
            "SpotifyCares",
        ]
    )
    SAMPLE_CONVERSATIONS_PER_BRAND: int = 5
    MAX_CONVERSATION_DEPTH: int = 50


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for Phase 2 dataset preprocessing and analysis pipeline."""

    BRAND: str = "AmazonHelp"
    SEED: int = int(os.getenv("RANDOM_SEED", "42"))
    ENGLISH_CONFIDENCE_THRESHOLD: float = 0.80
    MULTI_PART_MAX_GAP_SECONDS: float = 300.0  # 5 minutes
    MIN_RETRIEVAL_TURNS: int = 2
    INTENT_SAMPLE_SIZE: int = 10_000
    INTENT_CLUSTERS_K: int = 12
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"


@dataclass(frozen=True)
class Phase3Config:
    """Configuration for Phase 3 Golden Evaluation & Train-Only Retrieval."""

    SEED: int = int(os.getenv("RANDOM_SEED", "42"))
    TRAIN_SPLIT_RATIO: float = 0.80
    VAL_SPLIT_RATIO: float = 0.10
    TEST_SPLIT_RATIO: float = 0.10
    GOLDEN_SAMPLE_TARGET: int = 200
    RETRIEVAL_SIMILARITY_TOP_K: int = 5
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"


# Global instances
PATHS = PathConfig()
MODELS = ModelConfig()
AUDIT_CONFIG = AuditConfig()
PIPELINE_CONFIG = PipelineConfig()
PHASE3_CONFIG = Phase3Config()

# Ensure directories are ready
PATHS.ensure_directories()
# Set seed upon config import
set_seed(AUDIT_CONFIG.SEED)


