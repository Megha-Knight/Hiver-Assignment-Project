"""TF-IDF and Logistic Regression Baseline for Customer Intent Classification.

Trained strictly on the Train partition (42,909 pristine conversations).
Zero Validation or Test leakage.
"""

from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sample_golden_candidates import classify_inferred_intent
from src.annotation.annotator import APPROVED_INTENTS
from src.config import PATHS, PHASE3_CONFIG, set_seed
from src.utils.logger import get_logger

logger = get_logger("tfidf_logreg")


def format_input_text(customer_message: str, history: Optional[List[Dict[str, Any]]] = None) -> str:
    """Formats customer utterance with immediate prior support turn context.

    Input Representation:
    - Turn 1: "Customer: <customer_message>"
    - Turn > 1: "Support Context: <last_support_text> | Customer: <customer_message>"
    """
    msg = customer_message.strip()
    if not history:
        return f"Customer: {msg}"

    last_support = ""
    for h in reversed(history):
        if h.get("role") == "support":
            last_support = h.get("text", "").strip()
            break

    if last_support:
        return f"Support Context: {last_support} | Customer: {msg}"
    return f"Customer: {msg}"


class TfidfLogRegIntentClassifier:
    """TF-IDF + Logistic Regression Intent Classifier."""

    def __init__(
        self,
        max_features: int = 15000,
        ngram_range: Tuple[int, int] = (1, 2),
        C: float = 1.0,
        random_state: int = 42,
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.C = C
        self.random_state = random_state
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.model: Optional[LogisticRegression] = None
        self.classes_: List[str] = list(APPROVED_INTENTS)

    def fit(self, texts: List[str], labels: List[str]):
        """Fits TF-IDF vectorizer and Logistic Regression model on training data."""
        logger.info(f"Fitting TfidfVectorizer on {len(texts):,} training examples...")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=True,
            stop_words="english",
            token_pattern=r"(?u)\b\w+\b",
        )
        X = self.vectorizer.fit_transform(texts)

        logger.info(f"Fitting LogisticRegression (class_weight='balanced', C={self.C})...")
        self.model = LogisticRegression(
            C=self.C,
            max_iter=1000,
            class_weight="balanced",
            random_state=self.random_state,
            solver="lbfgs",
        )
        self.model.fit(X, labels)
        self.classes_ = list(self.model.classes_)
        logger.info(f"Model trained successfully. Number of classes: {len(self.classes_)}")

    def predict(self, texts: List[str]) -> List[str]:
        """Predicts intent labels for given input texts."""
        if not self.vectorizer or not self.model:
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")
        X = self.vectorizer.transform(texts)
        return list(self.model.predict(X))

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Returns predicted probability distribution across classes."""
        if not self.vectorizer or not self.model:
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)

    def predict_with_confidence(self, customer_message: str, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Classifies a single utterance and returns top intent and confidence."""
        formatted_text = format_input_text(customer_message, history)
        if not self.vectorizer or not self.model:
            raise RuntimeError("Model is not fitted.")

        X = self.vectorizer.transform([formatted_text])
        probas = self.model.predict_proba(X)[0]
        top_idx = int(np.argmax(probas))
        top_intent = self.classes_[top_idx]
        confidence = float(probas[top_idx])

        return {
            "intent": top_intent,
            "confidence": confidence,
            "distribution": {cls: float(p) for cls, p in zip(self.classes_, probas)},
        }

    def save(self, filepath: Path):
        """Persists fitted model and vectorizer to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "model": self.model,
                "classes_": self.classes_,
                "max_features": self.max_features,
                "ngram_range": self.ngram_range,
                "C": self.C,
                "random_state": self.random_state,
            },
            filepath,
        )
        logger.info(f"Persisted TF-IDF + Logistic Regression model to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "TfidfLogRegIntentClassifier":
        """Loads fitted model and vectorizer from disk."""
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found at {filepath}")
        data = joblib.load(filepath)
        instance = cls(
            max_features=data["max_features"],
            ngram_range=data["ngram_range"],
            C=data["C"],
            random_state=data["random_state"],
        )
        instance.vectorizer = data["vectorizer"]
        instance.model = data["model"]
        instance.classes_ = data["classes_"]
        logger.info(f"Loaded TF-IDF + Logistic Regression model from {filepath}")
        return instance


def build_train_intent_dataset(limit_conversations: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """Extracts customer turns strictly from the Train partition and derives intent labels.

    Partition Isolation:
    - Pristine dataset is chronologically sorted.
    - Strictly the first 80% (42,909 conversations) are used.
    - Zero Dev/Validation and zero Test conversations are touched.
    """
    logger.info(f"Loading Train partition from {PATHS.AMAZONHELP_PRISTINE_JSONL}...")
    all_convs = []
    with open(PATHS.AMAZONHELP_PRISTINE_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            all_convs.append(json.loads(line))

    all_convs.sort(key=lambda x: (x["start_timestamp"], x["conversation_id"]))
    n = len(all_convs)
    n_train = int(n * PHASE3_CONFIG.TRAIN_SPLIT_RATIO)
    train_convs = all_convs[:n_train]
    if limit_conversations:
        train_convs = train_convs[:limit_conversations]

    logger.info(f"Extracting customer turns from {len(train_convs):,} Train conversations...")
    texts = []
    labels = []

    for conv in train_convs:
        turns = conv["normalized_turns"]
        history: List[Dict[str, Any]] = []

        for turn in turns:
            if turn["role"] == "customer":
                cust_text = turn["text"].strip()
                if cust_text:
                    intent = classify_inferred_intent(cust_text)
                    formatted_input = format_input_text(cust_text, history)
                    texts.append(formatted_input)
                    labels.append(intent)

            history.append({
                "turn_index": turn["turn_index"],
                "role": turn["role"],
                "text": turn["text"],
            })

    logger.info(f"Extracted {len(texts):,} Train customer turns. Intent distribution: {dict(Counter(labels))}")
    return texts, labels
