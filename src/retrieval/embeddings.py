"""Modular local embedding interface for historical conversation retrieval.

Provides:
1. Abstract BaseEmbedder interface for pluggable representation models.
2. SentenceTransformerEmbedder for local open-weight dense embeddings.
3. LSAEmbedder (TF-IDF + TruncatedSVD) as a robust deterministic local fallback.
4. L2 normalization guaranteeing cosine similarity via dot product.
"""

from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.config import PIPELINE_CONFIG
from src.utils.logger import get_logger

logger = get_logger("embeddings")


class BaseEmbedder(ABC):
    """Abstract interface for local text embedding models."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Encodes a list of texts into a 2D float32 normalized numpy array."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Encodes a single query string into a 1D float32 normalized vector."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns embedding vector dimension."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns model identifier."""
        pass


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local SentenceTransformer dense embedding model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        import torch
        from sentence_transformers import SentenceTransformer
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(model_name)
        self._dim = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.model.encode([query], show_progress_bar=False, normalize_embeddings=True)[0]
        return np.asarray(vec, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return f"SentenceTransformer({self._model_name})"


class LSAEmbedder(BaseEmbedder):
    """Latent Semantic Analysis (TF-IDF + TruncatedSVD) dense embedding engine."""

    def __init__(self, n_components: int = 128, seed: int = 42):
        self._dim = n_components
        self._seed = seed
        self._model_name = f"LSA-TruncatedSVD-{n_components}d"
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import Normalizer

        self.tfidf = TfidfVectorizer(
            max_features=8000,
            sublinear_tf=True,
            ngram_range=(1, 2),
            stop_words="english",
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=seed)
        self.normalizer = Normalizer(copy=False)
        self.is_fitted = False

    def fit(self, texts: List[str]):
        tfidf_mat = self.tfidf.fit_transform(texts)
        n_comp = min(self._dim, tfidf_mat.shape[1] - 1)
        self.svd.n_components = n_comp
        self._dim = n_comp
        self.svd.fit(tfidf_mat)
        self.is_fitted = True
        return self

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            self.fit(texts)
        sparse_mat = self.tfidf.transform(texts)
        dense_vecs = self.svd.transform(sparse_mat)
        normalized = self.normalizer.transform(dense_vecs)
        return np.asarray(normalized, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("LSAEmbedder must be fitted on corpus before querying.")
        sparse = self.tfidf.transform([query])
        dense = self.svd.transform(sparse)
        normalized = self.normalizer.transform(dense)
        return np.asarray(normalized[0], dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name


def get_embedder(model_name: str = PIPELINE_CONFIG.EMBEDDING_MODEL_NAME) -> BaseEmbedder:
    """Factory creating local embedder with automatic graceful fallback."""
    try:
        embedder = SentenceTransformerEmbedder(model_name=model_name)
        logger.info(f"Loaded primary embedding model: {embedder.model_name}")
        return embedder
    except Exception as e:
        logger.warning(
            f"SentenceTransformerEmbedder failed to initialize ({e}). "
            "Falling back to local Latent Semantic Analysis (LSA: TF-IDF + TruncatedSVD)."
        )
        return LSAEmbedder(n_components=128, seed=PIPELINE_CONFIG.SEED)
