"""Retrieval package for historical conversation search and grounding."""
from .corpus_builder import (
    RetrievalDocument,
    build_train_retrieval_corpus,
    evaluate_retrieval_eligibility,
    generate_retrieval_report,
)
from .embeddings import BaseEmbedder, LSAEmbedder, SentenceTransformerEmbedder, get_embedder
from .retriever import HistoricalRetriever, RetrievalResult

__all__ = [
    "RetrievalDocument",
    "build_train_retrieval_corpus",
    "evaluate_retrieval_eligibility",
    "generate_retrieval_report",
    "BaseEmbedder",
    "LSAEmbedder",
    "SentenceTransformerEmbedder",
    "get_embedder",
    "HistoricalRetriever",
    "RetrievalResult",
]
