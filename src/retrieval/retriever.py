"""Retrieval query engine for historical AmazonHelp conversations.

Implements:
1. Dense vector similarity search using normalized embeddings.
2. Metadata filtering by intent or outcome evidence type.
3. Formatted exemplar return for grounded agent reasoning.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS
from src.retrieval.embeddings import BaseEmbedder, get_embedder
from src.utils.logger import get_logger

logger = get_logger("retriever")


@dataclass
class RetrievalResult:
    """A single retrieved historical conversation match with provenance and evidence."""
    similarity_score: float
    retrieval_id: str
    conversation_id: str
    derived_intent: str
    resolution_status: str
    outcome_evidence_type: str
    customer_problem_summary: str
    support_response_evidence: str
    conversation_text: str
    source_tweet_ids: List[int]
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HistoricalRetriever:
    """Dense vector retriever over the Train-only historical conversation index."""

    def __init__(
        self,
        index_npz_path: Path = PATHS.RETRIEVAL_INDEX_NPZ,
        meta_json_path: Path = PATHS.RETRIEVAL_INDEX_META,
        embedder: Optional[BaseEmbedder] = None,
    ):
        self.index_npz_path = index_npz_path
        self.meta_json_path = meta_json_path
        self.embedder = embedder or get_embedder()

        self.vectors: Optional[np.ndarray] = None
        self.documents: List[Dict[str, Any]] = []
        self._load_index()

    def _load_index(self):
        """Loads vector matrix and document metadata into memory."""
        if not self.index_npz_path.exists() or not self.meta_json_path.exists():
            logger.warning(f"Index files not found at {self.index_npz_path}. Index must be built first.")
            return

        logger.info(f"Loading retrieval index from {self.index_npz_path}...")
        data = np.load(self.index_npz_path)
        self.vectors = data["vectors"]

        with open(self.meta_json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            self.documents = meta["documents"]

        logger.info(f"Loaded {len(self.documents):,} historical retrieval documents ({self.vectors.shape}).")

    def query(
        self,
        customer_message: str,
        conversation_context: Optional[str] = None,
        intent_filter: Optional[str] = None,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Queries historical index for top-k semantically relevant conversations."""
        if self.vectors is None or len(self.documents) == 0:
            raise RuntimeError("Retriever index is empty or not loaded.")

        # Build combined query text
        query_text = customer_message.strip()
        if conversation_context:
            query_text = f"{conversation_context.strip()} Customer: {query_text}"

        query_vec = self.embedder.embed_query(query_text)

        # Dot product for cosine similarity (vectors are L2 normalized)
        scores = np.dot(self.vectors, query_vec)

        # Apply intent filter if specified
        if intent_filter:
            mask = np.array([doc.get("derived_intent") == intent_filter for doc in self.documents])
            scores[~mask] = -1.0

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < -0.5:
                continue
            doc = self.documents[idx]
            results.append(
                RetrievalResult(
                    similarity_score=round(score, 4),
                    retrieval_id=doc["retrieval_id"],
                    conversation_id=doc["conversation_id"],
                    derived_intent=doc.get("derived_intent", "UNKNOWN"),
                    resolution_status=doc.get("resolution_status", "UNKNOWN"),
                    outcome_evidence_type=doc.get("outcome_evidence_type", "UNKNOWN"),
                    customer_problem_summary=doc.get("customer_problem_summary", ""),
                    support_response_evidence=doc.get("support_response_evidence", ""),
                    conversation_text=doc.get("conversation_text", ""),
                    source_tweet_ids=doc.get("source_tweet_ids", []),
                    timestamp=doc.get("timestamp", 0.0),
                )
            )

        return results
