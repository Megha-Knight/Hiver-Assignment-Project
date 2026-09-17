"""Dataset audit module for Twitter Customer Support dataset.

Streams large CSV files in chunks, extracts brand interaction graphs,
computes comparative conversation metrics, and detects template usage.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.config import AUDIT_CONFIG, PATHS
from src.data.reconstruct import ConversationReconstructor, ReconstructedConversation
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BrandMetrics:
    """Metrics calculated during dataset audit for a specific brand."""
    brand: str
    total_tweets: int
    customer_inbound_count: int
    support_response_count: int
    reconstructable_conversations: int
    avg_conversation_length: float
    median_conversation_length: float
    pct_conversations_ge_3_turns: float
    pct_conversations_ge_5_turns: float
    duplicate_template_rate: float
    missing_broken_link_rate: float

    def to_dict(self) -> dict:
        return {
            "Brand": self.brand,
            "Total Tweets": self.total_tweets,
            "Customer Inbound": self.customer_inbound_count,
            "Support Responses": self.support_response_count,
            "Reconstructable Conversations": self.reconstructable_conversations,
            "Avg Conversation Length": round(self.avg_conversation_length, 2),
            "Median Conversation Length": round(self.median_conversation_length, 1),
            "Conversations >= 3 Turns (%)": round(self.pct_conversations_ge_3_turns, 2),
            "Conversations >= 5 Turns (%)": round(self.pct_conversations_ge_5_turns, 2),
            "Template Response Rate (%)": round(self.duplicate_template_rate, 2),
            "Broken Link Rate (%)": round(self.missing_broken_link_rate, 2),
        }


def normalize_support_text(text: str) -> str:
    """Normalizes support response text to detect canned boilerplate templates.

    Strips URLs, user handles, punctuation, agent initials (^AB, /XY), and extra whitespace.
    """
    if not isinstance(text, str):
        return ""
    # Strip URLs
    text = re.sub(r"https?://\S+", "", text)
    # Strip user mention IDs like @105837 or handles
    text = re.sub(r"@\w+", "", text)
    # Strip agent initials (^AB, /XY, -AB)
    text = re.sub(r"[\^/|-][A-Za-z]{1,4}\b", "", text)
    # Strip non-alphanumeric
    text = re.sub(r"[^\w\s]", " ", text).lower()
    # Normalize whitespace
    return " ".join(text.split())


class DatasetAuditor:
    """Streaming dataset auditor for Twitter customer support data."""

    def __init__(
        self,
        csv_path: Optional[Path] = None,
        candidate_brands: Optional[List[str]] = None,
        chunk_size: int = 250_000,
    ):
        self.csv_path = csv_path or PATHS.get_raw_dataset_path()
        self.candidate_brands = candidate_brands or AUDIT_CONFIG.CANDIDATE_BRANDS
        self.chunk_size = chunk_size

    def audit_brands(self) -> Tuple[pd.DataFrame, Dict[str, List[ReconstructedConversation]]]:
        """Performs two-pass streaming audit across all candidate brands."""
        logger.info(f"Initiating dataset audit on: {self.csv_path}")
        logger.info(f"Target candidate brands: {self.candidate_brands}")

        # PASS 1: Identify brand tweets and referenced customer tweet IDs
        logger.info("Pass 1/2: Discovering brand tweets and referenced conversation IDs...")
        brand_tweet_ids: Dict[str, Set[int]] = {b: set() for b in self.candidate_brands}
        referenced_ids: Dict[str, Set[int]] = {b: set() for b in self.candidate_brands}
        
        target_brands_set = set(self.candidate_brands)

        pass1_cols = ["tweet_id", "author_id", "response_tweet_id", "in_response_to_tweet_id"]
        total_rows_scanned = 0

        for chunk in pd.read_csv(self.csv_path, usecols=pass1_cols, chunksize=self.chunk_size):
            total_rows_scanned += len(chunk)
            brand_rows = chunk[chunk["author_id"].isin(target_brands_set)]
            if len(brand_rows) == 0:
                continue

            for b in self.candidate_brands:
                sub = brand_rows[brand_rows["author_id"] == b]
                if len(sub) == 0:
                    continue
                brand_tweet_ids[b].update(sub["tweet_id"].astype("int64"))

                parents = sub["in_response_to_tweet_id"].dropna().astype("int64")
                referenced_ids[b].update(parents)

                for resps in sub["response_tweet_id"].dropna():
                    for r in str(resps).split(","):
                        r_clean = r.strip()
                        if r_clean.isdigit():
                            referenced_ids[b].add(int(r_clean))

        logger.info(f"Pass 1 finished. Scanned {total_rows_scanned:,} total rows.")
        for b in self.candidate_brands:
            logger.info(
                f"[{b}] Found {len(brand_tweet_ids[b]):,} brand tweets and {len(referenced_ids[b]):,} referenced IDs"
            )

        # Union of all relevant IDs per brand
        relevant_ids_per_brand: Dict[str, Set[int]] = {
            b: brand_tweet_ids[b] | referenced_ids[b] for b in self.candidate_brands
        }
        all_target_ids = set().union(*relevant_ids_per_brand.values())
        logger.info(f"Total target tweet IDs to extract in Pass 2: {len(all_target_ids):,}")

        # PASS 2: Extract tweets using fast itertuples
        logger.info("Pass 2/2: Extracting tweet content and building conversation graphs...")
        reconstructors: Dict[str, ConversationReconstructor] = {
            b: ConversationReconstructor(brand=b) for b in self.candidate_brands
        }
        support_template_counters: Dict[str, Counter] = {b: Counter() for b in self.candidate_brands}
        brand_tweet_counts: Dict[str, int] = {b: 0 for b in self.candidate_brands}
        customer_tweet_counts: Dict[str, int] = {b: 0 for b in self.candidate_brands}

        for chunk in pd.read_csv(self.csv_path, chunksize=self.chunk_size):
            chunk_filtered = chunk[chunk["tweet_id"].isin(all_target_ids)]
            if len(chunk_filtered) == 0:
                continue

            for row in chunk_filtered.itertuples(index=False):
                tid = int(row.tweet_id)
                author = str(row.author_id)
                inbound = bool(row.inbound)
                created_at = str(row.created_at)
                text = str(row.text) if pd.notna(row.text) else ""

                parent_val = row.in_response_to_tweet_id
                parent_id = int(parent_val) if pd.notna(parent_val) and str(parent_val).strip() != "" else None

                resps = []
                resp_val = row.response_tweet_id
                if pd.notna(resp_val) and str(resp_val).strip() != "":
                    for r in str(resp_val).split(","):
                        r_clean = r.strip()
                        if r_clean.isdigit():
                            resps.append(int(r_clean))

                # Route tweet to matching brand reconstructors
                for b in self.candidate_brands:
                    if tid in relevant_ids_per_brand[b]:
                        reconstructors[b].add_tweet(
                            tweet_id=tid,
                            author_id=author,
                            inbound=inbound,
                            created_at=created_at,
                            text=text,
                            parent_id=parent_id,
                            response_ids=resps,
                        )
                        if author == b and not inbound:
                            brand_tweet_counts[b] += 1
                            norm_text = normalize_support_text(text)
                            if norm_text:
                                support_template_counters[b][norm_text] += 1
                        else:
                            customer_tweet_counts[b] += 1

        logger.info("Pass 2 complete. Reconstructing conversations and computing metrics...")

        all_conversations: Dict[str, List[ReconstructedConversation]] = {}
        metrics_list: List[BrandMetrics] = []

        for b in self.candidate_brands:
            reconstructor = reconstructors[b]
            conversations = reconstructor.reconstruct_all()
            all_conversations[b] = conversations

            lengths = [c.turn_count for c in conversations] if conversations else [0]
            avg_len = float(np.mean(lengths)) if lengths else 0.0
            median_len = float(np.median(lengths)) if lengths else 0.0

            ge_3 = sum(1 for ln in lengths if ln >= 3)
            pct_ge_3 = (ge_3 / len(conversations) * 100.0) if conversations else 0.0

            ge_5 = sum(1 for ln in lengths if ln >= 5)
            pct_ge_5 = (ge_5 / len(conversations) * 100.0) if conversations else 0.0

            broken_count = sum(1 for c in conversations if c.is_broken)
            broken_rate = (broken_count / len(conversations) * 100.0) if conversations else 0.0

            total_supp = brand_tweet_counts[b]
            counter = support_template_counters[b]
            duplicate_supp_count = sum(cnt for templ, cnt in counter.items() if cnt > 1)
            dup_rate = (duplicate_supp_count / total_supp * 100.0) if total_supp > 0 else 0.0

            brand_metric = BrandMetrics(
                brand=b,
                total_tweets=brand_tweet_counts[b] + customer_tweet_counts[b],
                customer_inbound_count=customer_tweet_counts[b],
                support_response_count=brand_tweet_counts[b],
                reconstructable_conversations=len(conversations),
                avg_conversation_length=avg_len,
                median_conversation_length=median_len,
                pct_conversations_ge_3_turns=pct_ge_3,
                pct_conversations_ge_5_turns=pct_ge_5,
                duplicate_template_rate=dup_rate,
                missing_broken_link_rate=broken_rate,
            )
            metrics_list.append(brand_metric)
            logger.info(f"[{b}] Audit summary: {brand_metric.to_dict()}")

        comparison_df = pd.DataFrame([m.to_dict() for m in metrics_list])
        return comparison_df, all_conversations
