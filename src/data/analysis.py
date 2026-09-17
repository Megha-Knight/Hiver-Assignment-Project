"""Exploratory conversation analysis and visualization module for AmazonHelp.

Computes distributions, escalation signals, repeated contacts, DM handoffs,
and generates clean, publication-ready matplotlib visualizations.
"""

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from src.config import PATHS
from src.data.preprocess import (
    ProcessedConversation,
    RE_ESCALATION_PHRASES,
    RE_HANDOFF_DM,
    RE_REPEATED_CONTACT_SIGNALS,
    RE_RESOLVED_SIGNALS,
    RE_UNRESOLVED_SIGNALS,
    clean_text_for_langdetect,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_alternations(turns: List[Dict[str, Any]]) -> int:
    """Computes the number of customer <-> support speaker role switches."""
    if len(turns) <= 1:
        return 0
    alternations = 0
    for i in range(len(turns) - 1):
        if turns[i]["role"] != turns[i + 1]["role"]:
            alternations += 1
    return alternations


def extract_top_ngrams(texts: List[str], n: int = 2, top_k: int = 20) -> List[Tuple[str, int]]:
    """Extracts frequent n-grams from a collection of customer opening texts."""
    stop_words = {
        "i", "me", "my", "myself", "we", "our", "ours", "you", "your", "yours",
        "he", "him", "his", "she", "her", "it", "its", "they", "them", "their",
        "what", "which", "who", "whom", "this", "that", "these", "those", "am",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but",
        "if", "or", "because", "as", "until", "while", "of", "at", "by", "for",
        "with", "about", "against", "between", "into", "through", "during",
        "before", "after", "above", "below", "to", "from", "up", "down", "in",
        "out", "on", "off", "over", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "any", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
        "can", "will", "just", "don", "should", "now", "amazonhelp", "amazon",
        "hi", "hello", "please", "help", "thanks", "thank",
    }
    counts = Counter()
    for text in texts:
        cleaned = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
        tokens = [w for w in cleaned.split() if len(w) > 2 and w not in stop_words]
        for i in range(len(tokens) - n + 1):
            gram = " ".join(tokens[i : i + n])
            counts[gram] += 1
    return counts.most_common(top_k)


def run_conversation_analysis(
    conversations: List[ProcessedConversation],
    output_dir: Path = PATHS.ANALYSIS_DIR,
) -> Dict[str, Any]:
    """Analyzes conversation metrics and generates matplotlib plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Running conversation analysis on {len(conversations):,} conversations...")

    total_convs = len(conversations)
    if total_convs == 0:
        logger.warning("No conversations provided to analysis.")
        return {}

    # Metric collections
    conv_lengths = []
    customer_msg_lengths = []
    support_msg_lengths = []
    alternation_counts = []
    resolution_counts = Counter()
    handoff_count = 0
    escalation_count = 0
    repeated_contact_count = 0
    customer_opening_texts = []

    for conv in conversations:
        conv_lengths.append(len(conv.normalized_turns))
        alternation_counts.append(compute_alternations(conv.normalized_turns))
        resolution_counts[conv.resolution_status] += 1

        has_dm = False
        has_escalation = False
        has_repeated = False
        first_customer_msg = None

        for t in conv.normalized_turns:
            words = len(t["text"].split())
            if t["role"] == "customer":
                customer_msg_lengths.append(words)
                if first_customer_msg is None:
                    first_customer_msg = t["text"]
                if RE_ESCALATION_PHRASES.search(t["text"]):
                    has_escalation = True
                if RE_REPEATED_CONTACT_SIGNALS.search(t["text"]):
                    has_repeated = True
            else:
                support_msg_lengths.append(words)
                if RE_HANDOFF_DM.search(t["text"]):
                    has_dm = True

        if has_dm:
            handoff_count += 1
        if has_escalation:
            escalation_count += 1
        if has_repeated:
            repeated_contact_count += 1
        if first_customer_msg:
            customer_opening_texts.append(first_customer_msg)

    # Top ngrams for opening customer complaints
    top_bigrams = extract_top_ngrams(customer_opening_texts, n=2, top_k=15)
    top_trigrams = extract_top_ngrams(customer_opening_texts, n=3, top_k=15)

    stats = {
        "total_analyzed_conversations": total_convs,
        "conversation_length": {
            "mean": float(np.mean(conv_lengths)),
            "median": float(np.median(conv_lengths)),
            "p90": float(np.percentile(conv_lengths, 90)),
            "max": int(np.max(conv_lengths)),
        },
        "customer_message_words": {
            "mean": float(np.mean(customer_msg_lengths)) if customer_msg_lengths else 0.0,
            "median": float(np.median(customer_msg_lengths)) if customer_msg_lengths else 0.0,
            "p90": float(np.percentile(customer_msg_lengths, 90)) if customer_msg_lengths else 0.0,
        },
        "support_message_words": {
            "mean": float(np.mean(support_msg_lengths)) if support_msg_lengths else 0.0,
            "median": float(np.median(support_msg_lengths)) if support_msg_lengths else 0.0,
            "p90": float(np.percentile(support_msg_lengths, 90)) if support_msg_lengths else 0.0,
        },
        "alternations": {
            "mean": float(np.mean(alternation_counts)),
            "median": float(np.median(alternation_counts)),
            "max": int(np.max(alternation_counts)),
        },
        "dm_handoff_rate_pct": round((handoff_count / total_convs) * 100, 2),
        "escalation_phrase_rate_pct": round((escalation_count / total_convs) * 100, 2),
        "repeated_contact_rate_pct": round((repeated_contact_count / total_convs) * 100, 2),
        "resolution_breakdown": {k: v for k, v in resolution_counts.items()},
        "resolution_percentages": {
            k: round((v / total_convs) * 100, 2) for k, v in resolution_counts.items()
        },
        "common_opening_bigrams": top_bigrams,
        "common_opening_trigrams": top_trigrams,
    }

    # -------------------------------------------------------------------------
    # Generate Matplotlib Visualizations
    # -------------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot 1: Conversation Length Distribution
    plt.figure(figsize=(8, 5))
    bins = np.arange(1, min(15, max(conv_lengths) + 2)) - 0.5
    counts, edges, patches = plt.hist(
        conv_lengths,
        bins=bins,
        color="#2563eb",
        edgecolor="white",
        rwidth=0.85,
    )
    plt.title("AmazonHelp: Conversation Length (Normalized Turns)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Turn Count (Customer + Support)", fontsize=11)
    plt.ylabel("Number of Conversations", fontsize=11)
    plt.xticks(range(1, min(15, max(conv_lengths) + 1)))
    plt.tight_layout()
    plt.savefig(output_dir / "conv_length_distribution.png", dpi=200)
    plt.close()

    # Plot 2: Message Length (Word Count) Comparison
    plt.figure(figsize=(8, 5))
    plt.hist(
        customer_msg_lengths,
        bins=range(0, 70, 2),
        alpha=0.65,
        color="#0284c7",
        label="Customer Turns",
        density=True,
    )
    plt.hist(
        support_msg_lengths,
        bins=range(0, 70, 2),
        alpha=0.65,
        color="#f97316",
        label="Support Turns",
        density=True,
    )
    plt.title("Turn Word Count Distribution: Customer vs. Support", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Word Count", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / "message_length_distribution.png", dpi=200)
    plt.close()

    # Plot 3: Turn Alternations
    plt.figure(figsize=(8, 5))
    alt_bins = np.arange(0, min(12, max(alternation_counts) + 2)) - 0.5
    plt.hist(
        alternation_counts,
        bins=alt_bins,
        color="#10b981",
        edgecolor="white",
        rwidth=0.85,
    )
    plt.title("Speaker Role Alternations per Conversation", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Number of Speaker Switches", fontsize=11)
    plt.ylabel("Number of Conversations", fontsize=11)
    plt.xticks(range(0, min(12, max(alternation_counts) + 1)))
    plt.tight_layout()
    plt.savefig(output_dir / "turn_alternation_distribution.png", dpi=200)
    plt.close()

    # Plot 4: Resolution Status Distribution
    plt.figure(figsize=(9, 5))
    statuses = list(resolution_counts.keys())
    values = [resolution_counts[s] for s in statuses]
    colors = ["#3b82f6", "#10b981", "#ef4444", "#f59e0b", "#6b7280"][:len(statuses)]
    bars = plt.bar(statuses, values, color=colors, edgecolor="black", linewidth=0.5, width=0.6)
    for bar in bars:
        height = bar.get_height()
        pct = (height / total_convs) * 100
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(values) * 0.015,
            f"{pct:.1f}%\n({height:,})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.title("Conservative Resolution Status Distribution", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Conversation Count", fontsize=11)
    plt.ylim(0, max(values) * 1.18)
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "resolution_status_distribution.png", dpi=200)
    plt.close()

    # Plot 5: Escalation & Handoff Signals
    plt.figure(figsize=(8, 5))
    signal_names = ["DM / External Handoff", "Customer Frustration / Escalation", "Repeated Contact Signals"]
    signal_pcts = [
        stats["dm_handoff_rate_pct"],
        stats["escalation_phrase_rate_pct"],
        stats["repeated_contact_rate_pct"],
    ]
    bar_colors = ["#8b5cf6", "#f43f5e", "#ea580c"]
    sig_bars = plt.bar(signal_names, signal_pcts, color=bar_colors, edgecolor="black", linewidth=0.5, width=0.55)
    for bar in sig_bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.8,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    plt.title("Frequencies of Critical Dialogue Signals", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Percentage of Conversations (%)", fontsize=11)
    plt.ylim(0, max(signal_pcts) * 1.25)
    plt.xticks(rotation=10, ha="right", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "escalation_and_handoff_frequencies.png", dpi=200)
    plt.close()

    logger.info(f"Generated 5 analysis plots in {output_dir}")
    return stats
