"""Intent discovery module using semantic embeddings and clustering.

Implements:
1. Stratified sampling of opening customer turns
2. Dense semantic representation using SentenceTransformers
3. MiniBatchKMeans unsupervised clustering for theme discovery
4. Cluster centroid extraction and closest exemplar identification
5. Automatic generation of docs/intent_discovery_examples.md
"""

from collections import Counter
import json
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import PATHS, PIPELINE_CONFIG
from src.data.preprocess import ProcessedConversation, clean_text_for_langdetect
from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_customer_openings(
    conversations: List[ProcessedConversation],
) -> List[Dict[str, Any]]:
    """Extracts customer opening statements and conversation metadata for clustering."""
    records = []
    for conv in conversations:
        # Find first customer turn
        first_cust = None
        for t in conv.normalized_turns:
            if t["role"] == "customer":
                first_cust = t
                break
        if not first_cust:
            continue

        raw_text = first_cust["text"]
        cleaned = clean_text_for_langdetect(raw_text)
        if len(cleaned.split()) < 3:
            continue

        records.append({
            "conversation_id": conv.conversation_id,
            "tweet_id": first_cust["tweet_ids"][0],
            "raw_text": raw_text,
            "cleaned_text": cleaned,
            "turn_count": len(conv.normalized_turns),
            "resolution_status": conv.resolution_status,
            "full_conv": conv,
        })
    return records


def stratified_sample_records(
    records: List[Dict[str, Any]],
    sample_size: int = 10_000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Performs stratified sampling across resolution status and turn depth."""
    if len(records) <= sample_size:
        return records

    random.seed(seed)
    # Group by strata (resolution_status + is_multiturn)
    strata = {}
    for r in records:
        key = (r["resolution_status"], r["turn_count"] >= 3)
        strata.setdefault(key, []).append(r)

    sampled = []
    total_records = len(records)
    for key, group in strata.items():
        stratum_sample_size = max(1, int(round((len(group) / total_records) * sample_size)))
        sampled.extend(random.sample(group, min(len(group), stratum_sample_size)))

    random.shuffle(sampled)
    return sampled[:sample_size]


def run_intent_clustering(
    conversations: List[ProcessedConversation],
    k_clusters: int = PIPELINE_CONFIG.INTENT_CLUSTERS_K,
    sample_size: int = PIPELINE_CONFIG.INTENT_SAMPLE_SIZE,
    output_markdown_path: Path = PATHS.INTENT_DISCOVERY_EXAMPLES_MD,
) -> List[Dict[str, Any]]:
    """Executes embedding clustering on sampled customer turns and outputs discovery artifact."""
    records = extract_customer_openings(conversations)
    logger.info(f"Extracted {len(records):,} valid customer opening messages.")

    sampled_records = stratified_sample_records(records, sample_size=sample_size, seed=PIPELINE_CONFIG.SEED)
    logger.info(f"Selected {len(sampled_records):,} stratified customer turns for clustering.")

    # Load embedding model
    texts_to_embed = [r["cleaned_text"] for r in sampled_records]
    embeddings = None
    embedding_source = PIPELINE_CONFIG.EMBEDDING_MODEL_NAME

    try:
        import torch
        from sentence_transformers import SentenceTransformer
        logger.info(f"Encoding customer texts using SentenceTransformer ({PIPELINE_CONFIG.EMBEDDING_MODEL_NAME})...")
        model = SentenceTransformer(PIPELINE_CONFIG.EMBEDDING_MODEL_NAME)
        embeddings = model.encode(texts_to_embed, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    except Exception as e:
        logger.warning(f"SentenceTransformer embedding failed ({e}). Falling back to Latent Semantic Analysis (LSA: TF-IDF + TruncatedSVD) embeddings.")
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import Normalizer
        lsa_tfidf = TfidfVectorizer(max_features=5000, sublinear_tf=True, ngram_range=(1, 2), stop_words="english")
        tfidf_sparse = lsa_tfidf.fit_transform(texts_to_embed)
        n_comp = min(128, tfidf_sparse.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=PIPELINE_CONFIG.SEED)
        dense_vecs = svd.fit_transform(tfidf_sparse)
        embeddings = Normalizer(copy=False).fit_transform(dense_vecs)
        embedding_source = f"LSA-TruncatedSVD-{n_comp}d"

    # Perform clustering
    logger.info(f"Fitting MiniBatchKMeans with k={k_clusters} on {embedding_source} embeddings...")

    kmeans = MiniBatchKMeans(
        n_clusters=k_clusters,
        random_state=PIPELINE_CONFIG.SEED,
        batch_size=256,
        n_init=5,
    )
    cluster_labels = kmeans.fit_predict(embeddings)

    # TF-IDF to find distinctive terms per cluster
    tfidf = TfidfVectorizer(
        max_features=2000,
        stop_words="english",
        ngram_range=(1, 2),
    )
    cluster_docs = [""] * k_clusters
    for idx, label in enumerate(cluster_labels):
        cluster_docs[label] += " " + texts_to_embed[idx]
    
    tfidf_matrix = tfidf.fit_transform(cluster_docs)
    feature_names = tfidf.get_feature_names_out()

    cluster_summaries = []
    for cluster_id in range(k_clusters):
        member_indices = np.where(cluster_labels == cluster_id)[0]
        if len(member_indices) == 0:
            continue

        # Top TF-IDF keywords
        row = tfidf_matrix.getrow(cluster_id).toarray()[0]
        top_indices = np.argsort(row)[::-1][:8]
        top_keywords = [feature_names[i] for i in top_indices if row[i] > 0]

        # Calculate distances to centroid
        centroid = kmeans.cluster_centers_[cluster_id]
        cluster_vecs = embeddings[member_indices]
        dists = np.linalg.norm(cluster_vecs - centroid, axis=1)

        # Select top 5 closest exemplar turns
        closest_order = np.argsort(dists)[:5]
        exemplar_records = [sampled_records[member_indices[i]] for i in closest_order]

        # Suggest theme based on top keywords
        kw_str = " ".join(top_keywords).lower()
        if any(w in kw_str for w in ["delivery", "delivered", "package", "tracking", "courier", "delayed", "late", "arrive"]):
            theme = "Order Tracking & Delivery Delays"
        elif any(w in kw_str for w in ["refund", "return", "returned", "money", "replacement"]):
            theme = "Returns, Refunds & Item Replacements"
        elif any(w in kw_str for w in ["account", "password", "login", "otp", "locked", "email"]):
            theme = "Account Access & Login Security"
        elif any(w in kw_str for w in ["prime", "video", "membership", "subscription", "music"]):
            theme = "Prime Membership & Digital Streaming"
        elif any(w in kw_str for w in ["damaged", "broken", "wrong", "item", "empty", "box"]):
            theme = "Product Condition & Wrong/Damaged Item"
        elif any(w in kw_str for w in ["payment", "card", "charged", "bank", "declined", "gift"]):
            theme = "Payment, Billing & Gift Cards"
        elif any(w in kw_str for w in ["cancel", "order", "cancellation", "address"]):
            theme = "Order Cancellation & Address Modification"
        elif any(w in kw_str for w in ["kindle", "app", "fire", "device", "alexa", "echo"]):
            theme = "Digital Devices & App Troubleshooting"
        else:
            theme = f"Customer Inquiries ({', '.join(top_keywords[:3])})"

        cluster_summaries.append({
            "cluster_id": cluster_id,
            "size": int(len(member_indices)),
            "pct": round(len(member_indices) / len(sampled_records) * 100, 2),
            "top_keywords": top_keywords,
            "suggested_theme": theme,
            "exemplars": exemplar_records,
        })

    # Sort clusters by size descending
    cluster_summaries.sort(key=lambda x: x["size"], reverse=True)

    # -------------------------------------------------------------------------
    # Write docs/intent_discovery_examples.md
    # -------------------------------------------------------------------------
    output_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_markdown_path, "w", encoding="utf-8") as f:
        f.write("# Intent Discovery & Cluster Exploration Artifact\n\n")
        f.write("> **Unsupervised Semantic Clustering Analysis of Customer Initial Turns**  \n")
        f.write(f"> *Model: `{PIPELINE_CONFIG.EMBEDDING_MODEL_NAME}` | Clustered Sample Size: {len(sampled_records):,} | Clusters: {k_clusters}*\n\n")
        f.write("---\n\n")
        f.write("## 1. Executive Summary of Discovered Clusters\n\n")
        f.write("| Cluster ID | Suggested Theme | Sample Count | % Share | Distinguishing Keywords |\n")
        f.write("| :---: | :--- | :---: | :---: | :--- |\n")
        for c in cluster_summaries:
            kws = ", ".join(f"`{k}`" for k in c["top_keywords"][:5])
            f.write(f"| {c['cluster_id']} | **{c['suggested_theme']}** | {c['size']:,} | {c['pct']}% | {kws} |\n")
        f.write("\n---\n\n")

        f.write("## 2. Detailed Cluster Inspections & Real Multi-Turn Dialogue Exemplars\n\n")
        for c in cluster_summaries:
            f.write(f"### Cluster {c['cluster_id']}: {c['suggested_theme']}\n\n")
            f.write(f"- **Discovered Size**: {c['size']:,} conversations ({c['pct']}% of analyzed sample)\n")
            f.write(f"- **Top TF-IDF Salient Terms**: {', '.join(f'`{k}`' for k in c['top_keywords'])}\n")
            f.write("- **Empirical Customer Messages (Centroid Exemplars)**:\n\n")
            for idx, ex in enumerate(c["exemplars"], 1):
                f.write(f"  {idx}. *\"{ex['raw_text'].strip()}\"* (Tweet ID: `{ex['tweet_id']}`)\n")
            f.write("\n- **Representative Multi-Turn Conversation Thread**:\n\n")
            # Pick first exemplar that is multi-turn
            chosen_conv = None
            for ex in c["exemplars"]:
                if ex["turn_count"] >= 2:
                    chosen_conv = ex["full_conv"]
                    break
            if not chosen_conv and c["exemplars"]:
                chosen_conv = c["exemplars"][0]["full_conv"]

            if chosen_conv:
                f.write(f"> **Conversation ID**: `{chosen_conv.conversation_id}`  \n")
                f.write(f"> **Resolution Status**: `{chosen_conv.resolution_status}` | **Turns**: {len(chosen_conv.normalized_turns)}\n>\n")
                for t in chosen_conv.normalized_turns:
                    role_badge = "**Customer**" if t["role"] == "customer" else "**AmazonHelp Support**"
                    text_indented = t["text"].replace("\n", "\n> ")
                    f.write(f"> - {role_badge}: {text_indented}\n")
                f.write("\n")
            f.write("---\n\n")

    logger.info(f"Saved intent discovery artifact to {output_markdown_path}")
    return cluster_summaries
