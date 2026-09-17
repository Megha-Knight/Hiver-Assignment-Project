# Language Filtering & Verification Report

> **Dataset**: `AmazonHelp` Twitter Conversations  
> **Tool**: `langdetect` (Deterministic Seed: `42`)  
> **Configured English Confidence Threshold**: `0.8`

---

## 1. Summary of Language Filtering Results

| Metric | Conversation Count | Percentage (%) |
| :--- | :---: | :---: |
| **Total Reconstructed Conversations** | **85,087** | **100.0%** |
| **English Conversations (`is_english=True`)** | **63,493** | **74.62%** |
| **Non-English Conversations (`prob >= 0.80`)** | **14,540** | **17.09%** |
| **Uncertain Conversations (`prob < 0.80` / short / ambiguous)** | **7,054** | **8.29%** |

---

## 2. Confidence Score Distribution (All Conversations)

| Confidence Interval | Conversation Count | Share (%) | Description |
| :--- | :---: | :---: | :--- |
| **0.90 – 1.00** | 76,889 | 90.37% | High-certainty single-language prediction |
| **0.80 – 0.90** | 1,144 | 1.34% | Acceptable certainty above threshold |
| **0.70 – 0.80** | 525 | 0.62% | Sub-threshold; sent to uncertain bucket |
| **Below 0.70 / Failed** | 6,529 | 7.67% | Ultra-short queries, handles, or emojis |

---

## 3. Top Detected Non-English Languages

Amazon operates international storefronts (.de, .es, .fr, .in, .co.jp). When non-English inquiries reach `@AmazonHelp`, they are detected and partitioned cleanly:

| Detected ISO Code | Conversation Count | Storefront Domain Context |
| :---: | :---: | :--- |
| `es` | 4,418 | International customer query |
| `fr` | 3,363 | International customer query |
| `de` | 2,552 | International customer query |
| `pt` | 1,726 | International customer query |
| `ja` | 1,283 | International customer query |
| `it` | 829 | International customer query |
| `nl` | 42 | International customer query |
| `hi` | 37 | International customer query |

---

## 4. Destination File Mapping

- **English Benchmark Candidates**: `data/processed/amazonhelp_english.jsonl`
- **Non-English & Uncertain Holdout**: `data/processed/amazonhelp_non_english_or_uncertain.jsonl`

No data was silently discarded. All uncertain and non-English dialogues are preserved with their confidence metrics for future multilingual expansion.
