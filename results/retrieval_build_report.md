# Train-Only Retrieval Corpus Build Report

> **Dataset**: `AmazonHelp` Twitter Support Benchmark  
> **Source Partition**: Strict Train Split (Dec 23, 2015 – Nov 24, 2017)  
> **Corpus File**: `data/retrieval/amazonhelp_train_retrieval.jsonl`

---

## 1. Executive Summary & Filtration Performance

| Metric | Measured Value | Percentage (%) |
| :--- | :---: | :---: |
| **Total Train Pristine Conversations** | **42,909** | **100.0%** |
| **Accepted Retrieval Documents** | **5,502** | **12.82%** |
| **Excluded Non-Actionable Dialogues** | **37,407** | **87.18%** |
| **Average Turns per Retrieval Document** | **5.12 turns** | — |

---

## 2. Detailed Exclusion Breakdown

Unlike naive retrieval datasets, dead-end drop-offs and unresolved rants were eliminated to ensure high evidence quality:

| Exclusion Reason | Excluded Conversations | Share of Exclusions (%) | Description |
| :--- | :---: | :---: | :--- |
| `lacks_actionable_resolution_evidence` | 30,639 | 81.9% | Filtered out during quality audit |
| `unresolved_complaint_or_rant` | 6,768 | 18.1% | Filtered out during quality audit |

---

## 3. Actionable Outcome Evidence Breakdown

Every accepted document contains verified, grounded support evidence mapped to a controlled outcome category:

| Evidence Type | Document Count | Share of Corpus (%) | Retrieval Function |
| :--- | :---: | :---: | :--- |
| **`CONFIRMED_RESOLUTION`** | 3,806 | 69.2% | Historical resolution grounding |
| **`OFFICIAL_HANDOFF`** | 1,018 | 18.5% | Historical resolution grounding |
| **`POLICY_GUIDANCE`** | 461 | 8.4% | Historical resolution grounding |
| **`TROUBLESHOOTING_STEPS`** | 217 | 3.9% | Historical resolution grounding |

---

## 4. Leakage Verification Guarantee

- **Train Conversation IDs Only**: 100% of documents belong to the Train partition.
- **Dev/Validation Overlap**: **Strictly 0**.
- **Test Benchmark Overlap**: **Strictly 0**.
- **Audit Verification**: Asserted via automated unit test `scripts/verify_phase3_retrieval.py`.
