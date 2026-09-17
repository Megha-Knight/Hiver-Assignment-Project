# Historical Conversation Retrieval Corpus Specification

> **Phase 3 Retrieval Corpus Design, Document Schema, and Train-Only Filtering Policy**  
> *AmazonHelp Autonomous Multi-Turn Support Agent*

---

## 1. Executive Motivation & Grounding

In an autonomous multi-turn support agent, the historical retrieval corpus provides grounded few-shot exemplars, verified troubleshooting procedures, and standard policy answers.

### The Pitfalls of Naive Twitter Retrieval
1. **Unfiltered Drop-Offs**: Over 74% of Twitter support conversations end without explicit resolution because customers migrate to phone, email, or simply abandon the public thread. Indexing these dialogues would retrieve non-answers for future customers.
2. **Boilerplate Template Smothering**: 14.96% of human tweets contain generic redirects (`"Please connect with us here: https://amzn.to/..."`) that lack diagnostic problem-solving value.
3. **Data Leakage Catastrophe**: If conversations from the Validation or Test partitions are included in the retrieval corpus, the agent will have access to test dialogue histories during evaluation.

### Train-Only Isolation Guarantee
The retrieval corpus is built **strictly from the 42,909 Train conversations** (Dec 23, 2015 – Nov 24, 2017).
- Zero conversations from Validation (5,363 convs) or Test (5,365 convs) are permitted into the corpus.
- A runtime assertion fails loudly if any non-Train conversation ID is encountered.

---

## 2. Retrieval Document Schema

Each record in `data/retrieval/amazonhelp_train_retrieval.jsonl` is formatted as a structured retrieval document:

```json
{
  "retrieval_id": "ret_AmazonHelp_2565",
  "conversation_id": "conv_AmazonHelp_2565",
  "source_tweet_ids": [2565, 2566],
  "timestamp": 1509495123.0,
  "derived_intent": "DELIVERY_STATUS_AND_TRACKING",
  "resolution_status": "HANDOFF_OR_DM",
  "outcome_evidence_type": "OFFICIAL_HANDOFF",
  "turn_count": 2,
  "customer_problem_summary": "Where is my order? https://t.co/pXnKSCo2ex",
  "conversation_text": "Customer: Where is my order? https://t.co/pXnKSCo2ex\nAmazonHelp: I'm sorry it hasn't arrived! Please reach us here: https://amzn.to/help, so we can look into available options.",
  "support_response_evidence": "I'm sorry it hasn't arrived! Please reach us here: https://amzn.to/help, so we can look into available options.",
  "metadata": {
    "partition": "train",
    "is_broken": false,
    "has_cycle": false,
    "language": "en",
    "language_confidence": 1.0,
    "actionable_score": 0.85
  },
  "provenance": {
    "source_dataset": "twcs.csv",
    "brand": "AmazonHelp",
    "processed_by": "Phase3CorpusBuilder"
  }
}
```

### Key Field Semantics
- `retrieval_id`: Unique identifier formatted as `ret_AmazonHelp_<root_id>`.
- `derived_intent`: The primary functional intent of the dialogue.
- `customer_problem_summary`: Cleaned summary of the customer's initial problem description.
- `support_response_evidence`: The operative advice, instructions, or resolution pathway delivered by human support.
- `outcome_evidence_type`: Controlled categorization of the evidence utility:
  - `CONFIRMED_RESOLUTION`: Customer explicitly confirmed resolution ("thank you, that worked").
  - `OFFICIAL_HANDOFF`: Support routed customer to verified secure channel (`amzn.to/...` or DM).
  - `TROUBLESHOOTING_STEPS`: Support delivered concrete procedural steps or diagnostic instructions.
  - `POLICY_GUIDANCE`: Support provided factual store policy, timeframe, or fee explanation.

---

## 3. Strict Retrieval Eligibility Filtering Rules

To be accepted into `amazonhelp_train_retrieval.jsonl`, a conversation must satisfy **ALL** of the following requirements:

### Rule 1: Partition Verification
- `conv_id in train_partition_ids`. If not, immediate hard rejection.

### Rule 2: Structural Flawlessness
- `is_broken == False` (zero missing parent/child links).
- `has_cycle == False` (acyclic conversation graph).
- `validation['is_valid'] == True` (no duplicate tweet IDs, no missing text, chronological monotonicity).

### Rule 3: Meaningful Interaction Depth
- `len(normalized_turns) >= 2`.
- At least 1 customer turn AND at least 1 support turn.

### Rule 4: Actionable Support Evidence (Anti-Noise Filter)
Conversations are accepted **only** if they provide demonstrable, actionable support value:
1. **Always Accepted**: `resolution_status == 'APPARENTLY_RESOLVED'` (high-value positive resolution).
2. **Accepted with URL/DM Validation**: `resolution_status == 'HANDOFF_OR_DM'` where support message contains verified official help link or explicit DM routing.
3. **Selectively Accepted from `UNKNOWN`**: Allowed **only** if the support turn contains substantive troubleshooting or policy explanations (e.g. contains restart steps, app cache instructions, delivery timeframe facts, or links), marked explicitly with `outcome_evidence_type in ['TROUBLESHOOTING_STEPS', 'POLICY_GUIDANCE']`.
4. **Strictly Excluded**:
   - `resolution_status == 'UNRESOLVED'` (unresolved frustration rants).
   - `UNKNOWN` conversations where support merely asked a clarifying question and the customer never responded (incomplete dead-ends).
   - Boilerplate responses lacking specific troubleshooting guidance.
