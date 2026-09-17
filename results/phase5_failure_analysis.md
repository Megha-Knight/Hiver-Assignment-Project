# Phase 5 Retrieval Failure Analysis & Edge Cases

> **Empirical Diagnostics: Why Dense Retrieval Fails and How Phase 6 Mitigates It**  
> *AmazonHelp Autonomous Support Agent Benchmark*

---

## 1. Overview

While historical retrieval augmentation substantially boosts escalation recall (+35.6%) and hard-case performance, dense vector retrieval alone exhibits specific structural failure modes.

This report investigates 5 verified empirical failure modes discovered during benchmarking on the 200 golden evaluation checkpoints.

---

## Failure Mode 1: Prime Keyword Semantic Dominance

- **Checkpoint ID**: `chk_AmazonHelp_437905_turn1`
- **Customer Message**: *"What is the point of having a #prime membership , if you haven't received your product even after 3 days from ordering it ? 
#Amazon #AmazonPrime @115850"*
- **Max Cosine Similarity**: `0.7338`
- **Ground Truth Target**: `DELIVERY_STATUS_AND_TRACKING`
- **Retrieved Precedent**: *"Ordered a @132994 5t on 21st Nov and selected one day delivery on @115850.. The product is still not delivered.. What's ..."*

### Root Cause & Failure Mechanism:
The token 'Prime' pulled dense embedding vectors toward Prime subscription/membership dialogues, despite the customer's actual grievance being a delayed package.

### Hypothesis:
Dense bi-encoders without token-level cross-attention over-index on brand product tokens (e.g. 'Prime') over relational verbs (e.g. 'didn't receive').

### Proposed Phase 6 LLM Mitigation:
> **Mitigation Strategy**: In Phase 6, prompt the local LLM with explicit intent boundary instructions to disentangle subscription fee issues from physical delivery delays.

---

## Failure Mode 2: Severe Customer Frustration & Sarcastic Escalation

- **Checkpoint ID**: `chk_AmazonHelp_2894680_turn1`
- **Customer Message**: *"@115821 @115830 is a joke! I have cancelled an order (that I diddnt even buy) I was assured a refund, I haven’t got my refund or the delivery that cost 40 pound are you having a laugh! You’ve literally just taken 40 pound out my wallet and given nothing back! Useless"*
- **Max Cosine Similarity**: `0.5982`
- **Ground Truth Target**: `PAYMENT_ACCOUNT_DISPUTE`
- **Retrieved Precedent**: *"Just wondering when @115830 will do my 44 euros refund...is it a bad joke? Never got the product, never got the refund...."*

### Root Cause & Failure Mechanism:
Customer expresses severe frustration or sarcasm. The dense embedder matched polite standard routine dialogues with low escalation weight rather than detecting the emotional intensity.

### Hypothesis:
Dense bi-encoders focus on topical nouns and verbs rather than affective tone or subtle emotional exasperation.

### Proposed Phase 6 LLM Mitigation:
> **Mitigation Strategy**: Ollama LLM in Phase 6 will analyze pragmatic tone and frustration markers rather than relying solely on cosine similarity over topical keywords.

---

## Failure Mode 3: Multi-Issue Composite Grievance

- **Checkpoint ID**: `chk_AmazonHelp_2899622_turn2`
- **Customer Message**: *"@115850 @AmazonHelp @4030 They can,'t assist.. keep the amount. Let me aware the people or tell them about your fraud policy. You are just cheating the people and use their money. 9929029996 This is my contact no. Let me assist if u can."*
- **Max Cosine Similarity**: `0.7754`
- **Ground Truth Target**: `POLICY_AND_GENERAL_INQUIRIES`
- **Retrieved Precedent**: *"@115850 fraud site as neither given my product nor refund my money since 4 Oct and showing just on the way https://t.co/..."*

### Root Cause & Failure Mechanism:
Customer mentions both missing delivery and missing refund simultaneously. Retrieval matched the secondary delivery aspect rather than the primary refund grievance.

### Hypothesis:
Cosine similarity over pooled sentence embeddings computes a centroid representation that dilutes the primary hierarchical clause.

### Proposed Phase 6 LLM Mitigation:
> **Mitigation Strategy**: Phase 6 prompt will enforce hierarchical precedence rules (e.g. Remedy overrides transit tracking) on top of retrieved exemplars.

---

## Failure Mode 4: Low-Similarity / Vocabulary Outlier Query

- **Checkpoint ID**: `chk_AmazonHelp_2830662_turn1`
- **Customer Message**: *"So @115821 has that “buy now” button and Ruy freaking “accidentally” bought some stupid shit for his truck and it charged to my card!!😂😡😭 https://t.co/Ie0fh53Jsr"*
- **Max Cosine Similarity**: `0.4964`
- **Ground Truth Target**: `PAYMENT_BILLING_AND_PROMOTIONS`
- **Retrieved Precedent**: *"@115850 Please help me out. My Rupay Card shows during ordering but not in the *Manage Payment* section. Bug? https://t...."*

### Root Cause & Failure Mechanism:
Top-1 similarity was only 0.50 due to atypical customer phrasing or rare third-party integration mentions.

### Hypothesis:
When historical support conversations lack semantically identical phrasing, dense similarity degrades to generic nearest neighbors.

### Proposed Phase 6 LLM Mitigation:
> **Mitigation Strategy**: Gated fallback: when retrieval confidence is LOW/NONE, instruct the LLM to rely primarily on internal parametric reasoning rather than forcing ungrounded retrieved exemplars.

---

## Failure Mode 5: Generic Macro/Template Action Bias

- **Checkpoint ID**: `chk_AmazonHelp_2915900_turn1`
- **Customer Message**: *"@115830 hey I’ve got a problem with one of my recent purchases, how do I get in contact about this?"*
- **Max Cosine Similarity**: `0.6661`
- **Ground Truth Target**: `PROVIDE_INFORMATION`
- **Retrieved Precedent**: *"@427428 Hi Tim- You can contact us directly via email, chat or phone via the below link: https://t.co/JzP7hlA23B ^NV
@42..."*

### Root Cause & Failure Mechanism:
The historical dataset contains a high proportion of support agents immediately posting DM routing links, biasing retrieval toward HANDOFF even when public policy information was requested.

### Hypothesis:
Real human support agents frequently default to standard boilerplate DM links for convenience, creating an empirical action bias in the historical corpus.

### Proposed Phase 6 LLM Mitigation:
> **Mitigation Strategy**: Maintain deterministic action guardrails outside the generative model to select PROVIDE_INFORMATION for explicit informational inquiries.

---

