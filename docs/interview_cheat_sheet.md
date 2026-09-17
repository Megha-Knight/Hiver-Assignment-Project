# Engineering Interview Cheat Sheet: 30 Technical Questions & Defensible Answers

**Project**: AmazonHelp Autonomous Customer Support AI Agent  
**Role**: Senior Engineering Candidate / Author

---

### 1. Why this dataset?
The Kaggle *Customer Support on Twitter* dataset contains 2.8 million real-world customer support interactions across 108 brands, capturing genuine customer colloquialisms, frustrations, multi-turn dynamics, and real agent responses rather than clean synthetic benchmarks.

### 2. Why AmazonHelp?
AmazonHelp is the single largest e-commerce brand in the corpus, providing 85,087 reconstructed conversations. It offers dense multi-turn dialogues, clear transactional states (inbound, awaiting info, escalated, resolved), and high operational relevance for customer support engineering.

### 3. How did you reconstruct conversations?
Raw tweets are linked only via `in_response_to_tweet_id`. We implemented a breadth-first tree traversal algorithm that reconstructed root-to-leaf paths, filtering out orphaned tweets, loops, and bot spam to create clean, chronologically ordered multi-turn dialogue trees.

### 4. How did you prevent data leakage?
We enforced three strict firewalls: (1) Chronological train/dev/test partitioning (earliest 70% Train, middle 15% Dev, latest 15% Test); (2) Train-only retrieval indexing (5,502 dialogues strictly from Train); and (3) Complete isolation of the 5,365-dialogue Test partition.

### 5. Why a chronological split instead of random?
Customer support dialogues exhibit temporal drift—seasonal shopping surges, policy shifts, and evolving product lines. Random splits leak future phrasing and agent templates into training, causing unrealistically optimistic evaluation metrics.

### 6. How did you define the intent taxonomy?
We performed empirical topic discovery and clustered frequent customer query patterns, balancing operational actionability with machine learnability.

### 7. Why 10 intents?
10 intents (`DELIVERY_STATUS_AND_TRACKING`, `REFUND_REQUEST`, `RETURN_INQUIRY`, `ACCOUNT_ACCESS_ISSUE`, `PAYMENT_ISSUE`, `ORDER_CANCELLATION`, `PRODUCT_INQUIRY`, `SUBSCRIPTION_INQUIRY`, `DAMAGED_DEFECTIVE_ITEM`, `FEEDBACK_COMPLAINT`) cover >92% of support volume without diluting classifier precision into long-tail noise.

### 8. Why a TF-IDF + Logistic Regression baseline?
Before evaluating complex neural or generative models, a simple, interpretable, sub-millisecond baseline is mandatory to quantify the actual value added by LLMs and dense retrieval. It achieved 87.50% Intent Accuracy.

### 9. Why use dense retrieval?
Customer support agents rely heavily on standardized historical resolutions and approved brand phrasing. Dense retrieval embeds customer queries into a continuous semantic space, surfacing contextually relevant exemplars even when vocabulary varies.

### 10. Why a local LLM instead of cloud APIs?
Data privacy, latency control, zero cloud API fees, and strict reproducibility. Enterprise support contracts frequently prohibit streaming customer communications to external multi-tenant cloud APIs.

### 11. Why `llama3.2:1b`?
`llama3.2:1b` is a modern, instruction-tuned open-weight model with a compact memory footprint (~1.3 GB) that runs efficiently on commodity CPU hardware without requiring dedicated GPUs.

### 12. Why not a larger model like 8B or 70B?
An 8B model requires 8–16 GB of VRAM or introduces 10–20 second CPU latencies per turn. In an autonomous support SLA requiring sub-2-second turns on commodity infrastructure, 1B is the pragmatic engineering sweet spot.

### 13. Why a hybrid architecture instead of LLM-only?
End-to-end LLM prompting (Phase 6B) collapsed to 60.00% intent accuracy and 19.50% exact match due to hallucinated states and unconstrained decision spaces. Hybrid tri-layer architecture anchors generation with deterministic signals and retrieval context.

### 14. Why $K=5$ for retrieval?
$K=5$ provides sufficient exemplar diversity across varied customer situations without saturating the prompt token budget or confusing the 1B model with conflicting historical precedents.

### 15. Why a deterministic safety layer?
Neural self-moderation is probabilistic and vulnerable to jailbreaks. Deterministic regex and heuristic filters guarantee that credential solicitation (passwords, CVVs, OTPs) and unauthorized commitments are hard-blocked 100% of the time.

### 16. How is escalation decided?
Escalation uses a dual-trigger mechanism: (1) Deterministic policy triggers (security threats, repeated failed turns, severe legal/financial complaints) force immediate escalation; (2) Model-predicted escalation flags are validated against support action capabilities.

### 17. What does FAHR mean?
**False Auto-Handle Rate (FAHR)** measures the proportion of turns requiring human escalation where the system mistakenly attempted autonomous handling. In Phase 6C, FAHR was 66.67% (30 missed escalations out of 45 true escalation checkpoints).

### 18. Why is Exact Match only 38.50%?
Exact Match is a strict joint metric: **Intent + State + Action + Escalation must all match simultaneously**. While individual Intent Accuracy is 86.00% and State Accuracy is 77.50%, compound probability across all four dimensions lowers exact match to 38.50%.

### 19. Why did Action Accuracy drop from baseline (88.5%) to hybrid (67.5%)?
The non-LLM baseline used deterministic rules mapping intents directly to actions. The hybrid LLM agent allows generative flexibility, which occasionally chose plausible but non-standard actions (e.g. `PROVIDE_INSTRUCTIONS` instead of `REQUEST_SAFE_DETAILS`).

### 20. Why is State Macro-F1 low (40.04%)?
Severe class imbalance across the 8 dialogue states. The majority of turns reside in `STATE_INITIAL_INBOUND` and `STATE_AWAITING_INFO`, while rare intermediate states (`STATE_DISPUTE_RAISED`, `STATE_AGENT_ESCALATED`) have fewer examples, depressing unweighted macro-F1.

### 21. What does 99.50% grounding actually mean?
It is an **automated evidence-alignment evaluator result** confirming that 199/200 generated responses contained zero unsupported policy commitments, fabricated order tracking IDs, or ungrounded promises. It does NOT prove independent human factual verification.

### 22. Why was the heuristic audit score (3.93/5) lower than the same-family LLM judge score (4.29/5)?
The same-family local LLM judge suffers from leniency bias (+0.36 points) and mode-collapse at integer 4.0. The rule-based heuristic audit penalized premature canned closures and generic troubleshooting on physical defects, which the automated judge overlooked. Genuine manual human evaluation remains pending manual review.

### 23. What did the response-quality review packet reveal?
The stratified $N=40$ review packet revealed an 83.8% adjacent agreement ($\pm 1$) between the heuristic audit and LLM judge, confirmed 0 credential solicitations, identified `Helpfulness` as the weakest dimension (3.15/5.0), and showed that action mismatches degrade customer experience far more than minor dialogue state discrepancies.

### 24. What is the biggest failure mode?
`WRONG_ACTION` (65/200 cases), where the agent misjudges the appropriate operational protocol (e.g., offering a general FAQ link when immediate account escalation is required).

### 25. What would you improve in a one-week sprint?
1. Constrain action selection using deterministic decision trees conditioned on intent and turn depth.
2. Implement multi-intent query decomposition for compound customer complaints.
3. Calibrate escalation thresholds to reduce the 66.67% FAHR.

### 26. How would this system scale in production?
Decouple components into asynchronous microservices: (1) Inbound Gateway & Conversation State Store (Redis); (2) Classifier & Retrieval Service (FAISS / Milvus); (3) GPU-backed vLLM / Triton inference cluster; (4) Policy & Safety Gateway; and (5) Human-in-the-loop escalation queue.

### 27. How would you deploy it?
Deploy containerized services on Kubernetes with horizontal pod autoscaling (HPA) driven by turn request rate. The deterministic safety layer acts as an inline proxy before message egress.

### 28. What observability metrics would you monitor?
P95/P99 latency, intent confidence distribution, escalation rate, human override rate, CSAT, customer reopen rate, and automated safety/credential trigger counters.

### 29. What would you test in production before 100% rollout?
A canary shadow-deployment: Route 10% of live traffic through the agent in shadow mode, comparing its decisions and responses against human agents without sending messages to customers.

### 30. What did you personally learn from this project?
1. Single headline metrics (like 86% accuracy) are dangerously misleading without strict joint metrics like Exact Match.
2. Same-family LLM judges have inherent leniency biases that require human calibration.
3. In mission-critical enterprise systems, deterministic safety rules must always supersede generative models.
