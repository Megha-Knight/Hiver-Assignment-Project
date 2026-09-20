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
TF-IDF + Logistic Regression is our only **trained ML classifier** in the pipeline. It underwent actual `.fit()` training on the 70% Train partition, learned specific sparse feature coefficients across word and character n-grams, and was serialized as a saved model artifact (`models/baselines/tfidf_logreg_intent.joblib`). A fast, interpretable, sub-millisecond baseline is mandatory to quantify the actual value added by generative prompting and dense retrieval. It achieved 87.50% Intent Accuracy.

### 9. Why use dense retrieval and what model is used?
Customer support agents rely heavily on standardized historical resolutions and approved brand phrasing. We utilize `all-MiniLM-L6-v2` as a **pretrained embedding model** (producing 384-dimensional dense vectors) to surface contextually relevant exemplars from the 5,502 Train-only dialogues. We apply **L2 embedding normalization** on all vectors so cosine similarity reduces to efficient inner products. Crucially, `all-MiniLM-L6-v2` was **not fine-tuned** by this project; it is used purely out-of-the-box for inference and vector similarity.

### 10. Why a local LLM instead of cloud APIs?
Data privacy, latency control, zero cloud API fees, and strict reproducibility. Enterprise support contracts frequently prohibit streaming customer communications to external multi-tenant cloud APIs.

### 11. Why `llama3.2:1b`? Was it trained?
`llama3.2:1b` is a **pretrained local generative open-weight model** running via Ollama. It was **NOT trained or fine-tuned by this project**. Instead, it is prompted dynamically using in-context retrieved exemplars, conversation history, and strict Pydantic JSON schemas. It offers a compact memory footprint (~1.3 GB) that runs efficiently on commodity CPU hardware without requiring dedicated GPUs.

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
**False Auto-Handle Rate (FAHR)** measures the proportion of turns requiring human escalation where the system mistakenly attempted autonomous handling. In the final evaluation, FAHR was 66.67% (30 missed escalations out of 45 true escalation checkpoints), representing the primary operational risk.

### 18. Why is Exact Match 59.50%?
Exact Match is a strict joint metric: **Intent + State + Action + Escalation must all match simultaneously**. While individual Intent Accuracy is 86.00%, Action Accuracy is 87.00%, and State Accuracy is 77.50%, compound probability across all four dimensions lowers exact match to **59.50%** overall. However, stratification demonstrates high reliability on routine inquiries: **81.63% on Easy**, **70.33% on Medium**, and **25.00% on Hard** multi-turn edge cases.

### 19. How did Action Accuracy perform in the final system?
The final unified system achieved **87.00% Action Accuracy**, aligning generative suggestions with deterministic action policy constraints. Where action discrepancies occurred, they typically involved plausible alternative customer guidance (e.g. `PROVIDE_INSTRUCTIONS` vs `REQUEST_SAFE_DETAILS`).

### 20. Why is State Macro-F1 low (40.04%)?
Severe class imbalance across the 8 dialogue states. The majority of turns reside in `STATE_INITIAL_INBOUND` and `STATE_AWAITING_INFO`, while rare intermediate states (`STATE_DISPUTE_RAISED`, `STATE_AGENT_ESCALATED`) have fewer examples, depressing unweighted macro-F1.

### 21. What does the 82.50% grounding metric actually mean?
In the verified final evaluation, **82.50% of responses were Evidence-Supported & Policy-Compliant**. We explicitly decouple this into:
- **Direct Exemplar Support (46.50%)**: Specific assertions, instructions, or links directly cited from retrieved historical exemplars.
- **Procedural / Policy Support (36.00%)**: Procedurally compliant routing (e.g., standard secure DM transfers) conforming to Amazon support policy without verbatim exemplar borrowing.
- **Unsupported (17.50%)**: Generic templates when retrieval similarity fell below threshold.
- **Contradictory (0.00%)**: Zero hallucinations contradicting retrieved policy.
Historical automated heuristics reported 99.50% "grounding" merely by checking for absence of prohibited tokens, which masked non-grounded generic generation.

### 22. Why was the genuine human review score (3.76/5) lower than the same-family LLM judge score (4.17/5)?
The local `llama3.2:1b` LLM judge is a **secondary, diagnostic tool, not ground truth**. Because it shares the same model family as the response generator, it exhibits substantial **leniency bias** (+0.41 composite inflation, scoring 4.17 vs human 3.76). It is substantially lenient on **Relevance** (scoring 4.88 vs human 3.42) and suffers from mode collapse (clustering at integer 4.0). In contrast, genuine human reviewers penalize non-specific troubleshooting, generic FAQ deflection, and partial address of compound customer inquiries.

### 23. What did the genuine human evaluation and LLM judge calibration reveal?
Across the completed post-fix genuine human evaluation ($N=40$ stratified across Easy, Medium, Hard), human expert composite was **3.76/5.00** vs the LLM judge's **4.17/5.00**:
- **Adjacent Agreement ($\pm 1$ grade)**: **83.33%** (exact agreement: 59.17%)
- **Quadratic Weighted Kappa (QWK)**: **0.0309**
- **Spearman Correlation ($\rho$)**: **0.0634**
- **Dimension Breakdown**:
  - **Relevance**: human **3.42** vs judge **4.88** (judge +1.46 lenient)
  - **Helpfulness**: human **3.30** vs judge **4.00** (judge +0.70 lenient)
  - **Groundedness**: human **3.58** vs judge **4.00** (judge +0.42 lenient)
  - **Action Appropriateness**: human **3.40** vs judge **3.77** (judge +0.37 lenient)
  - **Safety**: human **5.00** vs judge **4.70** (human confirmed 100% safety compliance)
  - **Communication Quality**: human **3.85** vs judge **4.00** (judge +0.15 lenient)
The near-zero rank correlation demonstrates that while adjacent agreement is high due to mode clustering, the automated judge cannot reliably rank-order response quality and must always be calibrated against genuine human reviews.

### 24. What is the biggest failure mode?
`WRONG_STATE` (45/200 cases, 22.5%) and `WEAK_EVIDENCE_GROUNDING` (35/200 cases, 17.5%), followed by `ESCALATION_MISS` (30/200 cases, 15.0%), where the agent misjudges customer dialogue progression or relies on fallback templates when retrieval similarity is weak.

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

### 31. How do you distinguish Text Normalization vs Embedding Normalization?
- **Text Normalization**: Deterministic string preprocessing operations applied to incoming raw tweet text (masking URLs to `https://t.co`, masking user handles to `@customer` and `@agent`, collapsing redundant whitespace, and standardizing casing).
- **Embedding Normalization**: Mathematical L2 unit-norm scaling ($\|v\|_2 = 1.0$) applied to the 384-dimensional dense vectors output by `all-MiniLM-L6-v2`. This ensures Euclidean distance and cosine similarity are mathematically equivalent, enabling high-performance dot-product vector search across the 5,502 retrieval exemplars.

