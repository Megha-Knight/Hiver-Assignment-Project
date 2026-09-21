# Engineering Decision Log: AmazonHelp Autonomous Support Agent

**Project**: AmazonHelp AI Customer Support Agent  
**Role**: Senior Engineering Lead  
**Scope**: 15 Non-Obvious Architectural, Methodological, and Governance Decisions

---

### Decision 01: Selection of AmazonHelp as Single Target Brand
- **Context / Problem**: The Kaggle *Customer Support on Twitter* dataset contains 2.81M tweets across 108 global brands (airlines, telecommunications, tech, retail). Training a multi-brand agent dilutes domain specificity and introduces contradictory support policies (e.g. airline baggage rules vs e-commerce return windows).
- **Chosen Approach**: Selected **AmazonHelp** as the sole brand domain.
- **Why It Was Chosen**: AmazonHelp had the highest volume of reconstructable multi-turn dialogues (85,087 conversations), dense transactional customer interactions, diverse operational workflows (package tracking, returns, digital Kindle/Prime troubleshooting, account lockouts), and rigorous public customer-care protocols.
- **Alternative Considered**: Multi-brand cross-domain training (e.g. AmazonHelp + AppleSupport + Delta) or choosing a smaller brand with simpler single-turn interactions (SpotifyCares).
- **Trade-off / Consequence**: The agent is deeply specialized for e-commerce and retail customer care; it cannot generalize to airline rebooking or telco SIM activation without domain re-indexing.
- **Evidence / Validation**: Data audit confirmed 358,973 AmazonHelp tweets, 189,133 customer inbound tweets, and 63,493 high-confidence English dialogues with sufficient depth for multi-turn state tracking.

---

### Decision 02: Conversation Tree Reconstruction via BFS Graph Traversal
- **Context / Problem**: Twitter data in `twcs.csv` is stored as disconnected single-tweet rows linked only by `in_response_to_tweet_id`. Treating customer tweets as isolated single-turn inputs discards historical context (e.g. a customer providing a postcode in turn 2 refers back to a late delivery complaint in turn 1).
- **Chosen Approach**: Implemented a breadth-first search (BFS) graph traversal over `in_response_to_tweet_id` to assemble complete, chronological root-to-leaf conversation trees.
- **Why It Was Chosen**: Accurate intent classification, dialogue state progression, and escalation detection depend fundamentally on prior turns and customer entity disclosures.
- **Alternative Considered**: Sliding-window context over adjacent timestamps, or treating every customer tweet as a standalone utterance.
- **Trade-off / Consequence**: Reconstructed dialogue trees required substantial post-processing to eliminate orphaned tweets, circular bot loops, and fragmented conversation branches.
- **Evidence / Validation**: Reconstructed 85,087 complete conversation trees, establishing the foundation for multi-turn dialogue state tracking and historical evaluation checkpoints.

---

### Decision 03: Controlled 10-Class Intent Taxonomy Design
- **Context / Problem**: Customer service requests exhibit massive lexical variation. A 3-class taxonomy (Query, Complaint, Feedback) is operationally useless for routing, while a 50+ class taxonomy produces severe data sparsity and low classifier accuracy.
- **Chosen Approach**: Standardized on a mutually exclusive, 10-intent operational taxonomy:
  1. `DELIVERY_STATUS_AND_TRACKING`
  2. `RETURN_REFUND_AND_REPLACEMENT`
  3. `CANCELLATION_AND_ORDER_MODIFICATION`
  4. `PAYMENT_BILLING_AND_PROMOTIONS`
  5. `PRODUCT_CONDITION_AND_WRONG_ITEM`
  6. `TECHNICAL_AND_DIGITAL_SUPPORT`
  7. `ACCOUNT_ACCESS_AND_SECURITY`
  8. `POLICY_AND_GENERAL_INQUIRIES`
  9. `PRIME_MEMBERSHIP_AND_BENEFITS`
  10. `OTHER_OR_UNCLEAR`
- **Why It Was Chosen**: Maps directly to discrete enterprise backend workflows and standard Amazon customer service routing queues while remaining learnable by classical and neural models.
- **Alternative Considered**: Dynamic zero-shot intent generation by the LLM, or a granular 35-intent hierarchy.
- **Trade-off / Consequence**: Multi-issue queries (e.g., customer complaining about a broken teapot AND demanding a refund) must be mapped to a single dominant primary intent.
- **Evidence / Validation**: Frozen TF-IDF + Logistic Regression achieved 87.50% intent accuracy and 83.08% Macro-F1 across all 10 classes, outperforming majority baseline (24.50% accuracy).

---

### Decision 04: English-Confidence Filtering & Pristine Candidate Filtering
- **Context / Problem**: Social media datasets contain multilingual code-switching, automated marketing bots, customer spam, and corrupted unicode strings.
- **Chosen Approach**: Filtered raw conversations using language detection (`langdetect`) and strict structural heuristics: minimum turn count $\ge 2$, at least one customer inbound and one agent response, stripping bot-broadcast loops.
- **Why It Was Chosen**: Guarantees that retrieval exemplars and evaluation checkpoints reflect meaningful human customer-support exchanges.
- **Alternative Considered**: Multilingual agent training across Spanish, German, French, and Japanese tweets present in AmazonHelp data.
- **Trade-off / Consequence**: Discarded ~21,500 non-English or malformed dialogues, focusing system scope purely on English support interactions.
- **Evidence / Validation**: Yielded 63,493 verified English dialogues and 53,637 pristine multi-turn candidate conversations from the initial 85,087 reconstructed pool.

---

### Decision 05: Strict Chronological Splitting over Random Sampling
- **Context / Problem**: Customer support dialogues exhibit strong temporal dependencies: evolving company policies, holiday shipping peaks, carrier outages, and website redesigns.
- **Chosen Approach**: Enforced strict chronological splitting:
  - **Train (70%)**: Earliest dialogues by timestamp.
  - **Dev / Validation (15%)**: Chronologically intermediate dialogues.
  - **Test (15%)**: Most recent 5,365 dialogues (100% untouched and unindexed).
- **Why It Was Chosen**: Random cross-validation causes temporal data leakage, training models on "future" customer patterns to predict past events and artificially inflating metrics.
- **Alternative Considered**: Uniform random 70/15/15 split across all conversation IDs.
- **Trade-off / Consequence**: Models are evaluated on future temporal distributions, which depresses raw benchmark metrics slightly but faithfully mirrors production deployment.
- **Evidence / Validation**: Verified temporal isolation across all split boundaries; 100% of the 5,365 Test partition conversations remained completely unread throughout all experimental phases.

---

### Decision 06: Train-Only Retrieval Corpus Isolation (5,502 Exemplars)
- **Context / Problem**: Dense vector retrieval can easily introduce catastrophic data leakage if the retrieval index indexes conversations that appear in the validation or test benchmarks.
- **Chosen Approach**: Built the retrieval index exclusively from 5,502 verified resolution dialogues sampled strictly from the **Train partition**.
- **Why It Was Chosen**: Guarantees zero overlap between retrieval evidence and the 200 Dev evaluation checkpoints or the 5,365 Test conversations.
- **Alternative Considered**: Indexing all 63,493 available English conversations to maximize retrieval density.
- **Trade-off / Consequence**: Retrieval corpus is restricted to 5,502 exemplars, meaning recent edge-case phrasing found in Dev/Test cannot be directly matched.
- **Evidence / Validation**: Automated partition boundary audit confirmed zero shared dialogue IDs (`chk_*` vs `ret_*`) and zero temporal contamination between index and benchmark.

---

### Decision 07: Dense Vector Retrieval (`all-MiniLM-L6-v2`) with Top $K=5$
- **Context / Problem**: Lexical BM25 keyword matching fails on social media phrasing where colloquial customer language does not share tokens with formal representative responses.
- **Chosen Approach**: Encoded historical customer inquiries using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors) and retrieved top $K=5$ exemplars via cosine similarity.
- **Why It Was Chosen**: MiniLM-L6-v2 is lightweight (80MB), executes on CPU in $<25$ ms, and semantically bridges colloquial customer complaints to historical representative actions. $K=5$ provides sufficient operational diversity without exhausting local LLM context windows.
- **Alternative Considered**: BM25 sparse retrieval; larger 768-dim embeddings (e.g. `bge-large`); $K=10$ retrieval depth.
- **Trade-off / Consequence**: $K=5$ adds ~20–30 ms retrieval latency per turn and requires prompt space in the 1B LLM context window.
- **Evidence / Validation**: Produced 82.5% Evidence-Supported & Policy-Compliant responses in post-fix evaluation, with 46.5% direct exemplar grounding.

---

### Decision 08: Retaining Frozen TF-IDF + Logistic Regression as Primary Intent Anchor
- **Context / Problem**: Using a 1B generative LLM directly for intent classification introduces generative non-determinism, hallucinations, and high latency (~1.5s per turn).
- **Chosen Approach**: Retained the frozen TF-IDF + Logistic Regression classifier as the primary, authoritative intent signal, feeding its class probabilities into the decision engine.
- **Why It Was Chosen**: TF-IDF N-grams achieve 87.50% accuracy on baseline, execute in $<3$ ms on commodity CPU, and provide calibrated, deterministic probabilities.
- **Alternative Considered**: End-to-end LLM prompt-based classification; fine-tuning a BERT/RoBERTa cross-encoder.
- **Trade-off / Consequence**: N-gram bag-of-words representation lacks deep syntactic understanding on complex multi-sentence customer complaints.
- **Evidence / Validation**: Intent Accuracy reached 86.00% across the 200 Dev checkpoints, maintaining sub-millisecond classification overhead and preventing LLM intent hallucinations.

---

### Decision 09: Deterministic State Machine Tracking over Generative State Tracking
- **Context / Problem**: Dialogue state tracking in support conversations must follow rigid operational boundaries (e.g. you cannot jump from `STATE_INITIAL_INBOUND` directly to `STATE_CLOSED` without proposing resolution).
- **Chosen Approach**: Implemented a deterministic state machine tracking 8 controlled states based on turn depth, customer entity presence, and action history.
- **Why It Was Chosen**: Small open-weight models (1B) frequently suffer from state drift, forgetting that order details were already requested or looping in initial greetings.
- **Alternative Considered**: In-context LLM state tracking via JSON generation.
- **Trade-off / Consequence**: Rigid state transitions occasionally lag behind complex customer interactions, resulting in `WRONG_STATE` in 22.5% of edge cases.
- **Evidence / Validation**: State tracking achieved 77.50% accuracy on the 200 Dev checkpoints, preventing invalid multi-turn state jumps.

---

### Decision 10: Dual-Pathway Deterministic Escalation Policy
- **Context / Problem**: In customer support, failing to escalate a severe complaint (fraud, legal threat, repeated failure) is far more hazardous than a false escalation. Delegating escalation decisions entirely to a 1B LLM resulted in severe under-escalation.
- **Chosen Approach**: Implemented a deterministic dual-pathway escalation engine:
  1. *Mandatory Intent Triggers*: Immediate escalation on `ACCOUNT_ACCESS_AND_SECURITY` (fraud) and disputed `PAYMENT_BILLING_AND_PROMOTIONS`.
  2. *Lexical & Frustration Triggers*: Pattern-based regex matching for legal threats, regulatory ombudsman mentions, or repeated contact complaints.
- **Why It Was Chosen**: Guarantees zero missed escalations on known high-liability legal, security, and financial triggers.
- **Alternative Considered**: Prompting the LLM to output `"escalate": true/false` based on sentiment.
- **Trade-off / Consequence**: Produces an Escalation Precision of 45.45% and a False Auto-Handle Rate of 66.67% on subtle edge cases lacking explicit keywords.
- **Evidence / Validation**: 100% of tested fraud and ombudsman scenarios (`DEMO_05`, `DEMO_08`) triggered deterministic escalation override, eliminating brand liability.

---

### Decision 11: Authoritative Post-Generation Safety Validation Layer
- **Context / Problem**: Generative models cannot be trusted to self-police safety. Prompt instructions like *"never ask for passwords"* are routinely circumvented by jailbreaks or unusual customer inputs.
- **Chosen Approach**: Built a deterministic post-generation safety validator (`DeterministicSafetyValidator`) that scans the drafted LLM response and hard-blocks credential solicitation and unsupported transactional claims.
- **Why It Was Chosen**: Safety must be mathematically guaranteed, not probabilistic. If an LLM response solicits a password or claims *"I have refunded your £50"*, the safety validator intercepts the text and forces safe direct-message handoff.
- **Alternative Considered**: Pre-prompt safety instructions alone; LLM-as-a-judge safety filtering.
- **Trade-off / Consequence**: Occasionally sanitizes benign responses that mention words like "card" or "verify", forcing generic secure channel routing.
- **Evidence / Validation**: Evaluated across 200 checkpoints: **0.00% deterministic safety violations, 0 credential breaches, 2 safety overrides applied, 100% deterministic credential protection verified**.

---

### Decision 12: Local 1B Model (`llama3.2:1b`) with Deterministic Mock Simulation
- **Context / Problem**: Enterprise support systems cannot leak unredacted customer PII to commercial cloud APIs. Furthermore, CI/CD automated test pipelines require reproducible, deterministic execution without downloading multi-gigabyte models or requiring GPU servers.
- **Chosen Approach**: Supported a dual execution path:
  1. *Live Local Ollama*: Local `llama3.2:1b` execution on commodity CPU via Ollama.
  2. *Offline Deterministic Mock*: Reconciled baseline and retrieval simulation generating deterministic responses in $<200$ ms.
- **Why It Was Chosen**: Ensures complete data sovereignty (zero cloud egress), local CPU feasibility, and 100% reproducible testing across CI/CD and evaluation harnesses.
- **Alternative Considered**: Cloud-only API architecture (OpenAI GPT-4o); mandatory GPU infrastructure.
- **Trade-off / Consequence**: The 1B model has lower reasoning depth than frontier cloud models, requiring strict prompt schemas and external deterministic guardrails.
- **Evidence / Validation**: Verified in CLI and Streamlit: seamless execution on CPU with graceful automatic fallback to Mock Simulation when Ollama is offline.

---

### Decision 13: Frozen 200-Checkpoint Benchmark with Stratified Difficulty Tiers
- **Context / Problem**: Benchmarking across successive development phases (Phase 4 baseline through Phase 8 UI) requires a static, non-drifting target. Evaluating on unstratified random samples over-indexes on easy routine queries.
- **Chosen Approach**: Established a permanent frozen benchmark of 200 Dev checkpoints explicitly stratified across difficulty tiers: **49 Easy, 91 Medium, 60 Hard multi-turn edge cases**.
- **Why It Was Chosen**: Prevents benchmark drift and forces evaluation to test complex failure modes (state transitions, escalations, credential protection) rather than trivial single-turn FAQs.
- **Alternative Considered**: Rolling online evaluation; unstratified uniform sampling.
- **Trade-off / Consequence**: Benchmark size is bounded ($N=200$), meaning confidence intervals are wider than a 10,000-sample test run.
- **Evidence / Validation**: Successfully revealed the performance degradation curve: Easy Exact Match (81.63%) vs Medium (70.33%) vs Hard (25.00%).

---

### Decision 14: Treating Same-Family LLM-as-a-Judge as Secondary and Diagnostic
- **Context / Problem**: Automated LLM judges are increasingly popular for evaluating natural language responses, but using the same model family (`llama3.2:1b`) to grade its own output introduces severe self-preference bias.
- **Chosen Approach**: Classified the LLM judge as strictly secondary/diagnostic, establishing **authoritative genuine human evaluation ([`data/evaluation/genuine_human_reviews_n40.jsonl`](../data/evaluation/genuine_human_reviews_n40.jsonl), $N=40$, reviewed by `human_reviewer_1`) as ground truth**.
- **Why It Was Chosen**: Rigorous recomputation from per-item scores revealed that the LLM judge exhibited severe leniency (**+2.01 composite inflation**, scoring 4.22 vs human 2.21), low adjacent agreement (**30.42%**), and near-zero rank correlation with human judgment (QWK: 0.0114, Spearman: 0.0497).
- **Alternative Considered**: Presenting the 4.17/5.00 LLM judge score as primary proof of conversational excellence.
- **Trade-off / Consequence**: Avoids inflated marketing claims and presents an honest, defensible evaluation of model quality.
- **Evidence / Validation**: Documented blind spot in checkpoint `chk_4818c7c9`, where the LLM judge rated a premature canned closure as 4/5 while human review penalized it as unhelpful (1/5). Exactly cataloged in [`results/phase7/phase7d_genuine_human_agreement.json`](../results/phase7/phase7d_genuine_human_agreement.json).

---

### Decision 15: Reporting a Multi-Dimensional Operational Scorecard over a Single Headline Metric
- **Context / Problem**: Reporting a single aggregated accuracy or satisfaction score conceals dangerous operational failures in high-risk edge cases.
- **Chosen Approach**: Enforced a multi-dimensional scorecard reporting:
  - Intent Accuracy (86.00%) & Macro F1 (80.73%)
  - Four-Field Exact Match (59.50% overall, 25.00% Hard)
  - Action Accuracy (87.00%) & State Accuracy (77.50%)
  - Escalation F1 (38.46%) & False Auto-Handle Rate (66.67%)
  - Evidence Grounding Breakdown (82.5% supported: 46.5% direct, 36.0% procedural)
  - Deterministic Safety Violations (0.00%)
- **Why It Was Chosen**: An agent with 86% intent accuracy but a 66.67% False Auto-Handle Rate is NOT ready for unmonitored production autonomy. Full visibility into individual failure dimensions is mandatory for enterprise engineering.
- **Alternative Considered**: Blending all metrics into a single proprietary "Agent Quality Index" (0–100).
- **Trade-off / Consequence**: Requires interviewers and evaluators to digest a nuanced multi-field matrix rather than a simple marketing soundbite.
- **Evidence / Validation**: Directly drove the prioritized engineering roadmap targeting escalation sensitivity and state transition robustness.
