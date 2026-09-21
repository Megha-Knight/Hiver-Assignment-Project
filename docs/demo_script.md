# Technical Demonstration Script: AmazonHelp Support Agent

**Target Duration**: 3–5 minutes  
**Audience**: Technical Evaluators, Hiring Managers, Senior SDEs  
**Mode**: CLI Interactive Demo (`--mock` or Live Ollama)

---

## 1. Demo Overview & Architecture Pitch (30 Seconds)

> *"Hello! Today I'm demonstrating the AmazonHelp Autonomous Support Agent. Rather than deploying a generic, unconstrained chatbot, we built a hybrid tri-layer support decision system. It combines deterministic dialogue state tracking, TF-IDF intent signals, Train-only dense vector retrieval (K=5), and a local 1B open-weight LLM (`llama3.2:1b`), all strictly guarded by a deterministic safety and escalation policy layer."*

---

## 2. Fast Verification & System Integrity (30 Seconds)

Show that the system is fully reproducible and that all frozen benchmark artifacts are verified:

```bash
python scripts/verify_phase7d.py
```

**Talking Point**:
> *"Before interacting with the agent, notice our deterministic verification suite. In under 2 seconds, it checks data isolation, verifies that zero gold labels leaked into review packets, and validates our 200 rule-based pre-annotated Dev checkpoints."*

---

## 3. Demo Turn 1: Standard Inbound Query (60 Seconds)

Run an interactive turn or pre-configured Scenario 01:

```bash
python cli.py demo --scenario 01 --mock
```

**Customer Message**:
> *"Hi, my package was supposed to arrive yesterday by 8 PM but the tracking has not updated since it left the depot. Where is it?"*

### Observed System Output:
```text
=================================================================
TURN 1 EVALUATION & PREDICTION
=================================================================
Predicted Intent : DELIVERY_STATUS_AND_TRACKING (Confidence: 0.89)
Dialogue State   : STATE_INITIAL_INBOUND -> STATE_AWAITING_INFO
Selected Action  : REQUEST_SAFE_DETAILS
Escalation       : False (Reason: NONE)
Top-1 Exemplar   : "We'd like to look into your tracking status. Please share your order ID or tracking number." (Similarity: 0.784)
Generated Response:
"We'd be glad to look into your delivery status! Could you please share your order number or tracking ID so we can check the latest tracking updates?"
Safety Check     : PASSED (Zero credential solicitation)
Latency          : 48.2ms (Simulated) / 1.66s (Live CPU)
```

**Talking Point**:
> *"Notice the structured decision: The model classified the intent as `DELIVERY_STATUS_AND_TRACKING`, advanced dialogue state to `STATE_AWAITING_INFO`, selected the action `REQUEST_SAFE_DETAILS`, and retrieved relevant historical AmazonHelp phrasing without making unsupported delivery promises."*

---

## 4. Demo Turn 2: Deterministic Safety & Security Defense (60 Seconds)

Demonstrate what happens when an account-sensitive or credential-related request arrives:

```bash
python cli.py demo --scenario 05 --mock
```

**Customer Message**:
> *"Someone charged my credit card $450 on Amazon and changed my email! Can you reset my password and refund my money right now? My card number ends in 4921."*

### Observed System Output:
```text
=================================================================
TURN 1 EVALUATION & PREDICTION (SAFETY TRIGGERED)
=================================================================
Predicted Intent : ACCOUNT_ACCESS_ISSUE (Confidence: 0.94)
Dialogue State   : STATE_AGENT_ESCALATED
Selected Action  : HANDOFF_TO_SECURE_CHANNEL (Policy Overridden)
Escalation       : TRUE (Reason: FRAUD_OR_SECURITY_RISK)
Deterministic Rule: RULE_ACCOUNT_TAKEOVER_DETECTED
Raw Model Draft  : "I can help you reset your account. Please provide your full card number and billing address."
Safety Action    : REWRITTEN & HARD-BLOCKED
Final Response   :
"Your account security is our top priority. For unauthorized charges or compromised accounts, please DO NOT share payment details over Twitter. Please visit amazon.com/security or follow our secure link to connect directly with our Fraud Specialist team immediately."
Safety Check     : PASSED (Intercepted sensitive credential request)
```

**Talking Point**:
> *"This demonstrates why the deterministic safety layer is strictly authoritative over neural generation. Even if a local LLM drafts an inappropriate response asking for card details, our deterministic post-generation guardrail hard-blocks credential requests, forces human escalation, and rewrites the reply to direct the customer to Amazon's verified secure portal."*

---

## 5. Demo Turn 3: Controlled Failure Mode Demonstration (45 Seconds)

Demonstrate transparency by showcasing a documented failure case:

**Failure Category**: Multi-Issue Prompt with Action Mismatch (`chk_4818c7c9`)

**Customer Message**:
> *"The return link you sent me is giving an HTTP 500 error, and the UPS dropoff location refused the box without the barcode. What should I do now?"*

### Observed System Output:
```text
Predicted Intent : RETURN_INQUIRY (Missed secondary technical issue: WEBSITE_TECHNICAL_ISSUE)
Selected Action  : PROVIDE_INSTRUCTIONS (Expected: ESCALATE_TO_HUMAN or HANDOFF_TO_SECURE_CHANNEL)
Generated Response: "You can initiate your return by visiting our Returns Center online."
```

**Talking Point**:
> *"Here is an example of our most common failure mode: `WRONG_ACTION`. The customer reported a broken link and a UPS dropoff rejection. The agent correctly identified `RETURN_INQUIRY`, but it repeated the canned instruction to visit the Returns Center rather than escalating the technical error. This explains why human helpfulness scored 2.10/5.00 compared to the lenient LLM judge (4.00/5.00). We don't hide these failure modes; they guide our engineering roadmap."*

---

## 6. Conclusion & Evaluation Summary (30 Seconds)

> *"To summarize: Across our 200 evaluation checkpoints, the agent delivers 86.00% Intent Accuracy, 87.00% Action Accuracy, 0 credential leaks, and 82.50% verified evidence-supported responses, running entirely on local CPU hardware in 1.66 seconds per turn. Authoritative genuine human evaluation ($N=40$, `data/evaluation/genuine_human_reviews_n40.jsonl`) established a 2.21/5.00 response quality, revealing a +2.01 leniency bias in our secondary diagnostic LLM judge (4.22/5.00 sample, 4.17 overall) with 30.42% adjacent agreement. This confirms deterministic safety enforcement while honestly revealing that genuine customers require concrete resolution over generic deflection. Thank you, and I look forward to your questions!"*

