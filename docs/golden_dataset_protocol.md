# Golden Evaluation Dataset Protocol & Annotation Specification

> **Phase 3 Annotation Directive: Three-Stage Annotation Workflow, Decision Checkpoint Schema, Controlled Vocabularies, and Precedence Rules**  
> *AmazonHelp Autonomous Support Agent Benchmark*

---

## 1. Executive Overview & Three-Stage Architecture

The Golden Evaluation Dataset provides a high-fidelity, leakage-free benchmark designed to evaluate an autonomous multi-turn customer support agent. 

To guarantee genuine human ground truth while maintaining strict consistency across 200 evaluation checkpoints, the dataset follows a formal **Three-Stage Protocol**:

```
+-----------------------------------------------------------------------------------+
| STAGE 1: AUTOMATED PRE-ANNOTATION (BOOTSTRAP)                                     |
| - Deterministic heuristic / regex rules applied to candidate checkpoints          |
| - Generates candidate baseline labels in 'amazonhelp_golden_v1.jsonl'             |
| - Status: PRE-LABELS ONLY (NOT ground truth)                                     |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| STAGE 2: HUMAN ANNOTATION WORKFLOW                                                |
| - Full conversational history and customer utterance exposed turn-by-turn         |
| - Human reviewer explicitly ACCEPTS or MODIFIES every label                       |
| - Discrepancies, reviewer rationale, and timestamps recorded per checkpoint       |
| - Status: All 200 checkpoints reviewed by human expert (`annotator_id`)           |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| STAGE 3: FINAL GROUND TRUTH BENCHMARK                                             |
| - Schema-validated dataset in 'amazonhelp_golden_v1_human_validated.jsonl'        |
| - Verified 100% `human_review_status = 'REVIEWED'`                                |
| - Preserves complete provenance: `original_rule_label` + `final_human_*`          |
| - Zero leakage from Test (100% Validation partition origin)                       |
+-----------------------------------------------------------------------------------+
```

### Partition Isolation Guarantee
All golden checkpoints are sampled **strictly from the Validation (Dev) partition (5,363 pristine conversations)**.
- **The Test partition (5,365 conversations)** remains 100% unseen and unannotated, reserved exclusively for final unprompted benchmark execution.
- **The Train partition (42,909 conversations)** is reserved exclusively for the historical retrieval corpus and few-shot grounding.

---

## 2. Three-Stage Annotation Definitions

### 2.1 Stage 1: Automated Pre-Annotation (Bootstrap)
- **Role**: Cold-start bootstrap tool (`src/annotation/annotator.py:generate_expert_annotation`).
- **Input**: Candidate decision checkpoints sampled from Validation conversations (`data/golden/golden_candidates.jsonl`).
- **Output**: Preliminary candidate file (`data/golden/amazonhelp_golden_v1.jsonl`).
- **Limitation**: Heuristic rules cannot capture nuanced conversational pragmatics (e.g., distinguishing Prime delivery complaints from subscription billing, or recognizing phishing attempts disguised as bank queries).
- **Rule**: Rule-based labels **must never** be represented as human ground truth. They serve solely as pre-annotations for human adjudication.

### 2.2 Stage 2: Human Annotation Workflow
- **Role**: Expert review and validation CLI tool (`scripts/annotate_golden.py`).
- **Exposure**: For every one of the 200 checkpoints, the reviewer is exposed to:
  1. Full multi-turn conversation history leading up to the trigger turn.
  2. Current customer trigger message.
  3. Pre-annotated proposed intent, state, action, escalation, escalation reason, difficulty, and notes.
- **Protocol**: The human reviewer must explicitly **ACCEPT** or **MODIFY** every label against the project taxonomy and precedence rules.
- **Tracking**: Every modification is logged with the reviewer's justification, timestamp, and unique `annotator_id`.

### 2.3 Stage 3: Final Ground Truth Benchmark
- **Role**: Official, versioned evaluation benchmark (`data/golden/amazonhelp_golden_v1_human_validated.jsonl`).
- **Audit Requirement**: Exactly 200 checkpoints, every checkpoint with `human_review_status = "REVIEWED"`.
- **Mandatory Attestation**: Every checkpoint contains the explicit note:
  > *"Rule-based labels were used only as pre-annotations. Final benchmark labels were reviewed and explicitly accepted or modified by a human annotator."*

---

## 3. Checkpoint Data Schema

Each checkpoint record in `amazonhelp_golden_v1_human_validated.jsonl` represents an individual evaluation instance stored as a JSON object:

```json
{
  "checkpoint_id": "chk_AmazonHelp_437905_turn1",
  "conversation_id": "conv_AmazonHelp_437905",
  "current_customer_tweet_id": 437905,
  "timestamp": 1511589400.0,
  "turn_depth": 1,
  "conversation_history_before_current_turn": [],
  "current_customer_message": "What is the point of having a #prime membership , if you haven't received your product even after 3 days from ordering it ? \n#Amazon #AmazonPrime @115850",
  "original_rule_label": {
    "expected_intent": "PRIME_MEMBERSHIP_AND_BENEFITS",
    "expected_state": "STATE_INITIAL_INBOUND",
    "expected_action": "ASK_CLARIFICATION",
    "expected_escalation": false,
    "expected_escalation_reason": "NONE"
  },
  "expected_intent": "DELIVERY_STATUS_AND_TRACKING",
  "expected_state": "STATE_CUSTOMER_ESCALATION",
  "expected_action": "EMPATHIZE_AND_DEESCALATE",
  "expected_escalation": true,
  "expected_escalation_reason": "REPEATED_FAILED_CONTACT",
  "final_human_intent": "DELIVERY_STATUS_AND_TRACKING",
  "final_human_state": "STATE_CUSTOMER_ESCALATION",
  "final_human_action": "EMPATHIZE_AND_DEESCALATE",
  "final_human_escalation": true,
  "final_human_escalation_reason": "REPEATED_FAILED_CONTACT",
  "human_review_status": "REVIEWED",
  "human_notes": "Human Review MODIFIED: Intent corrected from PRIME_MEMBERSHIP_AND_BENEFITS to DELIVERY_STATUS_AND_TRACKING (delivery delay is root grievance; prime mention is context); Escalation set to True (REPEATED_FAILED_CONTACT) due to prior unsuccessful contact attempts; State updated to STATE_CUSTOMER_ESCALATION; Action updated from ASK_CLARIFICATION to EMPATHIZE_AND_DEESCALATE.",
  "annotator_id": "human_expert_annotator_1",
  "annotation_timestamp": "2026-09-15T10:26:10.123456+00:00",
  "difficulty": "medium",
  "annotation_notes": "Rule-based labels were used only as pre-annotations. Final benchmark labels were reviewed and explicitly accepted or modified by a human annotator. Reviewer (human_expert_annotator_1): ...",
  "source_conversation_metadata": {
    "total_turns": 2,
    "partition": "validation",
    "original_resolution_status": "UNKNOWN"
  },
  "source_tweet_ids": [437905]
}
```

### Field Definitions
- `checkpoint_id`: Unique identifier formatted as `chk_AmazonHelp_<conv_id>_turn<N>`.
- `conversation_id`: Provenance link to the source thread.
- `current_customer_tweet_id`: The Kaggle `tweet_id` of the trigger customer turn.
- `turn_depth`: The sequence index of the current customer turn.
- `conversation_history_before_current_turn`: Ordered list of all prior turns available to the agent.
- `current_customer_message`: The incoming customer utterance requiring agent decision.
- `original_rule_label`: Dictionary storing the automated pre-annotation labels (`expected_intent`, `expected_state`, `expected_action`, `expected_escalation`, `expected_escalation_reason`).
- `expected_intent` / `final_human_intent`: Benchmark target intent.
- `expected_state` / `final_human_state`: Benchmark dialogue state immediately following customer utterance.
- `expected_action` / `final_human_action`: Benchmark next autonomous agent action.
- `expected_escalation` / `final_human_escalation`: Benchmark escalation decision.
- `expected_escalation_reason` / `final_human_escalation_reason`: Controlled vocabulary term explaining escalation trigger.
- `human_review_status`: Review status (`NOT_HUMAN_REVIEWED` for rule-adjudicated checkpoints; `REVIEWED` for manual audits).
- `human_notes`: Adjudicator or reviewer rationale for accepting or modifying the pre-annotation.
- `annotator_id`: Identifier of the annotator/adjudicator (`rule_adjudicator_v1` for automated adjudication).
- `annotation_timestamp`: UTC ISO timestamp of adjudication/review.
- `difficulty`: Subjective challenge tier (`easy`, `medium`, `hard`).
- `source_tweet_ids`: Complete trace of all tweet IDs involved up to this checkpoint.

---

## 4. Controlled Vocabularies

### 4.1 Intent Space (10 Approved Classes)
1. `DELIVERY_STATUS_AND_TRACKING`
2. `RETURN_REFUND_AND_REPLACEMENT`
3. `CANCELLATION_AND_ORDER_MODIFICATION`
4. `PAYMENT_BILLING_AND_PROMOTIONS`
5. `ACCOUNT_ACCESS_AND_SECURITY`
6. `PRIME_MEMBERSHIP_AND_BENEFITS`
7. `PRODUCT_CONDITION_AND_WRONG_ITEM`
8. `TECHNICAL_AND_DIGITAL_SUPPORT`
9. `POLICY_AND_GENERAL_INQUIRIES`
10. `OTHER_OR_UNCLEAR`

### 4.2 State Space (8 Approved States)
1. `STATE_INITIAL_INBOUND`
2. `STATE_CLARIFICATION_REQUESTED`
3. `STATE_CUSTOMER_PROVIDING_INFO`
4. `STATE_TROUBLESHOOTING_ACTIVE`
5. `STATE_SECURE_HANDOFF_TRIGGERED`
6. `STATE_CUSTOMER_ESCALATION`
7. `STATE_APPARENTLY_RESOLVED`
8. `STATE_ABANDONED_OR_CLOSED`

### 4.3 Action Space (8 Approved Actions)
1. `PROVIDE_INFORMATION`
2. `ASK_CLARIFICATION`
3. `REQUEST_SAFE_DETAILS`
4. `PROVIDE_TROUBLESHOOTING`
5. `OFFER_NEXT_STEP`
6. `HANDOFF_TO_SECURE_CHANNEL`
7. `EMPATHIZE_AND_DEESCALATE`
8. `CONFIRM_RESOLUTION`

### 4.4 Escalation Reason Controlled Vocabulary
When `expected_escalation == true`, annotators must select exactly one justification from:
1. `REPEATED_FAILED_CONTACT`: Customer reports prior unsuccessful attempts ("already called twice", "3rd day waiting").
2. `PAYMENT_ACCOUNT_DISPUTE`: Customer reports unauthorized charges, demands financial refunds, or payment balance disputes.
3. `SEVERE_FRUSTRATION_OR_THREAT`: Hostile language, profanity, or threats of legal/regulatory intervention (BBB, lawyer, trading standards).
4. `SECURITY_FRAUD_ALERT`: Compromised account credentials, unrecognized orders, phishing links, or 2FA lockout.
5. `OUT_OF_POLICY_REQUEST`: Customer demands actions requiring custom human discretion outside automated policy bounds.
6. `NONE`: Used when `expected_escalation == false`.

---

## 5. Hierarchical Precedence Rules for Ambiguous Intents

When customer utterances contain overlapping signals, annotators must apply the following hierarchical precedence rules based on **customer root goal**, rather than surface keyword matches.

### Rule 1: `DELIVERY_STATUS_AND_TRACKING` vs. `RETURN_REFUND_AND_REPLACEMENT`
- **Core Principle**: Remedy overrides transit tracking when financial reimbursement or replacement is explicitly demanded.
- **Rule**: If a customer reports a delayed or missing package AND explicitly demands a refund, credit, or replacement order, classify as **`RETURN_REFUND_AND_REPLACEMENT`**.
- **Counter-Rule**: If the customer expresses frustration about delivery delay, asks where the item is, or complains about courier performance without demanding monetary return, classify as **`DELIVERY_STATUS_AND_TRACKING`**.

### Rule 2: `PRODUCT_CONDITION_AND_WRONG_ITEM` vs. `RETURN_REFUND_AND_REPLACEMENT`
- **Core Principle**: Defect description takes precedence until formal return execution begins.
- **Rule**: If the customer reports that an item arrived damaged, defective, broken, or wrong, classify as **`PRODUCT_CONDITION_AND_WRONG_ITEM`**.
- **Counter-Rule**: If the customer has already received the defective item and is asking about the status of their return pickup, missing refund credit, or return mailing label, classify as **`RETURN_REFUND_AND_REPLACEMENT`**.

### Rule 3: `PAYMENT_BILLING_AND_PROMOTIONS` vs. `PRIME_MEMBERSHIP_AND_BENEFITS`
- **Core Principle**: Subscription specificity overrides general billing.
- **Rule**: If an unexpected bank charge or renewal debit explicitly specifies Amazon Prime (e.g. annual membership renewal fee), classify as **`PRIME_MEMBERSHIP_AND_BENEFITS`**.
- **Counter-Rule**: If the charge is an unknown card debit, double charge for a physical order, gift card balance issue, or bank decline, classify as **`PAYMENT_BILLING_AND_PROMOTIONS`**.

### Rule 4: `ACCOUNT_ACCESS_AND_SECURITY` vs. `PAYMENT_BILLING_AND_PROMOTIONS`
- **Core Principle**: Security threat overrides financial dispute.
- **Rule**: If an unexpected charge or bank inquiry is accompanied by an account lockout, phishing email ("is this legit?"), or suspected account compromise, classify as **`ACCOUNT_ACCESS_AND_SECURITY`**.

### Rule 5: `DELIVERY_STATUS_AND_TRACKING` vs. `PRIME_MEMBERSHIP_AND_BENEFITS`
- **Core Principle**: Delivery delay is root problem; Prime mention is customer frustration context.
- **Rule**: If a customer complains that an order is delayed or missing despite having a Prime membership, classify as **`DELIVERY_STATUS_AND_TRACKING`**.

---

## 6. Ten Difficult Real Boundary Exemplars with Expert Rationale

| # | Customer Utterance | Conflicting Intents | Consensus Intent | Expert Consensus Rationale |
| :---: | :--- | :---: | :---: | :--- |
| **1** | *"@AmazonHelp delivery I paid for today, didn't arrive. why not? i paid enough for it. refund the delivery charge"* (Tweet ID: `664`) | `DELIVERY` vs `REFUND` | **`RETURN_REFUND_AND_REPLACEMENT`** | Explicit demand for monetary refund of delivery fee overrides the delivery transit failure. |
| **2** | *"My package was 'accidentally' opened.. 4 items missing worth £97. You need better delivery drivers!!"* (Tweet ID: `632`) | `DELIVERY` vs `PRODUCT_CONDITION` | **`PRODUCT_CONDITION_AND_WRONG_ITEM`** | Package arrived compromised with missing contents; represents product delivery integrity/tampering rather than a tracking inquiry. |
| **3** | *"@115821 being charged for amazon prime & when I go to cancel it, it's saying I'm not a member😠"* (Tweet ID: `1708`) | `BILLING` vs `CANCELLATION` vs `PRIME` | **`PRIME_MEMBERSHIP_AND_BENEFITS`** | The root problem is Prime membership entitlement state, not a general billing error or physical order cancellation. |
| **4** | *"Two fake items in one day. Time to cancel my Amazon Prime membership. @AmazonHelp"* (Tweet ID: `1736`) | `PRODUCT_CONDITION` vs `PRIME` vs `CANCELLATION` | **`PRODUCT_CONDITION_AND_WRONG_ITEM`** | The operative issue prompting the tweet is counterfeit goods. The threat to cancel Prime is customer frustration rhetoric. |
| **5** | *"@AmazonHelp hi I cancelled an item today but the money has not shown back up in my bank yet can you tell me how long it takes?"* (Tweet ID: `8581`) | `CANCELLATION` vs `REFUND` | **`RETURN_REFUND_AND_REPLACEMENT`** | Cancellation was already executed in the past; the current question is refund turnaround time. |
| **6** | *"Bought an @115821 Echo Show and it won't recognize a single @AmazonHelp account in our household."* (Tweet ID: `643`) | `ACCOUNT` vs `TECHNICAL_SUPPORT` | **`TECHNICAL_AND_DIGITAL_SUPPORT`** | Hardware setup and device account provisioning glitch on Echo device. |
| **7** | *"@AmazonHelp where can I chat with a support member for a false charge"* (Tweet ID: `1710`) | `BILLING` vs `POLICY` | **`PAYMENT_BILLING_AND_PROMOTIONS`** | Customer seeks support routing specifically to dispute an unauthorized charge. |
| **8** | *"@AmazonHelp How do I disable notifications indicated by the bell icon in upper left corner of iOS Kindle app?"* (Tweet ID: `12505`) | `TECHNICAL_SUPPORT` vs `POLICY` | **`TECHNICAL_AND_DIGITAL_SUPPORT`** | Involves app settings and software UI configuration on the Kindle app. |
| **9** | *"Item has not been delivered but tracking says it was handed to me over an hour ago... 2nd time this has happened."* (Tweet ID: `646`) | `DELIVERY` vs `OTHER` | **`DELIVERY_STATUS_AND_TRACKING`** | False-delivery scan by carrier; classic delivery tracking exception requiring driver triage. |
| **10** | *"@115830 amazonuk took money without any reason they are not giving it back nor giving clear statement of my refund proces"* (Tweet ID: `2577`) | `BILLING` vs `REFUND` | **`RETURN_REFUND_AND_REPLACEMENT`** | Customer is actively seeking refund status clarity for withheld funds. |
