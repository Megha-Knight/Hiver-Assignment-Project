# Phase 4 Deterministic Decision Policy Specification

> **State Transitions, Escalation Triggers, Action Selection, and Safety Guardrails**  
> *AmazonHelp Autonomous Support Agent Policy Layer*

---

## 1. Overview

The deterministic policy layer operates as an auditable rule engine that translates intent classifications and multi-turn dialogue context into concrete agent actions and escalation decisions.

---

## 2. Escalation Policy Specification

### 2.1 Escalation Trigger Precedence
When evaluating customer utterances, triggers are evaluated in strict hierarchical precedence:

1. **`SECURITY_FRAUD_ALERT`**:
   - *Triggers*: Phishing alerts, fake emails ("is this legit?"), account compromise, unauthorized password resets, 2FA bypass attempts.
   - *Required Action*: Immediate handoff to secure verification channel.
2. **`SEVERE_FRUSTRATION_OR_THREAT`**:
   - *Triggers*: Threats of litigation ("lawyer", "court"), regulatory filings ("trading standards", "BBB"), severe hostility or profanity.
   - *Required Action*: Immediate de-escalation by agent and human supervisory routing.
3. **`PAYMENT_ACCOUNT_DISPUTE`**:
   - *Triggers*: Multiple charges ("charged three times", "double charged"), unauthorized card debits, disputed refund amounts.
   - *Required Action*: Secure channel handoff to inspect billing records.
4. **`REPEATED_FAILED_CONTACT`**:
   - *Triggers*: Multi-channel failure reports ("called twice", "still waiting after 3 days", "no response from call center").
   - *Required Action*: Acknowledge prior inconvenience and transfer to priority support.
5. **`OUT_OF_POLICY_REQUEST`**:
   - *Triggers*: Demands for policy overrides, fee waivers, or manual compensation exceptions.
   - *Required Action*: Clear policy statement or human discretion routing.

### 2.2 Output Schema
```json
{
  "escalation": true,
  "reason": "PAYMENT_ACCOUNT_DISPUTE",
  "confidence": 0.94,
  "evidence": ["Payment dispute match: 'charged three times'"]
}
```

---

## 3. Conversation State Tracker (Transition Table)

| Current State | Trigger / Event | Next State | Rationale |
| :--- | :--- | :--- | :--- |
| `NONE` | Turn 1 inbound customer message | `STATE_INITIAL_INBOUND` | Standard initiation of dialogue. |
| `NONE` | Turn 1 customer message with severe threat or repeated contact | `STATE_CUSTOMER_ESCALATION` | Immediate escalation due to extreme inbound hostility. |
| `STATE_INITIAL_INBOUND` | Support asks for non-sensitive order identifier | `STATE_CLARIFICATION_REQUESTED` | Agent prompts customer for missing context. |
| `STATE_CLARIFICATION_REQUESTED` | Customer provides requested order ID or courier name | `STATE_CUSTOMER_PROVIDING_INFO` | Customer supplies missing details requested by agent. |
| `STATE_SECURE_HANDOFF_TRIGGERED` | Customer continues conversation in public thread | `STATE_CUSTOMER_PROVIDING_INFO` | Customer provides contextual followup in thread. |
| `ANY` | Customer confirms resolution ("thanks for speedy fix", "got it sorted") | `STATE_APPARENTLY_RESOLVED` | Customer explicitly declares issue resolved. |
| `ANY` | Customer rejects support response with hostility or repeat delay | `STATE_CUSTOMER_ESCALATION` | Conversation derailed into escalation. |
| `STATE_CUSTOMER_PROVIDING_INFO` | Agent provides troubleshooting instructions | `STATE_TROUBLESHOOTING_ACTIVE` | Ongoing diagnostic steps. |

---

## 4. Action Policy Engine & Disambiguation Rules

### 4.1 Disambiguation: `ASK_CLARIFICATION` vs. `REQUEST_SAFE_DETAILS`
- **`ASK_CLARIFICATION`**:
  - *Applicability*: The customer's core intent or goal is uncertain, ambiguous, or underspecified (e.g. *"What is the difference between these accounts?"*, or vague feedback).
  - *Goal*: Clarify what service the customer requires.
- **`REQUEST_SAFE_DETAILS`**:
  - *Applicability*: The intent is clear (e.g. `DELIVERY_STATUS_AND_TRACKING` or `PRODUCT_CONDITION_AND_WRONG_ITEM`) but a public identifier (e.g. carrier name, postal code) is needed to proceed.
  - *Goal*: Solicit non-sensitive tracking information.

### 4.2 Disambiguation: `PROVIDE_INFORMATION` vs. `OFFER_NEXT_STEP`
- **`PROVIDE_INFORMATION`**:
  - *Applicability*: Factual, policy, or knowledge-base questions (e.g. *"How does buyer-seller messaging work?"*, *"What is your return policy?"*).
  - *Goal*: Answer the customer's question directly.
- **`OFFER_NEXT_STEP`**:
  - *Applicability*: Action-oriented procedural workflows where the customer needs to execute an action (e.g. cancelling an in-flight order, initiating a return label).
  - *Goal*: Provide the specific step-by-step next action for the user to take.

---

## 5. Uncompromising Safety Boundaries

> [!CAUTION]
> **Authentication Secrets Guardrail**:
> The support agent operates in a public social channel (Twitter/X). Under **zero circumstances** may the agent solicit:
> - Passwords or passcodes
> - One-Time Passwords (OTPs)
> - Personal Identification Numbers (PINs)
> - Card Verification Values (CVV / CVC)
> - Full credit/debit card numbers
>
> When private account identification is required, the agent **must** execute `HANDOFF_TO_SECURE_CHANNEL` directing the customer to verified Amazon Direct Message or authenticated account portal.
