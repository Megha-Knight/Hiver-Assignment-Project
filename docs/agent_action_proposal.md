# Agent Action Space Proposal

> **Minimal, Deterministic Agent Action Space for Autonomous Multi-Turn Support**  
> *Derived from Real Twitter Customer Support Patterns across AmazonHelp*

---

## 1. Action Space Philosophy

To prevent hallucination, out-of-scope commitments, and boundary violations, an autonomous customer support agent operating on open or social channels must not generate unstructured text without an explicit behavioral intent.

This proposal establishes a **compact, measurable 8-action taxonomy**. Every model response in future phases will be conditioned on selecting one primary action (or an action pair) from this space.

---

## 2. Action Taxonomy Summary

| Action Code | Action Name | Primary Purpose | Autonomy Level |
| :---: | :--- | :--- | :---: |
| **ACT-01** | `PROVIDE_INFORMATION` | Share public store policies, FAQ answers, or timelines | Fully Autonomous |
| **ACT-02** | `ASK_CLARIFICATION` | Resolve ambiguous queries by asking focused scoping questions | Fully Autonomous |
| **ACT-03** | `REQUEST_SAFE_DETAILS` | Request non-PII parameters (e.g., carrier name, marketplace) | Fully Autonomous |
| **ACT-04** | `PROVIDE_TROUBLESHOOTING` | Deliver procedural troubleshooting steps for devices/apps | Fully Autonomous |
| **ACT-05** | `OFFER_NEXT_STEP` | Guide customer to self-service portals (e.g. "Your Orders") | Fully Autonomous |
| **ACT-06** | `HANDOFF_TO_SECURE_CHANNEL` | Route customer to private DM or authenticated account link | Policy-Forced Escalation |
| **ACT-07** | `EMPATHIZE_AND_DEESCALATE` | Calm frustrated customer and prepare for priority handling | Semi-Autonomous + Handoff |
| **ACT-08** | `CONFIRM_RESOLUTION` | Acknowledge customer satisfaction and close dialogue | Fully Autonomous |

---

## 3. Detailed Action Specifications

### 1. `PROVIDE_INFORMATION`
- **Definition**: Delivering factual, verified information regarding Amazon policies, standard shipping windows, holiday schedules, or general capabilities.
- **When Allowed**: The customer inquiry is general, policy-oriented, or refers to standard procedures (e.g. "How long does standard delivery take to Alaska?").
- **When Forbidden**: The customer is asking for account-specific balance figures, order-specific delivery dates that require backend order lookup, or confidential internal seller metrics.
- **Real Examples**:
  - *"Standard international shipping to Germany typically takes 5–8 business days via AmazonGlobal."*
  - *"You can stream Prime Video on up to 3 devices simultaneously using the same Amazon account."*

---

### 2. `ASK_CLARIFICATION`
- **Definition**: Asking targeted scoping questions to disambiguate an unclear customer complaint without requesting specific IDs.
- **When Allowed**: The issue description lacks sufficient context to identify the root cause (e.g., "the video won't play" $\rightarrow$ ask which device and title).
- **When Forbidden**: The customer has already provided all necessary information; or when clarification is used as a stalling tactic.
- **Real Examples**:
  - *"Are you experiencing this playback error on your web browser or through the Prime Video Smart TV app?"*
  - *"To help us narrow this down, which marketplace (.com, .co.uk, or .in) was this order placed on?"*

---

### 3. `REQUEST_SAFE_DETAILS`
- **Definition**: Soliciting non-sensitive, public-safe operational identifiers necessary for triage.
- **When Allowed**: Requesting public tracking numbers, carrier names, device firmware versions, or error code numbers.
- **When Forbidden**: **CRITICAL VIOLATION**: Never request email addresses, phone numbers, home addresses, passwords, credit card numbers, or OTP codes in public!
- **Real Examples**:
  - *"Could you share the courier name and the public tracking number provided in your dispatch email?"*
  - *"What exact error message or code appears on your Echo screen during setup?"*

---

### 4. `PROVIDE_TROUBLESHOOTING`
- **Definition**: Providing clear, step-by-step diagnostic procedures for self-service problem resolution.
- **When Allowed**: Digital device issues (Kindle, Echo, Fire TV), app sync issues, browser cache problems, or basic hardware restarts.
- **When Forbidden**: Hardware has physical liquid/impact damage; or the customer has already performed these exact steps twice.
- **Real Examples**:
  - *"Let's try resetting the app: go to Settings > Applications > Manage Installed Applications > Prime Video, then select 'Clear Cache' and restart your device."*
  - *"Please hold down the power button on your Kindle Paperwhite for a full 40 seconds until the screen flashes, then release."*

---

### 5. `OFFER_NEXT_STEP`
- **Definition**: Instructing the customer on how to initiate actions themselves within their authenticated Amazon account interface.
- **When Allowed**: Actions supported by self-service (e.g. printing a return label from "Your Orders", updating an address prior to dispatch, canceling an unshipped item).
- **When Forbidden**: The order has already entered dispatch and self-service cancellation is disabled; or when the customer's account is locked.
- **Real Examples**:
  - *"You can initiate a return label directly by visiting 'Your Orders' > select the item > click 'Return or Replace Items'."*
  - *"To update your delivery address, visit 'Your Orders' before dispatch and click 'Change Shipping Address'."*

---

### 6. `HANDOFF_TO_SECURE_CHANNEL`
- **Definition**: Explicitly directing the customer away from the public thread into a private, authenticated channel (Twitter DM or verified Amazon customer service portal).
- **When Allowed**: **MANDATORY** whenever handling refunds, order cancellations requiring agent action, payment disputes, account security/lockouts, or PII.
- **When Forbidden**: Simple public FAQ questions where no account data is required.
- **Real Examples**:
  - *"Because we'll need to look into your specific order details securely, please send us a Direct Message with your order number: https://amzn.to/dm-help. ^AB"*
  - *"To protect your account privacy, please connect with our secure account specialist team directly here: https://amzn.to/contact-us. ^MK"*

---

### 7. `EMPATHIZE_AND_DEESCALATE`
- **Definition**: Validating customer frustration, apologizing for service failure, and ensuring priority attention before immediate routing.
- **When Allowed**: Customer displays strong anger, reports multiple failed contacts, threatens escalation (BBB, legal), or reports lost high-value shipments.
- **When Forbidden**: Overusing apologetic boilerplate without providing an actionable resolution pathway.
- **Real Examples**:
  - *"I completely understand your frustration with this delay, especially after you were promised delivery yesterday. Let's get this escalated immediately via DM so we can make this right. ^SJ"*
  - *"We sincerely apologize for the condition this parcel arrived in—this is certainly not the standard we aim for. Please DM us your details so we can arrange an immediate replacement. ^CB"*

---

### 8. `CONFIRM_RESOLUTION`
- **Definition**: Formally acknowledging the customer's positive confirmation that their issue is resolved and offering a polite parting sign-off.
- **When Allowed**: Customer says "thank you", "that fixed it", or confirms they are all set.
- **When Forbidden**: The customer is still asking questions or expresses residual doubt.
- **Real Examples**:
  - *"You're very welcome! Glad we could get that sorted for you. Have a wonderful rest of your day! ^GS"*
  - *"Happy to help! Don't hesitate to reach back out if anything else comes up. :) ^MK"*
