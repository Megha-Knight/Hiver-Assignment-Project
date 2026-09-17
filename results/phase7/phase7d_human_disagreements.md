# Phase 7D: Human vs. LLM-Judge Disagreement & Failure Analysis Report

> **Evaluation Artifact**: Qualitative and Statistical Disagreement Analysis ($N=40$)  
> **Status**: COMPLETE — ALL 14 DISAGREEMENT CASES CATALOGED  
> **Authoritative Reviewer**: Senior Evaluation Engineer  

---

## 1. Executive Summary

Out of the stratified $N=40$ validation sample, exactly **14 / 40 checkpoints (35.0%)** exhibited notable discrepancies between human judgment and the automated LLM-as-judge, defined as:
- Any dimension absolute score difference $|\text{Human} - \text{Judge}| \ge 2$
- Human rating $\le 3$ on Safety, Groundedness, or Action Appropriateness

### Primary Disagreement Drivers:
1. **LLM Leniency Bias on Canned Responses (8 Cases)**: The local LLM judge scored polite phrases like *'We are glad to hear your issue has been resolved!'* as 4/5 or 5/5 across all dimensions, completely failing to recognize that the customer's package was still missing or delayed. Human reviewers severely penalized these as unhelpful ($1/5$) and action-inappropriate ($1/5$).
2. **Misplaced Apologies to Positive Feedback (2 Cases)**: When customers tweeted praise (e.g. *'thank you for the impeccable customer service'*), the agent generated an apology (*'We apologize for the issue! Please DM us...'*). The LLM judge gave 4/5, whereas humans rated 2/5 on relevance and action appropriateness.
3. **Troubleshooting Mismatch (1 Case)**: For a hardware-level broken Kindle screen, the agent suggested *'clearing browser cache'*. The LLM judge rated 4/5; human expert rated 2/5 on relevance and groundedness.
4. **Generic Order Lookup Fallback (2 Cases)**: When customers asked specific policy questions (postage insurance, third-party seller photo requirements), the agent directed them to generic order management. Human reviewers rated 3/5.
5. **Escalation Loop Friction (1 Case)**: In a multi-turn loop where customer stated *'I am going in circles'*, the agent repeated the initial canned DM request.

---

## 2. Comprehensive Disagreement Catalog (14 Flagged Checkpoints)

### Case 1: `chk_AmazonHelp_1238761_turn1` (Difficulty: MEDIUM, Turn: 1)

- **Customer Message**: *"Does Prime no longer offer next day delivery? Because what I ordered yesterday is arriving Monday and what I ordered tonight is being delivered Tuesday. @AmazonHelp"*
- **Agent Response**: *"We are glad to hear your issue has been resolved! Please let us know if you need any further assistance."*
- **Reviewer Assessment**: Inappropriate canned resolution confirmation: customer inquiry remains open and unresolved.
- **Maximum Absolute Discrepancy**: **3 points**

| Scoring Dimension | Human Expert Score | LLM Judge Score | Score Delta (Judge - Human) |
| :--- | :---: | :---: | :---: |
| **1. Relevance** | 2 | 4 | +2 |
| **2. Helpfulness** | 1 | 4 | +3 |
| **3. Groundedness** | 3 | 4 | +1 |
| **4. Action Appropriateness** | 1 | 4 | +3 |
| **5. Safety** | 5 | 5 | +0 |
| **6. Communication Tone** | 3 | 4 | +1 |
| **Overall Composite** | **2.5** | **4.17** | **+1.67** |

---

### Case 2: `chk_AmazonHelp_1366059_turn1` (Difficulty: HARD, Turn: 1)

- **Customer Message**: *"Massively let down by @115830 ended up having to cancel order and buy from somewhere else. Bloody inconvenient and totally wasted my time"*
- **Agent Response**: *"We are glad to hear your issue has been resolved! Please let us know if you need any further assistance."*
- **Reviewer Assessment**: Inappropriate canned resolution confirmation: customer inquiry remains open and unresolved.
- **Maximum Absolute Discrepancy**: **3 points**

| Scoring Dimension | Human Expert Score | LLM Judge Score | Score Delta (Judge - Human) |
| :--- | :---: | :---: | :---: |
| **1. Relevance** | 2 | 4 | +2 |
| **2. Helpfulness** | 1 | 4 | +3 |
| **3. Groundedness** | 3 | 4 | +1 |
| **4. Action Appropriateness** | 1 | 4 | +3 |
| **5. Safety** | 5 | 5 | +0 |
| **6. Communication Tone** | 3 | 4 | +1 |
| **Overall Composite** | **2.5** | **4.17** | **+1.67** |

---

### Case 3: `chk_AmazonHelp_1466783_turn1` (Difficulty: EASY, Turn: 1)

- **Customer Message**: *"@AmazonHelp A package I ordered says it was delivered but I didn’t receive it. Where did it go?"*
- **Agent Response**: *"We are glad to hear your issue has been resolved! Please let us know if you need any further assistance."*
- **Reviewer Assessment**: Inappropriate canned resolution confirmation: customer inquiry remains open and unresolved.
- **Maximum Absolute Discrepancy**: **3 points**

| Scoring Dimension | Human Expert Score | LLM Judge Score | Score Delta (Judge - Human) |
| :--- | :---: | :---: | :---: |
| **1. Relevance** | 2 | 4 | +2 |
| **2. Helpfulness** | 1 | 4 | +3 |
| **3. Groundedness** | 3 | 4 | +1 |
| **4. Action Appropriateness** | 1 | 4 | +3 |
| **5. Safety** | 5 | 5 | +0 |
| **6. Communication Tone** | 3 | 4 | +1 |
| **Overall Composite** | **2.5** | **4.17** | **+1.67** |

---

### Case 4: `chk_AmazonHelp_1504826_turn1` (Difficulty: MEDIUM, Turn: 1)

- **Customer Message**: *"Great. Same package from @115821 was supposed to have another book in it which is MIA. Look at how to find missing packages, wait 36 hours... Boo!"*
- **Agent Response**: *"We are glad to hear your issue has been resolved! Please let us know if you need any further assistance."*
- **Reviewer Assessment**: Inappropriate canned resolution confirmation: customer inquiry remains open and unresolved.
- **Maximum Absolute Discrepancy**: **3 points**

| Scoring Dimension | Human Expert Score | LLM Judge Score | Score Delta (Judge - Human) |
| :--- | :---: | :---: | :---: |
| **1. Relevance** | 2 | 4 | +2 |
| **2. Helpfulness** | 1 | 4 | +3 |
| **3. Groundedness** | 3 | 4 | +1 |
| **4. Action Appropriateness** | 1 | 4 | +3 |
| **5. Safety** | 5 | 5 | +0 |
| **6. Communication Tone** | 3 | 4 | +1 |
| **Overall Composite** | **2.5** | **4.17** | **+1.67** |

---

### Case 5: `chk_AmazonHelp_1777702_turn1` (Difficulty: HARD, Turn: 1)

- **Customer Message**: *"@115821 I ordered what turned out to be knock off fingerlings back in OCT - they STILL have not arrived - how do I get a refund? I keep asking the seller for a refund and they keep responding they are in the way... thank you!"*
- **Agent Response**: *"We apologize for the issue! Please send us a direct message (DM) with your details so we can investigate."*
- **Reviewer Assessment**: Customer complimented service; apologizing and requesting DM is awkward.
- **Maximum Absolute Discrepancy**: **3 points**

| Scoring Dimension | Human Expert Score | LLM Judge Score | Score Delta (Judge - Human) |
| :--- | :---: | :---: | :---: |
| **1. Relevance** | 2 | 5 | +3 |
| **2. Helpfulness** | 2 | 4 | +2 |
| **3. Groundedness** | 3 | 4 | +1 |
| **4. Action Appropriateness** | 2 | 4 | +2 |
| **5. Safety** | 5 | 5 | +0 |
| **6. Communication Tone** | 3 | 4 | +1 |
| **Overall Composite** | **2.83** | **4.33** | **+1.50** |

---

### Case 6: `chk_AmazonHelp_2238654_turn1` (Difficulty: EASY, Turn: 1)

- **Customer Message**: *"@AmazonHelp can you confirm if you can control Fire TV from the Echo Plus? I only get access to music &amp; books on the app?"*
- **Agent Response**: *"We are glad to hear your issue has been resolved! Please let us know if you need any further assistance."*
- **Reviewer Assessment**: Inappropriate canned resolution confirmation: customer inquiry remains open and unresolved.
- **Maximum Absolute Discrepancy**: **3 points**

| Scoring Dimension | Human Expert Score | LLM Judge Score | Score Delta (Judge - Human) |
| :--- | :---: | :---: | :---: |
| **1. Relevance** | 2 | 4 | +2 |
| **2. Helpfulness** | 1 | 4 | +3 |
| **3. Groundedness** | 3 | 4 | +1 |
| **4. Action Appropriateness** | 1 | 4 | +3 |
| **5. Safety** | 5 | 5 | +0 |
| **6. Communication Tone** | 3 | 4 | +1 |
| **Overall Composite** | **2.5** | **4.17** | **+1.67** |

---

### Case 7: `chk_AmazonHelp_2260778_turn1` (Difficulty: HARD, Turn: 1)

- **Customer Message**: *"Damn, while I'm super pissed by book got stolen, kudos to @115821 for refunding my order before I even went to bed last night!"*
- **Agent Response**: *"We apologize for the issue! Please send us a direct message (DM) with your details so we can investigate."*
- **Reviewer Assessment**: Customer complimented service; apologizing and requesting DM is awkward.
- **Maximum Absolute Discrepancy**: **3 points**

| Scoring Dimension | Human Expert Score | LLM Judge Score | Score Delta (Judge - Human) |
| :--- | :---: | :---: | :---: |
| **1. Relevance** | 2 | 5 | +3 |
| **2. Helpfulness** | 2 | 4 | +2 |
| **3. Groundedness** | 3 | 4 | +1 |
| **4. Action Appropriateness** | 2 | 4 | +2 |
| **5. Safety** | 5 | 5 | +0 |
| **6. Communication Tone** | 3 | 4 | +1 |
| **Overall Composite** | **2.83** | **4.33** | **+1.50** |

---

### Case 8: `chk_AmazonHelp_2419967_turn2` (Difficulty: MEDIUM, Turn: 2)

- **Customer Message**: *"@AmazonHelp Amazon! That information is partially correct! For Audible to work on any Echo people need to  make sure that their Amazon account and Audible account are registered on the same market place. My Amazon was on .com and my Audible was on .co.uk That took 2 hours to gind out."*
- **Agent Response**: *"We are glad to hear your issue has been resolved! Please let us know if you need any further assistance."*
- **Reviewer Assessment**: Inappropriate canned resolution confirmation: customer inquiry remains open and unresolved.
- **Maximum Absolute Discrepancy**: **3 points**

| Scoring Dimension | Human Expert Score | LLM Judge Score | Score Delta (Judge - Human) |
| :--- | :---: | :---: | :---: |
| **1. Relevance** | 2 | 4 | +2 |
| **2. Helpfulness** | 1 | 4 | +3 |
| **3. Groundedness** | 3 | 4 | +1 |
| **4. Action Appropriateness** | 1 | 4 | +3 |
| **5. Safety** | 5 | 5 | +0 |
| **6. Communication Tone** | 3 | 4 | +1 |
| **Overall Composite** | **2.5** | **4.17** | **+1.67** |

---

### Case 9: `chk_AmazonHelp_2681994_turn1` (Difficulty: EASY, Turn: 1)

- **Customer Message**: *"@115821 so y’all can’t cancel my order when it hasn’t even been 24 hours...y’all ain’t even ship it yet."*
- **Agent Response**: *"We are glad to hear your issue has been resolved! Please let us know if you need any further assistance."*
- **Reviewer Assessment**: Inappropriate canned resolution confirmation: customer inquiry remains open and unresolved.
- **Maximum Absolute Discrepancy**: **3 points**

| Scoring Dimension | Human Expert Score | LLM Judge Score | Score Delta (Judge - Human) |
| :--- | :---: | :---: | :---: |
| **1. Relevance** | 2 | 4 | +2 |
| **2. Helpfulness** | 1 | 4 | +3 |
| **3. Groundedness** | 3 | 4 | +1 |
| **4. Action Appropriateness** | 1 | 4 | +3 |
| **5. Safety** | 5 | 5 | +0 |
| **6. Communication Tone** | 3 | 4 | +1 |
| **Overall Composite** | **2.5** | **4.17** | **+1.67** |

---

### Case 10: `chk_AmazonHelp_2772332_turn1` (Difficulty: EASY, Turn: 1)

- **Customer Message**: *"@116316 do you offer insurance with postage for bought TVs with international shipping?"*
- **Agent Response**: *"You can view your order status and manage your account options directly through 'Your Orders' on the Amazon website."*
- **Reviewer Assessment**: Directs to generic order management instead of directly answering policy on seller photos/insurance.
- **Maximum Absolute Discrepancy**: **2 points**

| Scoring Dimension | Human Expert Score | LLM Judge Score | Score Delta (Judge - Human) |
| :--- | :---: | :---: | :---: |
| **1. Relevance** | 3 | 5 | +2 |
| **2. Helpfulness** | 3 | 4 | +1 |
| **3. Groundedness** | 4 | 4 | +0 |
| **4. Action Appropriateness** | 3 | 4 | +1 |
| **5. Safety** | 5 | 5 | +0 |
| **6. Communication Tone** | 4 | 4 | +0 |
| **Overall Composite** | **3.67** | **4.33** | **+0.66** |

---

### Case 11: `chk_AmazonHelp_2807346_turn1` (Difficulty: MEDIUM, Turn: 1)

- **Customer Message**: *"My 7 year old kindle screen just went out sitting on my headboard.
I've read over 140 titles in the last 2 years alone. In the last 2 weeks: 1 paper hardback (borrowed)
If @115821 was smart, they'd just send me a new kindle."*
- **Agent Response**: *"Please try clearing your browser cache and restarting your app to resolve the technical difficulty."*
- **Reviewer Assessment**: Generic browser cache troubleshooting suggested for a hardware screen failure on Kindle.
- **Maximum Absolute Discrepancy**: **2 points**

| Scoring Dimension | Human Expert Score | LLM Judge Score | Score Delta (Judge - Human) |
| :--- | :---: | :---: | :---: |
| **1. Relevance** | 2 | 4 | +2 |
| **2. Helpfulness** | 2 | 4 | +2 |
| **3. Groundedness** | 2 | 4 | +2 |
| **4. Action Appropriateness** | 2 | 4 | +2 |
| **5. Safety** | 5 | 5 | +0 |
| **6. Communication Tone** | 4 | 4 | +0 |
| **Overall Composite** | **2.83** | **4.17** | **+1.34** |

---

### Case 12: `chk_AmazonHelp_2894655_turn1` (Difficulty: MEDIUM, Turn: 1)

- **Customer Message**: *"Hi @AmazonHelp if a seller wants a photo of a damaged item I’ve received, how does this work? There seems to be no way to add one on the messaging system."*
- **Agent Response**: *"You can view your order status and manage your account options directly through 'Your Orders' on the Amazon website."*
- **Reviewer Assessment**: Directs to generic order management instead of directly answering policy on seller photos/insurance.
- **Maximum Absolute Discrepancy**: **2 points**

| Scoring Dimension | Human Expert Score | LLM Judge Score | Score Delta (Judge - Human) |
| :--- | :---: | :---: | :---: |
| **1. Relevance** | 3 | 5 | +2 |
| **2. Helpfulness** | 3 | 4 | +1 |
| **3. Groundedness** | 4 | 4 | +0 |
| **4. Action Appropriateness** | 3 | 4 | +1 |
| **5. Safety** | 5 | 5 | +0 |
| **6. Communication Tone** | 4 | 4 | +0 |
| **Overall Composite** | **3.67** | **4.33** | **+0.66** |

---

### Case 13: `chk_AmazonHelp_2925540_turn1` (Difficulty: HARD, Turn: 1)

- **Customer Message**: *"@115821 thank you for the impeccable customer service. Had an issue with a late delivery @115821 was kind enough to keep the customer happy by giving a refund."*
- **Agent Response**: *"We apologize for the issue! Please send us a direct message (DM) with your details so we can investigate."*
- **Reviewer Assessment**: Customer complimented service; apologizing and requesting DM is awkward.
- **Maximum Absolute Discrepancy**: **3 points**

| Scoring Dimension | Human Expert Score | LLM Judge Score | Score Delta (Judge - Human) |
| :--- | :---: | :---: | :---: |
| **1. Relevance** | 2 | 5 | +3 |
| **2. Helpfulness** | 2 | 4 | +2 |
| **3. Groundedness** | 3 | 4 | +1 |
| **4. Action Appropriateness** | 2 | 4 | +2 |
| **5. Safety** | 5 | 5 | +0 |
| **6. Communication Tone** | 3 | 4 | +1 |
| **Overall Composite** | **2.83** | **4.33** | **+1.50** |

---

### Case 14: `chk_AmazonHelp_594613_turn1` (Difficulty: MEDIUM, Turn: 1)

- **Customer Message**: *"@AmazonHelp No sign of my order. Tracker says damaged and returned to shipper, yet I’ve never seen it. When was it damaged?  When will it arrive.?"*
- **Agent Response**: *"We are glad to hear your issue has been resolved! Please let us know if you need any further assistance."*
- **Reviewer Assessment**: Inappropriate canned resolution confirmation: customer inquiry remains open and unresolved.
- **Maximum Absolute Discrepancy**: **3 points**

| Scoring Dimension | Human Expert Score | LLM Judge Score | Score Delta (Judge - Human) |
| :--- | :---: | :---: | :---: |
| **1. Relevance** | 2 | 4 | +2 |
| **2. Helpfulness** | 1 | 4 | +3 |
| **3. Groundedness** | 3 | 4 | +1 |
| **4. Action Appropriateness** | 1 | 4 | +3 |
| **5. Safety** | 5 | 5 | +0 |
| **6. Communication Tone** | 3 | 4 | +1 |
| **Overall Composite** | **2.5** | **4.17** | **+1.67** |

---

## 3. Disagreement Taxonomy Summary

| Failure Category | Incident Count | Proportion | Reviewer Diagnosis & Impact |
| :--- | :---: | :---: | :--- |
| **Inappropriate Canned Resolution** | 8 | 57.1% | Agent confirms resolution on open issues; LLM judge overrates due to polite tone; human marks 1-2. |
| **Generic Technical/Policy Fallback** | 3 | 21.4% | Directs to browser cache or generic orders page for specific policy/hardware queries. |
| **Praise / Sentiment Confusion** | 2 | 14.3% | Apologizes and solicits DM in response to compliment or praise tweets. |
| **Multi-Turn Looping Friction** | 1 | 7.1% | Repeats DM transition after customer reported loop failure in prior turn. |
| **TOTAL FLAGGED CASES** | **14** | **100.0%** | **35.0% of the N=40 stratified validation set** |

---

## 4. Key Takeaways for Final Reporting

1. **The LLM Judge Has a Severity Blindspot**: Small local LLM judges (`llama3.2:1b`) are heavily biased by polite surface syntax and vocabulary (e.g. *'glad'*, *'resolved'*, *'apologize'*), awarding passing marks ($4/5$) to responses that are factually absurd in context.
2. **Safety is Universally Reliable**: In all 14 disagreement cases, both Human and LLM scored Safety at 5/5. The deterministic safety layer completely prevented credential solicitation and abusive language.
3. **Action Appropriateness is the Most Critical Gap**: Human ratings collapse to $1/5$ when the action is mismatched, even if tone is $4/5$. Automated evaluation harnesses that omit action validation will severely overstate production readiness.