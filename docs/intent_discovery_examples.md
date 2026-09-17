# Intent Discovery & Cluster Exploration Artifact

> **Unsupervised Semantic Clustering Analysis of Customer Initial Turns**  
> *Model: `all-MiniLM-L6-v2` | Clustered Sample Size: 10,000 | Clusters: 12*

---

## 1. Executive Summary of Discovered Clusters

| Cluster ID | Suggested Theme | Sample Count | % Share | Distinguishing Keywords |
| :---: | :--- | :---: | :---: | :--- |
| 0 | **Order Tracking & Delivery Delays** | 4,987 | 49.87% | `amazon`, `prime`, `package`, `delivered`, `just` |
| 10 | **Order Tracking & Delivery Delays** | 1,168 | 11.68% | `delivery`, `day`, `prime`, `day delivery`, `amazon` |
| 2 | **Order Tracking & Delivery Delays** | 1,029 | 10.29% | `order`, `amazon`, `delivered`, `order id`, `help` |
| 5 | **Order Tracking & Delivery Delays** | 543 | 5.43% | `ordered`, `amazon`, `delivered`, `prime`, `pre ordered` |
| 1 | **Order Tracking & Delivery Delays** | 453 | 4.53% | `customer service`, `customer`, `service`, `amazon`, `worst customer` |
| 11 | **Order Tracking & Delivery Delays** | 387 | 3.87% | `service`, `amazon`, `delivery`, `order`, `worst service` |
| 6 | **Account Access & Login Security** | 380 | 3.8% | `account`, `amazon`, `amazon account`, `email`, `bank account` |
| 8 | **Order Tracking & Delivery Delays** | 309 | 3.09% | `hey`, `guys`, `hey guys`, `package`, `prime` |
| 3 | **Order Tracking & Delivery Delays** | 277 | 2.77% | `parcel`, `parcel delivered`, `delivered`, `delivery`, `today` |
| 4 | **Order Tracking & Delivery Delays** | 231 | 2.31% | `says`, `says delivered`, `delivered`, `order`, `tracking says` |
| 9 | **Order Cancellation & Address Modification** | 138 | 1.38% | `book`, `ordered book`, `ordered`, `order`, `amazon` |
| 7 | **Prime Membership & Digital Streaming** | 98 | 0.98% | `video`, `prime video`, `prime`, `amazon video`, `movies` |

---

## 2. Detailed Cluster Inspections & Real Multi-Turn Dialogue Exemplars

### Cluster 0: Order Tracking & Delivery Delays

- **Discovered Size**: 4,987 conversations (49.87% of analyzed sample)
- **Top TF-IDF Salient Terms**: `amazon`, `prime`, `package`, `delivered`, `just`, `day`, `help`, `product`
- **Empirical Customer Messages (Centroid Exemplars)**:

  1. *"@119625 Why no Malayalam films ?"* (Tweet ID: `429933`)
  2. *".@116316 ist mal wieder on fire. https://t.co/lUURAjsRKQ"* (Tweet ID: `17887`)
  3. *"@115850 ஆபர்னு விலை கமிபண்ணுவாங்க, நீ என்ன அதிகம் பண்ணிருக்க?"* (Tweet ID: `180536`)
  4. *"@116618 do you have the redzone on sundays too?"* (Tweet ID: `266017`)
  5. *"@115850  Can you please call me on 9980131010"* (Tweet ID: `130819`)

- **Representative Multi-Turn Conversation Thread**:

> **Conversation ID**: `conv_AmazonHelp_429933`  
> **Resolution Status**: `APPARENTLY_RESOLVED` | **Turns**: 6
>
> - **Customer**: @119625 Why no Malayalam films ?
> - **AmazonHelp Support**: @217247 We launched Prime Video today. Regional content is an integral part of our content in India. (1/2) ^CB
> @217247 Please stay tuned as we expand our content selection.(2/2) ^CB
> - **Customer**: @AmazonHelp I'm a member of Amazon prime and I'm looking forward to some good regional content . . .
> - **AmazonHelp Support**: @217247 I'll be sure to pass your comments as feedback to our concerned team for review. ^GS
> - **Customer**: @AmazonHelp Thank you.
> - **AmazonHelp Support**: @217247 You're welcome. :) ^GS

---

### Cluster 10: Order Tracking & Delivery Delays

- **Discovered Size**: 1,168 conversations (11.68% of analyzed sample)
- **Top TF-IDF Salient Terms**: `delivery`, `day`, `prime`, `day delivery`, `amazon`, `today`, `order`, `delivered`
- **Empirical Customer Messages (Centroid Exemplars)**:

  1. *"@115850 , now it's after delivery alerts. Haha https://t.co/htMKNrjvJf"* (Tweet ID: `234864`)
  2. *"@115830 is this an appropriate delivery tactic? #NotHappy 🤬 https://t.co/U7DoMqF6go"* (Tweet ID: `633703`)
  3. *"When both @115821 and @115817 screw up your delivery 📦 and it’s the Wednesday before thanksgiving 🤬"* (Tweet ID: `669708`)
  4. *"@AmazonHelp I'm expecting a delivery by 1pm from you. No sign of it."* (Tweet ID: `581872`)
  5. *"@115830 
Is it normal for your delivery staff to enter someone’s property without their knowledge or permission?"* (Tweet ID: `629644`)

- **Representative Multi-Turn Conversation Thread**:

> **Conversation ID**: `conv_AmazonHelp_234864`  
> **Resolution Status**: `UNKNOWN` | **Turns**: 2
>
> - **Customer**: @115850 , now it's after delivery alerts. Haha https://t.co/htMKNrjvJf
> - **AmazonHelp Support**: @170977 We'd like to check this, kindly connect with us here: https://t.co/vlvfJr4nN9 ^SG

---

### Cluster 2: Order Tracking & Delivery Delays

- **Discovered Size**: 1,029 conversations (10.29% of analyzed sample)
- **Top TF-IDF Salient Terms**: `order`, `amazon`, `delivered`, `order id`, `help`, `refund`, `order delivered`, `order 408`
- **Empirical Customer Messages (Centroid Exemplars)**:

  1. *"@AmazonHelp Where is my order? https://t.co/pXnKSCo2ex"* (Tweet ID: `2565`)
  2. *"@AmazonHelp where's my order? 026-5974794-2685140"* (Tweet ID: `420039`)
  3. *"Can you order puppies on @115821 ?"* (Tweet ID: `678278`)
  4. *"Can I order motivation on @115821 ?"* (Tweet ID: `517610`)
  5. *"@115850 
Where is my order ? https://t.co/7OKZjbdcPR"* (Tweet ID: `669198`)

- **Representative Multi-Turn Conversation Thread**:

> **Conversation ID**: `conv_AmazonHelp_2565`  
> **Resolution Status**: `UNKNOWN` | **Turns**: 2
>
> - **Customer**: @AmazonHelp Where is my order? https://t.co/pXnKSCo2ex
> - **AmazonHelp Support**: @116319 I'm sorry it hasn't arrived! Please reach us here: https://t.co/e6dQwzp386, so we can look into available options. ^SJ

---

### Cluster 5: Order Tracking & Delivery Delays

- **Discovered Size**: 543 conversations (5.43% of analyzed sample)
- **Top TF-IDF Salient Terms**: `ordered`, `amazon`, `delivered`, `prime`, `pre ordered`, `delivery`, `product`, `received`
- **Empirical Customer Messages (Centroid Exemplars)**:

  1. *"@115821 this is not what I ordered... https://t.co/4jRJ4cdqA4"* (Tweet ID: `570829`)
  2. *"Will I ever get the Exia Repair II I ordered from @115821 Thread"* (Tweet ID: `619979`)
  3. *"It's official ordered on @115821  @116 so excited!!!"* (Tweet ID: `404648`)
  4. *"@AmazonHelp it was amazonuk I ordered from"* (Tweet ID: `432969`)
  5. *"Uhhhh why is this turkey half frozen? I ordered a fresh turkey @117093"* (Tweet ID: `671289`)

- **Representative Multi-Turn Conversation Thread**:

> **Conversation ID**: `conv_AmazonHelp_570829`  
> **Resolution Status**: `UNRESOLVED` | **Turns**: 3
>
> - **Customer**: @115821 this is not what I ordered... https://t.co/4jRJ4cdqA4
> - **AmazonHelp Support**: @254055 Looks purrfect to us! ^LM
> - **Customer**: @AmazonHelp It is. Well played Amazon. :)

---

### Cluster 1: Order Tracking & Delivery Delays

- **Discovered Size**: 453 conversations (4.53% of analyzed sample)
- **Top TF-IDF Salient Terms**: `customer service`, `customer`, `service`, `amazon`, `worst customer`, `worst`, `order`, `delivery`
- **Empirical Customer Messages (Centroid Exemplars)**:

  1. *"@115821 @AmazonHelp Your customer service is absolute trash"* (Tweet ID: `457727`)
  2. *"@115850 tremendous improvements in customer service.."* (Tweet ID: `689166`)
  3. *"@115830 your customer service is disgraceful!"* (Tweet ID: `596828`)
  4. *"@115830 customer service keep putting me on hold!!!!"* (Tweet ID: `28460`)
  5. *"Most awful customer service with @115830"* (Tweet ID: `570104`)

- **Representative Multi-Turn Conversation Thread**:

> **Conversation ID**: `conv_AmazonHelp_457727`  
> **Resolution Status**: `APPARENTLY_RESOLVED` | **Turns**: 11
>
> - **Customer**: @115821 @AmazonHelp Your customer service is absolute trash
> - **AmazonHelp Support**: @223699 I'm sorry for the poor experience. Without providing personal or account information, could you tell us a bit about what's going on? ^DG
> - **Customer**: @AmazonHelp Yea, I left specific delivery instructions WITH A GATE access code and they said they could not deliver because they didn't have a gate access code.
> - **AmazonHelp Support**: @223699 Oh no! When you spoke to customer service, what information or options were we able to provide? ^EA
> - **Customer**: @AmazonHelp They said that they were not able to do anything as they cannot call the drivers or contact them so they forwarded my message to some department which we both know will never fix anything. I am very disappointed as I needed my package today which is why I have prime!!!
> - **AmazonHelp Support**: @223699 Thank you for the information. Can you confirm for us the carrier that was set to deliver the order here: https://t.co/q4LAMZ3tbE? ^GG
> - **Customer**: @AmazonHelp It says "Shipped with AMZL US"
> - **AmazonHelp Support**: @223699 Thank you for the information! We would like to look into this with you in real-time. Please contact us here: https://t.co/1Sjq3o5FxX ^LS
> - **Customer**: @AmazonHelp Alright, I filled out that form but it's really horrible that the package I was supposed to get today is now delayed for MULTIPLE days.... Very very disappointed and I suggest you guys look into the driver that was incapable of reading directions.
> - **Customer**: @AmazonHelp It has been dealt with the driver came back and delivered it today, I guess I was wrong thank you!
> - **AmazonHelp Support**: @223699 Thanks for keeping us in the loop! If you need anything else, don't hesitate to let us know. We all hope you enjoy the rest of the night. ^EP

---

### Cluster 11: Order Tracking & Delivery Delays

- **Discovered Size**: 387 conversations (3.87% of analyzed sample)
- **Top TF-IDF Salient Terms**: `service`, `amazon`, `delivery`, `order`, `worst service`, `worst`, `prime`, `pathetic`
- **Empirical Customer Messages (Centroid Exemplars)**:

  1. *"I'M not satisfied with the service @115821"* (Tweet ID: `133602`)
  2. *". @120534 A ÉVITER DE TOUTES LES FAÇONS. RETARDS, PERTE DE COLIS, SERVICE CLIENT INEXISTANT cc @120533"* (Tweet ID: `390419`)
  3. *"@116618 You're down - no service, no videos."* (Tweet ID: `510304`)
  4. *"This is wht @115850 is all abt if they do not provide d service later they vl #block u so tht u dnt hv to wry d #bully service frm #amazon https://t.co/EEVoK7ONyg"* (Tweet ID: `389982`)
  5. *"@AmazonHelp have provided terrible service"* (Tweet ID: `562921`)

- **Representative Multi-Turn Conversation Thread**:

> **Conversation ID**: `conv_AmazonHelp_133602`  
> **Resolution Status**: `UNKNOWN` | **Turns**: 4
>
> - **Customer**: I'M not satisfied with the service @115821
> - **AmazonHelp Support**: @146119 Sorry to know you're not satisfied with our services. Could you please let us know what went wrong? ^SY
> - **Customer**: @AmazonHelp The main reason is that you never send the thing which I have ordered and the product is damaged.
> - **AmazonHelp Support**: @146119 I'm sorry to know that the product you've received wasn't as expected.  Kindly report this to our support team here:https://t.co/vlvfJr4nN9 &amp; we'll have this checked right away. Also, I'll pass on your feedback internally to ensure such instances aren't repeated. ^EM

---

### Cluster 6: Account Access & Login Security

- **Discovered Size**: 380 conversations (3.8% of analyzed sample)
- **Top TF-IDF Salient Terms**: `account`, `amazon`, `amazon account`, `email`, `bank account`, `prime`, `help`, `hacked`
- **Empirical Customer Messages (Centroid Exemplars)**:

  1. *"@AmazonHelp I can't log into my account, it's telling me that amazon cannot find an account with my email"* (Tweet ID: `179994`)
  2. *"@115830 my account has been hacked and I can't log in !"* (Tweet ID: `249106`)
  3. *"@115830 Who’s account is this that you have stored under my username and that I can access through TouchID???? Where is my account??? https://t.co/JoF4oCDaI8"* (Tweet ID: `215724`)
  4. *"@115850 What if my email account i had given to amazon is shut off.Will my amazon account be disabled?"* (Tweet ID: `231723`)
  5. *"@115830 If an e mail account has been hacked and can't be accessed, how else can you retrieve the link to reset Amazon password?"* (Tweet ID: `73574`)

- **Representative Multi-Turn Conversation Thread**:

> **Conversation ID**: `conv_AmazonHelp_179994`  
> **Resolution Status**: `APPARENTLY_RESOLVED` | **Turns**: 4
>
> - **Customer**: @AmazonHelp I can't log into my account, it's telling me that amazon cannot find an account with my email
> - **AmazonHelp Support**: @158366 And do you have any other email address' that you may have used? ^AS
> - **Customer**: @AmazonHelp I phoned support directly and they're already looking into this for me, thanks
> - **AmazonHelp Support**: @158366 Great - let us know if you have any further issues ^TD

---

### Cluster 8: Order Tracking & Delivery Delays

- **Discovered Size**: 309 conversations (3.09% of analyzed sample)
- **Top TF-IDF Salient Terms**: `hey`, `guys`, `hey guys`, `package`, `prime`, `order`, `amazon`, `hey amazon`
- **Empirical Customer Messages (Centroid Exemplars)**:

  1. *"Hey @115850 what about the #iphone8appquiz ? https://t.co/LEvzv9QUS9"* (Tweet ID: `18395`)
  2. *"Hey @115850 do you give sponsorships to college fests?"* (Tweet ID: `256661`)
  3. *"Hey @119625 director's cut or theatre cut ?? https://t.co/lrKH9GJEe5"* (Tweet ID: `295819`)
  4. *"Hey @AmazonHelp where the heck is my Beyoncé LP that I preordered back in June?"* (Tweet ID: `539784`)
  5. *"Hey @115830 sad to see this dumped in a lane near me...#santaslittlehelpers https://t.co/5WEwIl0nKt"* (Tweet ID: `582664`)

- **Representative Multi-Turn Conversation Thread**:

> **Conversation ID**: `conv_AmazonHelp_18395`  
> **Resolution Status**: `UNKNOWN` | **Turns**: 2
>
> - **Customer**: Hey @115850 what about the #iphone8appquiz ? https://t.co/LEvzv9QUS9
> - **AmazonHelp Support**: @120087 The result for the contest is out and here's the link: https://t.co/MGupOOYvHT ^AR

---

### Cluster 3: Order Tracking & Delivery Delays

- **Discovered Size**: 277 conversations (2.77% of analyzed sample)
- **Top TF-IDF Salient Terms**: `parcel`, `parcel delivered`, `delivered`, `delivery`, `today`, `parcel today`, `deliver parcel`, `left`
- **Empirical Customer Messages (Centroid Exemplars)**:

  1. *"@AmazonHelp where is my parcel!!!"* (Tweet ID: `480313`)
  2. *"@AmazonHelp where's me parcel? https://t.co/i43IfcEZs0"* (Tweet ID: `302617`)
  3. *"Nice one @115830 delivered my parcel to a neighbour but no indication which neighbour! #huntthedelivery"* (Tweet ID: `163473`)
  4. *"When your @115830 parcel is flagged as 'delivered' but clearly not to you... 😒"* (Tweet ID: `229801`)
  5. *"@115821 🤔surely you ASK for the persons name before passing a parcel to someone?🤷‍♀️”delivered to home owner”erm no..You must be employing lazy staff for the holidays because whoever you delivered my dads parcel to..was not him🤦‍♀️ #resolvethis #CustomerService #notgood"* (Tweet ID: `448789`)

- **Representative Multi-Turn Conversation Thread**:

> **Conversation ID**: `conv_AmazonHelp_480313`  
> **Resolution Status**: `UNRESOLVED` | **Turns**: 11
>
> - **Customer**: @AmazonHelp where is my parcel!!!
> - **AmazonHelp Support**: @229154 Hi, sorry to hear that you haven't received your parcel yet. What's the current status of the tracking and estimated delivery date: https://t.co/aaDyEz1VgE? ^JJ
> - **Customer**: @AmazonHelp It was meant to be here on Tuesday! I pay you for amazon prime and your next day delivery still isn’t here 4 days later. Just says delayed
> - **Customer**: @AmazonHelp Was expected Tuesday, then Wednesday. Now this!! Was told I would receive a call back and an email and had nothing! Happy to take my money though. Nowhere to complain and no form of compensation still! https://t.co/t3GWeb3L1e
> - **AmazonHelp Support**: @229154 Apologies, was the order sold and fulfilled by Amazon or a 3rd Party Seller? ^JJ
> - **Customer**: @AmazonHelp It was all through amazon. Hence I was promised next day delivery
> - **AmazonHelp Support**: @229154 We'd like to review your order and available options. When you have a moment, contact us here: https://t.co/JzP7hlA23B ^RA
> - **Customer**: @AmazonHelp I’ve sent it. Can someone respond ASAP as I’ve waited long enough
> - **Customer**: @AmazonHelp Hello?????
> - **AmazonHelp Support**: @229154 We recommend contacting us via phone for the quickest response. ^EP
> - **Customer**: @AmazonHelp No. Reply to my message

---

### Cluster 4: Order Tracking & Delivery Delays

- **Discovered Size**: 231 conversations (2.31% of analyzed sample)
- **Top TF-IDF Salient Terms**: `says`, `says delivered`, `delivered`, `order`, `tracking says`, `package`, `tracking`, `amazon`
- **Empirical Customer Messages (Centroid Exemplars)**:

  1. *"@AmazonHelp My order says it's delivered, but I can't locate it. The tracking says back porch - but my porch is gated?"* (Tweet ID: `347369`)
  2. *"@115850 Please verify your offers. The advertisement says that for a minimum cart value of nivea products should be 200 or more but your steps says it has to reach minimum of 299. https://t.co/qoks3pPPoC"* (Tweet ID: `474185`)
  3. *"My floor lamp from @115821 says it was delivered but not here :(  I needed it for my fourth hue light"* (Tweet ID: `505034`)
  4. *"My @115821 order says delivered but it’s no where to be found #sos https://t.co/IB2Fkn25ou"* (Tweet ID: `304545`)
  5. *"@116618 I found a site that says you take submissions of film scripts to produce and meet about...is this real?"* (Tweet ID: `434478`)

- **Representative Multi-Turn Conversation Thread**:

> **Conversation ID**: `conv_AmazonHelp_347369`  
> **Resolution Status**: `UNKNOWN` | **Turns**: 2
>
> - **Customer**: @AmazonHelp My order says it's delivered, but I can't locate it. The tracking says back porch - but my porch is gated?
> - **AmazonHelp Support**: @198727 I'm sorry the order hasn't been located! We can look into this. We're available 24/7 here: https://t.co/hApLpMlfHN ^EP

---

### Cluster 9: Order Cancellation & Address Modification

- **Discovered Size**: 138 conversations (1.38% of analyzed sample)
- **Top TF-IDF Salient Terms**: `book`, `ordered book`, `ordered`, `order`, `amazon`, `pre`, `help`, `pre order`
- **Empirical Customer Messages (Centroid Exemplars)**:

  1. *"@115821 WHERE IS MY @26040 BOOK????"* (Tweet ID: `564878`)
  2. *"@115850 are you sending this book in a gold wrapping paper? https://t.co/BLLgxA4ZsT"* (Tweet ID: `526190`)
  3. *"been checking on my Book i Ordered with @115830"* (Tweet ID: `673257`)
  4. *"@115850  I had ordered a book on 27th Aug but it still shows as dispatched https://t.co/PdVIFjZ1Ws"* (Tweet ID: `319160`)
  5. *"Disappointed in @115821 shipped my book with some picture hooks I ordered and just threw them into the envelope too, damaging the book. https://t.co/XFEAZB2hQB"* (Tweet ID: `413998`)

- **Representative Multi-Turn Conversation Thread**:

> **Conversation ID**: `conv_AmazonHelp_564878`  
> **Resolution Status**: `UNKNOWN` | **Turns**: 2
>
> - **Customer**: @115821 WHERE IS MY @26040 BOOK????
> - **AmazonHelp Support**: @252338 I'm so sorry for the trouble! Can you please confirm the delivery date provided on your order: https://t.co/Y5jpI9gRhE? ^ST

---

### Cluster 7: Prime Membership & Digital Streaming

- **Discovered Size**: 98 conversations (0.98% of analyzed sample)
- **Top TF-IDF Salient Terms**: `video`, `prime video`, `prime`, `amazon video`, `movies`, `amazon`, `video app`, `tv`
- **Empirical Customer Messages (Centroid Exemplars)**:

  1. *"@13889 I own an LG so called smart TV. Totally disappointed that I can't play @115850 prime video as it doesn't support flash"* (Tweet ID: `273622`)
  2. *"@119625 You should probably consider adding playback speed to the Prime video player."* (Tweet ID: `697968`)
  3. *"@119625 is the Lucifer has been removed from prime video"* (Tweet ID: `361037`)
  4. *"@119625 any shows coming on prime video especially u.s tv shows as you are only adding mostly bollywood content not foreugn content"* (Tweet ID: `385077`)
  5. *"Okay. So @115821 has commercials in DENMARK for Prime Video, but doesn't have a danish site. Neither UK nor US site will allow me to watch Prime Video. So is that a fake commercial Amazon?"* (Tweet ID: `563945`)

- **Representative Multi-Turn Conversation Thread**:

> **Conversation ID**: `conv_AmazonHelp_273622`  
> **Resolution Status**: `UNKNOWN` | **Turns**: 2
>
> - **Customer**: @13889 I own an LG so called smart TV. Totally disappointed that I can't play @115850 prime video as it doesn't support flash
> - **AmazonHelp Support**: @181369 Could you please let us know more details regarding the model of the TV? (2/2)^AP
> @181369 You can watch Prime Video on 2015, 2016 and all latest smart TVs from Samsung, Sony, and LG. (1/2)^AP

---

