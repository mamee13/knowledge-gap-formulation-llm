# Day 2 — Tweet Thread

**Tweet 1 — Introduction + The Question (280 chars max)**

Week 12 of TRP1 @10academy. Day 2 topic: agent tool-use internals.

My partner's agent hallucinated an order_id instead of asking "what's your order ID?" Why does a missing parameter trigger fabrication instead of a clarification question? Here's the token-level answer

---

**Tweet 2 — The Mechanism: The Schema Shifts Probability**

JSON tool schema in the system prompt shifts token probabilities toward <tool_call> before the user speaks.

It's a learned prior from RLHF: millions of training examples rewarded tool execution as task success. Not a decision—a statistical bias baked into the weights.

---

**Tweet 3 — The Mechanism: Constrained Decoding Locks the Path**

Once the model emits <tool_call>, grammar-constrained decoding activates.

The JSON schema compiles into a formal grammar. Every violating token gets masked to -∞.

"What is your order ID?" is now structurally impossible. The model MUST fill order_id with something.

---

**Tweet 4 — The Mechanism: Why Hallucination Is Inevitable**

The grammar requires order_id. The model samples from valid string tokens.

It produces a plausible-looking but fabricated ID.

This isn't a reasoning failure. It's the sampling process doing exactly what it was designed to do: produce a valid JSON completion.

---

**Tweet 5 — The Fix: Absorb the Bias, Don't Fight It**

Wrong fix: prompt engineering ("ask, don't assert") — fights a learned prior with text. Fragile.

Right fix: add ask_clarifying_question(question_text) as a tool.

Now the model satisfies its tool-call bias AND asks for the order_id. Absorb the bias, don't fight it.

---

**Tweet 6 — Production Implication + Link**

Any agent with destructive tools (cancel, delete, transfer) needs a clarification tool as a first-class schema citizen.

The model's execution bias is an asset once you give it a safe target.

Full explainer: [BLOG_URL_HERE]
