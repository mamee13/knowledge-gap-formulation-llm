# Why Your Agent Hallucinates Parameters Instead of Asking: The Token-Level Mechanics of Premature Execution Bias

Written by Mamaru Yirga | For Yonas Eshete's question

**Question I'm answering:**

"When a prospect asked 'cancel my last order', my agent lacked the required `order_id`. Instead of asking for it, the agent fabricated an ID and immediately fired the `cancel_pending_order` tool. What is happening at the token level during function-calling that makes generating a conversational clarification question so mathematically unlikely compared to forcing a tool call?"

---

## The Load-Bearing Mechanism

When your agent sees a tool schema in its context, it doesn't "decide" to call a tool the way a human decides. It generates tokens autoregressively, and the presence of a JSON tool schema in the system prompt shifts the probability distribution over the entire vocabulary at every subsequent generation step.

Here's what actually happens at the token level:

**Step 1 — Logit computation.** At each generation step, the model produces a vector of raw scores (logits) over its full vocabulary—roughly 150,000–200,000 tokens. These encode how strongly the model "wants" to produce each token next, given everything it has seen.

**Step 2 — The schema creates a gravitational pull.** Tool schemas are serialized into the prompt as structured JSON. The model has been trained on millions of examples where a JSON schema in context is followed by a `<tool_call>` control token and a valid JSON argument object. This creates a strong conditional probability: `P(<tool_call> | JSON schema in context)` is very high. The schema tokens act as a prior that shifts probability mass toward the tool-call pathway before the user even speaks.

**Step 3 — Constrained decoding locks in the path.** Once the model emits the `<tool_call>` control token, the inference engine activates grammar-constrained decoding. The JSON schema is compiled into a context-free grammar, and at every subsequent token step, any token that would violate the schema is masked to `-∞` probability. The model is now on a one-way track: it must produce a valid JSON object matching the schema. Generating "What is your order ID?" is no longer a valid token sequence—it's been structurally excluded.

**Step 4 — Missing parameters get hallucinated.** The grammar requires the `order_id` field. The model must fill it. It samples from whatever probability mass remains over valid string tokens, producing a plausible-looking but fabricated ID. This isn't a reasoning failure—it's the sampling process doing exactly what it was designed to do.

**Why clarification is suppressed:** A conversational clarification question like "What is your order ID?" requires the model to *not* emit the `<tool_call>` token. But the schema in context has already shifted probability mass heavily toward that token. The model would need to "resist" a strong learned prior to stay in conversational mode. Without explicit training signal rewarding clarification, it doesn't.

---

## Show It

The following demonstrates the token-level decision point. The critical fork happens at the very first generation step after the user's message:

```python
# Conceptual demonstration of the probability fork
# At the first generation step, the model chooses between two paths:

# PATH A: Conversational clarification
# Requires generating: "I" -> " need" -> " your" -> " order" -> " ID"
# P(this path) is LOW because JSON schema in context shifts mass away from prose

# PATH B: Tool call execution  
# Requires generating: <tool_call> -> {"order_id": ...}
# P(this path) is HIGH because:
#   1. Schema in context creates strong conditional prior
#   2. RLHF training rewarded tool execution as task completion
#   3. Once <tool_call> is emitted, constrained decoding forces valid JSON

# The fix: add ask_clarifying_question as a tool
# Now PATH C exists:
# <tool_call> -> {"question_text": "What is your order ID?"}
# P(this path) is HIGH (still a tool call, still valid JSON)
# And it produces the right behavior

tools = [
    {
        "name": "cancel_pending_order",
        "description": "Cancels a pending order. Requires order_id.",
        "parameters": {"order_id": {"type": "string", "required": True}}
    },
    {
        "name": "ask_clarifying_question",  # ← absorb the bias, don't fight it
        "description": "Ask the user for missing information before taking action.",
        "parameters": {"question_text": {"type": "string", "required": True}}
    }
]
# Now the model can satisfy its tool-call bias AND ask for the order_id
```

This is the "absorb, don't fight" pattern. Your SCAP postscript was trying to suppress the `<tool_call>` token with prompt instructions—fighting a learned prior with text. Adding `ask_clarifying_question` as a tool redirects the bias into a safe channel instead.

---

## Adjacent Concepts

**Why RLHF makes this worse.** Tool-use training data overwhelmingly consists of complete, successful tool calls. "Stalling" to ask a clarifying question is rarely labeled as the correct action—it looks like task failure to a reward model scoring on tool execution success. The model learned that emitting `<tool_call>` is rewarded; staying in conversational mode is not. This is the training-time origin of the bias your probes P023–P025 are measuring.

**Why prompt engineering loses.** Your SCAP postscript adds tokens saying "ask, don't assert." But these compete against a learned prior reinforced across millions of training examples. Schema tokens in the system prompt carry more statistical weight than postscript instructions. Prompt engineering can nudge probabilities but cannot override a structural training signal—which is why your τ²-Bench pass rate remained fragile.

**The structural fix generalizes.** The `ask_clarifying_question` tool pattern works because it doesn't fight the model's bias—it routes the bias into a safe action. Any FDE engagement with destructive tools (delete, cancel, transfer, submit) should include a clarification tool as a first-class citizen in the schema. The model's tendency to execute is an asset once you give it a safe execution target.

---

## Pointers

- **Paper 1:** Schick, T., et al. (2023). [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761). Describes how tool-use behavior is trained into models via self-supervised API call examples—the origin of the execution bias.

- **Paper 2:** Zhang, X., et al. (2025). [AskToAct: Enhancing LLMs Tool Use via Self-Correcting Clarification](https://arxiv.org/abs/2503.01940). Directly addresses the premature execution problem; shows that adding clarification as a structured tool action outperforms prompt-based approaches by 10.46% on clarification efficiency.

- **Tool/pattern used:** Constrained decoding pipeline analysis based on [Function Calling Internals: Grammars and Constrained Sampling](https://www.salmanq.com/blog/llm-constrained-sampling/) — traces the exact token-level pipeline from logit computation through grammar-constrained sampling.

- **Further reading:** The AskToAct paper's key insight—that tool parameters represent explicit user intents, so missing parameters are the signal to clarify—is directly applicable to your `cancel_pending_order` case.

---

*All technical claims trace to cited canonical sources or runnable code in this repository.*
