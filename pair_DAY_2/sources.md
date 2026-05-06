# Day 2 — Sources

**Canonical sources:**

1. **Toolformer: Language Models Can Teach Themselves to Use Tools** — Schick, T., et al. (2023) — https://arxiv.org/abs/2302.04761
   - Describes the self-supervised training pipeline that teaches models when and how to call tools. Explains the origin of the execution bias: models are trained on examples where tool calls produce better next-token predictions, creating a strong learned prior toward tool execution over conversational responses.

2. **AskToAct: Enhancing LLMs Tool Use via Self-Correcting Clarification** — Zhang, X., et al. (2025) — https://arxiv.org/abs/2503.01940
   - Directly addresses premature execution bias. Key insight: tool parameters represent explicit user intents, so missing parameters are the signal to clarify rather than hallucinate. Shows that adding clarification as a structured tool action outperforms prompt-based approaches by 10.46% on clarification efficiency.

**Additional references:**

- **Function Calling Internals: Grammars and Constrained Sampling** — Salman Q. (2026) — https://www.salmanq.com/blog/llm-constrained-sampling/
  - Authoritative technical walkthrough of the constrained decoding pipeline: logit computation, temperature scaling, softmax, grammar compilation from JSON schema, token masking, and renormalization. Used as the primary source for the token-level mechanism explanation.
