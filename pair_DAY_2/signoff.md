# Sign-off - Day 2

**Asker:** Yonas Eshete  
**Explainer:** Mamaru Yirga
**Question:** Why does a function-calling LLM fabricate missing required arguments instead of asking a clarifying question?

---

**Gap closed.**

Before this explainer, I thought the premature execution failure in my conversion engine — documented in probes P023–P025 — was a prompt problem I could patch with a better SCAP postscript. The explainer corrected that: the bias toward tool execution is baked into the model's weights through RLHF post-training, where stalling to ask a clarifying question gets penalized because benchmark evaluators reward task completion over safe information gathering. The token-level consequence is that `<tool_call>` tokens carry inflated probability mass, and natural language clarification responses are suppressed — not because the prompt is wrong, but because the training signal shaped the prior. I now understand why the SCAP postscript can shift behavior at the margin but cannot eliminate the failure mode, and why the structural fix — adding an explicit `ask_clarifying_question(question_text: str)` tool to the schema so that asking becomes a valid action the model can take — works with the RLHF prior instead of against it.
