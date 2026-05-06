# Grounding Commit — Day 2

**Artifact edited:** `conversion-engine/eval/run_heldout.py`  
**Commit:** [`ed68d70`](https://github.com/Sanoy24/conversion-engine/commit/ed68d70ce3be3a86c7e03a4628880fe15bc9271c)

Mamaru's explainer on premature execution bias identified that the SCAP postscript in `run_heldout.py` fights the model's RLHF-trained tool-call prior with prompt text — a strategy that works at the margin but cannot override a strong learned bias. The structural alternative is to make clarification a valid tool call so the model's tendency to emit `<tool_call>` tokens routes into a safe action instead of being suppressed. The commit adds a 7th evaluation condition (`clarification_tool`) that injects `ask_clarifying_question(question_text: str)` as a declared tool alongside the existing destructive tools, and adds a comment block on the SCAP postscripts explaining why they are the prompt-based approach and citing the Day 2 explainer as the source of the structural insight.
