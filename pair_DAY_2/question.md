# Day 2 — Question

**Topic:** Agent and tool-use internals.

**My Question:** Token-Level Classification Mechanics in Intent Classifiers

In agent/llm/reply_agent.py:L195-220, my intent classifier sends a system prompt with seven intent labels and asks the model to return JSON with the chosen intent. The model reliably produces one of these seven strings, but I don't know what's happening at the token level when it makes that choice.

What token-level mechanism makes the model output one of exactly seven strings? And why does keyword pre-screening (L130-180) catch 80% of cases before the LLM even runs—what does that tell me about when the LLM's classification actually adds value?

Closing this gap means either adding structured output constraints to the classify_reply() call (if the model supports it) or documenting why keyword pre-screening handles the majority of cases and the LLM is only needed for ambiguous replies. It also lets me add a note to the Week 11 model card explaining why the intent classifier doesn't need a fine-tuned model.
