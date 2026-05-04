# Day 1 — Question

**Topic:** Inference-time mechanics

**My Question:**

Why Do Output Tokens Cost 2× More Than Input Tokens?

In agent/llm/client.py:L91, I hardcoded the cost formula as (input_tokens × $0.0000014) + (output_tokens × $0.0000028) — output tokens are priced exactly 2× input tokens. I copied this from OpenRouter's pricing page without understanding what hardware reality this ratio reflects.

What are the prefill and decode phases of a single inference call, why does the autoregressive decode phase cost more per token than the parallelized prefill phase, and for my typical call shape, is my cost prefill-dominated or decode-dominated?

**Connection to existing artifact:**

Closing this gap means fixing ClaudeClient (agent/llm/client.py:L108-110) which currently inherits DeepSeek's pricing instead of Claude's actual rates — a concrete bug that makes every eval call cost calculation wrong. It also lets me add a note to my CFO memo explaining whether the $3.53/lead cost is prefill-dominated or decode-dominated, which changes my optimization target.
