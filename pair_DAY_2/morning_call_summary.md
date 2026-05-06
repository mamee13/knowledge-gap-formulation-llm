# Morning call summary - Day 2

Prepared by Yonas Eshete approved by Mamaru Yirga

## Sharpening Yonas's question

Yonas's Day 2 gap came from the Week 10 Conversion Engine evaluation. In the tau2-Bench retail-style probes, the agent showed premature tool execution: when the user asked to cancel an order but did not provide the required `order_id`, the agent fabricated an ID and called `cancel_pending_order` instead of asking a clarification question.

The original framing risked becoming too broad: tool-use safety, RLHF incentives, JSON schemas, destructive tool constraints, and prompt design were all mixed together. We narrowed it to the mechanism that was actually blocking the artifact:

**Why does a function-calling LLM prefer fabricating missing required arguments and executing a tool over asking the user a clarifying question?**

The artifact connection was clear. Yonas had added a heavy `SCAP_POSTSCRIPT` in `eval/run_heldout.py` to force the model to "ask, not assert." The sharper question asks whether that prompt patch should be replaced with a structural interface change, such as an explicit `ask_clarifying_question(question_text)` tool or stricter validation before destructive tool calls.

The final scope for Yonas's question:

- explain premature tool execution as a model/interface mechanism
- avoid turning the answer into generic tool-calling best practices
- connect the answer back to deleting or simplifying the fragile SCAP prompt patch
- identify a safer agent boundary for missing required arguments

## Sharpening the partner's question

The partner's Day 2 gap came from `signal-driven-sales-conversion-engine/agent/llm/reply_agent.py`. Their intent classifier sends a system prompt with seven intent labels and asks the model to return JSON with one selected intent. Keyword pre-screening handles most replies before the LLM fallback runs.

The initial question was:

**What token-level mechanism makes the model output one of exactly seven strings, and why does keyword pre-screening catch most cases before the LLM even runs?**

We sharpened this away from general classifier advice and toward the actual repo mechanism:

- The LLM fallback is not a separate classifier head.
- It generates JSON token by token.
- The seven labels in the prompt bias the next-token distribution, but do not guarantee a valid enum.
- Keyword pre-screening handles replies where the intent is visible in surface phrases.
- The LLM fallback is only valuable when intent depends on sentence structure, contrast, qualification, or competing signals.

The final scope for the partner's question:

- explain LLM classification as next-token generation under label pressure
- explain why keyword rules handle lexical intent cheaply
- explain when the LLM adds value over keyword matching
- connect the answer to structured output constraints or enum validation in `classify_reply()`
- support a Week 11 model-card note explaining why this classifier does not obviously need a fine-tuned model yet

## Pair-day split

The two questions were related but distinct.

Yonas's question was about **tool execution under missing information**: why the model acts too early.

The partner's question was about **intent classification under a fixed label set**: why the model emits one of several labels and when the LLM layer is worth using.

That split gave each explainer a clear job and prevented the pair-day from collapsing into one broad discussion about "how LLM agents work."