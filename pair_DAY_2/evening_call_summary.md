# Evening call summary - Day 2

Prepared by Yonas Eshete approved by Mamaru Yirga

## Feedback on the partner's explainer for Yonas's question

Yonas's question asked why a function-calling agent fabricates missing required arguments and executes a tool instead of asking a clarifying question. The useful answer needed to explain the mechanism behind premature tool execution and give a safer design pattern for the Week 10 Conversion Engine.

The feedback target was specific:

- The explainer should not only say "prompt the model better." Yonas already had a prompt patch in `eval/run_heldout.py`, and the point of the gap was to understand why that patch felt fragile.
- The explainer needed to name the mechanism plainly: the model has a learned/action-interface bias toward completing the tool-call pattern once tools are available, especially when the benchmark or prompt rewards task completion.
- The answer needed to avoid pretending that exact token probabilities had been inspected unless a real trace was shown.
- The fix needed to be structural: give the model a safe action path such as `ask_clarifying_question(question_text)`, and validate required arguments before allowing destructive tools to run.

The requested revision was to connect the mechanism directly to the artifact:

If the agent lacks `order_id`, the interface should make "ask for the missing order ID" a valid tool/action, not merely a sentence buried in the prompt. That lets the system absorb the model's tendency to act instead of fighting it with a longer SCAP postscript.

Note: the partner's revised explainer artifact for Yonas's question is not present in this local Day 2 folder, so this summary records the feedback direction and expected revision rather than a file-level verification.

## Feedback on Yonas's explainer for the partner's question

Yonas's explainer answered the partner's intent-classifier question by naming the mechanism:

**LLM classification here is next-token generation under label pressure, not a separate classification module.**

The first revised version clarified why the seven labels in `_CLASSIFY_SYSTEM` make valid labels likely but do not guarantee them. It also explained why the keyword tier handles many replies: obvious sales replies are often lexical, meaning the intent is carried by phrases like `not interested`, `pricing`, or `tell me more`.

The partner-side gap needed more than that diagnosis, so the evening feedback focused on making the explanation concrete and defensible:

- Avoid saying the model outputs "exactly" one of seven labels unless constrained decoding or enum validation enforces it.
- Explain that labels like `positive_interest` may be multi-token, so the model walks through the label token by token.
- Avoid unverified attention-head claims. Without attention traces, describe the keyword behavior as a behavioral correlation rather than a measured internal map.
- Add a hands-on demonstration from the actual repo rather than only a conceptual example.
- Explain why this classifier does not obviously need a fine-tuned model yet.

## Revisions made to Yonas's explainer

The final explainer now includes:

- the mechanism name: **next-token generation under label pressure**
- the distinction between **lexical intent** and **compositional intent**
- a section showing why prompt-only JSON can produce valid JSON but invalid enum values
- a worked example of unconstrained vs constrained label decoding
- a hands-on local check against `signal-driven-sales-conversion-engine/agent/llm/reply_agent.py`
- a model-card implication: measure the ambiguous fallback bucket before deciding a fine-tuned classifier is needed

The hands-on check used Python's `ast` module to inspect the actual file. It extracted the seven `ReplyIntent` labels, counted the keyword lists, ran sample replies through the keyword-tier logic, and showed that `{"intent": "interested"}` is valid JSON but not a valid `ReplyIntent`.

That made the mechanism visible without claiming a live LLM trace.

## Final pair-day outcome

Both sides converged on the same engineering principle from different angles:

Prompt instructions are useful, but the production boundary should carry the guarantee.

For Yonas's tool-use question, that means missing required arguments should route to a safe clarification action before destructive tools can run.

For the partner's classifier question, that means the intent label should be constrained or validated as an enum instead of trusting prompt-only JSON.