# Day 3 — Sign-Off

**Asker:** Liul Teshome

**Gap closure status:** Closed

**What I understand now that I didn't before:**

Before reading this explainer, I assumed a prompt instruction was sufficient to enforce confidence-gated phrasing — that telling the model to "hedge when uncertain" would reliably produce probabilistic language when confidence dropped. The explainer showed me why this fails mechanically: RLHF reward models are biased toward confident-sounding responses regardless of correctness, so the model's post-training distribution actively resists hedging. A prompt instruction competes against this learned prior and loses silently, with no signal that it failed.

The load-bearing mechanism I now understand is the `LogitsProcessor`. It intercepts the raw logit vector at every generation step and masks assertive sentence-opening tokens to `-∞` when confidence falls below threshold — making assertion structurally impossible rather than merely discouraged. This is the same mechanism that makes function-calling reliable: constrained decoding doesn't hope the model complies, it makes non-compliance impossible.

The second thing I now understand is why generation and send eligibility must be separated. If they are coupled in a single prompt, I cannot distinguish between "the model generated a hedged message" and "the eligibility check blocked an assertive message." The two-step architecture — generate with the processor, then gate independently — is what makes the ablation clean and the over-claim reduction claim defensible.

I can now rewrite the Core Mechanism and Ablation Variants sections of my method.md with a mechanically specific description, and refactor agent/composer.py and agent/policy.py to implement the structural separation rather than relying on prompt instructions.
