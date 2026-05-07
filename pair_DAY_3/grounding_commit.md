# Day 3 — Grounding Commit

**Asker:** Liul Teshome

**Artifact edited:** `method.md` — Core Mechanism and Ablation Variants sections; `agent/composer.py`; `agent/policy.py`

**Link / file path:** `method.md`, `agent/composer.py`, `agent/policy.py`

**What changed and why:**

In `method.md`, I revised the Core Mechanism section from "The mechanism links source reliability and signal confidence directly to message phrasing and send eligibility" to a mechanically specific description: the mechanism operates in two layers — a `ConfidenceGatedPhrasingProcessor` that masks assertive tokens at generation time when confidence falls below threshold, and a separate `send_eligible` gate that scores the generated message and suppresses it if an assertion slipped through. I also added an Ablation Variants section describing the four-condition design (baseline / gate only / processor only / gate + processor) that isolates the confidence gate as the causal mechanism for over-claim reduction.

In `agent/composer.py`, I replaced the SCAP_POSTSCRIPT prompt injection with a `ConfidenceGatedPhrasingProcessor` instantiated from the confidence score returned by `agent/policy.py`. In `agent/policy.py`, I added a `get_confidence` method that returns a float score from the signal pipeline, which is now passed to the processor rather than embedded in the prompt string.

Before the explainer, I was attributing over-claim reduction to "prompt cleanup" because I had no structural mechanism to point to. Now I can defend the claim: the reduction comes from the confidence gate, not from the prompt being generally more careful.
