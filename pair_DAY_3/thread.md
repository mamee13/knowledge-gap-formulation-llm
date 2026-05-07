# Day 3 — Tweet Thread

**Tweet 1:**
Week 12 of TRP1 @10academy. Day 3: post-training mechanics.

My method.md claims confidence gates phrasing. But I couldn't defend it. Why can't a prompt enforce hedging when confidence drops — and how do you separate send eligibility from generation for a clean ablation? 

---

**Tweet 2:**
RLHF reward models are biased toward confident-sounding responses regardless of correctness.

A prompt saying "hedge when uncertain" fights a learned prior reinforced across millions of training examples.

It works sometimes. It fails silently. You can't ablate it cleanly.

---

**Tweet 3:**
The structural fix: a LogitsProcessor.

It intercepts the raw logit vector at every generation step. When confidence < threshold, assertive openers ("Your", "This") get masked to -∞. Hedging tokens ("Based", "Could") get boosted.

Cannot assert. Not discouraged — impossible.

---

**Tweet 4:**
But phrasing enforcement ≠ send eligibility.

Generation and gating must be two separate steps:
1. Generate with LogitsProcessor
2. Score the output independently
3. Gate: suppress if assertion slipped through

This separation is what makes the ablation clean.

---

**Tweet 5:**
With the two-step architecture you can run 4 ablation conditions:
- Baseline (no gate, no processor)
- Gate only
- Processor only
- Gate + processor

Over-claim reduction only in condition 4 = confidence gate is the mechanism, not prompt cleanup.

---

**Tweet 6:**
This directly fixes my method.md claim and lets me refactor agent/composer.py + agent/policy.py with a mechanically defensible intervention.

Full explainer + code + papers: [BLOG_URL_HERE]
