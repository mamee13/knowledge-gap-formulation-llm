# Day 3 — Question

**Topic:** Training and post-training mechanics

**My Question:**

Why DPO, SimPO, and ORPO Produce Different Judges Even on the Same Preference Pairs

In `docs/methodology.md`, I selected SimPO over ORPO after reading both papers, but my rationale in
`docs/methodology_rationale.md` treats the choice as a data-efficiency argument without explaining what
each method's loss function is actually penalizing differently at the gradient level. My 121 preference
pairs include 9 of 12 style-guide seeds with `required_signal_framing: interrogative`, which biased the
preference signal and caused the trained judge to approve assertive outputs on ~8/65 dev tasks. I cannot
explain whether this bias would manifest differently under DPO vs SimPO vs ORPO — specifically, whether
DPO's reference-model KL term would have dampened the bias by anchoring to the base model's prior, while
SimPO's reference-free formulation amplified it. Knowing this would let me revise the "One unresolved
training failure" paragraph in `docs/memo.md` to name the algorithmic reason the bias appeared, not just
the data reason.

**Artifact pointer:** `docs/memo.md` → Skeptic's Appendix, "One unresolved training failure";
`docs/methodology.md` → Path B justification
