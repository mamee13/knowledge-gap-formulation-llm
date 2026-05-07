# Day 3 — Sources

**Canonical sources read (minimum 2):**

1. **Taming Overconfidence in LLMs: Reward Calibration in RLHF** — Leng, J., et al. (2024) — https://arxiv.org/abs/2410.09724
   - Demonstrates that PPO reward models are systematically biased toward high-confidence responses regardless of actual correctness. Proposes PPO-M and PPO-C as calibration fixes. Directly explains why prompt-only confidence control is unreliable: the RLHF training distribution actively resists hedging, making prompt instructions compete against a learned prior rather than enforce a structural constraint.

2. **Controlling Language Model Generation with NVIDIA's LogitsProcessorZoo** — NVIDIA / Hugging Face (2024) — https://huggingface.co/blog/logits-processor-zoo
   - Authoritative documentation for the `LogitsProcessor` API in the Hugging Face transformers library. Explains how logits processors intercept raw logit vectors before sampling, enabling structural enforcement of generation constraints. Used as the primary source for the `ConfidenceGatedPhrasingProcessor` implementation pattern in the explainer.

**Tool or pattern used hands-on:**

- **ConfidenceGatedPhrasingProcessor implementation** — Designed and implemented the two-layer architecture: a `LogitsProcessor` subclass that masks assertive tokens and boosts hedging tokens when confidence falls below threshold, paired with a separate `send_eligible` function that gates on the generated output independently. The code demonstrates the architectural separation that makes the ablation clean: generation enforcement and send eligibility are decoupled, allowing each to be disabled independently in ablation conditions.

**Additional references:**

- **Calibrating the Confidence of Large Language Models by Eliciting Fidelity** — (2024) — https://arxiv.org/abs/2404.02655
  - Additional context on post-alignment overconfidence: RLHF-optimized models exhibit overconfidence where expressed confidence does not calibrate with correctness rate. Supports the claim that prompt-only hedging instructions are insufficient.
