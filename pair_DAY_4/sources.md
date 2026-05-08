# Day 4 — Sources

**Canonical sources read (minimum 2):**

1. **Evaluating Large Language Models Trained on Code** — Chen, M., et al. (2021) — https://arxiv.org/abs/2107.03374
   - Introduces the unbiased pass@k estimator: `1 - C(n-c, k) / C(n, k)`. This is the canonical formula for any pass@k implementation. Load-bearing in the explainer's mechanism section and the `pass_at_k` code implementation.

2. **Why Pass@k Optimization Can Degrade Pass@1** — (2025) — https://arxiv.org/abs/2602.21189
   - Explains the fundamental tension between pass^k and pass^1 as both training objectives and evaluation metrics. Shows that optimizing for pass^k shifts gradient emphasis toward low-success instances, which can degrade single-shot accuracy. Load-bearing in the "when pass^k misleads you" section.

**Tool or pattern used hands-on:**

- **pass_at_k implementation** — Implemented the Chen et al. unbiased estimator in Python and applied it to the partner's benchmark numbers (n=10, c=0 vs c=3, k=5) to demonstrate the deterministic vs stochastic failure distinction concretely. Also designed the minimal harness change for `run_act4_ablations.py` — adding `--n-samples` and `--temperature` flags and modifying `summarize_variant` to compute per-task pass^k estimates before bootstrapping.

**Additional references:**

- **Evaluating LLM Agent Reliability Under Production-Like Stress Conditions** — (2025) — https://arxiv.org/html/2601.06112
  - Confirms that existing agent benchmarks primarily measure single-run success rates, failing to capture reliability characteristics critical for production deployment. Supports the pass^1 vs pass^k deployment argument.
