# Sign-off — Day 4, Question 1 (pass^1 vs pass^k)

## Gap-closure judgment

**Closed** for the conceptual gap and **implemented** for the harness gap.

The explanation’s core claims are now either encoded in the runner or documented on the model card: pass^1 is the single-draw / first-sample metric appropriate for a one-shot send path; pass^k (implemented as unbiased pass@k from Chen et al., 2021 over \(n\) samples per task) estimates how often **at least one** of \(k\) randomly chosen draws from those \(n\) would pass — a **capability-ceiling** diagnostic that can exceed pass^1 when failures are **stochastic near-misses**, and equals pass^1 when failures are **deterministic** (every draw fails). The harness now computes pass@k only when you explicitly run multi-sample inference (**`--trained-n-samples > 1`** and **`--trained-temperature > 0`**); it does not silently upgrade an old **`ablation_results.json`** number to pass^k.

Residual scope (honest boundary): this repo does not re-run GPU inference in CI here, so the **numerical** pass@k lift for your specific nine tasks is not populated in committed artifacts until you execute a multi-sample ablation locally or in Colab and commit the new JSON.

## What I understand now that I did not before (one paragraph)

I used to treat **`pass_rate: 0.82`** in **`ablation_results.json`** as a generic “fraction of tasks that pass” without naming the implicit **`k=1`** and **`temperature=0`** contract — which made it easy to confuse with pass^k numbers from coding benchmarks or multi-sample reports. I now separate three ideas: **pass^1** is whether the **first** output on the harness passes (here aligned with greedy single-shot deployment); **pass@k** from \(n\) samples per task answers whether the model **could** pass if the product drew multiple times and picked one — useful for diagnosing stochastic vs deterministic failures; and **reporting pass^k as “reliability” for a single-send product** is misleading when **`pass^k > pass^1`**, because production does not take \(k\) tries unless the workflow explicitly allows retries. The mental model “optimize pass^k and hurt pass^1” from recent training literature is now something I can connect to **which metric** we put on the dashboard versus which we use for diagnosis — not two names for the same quantity.
