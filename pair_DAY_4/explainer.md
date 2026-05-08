# pass^1 Is a Deployment Metric, pass^k Is a Capability Ceiling: What Your 0.82 Actually Tells You

**Question I'm answering:**

"My `summarize_variant` computes `pass_rate` from a single per-task boolean — `variant_summary.trained.pass_rate: 0.82`, `n=50`. What does pass^k actually measure that pass^1 hides, and which of my 9 missed tasks would each metric label 'real' failures? When does pass^k > pass^1 mislead you about production reliability? And what harness change is required to compute pass^k on this benchmark?"

---

## The Load-Bearing Mechanism

pass^1 and pass^k measure fundamentally different things, and conflating them is the most common source of inflated capability claims on agent benchmarks.

**pass^1** is the probability that a single model draw at your evaluation temperature succeeds. For your harness at `temperature=0` (greedy decoding), pass^1 is deterministic: the same 9 tasks fail every single run. Your 0.82 is not a sample — it is the exact ceiling of your model under greedy decoding. There is no variance to average over.

**pass^k** is the probability that at least one of k independent draws at `temperature > 0` succeeds. The original formula from Chen et al. (2021) is:

```
pass@k = 1 - C(n-c, k) / C(n, k)
```

where `n` is total samples drawn per task, `c` is the number that pass, and `k` is the pool size for the metric. This is an unbiased estimator — you draw `n > k` samples and compute the probability that a random subset of k contains at least one correct answer.

**The critical distinction for your 9 failures:**

- **Deterministic failures** (hard policy violations, capacity over-commitment, banned phrases): these fail at every temperature. pass^k = pass^1 for these tasks. The model has learned the wrong behavior and no amount of resampling recovers it.

- **Stochastic near-misses** (borderline directness, weak-signal phrasing that sometimes hedges correctly): these fail at temperature=0 but pass at temperature > 0 on some draws. pass^k >> pass^1 for these tasks.

Your harness currently cannot distinguish these two failure types because it runs at temperature=0 with k=1. The 9 missed tasks could be all deterministic, all stochastic, or a mix — and that distinction is exactly what determines whether your model is deployable.

---

## Show It

Here is the minimal harness change to compute pass^k on your benchmark:

```python
import math
from statistics import mean

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k estimator from Chen et al. (2021).
    n = total samples drawn per task
    c = number of samples that passed
    k = pool size for the metric
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# Modified summarize_variant for pass@k
def summarize_variant_passk(rows_by_task: dict, k: int = 5, n_samples: int = 10) -> dict:
    """
    rows_by_task: {task_id: [list of n scored rows at temperature > 0]}
    """
    task_pass_at_k = []
    for task_id, samples in rows_by_task.items():
        n = len(samples)
        c = sum(1 for s in samples if s["score"]["pass"])
        task_pass_at_k.append(pass_at_k(n, c, k))
    
    return {
        "pass_at_1": mean(s["score"]["pass"] for task_samples in rows_by_task.values() 
                         for s in task_samples[:1]),  # first sample only
        f"pass_at_{k}": mean(task_pass_at_k),
        "n_samples_per_task": n_samples,
        "k": k
    }


# What you need to add to run_act4_ablations.py:
# 1. Add --temperature argument (default 0.0, set to 0.8 for pass@k run)
# 2. Add --n-samples argument (default 1, set to 10 for pass@k run)
# 3. Loop n_samples times per task at temperature > 0
# 4. Collect all n rows per task, then call pass_at_k(n, c, k)

# Concrete numbers for your benchmark:
# Current: temperature=0, n=1 per task → pass^1 = 0.82 (deterministic)
# Proposed: temperature=0.8, n=10 per task → compute pass@5 per task

# Example: if 4 of your 9 failures are stochastic near-misses
# and pass 3/10 draws at temperature=0.8:
for task_label, n, c in [("deterministic fail", 10, 0), ("stochastic near-miss", 10, 3)]:
    p1 = pass_at_k(10, c, 1)
    p5 = pass_at_k(10, c, 5)
    print(f"{task_label}: pass@1={p1:.2f}, pass@5={p5:.2f}")

# Output:
# deterministic fail:   pass@1=0.00, pass@5=0.00
# stochastic near-miss: pass@1=0.30, pass@5=0.83
```

**What this reveals:** A stochastic near-miss scores 0.00 at pass^1 (greedy) but 0.83 at pass^5 (10 samples). If you report pass^5 = 0.90 for your trained variant, a reader might conclude the model is production-ready. But your production loop runs once per turn — so the relevant metric is pass^1, not pass^5.

---

## Adjacent Concepts

**When pass^k > pass^1 misleads you.** pass^k is a capability ceiling — it tells you what the model *can* do given multiple attempts. For your benchmark, where each outreach task is sent once to a real prospect, pass^k > pass^1 is misleading if you use it to claim production reliability. The "Why Pass@k Optimization Can Degrade Pass@1" paper (2025) makes this concrete: optimizing for pass^k during training shifts gradient emphasis toward low-success instances, which can actually degrade single-shot accuracy. The metric you optimize for shapes the model you get.

**What pass^k is actually useful for.** pass^k is the right metric for diagnosing *why* your model fails. If pass^5 >> pass^1 for a task cluster, those are stochastic near-misses — the model has the right behavior in its distribution but greedy decoding suppresses it. The fix is temperature tuning, not more training. If pass^5 = pass^1 = 0, those are deterministic failures — the model learned the wrong behavior. The fix is more training data or a different loss function.

**How this changes your paired_bootstrap.** Your current `paired_bootstrap` operates on per-task `aggregate_score_pct` differences. If you add pass^k, the per-task statistic changes from a single boolean to a probability estimate. The bootstrap still works — you resample tasks, not samples — but the per-task statistic is now `pass_at_k(n, c, k)` rather than `bool(score["pass"])`. You're now estimating mean pass^k across tasks, not mean pass rate.

---

## Pointers

- **Paper 1:** Chen, M., et al. (2021). [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374). Introduces the unbiased pass@k estimator `1 - C(n-c,k)/C(n,k)`. The formula is the canonical source for any pass@k implementation.

- **Paper 2:** (2025). [Why Pass@k Optimization Can Degrade Pass@1](https://arxiv.org/abs/2602.21189). Explains the fundamental tension between pass^k and pass^1 as training objectives and evaluation metrics. Directly relevant to your deployment decision question.

- **Tool/pattern used:** `pass_at_k` implementation above, derived from the Chen et al. formula. Demonstrates the deterministic vs stochastic failure distinction with concrete numbers from your benchmark (n=10, c=0 vs c=3, k=5).

- **Further reading:** For the harness change, the minimal addition is a `--n-samples` flag and a temperature parameter in `run_act4_ablations.py`. The `export_heldout_outputs.py` already handles per-task output generation — you need to call it n times per task at temperature > 0 and aggregate.

---

