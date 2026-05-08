# Day 4 — Tweet Thread

**Tweet 1:**
Week 12 of TRP1 @10academy. Day 4: evaluation & statistics.

My partner reports pass_rate: 0.82 on their agent benchmark. But their harness runs at temperature=0, k=1. That's pass^1, not pass^k. The difference determines whether their 9 failures are fixable or fundamental. 🧵

---

**Tweet 2:**
pass^1 = probability a single greedy draw succeeds.
pass^k = probability at least one of k draws at temperature > 0 succeeds.

At temperature=0, pass^1 is deterministic. The same 9 tasks fail every run. There's no variance to average over. 0.82 is the exact ceiling.

---

**Tweet 3:**
The formula (Chen et al. 2021):

pass@k = 1 - C(n-c, k) / C(n, k)

n = samples drawn, c = samples that passed, k = pool size.

For a task with 10 draws, 0 passing: pass@5 = 0.00
For a task with 10 draws, 3 passing: pass@5 = 0.83

Same pass^1. Completely different story.

---

**Tweet 4:**
This separates two failure types:

Deterministic: hard policy violations, banned phrases. Fail at every temperature. pass^k = pass^1. Fix = more training.

Stochastic near-miss: borderline phrasing that sometimes hedges correctly. pass^k >> pass^1. Fix = temperature tuning.

---

**Tweet 5:**
When does pass^k mislead you?

When production runs the model once per turn. Your agent sends one email per prospect — k>1 recovery never happens. Reporting pass^5 = 0.90 implies reliability that doesn't exist.

pass^1 is the deployment metric. pass^k is the capability ceiling.

---

**Tweet 6:**
Harness fix: add --n-samples and --temperature flags to run_act4_ablations.py. Run 10 samples per task at temperature=0.8. Compute pass_at_k(n, c, k) per task. Bootstrap over tasks, not samples.

Full explainer + code: [BLOG_URL_HERE]
