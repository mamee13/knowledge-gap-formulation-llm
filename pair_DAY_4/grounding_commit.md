# Grounded commit — Day 4, Question 1 (pass^1 vs pass^k)

## Pointers to the actual edits

1. **`generation_scripts/run_act4_ablations.py`**
   - Added **`pass_at_k(n, c, k)`** using the Chen et al. (2021) unbiased estimator \(1 - \binom{n-c}{k}/\binom{n}{k}\) when \(n-c \ge k\), else \(1.0\).
   - Added **`compute_pass_at_k_summary`** to aggregate per-task pass@k and mean pass@k across tasks from scored traces.
   - Extended **`parse_args`** with `--trained-temperature`, `--trained-n-samples`, and `--pass-at-k` (comma-separated \(k\) values).
   - **Trained outputs** are now **`Dict[task_id] → List[candidate outputs]`**: file-based loads wrap one row per task as a single-element list; live inference calls **`build_trained_by_task_via_inference`**, which wraps **`export_trained`** with temperature / n-samples / seed.
   - **Main loop**: baseline and prompt-only unchanged per task; trained runs **`enumerate(trained_outputs)`** with **`sample_index`** on each trace row. **`pass_rate` / `mean_score_pct` / paired bootstrap** use only **`sample_index == 0`** (pass^1, aligned with historical single-sample runs). Full trained traces feed **`pass_at_k_*`** when **`trained_n_samples > 1`**.
   - **`ablation_results.json`** `inputs` now records **`trained_temperature`**, **`trained_n_samples`**, **`pass_at_k_requested`**, **`pass_at_k_computed`**; **`notes`** explain pass^1 vs pass@k.

2. **`generation_scripts/export_heldout_outputs.py` — `export_trained`**
   - New parameters: **`temperature`**, **`n_samples_per_task`**, **`sample_seed`**.
   - Inner loop over tasks × samples; **`do_sample`** when **`temperature > 0`** (with **`top_p=0.95`**); **`torch.manual_seed`** (and CUDA seed) derived from **`sample_seed + ti * 100_003 + si`** for reproducibility.
   - Each exported row includes **`sample_index`**.

3. **`generation_scripts/export_heldout_outputs.py` CLI (`main`)**
   - **`--temperature`**, **`--n-samples`**, **`sample-seed`** passed through to **`export_trained`** for **`trained`** / **`trained_intervene`**.
   - Validates **`n_samples > 1` ⇒ temperature > 0**.

4. **`model_card.md` — §Evaluation → Metrics**
   - Defines **`pass_rate` as pass^1** (first trained sample), documents default greedy single-sample behavior, paired bootstrap on that first sample, and optional **pass@k** from multi-sample runs — explicitly stating pass@k is a **capability-ceiling** diagnostic, not a deployment substitute for pass^1.

## What changed and why (one paragraph)

Question 1 asked for a clear separation between **what pass^k measures** (probability that at least one of \(k\) draws passes after drawing \(n\) samples per task) and **what the shipped harness actually reported** (one boolean per task — structurally pass^1). The fix implements that separation in code instead of only in prose: the default path is unchanged for reproducibility (**first sample only** for aggregate metrics and Delta A/B on **`aggregate_score_pct`**), while opt-in **`--trained-n-samples`** and **`--trained-temperature`** enable pass@k estimation with the canonical unbiased formula, explicit **`sample_index`** in **`held_out_traces.jsonl`**, and **`variant_summary.trained.pass_at_k_chen_et_al_2021`** in **`ablation_results.json`**. Stopping pass^k from being mistaken for production reliability is handled by **`pass_metric_definition`** text on the trained variant summary and by the model card metrics section, which states that pass@k does not replace pass^1 for deployment claims.
