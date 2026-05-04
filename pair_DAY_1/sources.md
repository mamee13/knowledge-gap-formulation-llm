# Day 1 — Sources

**Canonical sources read (minimum 2):**

1. **LoRA: Low-Rank Adaptation of Large Language Models** — Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021) — https://arxiv.org/abs/2106.09685
   - Describes the low-rank adaptation mechanism and explicitly addresses inference latency ("no additional inference latency" as a design goal)

2. **SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills** — Agrawal, A., et al. (2023) — https://arxiv.org/abs/2308.16369
   - Explains the prefill/decode split, why decode phase is memory-bound and dominates production latency, and how decode-maximal batching addresses the bottleneck

**Tool or pattern used hands-on:**

- **Python simulation** — Created `demo_decode_latency.py` to demonstrate the linear relationship between output length and decode-phase latency. Ran the simulation to produce concrete numbers showing how identical prefill costs with different output lengths produce the observed 3.2× speedup.

**Additional references:**

- **What APM Tools Don't Show About Inference Latency** — Tian Pan (2026) — https://tianpan.co/blog/2026-04-23-llm-span-is-lying-apm-inference-latency
  - Production-scale analysis of prefill vs decode bottlenecks in real deployments

- **Two-Stage Models for Efficient Language Model Decoding** — OverFill paper (2025) — https://arxiv.org/abs/2508.08446
  - Additional context on compute-bound prefill vs memory-bound decode characteristics
