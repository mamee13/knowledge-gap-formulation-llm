# Day 1 — Sources

**Canonical sources read (minimum 2):**

1. **LoRA: Low-Rank Adaptation of Large Language Models** — Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021) — https://arxiv.org/abs/2106.09685
   - Describes the low-rank adaptation mechanism and explicitly addresses inference latency ("no additional inference latency" as a design goal)

2. **SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills** — Agrawal, A., et al. (2023) — https://arxiv.org/abs/2308.16369
   - Explains the prefill/decode split, why decode phase is memory-bound and dominates production latency, and how decode-maximal batching addresses the bottleneck

**Tool or pattern used hands-on:**

- **Real inference profiling** — Created `demo_decode_latency.py` that uses transformers library to profile actual GPT-2 inference. Measured prefill and decode times separately across different output lengths (20, 50, 100 tokens) to demonstrate that prefill time remains constant (~45ms) while decode time scales linearly with output length. The profiling shows a 4.6× speedup when reducing output from 100 to 20 tokens, with the speedup coming entirely from reduced decode iterations.

**Additional references:**

- **What APM Tools Don't Show About Inference Latency** — Tian Pan (2026) — https://tianpan.co/blog/2026-04-23-llm-span-is-lying-apm-inference-latency
  - Production-scale analysis of prefill vs decode bottlenecks in real deployments

- **Two-Stage Models for Efficient Language Model Decoding** — OverFill paper (2025) — https://arxiv.org/abs/2508.08446
  - Additional context on compute-bound prefill vs memory-bound decode characteristics
