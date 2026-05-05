# Day 1 — Grounding Commit

**Artifact edited:** `README.md` (latency claim + References) and `methodology.md` (§4 Why Not Path C)

**Commit link:** https://github.com/YohannesDereje/Sales-Agent-Evaluation-Bench/commit/f1a59a21a9d76ca3084e6e11af2c8be72550266f

**What changed and why:**

Both artifacts contained the 3.2× inference speedup number (10,652ms baseline → 3,266ms trained LoRA) without explaining the mechanism behind it. The README stated it as a bare fact; the methodology section justified Path A over prompt engineering on quality grounds only, ignoring the latency dimension entirely.

The explainer taught me that autoregressive LLM inference splits into two fundamentally different phases: a prefill phase (all input tokens processed in parallel, compute-bound, cost fixed for a given prompt length) and a decode phase (one token generated per step, memory-bound, cost scales linearly with output length). The decode phase is the dominant cost for short-output tasks like outreach emails. The baseline model, given no constraints on its output distribution, generates verbose exploratory responses — more decode iterations, higher latency. The LoRA adapter, trained on (instruction, concise_output) pairs via SFT, internalised when to emit EOS: it encodes the concise output distribution in its weights rather than inferring it from context at each call. This is why the speedup is mechanistic and reliable, not coincidental. It also explains why prompt engineering only partially closes the gap (8,421ms): a system prompt steers output length via in-context learning, which is less reliable than a weight-level update and still produces longer outputs than a fine-tuned model.

The explainer also clarified that LoRA adds zero architectural overhead at inference time because the low-rank matrices are merged into the frozen weights before serving (`W' = W + BA`, Hu et al. 2021) — the adapter is not a separate module that runs at inference, it is dissolved into the base weights.

In `README.md`, the single-sentence latency claim was expanded to name the prefill/decode two-phase model, explain the zero-overhead merge, and contrast SFT's weight-level EOS control against prompt engineering's in-context output-length steering. Hu et al. (2021) and Agrawal et al. (2023) were added to the References section as the canonical sources for the mechanism.

In `methodology.md` §4, a new paragraph "Inference latency argument against Path C" was added, formally integrating the decode-phase cost model as a second independent justification for Path A over prompt engineering — the first justification (quality) was already there; the latency justification was missing and is now grounded in the same mechanism.

This changes how I would defend the production recommendation: previously I could only say "the adapter is faster." Now I can say why — the adapter reduced the number of memory-bound decode iterations by training the model to produce shorter, task-specific outputs — and I can predict when the speedup would not hold (e.g. tasks requiring long outputs, where decode cost is dominated by content length regardless of training).
