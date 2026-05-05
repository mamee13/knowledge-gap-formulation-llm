# Day 1 — Sign-Off

**Asker:** Yohannes Dereje

**Gap closure status:** Closed

**What I understand now that I didn't before:**

The 3.2× speedup in my ablation results has nothing to do with the LoRA adapter making the model computationally faster—it is not doing fewer matrix multiplications, it is not running lighter attention, and it is not skipping any layers. The mechanism is entirely in the decode phase and comes down to one thing: the fine-tuned model generates fewer output tokens.

Before reading this explainer, I assumed adding a system prompt and a LoRA adapter would add overhead and make inference slower. That assumption treated latency as a function of input complexity. The explainer corrected this by showing that total latency is `prefill_time + (output_tokens × time_per_decode_step)`. Prefill is fixed for a given prompt—it runs once, in parallel, and the cost is dominated by input length. Decode is what scales with output length, and decode is memory-bound: every output token requires a full sequential pass that reloads model weights and the growing KV cache from GPU memory.

The baseline model, unconstrained, generates verbose outputs. The LoRA adapter, trained on concise task-specific examples, learned when to emit the EOS token earlier. Fewer output tokens means fewer decode iterations. Fewer decode iterations means lower total latency. The 3.2× speedup is a 3.2× reduction in decode steps—not a 3.2× improvement in per-step efficiency.

The profiling demonstration made this concrete. Prefill time stayed flat at ~52ms regardless of output length. Decode time scaled linearly at ~15ms per token. The entire latency difference between short and long outputs lived in the decode phase. This directly maps to my ablation chart: baseline generates long outputs, prompt engineering constrains them moderately, the LoRA adapter constrains them most reliably because the behavior is encoded in the weights rather than inferred from context.

I can now correctly explain the Inference Speed chart in my ablation results, revise the latency claim in my README with the actual mechanism rather than just asserting the number, and defend the production recommendation on Slide 8: fine-tuning beats prompt engineering for latency not because it is computationally lighter, but because it more reliably reduces output length—and output length is the dominant cost factor in any memory-bound decode workload.
