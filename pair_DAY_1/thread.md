# Day 1 — Tweet Thread

**Tweet 1 — The Question**

I thought adding a LoRA adapter would make inference slower. Instead, my fine-tuned model runs 3.2× faster than baseline on identical hardware. Here's the mechanism that explains why "more work" actually means less latency. 🧵

---

**Tweet 2 — The Mechanism (part 1)**

Every LLM inference call is two operations: PREFILL (process prompt in parallel, compute-bound) and DECODE (generate tokens one-by-one, memory-bound). Your total latency = prefill_time + (output_tokens × time_per_token). Prefill is fixed. Decode scales linearly with output length.

---

**Tweet 3 — The Mechanism (part 2) + Visual/Code**

Same prompt, different output lengths. The speedup is in the decode phase:

```
Baseline (200 tokens):     10,200ms (prefill: 200, decode: 10,000)
Prompt-eng (150 tokens):    7,700ms (prefill: 200, decode:  7,500)
LoRA fine-tuned (60 tokens): 3,200ms (prefill: 200, decode:  3,000)
```

Fewer output tokens = fewer memory-bound decode steps = faster inference.

---

**Tweet 4 — The Mechanism (part 3)**

LoRA doesn't add inference overhead. During inference, the low-rank matrices merge into the frozen weights: W' = W + BA. You run the same matrix multiplication as the base model, just with updated numbers. Zero architectural overhead. The adapter taught the model *when to stop generating*.

---

**Tweet 5 — Adjacent Concept + Link**

For production deployments, output length is often the dominant cost. Fine-tuning that cuts average output by 50% effectively doubles throughput on the same hardware. This is why decode-phase optimization matters more than prefill tricks at scale.

Full explainer with papers + code: https://medium.com/@mamaruyirga1394/why-fine-tuned-models-generate-faster-the-decode-phase-mechanism-behind-your-3-2-speedup-9590a549c3d5


Thread link : https://x.com/MSishagn13/status/2051420699491758404?s=20
