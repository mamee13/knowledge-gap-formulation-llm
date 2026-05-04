"""
Demonstration: How output length drives decode-phase latency in autoregressive generation
"""

import time

def simulate_inference(prompt_tokens, output_tokens, time_per_token_ms=50):
    """
    Simulate LLM inference with prefill and decode phases.
    
    Args:
        prompt_tokens: Number of input tokens
        output_tokens: Number of tokens to generate
        time_per_token_ms: Time per decode step (memory-bound, constant per token)
    
    Returns:
        dict with prefill_time, decode_time, total_time
    """
    # Prefill: process all prompt tokens in parallel (compute-bound, scales with prompt length)
    # Simplified: assume prefill is fast and scales sublinearly due to parallelism
    prefill_time_ms = prompt_tokens * 2  # ~2ms per token in parallel batch
    
    # Decode: generate tokens one-by-one (memory-bound, scales linearly with output length)
    decode_time_ms = output_tokens * time_per_token_ms
    
    total_time_ms = prefill_time_ms + decode_time_ms
    
    return {
        "prefill_ms": prefill_time_ms,
        "decode_ms": decode_time_ms,
        "total_ms": total_time_ms,
        "output_tokens": output_tokens
    }

# Scenario: Same prompt, different output lengths
prompt_length = 100  # tokens

print("=" * 60)
print("Autoregressive Inference: Prefill vs Decode Phase")
print("=" * 60)
print(f"\nPrompt length: {prompt_length} tokens\n")

scenarios = [
    ("Baseline (verbose)", 200),
    ("Prompt-engineered (concise)", 150),
    ("LoRA fine-tuned (task-specific)", 60)
]

for name, output_len in scenarios:
    result = simulate_inference(prompt_length, output_len)
    print(f"{name}:")
    print(f"  Output tokens: {result['output_tokens']}")
    print(f"  Prefill time:  {result['prefill_ms']} ms")
    print(f"  Decode time:   {result['decode_ms']} ms  ← scales with output length")
    print(f"  Total time:    {result['total_ms']} ms")
    print()

print("Key insight:")
print("Decode phase is memory-bound and scales linearly with output length.")
print("A model that generates fewer tokens completes faster, even with")
print("identical prefill cost and no architectural changes.")
