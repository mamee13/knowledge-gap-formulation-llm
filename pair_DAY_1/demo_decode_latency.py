"""
Demonstration: Measuring actual prefill vs decode latency in autoregressive generation

This script profiles real inference to show:
1. Prefill time is fixed for a given prompt (compute-bound, parallel)
2. Decode time scales linearly with output length (memory-bound, sequential)
"""

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

def profile_inference(model, tokenizer, prompt, max_new_tokens):
    """
    Profile actual inference to measure prefill and decode phases separately.
    
    Returns:
        dict with prefill_time, decode_time, total_time, tokens_generated
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_length = inputs.input_ids.shape[1]
    
    # Warm up GPU
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    
    # Measure total generation time
    start_total = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    total_time = (time.time() - start_total) * 1000  # Convert to ms
    
    # Measure prefill time (generate just 1 token)
    start_prefill = time.time()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    prefill_time = (time.time() - start_prefill) * 1000  # Convert to ms
    
    # Calculate decode time
    tokens_generated = outputs.shape[1] - input_length
    decode_time = total_time - prefill_time
    time_per_token = decode_time / max(tokens_generated - 1, 1)  # Exclude first token (prefill)
    
    return {
        "prefill_ms": round(prefill_time, 1),
        "decode_ms": round(decode_time, 1),
        "total_ms": round(total_time, 1),
        "tokens_generated": tokens_generated,
        "ms_per_token": round(time_per_token, 1)
    }

def main():
    print("=" * 70)
    print("Real Inference Profiling: Prefill vs Decode Phase")
    print("=" * 70)
    print("\nLoading model (GPT-2 small for demonstration)...")
    
    # Use small model for fast demonstration
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model.eval()
    
    # Use CPU for reproducibility (GPU would be faster but requires CUDA)
    device = "cpu"
    model = model.to(device)
    
    prompt = "The relationship between prefill and decode phases in LLM inference"
    print(f"\nPrompt: '{prompt}'")
    print(f"Prompt tokens: {len(tokenizer.encode(prompt))}\n")
    
    # Test different output lengths
    scenarios = [
        ("Short output", 20),
        ("Medium output", 50),
        ("Long output", 100)
    ]
    
    results = []
    for name, max_tokens in scenarios:
        print(f"Profiling: {name} ({max_tokens} max tokens)...")
        result = profile_inference(model, tokenizer, prompt, max_tokens)
        results.append((name, result))
    
    print("\n" + "=" * 70)
    print("Results: Prefill vs Decode Breakdown")
    print("=" * 70)
    
    for name, result in results:
        print(f"\n{name}:")
        print(f"  Tokens generated:  {result['tokens_generated']}")
        print(f"  Prefill time:      {result['prefill_ms']} ms  ← fixed cost")
        print(f"  Decode time:       {result['decode_ms']} ms  ← scales with output length")
        print(f"  Time per token:    {result['ms_per_token']} ms")
        print(f"  Total time:        {result['total_ms']} ms")
    
    # Calculate speedup from reducing output length
    baseline_time = results[2][1]['total_ms']  # Long output
    short_time = results[0][1]['total_ms']     # Short output
    speedup = baseline_time / short_time
    
    print("\n" + "=" * 70)
    print("Key Insight:")
    print("=" * 70)
    print(f"Reducing output from {results[2][1]['tokens_generated']} to {results[0][1]['tokens_generated']} tokens")
    print(f"achieves {speedup:.1f}× speedup ({baseline_time}ms → {short_time}ms)")
    print("\nPrefill time is nearly identical across all runs.")
    print("Decode time scales linearly with output length.")
    print("A model that generates fewer tokens completes faster,")
    print("even with identical prefill cost and no architectural changes.")

if __name__ == "__main__":
    main()
