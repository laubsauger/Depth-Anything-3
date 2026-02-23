#!/usr/bin/env python3
"""
Test torch.compile() on MPS with attention-heavy models.

Specifically tests if torch.compile provides any speedup for:
1. Simple attention layers
2. Full transformer blocks (like DinoV2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np


class AttentionLayer(nn.Module):
    """Simple multi-head attention layer similar to DinoV2."""

    def __init__(self, dim=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Use SDPA (what DA3 uses)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """Full transformer block with attention + MLP."""

    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = AttentionLayer(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def benchmark_model(model, x, name, num_warmup=10, num_iters=50):
    """Benchmark a model."""
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")

    device = x.device

    # Warmup
    print(f"Warming up ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        with torch.inference_mode():
            _ = model(x)
        torch.mps.synchronize()

    # Benchmark
    print(f"Benchmarking ({num_iters} iterations)...")
    times = []
    for _ in range(num_iters):
        torch.mps.synchronize()
        start = time.time()
        with torch.inference_mode():
            out = model(x)
        torch.mps.synchronize()
        times.append((time.time() - start) * 1000)

    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print(f"Results:")
    print(f"  Mean: {mean_time:.3f}ms ± {std_time:.3f}ms")
    print(f"  Min:  {min_time:.3f}ms")
    print(f"  Max:  {max_time:.3f}ms")

    return mean_time


def main():
    print("="*70)
    print("torch.compile() Benchmark on MPS")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print()

    device = torch.device("mps")
    dtype = torch.float16

    # Test configurations matching DA3
    configs = [
        # (name, batch_size, seq_len, dim, num_heads)
        ("DA3-SMALL (B=1, N=1296)", 1, 1296, 384, 6),
        ("DA3-BASE (B=1, N=1296)", 1, 1296, 768, 12),
        ("Streaming (B=8, N=1296)", 8, 1296, 384, 6),
    ]

    results = []

    for config_name, batch_size, seq_len, dim, num_heads in configs:
        print(f"\n{'#'*70}")
        print(f"# {config_name}")
        print(f"# Shape: B={batch_size}, N={seq_len}, C={dim}, H={num_heads}")
        print(f"{'#'*70}")

        # Create input
        x = torch.randn(batch_size, seq_len, dim, dtype=dtype, device=device)

        # Test 1: Single attention layer
        print(f"\n--- Test 1: Single Attention Layer ---")
        model_eager = AttentionLayer(dim, num_heads).to(device).to(dtype)
        model_compiled = torch.compile(model_eager, backend='inductor')

        time_eager = benchmark_model(model_eager, x, "Eager Mode", num_warmup=5, num_iters=30)
        time_compiled = benchmark_model(model_compiled, x, "torch.compile()", num_warmup=5, num_iters=30)

        speedup_attn = time_eager / time_compiled
        print(f"\n📊 Attention Layer Speedup: {speedup_attn:.2f}x")

        # Test 2: Full transformer block
        print(f"\n--- Test 2: Full Transformer Block ---")
        block_eager = TransformerBlock(dim, num_heads).to(device).to(dtype)
        block_compiled = torch.compile(block_eager, backend='inductor')

        time_block_eager = benchmark_model(block_eager, x, "Eager Mode", num_warmup=5, num_iters=30)
        time_block_compiled = benchmark_model(block_compiled, x, "torch.compile()", num_warmup=5, num_iters=30)

        speedup_block = time_block_eager / time_block_compiled
        print(f"\n📊 Transformer Block Speedup: {speedup_block:.2f}x")

        results.append({
            'config': config_name,
            'attn_eager_ms': time_eager,
            'attn_compiled_ms': time_compiled,
            'attn_speedup': speedup_attn,
            'block_eager_ms': time_block_eager,
            'block_compiled_ms': time_block_compiled,
            'block_speedup': speedup_block,
        })

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Configuration':<30} {'Attn Speedup':<15} {'Block Speedup':<15}")
    print("-"*70)
    for r in results:
        attn_str = f"{r['attn_speedup']:.2f}x"
        block_str = f"{r['block_speedup']:.2f}x"

        if r['attn_speedup'] < 0.95:
            attn_str += " (SLOWER)"
        elif r['attn_speedup'] > 1.05:
            attn_str += " (FASTER)"
        else:
            attn_str += " (SAME)"

        if r['block_speedup'] < 0.95:
            block_str += " (SLOWER)"
        elif r['block_speedup'] > 1.05:
            block_str += " (FASTER)"
        else:
            block_str += " (SAME)"

        print(f"{r['config']:<30} {attn_str:<15} {block_str:<15}")

    print(f"{'='*70}")

    # Detailed breakdown
    print("\nDetailed Timing Breakdown:")
    print(f"{'Configuration':<30} {'Component':<20} {'Eager (ms)':<12} {'Compiled (ms)':<15} {'Speedup':<10}")
    print("-"*90)
    for r in results:
        print(f"{r['config']:<30} {'Attention':<20} {r['attn_eager_ms']:>10.2f}ms {r['attn_compiled_ms']:>13.2f}ms {r['attn_speedup']:>8.2f}x")
        print(f"{'':30} {'Full Block':<20} {r['block_eager_ms']:>10.2f}ms {r['block_compiled_ms']:>13.2f}ms {r['block_speedup']:>8.2f}x")
    print(f"{'='*90}")

    # Verdict
    avg_attn_speedup = np.mean([r['attn_speedup'] for r in results])
    avg_block_speedup = np.mean([r['block_speedup'] for r in results])

    print(f"\n🎯 Average Speedup:")
    print(f"   Attention Layer: {avg_attn_speedup:.2f}x")
    print(f"   Transformer Block: {avg_block_speedup:.2f}x")

    if avg_block_speedup > 1.1:
        print(f"\n✅ VERDICT: torch.compile() provides significant speedup (~{(avg_block_speedup-1)*100:.0f}%)")
        print("   RECOMMENDATION: Use torch.compile() on DA3 models for MPS")
    elif avg_block_speedup > 1.0:
        print(f"\n⚠️  VERDICT: torch.compile() provides minor speedup (~{(avg_block_speedup-1)*100:.0f}%)")
        print("   RECOMMENDATION: Optional - small benefit but adds complexity")
    else:
        print(f"\n❌ VERDICT: torch.compile() is slower on MPS (~{(1-avg_block_speedup)*100:.0f}% slower)")
        print("   RECOMMENDATION: DO NOT use torch.compile() on MPS")


if __name__ == "__main__":
    main()
