"""
Two questions:
  1. Training scaling: how does fwd+bwd scale with sequence length up to a
     full ~28000-frame Melee match (8 min at 60 Hz)?
  2. Inference: what does the "KV cache equivalent" look like for Mamba,
     how big is it, and how fast is per-step decoding?

Mamba's inference state is *constant size in L* — that's the structural
advantage over transformer KV cache, which grows linearly. Per-step
decode just advances the state.
"""
from __future__ import annotations
import time
import torch
from mamba_ssm import Mamba2

DEVICE = "cuda"
DTYPE = torch.bfloat16


def time_train(L, bsz, d_model=384, d_state=128, n_layer=8, headdim=64,
               chunk_size=256, grad_checkpoint=False):
    """One forward+backward through n_layer stacked Mamba2 blocks at length L."""
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    blocks = torch.nn.ModuleList([
        Mamba2(d_model=d_model, d_state=d_state, headdim=headdim, chunk_size=chunk_size)
        for _ in range(n_layer)
    ]).to(DEVICE).to(DTYPE)
    x0 = torch.randn(bsz, L, d_model, device=DEVICE, dtype=DTYPE)
    # warmup
    for _ in range(2):
        for p in blocks.parameters():
            if p.grad is not None: p.grad = None
        x = x0.clone().requires_grad_(True)
        for b in blocks:
            x = b(x) + x if not grad_checkpoint else (
                torch.utils.checkpoint.checkpoint(b, x, use_reentrant=False) + x
            )
        x.sum().backward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(3):
        for p in blocks.parameters():
            if p.grad is not None: p.grad = None
        x = x0.clone().requires_grad_(True)
        for b in blocks:
            x = b(x) + x if not grad_checkpoint else (
                torch.utils.checkpoint.checkpoint(b, x, use_reentrant=False) + x
            )
        x.sum().backward()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 3 * 1000
    gb = torch.cuda.max_memory_allocated() / 1e9
    return dt, gb


def measure_step(d_model=384, d_state=128, n_layer=8, headdim=64, max_L=8192):
    """Inference: prefill + per-step decode latency. Cache size."""
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    blocks = torch.nn.ModuleList([
        Mamba2(d_model=d_model, d_state=d_state, headdim=headdim, layer_idx=i, chunk_size=256)
        for i in range(n_layer)
    ]).to(DEVICE).to(DTYPE).eval()
    bsz = 1
    # Allocate inference cache for each layer
    caches = {i: blocks[i].allocate_inference_cache(bsz, max_L) for i in range(n_layer)}
    # caches[i] is a tuple (conv_state, ssm_state)
    cache_bytes = 0
    for i, (conv, ssm) in caches.items():
        cache_bytes += conv.numel() * conv.element_size() + ssm.numel() * ssm.element_size()
    print(f"  cache for {n_layer} layers, bsz=1, d_model={d_model}, d_state={d_state}: "
          f"{cache_bytes/1024:.1f} KB")
    print(f"    per-layer conv_state: {tuple(caches[0][0].shape)} "
          f"({caches[0][0].numel() * caches[0][0].element_size()} B)")
    print(f"    per-layer ssm_state:  {tuple(caches[0][1].shape)} "
          f"({caches[0][1].numel() * caches[0][1].element_size()} B)")

    # Per-step decode: one token at a time. Mamba2.step expects (B, 1, D).
    x = torch.randn(bsz, 1, d_model, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        for _ in range(20):
            h = x
            for i, b in enumerate(blocks):
                conv, ssm = caches[i]
                h, conv2, ssm2 = b.step(h, conv, ssm)
                caches[i] = (conv2, ssm2)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_steps = 200
    with torch.no_grad():
        for _ in range(n_steps):
            h = x
            for i, b in enumerate(blocks):
                conv, ssm = caches[i]
                h, conv2, ssm2 = b.step(h, conv, ssm)
                caches[i] = (conv2, ssm2)
    torch.cuda.synchronize()
    dt_us = (time.perf_counter() - t0) / n_steps * 1e6
    print(f"  per-step decode (n_layer={n_layer}, no_grad): {dt_us:.1f} µs/step")
    print(f"    → {1e6/dt_us:.0f} steps/sec  (vs 60 Hz target: 1 frame = 16667 µs, "
          f"so {dt_us/16667*100:.3f}% of frame budget)")


def main():
    print("=" * 64)
    print("Training scaling: 8-layer Mamba-2 stack, d_model=384, N=128, bf16")
    print("=" * 64)
    print(f"\n{'L':>6}  {'bsz':>4}  {'fwd+bwd time':>14}  {'peak mem':>10}")

    # Standard (no grad-checkpointing)
    print("\nStandard (full activations stored):")
    for L, bsz in [(1024, 32), (2048, 16), (4096, 8), (8192, 4), (16384, 2)]:
        try:
            ms, gb = time_train(L, bsz)
            print(f"  {L:>6}  {bsz:>4}  {ms:>11.1f} ms  {gb:>7.2f} GB")
        except torch.cuda.OutOfMemoryError:
            print(f"  {L:>6}  {bsz:>4}  {'OOM':>14}  {'OOM':>10}")
        torch.cuda.empty_cache()

    # With activation checkpointing
    print("\nWith gradient checkpointing (trade compute for memory):")
    for L, bsz in [(8192, 8), (16384, 4), (28800, 2), (28800, 1)]:
        try:
            ms, gb = time_train(L, bsz, grad_checkpoint=True)
            print(f"  {L:>6}  {bsz:>4}  {ms:>11.1f} ms  {gb:>7.2f} GB")
        except torch.cuda.OutOfMemoryError:
            print(f"  {L:>6}  {bsz:>4}  {'OOM':>14}  {'OOM':>10}")
        torch.cuda.empty_cache()

    print()
    print("=" * 64)
    print("Inference: KV-cache equivalent for Mamba")
    print("=" * 64)
    measure_step()


if __name__ == "__main__":
    main()
