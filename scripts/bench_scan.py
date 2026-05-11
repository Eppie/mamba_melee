"""
Compare selective scan implementations: correctness + fwd+bwd time + peak memory.

Variants exercised:
  - seq          — Python for-loop reference
  - seq+compile  — torch.compile wrap of the for-loop
  - parallel-cumsum — Heinsen-style parallel scan in pure PyTorch
  - mamba-ssm    — official CUDA fused kernel (if installed)

Correctness: all variants compared to seq, max abs error reported.
"""
from __future__ import annotations

import argparse
import time

import torch

from mamba_melee.scan_variants import VARIANTS, selective_scan_seq


def make_inputs(bsz, L, d_inner, d_state, device, dtype=torch.float32, requires_grad=True):
    g = torch.Generator(device=device).manual_seed(0)
    u = torch.randn(bsz, L, d_inner, device=device, dtype=dtype, generator=g)
    delta = torch.rand(bsz, L, d_inner, device=device, dtype=dtype, generator=g) * 0.1
    A = -torch.rand(d_inner, d_state, device=device, dtype=dtype, generator=g) - 0.1
    B = torch.randn(bsz, L, d_state, device=device, dtype=dtype, generator=g)
    C = torch.randn(bsz, L, d_state, device=device, dtype=dtype, generator=g)
    D = torch.randn(d_inner, device=device, dtype=dtype, generator=g)
    if requires_grad:
        for t in (u, delta, A, B, C, D):
            t.requires_grad_(True)
    return u, delta, A, B, C, D


def time_fn(fn, args, n_warmup=2, n_iter=5):
    """Time fwd + bwd of fn(*args). Returns (mean_ms, peak_mem_GB)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # Reset grads on inputs so backward works each iter
    def reset_grads():
        for t in args:
            if t.grad is not None:
                t.grad = None
    for _ in range(n_warmup):
        reset_grads()
        y = fn(*args)
        y.sum().backward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        reset_grads()
        y = fn(*args)
        y.sum().backward()
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) / n_iter * 1000
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    return dt_ms, peak_gb


def correctness(args, ref_y):
    """For each variant, check max abs error vs ref_y in fp32."""
    out = {}
    for name, fn in VARIANTS.items():
        try:
            with torch.no_grad():
                y = fn(*args)
            err = (y - ref_y).abs().max().item()
            out[name] = err
        except Exception as e:
            out[name] = f"ERROR: {type(e).__name__}: {e}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    print(f"Variants present: {list(VARIANTS)}")
    print(f"Device: {args.device}")

    # --- Correctness check on a tiny config ---
    print("\n=== Correctness vs sequential reference (fp32) ===")
    inps = make_inputs(2, 32, 16, 8, args.device, requires_grad=False)
    with torch.no_grad():
        ref = selective_scan_seq(*inps)
    errs = correctness(inps, ref)
    for k, v in errs.items():
        print(f"  {k:<20} max_abs_err = {v if isinstance(v, str) else f'{v:.2e}'}")

    # --- Timing across configs ---
    configs = [
        (32, 256,  384, 16),
        (32, 512,  384, 16),
        (32, 1024, 384, 16),
        (8,  1024, 384, 64),
    ]
    print(f"\n=== fwd+bwd timing (mean of 5 iters after 2 warmup) ===")
    print(f"  {'config':<32} | " + " | ".join(f"{name:<20}" for name in VARIANTS))
    for bsz, L, d_inner, d_state in configs:
        cfg_str = f"bsz={bsz} L={L} D={d_inner} N={d_state}"
        line = f"  {cfg_str:<32} |"
        for name, fn in VARIANTS.items():
            inps = make_inputs(bsz, L, d_inner, d_state, args.device)
            try:
                ms, gb = time_fn(fn, inps, n_warmup=1, n_iter=3)
                line += f" {ms:>7.1f}ms {gb:>5.2f}GB |"
            except torch.cuda.OutOfMemoryError:
                line += f" {'OOM':<20} |"
            except Exception as e:
                line += f" {'ERR:'+type(e).__name__:<20} |"
            torch.cuda.empty_cache()
        print(line)


if __name__ == "__main__":
    main()
