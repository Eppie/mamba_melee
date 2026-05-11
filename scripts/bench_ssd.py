"""
Compare Mamba-1 selective scan kernel vs Mamba-2 SSD kernel at training shapes.

Mamba-2 (SSD) has a different parameterization: multi-head with shared state
across head channels, scalar A per head (vs Mamba-1's (D, N) state matrix).
This is what makes it ~2-3x faster than Mamba-1 in practice.

We bench fwd+bwd time + peak memory for both, at d_model=384, L=1024, bf16
(typical training settings on a 4090).
"""
from __future__ import annotations
import time
import torch

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


def time_fn(fn, args, n_warmup=2, n_iter=10):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    def reset():
        for a in args:
            if isinstance(a, torch.Tensor) and a.grad is not None:
                a.grad = None
    for _ in range(n_warmup):
        reset(); y = fn(*args); y.sum().backward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        reset(); y = fn(*args); y.sum().backward()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_iter * 1000
    gb = torch.cuda.max_memory_allocated() / 1e9
    return dt, gb


def make_mamba1_inputs(bsz, L, D, N, dtype, device):
    g = torch.Generator(device=device).manual_seed(0)
    u     = torch.randn(bsz, D, L, device=device, dtype=dtype, generator=g, requires_grad=True)
    delta = (torch.rand(bsz, D, L, device=device, dtype=dtype, generator=g) * 0.1).requires_grad_(True)
    A     = (-torch.rand(D, N, device=device, dtype=torch.float32, generator=g) - 0.1).requires_grad_(True)
    B     = torch.randn(bsz, 1, N, L, device=device, dtype=dtype, generator=g, requires_grad=True)
    C     = torch.randn(bsz, 1, N, L, device=device, dtype=dtype, generator=g, requires_grad=True)
    D_p   = torch.randn(D, device=device, dtype=torch.float32, generator=g, requires_grad=True)
    return u, delta, A, B, C, D_p


def make_mamba2_inputs(bsz, L, D, N, headdim, ngroups, chunk_size, dtype, device):
    H = D // headdim
    g = torch.Generator(device=device).manual_seed(0)
    x  = torch.randn(bsz, L, H, headdim, device=device, dtype=dtype, generator=g, requires_grad=True)
    dt = (torch.rand(bsz, L, H,           device=device, dtype=dtype, generator=g) * 0.1).requires_grad_(True)
    A  = (-torch.rand(H,                  device=device, dtype=torch.float32, generator=g) - 0.1).requires_grad_(True)
    B  = torch.randn(bsz, L, ngroups, N,  device=device, dtype=dtype, generator=g, requires_grad=True)
    C  = torch.randn(bsz, L, ngroups, N,  device=device, dtype=dtype, generator=g, requires_grad=True)
    return x, dt, A, B, C, chunk_size


def main():
    device = "cuda"
    print(f"device={device}, fwd+bwd time, mean of 10 iters\n")
    print(f"{'config':<46} | {'Mamba-1 kernel':>22} | {'Mamba-2 SSD kernel':>22}")

    configs = [
        # (bsz, L, D, N_for_mamba1, N_for_mamba2, headdim, ngroups, chunk, dtype)
        (32, 256,  384,  16, 128, 64, 1, 256, torch.bfloat16),
        (32, 1024, 384,  16, 128, 64, 1, 256, torch.bfloat16),
        (32, 1024, 384, 128, 128, 64, 1, 256, torch.bfloat16),
        (16, 4096, 384, 128, 128, 64, 1, 256, torch.bfloat16),
        ( 8, 1024, 768, 128, 128, 64, 1, 256, torch.bfloat16),  # bigger d_model
    ]
    for bsz, L, D, N1, N2, hd, ng, ck, dt in configs:
        cfg = f"bsz={bsz} L={L} D={D} N1={N1} N2={N2}"
        line = f"{cfg:<46} |"

        try:
            args1 = make_mamba1_inputs(bsz, L, D, N1, dt, device)
            ms, gb = time_fn(
                lambda u, dl, A, B, C, Dp: selective_scan_fn(u, dl, A, B, C, Dp, z=None,
                                                              delta_bias=None, delta_softplus=False),
                args1)
            line += f" {ms:>9.2f}ms {gb:>5.2f}GB |"
        except Exception as e:
            line += f" {'ERR:'+type(e).__name__:<22} |"
        torch.cuda.empty_cache()

        try:
            args2 = make_mamba2_inputs(bsz, L, D, N2, hd, ng, ck, dt, device)
            ms, gb = time_fn(
                lambda x, dt_, A, B, C, ck_: mamba_chunk_scan_combined(x, dt_, A, B, C, ck_),
                args2)
            line += f" {ms:>9.2f}ms {gb:>5.2f}GB |"
        except Exception as e:
            line += f" {'ERR:'+type(e).__name__:<22} |"
        torch.cuda.empty_cache()
        print(line)


if __name__ == "__main__":
    main()
