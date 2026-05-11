"""
Compare full Mamba-1 block vs Mamba-2 block on the same training-realistic shapes.

Both come from mamba-ssm; this benches the *full block* including conv1d, gating,
and projections — not just the scan kernel — so the numbers reflect what training
will actually see per layer.
"""
from __future__ import annotations
import argparse, time

import torch
from mamba_ssm import Mamba, Mamba2


def time_fn(make_block, x_shape, n_warmup=2, n_iter=5):
    block = make_block().to(x_shape["device"]).to(x_shape["dtype"])
    x = torch.randn(x_shape["bsz"], x_shape["L"], x_shape["d_model"],
                    device=x_shape["device"], dtype=x_shape["dtype"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(n_warmup):
        for p in block.parameters():
            if p.grad is not None: p.grad = None
        x_ = x.clone().requires_grad_(True)
        y = block(x_); y.sum().backward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        for p in block.parameters():
            if p.grad is not None: p.grad = None
        x_ = x.clone().requires_grad_(True)
        y = block(x_); y.sum().backward()
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) / n_iter * 1000
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    n_params = sum(p.numel() for p in block.parameters())
    return dt_ms, peak_gb, n_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    args = ap.parse_args()

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    device = args.device

    print(f"device={device} dtype={args.dtype}\n")
    print(f"{'config':<36} | {'Mamba1':>30} | {'Mamba2':>30}")
    print(f"{'':<36} | {'time   mem   params':>30} | {'time   mem   params':>30}")
    configs = [
        # (bsz, L, d_model, d_state)
        (32, 256,  384, 16),
        (32, 1024, 384, 16),
        (32, 1024, 384, 64),
        (32, 1024, 384, 128),
        (16, 4096, 384, 128),
    ]
    for bsz, L, d_model, d_state in configs:
        cfg_str = f"bsz={bsz} L={L} D={d_model} N={d_state}"
        line = f"{cfg_str:<36} |"
        x_shape = dict(bsz=bsz, L=L, d_model=d_model, device=device, dtype=dtype)

        # Mamba-1
        try:
            ms, gb, n = time_fn(lambda: Mamba(d_model=d_model, d_state=d_state, expand=2), x_shape)
            line += f" {ms:>7.1f}ms {gb:>5.2f}GB {n/1e6:>5.2f}M |"
        except Exception as e:
            line += f" {'ERR:'+type(e).__name__:>30} |"
        torch.cuda.empty_cache()

        # Mamba-2
        try:
            ms, gb, n = time_fn(lambda: Mamba2(d_model=d_model, d_state=d_state, expand=2,
                                                headdim=64, chunk_size=min(256, L)),
                                 x_shape)
            line += f" {ms:>7.1f}ms {gb:>5.2f}GB {n/1e6:>5.2f}M |"
        except Exception as e:
            line += f" {'ERR:'+type(e).__name__:>30} |"
        torch.cuda.empty_cache()

        print(line)


if __name__ == "__main__":
    main()
