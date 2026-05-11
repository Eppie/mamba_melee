"""
Multiple implementations of the Mamba-1 selective scan, for benchmarking and
verification. All have the same signature and produce the same result (up to
fp tolerance):

    selective_scan_seq      — Python for-loop, reference
    selective_scan_compiled — same but wrapped in torch.compile
    selective_scan_parallel — cumsum-log-space trick, parallel over time
    selective_scan_ssm      — official mamba-ssm CUDA fused kernel (if installed)

All take (u, delta, A, B, C, D) with shapes:
    u, delta: (B, L, D)
    A:        (D, N)
    B, C:     (B, L, N)
    D:        (D,)
"""
from __future__ import annotations

import torch

from mamba_melee.mamba import selective_scan as selective_scan_seq

# --------------------------------------------------------------------------
# torch.compile of the sequential implementation
# --------------------------------------------------------------------------

_compiled = None


def selective_scan_compiled(u, delta, A, B, C, D):
    global _compiled
    if _compiled is None:
        _compiled = torch.compile(selective_scan_seq, mode="reduce-overhead", dynamic=False)
    return _compiled(u, delta, A, B, C, D)


# --------------------------------------------------------------------------
# Parallel-scan via cumsum-log-space trick (Heinsen 2023 style)
# --------------------------------------------------------------------------

def selective_scan_parallel(u, delta, A, B, C, D):
    """
    Mamba-1 selective scan, parallelized over time via cumsum-log-space.

    Math:
        A_bar_r = exp(Δ_r · A),  with Δ > 0 and A < 0  →  A_bar_r ∈ (0, 1)
        log A_bar_r = Δ_r · A                          → ≤ 0
        L_t = sum_{r≤t} log A_bar_r                    → monotone non-positive

        h_t = sum_{s≤t} exp(L_t - L_s) · (B_bar_s · u_s)
            = exp(L_t) · cumsum_t [exp(-L_s) · (Δ_s · B_s · u_s)]

    Numerical caveat: exp(-L_s) grows as |L_s| grows. For long sequences with
    small Δ·A, can overflow in fp32. Empirically OK for L ≲ 1024 with typical
    Mamba init; for longer or for fp16 use a chunked variant.
    """
    bsz, L, d_inner = u.shape
    d_state = A.shape[1]

    # log A_bar (per-step log of the discretized state matrix)
    # Shape: (B, L, D, N)
    log_A_bar = delta.unsqueeze(-1) * A  # broadcast (B, L, D, 1) * (D, N) → (B, L, D, N)

    # Cumulative log along time
    cum_log = log_A_bar.cumsum(dim=1)  # (B, L, D, N)

    # B_bar_s · u_s = (Δ_s · u_s) ⊗ B_s
    Bx = (delta * u).unsqueeze(-1) * B.unsqueeze(2)  # (B, L, D, N)

    # h_t = exp(L_t) · cumsum_t [exp(-L_s) · Bx_s]
    # Direct computation (vulnerable to over/underflow for large |L|)
    weighted = torch.exp(-cum_log) * Bx
    cum_weighted = weighted.cumsum(dim=1)
    h = torch.exp(cum_log) * cum_weighted  # (B, L, D, N)

    # y_t = sum_n C_t[n] · h_t[..., n] + D · u_t
    y = (h * C.unsqueeze(2)).sum(dim=-1)  # (B, L, D)
    y = y + u * D
    return y


# --------------------------------------------------------------------------
# Official mamba-ssm CUDA kernel wrapper
# --------------------------------------------------------------------------

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _ssm_fn

    HAS_MAMBA_SSM = True
except ImportError:
    HAS_MAMBA_SSM = False
    _ssm_fn = None


def selective_scan_ssm(u, delta, A, B, C, D):
    """
    Wrap mamba-ssm's CUDA fused selective_scan_fn with our (B, L, D) layout.
    mamba-ssm uses (B, D, L) for u/delta and (B, N, L) for B/C.
    """
    if not HAS_MAMBA_SSM:
        raise RuntimeError("mamba-ssm not installed")
    # Transpose to mamba-ssm's layout
    u_t = u.transpose(1, 2).contiguous()        # (B, D, L)
    delta_t = delta.transpose(1, 2).contiguous()  # (B, D, L)
    B_t = B.transpose(1, 2).unsqueeze(1).contiguous()  # (B, 1, N, L) — single SSM group
    C_t = C.transpose(1, 2).unsqueeze(1).contiguous()  # (B, 1, N, L)
    y = _ssm_fn(
        u_t, delta_t, A, B_t, C_t, D, z=None,
        delta_bias=None, delta_softplus=False,
    )
    return y.transpose(1, 2).contiguous()  # (B, L, D)


# --------------------------------------------------------------------------
# Registry for benchmarking
# --------------------------------------------------------------------------

VARIANTS = {
    "seq": selective_scan_seq,
    "parallel-cumsum": selective_scan_parallel,
}
# selective_scan_compiled (torch.compile) is not in VARIANTS by default — compiling
# a 1024-iteration Python for-loop unrolls into 1024 ops and takes ~10 min to compile
# without a meaningful speedup. The seq impl shape just isn't a good fit for compile.
if HAS_MAMBA_SSM:
    VARIANTS["mamba-ssm"] = selective_scan_ssm
