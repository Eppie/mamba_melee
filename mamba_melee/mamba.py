"""
Mamba-1 selective scan + MambaBlock, pure-PyTorch reference implementation.

The "kernel" here is `selective_scan` — the input-dependent state-space recurrence
that's the heart of Mamba:

    h_t = A_t ⊙ h_{t-1} + B_t * x_t              (state update, elementwise on N)
    y_t = sum_n C_t[n] * h_t[..., n] + D * x_t   (output projection)

where the discretized parameters are
    A_t = exp(Δ_t · A)        with A < 0 (parameterized as A = -exp(A_log))
    B_t = Δ_t · B_t           (B_t per-step; Δ_t per-step)

This is implemented as a Python for-loop over time. That's O(L) *sequential*
GPU launches instead of the official CUDA kernel's fused parallel scan, so
on a 4090 it's roughly:

    bsz=32 L=256 d_model=384:    126 ms / step (one block, fwd+bwd)
    bsz=32 L=1024 d_model=384:   1.8 s / step (one block, fwd+bwd)

…which is fine for prototyping and ablations on small configs, but not viable
for L=1024 training with multiple blocks. To swap in the fast path:

    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

and replace the call in MambaBlock.forward. Same math, same input layout
(modulo a transpose: mamba-ssm uses (B, D, L), this code uses (B, L, D)).

Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State
Spaces" (2023). https://arxiv.org/abs/2312.00752
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def selective_scan(
    u: torch.Tensor,        # (B, L, D)        input sequence
    delta: torch.Tensor,    # (B, L, D)        per-step discretization Δ > 0
    A: torch.Tensor,        # (D, N)           state matrix, typically negative
    B: torch.Tensor,        # (B, L, N)        input-conditioned input projection
    C: torch.Tensor,        # (B, L, N)        input-conditioned output projection
    D: torch.Tensor,        # (D,)             direct skip / feedthrough
) -> torch.Tensor:
    """
    Mamba-1 selective scan. Sequential reference implementation.

    Returns y of shape (B, L, D).

    Memory: keeps state h of shape (B, D, N) and accumulates outputs; never
    materialises the (B, L, D, N) discretised-A tensor. The Python loop is
    the bottleneck, not memory.
    """
    bsz, L, d_inner = u.shape
    d_state = A.shape[1]

    h = u.new_zeros(bsz, d_inner, d_state)
    ys: list[torch.Tensor] = []
    for t in range(L):
        delta_t = delta[:, t]                                    # (B, D)
        # A_bar_t = exp(Δ_t · A)
        A_bar_t = torch.exp(delta_t.unsqueeze(-1) * A)           # (B, D, N)
        # B_bar_t · x_t = (Δ_t · x_t) ⊗ B_t
        Bx_t = (delta_t * u[:, t]).unsqueeze(-1) * B[:, t].unsqueeze(1)  # (B, D, N)
        # Recurrence
        h = A_bar_t * h + Bx_t
        # Output: y_t = (h_t · C_t).sum(N) + D · x_t  (skip added at end)
        y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)             # (B, D)
        ys.append(y_t)

    y = torch.stack(ys, dim=1)                                   # (B, L, D)
    y = y + u * D
    return y


class MambaBlock(nn.Module):
    """
    Mamba-1 block (paper Figure 3, right).

    Layout:
        x ──> in_proj ──> split into [x', z]
        x' ──> depthwise causal Conv1d ──> SiLU ──> selective_scan ──> y
        y * SiLU(z) ──> out_proj ──> output

    The selective parameters (Δ, B, C) are produced by linear projections of
    the post-conv signal; A and D are learned parameters of the block.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # Input projection produces (x', z), each of size d_inner
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Depthwise causal Conv1d: mixes adjacent tokens per channel
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,  # causal: pad on left, crop on right
            bias=True,
        )

        # Project x' to (delta_input, B, C) — the selective parameters
        # delta is rank-reduced (dt_rank < d_inner) then projected up to d_inner
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialise dt_proj.weight small so initial Δ is dominated by the bias,
        # which we set so softplus(bias) ~ Uniform(dt_min, dt_max). This avoids
        # the model collapsing to A_bar ≈ 1 (no dynamics) early in training.
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        # Inverse-softplus: bias such that softplus(bias) = dt
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A: state matrix, parameterised as A = -exp(A_log) so it's always negative.
        # Init to a "HIPPO-like" geometric spread: A_d,n = -(n+1).
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # D: direct skip-connection scalar per channel
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection back to d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d_model)
        returns: (B, L, d_model)
        """
        bsz, L, _ = x.shape

        # Input projection -> split into (x', z)
        xz = self.in_proj(x)                                     # (B, L, 2*d_inner)
        x_prime, z = xz.chunk(2, dim=-1)                         # each (B, L, d_inner)

        # Causal Conv1d. nn.Conv1d expects (B, C, L). Padding on the left
        # makes the conv strictly causal once we crop the trailing positions.
        x_prime = x_prime.transpose(1, 2)                        # (B, d_inner, L)
        x_prime = self.conv1d(x_prime)[:, :, :L]                 # crop right pad
        x_prime = x_prime.transpose(1, 2)                        # (B, L, d_inner)
        x_prime = F.silu(x_prime)

        # Project to selective params
        x_dbl = self.x_proj(x_prime)                             # (B, L, dt_rank + 2*N)
        delta_in, B, C = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        # Δ > 0
        delta = F.softplus(self.dt_proj(delta_in))               # (B, L, d_inner)

        # Negative A
        A = -torch.exp(self.A_log.float())                       # (d_inner, d_state)

        # Selective scan
        y = selective_scan(x_prime, delta, A, B, C, self.D)      # (B, L, d_inner)

        # Gate + output projection
        y = y * F.silu(z)
        return self.out_proj(y)


# --------------------------------------------------------------------------
# Self-tests (run as `python -m mamba_melee.mamba`)
# --------------------------------------------------------------------------

def _check_selective_scan() -> None:
    """Forward + backward shape/finiteness check on the kernel."""
    torch.manual_seed(0)
    bsz, L, d_inner, d_state = 2, 16, 8, 4
    u = torch.randn(bsz, L, d_inner, requires_grad=True)
    delta = (torch.rand(bsz, L, d_inner) * 0.1).requires_grad_(True)
    A = (-torch.rand(d_inner, d_state) - 0.1).requires_grad_(True)
    B = torch.randn(bsz, L, d_state, requires_grad=True)
    C = torch.randn(bsz, L, d_state, requires_grad=True)
    D = torch.randn(d_inner, requires_grad=True)

    y = selective_scan(u, delta, A, B, C, D)
    assert y.shape == (bsz, L, d_inner), f"got {y.shape}"
    assert torch.isfinite(y).all(), "non-finite forward output"

    y.sum().backward()
    for name, t in [("u", u), ("delta", delta), ("A", A), ("B", B), ("C", C), ("D", D)]:
        assert t.grad is not None, f"no grad for {name}"
        assert torch.isfinite(t.grad).all(), f"non-finite grad for {name}"
    print("selective_scan: forward + backward OK")


def _check_recurrence_equivalence() -> None:
    """
    Verify the parallel batched implementation matches an even-more-naive
    explicit recurrence written one element at a time. This pins down the
    math, independent of the einsum/broadcast tricks above.
    """
    torch.manual_seed(1)
    bsz, L, d_inner, d_state = 1, 8, 3, 2
    u = torch.randn(bsz, L, d_inner)
    delta = torch.rand(bsz, L, d_inner) * 0.1
    A = -torch.rand(d_inner, d_state) - 0.1
    B = torch.randn(bsz, L, d_state)
    C = torch.randn(bsz, L, d_state)
    D = torch.randn(d_inner)

    # Reference: explicit per-(b,d,n) recurrence
    y_ref = torch.zeros(bsz, L, d_inner)
    for b in range(bsz):
        h = torch.zeros(d_inner, d_state)
        for t in range(L):
            for d in range(d_inner):
                for n in range(d_state):
                    a_bar = math.exp(delta[b, t, d].item() * A[d, n].item())
                    bx    = delta[b, t, d].item() * B[b, t, n].item() * u[b, t, d].item()
                    h[d, n] = a_bar * h[d, n] + bx
                y_ref[b, t, d] = sum(
                    C[b, t, n].item() * h[d, n].item() for n in range(d_state)
                ) + D[d].item() * u[b, t, d].item()

    y = selective_scan(u, delta, A, B, C, D)
    err = (y - y_ref).abs().max().item()
    print(f"selective_scan vs naive ref: max abs error = {err:.2e}")
    assert err < 1e-4, f"divergence: {err}"


def _check_block(device: str = "cpu") -> None:
    """Forward + backward through the full MambaBlock."""
    torch.manual_seed(2)
    bsz, L, d_model = 2, 32, 64
    block = MambaBlock(d_model=d_model, d_state=8, d_conv=4, expand=2).to(device)
    x = torch.randn(bsz, L, d_model, device=device, requires_grad=True)
    y = block(x)
    assert y.shape == (bsz, L, d_model), f"got {y.shape}"
    assert torch.isfinite(y).all(), "non-finite forward"
    y.sum().backward()
    for name, p in block.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
    n_params = sum(p.numel() for p in block.parameters())
    print(f"MambaBlock on {device}: forward + backward OK ({n_params:,} params)")


def _check_causality() -> None:
    """The block must be strictly causal: changing input at time t cannot affect output at time < t."""
    torch.manual_seed(3)
    bsz, L, d_model = 1, 16, 32
    block = MambaBlock(d_model=d_model, d_state=4, d_conv=4, expand=2).eval()
    x1 = torch.randn(bsz, L, d_model)
    x2 = x1.clone()
    t_perturb = 8
    x2[:, t_perturb] = torch.randn(bsz, d_model)
    with torch.no_grad():
        y1 = block(x1)
        y2 = block(x2)
    diff = (y1 - y2).abs().max(dim=-1).values[0]  # (L,)
    # Outputs at indices < t_perturb should be identical
    assert diff[:t_perturb].max().item() < 1e-5, \
        f"block is not causal: pre-perturbation diffs = {diff[:t_perturb]}"
    # Outputs at indices >= t_perturb should differ
    assert diff[t_perturb:].max().item() > 1e-3, \
        f"perturbation had no effect at all: {diff[t_perturb:]}"
    print("MambaBlock causality OK")


if __name__ == "__main__":
    _check_selective_scan()
    _check_recurrence_equivalence()
    _check_block(device="cpu")
    _check_causality()
    if torch.cuda.is_available():
        _check_block(device="cuda")
    print("\nall checks passed")
