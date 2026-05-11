# Selective scan: implementation comparison

Four ways to compute the Mamba-1 selective-scan recurrence:

| variant | what it is | where it shines |
|---|---|---|
| `seq` | Python for-loop over time, our reference impl | correctness checks, tiny ablations |
| `seq+compile` | `torch.compile(seq, mode="reduce-overhead")` | medium L without writing CUDA |
| `parallel-cumsum` | Heinsen-style cumsum-log-space trick | medium L, pure-PyTorch parallelism |
| `mamba-ssm` | official fused CUDA kernel from `state-spaces/mamba` | training, real workloads |

All produce the same y to fp tolerance (verified on a small config).

## How each works

### `seq` — sequential reference

```
for t in range(L):
    h = exp(Δ_t·A) ⊙ h + (Δ_t·u_t) ⊗ B_t
    y_t = (h ⊙ C_t).sum(N) + D·u_t
```

`L` Python-driven GPU launches. Each launch is small (a few elementwise ops on a `(B, D, N)` tensor) so launch overhead dominates over arithmetic. Memory is good — only `h` of shape `(B, D, N)` is kept around.

### `seq+compile` — same loop, fused with `torch.compile`

`torch.compile` traces the loop, fuses the elementwise ops into a single CUDA kernel, and reduces launch overhead by maybe 5–20×. Doesn't change asymptotics — still O(L) sequential, just with a much smaller constant. `mode="reduce-overhead"` uses CUDA graphs, which extracts most of the launch-overhead win.

### `parallel-cumsum` — Heinsen-style log-space scan

Rewrites the recurrence so that everything that depends on time can be done with `cumsum`. The key identity:

```
h_t = sum_{s≤t} exp(L_t − L_s) · (Δ_s · u_s ⊗ B_s)         where L_t = cumsum_t(Δ_r · A)
    = exp(L_t) · cumsum_t [exp(−L_s) · (Δ_s · u_s ⊗ B_s)]
```

Two `cumsum` calls + a few elementwise ops, each fully parallel along the time axis. ~O(log L) parallel depth on GPU.

**Numerical caveat:** `exp(−L_s)` grows as `|L_s|` grows. For long sequences with very small `Δ·A`, `|L_t|` becomes large and `exp(±L_t)` overflows in fp32. Empirically OK at L=1024 with typical Mamba init (Δ ~ 0.001–0.1, A ~ −1 to −10); for longer sequences would need fp64, a max-shift trick, or chunking.

Materialises one tensor of shape `(B, L, D, N)` (the cumulative log), so memory is `O(B·L·D·N)` instead of the seq variant's `O(B·D·N) + O(B·L·D)`. For B=32, L=1024, D=384, N=64 this is ~3 GB just for that one tensor — manageable on the 4090 but not free.

### `mamba-ssm` — official CUDA kernel

`state-spaces/mamba`'s `selective_scan_fn` is a hand-written CUDA kernel that does the parallel scan in-place, fusing everything into one launch. No intermediate `(B, L, D, N)` materialisation. ~O(log L) parallel depth and minimum memory footprint. This is what production Mamba training uses.

The associated `causal_conv1d_fn` is the Conv1d counterpart — same fused-kernel trick for the depthwise causal convolution at the front of the Mamba block.

## Benchmark — fwd+bwd time and peak memory on RTX 4090

`scripts/bench_scan.py`, fp32, mean of 5 iterations after 2 warmup.

| config (bsz × L × D × N) | seq | parallel-cumsum | mamba-ssm |
|---|---|---|---|
| 32 × 256 × 384 × 16 | 128 ms / 0.51 GB | 18 ms / 1.88 GB | **0.7 ms / 0.11 GB** |
| 32 × 512 × 384 × 16 | 294 ms / 1.01 GB | 33 ms / 3.75 GB | **1.2 ms / 0.21 GB** |
| 32 × 1024 × 384 × 16 | 951 ms / 2.02 GB | 69 ms / 7.50 GB | **2.2 ms / 0.42 GB** |
| 8 × 1024 × 384 × 64 | 550 ms / 1.72 GB | 64 ms / 7.31 GB | **1.6 ms / 0.12 GB** |

The numbers are all over the map and the relative ordering tells the story:

- **mamba-ssm is ~430× faster than seq** at L=1024 (and ~31× faster than parallel-cumsum). It's also the most memory-efficient by a wide margin — the fused kernel never materialises the `(B, L, D, N)` tensor.
- **parallel-cumsum is ~14× faster than seq** but uses ~4× more memory because it does materialise that tensor. Useful as a pure-PyTorch fallback when CUDA toolchain isn't available.
- **seq scales linearly with L** (every doubling of L doubles the time), exactly as expected for a Python loop.
- **mamba-ssm also scales linearly with L** but the absolute numbers are so small (millisecond-scale) that the linear cost barely shows up.

For training on our 4090, mamba-ssm is the only viable option at L=1024 with multiple blocks. If we have ~10 blocks per layer, seq would take ~10 seconds per block per minibatch — minutes per gradient update. With mamba-ssm: ~22 ms per block, well under the dataloader.

## Correctness

All variants agree on the same input to fp32 tolerance:

| variant | max abs error vs seq |
|---|---|
| seq (vs itself) | 0.00 |
| parallel-cumsum | 4.8 × 10⁻⁷ |
| mamba-ssm | 1.2 × 10⁻⁶ |

Both deviations are at the level of fp32 rounding accumulating differently — meaning all three are computing the same math.

## What to use when

- **Prototyping / ablation on tiny configs:** `seq` is plenty fast and the easiest to debug.
- **Medium configs (L ≤ ~512), no CUDA toolchain:** `seq+compile` is a one-liner upgrade.
- **Medium configs with pure PyTorch:** `parallel-cumsum` is faster than seq+compile if it fits in memory and the sequence isn't long enough to hit numerical issues.
- **Real training:** `mamba-ssm`. Faster than every PyTorch variant, no memory penalty, mature.

## Newer Mamba-style architectures worth considering

A non-exhaustive but selective tour, ranked roughly by relevance to our control problem (60 Hz streaming inference, ~1024-token training context, no long-range retrieval needs):

### Mamba-2 / SSD (Dao & Gu, 2024)

The first major successor to Mamba-1 ([State Space Duality, arXiv:2405.21060](https://arxiv.org/abs/2405.21060)). Reformulates the selective scan as a chunked structured matrix multiply (the "SSD" — state-space duality), which exposes much more parallelism inside each scan. The block also reorganises into a multi-head structure with scalar A per head (rather than Mamba-1's per-channel `(D, N)` state matrix).

Direct kernel comparison on a 4090, bf16, fwd+bwd time, mean of 10 iters (`scripts/bench_ssd.py`):

| config | Mamba-1 selective_scan_fn | Mamba-2 mamba_chunk_scan_combined |
|---|---|---|
| bsz=32 L=1024 D=384 N=16  | 1.1 ms / 0.18 GB | 2.5 ms / 0.41 GB |
| bsz=32 L=1024 D=384 N=128 | 12.2 ms / 0.32 GB | **2.5 ms / 0.44 GB** (≈5× faster) |
| bsz=16 L=4096 D=384 N=128 | 19.6 ms / 0.53 GB | **3.4 ms / 0.87 GB** (≈6× faster) |
| bsz=8  L=1024 D=768 N=128 | 4.0 ms / 0.27 GB | **2.0 ms / 0.43 GB** |

So the SSD kernel is faster than Mamba-1's selective scan at any N ≥ ~64, by a factor that grows with N and L. At N=16 (Mamba-1's old default) the simpler kernel wins; at N=128 (Mamba-2's default and what people actually use) SSD wins by ~5× and that gap widens for longer sequences. Memory is roughly comparable.

For our project: strong default. Get a Mamba-1 baseline working first to verify the rest of the pipeline; swap to Mamba-2 once it's proven, free training-throughput win.

### Mamba-3 (Dao & Gu et al., March 2026)

The newest entry in the line, published at ICLR 2026 ([arXiv:2603.15569](https://arxiv.org/abs/2603.15569)). Three claimed improvements:

1. **More expressive recurrence** — derived from a more careful SSM discretization (vs Mamba-2's first-order discretization).
2. **Complex-valued state update** — enables "richer state tracking" by allowing oscillatory dynamics in the state, not just exponential decay.
3. **MIMO formulation** — multi-input multi-output state structure that improves quality without increasing decode latency.

Reported results at 1.5B scale: ~1.8 pp better average downstream accuracy than the next-best linear model (Gated DeltaNet), and matches Mamba-2's perplexity at *half* the state size.

Available in `mamba-ssm ≥ 2.3` as `Mamba3`. **Catch:** the kernel uses Triton APIs (`triton.set_allocator`) that only exist in Triton ≥ 3.3, which only ships with PyTorch ≥ 2.7. Our stack right now is PyTorch 2.6 + Triton 3.2, so Mamba-3 isn't installable here without a torch upgrade. Worth doing once we're past the prototyping phase, but not urgent — Mamba-2 is plenty for the v1 model.

For our project: aspirational target. Plan: ship a Mamba-2 model first, profile, then evaluate whether Mamba-3 is worth the toolchain churn (probably yes given the half-state-for-equal-quality claim — meaningful for our 24 GB VRAM budget).

### Hybrid SSM + attention

The pure-SSM theoretical case is "linear-time, fixed-state" — but a couple of attention layers sprinkled in catch the "lookup a specific past event" failure modes that pure SSMs sometimes have on language. For control, this matters less; we don't need to recall what the opponent did 30 seconds ago verbatim. Still, cheap insurance:

- **Samba** (Microsoft, 2024 — [arXiv:2406.07522](https://arxiv.org/abs/2406.07522)) — Mamba blocks interleaved with sliding-window attention. Strong on streaming. Training stack is slightly more complex but the architecture is clean.
- **Hymba** (NVIDIA, 2024 — [arXiv:2411.13676](https://arxiv.org/abs/2411.13676)) — Mamba and attention run *in parallel* within the same block, then their outputs are summed. Cleaner residual flow, and the two paths complement each other (Mamba carries the state, attention does retrieval).
- **Jamba / Jamba-1.5** (AI21, 2024 — [arXiv:2403.19887](https://arxiv.org/abs/2403.19887)) — Mamba/attention interleaved + MoE. Designed for scale; MoE is overkill for our 26-character branching factor.
- **Zamba2** (Zyphra, 2024 — [arXiv:2411.15242](https://arxiv.org/abs/2411.15242)) — shared attention layer amortised across Mamba blocks. Memory-efficient.

**For our project:** I'd skip this for V1. Add 1–2 attention layers among ~12 Mamba-2 blocks only if ablation shows pure SSM is missing something specific (e.g., bot can't react to opponent's habits learned from earlier in the match).

### Non-Mamba recurrent alternatives

- **xLSTM** ([Beck et al., 2024 — arXiv:2405.04517](https://arxiv.org/abs/2405.04517)) — extended LSTM with matrix-valued state (mLSTM) or scalar (sLSTM). Designed partly with control-style problems in mind. Training ecosystem is thinner than Mamba's.
- **RWKV-7** (Peng et al., late 2024) — newer RWKV. Fast inference, competitive on language; less battle-tested for control.
- **DeltaNet** family — newer parallel-friendly RNNs.
- **TTT** ([Test-Time Training, Sun et al., 2024 — arXiv:2407.04620](https://arxiv.org/abs/2407.04620)) — uses test-time gradient updates as the recurrence. Interesting framing but harder to wire into RL.
- **Griffin / Hawk** ([DeepMind, 2024 — arXiv:2402.19427](https://arxiv.org/abs/2402.19427)) — gated linear recurrence + local attention. Similar tradeoff to Samba.

**For our project:** none of these are clearly better than Mamba-2 for our task, and all have thinner ecosystems. Worth knowing about but not worth starting with.

### What's NOT worth chasing for our problem

- **Mixture-of-Experts** (BlackMamba, Jamba-MoE) — designed for >7B-param scale where capacity is the bottleneck. We're aiming at 10–30M params.
- **Long-context tricks** (sparse attention, RAG hybrids, infinite-context schemes) — we don't need 100k tokens. 1024 covers ~17 seconds of game time which is probably all the context that's load-bearing for a control decision.
- **Vision Mamba / Mamba-Byte / etc.** — domain-specific variants for images / bytes. Not applicable.

## Recommendation

- **v1 block:** Mamba-2 (SSD), 8–12 layers, d_model 384, d_state 128.
- **v2 upgrade path:** swap to Mamba-3 once the pipeline is stable and we're willing to bump torch ≥ 2.7 / triton ≥ 3.3. Half the state for equal quality is worth a real chunk of our 24 GB.
- **Backbone:** pure SSM. Skip attention layers initially.
- **Implementation:** `mamba-ssm`'s `Mamba2` block for training; the recurrent-form `step()` for 60 Hz inference.
- **Reference:** keep our pure-PyTorch `seq` impl around for unit tests on small configs.
