# Long-sequence training + inference state for Mamba

Two practical questions:
1. Can we train on full 8-min Melee matches (28,800 frames at 60 Hz)?
2. What does the "KV cache equivalent" look like for streaming inference?

Both have clean answers thanks to Mamba's structure. `scripts/bench_long_seq.py` measures.

## Training: linear-time, linear-memory in L

8-layer Mamba-2 stack, `d_model=384`, `d_state=128`, bf16, on RTX 4090. Each row holds total tokens fixed (`L × bsz` ≈ 32k) so we can compare shapes:

### Standard (full activations stored)

| L | bsz | fwd+bwd | peak mem |
|---|---|---|---|
| 1024 | 32 | **43.9 ms** | 2.51 GB |
| 2048 | 16 | 43.8 ms | 2.24 GB |
| 4096 | 8 | 44.5 ms | 2.24 GB |
| 8192 | 4 | 43.6 ms | 2.23 GB |
| 16384 | 2 | 43.8 ms | 2.23 GB |

Both time and memory are flat across length — the SSD scan is genuinely linear in L. We can pick any (L, bsz) shape that fits the budget; only the *total token count per step* matters.

### With gradient checkpointing (trade compute for memory)

| L | bsz | fwd+bwd | peak mem |
|---|---|---|---|
| 8192 | 8 | 122 ms | 2.07 GB |
| 16384 | 4 | 122 ms | 2.07 GB |
| **28800** | **2** | **104 ms** | **1.83 GB** |
| 28800 | 1 | 51.5 ms | 0.97 GB |

Checkpointing costs ~3× more compute but lets us reach **full-match training (28,800 tokens = 8 minutes of game time)** at bsz=2 in under 2 GB. So if we *want* to train on whole matches, we can.

### Practical implication

We don't actually need full-match windows — most decisions in Melee depend on the last few seconds of game state, not minutes ago. Reasonable training plan:
- **Default:** sample random ~1024-token (~17 sec) windows from each match. Standard fwd+bwd, ~44 ms per step at bsz=32. Plenty of throughput.
- **If long-range matters (e.g., adapting to opponent tendencies built up over a match):** widen to 4k–8k windows or full matches with checkpointing.

The architecture lets us scale either way; we'll let the data tell us how much context matters.

## Inference: the "KV cache equivalent" for Mamba

Mamba has a streaming mode: prefill once on whatever history we have, then per-frame just advance the state. The cache that gets passed between frames is *constant size in L* — that's the core structural advantage over transformer KV cache.

For our 8-layer Mamba-2 stack at `d_model=384`, `d_state=128`, bsz=1, bf16:

```
cache for 8 layers, bsz=1: 1600.0 KB total
  per-layer conv_state:  shape (1, 1024, 4)        =     8 KB
  per-layer ssm_state:   shape (1, 12, 64, 128)    =   196 KB
```

So the entire "Mamba cache" is 1.6 MB. For comparison, a transformer KV cache for the same model at L=28,800 (full match):
```
KV cache = 2 (K+V) × n_layer (8) × L (28,800) × d_model (384) × 2 (bf16)
         ≈ 350 MB
```

Mamba's cache is **~220× smaller**, and **doesn't grow with L** — at frame 28,800 it's still 1.6 MB. The state has integrated the entire match history into a fixed-size summary; new frames advance it without lookback.

### Per-step latency

Stepping the cache forward by one frame, no_grad, 8-layer stack:

```
per-step decode:  2577 µs/step    (388 steps/sec)
60 Hz frame budget:  16667 µs
inference uses:   15.5% of budget
```

So the 4090 can do per-frame Mamba inference in ~2.6 ms, leaving ~14 ms for everything else (libmelee state extraction, action sampling, controller emission, Dolphin loopback). Comfortable margin.

A note on the latency number: 2.6 ms for *8 sequential layers* of Mamba-2 step. Each layer's `step()` is a tiny op (basically one matrix-vector product per layer for the SSM update + one conv1d advance), but the calls are launched serially from Python and that's where most of the time goes. At inference, this could be brought down further by:
- Fusing all 8 layers into a single CUDA graph
- Calling from a non-Python loop (TorchScript or compiled inference path)
- Lower-level inference runtime (e.g., torch.export or a custom C++ wrapper)

For our 60 Hz target, none of that is actually necessary — 15% of frame budget is fine.

## Summary

| | training | inference |
|---|---|---|
| **scaling in L** | linear time, linear memory | **constant** state size, constant per-step latency |
| **Mamba advantage** | gradient checkpointing makes 28,800-token training trivial | cache is 1.6 MB instead of 350 MB; per-step decode is 2.6 ms |
| **practical takeaway** | window 1k–4k tokens by default; wider only if metrics demand | per-frame inference fits comfortably inside the 60 Hz budget |

In short: long sequences aren't a problem — they're the regime Mamba was designed for. Training scales linearly; inference scales not at all (state size and step latency are independent of how many frames we've seen so far).
