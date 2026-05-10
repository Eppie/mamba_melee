# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

Train a **Mamba-based** neural network to play Super Smash Bros. Melee from **raw game state** (no pixels). Two phases:

1. **Imitation learning** from `.slp` replay files (current phase).
2. **Reinforcement learning** against in-game CPUs / self-play via Dolphin (later phase).

The architecture itself is not yet decided — selecting the Mamba variant (Mamba-1, Mamba-2, hybrid w/ attention), input tokenization of game state, and action head design are open design questions. Prior work on Melee bots (e.g. phillip, slippi-ai) is **reference only** — do not copy verbatim; justify deviations and similarities.

## Stack

- **Python env**: managed by `uv` (create with `uv venv`, install with `uv pip install` or `uv add`).
- **DL framework**: PyTorch.
- **Melee interfacing**: [libmelee](https://github.com/altf4/libmelee) for Dolphin control + slippi state parsing; Dolphin (Slippi-Ishiiruka or Slippi-Mainline) for emulation and RL rollouts.
- **Replay parsing for IL**: `peppi-py` or `py-slippi` are the usual choices for offline `.slp` parsing without booting Dolphin — pick based on speed/feature needs when that work begins.

## Hardware constraints

- Single machine, **single RTX 4090 (24 GB VRAM)**. All architecture, batch-size, and context-length choices must fit this budget. No multi-GPU assumptions.
- GPU monitoring during training:
  ```
  /lib/wsl/lib/nvidia-smi dmon -s pucvmt -d 1
  ```
  (`pucvmt` = power, util, clocks, violations, mem, temp; `-d 1` = 1s interval). Use this when diagnosing throughput / thermal issues.
- Platform is WSL2 on Linux — Dolphin GUI/audio may need extra setup; for headless RL rollouts prefer Dolphin's headless/exi-AI mode.

## Working notes

- **No code exists yet.** When bootstrapping, default layout: `data/` (raw + processed replays, gitignored), `src/mamba_melee/` (package), `scripts/` (training/eval entrypoints), `configs/` (yaml/hydra). Don't create scaffolding speculatively — only what the immediate task needs.
- Replay datasets get large fast. Plan for an offline preprocessing step (`.slp` → tensor shards on disk) rather than parsing during training.
- For Mamba: the official `mamba-ssm` package needs CUDA + a matching PyTorch build. On the 4090 (Ada / sm_89) confirm the wheel matches CUDA toolkit before installing into the uv venv.

## Replay format upgrade (`.slp` regeneration)

**Why it exists:** older `.slp` files lack newer fields (items 3.0+, velocity components 3.5+, instance IDs 3.16+, raw c-stick 3.17+, stage-specific events 3.18+). We can re-run them through the current Slippi Playback Dolphin to emit a new `.slp` at the latest format version.

**Mechanism:** Slippi Playback Dolphin (the `IS_PLAYBACK` build) has an undocumented `SlippiRegenerateReplays = True` config setting. When set, it writes a fresh `.slp` to `SlippiReplayRegenerateDir` while playing back a replay supplied via comm-file (`-i comm.json`).

**What is preserved (verified across all 12,487 frames of one replay):**
- All post-frame fields: positions, percent, stocks, action state, animation index, hurtbox state, shield, velocities, etc. — bit-exact.
- Pre-frame digital `buttons` (the field the character action engine responds to), `random_seed`, `raw_analog_x/y` stick values, joystick + cstick coordinates.
- Regeneration is deterministic — two runs of the same source produce bit-identical output.

**What is NOT preserved:**
- `buttons_physical` is zero on ~25% of frames where source had real button bits set. Hypothesis: the recording ASM reads physical state from a controller-side register that Dolphin doesn't populate during input replay. The digital `buttons` field captures the same press correctly, so character behaviour is unaffected — but anything that consumed `buttons_physical` directly (notably Slippi's UCF dashback/shield-drop detection logic recording) gets corrupted signal. **For ML training, use digital `buttons` + raw stick values; treat `buttons_physical` as unreliable from regenerated data.**

**Pipeline:**
- Driver: `scripts/regen_batch.py` — multiprocessing pool of N workers, each runs `xvfb-run + Slippi Playback AppImage` on one replay at a time. See README for setup + invocation.
- Comm-file `mode: "mirror"` is the critical setting — it triggers Hard FFW (`SlippiPlayback.cpp:766`), giving ~10× speedup over the playback default (which silently overrides `EmulationSpeed` to 1.0 at `EXI_DeviceSlippi.cpp:1990`).
- Binary acquisition: `scripts/setup_playback.sh` downloads `Slippi_Playback-x86_64.AppImage` v3.5.2 and extracts it to `.playback/squashfs-root/`.
- Completion detection: poll for `[GAME_END_FRAME]` in dolphin stdout (requires `--cout` flag), then 6s flush wait.
- A locally-rebuilt `slippi-Ishiiruka` nogui binary with `IS_PLAYBACK=ON` **does not produce playback output** despite Core::IsRunning succeeding. The xvfb + GUI AppImage path is the working approach. (Patched MainNoGUI.cpp to add `-i` flag and call `VideoBackendBase::ActivateBackend` — neither fixed it; root cause unknown.)
- Throughput (smoke test, 8 workers, 16 replays): 0.6 min wall-clock = ~26 replays/min. Full 1200-replay batch ≈ 45 min.

**Output completeness — verified explanation:**
The regenerated replay's pre/post-frame count equals the source's count of **unique (frame_index, port) pairs**, in every case. The source-vs-out frame-count gap is rollback duplicates being deduplicated. Confirmed by inspection: source replays with N "missing" frames after regen have exactly N rollback duplicates (frames where the same (frame_idx, port) appears 4× or 6× because the netplay client rolled back and re-played them). Frame ranges (min/max frame index) are identical between src and out.

For ML purposes, **the regenerated timeline is preferable** to the source — no stale rollback states polluting the data. Don't filter on frame-count delta; it just measures how rollback-heavy the source match was.
