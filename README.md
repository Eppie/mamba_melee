# mamba_melee

Train a Mamba-based neural network to play Super Smash Bros. Melee from raw game state.

Two phases:
1. **Imitation learning** from `.slp` replay files (current).
2. **Reinforcement learning** against in-game CPUs / self-play via Dolphin (later).

Architecture is not yet decided. See `CLAUDE.md` for design notes and constraints.

## What's here

- `scripts/regen_batch.py` — parallel `.slp` regenerator. Plays old replays through current Slippi Playback Dolphin and writes them out at the latest format version (v3.19), unlocking newer fields like instance IDs (3.16+), raw c-stick (3.17+), and stage-specific events (3.18+). Also deduplicates rollback frames.
- `scripts/Dolphin.ini.template` — Dolphin config template used by each worker (parameterised on ISO + regen output dir).
- `scripts/setup_playback.sh` — downloads and extracts the Slippi Playback AppImage.
- `CLAUDE.md` — full project context and design notes.

## Replay regeneration

### One-time setup

```bash
# 1. Get the Slippi Playback Dolphin (downloads ~75MB AppImage from GitHub)
./scripts/setup_playback.sh
export SLIPPI_PLAYBACK_DIR=$PWD/.playback/squashfs-root

# 2. Point at your Melee NTSC 1.02 ISO
export MELEE_ISO=/path/to/melee.iso

# 3. Install xvfb + Python deps
sudo apt-get install -y xvfb
uv venv && source .venv/bin/activate
uv pip install peppi-py numpy   # only needed for verification scripts
```

### Run

```bash
python scripts/regen_batch.py /path/to/raw_slp_dir /path/to/output_dir \
    --workers 8 --timeout 240
```

Throughput on a 32-core machine: ~25 replays/min (≈1 hour for 1500 replays). Each worker uses ~1.5 GB RAM (mostly Dolphin emulator state).

### What's preserved vs. what isn't

**Preserved bit-exact across all frames:**
- Positions, percent, stocks, action state, animation index, hurtbox state, shield, velocities
- Pre-frame digital `buttons` (the field the character action engine responds to), `random_seed`, raw analog stick values

**Not preserved:**
- `buttons_physical` is zero on ~25% of frames where source had press bits set. The digital `buttons` field captures the same press correctly, so character behaviour is unaffected — but anything that consumed `buttons_physical` directly (Slippi's UCF dashback / shield-drop detection logic) sees corrupted signal. **For ML training, use digital `buttons` + raw stick values; treat `buttons_physical` from regenerated data as unreliable.**

**Frame-count differences are rollback deduplication:** the regenerated replay's pre/post-frame count equals the source's count of unique `(frame_index, port)` pairs. Source replays from netplay record every frame the client computed — including frames that were rolled back and re-played due to network timing. Regen produces only the canonical timeline (same min/max frame index as source). This is preferable for ML — no stale rollback states polluting the dataset.

## Why this exists (and why the approach is unusual)

`.slp` is an input-driven format: pre-frame controller inputs + per-frame RNG seeds (added in 2.2.0) are sufficient to deterministically reproduce a match given Slippi-modified Melee ASM. So if you take an old replay (say v3.14) and re-run its inputs through current Playback Dolphin's recording path, you get a new replay at the current format version with all the fields the new build knows how to emit.

Slippi Playback Dolphin has an **undocumented** `SlippiRegenerateReplays = True` config setting that does exactly this. It's gated on the `IS_PLAYBACK` compile flag and was almost certainly added by Slippi devs for internal QA when the format changed.

The gotcha that took the longest to find: the playback build silently overrides `EmulationSpeed` to 1.0 (forces real-time), making naive use of the regenerate feature ~10× slower than necessary. Setting the comm-file `mode: "mirror"` triggers Hard FFW (`Source/Core/Core/Slippi/SlippiPlayback.cpp:766`), which speeds it up.

A full headless `nogui` build with `IS_PLAYBACK=ON` would be cleaner than the xvfb + GUI approach used here, but the nogui front-end in vladfi1's exi-ai-rebase fork hangs after BootCore in playback mode for reasons I haven't tracked down. xvfb works fine.
