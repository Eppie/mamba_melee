"""
Analyze controller-input distributions in regenerated Fox-ditto replays.
Output is consumed by notes/input_design.md.

Reads every .slp under <src>, pools per-port pre-frame data, computes:
  - main stick + c-stick magnitude / angle / 2D bin histograms
  - per-button press rate, "just pressed" rate, average hold length
  - L/R analog trigger distribution
  - coordination: stick magnitude conditional on A / B / Z press
  - top action states (post-frame state field)

Designed to inform the action head bin layout, which heads to fold together,
and which classes are rare enough to need weighting.
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from peppi_py import read_slippi


# Slippi physical button bits (from spec)
BITS = {
    "DpadL": 0x0001, "DpadR": 0x0002, "DpadD": 0x0004, "DpadU": 0x0008,
    "Z":     0x0010, "R":     0x0020, "L":     0x0040,
    "A":     0x0100, "B":     0x0200, "X":     0x0400, "Y":     0x0800,
    "Start": 0x1000,
}

# Proposed stick bin edges, snapped to mechanically meaningful thresholds
# (deadzone ~0.28, smash ~0.8)
STICK_EDGES = np.array([-1.0, -0.8, -0.5, -0.28, 0.0, 0.28, 0.5, 0.8, 1.0])
CSTICK_EDGES = np.array([-1.0, -0.7, 0.0, 0.7, 1.0])  # 4-class: full-, mid-, mid+, full+
MAG_EDGES = np.array([0.0, 0.05, 0.15, 0.28, 0.5, 0.7, 0.8, 0.95, 1.01])


def parse_one(path: Path) -> list[dict] | None:
    try:
        g = read_slippi(str(path))
    except Exception as e:
        print(f"  SKIP {path.name}: {e}", file=sys.stderr)
        return None
    out = []
    for port in (0, 1):
        pre = g.frames.ports[port].leader.pre
        post = g.frames.ports[port].leader.post
        # NOTE: triggers_physical.l/r is NOT preserved by the regenerate path
        # (same class of issue as buttons_physical). Use `triggers` scalar (the
        # effective post-processing value) which IS preserved across regen.
        t = np.asarray(pre.triggers)
        tl = t; tr = np.zeros_like(t)
        out.append({
            "jx":      np.asarray(pre.joystick.x),
            "jy":      np.asarray(pre.joystick.y),
            "cx":      np.asarray(pre.cstick.x),
            "cy":      np.asarray(pre.cstick.y),
            "tl":      tl,
            "tr":      tr,
            "buttons": np.asarray(pre.buttons),
            "state":   np.asarray(post.state),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--out", type=Path, default=Path("notes/input_stats.json"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    files = sorted(args.src.glob("*.slp"))
    if args.limit:
        files = files[:args.limit]
    print(f"parsing {len(files)} replays from {args.src} ...")

    chunks = {k: [] for k in ["jx", "jy", "cx", "cy", "tl", "tr", "buttons", "state"]}
    match_lens = []
    n_skipped = 0
    for i, p in enumerate(files):
        if i and i % 100 == 0:
            print(f"  {i}/{len(files)}")
        per_port = parse_one(p)
        if per_port is None:
            n_skipped += 1
            continue
        match_lens.append(len(per_port[0]["jx"]))
        for d in per_port:
            for k in chunks:
                chunks[k].append(d[k])

    arrs = {k: np.concatenate(v) for k, v in chunks.items()}
    n_frames = len(arrs["jx"])
    print(f"\ntotal port-frames: {n_frames:,} (skipped {n_skipped} replays)")

    stats = {
        "dataset": {
            "n_replays_parsed": len(files) - n_skipped,
            "n_replays_skipped": n_skipped,
            "n_port_frames": int(n_frames),
            "avg_match_len_frames": float(np.mean(match_lens)),
            "avg_match_len_seconds": float(np.mean(match_lens) / 60.0),
            "median_match_len_seconds": float(np.median(match_lens) / 60.0),
        },
    }

    # --- Main stick ---
    jx, jy = arrs["jx"], arrs["jy"]
    mag = np.sqrt(jx ** 2 + jy ** 2)
    stats["main_stick"] = {
        "jx_mean": float(jx.mean()), "jx_std": float(jx.std()),
        "jy_mean": float(jy.mean()), "jy_std": float(jy.std()),
        "magnitude_mean": float(mag.mean()),
        "magnitude_median": float(np.median(mag)),
        "fraction_centered_abs_lt_0.05": float((mag < 0.05).mean()),
        "fraction_in_deadzone_mag_lt_0.28": float((mag < 0.28).mean()),
        "fraction_walk_0.28_to_0.7": float(((mag >= 0.28) & (mag < 0.7)).mean()),
        "fraction_dash_0.7_to_0.8": float(((mag >= 0.7) & (mag < 0.8)).mean()),
        "fraction_smash_ge_0.8": float((mag >= 0.8).mean()),
    }
    # Magnitude histogram
    cnt, _ = np.histogram(mag, bins=MAG_EDGES)
    stats["main_stick"]["mag_hist"] = {
        f"[{a:.2f},{b:.2f})": int(c) for a, b, c in zip(MAG_EDGES[:-1], MAG_EDGES[1:], cnt)
    }
    # Angle (only for non-center frames)
    moving = mag >= 0.28
    angle = np.degrees(np.arctan2(jy[moving], jx[moving]))
    abins = np.arange(-180, 181, 22.5)
    cnt_a, _ = np.histogram(angle, bins=abins)
    stats["main_stick"]["angle_deg_hist_when_moving"] = {
        f"[{a:.0f},{b:.0f})": int(c) for a, b, c in zip(abins[:-1], abins[1:], cnt_a)
    }

    # 2D bin histogram (proposed 9x9)
    bx = np.searchsorted(STICK_EDGES, jx, side="right") - 1
    by = np.searchsorted(STICK_EDGES, jy, side="right") - 1
    bx = np.clip(bx, 0, len(STICK_EDGES) - 2)
    by = np.clip(by, 0, len(STICK_EDGES) - 2)
    bin2d = bx * (len(STICK_EDGES) - 1) + by
    cnt2d = np.bincount(bin2d, minlength=(len(STICK_EDGES) - 1) ** 2)
    stats["main_stick"]["bins_9x9_count"] = cnt2d.tolist()
    stats["main_stick"]["bins_9x9_max_pct"] = float(cnt2d.max() / n_frames)
    stats["main_stick"]["bins_9x9_min_pct"] = float(cnt2d.min() / n_frames)
    stats["main_stick"]["bins_9x9_n_used"] = int((cnt2d > 0).sum())

    # --- C-stick ---
    cx, cy = arrs["cx"], arrs["cy"]
    cmag = np.sqrt(cx ** 2 + cy ** 2)
    stats["cstick"] = {
        "fraction_centered_mag_lt_0.05": float((cmag < 0.05).mean()),
        "fraction_full_mag_ge_0.7": float((cmag >= 0.7).mean()),
        "fraction_partial_0.05_to_0.7": float(((cmag >= 0.05) & (cmag < 0.7)).mean()),
    }
    # Cardinal direction breakdown when full-pressed
    full = cmag >= 0.7
    if full.any():
        ang = np.degrees(np.arctan2(cy[full], cx[full]))
        # bin into 4 cardinals + 4 diagonals (8 cones)
        cone = ((ang + 22.5 + 360) % 360) // 45  # 0..7 starting east
        cone_names = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
        cone_cnt = np.bincount(cone.astype(int), minlength=8)
        stats["cstick"]["cone_when_full"] = {
            n: int(c) for n, c in zip(cone_names, cone_cnt)
        }
    # Bin into proposed 4-class scheme per axis
    bcx = np.searchsorted(CSTICK_EDGES, cx, side="right") - 1
    bcy = np.searchsorted(CSTICK_EDGES, cy, side="right") - 1
    bcx = np.clip(bcx, 0, len(CSTICK_EDGES) - 2)
    bcy = np.clip(bcy, 0, len(CSTICK_EDGES) - 2)
    bin2c = bcx * (len(CSTICK_EDGES) - 1) + bcy
    cnt2c = np.bincount(bin2c, minlength=(len(CSTICK_EDGES) - 1) ** 2)
    stats["cstick"]["bins_4x4_count"] = cnt2c.tolist()
    stats["cstick"]["bins_4x4_max_pct"] = float(cnt2c.max() / n_frames)
    stats["cstick"]["bins_4x4_n_used"] = int((cnt2c > 0).sum())

    # --- Triggers (effective scalar, preserved by regen) ---
    t = arrs["tl"]
    stats["triggers"] = {
        "note": "triggers_physical.l/r are NOT preserved by regen (same as buttons_physical). "
                "This is the `triggers` effective scalar which IS preserved.",
        "min": float(t.min()), "max": float(t.max()), "mean": float(t.mean()),
        "fraction_zero_lt_0.01": float((t < 0.01).mean()),
        "fraction_light_0.01_to_0.3": float(((t >= 0.01) & (t < 0.3)).mean()),
        "fraction_mid_0.3_to_0.7": float(((t >= 0.3) & (t < 0.7)).mean()),
        "fraction_high_ge_0.7": float((t >= 0.7).mean()),
        "fraction_full_ge_0.99": float((t >= 0.99).mean()),
    }

    # --- Buttons ---
    btn = arrs["buttons"]
    button_stats = {}
    for name, bit in BITS.items():
        pressed = (btn & bit) != 0
        # press edge: pressed now & not pressed prev frame (within port-trace concat,
        # off-by-one boundary errors are << 1 in 24M frames so fine)
        prev = np.roll(pressed, 1); prev[0] = False
        just = pressed & ~prev
        button_stats[name] = {
            "press_pct": float(pressed.mean() * 100),
            "just_press_pct": float(just.mean() * 100),
            "avg_hold_frames": float(pressed.sum() / max(int(just.sum()), 1)),
        }
    # Combined / collapsed
    jump = ((btn & (BITS["X"] | BITS["Y"])) != 0)
    shield = ((btn & (BITS["L"] | BITS["R"])) != 0) | (tl >= 0.3) | (tr >= 0.3)
    button_stats["jump_combined_X_or_Y"] = {
        "press_pct": float(jump.mean() * 100),
    }
    button_stats["shield_combined_LR_digital_or_analog"] = {
        "press_pct": float(shield.mean() * 100),
    }
    button_stats["X_only_no_Y"] = float(((btn & BITS["X"]) & ~(btn & BITS["Y"])).any() / max(1, n_frames))
    stats["buttons"] = button_stats

    # X vs Y preference per replay (do players use both, or just one?)
    # Sample: how often does a single match use only one of X/Y at all?
    # Quick proxy: for each port-trace, did it ever press X? did it ever press Y?
    x_used = []
    y_used = []
    cursor = 0
    for ml in match_lens:
        # each match has 2 port-traces of length ml, concatenated; loop over them
        for _port in range(2):
            seg = btn[cursor:cursor + ml]
            x_used.append(int(((seg & BITS["X"]) != 0).any()))
            y_used.append(int(((seg & BITS["Y"]) != 0).any()))
            cursor += ml
    stats["buttons"]["xy_usage_per_port_trace"] = {
        "n_traces": len(x_used),
        "used_X_only": int(sum(x and not y for x, y in zip(x_used, y_used))),
        "used_Y_only": int(sum(y and not x for x, y in zip(x_used, y_used))),
        "used_both":   int(sum(x and y     for x, y in zip(x_used, y_used))),
        "used_neither":int(sum(not x and not y for x, y in zip(x_used, y_used))),
    }

    # --- Coordination: stick magnitude conditional on press edges ---
    coord = {}
    for name in ["A", "B", "Z"]:
        bit = BITS[name]
        pressed = (btn & bit) != 0
        prev = np.roll(pressed, 1); prev[0] = False
        just = pressed & ~prev
        if just.sum() == 0:
            continue
        mag_at = mag[just]
        cnt_m, _ = np.histogram(mag_at, bins=MAG_EDGES)
        coord[f"stick_mag_when_{name}_just_pressed"] = {
            "n_events": int(just.sum()),
            "mean_mag": float(mag_at.mean()),
            "hist": {f"[{a:.2f},{b:.2f})": int(c)
                     for a, b, c in zip(MAG_EDGES[:-1], MAG_EDGES[1:], cnt_m)},
        }
        # Also c-stick for A (smashes via c-stick)
        if name == "A":
            cmag_at = cmag[just]
            coord[f"cstick_mag_when_A_just_pressed"] = {
                "mean_mag": float(cmag_at.mean()),
                "fraction_full_cstick_pressed_too": float((cmag_at >= 0.7).mean()),
            }
    # Conversely: when c-stick is fully pressed, what fraction of frames is A also pressed?
    cstick_full = cmag >= 0.7
    if cstick_full.any():
        a_when_cfull = ((btn & BITS["A"]) != 0)[cstick_full]
        coord["A_when_cstick_full_pressed"] = float(a_when_cfull.mean())
    stats["coordination"] = coord

    # --- Action state distribution ---
    st = arrs["state"]
    top = Counter(st.tolist()).most_common(30)
    stats["action_state"] = {
        "n_unique": int(len(set(st.tolist()))),
        "top_30": [{"state_id": int(s), "count": int(c), "pct": float(c / n_frames * 100)}
                   for s, c in top],
    }

    # --- Save + summary ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=2))
    print(f"\nwrote {args.out}")
    # Brief summary to stdout
    print("\n--- summary ---")
    print(f"replays: {stats['dataset']['n_replays_parsed']}")
    print(f"port-frames: {n_frames:,}")
    print(f"avg match: {stats['dataset']['avg_match_len_seconds']:.1f} sec/port")
    print(f"main stick: centered {stats['main_stick']['fraction_centered_abs_lt_0.05']*100:.1f}%, "
          f"deadzone {stats['main_stick']['fraction_in_deadzone_mag_lt_0.28']*100:.1f}%, "
          f"smash {stats['main_stick']['fraction_smash_ge_0.8']*100:.1f}%")
    print(f"c-stick: centered {stats['cstick']['fraction_centered_mag_lt_0.05']*100:.1f}%, "
          f"full pressed {stats['cstick']['fraction_full_mag_ge_0.7']*100:.1f}%")
    print(f"button press rates (% of frames):")
    for k in ["A", "B", "X", "Y", "Z", "L", "R", "Start"]:
        s = stats["buttons"][k]
        print(f"  {k}: held {s['press_pct']:.2f}%, edges {s['just_press_pct']:.3f}%, hold {s['avg_hold_frames']:.1f} frames")


if __name__ == "__main__":
    main()
