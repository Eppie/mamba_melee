# Action head design — backed by the actual Fox-ditto data

Source: 1,200 Fox-ditto replays, regenerated to format v3.19. Stats below are pooled across both player slots, **23,103,746 port-frames** total (avg ~160 sec per player per match). Numbers regenerable via `scripts/analyze_inputs.py`; raw output in `notes/input_stats.json`.

## TL;DR — what the data changes vs. my earlier proposal

1. **Main stick is bimodal, not gradient.** Players spend 33% of frames with the stick centered and 62% of frames with it at the cardinal edge (mag ≥ 0.8). The middle range (0.28–0.8) is **~5%** of frames. A 9×9 uniform grid is roughly half-empty but it's the easiest output to reason about.
2. **Y is the more common jump button, but most players use both, and the game can't tell them apart.** Collapse X|Y → single "jump" head.
3. **L vs R digital are also indistinguishable to the game.** Collapse to single shield head.
4. **B almost always means "directional special."** 84% of B-press events happen with the stick at full deflection.
5. **C-stick diagonals matter** (NE, NW, SE, SW collectively ~13% of full-press events). 9-class (center + 8 octants); 5-class loses the diagonals.
6. **Shield is bimodal too.** Within shielding events: 10% are in the lightshield-just-past-threshold band (0.30–0.40), 81% are slammed to 1.0, and the middle is sparse. **4-class shield head** `{none, light, mid, full}` captures the real structure.
7. **`triggers_physical.l/r` is lost in regenerated replays** (same class of issue as `buttons_physical`). The processed `triggers` scalar is preserved and is what we should train on.

## Dataset

| stat | value |
|---|---|
| replays | 1,200 |
| port-frames | 23,103,746 |
| avg match length | 160 sec/port (~5.3 min total) |
| median match length | 173 sec/port |

## Main stick

### Magnitude distribution

| bin | fraction | interpretation |
|---|---|---|
| [0.00, 0.05) | **33.3%** | exactly centered (deadzoned to 0) |
| [0.05, 0.28) | 0.0% | empty — `joystick` field zeroes the deadzone |
| [0.28, 0.50) | 1.9% | walk |
| [0.50, 0.70) | 2.2% | brisk walk / partial tilt |
| [0.70, 0.80) | 1.0% | dash threshold band |
| [0.80, 0.95) | 2.3% | sub-max |
| [0.95, 1.01] | **59.3%** | full deflection |

So **93% of frames are either centered or full-pressed**. The middle 5% is the entire "tilt / walk / partial" universe. Critically: any bin that spans the 0.28 deadzone boundary or the 0.7→0.8 smash threshold is mis-binned. (My earlier "9-bin scheme with edges at 0.28 and 0.8" already handles this.)

### Direction (when moving, mag ≥ 0.28)

The two dominant directions are **right (4.28M frames, 28%)** and **left (4.22M frames, 28%)**. After that come down-ish-and-forward (–90° to –68°, 14%) and up (90°–112°, 6%). Pure up is rare; pure down (-90°) is for fast-fall and is bucketed into the down-bias above.

### Bin occupancy (proposed 9×9, edges at `{-1.0, -0.8, -0.5, -0.28, 0, 0.28, 0.5, 0.8, 1.0}`)

- **45 of 81 cells are used at all** — 36 cells are dead from the deadzone alone (any cell straddling 0 in both axes collapses).
- Max bin = 33% (the centered cell), next biggest = 17% (full-right + zero-vertical), 16% (full-left + zero-vertical).
- Long tail: most non-cardinal full-deflection bins are 1–3% each.

**Decision:** stick with the 9×9 bin scheme (edges as above). Half the cells are never populated, which means those logits get no gradient and converge to "never predicted" — wasteful but simple, and the simplicity helps when ablating other parts of the head. Revisit if it shows up as a problem.

## C-stick

| bin | fraction |
|---|---|
| center (mag < 0.05) | **95.0%** |
| partial (0.05–0.7) | 0.3% |
| full (mag ≥ 0.7) | 4.7% |

When full-pressed, cone breakdown:

| cone | count | fraction of full-press events |
|---|---|---|
| down (S) | 329k | 30.5% |
| north (N) | 218k | 20.2% |
| east (E) | 203k | 18.8% |
| west (W) | 186k | 17.2% |
| NW (up-back) | 70k | 6.5% |
| SE | 35k | 3.2% |
| NE | 20k | 1.9% |
| SW | 19k | 1.8% |

So **9-class is right** (center + 8 octants). The diagonals are minority but non-trivial (~13% of all c-stick events). 5-class would lose them.

## Triggers / shield

The game treats the analog trigger as a single shield-strength scalar (`max(L_analog, R_analog)`). For our output we predict one head; the controller bridge can plug the value into either physical trigger.

`triggers` scalar across all frames:

| bin | % all frames |
|---|---|
| zero (< 0.01) | **81.2%** |
| 0.01–0.30 (no effect — game treats as 0 below threshold) | 0.0% |
| 0.30–0.70 (lightshield zone) | 2.7% |
| ≥ 0.70 (full shield) | 16.1% |
| ≥ 0.99 (slammed) | 15.3% |

Shielding is **18.8% of frames**. Within shielding events the distribution is sharply bimodal:

| range | % of shield events | % of all frames |
|---|---|---|
| 0.30–0.40 (lightshield-just-past-threshold) | **10.2%** | 1.9% |
| 0.40–0.95 (mid) | 7.7% | 1.4% |
| ≥ 0.95 (effectively full) | **82.1%** | 15.4% |

**Decision:** 4-class shield head `{none, light (0.30–0.40), mid (0.40–0.95), full (≥0.95)}`. This captures the two real modes (lightshield-min and full-press) plus a catch-all for the sparse middle. More than 4 bins doesn't help — the middle range is genuinely sparse.

Note that `triggers_physical.l/r` are zeroed by the regenerate path (verified directly); train on the `triggers` scalar, which IS preserved.

## Buttons

| button | held % | press-edge % | avg hold (frames) |
|---|---|---|---|
| A | 5.69 | 0.748 | 7.6 |
| B | 4.69 | 0.542 | 8.7 |
| X | 3.69 | 0.602 | 6.1 |
| Y | 5.69 | 0.829 | 6.9 |
| Z | 1.46 | 0.233 | 6.3 |
| L | 8.54 | 0.628 | 13.6 |
| R | 4.76 | 0.418 | 11.4 |
| Start | 0.00 | 0.001 | — |

**Start is never pressed mid-game** — mask out (zero logit) at both training and inference.

### X vs Y for jump

2,400 port-traces (1,200 replays × 2 ports):

| usage | count | % |
|---|---|---|
| both X and Y | 1,634 | 68% |
| only Y | 512 | 21% |
| only X | 254 | 11% |
| neither | 0 | 0% |

Most players use both, with a slight Y preference. Collapsing X|Y → single "jump" head loses the personal preference signal but is clearly fine for action prediction (combined jump rate ≈ 9.4% of frames, near-additive since the buttons are rarely held together).

### L vs R for shield

L held 8.5%, R held 4.8%. Some players prefer L, some R. Combined digital `L|R ≈ 13%`, plus the 19% shielding rate from triggers ≈ same population. Collapse to a single shield head.

## Action coordination

What is the stick magnitude on the frame A / B / Z is *just pressed*?

| press | n events | mag < 0.05 (centered) | mag in (0.28, 0.8) (mid) | mag ≥ 0.95 (full) |
|---|---|---|---|---|
| A | 173k | 39% | 7% | 52% |
| B | 125k | 15% | 0.6% | 84% |
| Z | 54k | 23% | 4.4% | 71% |

The pattern is bimodal in all three: presses cluster at *centered stick* and *full stick*, with the mid-magnitude range thinly populated. (The mid range is a mix of tilts, mid-stick aerials with drift, and dash attacks — this number alone doesn't disambiguate them; that would require correlating action_state at the moment of press.)

For B specifically the bimodality is extreme — 84% of B presses are with the stick at full deflection. B + full-stick = directional special (Fox shine if down, illusion if side, firefox if up). B + centered = neutral special (laser).

When the c-stick is full-pressed, A is *also* pressed on the same frame only 2.4% of the time → c-stick smashes and A presses are mostly separable events.

### Implication for the autoregressive A | stick link

A press probability changes substantially with stick magnitude: ~7× higher conditional on full-deflection vs. mid-magnitude (after normalising by base rates). The autoregressive ordering (predict stick, then A | stick) helps the model represent that. A fully independent factoring would still work; calibration on combined-input frames would be slightly worse.

## Action states (Fox)

181 unique action states observed in 23M frames (out of ~388 possible). Top 30 cover **62%** of all frames:

| rank | state id | name (libmelee enum) | % frames |
|---|---|---|---|
| 1 | 20 | DASHING | 8.5 |
| 2 | 25 | JUMPING_FORWARD | 5.4 |
| 3 | 90 | DAMAGE_FLY_TOP | 5.1 |
| 4 | 27 | JUMPING_ARIAL_FORWARD | 3.9 |
| 5 | 67 | BAIR | 3.3 |
| 6 | 43 | LANDING_SPECIAL | 3.2 |
| 7 | 88 | DAMAGE_FLY_NEUTRAL | 3.1 |
| 8 | 14 | STANDING | 3.0 |
| 9 | 24 | KNEE_BEND (jumpsquat) | 2.7 |
| 10 | 29 | FALLING | 2.5 |
| 11 | 65 | NAIR | 2.4 |
| 12 | 42 | LANDING | 2.3 |
| 13 | 354 | (Fox SHINE active / "down-special") | 2.0 |
| 14 | 69 | DAIR | 1.9 |
| 15 | 12 | (DAMAGE_AIR_3 / hitstun continuation) | 1.6 |
| 16 | 63 | UPSMASH | 1.6 |
| 17 | 183 | TECH_MISS_UP | 1.5 |
| 18 | 18 | TURNING | 1.5 |
| 19 | 68 | UAIR | 1.4 |
| 20 | 221 | THROW_UP | 1.4 |
| 21 | 212 | GRAB | 1.3 |
| 22 | 179 | SHIELD | 1.2 |
| 23 | 72 | BAIR_LANDING | 1.2 |
| 24 | 87 | DAMAGE_FLY_HIGH | 1.1 |
| 25 | 56 | UPTILT | 1.0 |
| 26 | 4 | DEAD_FLY_STAR (stock loss anim) | 1.0 |
| 27 | 21 | RUNNING | 1.0 |
| 28 | 74 | DAIR_LANDING | 0.95 |
| 29 | 50 | DASH_ATTACK | 0.95 |
| 30 | 356 | (Fox SHINE turnaround / "down-special2") | 0.92 |

Note that some IDs (354, 356) are character-specific — libmelee's `Action` enum gives the name from whichever character it associated with first; for Fox these are shine states.

**Implications for the input side** (we'll model action state as an embedding of the *opponent's* state, primarily):
- 181 unique observed states is well within an embedding table of size 388. Reserve an `<unknown>` slot for state IDs we haven't seen in training (will matter when opponent isn't Fox).
- Long tail: ranks 30+ are each <1%. Embedding dim of 32 is probably enough.

## Action head spec

```
main_stick:  81-class categorical (9×9 grid, edges {-1.0, -0.8, -0.5, -0.28, 0, 0.28, 0.5, 0.8, 1.0})
cstick:       9-class categorical {center, 8 octants}
shield:       4-class {none, light (0.30-0.40), mid (0.40-0.95), full (≥0.95)}
A:            Bernoulli, autoregressive on stick   (~7× rate change conditional on stick)
B:            Bernoulli                            (mostly independent of A)
jump:         Bernoulli, "X|Y combined"
Z:            Bernoulli                            (grabs)
[Start:       masked out everywhere — never pressed mid-game]
[L vs R digital: collapsed into shield head — game treats them identically]
```

Effective output size: 81 + 9 + 4 + 1 + 1 + 1 + 1 = 98 logits per frame.

The X|Y and L|R digital collapses are lossless from the game's perspective — the in-game ASM that polls the controller sets the same bit regardless of which physical button was pressed. Slippi just records them separately because they're physically separate buttons.

Loss: sum of CE / BCE on each head, with mild class weighting `1/sqrt(freq)` on the rare button heads (Z, B). Start with α=1 everywhere; reweight only if any head dominates the gradient. The A-conditional-on-stick term is the only autoregressive piece, sampled in stick→A order at inference.

### Open questions worth deciding before model code

1. **Edge-vs-hold for jump and A.** Right now I'd predict the *current* button state (hold included) and let the temporal model handle press transitions; the press-edge frequencies in the table above are 5–8× lower than holds and would need much heavier class weighting if used directly.
2. **Stick bin scheme refinement.** With 9×9, ~36 bins are dead from the deadzone. Worth revisiting if those wasted logits hurt training stability.
3. **B head conditioning.** B is so strongly bimodal on stick (84% with full deflection) that an autoregressive `B | stick` link would also help. Adding it doubles the autoregressive complexity without changing the architecture; cheap to try.
