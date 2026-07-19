# M6 - Visual Object-Goal Search: Fixed-Target Colour Discrimination Results

Date: 2026-07-18.
Branch: `m6-visual-object-goal-search`.

## Question

Can a from-scratch CNN (no hand-engineered colour perception) learn to discriminate a fixed target colour among visually-identical decoys - a signal a strong ray sensor is provably blind to - and, in doing so, match or approach an ablation upper bound that is *hand-told* the colour via distinct sensor tags?
M6 is a perception-only ablation on top of M5's terrain spine: memory (LSTM) and reward shaping are held identical across arms; only what the agent can *see* differs.

## Task and method

Each episode places 3 geometrically identical goals (same mesh/collider).
The **target is a fixed colour** (red); the two decoys are 2 random distinct non-target colours drawn per episode.
The agent must reach the red goal; touching a decoy first is a failure under the scoring rule below.
There is no per-episode cue to broadcast which goal is the target - colour identity alone carries the information, and it is the same fixed colour every episode.

Three arms differ **only in perception** (LSTM on, no RND, identical curriculum and reward in all three):

| Arm | Perception | Role |
|-----|-----------|------|
| `pixel` | 84x84 egocentric RGB camera, from-scratch `nature_cnn` | can the CNN learn colour from pixels? |
| `ray1` | ray fans, single undifferentiated `goal` tag | provable chance floor - the confound-detector |
| `rayC` | ray fans, per-colour tags (`goal_c0`..`c4`) | hand-told upper bound - colour is free information |

A `difficulty` curriculum ramps the terrain L0 (open, flat, goal cluster co-visible) -> L1 (occluding walls) -> L2 (elevated platform + pits) -> L3 (hidden, elevated, crowded), the same terrain spine as M5.

**Evaluation** is a **paired** protocol, forked from M5's: every arm and seed is scored on the **same** held-out layout set (colours, goal positions, terrain RNG re-seeded identically per episode).
Pairing is not assumed - it is **verified**: each arm emits a per-episode layout fingerprint (`m6_pairing_<arm>.csv`), and the three arms' fingerprint streams are byte-identical, both at probe scale and at the full 900-episode scale.
Scoring is **always-hard**: success requires reaching the red target *before* touching either decoy, computed by a pure geometric `Resolve()` that the harness (not the training reward) owns exclusively (`NavEnvironment.EvalMode` gates the training-time decoy-ends-episode behavior out of eval, so the harness is the sole arbiter of the eval boundary).

Aggregation: success rate with 95% **Wilson** score intervals (the convention used for every pre-registered gate below); **mean SPL** (Success weighted by inverse-optimal Path Length, oracle from the runtime-baked NavMesh) with 95% **stratified-bootstrap** CIs via `rliable`; and **probability-of-improvement (PoI)** on SPL, computed over the 12 (seed x level) cells per arm via `rliable`'s Mann-Whitney-based estimator.
(Mean, not IQM: SPL is heavily zero-inflated on the low-success arm - ray1 fails ~64% of episodes, so SPL=0 there - and the interquartile mean collapses toward zero under that skew, 0.026 vs a true mean of 0.162; mean SPL is the number that stays consistent with the success-rate story and is reported end to end here.)

A decision rule was **pre-registered before any full-scale training run**, via a five-gate calibration protocol (B1-B5, grilled and user-confirmed 2026-07-15) run on cheap single-seed probes before committing the ~34-hour, 9-run compute budget.
All five gates passed (see below) before the user authorized the full 3-arm x 3-seed run.

9 runs (3 arms x 3 seeds), 3M steps each, evaluated on 300 held-out episodes per arm (3 seeds x 4 levels x 25 episodes/level/seed) = 900 episodes total.
Figure: `training/eval/m6_search.png`.
Raw episodes: `training/eval/m6_search.csv`.
PoI: `training/eval/m6_poi.csv`.

## A note on the setup: the cued negative

M6's first design was different, and it failed - that failure is itself the most informative finding on the road to this result.

**v1 design (superseded).** The original task cued the target *per episode*: a 3-float RGB vector told the agent which of the 3 (randomly-coloured) goals to seek that episode, and the claim under test was whether a from-scratch CNN could bind an abstract vector cue to a rendered pixel colour.
Four training attempts, across four configurations, tried to make this learn:

1. **probe1** (scattered goal placement): reward plateaued; eval at 500k showed cued-reach 0.320 but the cued-*fraction*-of-reaches stayed pinned at chance the whole run - the agent reached goals more often as navigation improved, but picked the *right* one no better than 1-in-3.
2. **probe2** (clustered goal placement, to guarantee all 3 goals are visible in one camera frame): killed mid-run before it reached an eval verdict - code review caught a critical geometric-feasibility bug in the rejection-sampling cluster placement (the ~70% last-resort fallback rate meant it wasn't reliably testing the intended layout).
Redesigned as a rigid 120-degree ring (geometrically guaranteed spacing, no rejection) before re-running.
3. **probe3** (rigid-ring co-visible placement): cued-fraction opened above chance early (0.545 at 100k) - proof that co-visibility gives a real early colour signal - but **decayed back toward chance** as training continued (0.410 at 500k) while decoy-visits rose.
Diagnosis: a train/eval objective mismatch (soft L0 decoy penalties let training reward efficient goal-*sweeping*, which the always-hard eval scores as failure).
4. **probe4** (rigid-ring placement + a soft-to-hard decoy curriculum, to align training with the eval objective): training reward climbed strongly and never froze through the hardening transition - but eval cued-reach **stayed at chance** throughout (0.18 -> 0.30 -> 0.24 at 200k/400k/700k) while decoy-visits rose to 0.58.
The reward climbed for the wrong reason: hardening made wrong-pick episodes end faster, not more accurate.

All four attempts converged on the same diagnosis: the from-scratch CNN **learns navigation** (it reaches *a* goal reliably) but **not the cross-modal binding** - the 3-float vector cue never became reliably associated with the matching goal's rendered colour, at any placement geometry, decoy-hardness schedule, or checkpoint sampled.

**The controlled comparison.** Rather than keep tuning, the redesign changed exactly one variable: **drop the per-episode cue and fix the target colour** (always red), so there is no cross-modal binding left to learn - colour discrimination becomes a fixed-target detection problem, which a CNN should learn easily.
`probe5` ran this fixed-target task on **probe4's exact configuration** (same placement, same soft-to-hard decoy curriculum, same step budget) - the *only* change was de-conditioning the task on the cue.
Training reward went **positive** (+1.63 at 700k; no cued-design probe ever crossed 0) and eval climbed cleanly: 0.160 (200k, soft) -> 0.480 (400k, hard) -> **0.610** (700k, hard), N=100, Wilson CI (0.512, 0.700), with reach-any = 1.000 (the agent always reaches *some* goal; the question is only which one).
Target-fraction-of-reaches was 0.61 - about 1.8x chance, versus the ~1/3-flat cued-fraction in all four prior attempts.

Because probe4 and probe5 are the **same configuration with one variable changed**, the probe4-vs-probe5 gap (0.24 vs 0.610 target-reach, same budget, same curriculum, same placement) is controlled evidence that the **cross-modal cue-colour binding was the bottleneck**, not capacity, budget, or shaping - all three of which were already present and unchanged in probe4.
The fixed-target redesign that this document reports on is the one that followed from that finding.

## Results (n=3 seeds, 900 episodes: 3 arms x 3 seeds x 4 levels x 25 episodes)

Target-goal success rate and mean SPL (higher is better on both; decoy-visit is a failure-mode tracker, lower is better):

| Arm | success | 95% Wilson CI | mean SPL | decoy-visit | per-seed success |
|-----|--------:|---------------|---------:|-------------:|-------------------|
| `pixel` (from-scratch CNN) | **0.713** | (0.660, 0.762) | **0.398** | 0.260 | 0.59 / 0.81 / 0.74 |
| `ray1` (single-tag, chance floor) | 0.357 | (0.305, 0.412) | 0.162 | 0.590 | 0.28 / 0.35 / 0.44 |
| `rayC` (hand-told colour tags) | **0.773** | (0.723, 0.817) | 0.348 | 0.173 | 0.75 / 0.77 / 0.80 |

Per-level success (L0 open -> L3 hidden/elevated/crowded):

| Arm | L0 | L1 | L2 | L3 |
|-----|---:|---:|---:|---:|
| `pixel` | 0.653 | 0.693 | 0.733 | 0.773 |
| `ray1` | 0.360 | 0.360 | 0.387 | 0.320 |
| `rayC` | 0.747 | 0.760 | 0.747 | **0.840** |

Probability of improvement on SPL, over the 12 (seed x level) cells per arm:

| Comparison | PoI |
|-----------|----:|
| pixel > ray1 (**PRIMARY**) | **1.000** |
| rayC > ray1 | 0.965 |
| pixel > rayC (**STEELMAN**) | 0.632 |

Figure: `training/eval/m6_search.png`.

## Pre-registered claims and verdicts

Three claims were fixed before the full run; all three are now decided.

1. **PRIMARY - CONFIRMED.** The from-scratch pixel arm beats the ray1 confound-detector: success CIs are fully separated (0.660-0.762 vs 0.305-0.412, no overlap) and PoI(SPL, pixel > ray1) = **1.000**, clearing the pre-registered >= 0.75 bar with no ambiguity.
The CNN learned the fixed target's colour from raw pixels - a signal ray1 cannot represent at all.

2. **Confound-detector - HOLDS.** ray1's success CI, (0.305, 0.412), **contains 1/3** at full scale (300 episodes, 3 seeds).
A colour-blind sensor performing at chance is exactly the null the ablation design predicts if there is no leak (no shape/geometry/position tell that lets an undifferentiated ray sensor discriminate the target without colour).
This also rules out the failure mode where an apparent "pixel win" is really some non-colour signal leaking through the layout: ray1 gets none of it.

3. **Steelman - the strong outcome, honestly framed.** Success rate is **statistically indistinguishable** between `pixel` and `rayC`: the CIs overlap (0.723-0.762), and rayC is nominally ahead of pixel at every single difficulty level (L0 .747 vs .653, L1 .760 vs .693, L2 .747 vs .733, L3 .840 vs .773) as well as overall (0.773 vs 0.713).
Where pixel is nominally ahead is **path efficiency**: mean SPL 0.398 vs 0.348.
PoI(SPL, pixel > rayC) = 0.632 - above chance (0.5) but below the pre-registered 0.75 "strong effect" bar, so this SPL edge is a real-but-not-decisive signal at n=3 seeds, not a clean win.
Read honestly: **the from-scratch CNN matched the hand-told upper bound on the primary metric (success) and edged it, non-decisively, on efficiency** - "pixels learned what rays must be told," matched rather than merely approached, without overclaiming a win rayC's own numbers don't support.

## Methods detail: the pre-registered B1-B5 gates

Before authorizing the 9-run, ~34-hour full ablation, five gates were checked on cheap single-seed probes (N=100 episodes, L0 unless noted) - all numbers below are **probe-scale**, distinct from the full-run headline above, and are cited here for methodological completeness:

- **B1** (pixel learns the fixed-target task): probe5, target-reach 0.610, Wilson CI (0.512, 0.700), CI lower bound clears the >= 0.40 bar. PASS.
- **B2** (ray1 confound-detector at probe scale): target-reach 0.310, CI (0.228, 0.406), contains 1/3. PASS.
- **B3** (rayC solves it): target-reach 0.670, CI (0.573, 0.754). PASS.
- **B4** (no un-freeze collapse L0->L1): pixel continued 200k steps from the probe5 checkpoint into L1; target-reach held at 0.600 (vs 0.610 at L0 - no collapse), CI (0.462, 0.724), mean steps-to-goal 487 (the agent is acting, not instant-failing). PASS.
- **B5** (shared curriculum breakpoints): thresholds recalibrated from the probe curves to 0.20/0.40/0.65 (from placeholder 0.10/0.30/0.50), applied identically across all three arms' configs. DONE.

All five gates passed; pairing byte-identity was independently confirmed at probe scale (101-line fingerprints, `cmp` clean across all three arms) and re-confirmed at full scale after the real run.
The user authorized the full run only after this record.

## Honest caveats

- **Sensor-coverage caveat - the colour claim is attributed to L0/L1, not L2/L3.** The ray fans have a sensor-truth shaping gate with only ~8 degrees of vertical reach, versus the pixel camera's 90 degree vertical FOV (pitched down 12 degrees).
At elevated L2/L3 targets, ray arms get zero shaping toward the goal until they climb into range, while the camera sees an elevated goal from the ground.
This is a real perception difference, orthogonal to colour, and it inflates pixel's *apparent* edge at the harder, elevated levels if read carelessly.
The colour-discrimination claim (pixel vs ray1, pixel vs rayC) should be read off **L0/L1** - flat levels where both sensor types see the goals equally and only colour perception differs, which is exactly where the B1/B2/B3 pre-registered gates were measured.
Reassuringly, the coverage gap does **not** appear to have manufactured pixel's headline edge: `rayC` - a *ray* arm - wins L3 outright (0.840, the best success rate of any arm at any level), which would not happen if elevation coverage alone were carrying the result.

- **Seed variance is real and not small at n=3.** Pixel's per-seed success is 0.59 / 0.81 / 0.74 - a 0.22 spread.
This is reported plainly rather than smoothed into the point estimate; the Wilson CI on the pooled 300 episodes (0.660, 0.762) already reflects this dispersion, but a reader should not treat 0.713 as a tight number.

- **Probe-scale numbers are not the headline.** The B1-B5 probe gates (N=100, single seed, L0-only or L0->L1) and the full-run headline (N=300/arm, 3 seeds, all 4 levels) both say "ray1 is near chance," but at different scales and sample sizes (probe: 0.310 CI (0.228,0.406); full run: 0.357 CI (0.305,0.412)).
They should not be read as the same measurement, even though they agree.

## Findings

1. **The primary claim holds cleanly.** A from-scratch CNN discriminates a fixed target colour that a strong, otherwise-capable ray sensor is provably blind to (PoI 1.000, fully separated CIs).

2. **The confound-detector did its job.** ray1's chance-level result across both probe scale and full scale is the evidence that nothing besides colour is leaking through the ablation design.

3. **The cued negative was the real discovery.** Four attempts at a per-episode cued-colour task all failed in the same way (navigation learns, cross-modal binding doesn't), across four different placement and curriculum fixes - ruling out capacity, coverage, and shaping as the cause.
The probe4-vs-probe5 controlled comparison (identical configuration, only the task de-conditioned) is the evidence that isolates the cross-modal binding itself as the bottleneck.
That negative result is what redirected the milestone to the fixed-target design this document reports.

4. **The steelman result is a match, not a loss, and not an overclaim.** rayC is nominally ahead of pixel on success at every level; pixel is nominally ahead on path efficiency; neither gap is statistically decisive at n=3 seeds.
The honest reading is that the from-scratch CNN reached parity with a hand-told upper bound on the metric that matters most (does it reach the right goal), which is itself the strong form of the milestone's claim.

## Conclusion

M6 set out to show that a from-scratch CNN can learn a visual discrimination task a ray sensor cannot represent at all, at a cost approaching a sensor that is told the answer directly.
The pre-registered primary and confound-detector gates both confirm this without ambiguity, and the steelman comparison against the hand-told upper bound is, honestly read, a match rather than a partial result.
The path there was not a straight line: the original cued-colour design failed four times for a specific, diagnosed reason (an unlearnable cross-modal binding, not a capacity or shaping problem), and the fixed-target redesign that followed from that diagnosis is what actually works.
Both the negative and the positive results are reported here because both are load-bearing for anyone extending this task.

## Reproduce

- Train: `training/run_m6_ablation.sh` (3 arms x 3 seeds, serial, ~34 h wallclock).
  Configs: `training/configs/m6_{pixel,ray1,rayc}.yaml`.
- Eval (per arm, needs a StandaloneOSX player of the matching scene): `Unity -batchmode -projectPath NavSim -executeMethod M6EvalBatch.RunPixel` (`.RunRay1` / `.RunRayC`), writing to `training/eval/m6_search.csv`.
- Analyze: `.venv-nav/bin/python training/eval/plot_m6_search.py`.
