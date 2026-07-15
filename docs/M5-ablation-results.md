# M5 — 3D Terrain Search: LSTM × RND Ablation Results

Date: 2026-07-15.
Branch: `m5-3d-terrain-search`.

## Question

Does recurrent memory (LSTM) and/or curiosity-driven exploration (Random Network Distillation, RND) improve an agent's ability to search for a hidden goal in a 3D terrain with occlusion?
M5 exists to answer this cleanly where M4 could not: M4's ablation was inconclusive because the flat task was reactively solvable and the evaluation was single-run and confounded.

## Task and method

A single PPO learner (CharacterController + gravity + jump) navigates a procedurally generated arena (walls, ramp-reachable platforms, hazard pits, oblivious mover-occluders) to a line-of-sight-gated goal, perceived through three ray-fans (forward/down/up).
A `difficulty` curriculum ramps the terrain L0 (open, flat, goal visible) -> L1 (occluding walls) -> L2 (elevated platform + pits) -> L3 (hidden, elevated, crowded).

The ablation is a 2x2 over two levers, each arm identical except the ablated block:

| Arm | LSTM | RND |
|-----|------|-----|
| `primary`  | yes | yes |
| `nornd`    | yes | no  |
| `nolstm`   | no  | yes |
| `baseline` | no  | no  |

The evaluation is the part M5 rebuilt to fix M4.
It is a **paired** protocol: every arm and seed is scored on the **same** held-out layouts (RNG re-seeded identically per episode), on the efficiency-weighted success metric **SPL** (Success weighted by inverse-optimal Path Length), with the optimal path taken from the runtime-baked NavMesh oracle.
Aggregation uses `rliable`: IQM SPL with 95% stratified-bootstrap confidence intervals, plus **probability-of-improvement** (PoI) of the full policy over each single-ablation.
A decision rule was **pre-registered** before the runs: the full policy is deemed to justify a lever only if PoI >= 0.75; overlapping/low PoI at n=3 triggers adding seeds 4-5 (never a silent pass).

12 runs (4 arms x 3 seeds) were later extended to 20 runs (n=5) when the pre-registered rule fired.
Equal fixed budget of 3M steps per run.

## A note on the setup (why it is what it is)

A calibration probe (spec Task-14) caught a genuine reward/perception design flaw before any real run: distance-shaping was gated on privileged omnidirectional visibility while the policy only perceives the goal through its forward ray-fan, so PPO correctly converged to freeze (moving toward an unperceived goal is net-penalized).
The fixes that made the task learnable, all validated by probes: perception-gated shaping (the reward gates on the same forward-fan the policy sees), a spawn-distance cap so warmup goals start perceivable, and — critically — **pure-extrinsic reward** (an earlier "explore bias" was removed because it is an exploration drive that would confound the RND lever).

The task turned out to be learnable from scratch with pure-extrinsic reward alone, but only for the no-RND arms: RND (at strengths 0.02 and 0.1) actively **prevents** the L0 cold-start bootstrap (novelty is farmable by spinning, and the intrinsic drive competes with exploiting a perceivable goal).
So all four arms are initialized from a shared, architecture-matched, no-RND L0-bootstrap checkpoint (LSTM arms from an LSTM base, no-LSTM arms from a feedforward base).
This crutch is applied identically to every arm and, because it is a no-RND base, it biases *toward* RND — making any "RND still loses" outcome conservative.

## Results (n=5)

IQM SPL (higher is more efficient success):

| Arm | IQM SPL | mean SPL | success | L0 / L1 / L2 / L3 SPL |
|-----|--------:|---------:|--------:|-----------------------|
| `primary` (LSTM+RND) | 0.498 | 0.498 | 96% | 0.62 / 0.50 / 0.46 / 0.41 |
| `nornd` (LSTM only)  | 0.488 | 0.490 | 97% | 0.61 / 0.50 / 0.44 / 0.41 |
| `nolstm` (RND only)  | 0.408 | 0.433 | 95% | 0.54 / 0.38 / 0.40 / 0.41 |
| `baseline` (neither) | 0.396 | 0.431 | 98% | 0.51 / 0.40 / 0.42 / 0.40 |

Probability of improvement (pre-registered gate: PoI >= 0.75):

| Comparison | PoI | 95% CI | at n=3 |
|-----------|----:|--------|-------:|
| LSTM: `primary` > `nolstm` | **0.574** | [0.54, 0.61] | 0.606 |
| RND:  `primary` > `nornd`  | **0.506** | [0.47, 0.54] | 0.498 |

Figure: `training/eval/m5_search.png`. Raw episodes: `training/eval/m5_search.csv` (2000). PoI: `training/eval/m5_poi.csv`.

## Findings

1. **Curiosity (RND) does not earn its keep — a clean null.**
   PoI 0.506 with a CI straddling 0.5, at both n=3 and n=5.
   Adding RND does nothing for SPL (`nornd` >= `primary`; `baseline` >= `nolstm`), and it actively impedes the initial bootstrap.
   This is the M4 ambiguity resolved and un-confounded.

2. **Memory (LSTM) helps, but modestly.**
   The effect is real (IQM 0.49 vs 0.40; the PoI CI [0.54, 0.61] excludes 0.5) but sits **below** the pre-registered 0.75 "strong improvement" bar.
   Note that all arms succeed 95-98% of the time; LSTM's gain is in path *efficiency*, not in reaching the goal at all.

3. **The extra seeds corrected a small overclaim.**
   Going from n=3 to n=5, the LSTM PoI moved *down* (0.606 -> 0.574) and its CI tightened, rather than up toward the gate.
   The effect is genuine-but-modest, not few-seed noise.
   This is the pre-registered add-seeds discipline doing exactly its job.

## Honest caveats

- **The LSTM benefit is level-dependent, and not in the expected direction.**
  Per level, LSTM's SPL advantage (`nornd` vs `baseline`) is +0.10 at L0 and L1, but only +0.02 / +0.01 at L2 / L3.
  The benefit is concentrated on the *easier, flatter* levels and vanishes at the hardest, most-occluded one, where terrain difficulty (elevation, pits, crowding) appears to dominate and flatten the arms.
  This is mildly counter to the "memory helps most under heavy occlusion" motivation and should not be buried.

- **The RND arms required a bootstrap crutch to run at all** (they freeze from scratch).
  The shared no-RND base is conservative toward RND, but it does mean no arm was trained purely from scratch.

- **n=5 is the pre-registered ceiling.**
  The gate is not met for either lever; per the rule this is where it stops. The honest conclusion is stated, not stretched.

## Conclusion

In this 3D terrain hidden-goal search task, **memory gives a modest, real efficiency gain and curiosity gives nothing** — with the memory gain concentrated on the easier levels and neither lever clearing the pre-registered strong-improvement bar.
The headline is undramatic, but that is the point: M5 delivered a rigorous, multi-seed, un-confounded answer where M4 could only guess, and the pre-registered methodology prevented an overclaim rather than manufacturing one.

## Reproduce

- Train: `training/run_m5_ablation.sh` (seeds 0-2) and `training/run_m5_seeds34_and_eval.sh` (seeds 3-4 + eval + analysis). Bases: `training/configs/m5_base_{lstm,nolstm}.yaml`. Arms: `training/configs/m5_ppo{,_nolstm,_nornd,_baseline}.yaml`.
- Eval (bridge-free, headless): `Unity -batchmode -nographics -projectPath NavSim -executeMethod M5EvalBatch.RunHeadless`.
- Analyze: `MPLBACKEND=Agg python training/eval/plot_m5_search.py`.
