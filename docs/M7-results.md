# M7 - Cooperative Sacrifice: MA-POCA Credit Assignment Results

Date: 2026-07-20.
Branch: `m7-cooperative-sacrifice`.

## Question

Does MA-POCA's counterfactual credit assignment teach a shared-policy pair to *cooperate by sacrifice* - one agent holding a pressure plate so the other can cross a door and score - more than naive reward-sharing or selfish per-agent learning can?
M7 is a credit-assignment ablation on a two-agent plate-and-door task: the reward-routing scheme is the ablated lever, and everything else (network, observations, curriculum, the sparse extrinsic reward) is held identical across the three arms.

## The pre-registered claim

Fixed before the eval, this was the headline hypothesis under test:

> MA-POCA (credit assignment) will show the **highest cooperative fraction** - it can assign credit to the agent that held the plate without scoring, so it should learn the sacrifice.
> Selfish per-agent PPO, which pays an agent nothing for a partner's score, will show ~0 cooperation.
> Naive reward-sharing (shared PPO) will land in between.

This document reports the verdict against that claim honestly, in the same discipline as M5's RND null and M6's cued-binding negative: the negative is a feature of the benchmark, not a failure of it.

## Task and method

Two `CharacterController` agents spawn in the near chamber of a two-room arena divided by a wall whose only gap is a doorway blocked by a door.
The door opens only while an agent stands on a pressure plate.
The goal object sits in the far chamber, past the door.
A `coop_difficulty` curriculum ramps the geometry:

| Rung | Geometry | Solo-solvable? |
|------|----------|----------------|
| C0 | door always open, goal distance bootstrapped from near->far | trivially (warmup) |
| C1 | door closed, plate 2u beside the doorway, 4.0 s dwell grace after the plate is vacated | **yes** - one agent can tap the plate and sprint through before the door re-closes |
| C2 | door closed, plate in a back corner of the spawn chamber (off the goal-approach path), 2.0 s dwell | **no** - the plate is not on the way to the goal, so one agent must deliberately hold while the other crosses |

Three arms differ **only in reward routing** (identical network with LSTM memory, identical observations, identical curriculum and sparse `+1`-on-score / per-step time-cost reward in all three):

| Arm | Reward routing | Role |
|-----|----------------|------|
| `poca` | MA-POCA group reward with counterfactual credit assignment | the mechanism under test |
| `shared` | plain PPO, the `+1` outcome shared to both agents | naive reward-sharing baseline |
| `selfish` | plain PPO, each agent rewarded only for its own score | no-incentive-to-sacrifice floor |

**Evaluation** is a **paired** protocol forked from M5/M6: every arm and seed is scored on the **same** held-out episode set (spawn sides, goal position, plate side, terrain RNG re-seeded identically per episode), and the pairing is not assumed but **verified** - each arm emits a per-episode layout fingerprint (`m7_pairing_<arm>.csv`), and the three arms' fingerprint streams are byte-identical (`cmp` clean pairwise, 76 lines each).
Eval always forces the **hard** C1 setting: `EvalMode` bypasses the training-time dwell ramp, so the door-open grace is the final 4.0 s at eval, the tightest C1 sacrifice measurement the task supports.

**The cooperation discriminator** is computed within successful episodes only:

- **cooperative** = `holder_idx != scorer_idx and scorer_idx != -1` - one agent accumulated the most plate-time while the *other* agent crossed the door and scored (a division of labour).
- **solo** = `holder_idx == scorer_idx` - the same agent tapped the plate and crossed alone (tap-and-sprint).

9 models (3 arms x 3 seeds), evaluated on 75 held-out C1 episodes per arm (3 seeds x 25 episodes) = 225 episodes total.
Figure: `training/eval/m7_coop.png`.
Raw episodes: `training/eval/m7_coop.csv`.
Computed statistics: `training/eval/m7_stats.csv`.

## Results (n=3 seeds, 225 episodes: 3 arms x 3 seeds x 25 C1 episodes)

![M7 cooperative sacrifice](../training/eval/m7_coop.png)

### The clean positive: MA-POCA more than doubles task success

| Arm | successes | success rate | 95% Wilson CI | per-seed success |
|-----|----------:|-------------:|---------------|------------------|
| `poca` (MA-POCA credit) | 53 / 75 | **0.707** | (0.596, 0.798) | 0.80 / 0.36 / 0.96 |
| `shared` (naive reward-sharing) | 24 / 75 | 0.320 | (0.225, 0.432) | 0.08 / 0.00 / 0.88 |
| `selfish` (per-agent PPO) | 26 / 75 | 0.347 | (0.249, 0.459) | 0.40 / 0.20 / 0.44 |

MA-POCA's success CI (0.596, 0.798) is **fully separated** from both shared's (0.225, 0.432) and selfish's (0.249, 0.459) - no overlap - and poca more than doubles each on the point estimate.
Because n=3 seeds is small and the per-seed rates are dispersed, the pooled Wilson CI (which treats episodes as i.i.d.) is not the whole story, so the stronger, clustering-robust statement is worth stating plainly: **poca is the top arm in every single seed**, and even poca's *worst* seed (0.36) beats shared (0.00) and selfish (0.20) at that same paired seed.
The competence result does not rest on one lucky seed.

Probability-of-improvement on success, over the paired 3-seed x 25-episode matrix (method-parity with M5/M6):

| Comparison | PoI | 95% CI |
|-----------|----:|--------|
| poca > shared | 0.693 | (0.627, 0.760) |
| poca > selfish | 0.680 | (0.620, 0.740) |
| shared > selfish | 0.487 | (0.420, 0.553) |

Both poca comparisons sit above 0.5 with CIs that exclude it (poca favoured), but below the M5/M6 pre-registered 0.75 "strong effect" bar.
This is expected and carries no information the success CIs do not: PoI on a **binary** metric is mechanically about `0.5 + 0.5 x (success-rate gap)` - roughly `0.5 + 0.5 x 0.387 = 0.69` here - and the many failure-vs-failure ties (both arms fail the majority of episodes) pull it toward 0.5.
As in M6's own PoI-construction disclosure, the headline rests on the fully separated Wilson success CIs, not on this tie-diluted PoI leg.

### The refuted headline: MA-POCA is the *least* cooperative, not the most

Within successful episodes, the cooperative fraction:

| Arm | cooperative | solo | coop-fraction | 95% Wilson CI (n = successes) |
|-----|------------:|-----:|--------------:|-------------------------------|
| `poca` | 21 | 32 | **0.396** | (0.276, 0.531), n=53 |
| `shared` | 13 | 11 | 0.542 | (0.351, 0.721), n=24 |
| `selfish` | 14 | 12 | 0.538 | (0.355, 0.712), n=26 |

The pre-registered ordering was poca > shared > selfish on cooperation.
The observed ordering is the reverse: **poca is nominally the lowest**, and shared is statistically indistinguishable from selfish (0.542 vs 0.538).
This is reported without spin - no attempt was made to reshape the discriminator or select an alternate metric to rescue the hypothesis, per the project's experimental-validity directive.

The verdict has two honest halves, and it is important not to overclaim either:

1. The pre-registered directional claim - *credit assignment produces the most visible sacrifice* - is **refuted**: poca is not elevated on cooperation, it is the lowest of the three.
2. But poca is **not significantly *less* cooperative** than the others.
   A 2-proportion z-test gives poca-vs-shared z = -1.19 and poca-vs-selfish z = -1.20 (about 1.2 sigma each, two-sided p ~ 0.23), and the Wilson CIs overlap heavily.
   The right statement is "poca shows no more cooperation than the naive baselines," not "poca cooperates significantly less."

A second, independent measure agrees with the refutation.
Mean `plate_hold_frac` (fraction of steps an agent stood on the plate):

| Arm | all episodes | successful episodes |
|-----|-------------:|--------------------:|
| `poca` | 0.095 | 0.113 |
| `shared` | 0.088 | 0.098 |
| `selfish` | **0.118** | **0.156** |

The robust reading here is not "poca lowest" (poca and shared are tied at the low end within noise) but the direction that actually bears on the hypothesis: **`selfish` - the arm with zero incentive to hold a plate for a partner - holds the plate the *most*, on both the overall and the success-subset numbers**, and poca is never elevated.
Plate-holding does not track credit assignment in the hypothesized direction, on either metric.

## The mechanism: why the strongest policy cooperates least

C1 is **solo-solvable by construction**: with the 4.0 s door-open grace, one agent can tap the plate beside the doorway and sprint across alone before the door re-closes.
So at C1, cooperation is a *crutch that weaker policies fall back on* because they cannot reliably solo, while the strongest policy solves it solo - which is more efficient.
Competence and the cooperation proxy therefore **anti-correlate** at a solo-solvable rung, exactly inverting the naive hypothesis.

The data supports this reading directly.
Mean steps-to-goal on successful episodes:

| Arm | all successes | solo successes | cooperative successes |
|-----|--------------:|---------------:|----------------------:|
| `poca` | **847** | **683** | 1097 |
| `shared` | 1002 | 702 | 1257 |
| `selfish` | 1355 | 1231 | 1461 |

Two things fall out.
First, within poca, **solo successes are far faster than cooperative ones** (683 vs 1097 steps) - the solo tap-and-sprint is the efficient path, and the coordinated hold-and-cross is the slow one.
Second, **poca is the fastest arm overall** (847 steps, vs 1002 shared and 1355 selfish) and leans solo (32 solo vs 21 cooperative successes, 60% solo).
The most competent policy found the efficient solo route and used it most; the weaker arms fall back on the slower coordinated route more often, which inflates their cooperative fraction without reflecting any "learned sacrifice" the credit-assignment story predicted.

## The airtight boundary: the one rung where sacrifice was forced was not learnable

The only rung where genuine sacrifice was **geometrically forced** was C2: the plate moves from beside the doorway to a back corner of the spawn chamber, off the goal-approach path, so an agent heading to the goal never touches it incidentally - one agent must deliberately go to the corner and hold while the other crosses.
The benchmark never established a surface where sacrifice was **both required AND learnable at a local compute budget**, and this was established rigorously rather than assumed.

- **P4 (no bridge):** the poca policy, warm-started from its best C1-competent checkpoint (2M steps, C1 success 0.27) and continued into C2 for 600k steps, held a **flat -1.000 timeout floor across all 30 training summaries** and evaluated **0/100 with zero plate contact** (`holder_idx == -1` in all 100 episodes).
  The C1-competent policy goal-seeks toward the far chamber, which points directly *away* from the back-corner plate, so the incidental plate-clipping that carried C1 never occurs.
- **The scene is scoreable - the null is real.** Before asserting any mechanism, the C2 zero was discriminated against a broken scene using the harness's own `M7CoopEval.Selftest`: a scripted forced sacrifice at C2 geometry (one agent teleported onto the corner plate, the other onto the goal) **latches success**, and the corner plate registers a grounded holder and drives the door.
  So the -1.000 floor is a genuine discovery gap, not a scene artifact - the same "verify by driving the real flow" discipline that caught the earlier flatlines.
- **P5 (plate-distance bridge, 1M warm-started):** a competence-gated C2 plate-distance ramp (plate starts at the C1 doorway spot and Lerps toward the corner as C2 successes accrue) produced a genuinely different training signature - early off-floor variance instead of P4's dead floor, confirming the bridge changes dynamics - but that signal collapsed by ~220k and never became durable.
  Eval was **0/100 with zero plate contact at every checkpoint** (250k / 500k / 750k / 1M).

C2 is therefore **not reachable from the C1 checkpoint at a local (600k-1M step) budget, even with the plate-distance bridge**.
The hard, forced-sacrifice rung resists sparse-reward learning at this budget.

## The methodology arc: how C1 was made learnable at all

Reaching even the C1 result required diagnosing and fixing a sequence of flatlines, all found by driving the real training/eval flow rather than inferring from curves.
The arc: **three curriculum bridges** (all landed and reviewed), **eight-plus throwaway probes**, and **three code bug-fixes**.

Three curriculum bridges (each an arm-symmetric, competence-gated un-freeze, orthogonal to the ablated reward-routing lever):

1. **C0 goal-distance bootstrap** - the goal starts in the near chamber beside the doorway (door already open) and migrates out to the far chamber as C0 successes accrue.
2. **C1 dwell ramp** - the door-open grace starts at 30 s (so an incidental plate tap by a C0-competent policy converts into a full C1 success) and hardens to the target 4.0 s as C1 successes accrue; `EvalMode` always measures the full 4.0 s.
3. **C2 plate-distance ramp** - the plate migrates from the C1 doorway spot to the far corner as C2 successes accrue (the bridge P5 tested).

Three code bugs, each surfaced by a probe flatline and fixed:

1. **The C0 bootstrap never placed goals near the agents.** The original "goal-distance bootstrap" only varied depth *within* the far chamber, so even the easiest C0 goal sat ~8u away through a 3u doorway - no easy goal ever existed, so no learning gradient ever formed. Fixed to a true near->far ramp.
2. **The C0 ramp outran the policy.** After the placement fix, the ramp was time-gated *and* ran 2.5x too fast (an Academy-step vs trainer-step unit mismatch), so it dragged the goal out of the learnable corridor faster than the young policy could follow - the policy collapsed to the timeout floor immediately after the first gradient update. Fixed by making the ramp **competence-gated** (it advances per success, so it can never outrun the policy). A shared-PPO control run reproduced the same collapse with healthy policy entropy, confirming the fault was arm-independent curriculum timing, not a POCA-specific reward-routing bug.
3. **The closed door read as a wall.** The original door was opaque to the ray sensors, so the moment it closed at C1 the goal vanished from perception (the rays hit the door), breaking the perceptual continuity a C0-competent, goal-seeking policy relied on across the C0->C1 boundary. Fixed with a **barred door** - ray-transparent but physically blocking - so the goal stays visible through the closed door; this was the first fix to lift C1 off the structural true-zero floor.

The probe sequence (P1, P1b, P1c, P1d, P1e, P1f, P1g, P1h, the shared-PPO control, P4, P5) also settled a load-bearing question: **the two-agent sacrifice chain never emerges from undirected exploration.**
P1f warm-started a C0-competent policy into a maximally soft C1 (permanent 30 s door-open grace) for 400k steps and still got a **TRUE ZERO** - zero tap-then-partner-crosses events.
The first success has to be created by construction (a soft rung, a bridge), never discovered by chance in this sparse-reward arena.

## Honest limitations

- **The shared-policy-both-slots structure weakens the holder != scorer proxy.**
  All three arms load the *same* onnx file into *both* agent slots at eval time, so `holder_idx != scorer_idx` can arise from spawn/role symmetry as much as from an intentionally learned "I hold so you can score" strategy.
  This is a weaker proxy for genuine sacrifice than it would be under two independently-acting policies, and it plausibly explains why selfish's cooperative fraction is statistically indistinguishable from shared's (0.538 vs 0.542): without a per-agent identity difference, a "cooperative" pattern can emerge from geometry alone, regardless of the training incentive.

- **Seed variance is severe at n=3.**
  shared is 2/0/22 across seeds 0/1/2 - seed 2 alone carries the arm's aggregate - and poca is 20/9/24 (bimodal, seed 1 far below the other two).
  The cooperation-fraction Wilson CIs are therefore *optimistic*: because successes cluster by seed, the effective sample size is closer to 3 than to the raw success count, so the true uncertainty is wider than the intervals shown.
  This only reinforces the "not significant" verdict on the cooperation discriminator - it does not rescue the hypothesis.

- **C1 is solo-solvable, so the discriminator was measured where cooperation is optional.**
  The one rung where sacrifice was geometrically forced (C2) was not learnable at a local budget (P4/P5), so the benchmark could not measure the question it was really after - "does credit assignment help when sacrifice is *required*."
  The cooperation result is honest about the rung it was measured on, and that rung does not force the behavior the hypothesis was about.

- **PoI on binary success is tie-diluted** and carries no information beyond the separated success CIs; it is reported for method-parity with M5/M6, not as an independent leg of the argument.

## Findings

1. **The competence result is clean and strong.** MA-POCA counterfactual credit assignment more than doubles C1 task success versus both naive reward-sharing and selfish per-agent PPO (0.707 vs 0.320 / 0.347, fully separated Wilson CIs, poca the top arm in every seed). Credit assignment strongly aids solving this cooperative plate-and-door task.

2. **The headline cooperation hypothesis is refuted, unforced.** We pre-registered that poca would show the highest cooperative fraction (learned sacrifice); it shows the lowest (0.396 vs 0.542 / 0.538), and selfish is indistinguishable from shared. The refutation of the directional claim is clean; poca being *significantly less* cooperative is not established (~1.2 sigma, overlapping CIs).

3. **The mechanism makes sense and is data-backed.** C1 is solo-solvable, so cooperation is a crutch of the weaker policies; the strongest policy solves it solo and faster (poca solo successes 683 steps vs cooperative 1097; poca fastest overall at 847). Competence and the cooperation proxy anti-correlate at a solo-solvable rung.

4. **The hard boundary is characterized, not hand-waved.** The only rung where sacrifice was geometrically forced (C2) was not learnable at a local budget - 0/100 with zero plate contact, both without a bridge (P4) and with a plate-distance bridge (P5) - while a harness self-test confirms the scene *can* score a forced C2 sacrifice. The benchmark never established a surface where sacrifice was both required and learnable at this budget.

## Conclusion

M7 delivers four things.
(a) A clean credit-assignment **competence** result: MA-POCA more than doubles success on the cooperative task, with separated CIs and per-seed dominance.
(b) An honest **refutation** of the naive "credit assignment => more visible sacrifice" story at the measurable rung, reported as-is with the significance stated both ways.
(c) A **characterized boundary**: the hard, geometrically-forced sacrifice resists sparse-reward learning at a local budget even with curriculum bridging, established against a proven-scoreable scene.
(d) The full validated **infrastructure** (paired byte-verified eval, arm-symmetric curriculum) and a rigorous multi-probe debugging methodology (three bridges, eight-plus probes, three bug-fixes, all driven end to end).
The negative is a feature, not a failure - the same discipline as M5's RND null and M6's cued-binding negative.
The result that would have made the flashier headline (credit assignment teaches sacrifice) is simply not what the data shows at the rung the data could measure, and that is reported without spin.

## Reproduce

- Train: `training/run_m7_ablation.sh` (3 arms x 3 seeds).
  Configs: `training/configs/m7_{poca,shared,selfish}.yaml`.
- Eval (per arm, needs a StandaloneOSX player of the Coop scene): `M7_SEEDS=0,1,2 M7_LESSONS=1 M7_EPISODES=25 Unity -batchmode -projectPath NavSim -executeMethod M7EvalBatch.RunPoca` (`.RunShared` / `.RunSelfish`), writing to `training/eval/m7_coop.csv`.
- Analyze: `.venv-nav/bin/python training/eval/plot_m7_coop.py` (writes `training/eval/m7_coop.png` + `training/eval/m7_stats.csv`).
