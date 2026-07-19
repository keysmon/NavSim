# NavSim - Multi-Agent Navigation Simulator

A 2D Unity + ML-Agents simulator where a shared-policy crowd learns to *search out* hidden goals
and avoid collisions with each other and with obstacles.
Trained with reinforcement learning (PPO, and later MA-POCA), exported to ONNX, and playable
in-browser via a Unity WebGL build with in-engine inference (Unity Sentis).

## Status

Milestones M0-M6 complete. The full pipeline is proven end to end
(Unity -> ML-Agents -> ONNX -> Sentis -> WebGL -> Vercel):

- **M0** - single-agent pipeline smoke: an agent reaches a visible goal in-browser.
- **M1** - static obstacles + collision avoidance.
- **M2** - 2-8 shared-policy crowd with color-coded goals, a cooperative (congestion) reward, and
  visual-only observations (no compass oracle); interactive crowd-size slider.
- **M3** - a single collapsed difficulty curriculum (agent count x arena size x obstacle density) with a
  held-out generalization evaluation (see below); interactive difficulty + "new layout" controls.
- **M4** - hidden-goal search: ray length becomes a curriculum axis that fades from visible to hidden, the
  distance-shaping reward is gated on visibility, and the policy gains an LSTM + ICM curiosity; a 4-arm
  ablation (see below) with an interactive goal-visibility slider.
- **M5** - 3D terrain search: a single learner navigates procedurally generated 3D terrain (ramps, elevated
  platforms, hazard pits, oblivious movers) to a line-of-sight-gated goal via ray-fan perception; a paired,
  pre-registered LSTM x RND ablation (see below) found memory gives a modest, real efficiency gain and
  curiosity gives none.
- **M6** - visual object-goal search: perception itself becomes the ablated lever - a from-scratch CNN over
  egocentric RGB pixels vs. ray sensors with/without hand-told colour tags, on a fixed-target colour
  discrimination task; a paired, pre-registered ablation (see below) found the CNN matches the hand-told
  upper bound and clearly beats the colour-blind ray sensor.

**Live demo:** https://navsim-webgl.vercel.app

Remaining: M7 (MA-POCA benchmark), M8 (CI/CD).

## M3 - Generalization

M3 trains one shared PPO policy along a single **collapsed difficulty curriculum**: a lone `difficulty`
parameter co-varies agent count (2->8), arena size (half 6->11), and obstacle count (0->8) *together*, so
the curriculum advances along one monotonic diagonal through the (agents x arena x obstacles) cube. Ray
length is pinned to the largest arena's diagonal so goals stay visible at every lesson (keeping M3 in the
visible-goal regime, not hidden-goal search - that is M4). Training climbed all four lessons and broke
through on the hardest 8-agent config around 3M of 5M steps.

The milestone's exit criterion is **generalization**: the policy is evaluated on **held-out off-diagonal**
configurations it never trained as a tuple - e.g. 8 agents crammed into a small arena, 2 agents alone in
the huge arena, an untrained arena size, 2 agents in a dense obstacle field. The harness
(`M3GeneralizationEval`, run in-engine over the trained Sentis policy) sweeps both the trained diagonal and
the off-diagonal grid, reporting goals reached per agent and a collision proxy.

![M3 generalization](training/eval/m3_generalization.png)

**Result:** off-diagonal goal rates stay within the trained-diagonal range (no collapse toward zero) and
body-overlap stays ~0 across every held-out config - the policy transfers to unseen layouts and densities.
Near-encounter frequency rises at the densest off-diagonal configs (8 agents in the smallest arena), which
is a *density* artifact - agents pass closer more often - not a breakdown in avoidance: overlap stays at or
below ~0.4% and closest approach stays near body width. Absolute goal rates carry sampling noise (the
compass-free ray-search behavior yields a low goal rate, so per-config counts are modest); the robust
signal is the off-vs-on-diagonal comparison, which holds. Regenerate the chart with
`training/eval/plot_generalization.py` over the harness CSV.

## M4 - Pure Search (Hidden Goals + LSTM + Curiosity)

M4 crosses M3's explicit boundary. M3 pinned the ray length to the max arena diagonal so the goal was
visible everywhere; M4 unpins it and **fades it along the curriculum** (1.0x -> 0.6x -> 0.35x -> 0.2x of the
diagonal), so on the top rung the goal is visible only within a short radius and the agent must **search**.
The privileged distance-shaping reward is **visibility-gated** - it guides the agent only while a ray could
reach the goal; while the goal is hidden, an **LSTM** (memory) and an **ICM curiosity** intrinsic reward are
the only drives. Room geometry (walls, obstacles, peers) stays ray-visible, so the observation contract is
unchanged from M3. Training uses a deterministic **progress-based** curriculum (visible warmup for the first
15% of steps, then two intermediate rungs, then ~2.75M steps on the hidden rung) - reward-thresholded
lessons proved unusable because the per-episode reward is too noisy to gate on (a probe finding).

**The policy learns hidden-goal search.** On the hidden rung the primary (LSTM + curiosity) climbs from a
cold-start reward of ~-2.7 (reaching almost no goals) to ~+1.6 (reaching goals faster than the step penalty
bleeds), and in the in-engine eval it reaches goals at every visibility and keeps body-overlap at exactly 0.

The exit criterion is an **ablation**: the same env + curriculum trained four ways (LSTM+curiosity /
LSTM-only / curiosity-only / neither) and compared on the hidden rung.

![M4 ablation](training/eval/m4_search.png)

**Result (reported honestly).** The ablation is **inconclusive**. Within these runs the plain-PPO baseline
(no LSTM, no curiosity) is stably the weakest on the hidden rung (~1.40 vs ~1.60-1.70 for the three arms
that have at least one mechanism), but the three mechanism arms are **clustered within noise** and the
primary is not the best of them - so there is no clean "LSTM and curiosity are each necessary" story. Two
honest limits: (1) **n = 1 training seed per arm**, so no difference - including the baseline gap - can be
attributed to the mechanism rather than seed/initialization variance; (2) the eval's absolute reach rate is
low and *non-monotonic* in visibility (a compass-free-search + observation-clutter artifact), so it is shown
as exploratory, not as the headline. The most likely reading is that at 0.2x ray in a 22x22 arena the
"hidden" regime is still **solvable by reactive local search** - the agent bumps around ray-visible
structure and stumbles onto goals often enough that memory and curiosity are not *forced* to matter. A
sharper test would shorten the ray further, enlarge the arena, or occlude the goal behind obstacles, and run
multiple seeds. Regenerate with `Tools/NavSim/Run M4 Search Eval` (writes `m4_search.csv`) and
`training/eval/plot_m4_search.py` (reads that plus the per-arm `m4_l3_reward.csv`).

## M5 - 3D Terrain Search (LSTM x RND Ablation)

M5 moves off M4's flat 2D arena into procedurally generated 3D terrain: a single `CharacterController`
learner (gravity + jump) navigates walls, ramp-reachable elevated platforms, hazard pits, and oblivious
mover-occluders to a line-of-sight-gated goal, perceived through three ray-fans (forward/down/up). A
`difficulty` curriculum ramps L0 (open, flat, goal visible) -> L3 (hidden, elevated, crowded).

The exit criterion is a 2x2 ablation over two levers (LSTM memory x RND curiosity), scored by a **paired**
protocol - every arm and seed evaluated on the *same* held-out layouts, on SPL (Success weighted by
inverse-optimal Path Length), aggregated with `rliable` (IQM + 95% stratified-bootstrap CIs +
probability-of-improvement), against a decision rule pre-registered before any run.

![M5 ablation](training/eval/m5_search.png)

**Result.** Curiosity (RND) is a **clean null** (PoI 0.51, straddling 0.5) - it does nothing for SPL and
actively impedes the initial learning bootstrap. Memory (LSTM) helps, **modestly** (IQM SPL 0.49 vs 0.40;
PoI 0.57, below the pre-registered 0.75 "strong effect" bar), with the benefit concentrated on the easier,
flatter levels and vanishing at the hardest, most-occluded one. Full write-up, methodology, and honest
caveats: [`docs/M5-ablation-results.md`](docs/M5-ablation-results.md).

## M6 - Visual Object-Goal Search (Fixed-Target Colour Discrimination)

M6 makes **perception itself** the ablated lever on top of M5's terrain spine. Three geometrically
identical goals spawn per episode; the target is a **fixed colour** (red), the two decoys are random
distinct non-target colours. Three arms differ only in what they can see (LSTM on, no RND, identical
reward/curriculum in all three): `pixel` (an 84x84 egocentric RGB camera into a from-scratch CNN), `ray1`
(ray fans, one undifferentiated goal tag - a provable chance floor), and `rayC` (ray fans, hand-told
per-colour tags - the upper bound). Evaluation is the same paired, always-hard, `rliable`-aggregated
protocol as M5, gated by a five-point decision rule pre-registered before the full 9-run ablation.

![M6 ablation](training/eval/m6_search.png)

**Result.** The from-scratch CNN **matches the hand-told upper bound and clearly beats the colour-blind ray
sensor.** `pixel` success 0.713 (95% CI 0.660-0.762) is fully separated from `ray1`'s 0.357 (CI 0.305-0.412,
containing chance = 1/3 exactly, confirming no confound) with PoI(SPL) = 1.000. `pixel` vs `rayC` (0.773,
CI 0.723-0.817) overlap on success - a statistical match - with `pixel` nominally ahead on path efficiency
(mean SPL 0.398 vs 0.348). Getting here required diagnosing and discarding a failed first design (a
per-episode colour *cue* that a from-scratch CNN could not learn to bind to pixels across four training
attempts) - the honest negative result that led to this fixed-target redesign. Full write-up, the cued-negative
evidence, and honest caveats: [`docs/M6-results.md`](docs/M6-results.md).

## Layout

- `NavSim/` - the Unity project (created in M0 Task 1).
- `training/` - Python ML-Agents trainer configs and environment.
- `web/` - the WebGL build output deployed to Vercel.
- `docs/superpowers/` - design spec and milestone plans (local only, not pushed).

## Tech stack

Unity 6 LTS, com.unity.ml-agents 4.0.x, Python 3.10.x, Unity Inference Engine (Sentis) 2.x, WebGL, Vercel.
See `VERSIONS.md` for exact pins.
