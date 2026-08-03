# M8 Push-Only Demonstration Discriminator

## Goal

Determine whether behavioral cloning can learn the full 5-unit ramp push when
the climb segment is removed from each expert trajectory.

This is a placement-only Stage 1 discriminator. It must not start Probe B,
S1/S2, the real batch, another 600k run, or a combined-demonstration run.

## Why This Experiment

The mixed 40-episode demo contained ten hard-start trajectories. Replacing it
with 80 complete hard-start push-place-climb trajectories still produced zero
placements in ten hard-start checkpoint rollouts. BC loss was healthy, so the
demo was consumed, but the complete trajectory did not teach the long push.

The next cheapest discriminator removes the post-placement segment. This tests
whether the long push is learnable when every demonstration sequence ends at
the placement event. Goal completion is deliberately outside the Stage 1 pass
criterion.

## Push-Only Demonstration

Create a separate demonstration named `M8RampPush80` containing exactly 80
terminal episodes. Every episode:

1. starts at the full 5-unit ramp distance;
2. uses the existing deterministic expert's unchanged `Push` behavior;
3. ends immediately after `RampArena.RampAtTarget` becomes true; and
4. records placement as the terminal transition without requiring goal
   success.

The recording-only boundary must call `EndEpisode()` while the placement latch
is still true, then reset the arena for the next episode. It must never change
the canonical arena's normal success boundary.

Write the artifact to
`NavSim/Assets/Demonstrations/M8RampPush80.demo`. The name is within
ML-Agents' 16-character demonstration metadata limit. Preserve
`M8RampSoloExpert.demo`, `M8RampHard80.demo`, their metadata, and all previous
results and reports.

## Fail-Closed Recording Contract

The push-only recording mode must fail if:

- an episode reaches the goal instead of terminating at placement;
- placement does not occur within the existing 3000-step episode budget;
- any episode start distance differs from 5 units;
- any terminal episode lacks the placement latch;
- the completed or recorded episode count differs from 80;
- the demonstration metadata name or episode count is incorrect; or
- the canonical runtime scenes gain an expert, recorder, or recording
  controller.

The JSON recording report must identify the push-only mode and contain exact
attempt, placement, goal, episode-count, and start-distance evidence. Expected
totals are 80 attempts, 80 placements, zero goals, 80 recorded episodes, and
80 start distances equal to 5.0.

## Invariants

Change demonstration segmentation only. Keep unchanged:

- behavior name and action specification;
- vector and ray observations;
- expert push steering and action cadence;
- recording and training geometry;
- ramp physics and placement radius;
- canonical success requirements;
- runtime rewards and episode limit;
- PPO, network, and BC hyperparameters;
- curriculum parameters;
- trainer seed, environment count, player, time scale, and ports; and
- stochastic hard-start rollout mechanics.

The push-only placement boundary exists only in the recording player. Training
and evaluation continue using the canonical solo scene and its ordinary
push-place-climb success semantics.

## Validation Before Training

Add focused EditMode coverage for:

- explicit push-only mode parsing and identity;
- the 80-episode all-5-unit schedule;
- placement terminal acceptance and goal terminal rejection;
- one boundary action per episode;
- exact terminal metadata count; and
- preservation of the existing mixed and Hard80 modes.

Then:

1. build and validate the recording scene and recorder player;
2. record exactly 80 placement-only hard-start episodes under the new name;
3. import and validate the new demonstration metadata and sensor/action shapes;
4. load the demonstration through Python `demo_to_buffer`;
5. confirm existing demo and canonical-scene hashes are unchanged;
6. run the complete EditMode suite; and
7. run the existing M8 physics self-test.

Do not launch training unless every gate passes.

## Short Checkpoint Discriminator

Create `training/configs/m8_probeA_bc_push80_checkpoint_diag.yaml` by copying
`training/configs/m8_probeA_bc_hard80_checkpoint_diag.yaml`. Change only:

```yaml
demo_path: NavSim/Assets/Demonstrations/M8RampPush80.demo
```

Retain the 160k maximum, 50k checkpoint interval, four retained checkpoints,
150k BC duration, strength 0.3, all PPO/network settings, seed zero, four
environments, ports 5024–5027, the canonical solo player, and curriculum
parameters.

Use the new run ID `m8_probeA_bc_push80_checkpoint_diag`. Verify its result
directory is absent and the ports are free before launching exactly one
trainer invocation. Monitor startup once, then wait for terminal completion.

## Evaluation and Decision

Evaluate the checkpoints nearest 100k and 150k with five stochastic hard-start
episodes each. Reset recurrent state before each episode and retain the
existing 3000-step manual physics seam.

Record placement, goal, steps, minimum ramp-target distance, and maximum agent
height. Stage 1 passes if either checkpoint places the ramp in at least three
of five episodes. Goal completion is evidence only and is not required.

Apply the fixed outcomes:

- **At least 3/5 placements:** the long push is learnable from placement-only
  trajectories. Stop and design a separate short discriminator whose demo
  path is an isolated directory containing `M8RampPush80.demo` and
  `M8RampHard80.demo`.
- **One or two placements but no pass:** staging has weak signal. Stop and
  design one narrowly strengthened placement-only experiment; do not combine
  demos yet.
- **Zero placement at both checkpoints:** stop adding demonstrations and
  investigate the supervised recurrent/action representation seam.

Do not promote a model from this run.

## Preservation and Reporting

Preserve the new demonstration, diagnostic config, result directory,
checkpoints, rollout CSV, and a concise local report. Remove only temporary
rollout source and imported-model artifacts after evaluation. Confirm no
trainer or player remains, ports are free, protected hashes are unchanged,
authored diffs pass whitespace checks, and `docs/research/` remains unmodified
and untracked.
