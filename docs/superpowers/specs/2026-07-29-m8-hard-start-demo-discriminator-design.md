# M8 Hard-Start Demonstration Discriminator

## Goal

Determine, with the least additional recording and training compute, whether
substantially increasing complete 5-unit expert trajectory coverage allows
behavioral cloning to learn the hard-start push-place-climb sequence.

The experiment is diagnostic only. It must not start Probe B, S1/S2, the real
batch, or another 600k run.

## Demonstration

Create a separate demonstration named `M8RampHard80` containing
exactly 80 successful terminal episodes. Every episode starts with the ramp at
the full 5-unit push distance. This supplies eight times the hard-start
trajectory count in the existing mixed demonstration, which contains ten
5-unit episodes among forty total episodes.

Reuse the current deterministic expert and recording scene. Add a distinct
hard-recording mode or equivalent explicit recording configuration so the
existing mixed schedule remains available and unchanged. The new recording
must fail closed if:

- an episode does not reach the goal;
- an episode does not place the ramp before reaching the goal;
- the completed episode count differs from 80;
- any recorded start distance differs from 5 units; or
- the demonstration metadata name or episode count is incorrect.

Write the new artifact to
`NavSim/Assets/Demonstrations/M8RampHard80.demo`. Preserve
`M8RampSoloExpert.demo`, its metadata, and all previous reports and results.

## Invariants

The experiment changes demonstration coverage only. Keep all of the following
unchanged:

- behavior name and action specification;
- vector and ray observations;
- expert steering and state-machine behavior;
- recording and training scene geometry;
- ramp physics and placement semantics;
- goal-success requirements;
- rewards and episode limits;
- PPO and behavioral-cloning hyperparameters;
- curriculum parameters;
- trainer seed, environment count, player, time scale, and port layout; and
- hard-start rollout mechanics and stochastic inference.

No canonical runtime scene may gain a recorder, recording controller, or expert
agent component.

## Validation Before Training

Extend focused EditMode coverage for the hard-recording schedule, output
identity, terminal metadata handling, and failure conditions. Then:

1. build or validate the recording scene and recorder player;
2. run a bounded expert dry run at the 5-unit start;
3. record exactly 80 successful hard-start episodes under the new name;
4. import and validate the new demonstration metadata;
5. confirm the original demonstration hash is unchanged; and
6. run the complete EditMode suite and the existing M8 physics self-test.

Do not launch training unless all checks pass.

## Short Checkpoint Discriminator

Create `training/configs/m8_probeA_bc_hard80_checkpoint_diag.yaml` by copying
`training/configs/m8_probeA_bc_checkpoint_diag.yaml`. Change only
`behavioral_cloning.demo_path` to the new hard-start demonstration. Retain:

- `max_steps: 160000`;
- `checkpoint_interval: 50000`;
- `keep_checkpoints: 4`;
- BC strength `0.3` for `150000` steps;
- every PPO/network setting; and
- `arm_mode: 0.0` and `ramp_difficulty: 0.0`.

Use a new run ID, result directory, report, and evaluation CSV containing
`m8_probeA_bc_hard80_checkpoint_diag`. Verify the target result directory is
absent and ports 5024–5027 are clear before launch. Run the trainer once and
inspect it at terminal completion rather than repeatedly polling it.

## Evaluation and Decision

Evaluate the checkpoints nearest 100k and 150k using the same hard-start
checkpoint rollout harness as the preceding discriminator. Run five stochastic
episodes per checkpoint, resetting recurrent state before every episode.
Record placement, goal success, episode steps, and the same trajectory extrema
used in `training/eval/m8_bc_checkpoint_rollout.csv`.

A checkpoint passes only if at least three of five episodes complete the
push-place-climb sequence. Apply these outcomes:

- **At least 3/5 complete successes:** hard-start coverage allows BC to learn
  the sequence. Stop and design the next PPO-retention experiment; do not start
  it automatically.
- **Placement occurs but neither checkpoint passes:** BC has learned useful
  push behavior but not the full sequence. Stop and design staged
  demonstrations.
- **Zero placement at both checkpoints:** full-distance coverage alone is
  insufficient. Stop and design a staged push-first demonstration.

Do not promote a model from this discriminator.

## Preservation and Reporting

Preserve the new demonstration, diagnostic config, result directory,
checkpoints, evaluation CSV, and a concise local report. Remove only temporary
rollout source and imported-model artifacts after evaluation. Confirm no
trainer or player remains, ports are clear, protected artifacts retain their
hashes, authored diffs pass whitespace checks, and `docs/research/` remains
unmodified and untracked.
