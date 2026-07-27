# M8 BC checkpoint discriminator

## Goal

Determine, with the least additional compute and monitoring, whether the current
behavioral-cloning phase ever produces a complete hard-start
push-place-climb policy, or whether PPO destroys a policy that BC had already
learned.

## Experiment

Create one diagnostic config derived from `training/configs/m8_probeA_bc.yaml`.
Change only:

- `max_steps`: 160000;
- `checkpoint_interval`: 50000;
- retain enough checkpoints to preserve 50000, 100000, and 150000.

Keep the demo, BC strength and duration, PPO settings, seed, four environments,
ports, Unity player, curriculum, geometry, rewards, and observations unchanged.
Use a new run ID and preserve every existing result directory.

Run training foreground-supervised outside the restricted sandbox. Do not poll
the trainer repeatedly; inspect it once at terminal completion unless it
requests attention.

## Evaluation

First verify that the run completed, BC loss was nonzero, and the expected
checkpoints exist. Then evaluate the 100000 and 150000 checkpoints with five
stochastic hard-start solo rollouts each using the existing manual-step
evaluation pattern.

Record placement, goal success, episode steps, and trajectory extrema. A
complete BC policy requires at least three of five push-place-climb successes
at a checkpoint. Do not promote either model.

## Decision

- If a checkpoint passes, BC learned the sequence and later PPO retention is
  the failing seam. The next design should retain or gradually anneal BC.
- If neither checkpoint passes but placement occurs, BC learns only the push.
  The next design should improve sequence coverage through staged or expanded
  demonstrations.
- If neither placement nor goal occurs, audit demonstration consumption and
  policy/action alignment before changing training strength.

Regardless of outcome, do not start Probe B, S1/S2, or the real batch.

## Validation and cleanup

Preserve the diagnostic result directory and raw evaluation evidence. Remove
only temporary rollout source/model-import artifacts after evaluation. Verify
no trainer or player remains, ports are clear, protected project hashes are
unchanged, and authored diffs pass whitespace checks.
