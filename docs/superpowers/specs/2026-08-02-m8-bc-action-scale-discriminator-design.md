# M8 BC Continuous-Action Scale Discriminator

## Goal

Determine whether the M8 hard-start BC failure is caused by a mismatch between
the continuous-action space used by behavioral-cloning loss and the continuous
action actually sent from the exported policy to Unity.

This is a trainer-side action-scale discriminator. It must not change runtime
agent behavior, demonstrations, geometry, physics, rewards, observations,
action specifications, curriculum semantics, or canonical success rules.

## Root-Cause Evidence

The Push80 demonstration is behaviorally simple after padding is excluded:

```text
real experiences: 23,808
sequence padding: 1,536 / 25,344 (6.0606%)
forward > 0.9: 99.6682%
forward == 0: 0.3318%
abs(turn) < 1e-5: 99.5632%
jump action: 0 in 23,808 / 23,808 experiences
```

The 99,986- and 149,951-step Push80 checkpoints predict the expert action
accurately in the policy's raw Gaussian space:

```text
checkpoint 99,986:
  raw mean [forward, turn] = [1.001585, 0.003799]
  raw mean versus expert MSE = 0.001503
  Gaussian std = [0.955608, 0.945854]
  mean no-jump probability = 0.979326

checkpoint 149,951:
  raw mean [forward, turn] = [0.994612, 0.008409]
  raw mean versus expert MSE = 0.001435
  Gaussian std = [0.949208, 0.938905]
  mean no-jump probability = 0.984495
```

In the installed ML-Agents Python trainer,
`AgentAction.to_action_tuple(clip=True)` transforms continuous policy samples
with:

```python
torch.clamp(raw_action, -3, 3) / 3
```

The exported ONNX action path performs the same transform. The BC module,
however, compares the unscaled raw sampled action directly with demonstration
actions recorded in Unity's `[-1, 1]` environment space. Consequently a
correctly cloned full-forward label of `1.0` becomes approximately `0.333` at
the Unity boundary:

```text
checkpoint 99,986 environment-space deterministic mean = 0.333903
checkpoint 149,951 environment-space deterministic mean = 0.331575
```

The approximately `0.94–0.96` raw Gaussian standard deviation also explains
the observed approximately `0.90` pretraining loss: the continuous BC loss is
MSE against a sampled raw Gaussian action, not against its deterministic mean.
The loss trend therefore proves the demo was consumed but does not mean the
environment receives expert-scale control.

This evidence rejects recurrence, sequence coverage, and label complexity as
the primary cause. Recurrent and normalization behavior remain unchanged in
this discriminator.

## Chosen Approach

Add a tracked, version-pinned training entrypoint that applies the inverse of
ML-Agents' continuous environment-action transform only inside BC loss:

```python
raw_expert_continuous = environment_expert_continuous * 3.0
```

The wrapper must leave discrete cross-entropy unchanged. It must invoke the
installed ML-Agents CLI after installing the scoped patch in the same Python
process. It must not edit `.venv-nav`, site-packages, Unity packages, or
runtime C#.

The scale factor is not an arbitrary hyperparameter. The wrapper must derive
or assert the pinned factor against the installed ML-Agents behavior and fail
closed if the expected `clip(raw, -3, 3) / 3` convention is absent.

The implementation may replace
`BCModule._behavioral_cloning_loss` through a small, explicit wrapper around
the installed version. It must preserve the installed method's hybrid-action
behavior:

- continuous loss remains MSE on the raw policy sample;
- the continuous expert target alone is multiplied by `3.0`;
- discrete expert actions and branch cross-entropy remain byte-for-byte
  equivalent in meaning; and
- PPO loss, value loss, entropy, environment actions, action clipping, and
  model export remain untouched.

## Rejected Approaches

### Change runtime action gain

Multiplying `RampAgent` movement or turn inputs by three would change the
shared action contract for PPO, expert, evaluation, and deployment. It would
invalidate prior geometry/physics/reward evidence and is rejected.

### Record out-of-range expert actions

Recording action `3.0` while the runtime clamps to `1.0` could manufacture the
raw-space target the current BC implementation expects. It would encode
trainer internals into a demonstration, require new artifacts, and violate the
rule to stop adding demonstrations. It is rejected.

### Disable ML-Agents action clipping globally

Removing the `/3` transform from the actor or export path would affect PPO
sampling and every environment action, not just supervised labels. It is
broader than the diagnosed seam and is rejected.

## Tracked Components

Create one focused Python module under `training/tools/` with two
responsibilities:

1. validate the installed ML-Agents version and action-transform contract; and
2. install the continuous BC target correction before delegating to the normal
   ML-Agents CLI.

Keep the correction function independently testable. Its public test surface
must accept expert continuous tensors and return raw-space targets without
depending on a running Unity environment.

Add focused Python tests under the repository's existing training test
convention. If no Python test directory exists, create
`training/tests/test_m8_bc_action_scale.py` using the standard-library
`unittest` framework and run it with:

```text
.venv-nav/bin/python -m unittest training.tests.test_m8_bc_action_scale
```

Do not add or install a test dependency.

Create a new config:

```text
training/configs/m8_probeA_bc_push80_scale_checkpoint_diag.yaml
```

Copy `training/configs/m8_probeA_bc_push80_checkpoint_diag.yaml` and change
only:

```yaml
checkpoint_interval: 40000
max_steps: 80000
```

Keep:

- `M8RampPush80.demo`;
- BC strength `0.3`;
- BC duration `150000`;
- BC batch size `512`;
- BC epochs `3`;
- recurrent sequence length `64`;
- memory size `128`;
- normalization enabled;
- every PPO/network/reward setting;
- seed `0`;
- four environments;
- canonical solo player;
- time scale `20`; and
- ports `5024–5027`.

Use run ID:

```text
m8_probeA_bc_push80_scale_checkpoint_diag
```

## Offline Gates

No Unity trainer run may launch until all offline gates pass.

### Transform contract

Tests must prove:

- environment expert `[-1.0, 0.0, 1.0]` maps to raw targets
  `[-3.0, 0.0, 3.0]`;
- mapping back through `clip(raw, -3, 3) / 3` reproduces the original values;
- intermediate turn values such as `-0.2` and `0.2` round-trip exactly within
  floating-point tolerance;
- the input tensor is not mutated; and
- shape, dtype, and device are preserved.

### Hybrid loss contract

Using a synthetic hybrid action specification with two continuous actions and
one discrete branch of size two, tests must prove:

- corrected continuous loss is computed against the scaled raw target;
- discrete cross-entropy is identical to the installed implementation for the
  same logits and expert branch;
- zero and empty continuous batches fail explicitly rather than silently
  changing loss semantics; and
- unsupported installed ML-Agents behavior fails closed before CLI launch.

### Real demonstration audit

Load `M8RampPush80.demo` with `demo_to_buffer(sequence_length=64)` and preserve
a machine-readable audit showing:

- 25,344 processed experiences;
- 23,808 real experiences;
- 1,536 padding experiences;
- 80 terminal markers;
- action distribution totals stated in the root-cause evidence;
- 79 normal hard-push episodes of 300–302 decisions; and
- one 23-decision startup partial.

The 23-decision episode is a separate recorder-startup finding. It must be
reported but not fixed or re-recorded in this discriminator. Seventy-nine full
hard trajectories are sufficient for the action-scale test, and changing the
demo would add a second variable.

### Checkpoint replay proof

Load the existing 99,986 and 149,951 `.pt` checkpoints and the real Push80
buffer. Reproduce:

- raw deterministic forward mean approximately `1.0`;
- environment-space deterministic forward mean approximately `1/3`;
- raw mean expert MSE below `0.01`; and
- raw Gaussian standard deviation above `0.8`.

This gate ensures the new run is testing the diagnosed seam rather than an
assumed one.

## Short Discriminator

Before launch require:

- every offline gate passes;
- the target result directory is absent;
- ports 5024–5027 are free;
- no trainer or M8 player remains;
- the Push80 demo and all protected hashes are unchanged;
- complete Unity EditMode tests pass;
- M8 physics SELFTEST is true; and
- Push80 Unity and Python demo validation passes.

Launch exactly one trainer invocation through the tracked wrapper:

```text
caffeinate -i .venv-nav/bin/python training/tools/m8_bc_action_scale_runner.py \
  training/configs/m8_probeA_bc_push80_scale_checkpoint_diag.yaml \
  --run-id=m8_probeA_bc_push80_scale_checkpoint_diag \
  --seed=0 \
  --env=NavSim/Builds/M8RampSolo.app \
  --no-graphics --time-scale=20 --num-envs=4 --base-port=5024
```

The wrapper must accept the same positional configuration and CLI flags as
`mlagents-learn`. Do not use `--force` or `--resume`. Inspect startup once,
then monitor sparsely until terminal completion.

Require:

- trainer exit `0`;
- terminal step at least `80000`;
- nonzero pretraining loss;
- the correction's version/contract marker in the launch log; and
- checkpoints nearest 40k and 80k.

## Evaluation and Decision

Before Unity rollout, replay each new checkpoint against the real demo buffer
offline and record:

- raw deterministic forward and turn means;
- transformed environment-space forward and turn means;
- continuous raw-target MSE;
- Gaussian standard deviation; and
- no-jump probability.

The corrected checkpoint must send a deterministic environment-space forward
mean of at least `0.9` on real expert states. If neither checkpoint meets this
offline action gate, do not run Unity rollout; report that the correction did
not reach export.

For each eligible checkpoint, run five stochastic hard-start episodes with:

- recurrent state reset before every episode;
- lesson zero under `EvalMode` for the 5-unit endpoint;
- the unchanged manual physics seam;
- a 3000-step cap; and
- the existing placement, goal, steps, minimum ramp-target distance, and
  maximum agent-height columns.

Stage passes if either checkpoint places the ramp in at least three of five
episodes. Goal completion is evidence only.

Apply the fixed outcomes:

- **At least 3/5 placements:** action scaling is the primary BC blocker. Stop
  and design how to make the correction maintainable before any longer run.
- **One or two placements:** action scaling is causal but insufficient. Stop
  and investigate residual Gaussian variance; do not tune demos or runtime.
- **Zero placement with environment forward mean at least 0.9:** scaling
  reaches Unity but another control seam remains. Stop and capture action
  persistence/contact evidence; do not start another training run.
- **Offline action gate fails:** the wrapper or export path is not corrected.
  Stop and debug the trainer patch; do not evaluate or train again.

Do not promote a model from this run.

## Invariants and Preservation

Preserve all demonstrations, prior configs, results, checkpoints, CSVs, and
reports. Keep unchanged:

- Unity runtime and editor C#;
- behavior name and hybrid action specification;
- vector and ray observations;
- expert steering and action cadence;
- recording and training geometry;
- ramp physics and placement radius;
- runtime rewards and episode limit;
- canonical success requirements;
- curriculum parameters;
- PPO and network hyperparameters other than the explicit 80k cap and 40k
  checkpoint interval; and
- stochastic rollout mechanics.

Preserve the new runner, tests, config, offline audit, result directory,
checkpoints, rollout CSV, and concise local report under new names. Remove only
temporary evaluator and imported-model copies after evaluation.

Probe B, S1/S2, the real batch, combined demonstrations, a 600k run, runtime
changes, new demonstrations, and model promotion remain hard-stopped.
`docs/research/` remains unmodified and untracked.
