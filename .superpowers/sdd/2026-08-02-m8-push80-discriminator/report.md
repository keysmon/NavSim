# M8 Push80 BC checkpoint discriminator report

## Verdict

**COMPLETE — placement-only hard-start demonstrations still did not produce
hard-start placement.**

The 99,986-step and 149,951-step policies each scored 0/5 placement and 0/5
goal success from the hard 5-unit start. Removing the post-placement climb from
all 80 expert trajectories therefore did not make BC learn the full push.

The fixed decision rule now stops demonstration expansion. The next work is a
separate investigation of the supervised recurrent/action representation
seam. Probe B, S1/S2, the real batch, a combined-demo run, another 600k run,
and model promotion remain hard-stopped.

## Demonstration

The isolated placement-terminal demonstration is:

```text
NavSim/Assets/Demonstrations/M8RampPush80.demo
SHA-256 871a430f184a273b6a1ea87db8d4e0453d16b83dd9504486246a11acd1dc1330
```

Its bounded recording report contains exactly:

```text
mode=record-push80
completed=true
attempts=[0,0,0,80]
placements=[0,0,0,80]
successes=[0,0,0,0]
recordedEpisodes=80
episodeStartDistances=80 values equal to 5.0
```

Unity imported and validated:

```text
demonstrationName=M8RampPush80
episodes=80
steps=23809
continuous actions=2
discrete branches=[2]
observation shapes={[42], [42], [6], [78]}
```

Python `demo_to_buffer` loaded 25,344 experiences with the same action and
observation specification.

The prior demonstrations remained unchanged:

```text
M8RampSoloExpert.demo
SHA-256 3efcb84d2bfb7cea3bdcac1f2bf70823572a764813729d0d0bab345793ab4a54

M8RampHard80.demo
SHA-256 19682c947f9092e9b7dadcab3c1a1fb67f8fa2d05256931337fba79a62323c29
```

## Isolated configuration and run

`training/configs/m8_probeA_bc_push80_checkpoint_diag.yaml` differs from the
Hard80 discriminator config only in:

```yaml
demo_path: NavSim/Assets/Demonstrations/M8RampPush80.demo
```

Exact trainer command:

```text
caffeinate -i .venv-nav/bin/mlagents-learn training/configs/m8_probeA_bc_push80_checkpoint_diag.yaml --run-id=m8_probeA_bc_push80_checkpoint_diag --seed=0 --env=NavSim/Builds/M8RampSolo.app --no-graphics --time-scale=20 --num-envs=4 --base-port=5024
```

All four workers connected. The single invocation completed at 160,002 steps
in 212.501 seconds. Results and all checkpoints are preserved at
`results/m8_probeA_bc_push80_checkpoint_diag/`.

## Training evidence

`Losses/Pretraining Loss` was nonzero at every recorded sample:

```text
20k 1.482948; 40k 1.084426; 60k 1.000507; 80k 0.958449;
100k 0.939614; 120k 0.926137; 140k 0.913687; 160k 0.903409
```

This proves the Push80 demonstration was consumed. Easy-curriculum placement
was nonzero from 30k onward and peaked at 0.812500 at 90k. Easy-curriculum goal
success was intermittent and peaked at 0.117647 at 120k. The training start
distance moved only from 1.75 to 1.775809, so this telemetry is not a
hard-start result.

Evaluated checkpoints:

```text
RampAgent-99986.onnx
SHA-256 e6c4cc90289e6af338f46e046036d5cd6166055e37a60326b9ef07d5eb8f8307

RampAgent-149951.onnx
SHA-256 249572a3fbb64b4b8e20a923795ee86d1067ba8942c76050fbfa3064633fe8bd
```

## Hard-start rollout evidence

The temporary evaluator used stochastic Burst inference, reset recurrent
state with `EndEpisode()` before every episode, selected lesson zero under
`EvalMode` for the hard 5-unit endpoint, and stepped
`EnvironmentStep -> Physics.Simulate -> RampArena.Tick`. Every episode was
capped at 3,000 physics steps.

Ten final rows are preserved in
`training/eval/m8_bc_push80_checkpoint_rollout.csv`:

| Checkpoint | Placement | Goal | Minimum target-distance range | Result |
|---|---:|---:|---:|---|
| 99,986 | 0/5 | 0/5 | 3.891–4.867 | FAIL |
| 149,951 | 0/5 | 0/5 | 4.537–4.765 | FAIL |

CSV SHA-256:

```text
01fa0cf11eaf13a1719866c42536bda9296febbd52ba59b21c33c64fd5732017
```

All ten episodes timed out. Several policies jumped (`max_agent_y` up to
4.280), but no rollout placed the ramp.

## Validation and cleanup

Pretraining gates passed:

- 168/168 complete EditMode tests;
- M8 physics `SELFTEST -> True`;
- Unity Push80 demonstration validation;
- Python `demo_to_buffer` validation;
- unchanged canonical scene, prior-demo, Hard80-config, and project-setting
  hashes;
- absent target result directory and free ports 5024–5027.

The temporary evaluator initially exposed a one-segment output-path error:
`Application.dataPath` is `NavSim/Assets`, so `../training` wrote beneath
`NavSim/`. Comparison with the established M7 evaluator identified the correct
`../../training` path. The evaluator was rerun successfully and the accidental
file and empty directories were removed.

After evaluation, only temporary evaluator source/meta and imported ONNX
copy/meta artifacts were removed. The Push80 demo, config, results,
checkpoints, final CSV, and this report are preserved. No follow-up training
was launched. `docs/research/` remains unmodified and untracked.
