# M8 Hard80 BC checkpoint discriminator report

## Verdict

**COMPLETE — 80 full-distance demonstrations still did not produce hard-start placement.**

The 99,941-step and 149,940-step policies each scored 0/5 stochastic
push-place-climb successes from the hard 5-unit start. Neither policy placed
the ramp in any rollout. Increasing complete 5-unit trajectory coverage from
10 to 80 therefore did not make BC learn the hard-start push sequence.

The fixed decision rule selects a staged push-first demonstration as the next
design. Probe B, S1/S2, the real batch, another 600k run, and model promotion
remain hard-stopped.

## Demonstration

The new isolated demonstration is:

```text
NavSim/Assets/Demonstrations/M8RampHard80.demo
SHA-256 19682c947f9092e9b7dadcab3c1a1fb67f8fa2d05256931337fba79a62323c29
```

Its bounded recording report contains exactly 80 attempts, 80 placements, 80
goal successes, and 80 start distances equal to 5.0. Unity imported and
validated:

```text
demonstrationName=M8RampHard80
episodes=80
steps=31331
continuous actions=2
discrete branches=[2]
observation shapes={[42], [42], [6], [78]}
```

The originally approved name `M8RampSoloExpertHard80` exceeded ML-Agents'
16-character demonstration-name limit and truncated to the existing
`M8RampSoloExpert`. A failing boundary test reproduced this collision before
the implementation was corrected to the distinct 12-character
`M8RampHard80`. The first temporary candidate was rejected and never installed.

The existing mixed demo remained unchanged:

```text
NavSim/Assets/Demonstrations/M8RampSoloExpert.demo
SHA-256 3efcb84d2bfb7cea3bdcac1f2bf70823572a764813729d0d0bab345793ab4a54
```

## Isolated configuration and run

`training/configs/m8_probeA_bc_hard80_checkpoint_diag.yaml` differs from
`m8_probeA_bc_checkpoint_diag.yaml` only in:

```yaml
demo_path: NavSim/Assets/Demonstrations/M8RampHard80.demo
```

Exact trainer command:

```text
caffeinate -i .venv-nav/bin/mlagents-learn training/configs/m8_probeA_bc_hard80_checkpoint_diag.yaml --run-id=m8_probeA_bc_hard80_checkpoint_diag --seed=0 --env=NavSim/Builds/M8RampSolo.app --no-graphics --time-scale=20 --num-envs=4 --base-port=5024
```

All four workers connected. The single invocation completed at 160,012 steps
in 242.033919 seconds. The result directory is preserved at
`results/m8_probeA_bc_hard80_checkpoint_diag/`.

## Training evidence

`Losses/Pretraining Loss` was nonzero at every recorded BC sample:

```text
20k 1.397915; 40k 1.047020; 60k 0.974923; 80k 0.930517;
100k 0.902817; 120k 0.883099; 140k 0.873925; 160k 0.862552
```

The loss trend proves the Hard80 demo was consumed. Easy-curriculum placement
was nonzero from 30k onward and peaked at 0.444444 at 50k; reached-goal peaked
at 0.210526 at 110k. The training start distance remained on the easiest rung,
increasing only from 1.75 to 1.816912, so these telemetry values are not a
substitute for hard-start evaluation.

Evaluated checkpoints:

```text
RampAgent-99941.onnx
SHA-256 eabe9bb4f280b6d46d49e2e34cd7fbc4e8aadac972b08189bb049b33cbf66ee3

RampAgent-149940.onnx
SHA-256 f453c7f5cea01ab40ba1a106f06c30c7d5237abe65014ccb9492c0696cee12c1
```

## Hard-start rollout evidence

The temporary evaluator used the preceding discriminator's seam: stochastic
Burst inference, recurrent-state reset with `EndEpisode()` before each
episode, lesson zero under `EvalMode` for the hard 5-unit endpoint, and
`EnvironmentStep -> Physics.Simulate -> RampArena.Tick`. Every episode was
capped at 3000 physics steps.

Ten rows are preserved in
`training/eval/m8_bc_hard80_checkpoint_rollout.csv`:

| Checkpoint | Placement | Goal | Minimum target-distance range | Result |
|---|---:|---:|---:|---|
| 99,941 | 0/5 | 0/5 | 4.136–4.846 | FAIL |
| 149,940 | 0/5 | 0/5 | 4.223–4.892 | FAIL |

CSV SHA-256:

```text
09519c77335adbfd9bd725968879f166236799b953ce68116bfbbd63f7307249
```

All ten episodes timed out. Some policies jumped (`max_agent_y` up to 3.826),
but none reduced the ramp-target distance enough to place the ramp.

## Validation and cleanup

Pretraining gates passed:

- 161/161 complete EditMode tests;
- M8 physics `SELFTEST -> True`;
- Unity Hard80 demonstration validation;
- Python `demo_to_buffer` with 35,520 experiences;
- unchanged canonical scene, mixed-demo, prior-config, and project-setting
  hashes;
- absent target result directory and free ports 5024–5027.

After evaluation, only the temporary evaluator source/meta and imported ONNX
copy/meta were removed. The Hard80 demo, config, results, checkpoints, CSV, and
this report are preserved. Fresh post-cleanup verification passed 161/161
EditMode tests, physics `SELFTEST -> True`, Hard80 validation, process/port
audit, protected hashes, and authored diff checks. `docs/research/` remains
unmodified and untracked.
