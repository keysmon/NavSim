# M8 BC continuous-action scale discriminator report

## Verdict

**COMPLETE — action scaling was a real BC blocker, but correcting it did not
produce hard-start ramp contact or placement.**

The corrected 79,944-step policy reached a deterministic environment-space
forward mean of `0.958150` on real Push80 expert states, compared with
`0.331560` for the original 149,951-step policy. The correction therefore
reached the trained policy and action transform. Nevertheless, both corrected
checkpoints scored `0/5` hard-start placement and `0/5` goal success.

Under the fixed decision rule this is the **zero placement with offline
environment forward >= 0.9** outcome. The next investigation is bounded
action-persistence/contact capture at the Unity control seam. No follow-up
training was launched. Probe B, S1/S2, the real batch, combined
demonstrations, a 600k run, new demonstration recording, runtime changes, and
model promotion remain hard-stopped.

## Root cause and correction

ML-Agents Python `1.1.0` trains its Gaussian policy in raw action space, while
`AgentAction.to_action_tuple(clip=True)` sends
`clamp(raw_action, -3, 3) / 3` to Unity. The installed BC implementation
compared raw sampled policy actions directly with Unity-space demonstration
labels. The original Push80 checkpoints consequently learned raw forward
means near `1.0`, which became environment actions near `1/3`.

The tracked runner:

```text
training/tools/m8_bc_action_scale_runner.py
```

validates ML-Agents `1.1.0` and the real `[-3, 3] -> [-1, 1]` action
transform, then replaces only `BCModule._behavioral_cloning_loss` in process.
Continuous expert targets are multiplied by `3.0`; discrete cross-entropy is
semantically unchanged. PPO, runtime/export clipping, demonstrations, Unity
packages, site-packages, observations, geometry, physics, rewards, and
curriculum were not changed.

The standard-library test suite covers:

- pinned transform round-trip and version rejection;
- empty and mismatched continuous-action validation;
- corrected hybrid BC loss and continuous/discrete gradient flow;
- idempotent patch installation;
- a 100-update synthetic convergence proof (`raw > 2.9`,
  environment action `> 0.96`);
- the real Push80 sequence/action audit; and
- original-checkpoint replay.

Final Python result: `10/10` passing.

## Demonstration and sequence audit

The discriminator reused, without modification:

```text
NavSim/Assets/Demonstrations/M8RampPush80.demo
SHA-256 871a430f184a273b6a1ea87db8d4e0453d16b83dd9504486246a11acd1dc1330
```

The deterministic offline audit recorded:

```text
processed experiences: 25,344
real experiences:      23,808
padding experiences:    1,536 (6.0606%)
terminal markers:           80
episode lengths: 23 x1, 300 x11, 301 x51, 302 x17
forward > 0.9:  0.996682
zero-ish turn:  0.995632 (absolute turn <= 1e-5)
jump labels:     23,808 x action 0
```

The single 23-decision episode is the previously identified recorder-startup
partial. It was reported but not re-recorded; the other 79 trajectories are
full 300–302-decision hard pushes, so changing the demo would have introduced
a second variable.

The original checkpoint replay reproduced the diagnosed seam:

| Checkpoint | Raw forward | Environment forward | Raw expert-label MSE | Forward sigma | No-jump probability |
|---|---:|---:|---:|---:|---:|
| 99,986 | 1.001644 | 0.333881 | 0.001503 | 0.955615 | 0.979326 |
| 149,951 | 0.994679 | 0.331560 | 0.001435 | 0.949116 | 0.984495 |

The preserved machine-readable audit is
`training/eval/m8_bc_action_scale_offline_audit.json`, SHA-256
`c16311b2692909c22a42c27171603534bdcfc8cffed49d211371bda158cd4369`.

## Bounded training run

The diagnostic config differs from Push80 only in:

```yaml
checkpoint_interval: 40000
max_steps: 80000
```

Exact single trainer invocation:

```text
caffeinate -i .venv-nav/bin/python training/tools/m8_bc_action_scale_runner.py training/configs/m8_probeA_bc_push80_scale_checkpoint_diag.yaml --run-id=m8_probeA_bc_push80_scale_checkpoint_diag --seed=0 --env=NavSim/Builds/M8RampSolo.app --no-graphics --time-scale=20 --num-envs=4 --base-port=5024
```

Startup printed:

```text
[M8BCActionScale] validated=mlagents-1.1.0 scale=3.0
```

All four Unity workers connected. The process exited `0` at step `80,008`
after `114.312` seconds. BC duration remained 150k and the Push80 demo path
was preserved.

Pretraining loss was nonzero throughout:

```text
20k 3.280772; 40k 1.457880; 60k 1.104297; 80k 1.011599
```

Easy-curriculum placement appeared from 30k and peaked at `0.411765` at
70k. Reached-approach peaked at `0.875000` at 60k. Goal success remained
zero. Start distance stayed `1.750000`, so this telemetry is not a hard-start
result.

Scheduled evaluated checkpoints:

```text
RampAgent-39960.pt
SHA-256 61e787ac21d41772073de43efa1047daf019c7c53649b7bc4c9cdc12de0b48e0

RampAgent-79944.pt
SHA-256 d857f1f842a6253b74e64546119d1ebe39f3dfa5400942bed16ac7528f8964fb

RampAgent-39960.onnx
SHA-256 12567d6bcfff61a3eaeb36109408dcf4e762d1fc996d4f277373a352997420d5

RampAgent-79944.onnx
SHA-256 49964f4b11a504d895d5b883f8cb3da05ceca8397603ec7535953b02bf78901b
```

## Corrected checkpoint action gate

| Checkpoint | Raw forward | Environment forward | Raw scaled-target MSE | Forward sigma | No-jump probability |
|---|---:|---:|---:|---:|---:|
| 39,960 | 2.471229 | 0.823743 | 0.149122 | 0.973091 | 0.918501 |
| 79,944 | 2.874451 | 0.958150 | 0.019926 | 0.955812 | 0.964067 |

The 79,944 checkpoint passed the required
`environment_forward_mean >= 0.9` gate, authorizing the bounded Unity
rollout.

## Hard-start rollout

The temporary evaluator used stochastic Burst inference
(`DeterministicInference=false`), reset recurrent state with `EndEpisode()`,
seeded each layout, selected lesson zero under `EvalMode`, and stepped
`Academy.EnvironmentStep -> Physics.Simulate -> RampArena.Tick`. Each episode
was capped at 3,000 steps.

Ten rows are preserved at
`training/eval/m8_bc_push80_scale_checkpoint_rollout.csv`, SHA-256
`90705ae48219d364a6623def60eca8ad1545b75e35de2fbd5e1373716d20e871`.

| Checkpoint | Placement | Goal | Minimum ramp-target distance range | Maximum agent Y range |
|---|---:|---:|---:|---:|
| 39,960 | 0/5 | 0/5 | 4.262295–4.990689 | 1.826001–4.243051 |
| 79,944 | 0/5 | 0/5 | 4.878531–5.000000 | 2.869324–3.386471 |

All ten episodes timed out. The 79,944 policy produced a high offline forward
action but barely changed ramp-target distance in Unity, making
action persistence/contact capture the next discriminating measurement.

## Validation and preservation

Preflight passed:

- `168/168` complete EditMode tests;
- M8 physics `SELFTEST -> True`;
- Unity `ValidatePush80Demo`;
- Python `demo_to_buffer` validation;
- deterministic offline audit regeneration;
- unchanged demonstration, canonical-scene, project-setting, prior-config,
  prior-checkpoint, prior-CSV, and prior-report hashes;
- absent target result directory before launch; and
- no trainer/player process or listener on ports 5024–5027.

The result directory and all checkpoints are preserved under the new run ID.
Only the temporary evaluator source/meta and imported ONNX copies/meta were
removed. The accidental `NavSim/training/` path is absent. Final verification
repeated Python tests, offline audit regeneration, the complete Unity suite,
physics self-test, Push80 validation, Python demo loading, protected hashes,
process/port checks, `git diff --check`, and repository status. Unity's
incidental demo-meta whitespace and project define rewrites were restored.

`docs/research/` remains unmodified and untracked.
