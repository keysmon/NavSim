# M8 BC Continuous-Action Scale Discriminator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the environment-to-raw continuous-action target scale only inside ML-Agents behavioral cloning and test it with one bounded 80k M8 hard-start discriminator.

**Architecture:** A tracked Python runner validates the pinned ML-Agents 1.1.0 action transform, replaces only `BCModule._behavioral_cloning_loss` in process, and then delegates to the normal ML-Agents CLI. The same module exposes dependency-free offline demo/checkpoint audit functions, while Unity runtime, demonstrations, PPO, model export, and environment action clipping remain unchanged.

**Tech Stack:** Python 3.10 standard-library `unittest`, PyTorch 2.2.2, ML-Agents Python 1.1.0, Unity ML-Agents package 4.0.3, Unity 6000.5.3f1, YAML, ONNX/Sentis.

## Global Constraints

- Do not modify `.venv-nav`, site-packages, Unity packages, or Unity runtime/editor C#.
- Preserve every demonstration, prior config, result, checkpoint, CSV, and report.
- Keep behavior name, hybrid action specification, observations, expert actions, geometry, physics, rewards, episode limit, canonical success, curriculum, PPO, network settings, seed, environment count, player, time scale, ports, and stochastic rollout mechanics unchanged.
- The only trainer behavior change is multiplying continuous demonstration targets by `3.0` inside BC loss.
- Discrete BC cross-entropy must remain semantically identical to ML-Agents 1.1.0.
- Do not add or install a test dependency; use standard-library `unittest`.
- The only diagnostic-config changes from Push80 are `checkpoint_interval: 40000` and `max_steps: 80000`.
- Run every offline gate before launching exactly one bounded trainer invocation.
- Stage passes only if either eligible checkpoint records at least 3/5 hard-start placements; goal is evidence only.
- Do not start Probe B, S1/S2, the real batch, combined demonstrations, a 600k run, new demonstration recording, runtime changes, or model promotion.
- Do not modify or stage the unrelated untracked `docs/research/`.

---

### Task 1: Add the pinned BC action-scale correction

**Files:**
- Create: `training/tools/__init__.py`
- Create: `training/tools/m8_bc_action_scale_runner.py`
- Create: `training/tests/__init__.py`
- Create: `training/tests/test_m8_bc_action_scale.py`

**Interfaces:**
- Consumes: ML-Agents 1.1.0 `AgentAction.to_action_tuple(clip=True)`, `BCModule._behavioral_cloning_loss`, `AgentAction`, `ActionLogProbs`, and the normal `mlagents.trainers.learn.main()` CLI.
- Produces: `scale_expert_continuous(torch.Tensor) -> torch.Tensor`, `validate_installed_contract(version: str | None = None) -> None`, `corrected_behavioral_cloning_loss(module, selected_actions, log_probs, expert_actions) -> torch.Tensor`, `install_patch() -> None`, and the executable training runner.

- [ ] **Step 1: Write failing transform and contract tests**

Create package marker files and `training/tests/test_m8_bc_action_scale.py`.
Start with:

```python
import unittest

import numpy as np
from mlagents.torch_utils import torch

from training.tools.m8_bc_action_scale_runner import (
    EXPECTED_MLAGENTS_VERSION,
    scale_expert_continuous,
    validate_installed_contract,
)


class ActionScaleTests(unittest.TestCase):
    def test_environment_targets_round_trip_through_pinned_action_transform(self):
        expert = torch.tensor(
            [[-1.0, -0.2], [0.0, 0.2], [1.0, 1.0]],
            dtype=torch.float32,
        )
        original = expert.clone()

        raw = scale_expert_continuous(expert)
        environment = torch.clamp(raw, -3.0, 3.0) / 3.0

        torch.testing.assert_close(
            raw,
            torch.tensor(
                [[-3.0, -0.6], [0.0, 0.6], [3.0, 3.0]],
                dtype=torch.float32,
            ),
        )
        torch.testing.assert_close(environment, expert)
        torch.testing.assert_close(expert, original)
        self.assertEqual(raw.dtype, expert.dtype)
        self.assertEqual(raw.device, expert.device)
        self.assertEqual(raw.shape, expert.shape)

    def test_empty_expert_tensor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            scale_expert_continuous(torch.empty((0, 2)))

    def test_installed_contract_accepts_pinned_version(self):
        validate_installed_contract(EXPECTED_MLAGENTS_VERSION)

    def test_installed_contract_rejects_other_version(self):
        with self.assertRaisesRegex(RuntimeError, "requires ML-Agents 1.1.0"):
            validate_installed_contract("0.0.0")
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
.venv-nav/bin/python -m unittest training.tests.test_m8_bc_action_scale -v
```

Expected: import failure because
`training.tools.m8_bc_action_scale_runner` does not exist. Fix only package
import mistakes until the failure names the missing production module.

- [ ] **Step 3: Implement the minimal pinned transform validation**

Create `training/tools/m8_bc_action_scale_runner.py` with:

```python
from __future__ import annotations

import sys

import mlagents.trainers
import numpy as np
from mlagents.torch_utils import torch
from mlagents.trainers.torch_entities.agent_action import AgentAction


EXPECTED_MLAGENTS_VERSION = "1.1.0"
RAW_TO_ENVIRONMENT_SCALE = 3.0
PATCH_MARKER = "[M8BCActionScale] validated=mlagents-1.1.0 scale=3.0"


def scale_expert_continuous(expert: torch.Tensor) -> torch.Tensor:
    if expert is None or expert.numel() == 0:
        raise ValueError("continuous expert tensor must be non-empty")
    if not torch.is_floating_point(expert):
        raise TypeError("continuous expert tensor must use a floating dtype")
    return expert * RAW_TO_ENVIRONMENT_SCALE


def validate_installed_contract(version: str | None = None) -> None:
    actual = mlagents.trainers.__version__ if version is None else version
    if actual != EXPECTED_MLAGENTS_VERSION:
        raise RuntimeError(
            f"M8 BC action-scale correction requires ML-Agents "
            f"{EXPECTED_MLAGENTS_VERSION}; found {actual}"
        )

    raw = torch.tensor([[-6.0, -3.0, 0.0, 3.0, 6.0]])
    environment = AgentAction(raw, None).to_action_tuple(
        clip=True
    ).continuous
    np.testing.assert_allclose(
        environment,
        np.array([[-1.0, -1.0, 0.0, 1.0, 1.0]], dtype=np.float32),
        rtol=0,
        atol=1e-7,
    )
```

- [ ] **Step 4: Run transform tests and verify GREEN**

Run the unittest command from Step 2.

Expected: four tests pass.

- [ ] **Step 5: Write failing hybrid-loss and patch tests**

Extend the test file with real ML-Agents action types:

```python
from types import SimpleNamespace

from mlagents.trainers.torch_entities.action_log_probs import ActionLogProbs
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents_envs.base_env import ActionSpec, BehaviorSpec

from training.tools.m8_bc_action_scale_runner import (
    corrected_behavioral_cloning_loss,
    install_patch,
)


class HybridLossTests(unittest.TestCase):
    def setUp(self):
        behavior_spec = BehaviorSpec([], ActionSpec(2, (2,)))
        self.module = SimpleNamespace(
            policy=SimpleNamespace(behavior_spec=behavior_spec)
        )

    def test_corrected_loss_scales_continuous_target_and_preserves_discrete_ce(self):
        selected = AgentAction(
            torch.tensor([[3.0, 0.6]], requires_grad=True),
            [torch.tensor([0])],
        )
        expert = AgentAction(
            torch.tensor([[1.0, 0.2]]),
            [torch.tensor([0])],
        )
        branch_logits = torch.tensor([[2.0, -1.0]], requires_grad=True)
        log_probs = ActionLogProbs(None, [torch.tensor([0.0])], [branch_logits])

        loss = corrected_behavioral_cloning_loss(
            self.module, selected, log_probs, expert
        )
        expected_discrete = -torch.log_softmax(branch_logits, dim=1)[0, 0]

        torch.testing.assert_close(loss, expected_discrete)
        loss.backward()
        self.assertIsNotNone(selected.continuous_tensor.grad)
        self.assertIsNotNone(branch_logits.grad)

    def test_selected_and_expert_continuous_shapes_must_match(self):
        selected = AgentAction(torch.zeros((2, 2)), [torch.zeros(2, dtype=torch.long)])
        expert = AgentAction(torch.zeros((2, 1)), [torch.zeros(2, dtype=torch.long)])
        log_probs = ActionLogProbs(
            None,
            [torch.zeros(2)],
            [torch.zeros((2, 2))],
        )
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            corrected_behavioral_cloning_loss(
                self.module, selected, log_probs, expert
            )

    def test_install_patch_is_idempotent(self):
        install_patch()
        install_patch()
```

Add a synthetic optimization test that starts raw forward at zero, performs
100 SGD updates with corrected continuous loss against environment label
`1.0`, and asserts raw forward exceeds `2.9` while
`clip(raw, -3, 3) / 3` exceeds `0.96`.

- [ ] **Step 6: Run tests and verify RED**

Run the unittest command.

Expected: failures because the corrected loss and patch installer are absent.

- [ ] **Step 7: Implement corrected hybrid loss and idempotent patch**

Add imports:

```python
import numpy as np
from mlagents.trainers.torch_entities.components.bc.module import BCModule
from mlagents.trainers.torch_entities.utils import ModelUtils
```

Implement:

```python
def corrected_behavioral_cloning_loss(
    module, selected_actions, log_probs, expert_actions
) -> torch.Tensor:
    action_spec = module.policy.behavior_spec.action_spec
    loss = torch.tensor(
        0.0,
        dtype=selected_actions.continuous_tensor.dtype,
        device=selected_actions.continuous_tensor.device,
    )
    if action_spec.continuous_size > 0:
        selected = selected_actions.continuous_tensor
        expert = expert_actions.continuous_tensor
        if selected is None or expert is None or expert.numel() == 0:
            raise ValueError("continuous BC actions must be non-empty")
        if selected.shape != expert.shape:
            raise ValueError("selected and expert continuous actions require matching shapes")
        loss = loss + torch.nn.functional.mse_loss(
            selected, scale_expert_continuous(expert)
        )

    if action_spec.discrete_size > 0:
        one_hot_expert = ModelUtils.actions_to_onehot(
            expert_actions.discrete_tensor,
            action_spec.discrete_branches,
        )
        log_prob_branches = ModelUtils.break_into_branches(
            log_probs.all_discrete_tensor,
            action_spec.discrete_branches,
        )
        loss = loss + torch.mean(
            torch.stack(
                [
                    torch.sum(
                        -torch.nn.functional.log_softmax(branch, dim=1)
                        * expert_branch,
                        dim=1,
                    )
                    for branch, expert_branch in zip(
                        log_prob_branches, one_hot_expert
                    )
                ]
            )
        )
    return loss


def install_patch() -> None:
    validate_installed_contract()
    if getattr(BCModule, "_m8_action_scale_installed", False):
        return
    BCModule._behavioral_cloning_loss = corrected_behavioral_cloning_loss
    BCModule._m8_action_scale_installed = True
```

At normal CLI entry:

```python
def run_training_cli() -> None:
    install_patch()
    print(PATCH_MARKER, flush=True)
    from mlagents.trainers.learn import main as mlagents_main
    mlagents_main()
```

Do not edit site-packages.

- [ ] **Step 8: Run tests and verify GREEN**

Run the unittest command.

Expected: all transform, loss, gradient, shape, optimization, version, and
idempotence tests pass.

- [ ] **Step 9: Commit the correction and tests**

```bash
git add training/tools/__init__.py \
  training/tools/m8_bc_action_scale_runner.py \
  training/tests/__init__.py \
  training/tests/test_m8_bc_action_scale.py
git commit -m "feat(m8): correct BC continuous action scale"
```

---

### Task 2: Add real-demo and checkpoint offline auditing

**Files:**
- Modify: `training/tools/m8_bc_action_scale_runner.py`
- Modify: `training/tests/test_m8_bc_action_scale.py`
- Create: `training/eval/m8_bc_action_scale_offline_audit.json`
- Create: `training/configs/m8_probeA_bc_push80_scale_checkpoint_diag.yaml`

**Interfaces:**
- Consumes: `M8RampPush80.demo`, `demo_to_buffer(sequence_length=64)`, the existing 99,986 and 149,951 `.pt` checkpoints, `SimpleActor`, and the pinned M8 network settings.
- Produces: `build_demo_audit(path: str, sequence_length: int = 64) -> dict`, `replay_checkpoint(path: str, demo_path: str, sequence_length: int = 64) -> dict`, an `--m8-audit` runner mode, a machine-readable JSON audit, and the two-line config variant.

- [ ] **Step 1: Write failing real-demo audit tests**

Add:

```python
from training.tools.m8_bc_action_scale_runner import build_demo_audit


class RealDemoAuditTests(unittest.TestCase):
    DEMO = "NavSim/Assets/Demonstrations/M8RampPush80.demo"

    def test_push80_demo_counts_and_actions_match_recorded_evidence(self):
        audit = build_demo_audit(self.DEMO)

        self.assertEqual(audit["processed_experiences"], 25344)
        self.assertEqual(audit["real_experiences"], 23808)
        self.assertEqual(audit["padding_experiences"], 1536)
        self.assertEqual(audit["terminal_markers"], 80)
        self.assertEqual(audit["episode_length_counts"]["23"], 1)
        self.assertEqual(
            sum(
                audit["episode_length_counts"].get(str(length), 0)
                for length in (300, 301, 302)
            ),
            79,
        )
        self.assertAlmostEqual(audit["forward_gt_0_9_fraction"], 0.996682, places=6)
        self.assertAlmostEqual(audit["zero_turn_fraction"], 0.995632, places=6)
        self.assertEqual(audit["jump_counts"], {"0": 23808})
```

- [ ] **Step 2: Run tests and verify RED**

Run the unittest command.

Expected: import failure for `build_demo_audit`.

- [ ] **Step 3: Implement exact sequence-padding and action audit**

Use `demo_to_buffer`, `BufferKey`, and NumPy. Build the real-experience mask
by treating entries after each terminal marker through the end of its
64-element block as padding:

```python
def _real_mask(done: np.ndarray, sequence_length: int) -> np.ndarray:
    mask = np.ones(done.shape[0], dtype=bool)
    for terminal in np.flatnonzero(done):
        block_end = ((terminal // sequence_length) + 1) * sequence_length
        mask[terminal + 1 : block_end] = False
    return mask
```

Reconstruct episode lengths from the preceding padded block boundary and
return literal JSON-safe counts and fractions. Assert exactly 80 terminal
markers; fail if a terminal block contains a second terminal marker.

- [ ] **Step 4: Run tests and verify GREEN**

Run the unittest command.

Expected: real-demo audit assertions pass along with Task 1 tests.

- [ ] **Step 5: Add checkpoint replay with a failing threshold test**

Implement checkpoint replay using:

```python
NetworkSettings(
    normalize=True,
    hidden_units=256,
    num_layers=2,
    memory=NetworkSettings.MemorySettings(
        sequence_length=64,
        memory_size=128,
    ),
)
```

Instantiate `SimpleActor` with the demonstration behavior spec, load
`checkpoint["Policy"]`, feed the entire processed buffer as 64-step sequences
with zero initial memory, and collect the continuous Gaussian mean/std plus
the discrete no-jump probability.

Add an integration test for checkpoint 149,951:

```python
result = replay_checkpoint(
    "results/m8_probeA_bc_push80_checkpoint_diag/"
    "RampAgent/RampAgent-149951.pt",
    self.DEMO,
)
self.assertLess(result["raw_environment_label_mse"], 0.01)
self.assertGreater(result["raw_forward_mean"], 0.98)
self.assertLess(result["raw_forward_mean"], 1.02)
self.assertGreater(result["raw_forward_std"], 0.8)
self.assertGreater(result["no_jump_probability"], 0.95)
self.assertGreater(result["environment_forward_mean"], 0.32)
self.assertLess(result["environment_forward_mean"], 0.34)
```

Run once before implementation and require missing-symbol RED, then implement
and require GREEN.

- [ ] **Step 6: Add audit-only CLI and write the preserved JSON**

Parse the private first argument before delegating to ML-Agents:

```text
--m8-audit
--demo <path>
--checkpoint <step>=<path>  # repeatable
--output <path>
```

Audit mode validates the installed contract, writes sorted, indented JSON
containing the demo audit and checkpoint replay objects, prints
`[M8BCActionScale] offline-audit PASS`, and exits without importing or calling
the training CLI.

Run:

```bash
.venv-nav/bin/python training/tools/m8_bc_action_scale_runner.py \
  --m8-audit \
  --demo NavSim/Assets/Demonstrations/M8RampPush80.demo \
  --checkpoint 99986=results/m8_probeA_bc_push80_checkpoint_diag/RampAgent/RampAgent-99986.pt \
  --checkpoint 149951=results/m8_probeA_bc_push80_checkpoint_diag/RampAgent/RampAgent-149951.pt \
  --output training/eval/m8_bc_action_scale_offline_audit.json
```

Require the committed-spec values, including environment forward means near
one third.

- [ ] **Step 7: Create and compare the 80k config**

Create `training/configs/m8_probeA_bc_push80_scale_checkpoint_diag.yaml` from
the Push80 config. Change exactly:

```yaml
checkpoint_interval: 40000
max_steps: 80000
```

Verify the unified diff contains those two lines and still points to
`M8RampPush80.demo`.

- [ ] **Step 8: Run the complete offline gate**

Run:

```bash
.venv-nav/bin/python -m unittest training.tests.test_m8_bc_action_scale -v
```

Then rerun audit mode and compare its SHA-256 before and after the second run.
Expected: byte-identical JSON.

- [ ] **Step 9: Commit audit and config**

```bash
git add training/tools/m8_bc_action_scale_runner.py \
  training/tests/test_m8_bc_action_scale.py \
  training/eval/m8_bc_action_scale_offline_audit.json \
  training/configs/m8_probeA_bc_push80_scale_checkpoint_diag.yaml
git commit -m "feat(m8): add BC action-scale offline audit"
```

---

### Task 3: Run the single corrected 80k discriminator

**Files:**
- Create: `results/m8_probeA_bc_push80_scale_checkpoint_diag/`
- Build/log evidence only: `/tmp/m8-bc-action-scale-*`

**Interfaces:**
- Consumes: the tested runner, two-line config, validated Push80 demo, canonical solo player, and ports 5024–5027.
- Produces: one terminal 80k result with checkpoints near 40k and 80k.

- [ ] **Step 1: Capture preservation hashes and run every preflight**

Record SHA-256 for all three demos, canonical scenes, project settings,
Push80 config, prior Push80 results' eligible checkpoints, CSV, and report.
Require:

- target result directory absent;
- no trainer/recorder/solo player process;
- ports 5024–5027 free;
- runner unit/integration tests pass;
- offline audit reruns byte-identically;
- Python `demo_to_buffer` succeeds;
- complete Unity EditMode suite passes;
- M8 physics SELFTEST is true; and
- `ValidatePush80Demo` passes.

Restore Unity's incidental demonstration-meta whitespace and
`ProjectSettings.asset` define rewrites before comparing hashes. Do not train
if any gate fails.

- [ ] **Step 2: Launch exactly one corrected trainer invocation**

Run:

```bash
caffeinate -i .venv-nav/bin/python \
  training/tools/m8_bc_action_scale_runner.py \
  training/configs/m8_probeA_bc_push80_scale_checkpoint_diag.yaml \
  --run-id=m8_probeA_bc_push80_scale_checkpoint_diag \
  --seed=0 \
  --env=NavSim/Builds/M8RampSolo.app \
  --no-graphics --time-scale=20 --num-envs=4 --base-port=5024
```

Do not use `--force` or `--resume`.

- [ ] **Step 3: Verify startup once**

Require:

- marker `[M8BCActionScale] validated=mlagents-1.1.0 scale=3.0`;
- all four Unity workers connected;
- demo path is Push80;
- max steps is 80k;
- checkpoint interval is 40k;
- BC duration is 150k; and
- no unapproved config difference.

Then wait in coarse intervals for terminal completion.

- [ ] **Step 4: Establish evaluation eligibility**

Require:

- process exit `0`;
- terminal step at least 80,000;
- nonzero `Losses/Pretraining Loss`;
- checkpoints nearest 40k and 80k; and
- result configuration preserves the approved values.

Record checkpoint hashes. If any condition fails, preserve results, write a
failure report, and stop without Unity checkpoint rollout.

---

### Task 4: Apply the offline action gate, evaluate, and report

**Files:**
- Modify: `training/eval/m8_bc_action_scale_offline_audit.json`
- Create: `training/eval/m8_bc_push80_scale_checkpoint_rollout.csv`
- Create: `.superpowers/sdd/2026-08-02-m8-bc-action-scale-discriminator/report.md`
- Temporary only: Unity checkpoint evaluator source/meta and imported ONNX copies/meta

**Interfaces:**
- Consumes: eligible corrected checkpoints, `replay_checkpoint`, canonical solo scene, and the established stochastic manual-physics seam.
- Produces: checkpoint action-space evidence, ten rollout rows when offline-eligible, and one fixed stop decision.

- [ ] **Step 1: Append corrected checkpoints to the offline audit**

Run audit mode with the two original and two corrected checkpoints. Preserve
separate keys by checkpoint step/run label. Require at least one corrected
checkpoint to satisfy:

```text
environment_forward_mean >= 0.9
```

Also record raw forward/turn mean, raw scaled-target MSE, Gaussian std, and
no-jump probability. If neither corrected checkpoint passes, write the failure
report and skip Unity rollout.

- [ ] **Step 2: Recreate the bounded Unity evaluator**

Create temporary evaluator source under
`NavSim/Assets/Scripts/Editor/` and temporary ONNX copies under
`NavSim/Assets/Models/M8/`. The evaluator must:

```csharp
behavior.DeterministicInference = false;
agent.SetModel("RampAgent", model, InferenceDevice.Burst);
```

For every checkpoint and seed `0..4`:

```csharp
agent.EndEpisode();
arena.SeedLayoutRng(seed);
arena.SetLesson(0);
arena.ResetEpisode();
Physics.SyncTransforms();
```

Then step at most 3000 times:

```csharp
Academy.Instance.EnvironmentStep();
Physics.Simulate(Time.fixedDeltaTime);
arena.Tick(Time.fixedDeltaTime);
```

Write:

```text
checkpoint,seed,placed,success,steps,min_ramp_target_dist,max_agent_y
```

Use `Path.Combine(Application.dataPath,
"../../training/eval/m8_bc_push80_scale_checkpoint_rollout.csv")`; add a
focused path assertion before running so the earlier one-segment output bug
cannot recur.

- [ ] **Step 3: Run exactly five hard-start episodes per checkpoint**

Invoke the temporary batch entrypoint once. Require exactly ten data rows,
checkpoint identifiers matching the eligible artifacts, seeds `0..4` exactly
once per checkpoint, and every row capped at 3000 steps.

- [ ] **Step 4: Apply the fixed decision**

- At least 3/5 placements at either checkpoint: scaling is the primary BC
  blocker; stop and recommend a maintainability design.
- One or two placements: scaling is causal but insufficient; stop and
  investigate residual Gaussian variance.
- Zero placement with offline environment forward mean at least 0.9: another
  control seam remains; stop and recommend action-persistence/contact capture.
- Offline gate failure: stop and debug patch/export propagation.

Goal success does not change the placement-first decision. Do not launch the
recommended follow-up or promote a model.

- [ ] **Step 5: Remove only temporary evaluation artifacts**

Delete the exact temporary evaluator source/meta and imported ONNX copies/meta.
Confirm the accidental `NavSim/training/` tree is absent, prior artifacts
remain, and no process or port remains.

- [ ] **Step 6: Write the local report**

The report must contain:

- root-cause evidence and pinned transform;
- correction implementation and test evidence;
- demo/action/sequence audit including the single 23-decision startup partial;
- exact trainer command, marker, exit status, time, and terminal step;
- pretraining loss and easy-curriculum telemetry;
- checkpoint hashes and offline action metrics;
- ten rollout rows summarized by checkpoint when eligible;
- fixed decision and explicit hard stops;
- final Unity, Python, physics, demo, process, port, hash, and diff checks.

- [ ] **Step 7: Run fresh final verification**

Run:

1. standard-library unittest suite;
2. deterministic offline audit regeneration;
3. complete Unity EditMode suite;
4. physics SELFTEST;
5. Push80 Unity validation;
6. Python demo loader;
7. protected hash comparison;
8. process/port audit;
9. `git diff --check`; and
10. `git status --short --branch`.

Restore only Unity-generated metadata/project-setting rewrites. Leave
`docs/research/` untouched.

- [ ] **Step 8: Commit authored evidence**

```bash
git add training/eval/m8_bc_action_scale_offline_audit.json \
  training/eval/m8_bc_push80_scale_checkpoint_rollout.csv
git add -f .superpowers/sdd/2026-08-02-m8-bc-action-scale-discriminator/report.md
git commit -m "feat(m8): evaluate BC action-scale discriminator"
```

Do not commit ignored `results/` unless repository precedent changes. Push
`main` only after the user explicitly requests it.
