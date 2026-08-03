# M8 Push-Only Demonstration Discriminator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record an isolated 80-episode, all-5-unit demonstration that terminates at ramp placement and use it in one 160k BC checkpoint discriminator.

**Architecture:** Add an explicit `Push80` recording mode to the existing recording-only controller. The controller owns the placement boundary and ends the expert episode exactly once while `RampArena.RampAtTarget` is still latched; canonical arena, training, and evaluation episode semantics remain unchanged. Validate the artifact independently, point a copied diagnostic config at it, and reuse the existing stochastic hard-start checkpoint evaluator.

**Tech Stack:** Unity 6000.5.3f1, C#, Unity ML-Agents demonstration APIs, NUnit EditMode tests, ML-Agents Python trainer, YAML, ONNX/Sentis checkpoint inference.

## Global Constraints

- Preserve `NavSim/Assets/Demonstrations/M8RampSoloExpert.demo`, `NavSim/Assets/Demonstrations/M8RampHard80.demo`, their metadata, and every existing result.
- Do not modify geometry, physics, rewards, observations, actions, expert push steering, trainer hyperparameters, curriculum, seed, environment count, ports, player, time scale, or canonical success requirements.
- Restrict the placement-only terminal boundary to the recording player and the explicit `--m8-mode=record-push80` mode.
- Do not modify or stage the unrelated untracked `docs/research/`.
- Do not start Probe B, S1/S2, the real batch, a combined-demonstration run, or a 600k run.
- Use the ML-Agents-compatible demonstration name `M8RampPush80` and run ID `m8_probeA_bc_push80_checkpoint_diag`.
- Run one bounded 160k trainer invocation only after every preflight gate passes.
- Stage 1 passes if either evaluated checkpoint records at least 3/5 ramp placements; goal completion is evidence only.
- Stop after reporting the fixed discriminator decision; do not promote a model or launch the recommended follow-up.

---

### Task 1: Add a fail-closed push-only recording boundary

**Files:**
- Modify: `NavSim/Assets/Scripts/Runtime/RampExpertLogic.cs`
- Modify: `NavSim/Assets/Scripts/Runtime/M8RampRecordingController.cs`
- Modify: `NavSim/Assets/Scripts/Tests/EditMode/RampExpertLogicTests.cs`
- Modify: `NavSim/Assets/Scripts/Tests/EditMode/M8RampRecordingControllerTests.cs`

**Interfaces:**
- Consumes: existing `DryRun`, `Mixed40`, and `Hard80` modes; `RampExpertLogic.HardStartDistance(int)`; `RampArena.RampAtTarget`; the recording scene's single `M8RampExpertAgent`.
- Produces: `--m8-mode=record-push80`, `RecordingMode.Push80`, `RampExpertLogic.PushDemonstrationName`, `ShouldIssuePlacementBoundary(bool, bool, bool)`, and an 80-terminal-episode `M8RampPush80` contract.

- [ ] **Step 1: Write failing mode, identity, schedule, boundary, and terminal-count tests**

Add a recorder-name boundary test:

```csharp
[Test]
public void PushDemonstrationName_SurvivesRecorderSanitization()
{
    MethodInfo sanitize = typeof(DemonstrationRecorder).GetMethod(
        "SanitizeName", BindingFlags.Static | BindingFlags.NonPublic);

    Assert.NotNull(sanitize);
    string sanitized = (string)sanitize.Invoke(
        null, new object[] { RampExpertLogic.PushDemonstrationName, 16 });
    Assert.AreEqual("M8RampPush80", sanitized);
    Assert.AreNotEqual(RampExpertLogic.HardDemonstrationName, sanitized);
}
```

In `M8RampRecordingControllerTests`, add reflection helpers for private static
methods and assert the explicit mode contract:

```csharp
[TestCase("--m8-mode=record-push80", "Push80")]
public void ParseMode_RecognizesPushOnlyMode(string argument, string expected)
{
    object mode = InvokePrivateStatic("ParseMode", argument);
    Assert.AreEqual(expected, mode.ToString());
    Assert.AreEqual("record-push80", InvokePrivateStatic("ModeName", mode));
    Assert.AreEqual(80, InvokePrivateStatic("RequiredEpisodeCount", mode));
    Assert.AreEqual("M8RampPush80", InvokePrivateStatic("DemonstrationName", mode));
    Assert.AreEqual(5f, (float)InvokePrivateStatic("StartDistance", mode, 79), 1e-5f);
}

[TestCase(true, false, false, false)]
[TestCase(true, true, false, true)]
[TestCase(true, true, true, false)]
[TestCase(false, true, false, false)]
public void PlacementBoundary_IsIssuedExactlyOnce(
    bool pushOnly, bool rampAtTarget, bool alreadyIssued, bool expected)
{
    Assert.AreEqual(expected, InvokePrivateStatic(
        "ShouldIssuePlacementBoundary", pushOnly, rampAtTarget, alreadyIssued));
}
```

Extract the current terminal-writer fixture setup and retain both existing
40- and 80-episode assertions. Add a third exact-count assertion using
demonstration name `M8RampPush80` and `completedTerminalEpisodes = 80`.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
/Applications/Unity/Hub/Editor/6000.5.3f1/Unity.app/Contents/MacOS/Unity \
  -batchmode -nographics -projectPath NavSim \
  -runTests -testPlatform EditMode \
  -testFilter "NavSim.Tests.EditMode.RampExpertLogicTests;NavSim.Tests.EditMode.M8RampRecordingControllerTests" \
  -testResults /tmp/m8-push80-task1-red.xml \
  -logFile /tmp/m8-push80-task1-red.log
```

Expected: nonzero result because `PushDemonstrationName`, `Push80`, and
`ShouldIssuePlacementBoundary` do not exist. If Unity is blocked by licensing
or a live project lock, stop without killing an interactive editor.

- [ ] **Step 3: Implement the explicit mode and one-shot recording boundary**

In `RampExpertLogic`, add:

```csharp
public const string PushDemonstrationName = "M8RampPush80";
```

In `M8RampRecordingController`:

```csharp
private enum RecordingMode { Invalid, DryRun, Mixed40, Hard80, Push80 }
private const int PushEpisodeCount = 80;
private bool _placementBoundaryIssued;
```

Extend parsing and identity helpers:

```csharp
"--m8-mode=record-push80" => RecordingMode.Push80,
```

```csharp
RecordingMode.Push80 => "record-push80",
```

Make `IsRecording` include `Push80`; make `RequiredEpisodeCount` return `80`
for `Hard80` and `Push80`; make `DemonstrationName` return
`RampExpertLogic.PushDemonstrationName` for `Push80`; and make
`StartDistance` return `RampExpertLogic.HardStartDistance(episodeIndex)` for
both all-hard modes.

Add the pure predicate:

```csharp
private static bool ShouldIssuePlacementBoundary(
    bool pushOnly, bool rampAtTarget, bool alreadyIssued) =>
    pushOnly && rampAtTarget && !alreadyIssued;
```

In `FixedUpdate`, resolve the single expert and issue one recording-only
boundary:

```csharp
private void FixedUpdate()
{
    if (!ShouldIssuePlacementBoundary(
            _mode == RecordingMode.Push80,
            _arena != null && _arena.RampAtTarget,
            _placementBoundaryIssued))
        return;

    M8RampExpertAgent expert = _arena.Agents
        .OfType<M8RampExpertAgent>()
        .SingleOrDefault();
    if (expert == null)
    {
        Fail("push-only recording expert missing");
        return;
    }

    if (_arena.Success)
    {
        Fail("push-only episode reached goal before placement boundary");
        return;
    }

    _placementBoundaryIssued = true;
    expert.EndEpisode();
    _arena.ResetEpisode();
}
```

At the start of each new episode in `HandleEpisodeBegin`, capture the prior
placement and goal latches before any reset-side effects, then clear
`_placementBoundaryIssued` only after counting the completed terminal episode.
For `Push80`, accept the terminal only when placement is true and goal is
false. For `Mixed40` and `Hard80`, retain the existing placement-plus-goal
requirement. Count `recordedEpisodes` exactly once, require all 80 Push80 start
distances to be 5 units, and preserve `PrepareTerminalWriterForClose`.

Update the invalid-mode error to list `record-push80`. Do not edit
`RampArena`, `RampAgent`, scene geometry, rewards, observations, or expert
action logic.

- [ ] **Step 4: Run focused and complete EditMode tests**

Run the focused fixtures, then:

```bash
/Applications/Unity/Hub/Editor/6000.5.3f1/Unity.app/Contents/MacOS/Unity \
  -batchmode -nographics -projectPath NavSim \
  -runTests -testPlatform EditMode \
  -testResults /tmp/m8-push80-task1-full.xml \
  -logFile /tmp/m8-push80-task1-full.log
```

Expected: exit `0`, no failed tests, and every pre-existing mixed/Hard80 test
still green. Restore any incidental `ProjectSettings.asset` mutation before
continuing.

- [ ] **Step 5: Commit the recording contract**

```bash
git add NavSim/Assets/Scripts/Runtime/RampExpertLogic.cs \
  NavSim/Assets/Scripts/Runtime/M8RampRecordingController.cs \
  NavSim/Assets/Scripts/Tests/EditMode/RampExpertLogicTests.cs \
  NavSim/Assets/Scripts/Tests/EditMode/M8RampRecordingControllerTests.cs
git commit -m "feat(m8): add push-only expert recording mode"
```

---

### Task 2: Register independent validation and the copied discriminator config

**Files:**
- Modify: `NavSim/Assets/Scripts/Editor/M8RampDemonstrationSetup.cs`
- Create: `training/configs/m8_probeA_bc_push80_checkpoint_diag.yaml`

**Interfaces:**
- Consumes: `ValidateDemoAtPath(string, string, int)`, the canonical solo shape check, and `training/configs/m8_probeA_bc_hard80_checkpoint_diag.yaml`.
- Produces: `M8RampDemonstrationSetup.ValidatePush80Demo` and a diagnostic config whose only difference from Hard80 is `demo_path`.

- [ ] **Step 1: Add the push-only validator entrypoint**

Add:

```csharp
private const string Push80Demonstration =
    "Assets/Demonstrations/M8RampPush80.demo";
```

and:

```csharp
public static void ValidatePush80Demo() =>
    ValidateDemoAtPath(
        Push80Demonstration, RampExpertLogic.PushDemonstrationName, 80);
```

Reuse `ValidateDemoAtPath` unchanged so the new artifact must have 80 episodes,
positive steps, two continuous actions, discrete branches `[2]`, behavior
`RampAgent`, vector size `6`, and the exact canonical sensor-shape multiset.

- [ ] **Step 2: Create the copied checkpoint config**

Copy `training/configs/m8_probeA_bc_hard80_checkpoint_diag.yaml` to
`training/configs/m8_probeA_bc_push80_checkpoint_diag.yaml` and change exactly:

```yaml
demo_path: NavSim/Assets/Demonstrations/M8RampPush80.demo
```

Verify:

```bash
diff -u training/configs/m8_probeA_bc_hard80_checkpoint_diag.yaml \
  training/configs/m8_probeA_bc_push80_checkpoint_diag.yaml
```

Expected: one changed line only.

- [ ] **Step 3: Compile and validate both preserved demonstrations**

Run the complete EditMode suite plus:

```bash
/Applications/Unity/Hub/Editor/6000.5.3f1/Unity.app/Contents/MacOS/Unity \
  -batchmode -nographics -projectPath NavSim \
  -executeMethod M8RampDemonstrationSetup.ValidateDemo \
  -logFile /tmp/m8-push80-mixed-demo-validation.log

/Applications/Unity/Hub/Editor/6000.5.3f1/Unity.app/Contents/MacOS/Unity \
  -batchmode -nographics -projectPath NavSim \
  -executeMethod M8RampDemonstrationSetup.ValidateHard80Demo \
  -logFile /tmp/m8-push80-hard-demo-validation.log
```

Expected: all exit `0`; existing demos remain 40 and 80 episodes respectively.

- [ ] **Step 4: Commit validator and config**

```bash
git add NavSim/Assets/Scripts/Editor/M8RampDemonstrationSetup.cs \
  training/configs/m8_probeA_bc_push80_checkpoint_diag.yaml
git commit -m "feat(m8): register push80 BC discriminator"
```

---

### Task 3: Record and preflight the Push80 artifact

**Files:**
- Create: `NavSim/Assets/Demonstrations/M8RampPush80.demo`
- Create: `NavSim/Assets/Demonstrations/M8RampPush80.demo.meta`
- Regenerate only if intentional serialized output changes: `NavSim/Assets/Scenes/Ramp_recording.unity`
- Build output only: `NavSim/Builds/M8RampRecorder.app`

**Interfaces:**
- Consumes: `--m8-mode=record-push80`, `M8_DEMO_DIR`, `M8_RECORD_REPORT`, and `ValidatePush80Demo`.
- Produces: one validated 80-episode, all-hard, placement-terminal demonstration loadable by Python ML-Agents.

- [ ] **Step 1: Capture preservation hashes and create isolated evidence**

Record SHA-256 hashes for both existing demos, `Ramp.unity`,
`Ramp_solo.unity`, the Hard80 config, and `ProjectSettings.asset`. Create a
temporary evidence directory with `mktemp -d /tmp/m8-push80.XXXXXX` and verify
that it initially contains no `.demo` file.

- [ ] **Step 2: Rebuild the recording scene and recorder player**

Run:

```bash
/Applications/Unity/Hub/Editor/6000.5.3f1/Unity.app/Contents/MacOS/Unity \
  -batchmode -nographics -projectPath NavSim \
  -executeMethod M8RampDemonstrationSetup.BuildScene \
  -logFile /tmp/m8-push80-build-scene.log

/Applications/Unity/Hub/Editor/6000.5.3f1/Unity.app/Contents/MacOS/Unity \
  -batchmode -nographics -projectPath NavSim \
  -executeMethod M8RampDemonstrationSetup.BuildPlayer \
  -logFile /tmp/m8-push80-build-player.log
```

Expected: both exit `0` with `PASS` markers, and the canonical scenes still
contain no expert, recorder, or recording controller.

- [ ] **Step 3: Record exactly 80 placement-terminal episodes**

Run one bounded recorder invocation:

```bash
M8_DEMO_DIR="$M8_PUSH80_EVIDENCE_DIR" \
M8_RECORD_REPORT="$M8_PUSH80_EVIDENCE_DIR/record-push80.json" \
NavSim/Builds/M8RampRecorder.app/Contents/MacOS/NavSim \
  --m8-mode=record-push80 -batchmode -nographics \
  -logFile "$M8_PUSH80_EVIDENCE_DIR/record-push80.log"
```

Expected: exit `0`; report mode `record-push80`, `completed=true`, exactly 80
attempts, 80 placements, zero goals, 80 recorded episodes, and eighty start
distances equal to `5.0`. Exactly one unsuffixed `M8RampPush80.demo` must
exist. Reject the candidate if any episode reaches 3000 steps, reaches the
goal, lacks placement at its terminal boundary, or produces a mismatched
count.

- [ ] **Step 4: Install and validate the artifact**

Copy the candidate into `NavSim/Assets/Demonstrations/M8RampPush80.demo`, allow
Unity to create its `.meta`, then run:

```bash
/Applications/Unity/Hub/Editor/6000.5.3f1/Unity.app/Contents/MacOS/Unity \
  -batchmode -nographics -projectPath NavSim \
  -executeMethod M8RampDemonstrationSetup.ValidatePush80Demo \
  -logFile "$M8_PUSH80_EVIDENCE_DIR/validate-push80.log"
```

Expected: name `M8RampPush80`, exactly 80 episodes, and compatible action and
observation shapes.

- [ ] **Step 5: Run all pretraining gates**

Require:

1. complete EditMode suite passes;
2. `M8RampPhysicsSelftest.Run` reports `SELFTEST true`;
3. Python `mlagents.trainers.demo_loader.demo_to_buffer` loads the new file;
4. the Push80 config differs from Hard80 only in `demo_path`;
5. no trainer/player process remains and ports 5024–5027 are free;
6. both prior demo hashes and all protected canonical hashes are unchanged;
7. the recording report contains the exact Push80 evidence; and
8. `docs/research/` remains unmodified and untracked.

Do not train if any gate fails.

- [ ] **Step 6: Commit the validated demonstration**

```bash
git add NavSim/Assets/Demonstrations/M8RampPush80.demo \
  NavSim/Assets/Demonstrations/M8RampPush80.demo.meta
git commit -m "feat(m8): record push80 ramp demonstration"
```

Stage `NavSim/Assets/Scenes/Ramp_recording.unity` only if its intentional
serialized output changed; otherwise leave it out.

---

### Task 4: Run the single 160k discriminator and report the fixed decision

**Files:**
- Create: `results/m8_probeA_bc_push80_checkpoint_diag/`
- Create: `training/eval/m8_bc_push80_checkpoint_rollout.csv`
- Create: `.superpowers/sdd/2026-08-02-m8-push80-discriminator/report.md`
- Temporary only: checkpoint evaluator source and imported ONNX assets

**Interfaces:**
- Consumes: the validated Push80 demo, copied config, canonical solo player, and the Hard80 manual-physics checkpoint evaluator seam.
- Produces: two five-episode checkpoint placement verdicts and one fixed stop decision.

- [ ] **Step 1: Perform terminal launch preflight**

Require the target result directory to be absent, ports 5024–5027 to be free,
no `mlagents-learn` or M8 player process, the Push80 demo loader to succeed,
the config to point only to `M8RampPush80.demo`, and every Task 3 gate to
remain green.

- [ ] **Step 2: Launch exactly one bounded trainer run**

Run foreground-supervised:

```bash
caffeinate -i .venv-nav/bin/mlagents-learn \
  training/configs/m8_probeA_bc_push80_checkpoint_diag.yaml \
  --run-id=m8_probeA_bc_push80_checkpoint_diag \
  --seed=0 \
  --env=NavSim/Builds/M8RampSolo.app \
  --no-graphics --time-scale=20 --num-envs=4 --base-port=5024
```

Do not use `--force` or `--resume`. Inspect startup once, then monitor sparsely
until terminal completion.

- [ ] **Step 3: Establish evaluation eligibility**

Require trainer exit `0`, terminal step at least 160000, nonzero
`Losses/Pretraining Loss`, and checkpoints nearest 100k and 150k. Record their
SHA-256 hashes. If any condition fails, preserve the result and write a failure
report without fabricating rollout evidence.

- [ ] **Step 4: Run the unchanged hard-start checkpoint evaluator seam**

Recreate the temporary evaluator used by Hard80:

- load one checkpoint into the canonical solo `RampAgent`;
- select lesson zero under `EvalMode` for the hard 5-unit endpoint;
- use stochastic Burst inference;
- call `EndEpisode()` before each rollout to reset recurrent state;
- step `EnvironmentStep`, `Physics.Simulate`, then `RampArena.Tick`;
- stop at success or 3000 physics steps; and
- write `checkpoint`, `seed`, `placed`, `success`, `steps`,
  `min_ramp_target_dist`, and `max_agent_y`.

Run five episodes per eligible checkpoint and preserve
`training/eval/m8_bc_push80_checkpoint_rollout.csv`. Remove only the temporary
evaluator source, its `.meta`, and imported ONNX copies after preserving the
CSV.

- [ ] **Step 5: Apply the placement-first decision**

- If either checkpoint has at least 3/5 placements, Stage 1 passes. Stop and
  recommend a separate short discriminator whose demo path is an isolated
  directory containing only `M8RampPush80.demo` and `M8RampHard80.demo`.
- If placement occurs in only one or two episodes and neither checkpoint
  passes, stop and recommend one narrowly strengthened placement-only
  experiment.
- If both checkpoints have zero placement, stop adding demonstrations and
  investigate the supervised recurrent/action representation seam.

Goal totals are supporting evidence only. Do not launch the selected
follow-up, promote an ONNX model, or start any hard-stopped pipeline stage.

- [ ] **Step 6: Write the report and run final verification**

The report must include:

- exact demo metadata, report totals, and preservation hashes;
- exact trainer command, exit status, terminal step, and BC-loss evidence;
- checkpoint hashes and all ten rollout results;
- the placement-first decision and explicit Probe B/S1/S2/real-batch/600k
  stop;
- final EditMode, physics, demo-validation, process, port, hash, whitespace,
  and working-tree checks.

Confirm all prior demos/results remain, no trainer/player remains, ports are
free, only temporary evaluator/import artifacts were removed, and
`docs/research/` is untouched.

- [ ] **Step 7: Commit authored evidence**

Do not commit the ignored `results/` directory unless established repository
precedent requires it. Commit:

```bash
git add training/eval/m8_bc_push80_checkpoint_rollout.csv
git add -f .superpowers/sdd/2026-08-02-m8-push80-discriminator/report.md
git commit -m "feat(m8): evaluate push80 BC discriminator"
```

Finish with `git diff --check`, `git status --short --branch`, and a fresh
verification summary. Push `main` only after the user explicitly requests it.
