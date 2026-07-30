# M8 Hard-Start Demonstration Discriminator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record an isolated 80-episode all-5-unit expert demonstration and use it in one 160k BC checkpoint discriminator.

**Architecture:** Extend the existing recorder with an explicit hard-recording mode while preserving its mixed and dry-run modes. Validate the new demonstration independently, point a copied diagnostic YAML at it, and reuse the preceding hard-start rollout seam for a two-checkpoint decision.

**Tech Stack:** Unity 6000.5.3f1, C#, Unity ML-Agents demonstration APIs, NUnit EditMode tests, ML-Agents Python trainer, YAML, ONNX/Sentis checkpoint inference.

## Global Constraints

- Preserve `NavSim/Assets/Demonstrations/M8RampSoloExpert.demo` and every existing result.
- Do not modify geometry, physics, rewards, observations, actions, expert steering, trainer hyperparameters, curriculum, seed, environment count, ports, player, or time scale.
- Do not modify or stage the unrelated untracked `docs/research/`.
- Do not start Probe B, S1/S2, the real batch, or a 600k run.
- Use the ML-Agents-compatible demo name `M8RampHard80` (within its 16-character
  metadata limit) and run names containing `m8_probeA_bc_hard80_checkpoint_diag`.
- Run one bounded 160k trainer invocation only after every preflight gate passes.
- Stop after reporting the discriminator decision; do not promote a model.

---

### Task 1: Add a fail-closed all-hard recording mode

**Files:**
- Modify: `NavSim/Assets/Scripts/Runtime/RampExpertLogic.cs`
- Modify: `NavSim/Assets/Scripts/Runtime/M8RampRecordingController.cs`
- Modify: `NavSim/Assets/Scripts/Tests/EditMode/RampExpertLogicTests.cs`
- Modify: `NavSim/Assets/Scripts/Tests/EditMode/M8RampRecordingControllerTests.cs`

**Interfaces:**
- Consumes: the current mixed `--m8-mode=record` and grouped `--m8-mode=dry-run` behavior.
- Produces: `--m8-mode=record-hard80`, `RampExpertLogic.HardStartDistance`, and an 80-episode terminal contract named `M8RampHard80`.

- [ ] **Step 1: Write failing schedule and metadata tests**

Add tests that express the new contract without weakening existing assertions:

```csharp
[TestCase(0)]
[TestCase(1)]
[TestCase(79)]
[TestCase(80)]
public void HardStartDistance_IsAlwaysFiveUnits(int episode)
    => Assert.AreEqual(5f, RampExpertLogic.HardStartDistance(episode), 1e-5f);
```

Rename the existing metadata test to state the mixed-demo contract, then
extract its reflection setup into a helper and add:

```csharp
[Test]
public void RecorderClose_AfterEightyTerminalEpisodesPreservesMetadataCount()
{
    var fixture = CreateWriterFixture("M8RampHard80", 80);
    Assert.IsTrue(PrepareClose(fixture.Writer, 80));
    fixture.Close();
    Assert.AreEqual(80, fixture.EpisodeCount);
}
```

Retain the 40-episode test.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
/Applications/Unity/Hub/Editor/6000.5.3f1/Unity.app/Contents/MacOS/Unity \
  -batchmode -nographics -projectPath NavSim \
  -runTests -testPlatform EditMode \
  -testFilter NavSim.Tests.EditMode.RampExpertLogicTests \
  -testResults /tmp/m8-hard80-task1-red.xml \
  -logFile /tmp/m8-hard80-task1-red.log
```

Expected: nonzero result because `HardStartDistance` does not exist. If Unity
fails before compilation for licensing or a live project lock, stop and report
the environmental block; do not kill an interactive editor.

- [ ] **Step 3: Implement the minimal explicit mode contract**

In `RampExpertLogic`, add:

```csharp
public static float HardStartDistance(int episodeIndex)
{
    _ = episodeIndex;
    return 5f;
}
```

Do not add a second schedule array.

In `M8RampRecordingController`, replace the two recording booleans with an
explicit private mode enum:

```csharp
private enum RecordingMode { Invalid, DryRun, Mixed40, Hard80 }

private const int MixedEpisodeCount = 40;
private const int HardEpisodeCount = 80;
private const string MixedDemoName = "M8RampSoloExpert";
private const string HardDemoName = "M8RampHard80";
```

Parse:

```text
--m8-mode=dry-run
--m8-mode=record
--m8-mode=record-hard80
```

Centralize the varying values in pure private helpers:

```csharp
private static bool IsRecording(RecordingMode mode);
private static int RequiredEpisodeCount(RecordingMode mode);
private static string DemonstrationName(RecordingMode mode);
private static float StartDistance(RecordingMode mode, int episodeIndex);
```

Contracts:

- `DryRun`: existing grouped ten-attempt rungs and 9/10 threshold;
- `Mixed40`: existing interleaved 40-episode schedule and name;
- `Hard80`: always 5 units, exactly 80 successful episodes, hard demo name.

For every recording episode, require both `previousSuccess` and
`_arena.RampAtTarget` before incrementing `recordedEpisodes`. Include the chosen
start distance in the JSON report as an `episodeStartDistances` array or list,
and verify every Hard80 entry is exactly `5f` before closing the recorder.
Continue using `PrepareTerminalWriterForClose(writer, recordedEpisodes)` so
ML-Agents close semantics preserve the exact metadata count.

- [ ] **Step 4: Run focused and complete EditMode tests**

Run the two focused fixtures, then the complete suite:

```bash
/Applications/Unity/Hub/Editor/6000.5.3f1/Unity.app/Contents/MacOS/Unity \
  -batchmode -nographics -projectPath NavSim \
  -runTests -testPlatform EditMode \
  -testFilter "NavSim.Tests.EditMode.RampExpertLogicTests;NavSim.Tests.EditMode.M8RampRecordingControllerTests" \
  -testResults /tmp/m8-hard80-task1-focused.xml \
  -logFile /tmp/m8-hard80-task1-focused.log

/Applications/Unity/Hub/Editor/6000.5.3f1/Unity.app/Contents/MacOS/Unity \
  -batchmode -nographics -projectPath NavSim \
  -runTests -testPlatform EditMode \
  -testResults /tmp/m8-hard80-task1-full.xml \
  -logFile /tmp/m8-hard80-task1-full.log
```

Expected: exit `0`, no failed tests. Restore any incidental
`ProjectSettings.asset` scripting-define mutation before continuing.

- [ ] **Step 5: Commit the recorder contract**

```bash
git add NavSim/Assets/Scripts/Runtime/RampExpertLogic.cs \
  NavSim/Assets/Scripts/Runtime/M8RampRecordingController.cs \
  NavSim/Assets/Scripts/Tests/EditMode/RampExpertLogicTests.cs \
  NavSim/Assets/Scripts/Tests/EditMode/M8RampRecordingControllerTests.cs
git commit -m "feat(m8): add all-hard expert recording mode"
```

---

### Task 2: Add independent demo validation and diagnostic configuration

**Files:**
- Modify: `NavSim/Assets/Scripts/Editor/M8RampDemonstrationSetup.cs`
- Create: `training/configs/m8_probeA_bc_hard80_checkpoint_diag.yaml`

**Interfaces:**
- Consumes: the existing recording scene/player and `ValidateDemo` implementation.
- Produces: `M8RampDemonstrationSetup.ValidateHard80Demo` and a trainer config whose only semantic change from the preceding discriminator is `demo_path`.

- [ ] **Step 1: Generalize validation without changing the mixed validator**

Extract the body of `ValidateDemo` into:

```csharp
private static void ValidateDemoAtPath(
    string path, string expectedName, int expectedEpisodes)
```

Keep the existing public entrypoint:

```csharp
public static void ValidateDemo() =>
    ValidateDemoAtPath(
        "Assets/Demonstrations/M8RampSoloExpert.demo",
        "M8RampSoloExpert",
        40);
```

Add:

```csharp
public static void ValidateHard80Demo() =>
    ValidateDemoAtPath(
        "Assets/Demonstrations/M8RampHard80.demo",
        "M8RampHard80",
        80);
```

Both paths must validate imported name, exact episode count, positive step
count, continuous action count `2`, discrete branches `[2]`, behavior name
`RampAgent`, vector observation size `6`, and the full sensor-shape multiset
against `Ramp_solo`.

- [ ] **Step 2: Create the copied checkpoint config**

Copy `training/configs/m8_probeA_bc_checkpoint_diag.yaml` to
`training/configs/m8_probeA_bc_hard80_checkpoint_diag.yaml` and change exactly:

```yaml
demo_path: NavSim/Assets/Demonstrations/M8RampHard80.demo
```

Verify with:

```bash
diff -u training/configs/m8_probeA_bc_checkpoint_diag.yaml \
  training/configs/m8_probeA_bc_hard80_checkpoint_diag.yaml
```

Expected: one changed line only.

- [ ] **Step 3: Compile and validate the unchanged mixed demo**

Run the complete EditMode suite and:

```bash
/Applications/Unity/Hub/Editor/6000.5.3f1/Unity.app/Contents/MacOS/Unity \
  -batchmode -nographics -projectPath NavSim \
  -executeMethod M8RampDemonstrationSetup.ValidateDemo \
  -logFile /tmp/m8-hard80-existing-demo-validation.log
```

Expected: both exit `0`; existing demo reports 40 episodes.

- [ ] **Step 4: Commit validator and config**

```bash
git add NavSim/Assets/Scripts/Editor/M8RampDemonstrationSetup.cs \
  training/configs/m8_probeA_bc_hard80_checkpoint_diag.yaml
git commit -m "feat(m8): register hard80 BC discriminator"
```

---

### Task 3: Record and preflight the Hard80 artifact

**Files:**
- Create: `NavSim/Assets/Demonstrations/M8RampHard80.demo`
- Create: `NavSim/Assets/Demonstrations/M8RampHard80.demo.meta`
- Regenerate only if source changed: `NavSim/Assets/Scenes/Ramp_recording.unity`
- Build output only: `NavSim/Builds/M8RampRecorder.app`

**Interfaces:**
- Consumes: `--m8-mode=record-hard80` and `ValidateHard80Demo`.
- Produces: one validated 80-episode all-hard demonstration usable by Python ML-Agents.

- [ ] **Step 1: Capture preservation hashes and prepare isolated evidence**

Record SHA-256 hashes for the existing mixed demo, canonical scenes, config,
and project settings. Create a temporary evidence directory and ensure it
contains no `.demo` file:

```bash
M8_HARD80_EVIDENCE_DIR=$(mktemp -d /tmp/m8-hard80.XXXXXX)
find "$M8_HARD80_EVIDENCE_DIR" -name '*.demo' -print -quit
```

Expected: the `find` command prints nothing.

- [ ] **Step 2: Rebuild the recording scene and player**

Run:

```bash
/Applications/Unity/Hub/Editor/6000.5.3f1/Unity.app/Contents/MacOS/Unity \
  -batchmode -nographics -projectPath NavSim \
  -executeMethod M8RampDemonstrationSetup.BuildScene \
  -logFile /tmp/m8-hard80-build-scene.log

/Applications/Unity/Hub/Editor/6000.5.3f1/Unity.app/Contents/MacOS/Unity \
  -batchmode -nographics -projectPath NavSim \
  -executeMethod M8RampDemonstrationSetup.BuildPlayer \
  -logFile /tmp/m8-hard80-build-player.log
```

Expected: both exit `0` with their `PASS` markers.

- [ ] **Step 3: Run one bounded hard-start dry gate**

Use the recorder player with `--m8-mode=record-hard80` but direct output to the
temporary evidence directory. This is the real bounded recording: if any
episode fails, the controller exits `2` and the artifact is rejected.

```bash
M8_DEMO_DIR="$M8_HARD80_EVIDENCE_DIR" \
M8_RECORD_REPORT="$M8_HARD80_EVIDENCE_DIR/record-hard80.json" \
NavSim/Builds/M8RampRecorder.app/Contents/MacOS/NavSim \
  --m8-mode=record-hard80 -batchmode -nographics \
  -logFile "$M8_HARD80_EVIDENCE_DIR/record-hard80.log"
```

Expected: exit `0`; JSON says mode `record-hard80`, completed `true`,
recordedEpisodes `80`, eighty start distances of `5.0`, and 80/80 placement
plus goal success. Exactly one unsuffixed
`M8RampHard80.demo` must exist.

- [ ] **Step 4: Install and validate the new artifact**

Copy the candidate and its generated metadata into
`NavSim/Assets/Demonstrations/`, then run:

```bash
/Applications/Unity/Hub/Editor/6000.5.3f1/Unity.app/Contents/MacOS/Unity \
  -batchmode -nographics -projectPath NavSim \
  -executeMethod M8RampDemonstrationSetup.ValidateHard80Demo \
  -logFile "$M8_HARD80_EVIDENCE_DIR/validate-hard80.log"
```

Expected: name `M8RampHard80`, episodes `80`, compatible action and
observation shapes, exit `0`.

- [ ] **Step 5: Run final preflight gates**

Run:

1. complete EditMode suite;
2. `M8RampPhysicsSelftest.Run` and require `SELFTEST true`;
3. Python `mlagents.trainers.demo_loader.demo_to_buffer` against the new file;
4. YAML structural comparison proving only the demo path differs;
5. process and port checks proving no trainer/player and ports 5024–5027 free;
6. hash comparison proving the mixed demo and canonical protected files are unchanged.

Do not train if any gate fails.

- [ ] **Step 6: Commit the validated demonstration**

```bash
git add NavSim/Assets/Demonstrations/M8RampHard80.demo \
  NavSim/Assets/Demonstrations/M8RampHard80.demo.meta \
  NavSim/Assets/Scenes/Ramp_recording.unity
git commit -m "feat(m8): record hard80 ramp demonstration"
```

Stage `Ramp_recording.unity` only if its intentional serialized output changed;
otherwise omit it.

---

### Task 4: Run the single 160k discriminator and report the decision

**Files:**
- Create: `results/m8_probeA_bc_hard80_checkpoint_diag/`
- Create: `training/eval/m8_bc_hard80_checkpoint_rollout.csv`
- Create: `.superpowers/sdd/2026-07-29-m8-hard80-discriminator/report.md`
- Temporary only: checkpoint evaluator source and imported ONNX assets

**Interfaces:**
- Consumes: the validated Hard80 demo, copied config, solo player, and prior manual-step hard-start evaluator seam.
- Produces: two five-episode checkpoint verdicts and a stop decision.

- [ ] **Step 1: Perform the terminal launch preflight**

Require:

- `results/m8_probeA_bc_hard80_checkpoint_diag/` absent;
- ports 5024–5027 free;
- no `mlagents-learn` or M8 player process;
- the new demo loader succeeds;
- the config points only to the Hard80 artifact;
- the solo player exists and the full Unity preflight remains green.

- [ ] **Step 2: Launch exactly one bounded trainer run**

Run foreground-supervised:

```bash
caffeinate -i .venv-nav/bin/mlagents-learn \
  training/configs/m8_probeA_bc_hard80_checkpoint_diag.yaml \
  --run-id=m8_probeA_bc_hard80_checkpoint_diag \
  --seed=0 \
  --env=NavSim/Builds/M8RampSolo.app \
  --no-graphics --time-scale=20 --num-envs=4 --base-port=5024
```

Do not use `--force` or `--resume`. Monitor sparsely: inspect startup once, then
wait for terminal completion unless the process requests attention.

- [ ] **Step 3: Check run eligibility before evaluation**

Require exit `0`, terminal step at least 160000, nonzero
`Losses/Pretraining Loss`, and checkpoints nearest 100k and 150k. Record hashes
of those checkpoints. If any condition fails, stop; preserve results and write
the failure report without creating rollout evidence.

- [ ] **Step 4: Recreate and run the prior checkpoint evaluator seam**

Use the same temporary EditMode evaluator structure as the preceding
discriminator:

- load one checkpoint into the canonical solo `RampAgent`;
- set `EvalMode` and select lesson zero, which maps to the hard 5-unit endpoint;
- use stochastic Burst inference;
- call `EndEpisode()` before each rollout to reset LSTM memory;
- step via `EnvironmentStep`, `Physics.Simulate`, and `RampArena.Tick`;
- cap at 3000 physics steps;
- write the exact existing CSV columns: `checkpoint`, `seed`, `placed`,
  `success`, `steps`, `min_ramp_target_dist`, and `max_agent_y`.

Run five episodes for each eligible checkpoint and write
`training/eval/m8_bc_hard80_checkpoint_rollout.csv`. Remove the temporary
evaluator source, `.meta`, and imported ONNX copies after the CSV is preserved.

- [ ] **Step 5: Apply the fixed decision rule**

For each checkpoint, count full successes:

- at least 3/5 at either checkpoint: coverage succeeds; stop and recommend a
  separate PPO-retention design;
- placement occurs but neither passes: stop and recommend staged demos;
- zero placement at both: stop and recommend staged push-first demos.

Do not start the recommended follow-up and do not promote an ONNX file.

- [ ] **Step 6: Write the report and perform final cleanup**

The report must contain:

- exact config/run command and terminal status;
- Hard80 metadata and preservation hashes;
- BC-loss evidence and checkpoint hashes;
- ten rollout rows summarized by checkpoint;
- the fixed decision and explicit Probe B/S1/S2/real-batch stop;
- final test, physics, demo-validation, process, port, and diff checks.

Confirm only temporary evaluator/import artifacts were removed, no process
remains, ports are free, all existing demos/results remain, and
`docs/research/` is untouched.

- [ ] **Step 7: Commit authored evidence**

Do not commit the ignored `results/` directory unless repository precedent
explicitly requires it. Commit the CSV, local report, and any final authored
source/config changes:

```bash
git add training/eval/m8_bc_hard80_checkpoint_rollout.csv \
  .superpowers/sdd/2026-07-29-m8-hard80-discriminator/report.md
git commit -m "feat(m8): evaluate hard80 BC discriminator"
```

Finish with `git diff --check`, `git status --short`, and a fresh verification
summary. Do not push unless explicitly requested.
