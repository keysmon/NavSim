using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;  // ActionBuffers/ActionSegment (scripted Selftest step)
using Unity.MLAgents.Policies;   // BehaviorParameters, InferenceDevice, DeterministicInference
using Unity.InferenceEngine;     // ModelAsset (com.unity.ai.inference)
using NavSim.Runtime;

// SHOWCASE course-mode eval — the course-mode sibling of M6SearchEval + M6EvalBatch. Three batchmode entries share
// one Play-mode shell; each Exits its own code (NO -quit; graphics ON — the pixel CameraSensor needs a GPU or it
// feeds blank frames, so NEVER -nographics):
//   Selftest — no-model gate (cadence + ForceCourse reproducibility + Resolve + EvalMode-suppresses-reroll + import).
//     Unity -batchmode -projectPath NavSim -executeMethod ShowcaseEvalBatch.Selftest -logFile -
//   RunAB   — LSTM-carryover A/B on the representative cell (stage 2, mirror=false), as-is vs per-episode reset.
//     Unity -batchmode -projectPath NavSim -executeMethod ShowcaseEvalBatch.RunAB    -logFile -
//   Run     — full 300-episode forced grid (reset mode via env SHOWCASE_EVAL_RESET, chosen by the A/B).
//     SHOWCASE_EVAL_ONNX=Assets/Models/Showcase/showcase_s0.onnx [SHOWCASE_EVAL_RESET=1] \
//       Unity -batchmode -projectPath NavSim -executeMethod ShowcaseEvalBatch.Run    -logFile -
//
// INFERENCE MODE: DeterministicInference = true (mean action) for every model run — standard eval practice, and
// LOAD-BEARING here: it makes paired seeds produce bit-identical trajectories unless the LSTM memory differs, so the
// A/B carries ZERO sampling noise (any as-is-vs-reset difference is a real memory signal, not luck). Set BEFORE the
// (single) SetModel so GeneratePolicy reads it; the DET-CHECK (two fresh trials → identical first action) confirms it took.
//
// LSTM RESET (the review-mandated A/B lever): course-mode training ends every traversal in EndEpisode
// (EndsEpisodeOnGoal or MaxStep 3000), so the policy was trained — and the WebGL demo runs — starting each traversal
// from ZERO LSTM memory. This harness holds the boundary with MaxStep=0, so "as-is" carries memory across traversals
// (an artifact neither training nor deployment produces); "reset" restores the faithful condition. Mechanism:
// agent.EndEpisode() — NOT SetModel (which early-outs on an unchanged model, a no-op). EndEpisode queues a `done`
// AgentInfo; on the next DecideBatch GeneratorImpl does `if (info.done) m_Memories.Remove(episodeId)` then writes ZERO
// memory for the (same-episodeId) live decision — robust to whether done + live land in the same batch or sequential
// ones. OnEpisodeBegin's random placement is overwritten by the deterministic ForceCourse that follows.
//
// Idiom invariants (M5/M6-proven): Physics.simulationMode=Script for the whole run; Step()=EnvironmentStep()+
// Physics.Simulate(dt), so the ONNX policy (via the scene's DecisionRequester) acts once per step; Physics.SyncTransforms()
// after every ForceCourse; agent.MaxStep=0 + env.EvalMode=true so the HARNESS owns the boundary and outcomes are
// detected GEOMETRICALLY on VALUES captured at episode start (decoy-first, always hard). A pit is a NON-terminal delta.
public static class ShowcaseEvalBatch
{
    private const string ScenePath = "Assets/Scenes/Training_showcase.unity";
    private const string DefaultOnnx = "Assets/Models/Showcase/showcase_s0.onnx";
    private const int WarmupFrames = 40;      // Play + Academy/scene init before we seize control (P0/M6 idiom)
    private const int MaxSteps = 3000;        // per-episode academy-step budget == agent.MaxStep -> "timeout"
    private const int EpisodesPerCell = 25;   // episodes per (stage, mirror, variant) cell
    private const int SettleSteps = 20;       // scripted zero-action steps to canonical rest at episode start
    private const float ActionEps = 1e-4f;    // deterministic inference is bit-stable; a sampling diff would be O(1)

    // Forced grid: every stage on both mirror states; stage 3 ("The gap") on BOTH pit variants, the rest NoLoop.
    // -> stages 0,1,2,4: 2 mirror x 25 = 50 each; stage 3: 2 mirror x 2 variant x 25 = 100. Total 300 episodes.
    private static readonly int[] Stages = { 0, 1, 2, 3, 4 };

    private enum Mode { Grid, Selftest, AB, Debug }
    private enum Outcome { None, Success, Decoy }

    private static Mode _mode;
    private static bool _savedEpmoEnabled;
    private static EnterPlayModeOptions _savedEpmo;
    private static SimulationMode _savedSimMode;
    private static int _frames;
    private static string _onnxPath;

    private struct EpRow
    {
        public int stage; public bool mirrored; public CourseVariant variant; public int seed;
        public string outcome; public int steps; public int jumpUses; public int pitFalls;
        public Vector3 target0;   // captured red-slot position (A/B cell-integrity check across arms)
    }

    public static void Run()      { _mode = Mode.Grid;     Begin(bakeModel: true); }
    public static void Selftest() { _mode = Mode.Selftest; Begin(bakeModel: false); }
    public static void RunAB()    { _mode = Mode.AB;       Begin(bakeModel: true); }
    public static void RunDebug() { _mode = Mode.Debug;    Begin(bakeModel: true); }

    private static void Begin(bool bakeModel)
    {
        _savedEpmoEnabled = EditorSettings.enterPlayModeOptionsEnabled;
        _savedEpmo = EditorSettings.enterPlayModeOptions;
        _savedSimMode = Physics.simulationMode;
        _onnxPath = System.Environment.GetEnvironmentVariable("SHOWCASE_EVAL_ONNX");
        if (string.IsNullOrWhiteSpace(_onnxPath)) _onnxPath = DefaultOnnx;
        try
        {
            EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
            // Bake the model into BehaviorParameters in EDIT MODE (before Play) so the FIRST ModelRunner created at
            // LazyInitialize is deterministic. This sidesteps an ml-agents quirk: Academy.GetOrCreateModelRunner caches
            // runners by (model, device) ONLY — it ignores deterministicInference — so a non-deterministic runner
            // created first (e.g. by a SetModel-after-warmup path, where SetModel then early-outs on the unchanged
            // model) would pin sampling forever. Baking + BehaviorType.InferenceOnly is the M6DemoSceneSetup idiom;
            // the scene is NEVER saved (in-memory only, discarded on Exit). Selftest is no-model (bakeModel=false).
            if (bakeModel && !BakeModel()) { RestoreProjectSettings(); EditorApplication.Exit(2); return; }
            EditorSettings.enterPlayModeOptionsEnabled = true;
            EditorSettings.enterPlayModeOptions =
                EnterPlayModeOptions.DisableDomainReload | EnterPlayModeOptions.DisableSceneReload;
            _frames = 0;
            EditorApplication.update += Tick;
            EditorApplication.EnterPlaymode();
        }
        catch (System.Exception e)
        {
            RestoreProjectSettings();
            Debug.LogError("[ShowcaseEval] Begin setup FAILED: " + e);
            EditorApplication.Exit(2);
        }
    }

    private static void Tick()
    {
        if (!EditorApplication.isPlaying) return;
        _frames++;
        if (_frames < WarmupFrames) return;
        EditorApplication.update -= Tick;
        int code = 2;
        try
        {
            var env = Object.FindAnyObjectByType<NavEnvironment>();
            var agent = Object.FindAnyObjectByType<NavAgent>();
            if (env == null || agent == null) Debug.LogError("[ShowcaseEval] missing env/agent in scene");
            else if (!env.CourseMode) Debug.LogError("[ShowcaseEval] env.course not wired — scene is not in COURSE mode");
            else
                switch (_mode)
                {
                    case Mode.Selftest: code = RunSelftest(env, agent); break;
                    case Mode.AB: code = RunABImpl(env, agent); break;
                    case Mode.Debug: code = RunDebugImpl(env, agent); break;
                    default: code = RunGrid(env, agent); break;
                }
        }
        catch (System.Exception e) { Debug.LogError("[ShowcaseEval] EXCEPTION: " + e); code = 2; }
        finally { RestoreProjectSettings(); }
        EditorApplication.Exit(code);
    }

    private static void RestoreProjectSettings()
    {
        Physics.simulationMode = _savedSimMode;
        EditorSettings.enterPlayModeOptionsEnabled = _savedEpmoEnabled;
        EditorSettings.enterPlayModeOptions = _savedEpmo;
    }

    // ================= shared harness setup =================

    // Script physics, harness-owned boundary, one step to allocate the pixel CameraSensor texture before any SetModel.
    private static void SetupHarnessCommon(NavEnvironment env, NavAgent agent)
    {
        Physics.simulationMode = SimulationMode.Script;
        agent.MaxStep = 0;
        env.EvalMode = true;
        Step();
    }

    // Bake the ONNX into BehaviorParameters in EDIT MODE (called from Begin, before Play). InferenceOnly + Burst +
    // DeterministicInference=true, so LazyInitialize creates a deterministic (showcase_s0, Burst) runner FIRST — the
    // only reliable way past the (model, device)-only runner cache. Returns false on a missing bp / unimported model.
    private static bool BakeModel()
    {
        AssetDatabase.Refresh();   // ensure a freshly-copied ONNX is imported before we load it
        var agent = Object.FindAnyObjectByType<NavAgent>();
        if (agent == null) { Debug.LogError("[ShowcaseEval] BakeModel: no NavAgent in scene"); return false; }
        var bp = agent.GetComponent<BehaviorParameters>();
        if (bp == null) { Debug.LogError("[ShowcaseEval] BakeModel: agent has no BehaviorParameters"); return false; }
        var model = AssetDatabase.LoadAssetAtPath<ModelAsset>(_onnxPath);
        if (model == null)
        {
            Debug.LogError($"[ShowcaseEval] model not found/imported at '{_onnxPath}' — copy the ONNX in AND let Unity " +
                           "import it (a .onnx.meta must exist) before running; LoadAssetAtPath returns null otherwise.");
            return false;
        }
        // Inference mode is keyed off the run MODE, not a call-site env var (a forgotten var would silently make the
        // A/B stochastic and its DET-CHECK fail). STOCHASTIC (sample the action distribution) is the DEPLOYMENT-faithful
        // mode — the WebGL demo runs DeterministicInference=false, and deterministic argmax SUPPRESSES a probabilistic
        // jump (a lip jump the policy assigns p~0.3 never fires under argmax → gap stages become unpassable, an
        // artifact). Only the A/B forces deterministic (mean action) for its paired reproducibility on no-jump stage 2;
        // the grid runs stochastic. The env var can still force deterministic for ad-hoc debugging.
        bool deterministic = _mode == Mode.AB
            || ParseBool(System.Environment.GetEnvironmentVariable("SHOWCASE_EVAL_DETERMINISTIC"));
        bp.BehaviorName = "NavAgent";
        bp.Model = model;
        bp.InferenceDevice = InferenceDevice.Burst;
        bp.BehaviorType = BehaviorType.InferenceOnly;
        bp.DeterministicInference = deterministic;
        EditorUtility.SetDirty(bp);   // in-memory only; the scene is never saved
        Debug.Log($"[ShowcaseEval] baked model={_onnxPath} device=Burst InferenceOnly deterministic={deterministic}");
        return true;
    }

    // Two fresh (zero-memory) trials on the same seeded layout must yield the same action — proves inference is
    // deterministic AND that DeterministicInference actually took effect on the policy.
    private static bool AssertDeterministic(NavEnvironment env, NavAgent agent)
    {
        int s = SeedFor(2, false, CourseVariant.NoLoop, 0);
        float[] a1 = FirstAction(env, agent, 2, false, CourseVariant.NoLoop, s);
        float[] a2 = FirstAction(env, agent, 2, false, CourseVariant.NoLoop, s);
        bool ok = ActionsEqual(a1, a2);
        Debug.Log($"[ShowcaseEval] DET-CHECK a1=[{Fmt(a1)}] a2=[{Fmt(a2)}] identical={ok}");
        return ok;
    }

    // ================= per-episode driving =================

    // Seed both RNGs, force the deterministic cell, request a fresh decision on the new layout, capture outcome anchors
    // as VALUES. resetMemory: EndEpisode FIRST (zero the LSTM), then re-seed AFTER OnEpisodeBegin's random placement so
    // the seeded ForceCourse (which overwrites that placement) is stream-independent of it.
    private static (Vector3 target0, Vector3[] decoy0) SetupEpisode(
        NavEnvironment env, NavAgent agent, int stage, bool mir, CourseVariant variant, int seed, bool resetMemory)
    {
        if (resetMemory) agent.EndEpisode();
        UnityEngine.Random.InitState(seed);
        env.SeedColorRng(seed);
        env.ForceCourse(stage, mir, variant);   // CourseBuilder.ClearPieces DestroyImmediate's the prior stage (root fix): ghost-free
        ResetLighting();           // EndEpisode's OnEpisodeBegin jitters the sun; ForceCourse(jitter:false) doesn't reset it
        Physics.SyncTransforms();
        // Settle to canonical rest: PlaceAt teleports via a direct transform write, so CharacterController.velocity is
        // stale (the previous episode's last Move) until the CC moves again. Left unsettled, the first CollectObservations
        // reads that stale velocity — nondeterministic AND a per-episode as-is/reset confound. These scripted zero-action
        // steps drive the CC (zeroing velocity, grounding) via OnActionReceived DIRECTLY, so the policy never runs and the
        // LSTM memory stays zeroed (memory only updates inside Academy DecideBatch, which these steps never call).
        for (int i = 0; i < SettleSteps; i++) ScriptedZeroStep(agent);
        agent.RequestDecision();   // both modes: first step is a fresh decision on the new layout (no stale-action confound)
        return (env.GoalPositionFor(agent), env.DecoyPositions());
    }

    private static EpRow RunEpisode(NavEnvironment env, NavAgent agent, int stage, bool mir, CourseVariant variant,
        int seed, Vector3 target0, Vector3[] decoy0)
    {
        int pit0 = env.PitFalls, jump0 = agent.JumpUses;
        string outcome = "timeout";
        int steps = MaxSteps;
        for (int s = 0; s < MaxSteps; s++)
        {
            Step();
            Outcome o = Resolve(env, agent.transform.position, target0, decoy0);
            if (o == Outcome.Decoy) { outcome = "decoy"; steps = s + 1; break; }
            if (o == Outcome.Success) { outcome = "success"; steps = s + 1; break; }
        }
        return new EpRow
        {
            stage = stage, mirrored = mir, variant = variant, seed = seed,
            outcome = outcome, steps = steps,
            jumpUses = agent.JumpUses - jump0,
            pitFalls = env.PitFalls - pit0,
            target0 = target0,
        };
    }

    private static void RunCell(NavEnvironment env, NavAgent agent, int stage, bool mir, CourseVariant variant,
        bool resetMemory, List<EpRow> rows)
    {
        for (int ep = 0; ep < EpisodesPerCell; ep++)
        {
            int seed = SeedFor(stage, mir, variant, ep);
            var (t0, d0) = SetupEpisode(env, agent, stage, mir, variant, seed, resetMemory);
            rows.Add(RunEpisode(env, agent, stage, mir, variant, seed, t0, d0));
        }
    }

    // The single source of truth for outcome resolution (RunEpisode + the Selftest both call it). Decoy FIRST
    // (pessimistic tie-break, always hard), then the target; within GoalRadius (== 1.5).
    private static Outcome Resolve(NavEnvironment env, Vector3 pos, Vector3 target0, Vector3[] decoy0)
    {
        if (Within(env, pos, decoy0[0]) || Within(env, pos, decoy0[1])) return Outcome.Decoy;
        if (Within(env, pos, target0)) return Outcome.Success;
        return Outcome.None;
    }

    private static bool Within(NavEnvironment env, Vector3 pos, Vector3 slot) =>
        Vector3.Distance(pos, slot) < env.GoalRadius;

    // First policy action on a fresh zero-memory decision at the seeded layout (validation probe readout).
    private static float[] FirstAction(NavEnvironment env, NavAgent agent, int stage, bool mir, CourseVariant variant, int seed)
    {
        SetupEpisode(env, agent, stage, mir, variant, seed, resetMemory: true);
        Step();
        ActionBuffers ab = agent.GetStoredActionBuffers();
        return new float[] { ab.ContinuousActions[0], ab.ContinuousActions[1], ab.DiscreteActions[0] };
    }

    private static int SeedFor(int stage, bool mir, CourseVariant variant, int ep) =>
        stage * 100000 + (mir ? 1 : 0) * 10000 + (int)variant * 1000 + ep;

    private static void Step()
    {
        Academy.Instance.EnvironmentStep();
        Physics.Simulate(Time.fixedDeltaTime);
    }

    // A scripted zero-action physics step: drives the CC via OnActionReceived DIRECTLY (no Academy step, so no policy
    // decision and no LSTM memory update) — used only to settle the agent to rest at episode start. Zero forward means
    // zero horizontal CC velocity; a few steps ground it, giving a canonical, reproducible episode-start observation.
    private static void ScriptedZeroStep(NavAgent agent)
    {
        agent.OnActionReceived(new ActionBuffers(
            new ActionSegment<float>(new float[] { 0f, 0f }), new ActionSegment<int>(new int[] { 0 })));
        Physics.Simulate(Time.fixedDeltaTime);
    }

    // ================= GRID (full 300-episode forced grid) =================

    private static int RunGrid(NavEnvironment env, NavAgent agent)
    {
        SetupHarnessCommon(env, agent);   // model baked in Begin: InferenceOnly + STOCHASTIC (deployment-faithful).
        // No determinism assert here — the grid is stochastic BY DESIGN (deterministic argmax suppresses the gap jump).
        bool reset = ParseBool(System.Environment.GetEnvironmentVariable("SHOWCASE_EVAL_RESET"));
        Debug.Log($"[ShowcaseEval] GRID reset-memory-per-episode={reset} inference=STOCHASTIC (single draw) " +
                  "(reset + stochastic = training/deployment-faithful; deterministic argmax would suppress the gap jump)");

        var rows = new List<EpRow>(300);
        foreach (int stage in Stages)
            foreach (bool mir in new[] { false, true })
            {
                if (stage == 3)
                {
                    RunCell(env, agent, 3, mir, CourseVariant.SafeLoop, reset, rows);
                    RunCell(env, agent, 3, mir, CourseVariant.NoLoop, reset, rows);
                }
                else RunCell(env, agent, stage, mir, CourseVariant.NoLoop, reset, rows);
            }

        WriteGridCsv(rows, reset);
        LogSummary(rows, reset);
        Debug.Log($"[ShowcaseEval] GRID COMPLETE — {rows.Count} episodes (reset={reset}).");
        return 0;
    }

    // ================= DEBUG (instrumented cross-stage ghost-collider repro) =================

    // Cycles stage 0->1->2 in reset mode (like the grid), logging per rebuild the course-child + collider count
    // under the course root (growth across stages = deferred-Destroy ghost accumulation), and on the stage-2
    // episodes per 100 steps: pos, grounded, jump press, live JumpUses/pitFalls deltas, and live GoalPositionFor
    // vs captured target0 drift (nonzero = a mid-episode re-roll). Diagnostic only.
    private static int RunDebugImpl(NavEnvironment env, NavAgent agent)
    {
        SetupHarnessCommon(env, agent);
        var cc = agent.GetComponent<CharacterController>();
        // Cover all stages incl. the mandatory-jump gaps (3 NoLoop, 4), with repeats so a CONSTANT collider count
        // proves the ghost fix and a gap-stage success with jumpUses>=1 rules out a broken jump counter.
        int[] cyc = { 0, 1, 2, 3, 4, 2, 3, 4 };
        int epIdx = 0;
        foreach (int stage in cyc)
        {
            int seed = SeedFor(stage, false, CourseVariant.NoLoop, epIdx);
            var (t0, d0) = SetupEpisode(env, agent, stage, false, CourseVariant.NoLoop, seed, resetMemory: true);
            int children = env.Course.transform.childCount;
            int colliders = env.Course.transform.GetComponentsInChildren<Collider>(true).Length;
            Debug.Log($"[DBG] epIdx={epIdx} stage={stage} courseChildren={children} colliders={colliders} target0={Fmt3(t0)}");
            int pit0 = env.PitFalls, jump0 = agent.JumpUses;
            string outcome = "timeout";
            int steps = MaxSteps;
            for (int s = 0; s < MaxSteps; s++)
            {
                Step();
                if (stage >= 2 && s % 100 == 0)
                {
                    Vector3 pos = agent.transform.position;
                    Vector3 live = env.GoalPositionFor(agent);
                    var da = agent.GetStoredActionBuffers().DiscreteActions;
                    int jp = da.Length > 0 ? da[0] : -1;
                    Debug.Log($"[DBG]   stage={stage} step={s} pos={Fmt3(pos)} grounded={cc.isGrounded} jumpPress={jp} " +
                              $"jumpUses={agent.JumpUses - jump0} pitFalls={env.PitFalls - pit0} driftLiveVsT0={(live - t0).magnitude:F2}");
                }
                Outcome o = Resolve(env, agent.transform.position, t0, d0);
                if (o == Outcome.Decoy) { outcome = "decoy"; steps = s + 1; break; }
                if (o == Outcome.Success) { outcome = "success"; steps = s + 1; break; }
            }
            Debug.Log($"[DBG] epIdx={epIdx} stage={stage} OUTCOME={outcome} steps={steps} " +
                      $"jumpUses={agent.JumpUses - jump0} pitFalls={env.PitFalls - pit0}");
            epIdx++;
        }
        return 0;
    }

    private static string Fmt3(Vector3 v) =>
        string.Format(CultureInfo.InvariantCulture, "({0:F1},{1:F1},{2:F1})", v.x, v.y, v.z);

    // ================= A/B (LSTM carryover on the representative cell) =================

    private static int RunABImpl(NavEnvironment env, NavAgent agent)
    {
        SetupHarnessCommon(env, agent);   // model already baked (InferenceOnly, deterministic) in Begin
        bool detOk = AssertDeterministic(env, agent);
        if (!detOk) { Debug.LogError("[ShowcaseEval] AB abort: inference not deterministic — A/B would be noise"); return 3; }

        // RESET-CHECK: fresh vs after-priming first action. Under deterministic inference the ONLY thing that could
        // differ is the LSTM memory, so identical ⇒ EndEpisode truly zeroed it (history-independent).
        int sSeed = SeedFor(2, false, CourseVariant.NoLoop, 0);
        float[] aFresh = FirstAction(env, agent, 2, false, CourseVariant.NoLoop, sSeed);
        for (int k = 1; k <= 5; k++)   // priming episodes (each reset) build then clear memory
        {
            int pseed = SeedFor(2, false, CourseVariant.NoLoop, 100 + k);
            var (pt, pd) = SetupEpisode(env, agent, 2, false, CourseVariant.NoLoop, pseed, resetMemory: true);
            RunEpisode(env, agent, 2, false, CourseVariant.NoLoop, pseed, pt, pd);
        }
        float[] aAfter = FirstAction(env, agent, 2, false, CourseVariant.NoLoop, sSeed);
        bool resetOk = ActionsEqual(aFresh, aAfter);
        Debug.Log($"[ShowcaseEval] RESET-CHECK aFresh=[{Fmt(aFresh)}] aAfterPriming=[{Fmt(aAfter)}] identical={resetOk}");

        // A/B paired cell (stage 2, mirror=false, 25 eps). Both arms start the cell from ZERO memory (EndEpisode below),
        // so the only difference is WITHIN-cell carryover.
        agent.EndEpisode();
        var asis = new List<EpRow>(EpisodesPerCell);
        RunCell(env, agent, 2, false, CourseVariant.NoLoop, resetMemory: false, asis);
        var reset = new List<EpRow>(EpisodesPerCell);
        RunCell(env, agent, 2, false, CourseVariant.NoLoop, resetMemory: true, reset);

        // Paired per-episode diff (deterministic + paired seeds -> zero sampling noise; any diff is a real memory signal).
        int nIdentical = 0, nDiff = 0, integrityFail = 0;
        var diffs = new List<string>();
        for (int i = 0; i < asis.Count; i++)
        {
            if ((asis[i].target0 - reset[i].target0).sqrMagnitude > 1e-6f) integrityFail++; // same layout by construction
            bool same = asis[i].outcome == reset[i].outcome && asis[i].steps == reset[i].steps;
            if (same) nIdentical++;
            else { nDiff++; diffs.Add($"seed={asis[i].seed}: asis={asis[i].outcome}/{asis[i].steps} reset={reset[i].outcome}/{reset[i].steps}"); }
        }
        int succAsis = CountSuccess(asis), succReset = CountSuccess(reset);
        int delta = Mathf.Abs(succAsis - succReset);
        string gridMode = delta > 2 ? "reset" : "as-is";

        WriteAbCsv(asis, reset);

        var sb = new StringBuilder();
        sb.AppendLine("[ShowcaseEval] ===== LSTM CARRYOVER A/B (stage 2 The wall, mirror=false, 25 eps, deterministic) =====");
        sb.AppendLine($"  det-check   (deterministic inference live) : {detOk}");
        sb.AppendLine($"  reset-check (EndEpisode zeroes LSTM memory): {resetOk}");
        sb.AppendLine($"  cell integrity (target0 identical per seed): {(integrityFail == 0)} ({integrityFail} mismatch)");
        sb.AppendLine($"  success   as-is={succAsis}/25   reset={succReset}/25   |delta|={delta}");
        sb.AppendLine($"  paired per-episode: identical={nIdentical}/25  differ={nDiff}/25");
        foreach (var d in diffs) sb.AppendLine("     DIFF " + d);
        sb.AppendLine("  NOTE: n=25 at p~0.8 has INDEPENDENT-run SD ~= 2.0, but paired+deterministic collapses sampling noise to 0.");
        sb.AppendLine($"  GRID MODE by the >2/25 rule: {gridMode}  (reset is training/deployment-faithful; as-is is the MaxStep=0 artifact).");
        Debug.Log(sb.ToString());
        return 0;
    }

    // ================= no-model SELFTEST =================

    private static int RunSelftest(NavEnvironment env, NavAgent agent)
    {
        SetupHarnessCommon(env, agent);   // no InstallModel -> heuristic (zero) action; the gates below need no policy
        bool allPass = true;

        // (1) CADENCE gate: a MISSING DecisionRequester is the silent all-timeout failure mode (P0 can't catch it — it
        // bypasses the policy pipeline). Same scene as training, so the period matches training by construction.
        var dr = agent.GetComponent<DecisionRequester>();
        bool cadenceOk = dr != null;
        Debug.Log($"[ShowcaseEval] SELFTEST cadence: DecisionRequester={(dr != null ? "present" : "MISSING")} " +
                  $"period={(dr != null ? dr.DecisionPeriod.ToString() : "n/a")} " +
                  $"takeActionsBetween={(dr != null ? dr.TakeActionsBetweenDecisions.ToString() : "n/a")}");
        allPass &= cadenceOk;

        // (2) same-seed ForceCourse reproducibility: SeedColorRng drives which fixed slot is the red target, so the
        // captured target0/decoy0 must reproduce for a seed and (almost always) vary across seeds.
        var (t1, d1) = PlaceSeeded(env, agent, 2, false, CourseVariant.NoLoop, 123);
        var (t2, d2) = PlaceSeeded(env, agent, 2, false, CourseVariant.NoLoop, 123);
        bool repro = Approx(t1, t2) && Approx(d1[0], d2[0]) && Approx(d1[1], d2[1]);
        var (t3, _) = PlaceSeeded(env, agent, 2, false, CourseVariant.NoLoop, 999);
        bool varies = !Approx(t1, t3);
        allPass &= repro && varies;

        // (3) Resolve correctness (the SAME function RunEpisode uses): on-target -> Success, on-decoy -> Decoy, separated.
        var (tgt, dcy) = PlaceSeeded(env, agent, 2, false, CourseVariant.NoLoop, 55);
        bool onTarget = Resolve(env, tgt, tgt, dcy) == Outcome.Success;
        bool onDecoy = Resolve(env, dcy[0], tgt, dcy) == Outcome.Decoy;
        bool separated = !Within(env, tgt, dcy[0]) && !Within(env, tgt, dcy[1]);
        bool resolveOk = onTarget && onDecoy && separated;
        allPass &= resolveOk;

        // (4) EvalMode-suppresses-reroll: move a decoy onto the agent, drive ONE scripted step (OnActionReceived
        // directly, P0 idiom — no policy), assert the decoy fired AND the target did NOT re-roll (no EndEpisode).
        var (tgtL, dcyL) = PlaceSeeded(env, agent, 2, false, CourseVariant.NoLoop, 77);
        Vector3 targetBefore = tgtL;
        GameObject decoyGo = FindGoalAt(dcyL[0]);
        bool ran = decoyGo != null, touched = false, noReroll = false;
        if (ran)
        {
            decoyGo.transform.position = agent.transform.position + agent.transform.forward * 1.0f; // within GoalRadius
            Physics.SyncTransforms();
            agent.OnActionReceived(new ActionBuffers(
                new ActionSegment<float>(new float[] { 0f, 0f }), new ActionSegment<int>(new int[] { 0 })));
            touched = env.TouchedDecoy(agent);
            noReroll = Approx(env.GoalPositionFor(agent), targetBefore);
        }
        bool evalModeOk = !ran || (touched && noReroll);
        allPass &= evalModeOk;

        // (5) IMPORT gate (team-lead): opening the project imports the staged ONNX; assert it's loadable + has a .meta,
        // so the downstream A/B / grid never discover a null model.
        AssetDatabase.Refresh();
        var model = AssetDatabase.LoadAssetAtPath<ModelAsset>(_onnxPath);
        string metaPath = Path.GetFullPath(Path.Combine(Application.dataPath, "..", _onnxPath)) + ".meta";
        bool metaExists = File.Exists(metaPath);
        bool importOk = model != null && metaExists;
        allPass &= importOk;

        Debug.Log($"[ShowcaseEval] SELFTEST — cadence={cadenceOk} reproducible={repro} varies={varies} " +
                  $"resolve(onTarget={onTarget},onDecoy={onDecoy},separated={separated})={resolveOk} " +
                  $"evalMode-suppresses-reroll(ran={ran},touched={touched},noReroll={noReroll})={evalModeOk} " +
                  $"import(model={(model != null)},meta={metaExists})={importOk} => {(allPass ? "ALL PASS" : "FAIL")}");
        return allPass ? 0 : 1;
    }

    private static (Vector3, Vector3[]) PlaceSeeded(NavEnvironment env, NavAgent agent, int stage, bool mir, CourseVariant v, int seed)
    {
        UnityEngine.Random.InitState(seed);
        env.SeedColorRng(seed);
        env.ForceCourse(stage, mir, v);   // CourseBuilder.ClearPieces DestroyImmediate's the prior stage (root fix): ghost-free
        ResetLighting();
        Physics.SyncTransforms();
        return (env.GoalPositionFor(agent), env.DecoyPositions());
    }

    // Deterministic sun: EndEpisode's OnEpisodeBegin path runs a jittered SetCourseStage (JitterLighting randomizes the
    // directional light) that the subsequent deterministic ForceCourse does NOT reset — so a pixel policy would see
    // different lighting per episode (nondeterministic obs, and a reset-vs-as-is confound). Pin the sun to
    // JitterLighting's un-jittered base so every episode/arm/trial renders under identical light.
    private static void ResetLighting()
    {
        foreach (Light l in Object.FindObjectsByType<Light>(FindObjectsSortMode.None))
            if (l != null && l.type == LightType.Directional)
            {
                l.transform.rotation = Quaternion.Euler(45f, -35f, 0f);
                l.intensity = 1f;
                return;
            }
    }

    private static GameObject FindGoalAt(Vector3 pos)
    {
        foreach (var g in GameObject.FindGameObjectsWithTag("goal"))
            if ((g.transform.position - pos).sqrMagnitude < 0.01f) return g;
        return null;
    }

    // ================= CSV + summary =================

    private static void WriteGridCsv(List<EpRow> rows, bool reset)
    {
        string path = CsvPath("showcase_eval.csv");
        var sb = new StringBuilder("stage,mirrored,variant,outcome,steps,jump_uses,pit_falls,stage_name,reset_mode\n");
        foreach (var r in rows)
            sb.Append(string.Format(CultureInfo.InvariantCulture, "{0},{1},{2},{3},{4},{5},{6},{7},{8}\n",
                r.stage, r.mirrored ? "true" : "false", r.variant, r.outcome, r.steps, r.jumpUses, r.pitFalls,
                CourseSpec.StageNames[r.stage], reset ? "reset" : "as-is"));
        WriteFile(path, sb.ToString());
        Debug.Log("[ShowcaseEval] wrote " + rows.Count + " rows -> " + path);
    }

    private static void WriteAbCsv(List<EpRow> asis, List<EpRow> reset)
    {
        string path = CsvPath("showcase_ab.csv");
        var sb = new StringBuilder("arm,stage,mirrored,variant,seed,outcome,steps,jump_uses,pit_falls\n");
        AppendAbRows(sb, "as-is", asis);
        AppendAbRows(sb, "reset", reset);
        WriteFile(path, sb.ToString());
        Debug.Log("[ShowcaseEval] wrote A/B (" + (asis.Count + reset.Count) + " rows) -> " + path);
    }

    private static void AppendAbRows(StringBuilder sb, string arm, List<EpRow> rows)
    {
        foreach (var r in rows)
            sb.Append(string.Format(CultureInfo.InvariantCulture, "{0},{1},{2},{3},{4},{5},{6},{7},{8}\n",
                arm, r.stage, r.mirrored ? "true" : "false", r.variant, r.seed, r.outcome, r.steps, r.jumpUses, r.pitFalls));
    }

    private static void LogSummary(List<EpRow> rows, bool reset)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"[ShowcaseEval] ===== GRID SUMMARY (per stage, reset-mode={reset}) =====");
        sb.AppendLine("stage  name            n   success   meanJumps   pit%    mirF_succ  mirT_succ");
        foreach (int stage in Stages)
        {
            var st = rows.FindAll(r => r.stage == stage);
            if (st.Count == 0) continue;
            int succ = CountSuccess(st);
            int pit = st.FindAll(r => r.pitFalls > 0).Count;
            float meanJumps = 0f;
            foreach (var r in st) meanJumps += r.jumpUses;
            meanJumps /= st.Count;
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "{0,-5}  {1,-14} {2,4}    {3,6:F3}   {4,7:F3}   {5,5:F1}    {6,7:F3}    {7,7:F3}",
                stage, CourseSpec.StageNames[stage], st.Count, succ / (float)st.Count, meanJumps,
                100f * pit / st.Count, MirrorSucc(st, false), MirrorSucc(st, true)));
        }
        // Jump-on-NoLoop-gap-success rate: stages 3(NoLoop)/4 successes that used >=1 jump (the gap needs a jump).
        var gapSucc = rows.FindAll(r => (r.stage == 4 || (r.stage == 3 && r.variant == CourseVariant.NoLoop)) && r.outcome == "success");
        int gapJumped = gapSucc.FindAll(r => r.jumpUses >= 1).Count;
        sb.AppendLine($"jump-on-NoLoop-gap success: {gapJumped}/{gapSucc.Count} used >=1 jump" +
                      (gapSucc.Count > 0 ? $" ({100f * gapJumped / gapSucc.Count:F0}%)" : ""));
        Debug.Log(sb.ToString());
    }

    private static int CountSuccess(List<EpRow> rows) => rows.FindAll(r => r.outcome == "success").Count;

    private static float MirrorSucc(List<EpRow> rows, bool mir)
    {
        var m = rows.FindAll(r => r.mirrored == mir);
        return m.Count == 0 ? 0f : CountSuccess(m) / (float)m.Count;
    }

    private static string CsvPath(string name) =>
        Path.GetFullPath(Path.Combine(Application.dataPath, "../../training/eval/" + name));

    private static void WriteFile(string path, string content)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllText(path, content);
    }

    // ================= small helpers =================

    private static bool ParseBool(string s) =>
        !string.IsNullOrWhiteSpace(s) && (s == "1" || s.Equals("true", System.StringComparison.OrdinalIgnoreCase)
            || s.Equals("yes", System.StringComparison.OrdinalIgnoreCase));

    private static bool ActionsEqual(float[] a, float[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (Mathf.Abs(a[i] - b[i]) > ActionEps) return false;
        return true;
    }

    private static string Fmt(float[] a) =>
        string.Join(",", System.Array.ConvertAll(a, x => x.ToString("F4", CultureInfo.InvariantCulture)));

    private static bool Approx(Vector3 a, Vector3 b) => (a - b).sqrMagnitude < 1e-6f;
}
