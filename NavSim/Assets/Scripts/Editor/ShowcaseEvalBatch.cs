using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;   // InferenceDevice
using Unity.InferenceEngine;     // ModelAsset (com.unity.ai.inference)
using NavSim.Runtime;

// SHOWCASE course-mode eval — the course-mode sibling of M6SearchEval + M6EvalBatch, fused into one batchmode entry
// (there is a single scene and a single model, so no per-arm shell is needed). Drives the trained showcase policy
// through the FORCED 5-stage grid and writes the reproducible artifact training/eval/showcase_eval.csv.
//
// Batchmode (NO -quit: Play mode needs the game loop; we Exit() ourselves; NOT -nographics: the pixel CameraSensor
// needs a graphics device or it feeds blank frames):
//   SHOWCASE_EVAL_ONNX=Assets/Models/Showcase/showcase_s0.onnx \
//     Unity -batchmode -projectPath NavSim -executeMethod ShowcaseEvalBatch.Run -logFile -
//
// Model wiring mirrors the PROVEN M6SearchEval path EXACTLY: enter Play, step ONCE (so the pixel CameraSensor's
// Texture2D is allocated before SetModel validates the model against the sensor spec), then agent.SetModel — the
// only inference install. No BehaviorType manipulation (M6 never touches it; InferenceOnly + a null bp.Model at any
// re-init would throw, and pre-wiring bp.Model in edit mode risks an auto-init before the camera texture exists).
//
// Idiom invariants (M5/M6-proven, project standing practice):
//   - Physics.simulationMode = Script for the whole run: Step() = Academy.EnvironmentStep() + Physics.Simulate(dt),
//     so the ONNX policy (via the scene's DecisionRequester) produces one action per step, deterministically, with
//     no FixedUpdate/Academy interleave. Physics.SyncTransforms() after EVERY ForceCourse (batchmode has no auto-
//     sync; a deferred transform write would let step 1 perceive stale geometry).
//   - agent.MaxStep = 0 + env.EvalMode = true: the HARNESS owns the episode boundary. EvalMode suppresses the
//     agent-side EndEpisode on both success and a hard-decoy touch (verified in NavAgent.OnActionReceived's reached
//     branch: `EndsEpisodeOnGoal && !EvalMode` and `decoyEnter && !EvalMode && DecoyHard` are both skipped), so a
//     reached-target only redraws COLOURS (course-mode RespawnGoal keeps the goal POSITIONS fixed) and never re-rolls
//     the geometry mid-episode.
//   - Outcomes are detected GEOMETRICALLY against VALUES captured at episode start (target0 = the red slot, decoy0 =
//     the two non-red slots), decoy-first (pessimistic tie-break), always hard — independent of the training
//     soften->harden schedule. A pit fall is a NON-terminal delta count (the agent respawns on DeckA and continues).
//
// Exit codes: 0 == all grid episodes completed (regardless of rates — the VERDICT is a deferred controller step);
// non-zero == harness failure (missing scene/env/agent, not course mode, no model, or an exception).
public static class ShowcaseEvalBatch
{
    private const string ScenePath = "Assets/Scenes/Training_showcase.unity";
    private const string DefaultOnnx = "Assets/Models/Showcase/showcase_s0.onnx";
    private const int WarmupFrames = 40;      // Play + Academy/scene init before we seize control (P0/M6 idiom)
    private const int MaxSteps = 3000;        // per-episode academy-step budget == agent.MaxStep -> "timeout"
    private const int EpisodesPerCell = 25;   // episodes per (stage, mirror, variant) cell

    // Forced grid: every stage on both mirror states; stage 3 ("The gap") on BOTH pit variants, the rest NoLoop.
    // -> stages 0,1,2,4: 2 mirror x 25 = 50 each; stage 3: 2 mirror x 2 variant x 25 = 100. Total 300 episodes.
    private static readonly int[] Stages = { 0, 1, 2, 3, 4 };

    private static bool _savedEpmoEnabled;               // snapshot of the tracked EditorSettings play-mode options
    private static EnterPlayModeOptions _savedEpmo;      // (restored before Exit so EditorSettings.asset stays un-dirtied)
    private static SimulationMode _savedSimMode;         // restored before Exit (scene is never saved; in-memory only)
    private static int _frames;
    private static string _onnxPath;                     // resolved in Run(); survives into Tick via DisableDomainReload

    private struct EpRow
    {
        public int stage; public bool mirrored; public CourseVariant variant;
        public string outcome; public int steps; public int jumpUses; public int pitFalls;
    }

    public static void Run()
    {
        // Snapshot the tracked project settings BEFORE mutating them so we can restore the exact prior values before
        // Exit — the git-tracked ProjectSettings/{EditorSettings,DynamicsManager}.asset must stay un-dirtiable
        // regardless of Unity's on-exit flush. (DisableDomainReload keeps these statics alive into Tick.)
        _savedEpmoEnabled = EditorSettings.enterPlayModeOptionsEnabled;
        _savedEpmo = EditorSettings.enterPlayModeOptions;
        _savedSimMode = Physics.simulationMode;
        _onnxPath = System.Environment.GetEnvironmentVariable("SHOWCASE_EVAL_ONNX");
        if (string.IsNullOrWhiteSpace(_onnxPath)) _onnxPath = DefaultOnnx;
        try
        {
            EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
            // DisableDomainReload keeps this class's static state (_frames, the update callback, _onnxPath) alive
            // across the Play transition; DisableSceneReload keeps the loaded scene. Same idiom as M6EvalBatch.
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
            Debug.LogError("[ShowcaseEval] Run setup FAILED: " + e);
            EditorApplication.Exit(2);
        }
    }

    private static void Tick()
    {
        if (!EditorApplication.isPlaying) return;   // wait for Play + Academy
        _frames++;
        if (_frames < WarmupFrames) return;
        EditorApplication.update -= Tick;
        int code = 2;
        try { code = RunEval(); }
        catch (System.Exception e) { Debug.LogError("[ShowcaseEval] EXCEPTION: " + e); code = 2; }
        finally { RestoreProjectSettings(); } // restore physics + EditorSettings before we exit, always
        EditorApplication.Exit(code);
    }

    private static void RestoreProjectSettings()
    {
        Physics.simulationMode = _savedSimMode;
        EditorSettings.enterPlayModeOptionsEnabled = _savedEpmoEnabled;
        EditorSettings.enterPlayModeOptions = _savedEpmo;
    }

    private static int RunEval()
    {
        var env = Object.FindAnyObjectByType<NavEnvironment>();
        var agent = Object.FindAnyObjectByType<NavAgent>();
        if (env == null || agent == null) { Debug.LogError("[ShowcaseEval] missing env/agent in scene"); return 2; }
        if (!env.CourseMode) { Debug.LogError("[ShowcaseEval] env.course not wired — scene is not in COURSE mode"); return 2; }

        // M6SearchEval.SetupHarness idiom: the harness owns the boundary; step ONCE before SetModel so the pixel
        // CameraSensor's Texture2D is allocated (SetModel validates the model against the live sensor spec).
        Physics.simulationMode = SimulationMode.Script;
        agent.MaxStep = 0;      // no agent-side max-step EndEpisode; the MaxSteps budget below is the sole timeout
        env.EvalMode = true;    // suppress EndEpisode on success AND on a hard-decoy touch; outcomes are geometric here
        Step();                 // allocate the CameraSensor texture BEFORE the first SetModel (pixel fork)

        var model = AssetDatabase.LoadAssetAtPath<ModelAsset>(_onnxPath);
        if (model == null)
        {
            Debug.LogError($"[ShowcaseEval] model not found at '{_onnxPath}' — set SHOWCASE_EVAL_ONNX or bake the ONNX " +
                           "into Assets/Models/Showcase/. Eval cannot run without a trained policy.");
            return 2;
        }
        agent.SetModel("NavAgent", model, InferenceDevice.Burst);
        Debug.Log($"[ShowcaseEval] model={_onnxPath} device=Burst episodesPerCell={EpisodesPerCell} maxSteps={MaxSteps} " +
                  "grid=stages0-4 x mirror{false,true} x (stage3:{SafeLoop,NoLoop} else NoLoop) -> 300 episodes");

        var rows = new List<EpRow>(300);
        foreach (int stage in Stages)
            foreach (bool mir in new[] { false, true })
            {
                if (stage == 3)
                {
                    RunCell(env, agent, 3, mir, CourseVariant.SafeLoop, rows);
                    RunCell(env, agent, 3, mir, CourseVariant.NoLoop, rows);
                }
                else RunCell(env, agent, stage, mir, CourseVariant.NoLoop, rows);
            }

        WriteCsv(rows);
        LogSummary(rows);
        Debug.Log($"[ShowcaseEval] COMPLETE — {rows.Count} episodes.");
        return 0;
    }

    // One grid cell: EpisodesPerCell deterministic episodes at a fixed (stage, mirror, variant).
    private static void RunCell(NavEnvironment env, NavAgent agent, int stage, bool mir, CourseVariant variant, List<EpRow> rows)
    {
        for (int ep = 0; ep < EpisodesPerCell; ep++)
        {
            int seed = SeedFor(stage, mir, variant, ep);
            UnityEngine.Random.InitState(seed);   // belt: reproducible even if ForceCourse ever gains a jitter path
            env.SeedColorRng(seed);                // the ONLY stochastic element in ForceCourse's jitter:false path
            env.ForceCourse(stage, mir, variant);  // deterministic place: agent at spawn, goals at the fixed slots
            Physics.SyncTransforms();              // Script sim defers transform->physics; push so step 1 perceives fresh geometry

            // Capture outcome anchors as VALUES BEFORE stepping (M6 idiom). A reached target recolours the triad
            // (RespawnGoal -> AssignColorsAndTarget), but course-mode goal POSITIONS never move, so these captured
            // values stay authoritative for the whole episode; we also break the same step a reach is detected.
            Vector3 target0 = env.GoalPositionFor(agent);   // the red (target) slot
            Vector3[] decoy0 = env.DecoyPositions();          // the two non-red slots

            int pit0 = env.PitFalls, jump0 = agent.JumpUses;
            string outcome = "timeout";
            int steps = MaxSteps;
            for (int s = 0; s < MaxSteps; s++)
            {
                Step();
                Vector3 pos = agent.transform.position;
                // Decoy FIRST (pessimistic tie-break, always hard), then the target. Within GoalRadius (== 1.5).
                if (Within(env, pos, decoy0[0]) || Within(env, pos, decoy0[1])) { outcome = "decoy"; steps = s + 1; break; }
                if (Within(env, pos, target0)) { outcome = "success"; steps = s + 1; break; }
            }
            rows.Add(new EpRow
            {
                stage = stage,
                mirrored = mir,
                variant = variant,
                outcome = outcome,
                steps = steps,
                jumpUses = agent.JumpUses - jump0,   // agent.JumpUses delta over the episode
                pitFalls = env.PitFalls - pit0,      // NON-terminal: the agent respawns and continues
            });
        }
    }

    private static bool Within(NavEnvironment env, Vector3 pos, Vector3 slot) =>
        Vector3.Distance(pos, slot) < env.GoalRadius;

    // Deterministic per-episode seed: unique + collision-free per (stage, mirror, variant, ep) so a re-run of the
    // whole grid reproduces identical colour/target pairings. ep < 25, variant in {0,1}, mirror in {0,1}, stage 0-4.
    private static int SeedFor(int stage, bool mir, CourseVariant variant, int ep) =>
        stage * 100000 + (mir ? 1 : 0) * 10000 + (int)variant * 1000 + ep;

    private static void Step()
    {
        Academy.Instance.EnvironmentStep();
        Physics.Simulate(Time.fixedDeltaTime);
    }

    // Single clobber-write (this is ONE run producing all 300 rows — no per-arm merge like M6). The required columns
    // (stage,mirrored,variant,outcome,steps,jump_uses,pit_falls) keep fixed positions for the deferred verdict step;
    // stage_name is appended LAST as a legibility nicety.
    private static void WriteCsv(List<EpRow> rows)
    {
        string path = Path.GetFullPath(Path.Combine(Application.dataPath, "../../training/eval/showcase_eval.csv"));
        var sb = new StringBuilder("stage,mirrored,variant,outcome,steps,jump_uses,pit_falls,stage_name\n");
        foreach (var r in rows)
            sb.Append(string.Format(CultureInfo.InvariantCulture, "{0},{1},{2},{3},{4},{5},{6},{7}\n",
                r.stage, r.mirrored ? "true" : "false", r.variant, r.outcome, r.steps, r.jumpUses, r.pitFalls,
                CourseSpec.StageNames[r.stage]));
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllText(path, sb.ToString());
        Debug.Log("[ShowcaseEval] wrote " + rows.Count + " rows -> " + path);
    }

    // Per-stage success rate, mean jumps/episode, pit-episode fraction, and the both-mirror success split — the
    // legible signal the deferred controller checks against the spec acceptance (success >= 0.8 every stage, etc).
    private static void LogSummary(List<EpRow> rows)
    {
        var sb = new StringBuilder();
        sb.AppendLine("[ShowcaseEval] ===== SUMMARY (per stage) =====");
        sb.AppendLine("stage  name            n   success   meanJumps   pit%    mirF_succ  mirT_succ");
        foreach (int stage in Stages)
        {
            var st = rows.FindAll(r => r.stage == stage);
            if (st.Count == 0) continue;
            int succ = st.FindAll(r => r.outcome == "success").Count;
            int pit = st.FindAll(r => r.pitFalls > 0).Count;
            float meanJumps = 0f;
            foreach (var r in st) meanJumps += r.jumpUses;
            meanJumps /= st.Count;
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "{0,-5}  {1,-14} {2,4}    {3,6:F3}   {4,7:F3}   {5,5:F1}    {6,7:F3}    {7,7:F3}",
                stage, CourseSpec.StageNames[stage], st.Count, succ / (float)st.Count, meanJumps,
                100f * pit / st.Count, MirrorSucc(st, false), MirrorSucc(st, true)));
        }
        Debug.Log(sb.ToString());
    }

    private static float MirrorSucc(List<EpRow> rows, bool mir)
    {
        var m = rows.FindAll(r => r.mirrored == mir);
        return m.Count == 0 ? 0f : m.FindAll(r => r.outcome == "success").Count / (float)m.Count;
    }
}
