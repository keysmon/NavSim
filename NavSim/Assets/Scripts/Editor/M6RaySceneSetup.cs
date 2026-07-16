using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Policies;
using NavSim.Runtime;

// Phase-4 ray-arm scene builder. Mirrors M6PixelSceneSetup: opens the M5 Training scene and saves two derived
// arm scenes that KEEP the M5 three ray fans (no camera) — the only lever vs the pixel arm is perception.
//   - Training_ray1.unity : fans UNCHANGED (goals stay tag "goal") + tagGoalsByColor=false -> provably ~1/3
//     (the confound-detector: ray1 has ZERO color information).
//   - Training_rayc.unity : each goal-bearing fan's "goal" tag is SURGICALLY spliced to goal_c0..goal_c4
//     (terrain tags wall/obstacle/mover preserved in place) + tagGoalsByColor=true -> rayC is hand-told the
//     color categories (the declared steelman; its wider ray obs IS the manipulation, not a confound).
// Both arms keep the 8-float vector obs (proprioception + RGB cue) via NavAgent.CollectObservations.
//
// SELF-VERIFYING: after building each arm the script HARD-ASSERTS the full post-splice DetectableTags per fan
// (advisor, non-optional) — the surgical splice exists to protect wall/obstacle/mover, so the build FAILS LOUDLY
// (nonzero exit) if any terrain tag were dropped or ray1 ever carried a color tag.
//
// The SCRIPT is throwaway tooling; the committed artifacts are Training_ray1.unity + Training_rayc.unity.
// Batchmode:  Unity -batchmode -projectPath NavSim -executeMethod M6RaySceneSetup.Build -logFile -
public static class M6RaySceneSetup
{
    private static readonly string[] ColorTags = { "goal_c0", "goal_c1", "goal_c2", "goal_c3", "goal_c4" };
    private const string BaseScene = "Assets/Scenes/Training.unity";

    public static void Build()
    {
        try
        {
            bool ok1 = BuildArm(byColor: false, outPath: "Assets/Scenes/Training_ray1.unity");
            bool ok2 = BuildArm(byColor: true, outPath: "Assets/Scenes/Training_rayc.unity");
            EditorApplication.Exit(ok1 && ok2 ? 0 : 1);
        }
        catch (System.Exception e)
        {
            Debug.LogError("[M6RayScene] FAILED: " + e);
            EditorApplication.Exit(2);
        }
    }

    private static bool BuildArm(bool byColor, string outPath)
    {
        string arm = byColor ? "rayC" : "ray1";
        EditorSceneManager.OpenScene(BaseScene, OpenSceneMode.Single); // fresh open each arm -> no cross-contamination
        var env = Object.FindAnyObjectByType<NavEnvironment>();
        var agent = Object.FindAnyObjectByType<NavAgent>();
        if (env == null || agent == null) { Debug.LogError($"[M6RayScene:{arm}] missing env/agent in {BaseScene}"); return false; }

        // (0) Defensive: the ray arms have NO camera. Strip any CameraSensor + eye-cam if the base ever gained one,
        //     and reset the agent hierarchy to the Default layer so it matches M5 exactly (the "Agent" cull layer
        //     is a pixel-only trick; ray sensors must see the same world M5 trained on).
        foreach (var cs in agent.GetComponents<CameraSensorComponent>()) Object.DestroyImmediate(cs);
        Transform eye = agent.transform.Find("AgentEyeCam");
        if (eye != null) Object.DestroyImmediate(eye.gameObject);
        SetLayerRecursively(agent.gameObject, 0); // Default

        // (1) Read the fans. Capture ORIGINAL per-fan tags before touching anything.
        var fans = agent.GetComponents<RayPerceptionSensorComponent3D>();
        if (fans.Length == 0) { Debug.LogError($"[M6RayScene:{arm}] no RayPerceptionSensorComponent3D on the agent"); return false; }
        var original = fans.ToDictionary(f => f, f => new List<string>(f.DetectableTags));

        // (2) SOURCE-INTEGRITY assert (ray1 confound-detector foundation): the base scene must carry the plain
        //     "goal" tag and NEVER a color tag — else Training.unity was contaminated and ray1 would leak color.
        foreach (var f in fans)
        {
            if (original[f].Any(t => t.StartsWith("goal_c")))
            {
                Debug.LogError($"[M6RayScene:{arm}] SOURCE CONTAMINATED: fan '{f.SensorName}' already has a color tag " +
                               $"[{string.Join(",", original[f])}] in {BaseScene} — ray1 integrity broken. Abort.");
                return false;
            }
        }

        // (3) Build the EXPECTED post-splice tag set per fan, and apply it for rayC.
        var expected = new Dictionary<RayPerceptionSensorComponent3D, List<string>>();
        foreach (var f in fans)
        {
            var tags = new List<string>(original[f]);
            if (byColor && tags.Remove("goal")) tags.AddRange(ColorTags); // splice ONLY goal-bearing fans; keep the rest
            expected[f] = tags;
            f.DetectableTags = new List<string>(tags);
        }

        // (4) BehaviorParameters: vector obs is 8 (proprioception 5 + RGB cue 3); ray obs is separate (sensor-inferred).
        var bp = agent.GetComponent<BehaviorParameters>();
        if (bp != null)
        {
            bp.BehaviorName = "NavAgent";
            bp.BehaviorType = BehaviorType.Default;
            bp.BrainParameters.VectorObservationSize = 8;
        }

        // (5) env: ray arms have NO camera (agentCamera==null routes TargetPerceivable to the SENSOR-TRUTH gate —
        //     an actual goal-detecting ray hit; no angular-cone params to pin). tagGoalsByColor selects the arm.
        var so = new SerializedObject(env);
        so.FindProperty("agentCamera").objectReferenceValue = null;
        so.FindProperty("tagGoalsByColor").boolValue = byColor;
        so.ApplyModifiedProperties();

        // (6) HARD VERIFY (advisor, non-optional): read back each fan and assert the full tag SET matches expected,
        //     proving wall/obstacle/mover survived and the goal splice happened exactly where intended.
        bool allOk = true;
        foreach (var f in agent.GetComponents<RayPerceptionSensorComponent3D>())
        {
            var got = new HashSet<string>(f.DetectableTags);
            var want = new HashSet<string>(expected[f]);
            bool match = got.SetEquals(want);
            Debug.Log($"[M6RayScene:{arm}] fan '{f.SensorName}': [{string.Join(",", original[f])}] -> " +
                      $"[{string.Join(",", f.DetectableTags)}]  ({(match ? "OK" : "MISMATCH")})");
            if (!match)
            {
                Debug.LogError($"[M6RayScene:{arm}] TAG MISMATCH on '{f.SensorName}': got [{string.Join(",", got)}] " +
                               $"expected [{string.Join(",", want)}]");
                allOk = false;
            }
            // Terrain tags must never vanish from a fan that had them.
            foreach (var terrain in new[] { "wall", "obstacle", "mover" })
                if (original[f].Contains(terrain) && !got.Contains(terrain))
                { Debug.LogError($"[M6RayScene:{arm}] DROPPED terrain tag '{terrain}' from '{f.SensorName}'"); allOk = false; }
            // ray1 must have zero color info; rayC must have zero plain-"goal" info on goal-bearing fans.
            if (!byColor && got.Any(t => t.StartsWith("goal_c")))
            { Debug.LogError($"[M6RayScene:{arm}] ray1 fan '{f.SensorName}' leaked a color tag"); allOk = false; }
            if (byColor && original[f].Contains("goal") && got.Contains("goal"))
            { Debug.LogError($"[M6RayScene:{arm}] rayC fan '{f.SensorName}' still has plain 'goal' after splice"); allOk = false; }
            // INDEPENDENT of the splice expression (not just SetEquals vs `expected`): a goal-bearing rayC fan
            // MUST actually contain every runtime color tag, else a "removed goal but added nothing" regression
            // would ship rayC blind to goals yet pass every other check.
            if (byColor && original[f].Contains("goal"))
                foreach (var c in ColorTags)
                    if (!got.Contains(c))
                    { Debug.LogError($"[M6RayScene:{arm}] rayC fan '{f.SensorName}' MISSING color tag '{c}' after splice"); allOk = false; }
        }
        if (!allOk) { Debug.LogError($"[M6RayScene:{arm}] tag verification FAILED — not saving {outPath}"); return false; }

        var scene = EditorSceneManager.GetActiveScene();
        bool saved = EditorSceneManager.SaveScene(scene, outPath, true); // saveAsCopy -> leaves Training.unity untouched
        Debug.Log($"[M6RayScene:{arm}] saved={saved} fans={fans.Length} tagGoalsByColor={byColor} vecObs=8 -> {outPath}");
        return saved;
    }

    private static void SetLayerRecursively(GameObject go, int layer)
    {
        go.layer = layer;
        foreach (Transform c in go.transform) SetLayerRecursively(c.gameObject, layer);
    }

    // StandaloneOSX training players for the ray arms (mirrors M6PixelSceneSetup.BuildPlayer). Ray arms have no
    // camera, so mlagents-learn may run them headless with -nographics; the build itself is standard.
    //   Unity -batchmode -projectPath NavSim -executeMethod M6RaySceneSetup.BuildRay1Player -logFile -
    public static void BuildRay1Player() => BuildArmPlayer("Assets/Scenes/Training_ray1.unity", "Builds/M6Ray1Train.app");
    public static void BuildRayCPlayer() => BuildArmPlayer("Assets/Scenes/Training_rayc.unity", "Builds/M6RayCTrain.app");

    private static void BuildArmPlayer(string scene, string outRel)
    {
        try
        {
            if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.StandaloneOSX)
                EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Standalone, BuildTarget.StandaloneOSX);
            var opts = new BuildPlayerOptions
            {
                scenes = new[] { scene },
                locationPathName = System.IO.Path.GetFullPath(Application.dataPath + "/../" + outRel),
                target = BuildTarget.StandaloneOSX,
                options = BuildOptions.None
            };
            var summary = BuildPipeline.BuildPlayer(opts).summary;
            Debug.Log("[M6RayScene] BuildPlayer result=" + summary.result + " errors=" + summary.totalErrors + " out=" + outRel);
            EditorApplication.Exit(summary.result == BuildResult.Succeeded ? 0 : 1);
        }
        catch (System.Exception e) { Debug.LogError("[M6RayScene] BuildArmPlayer FAILED: " + e); EditorApplication.Exit(2); }
    }
}
