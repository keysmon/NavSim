using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using Unity.MLAgents.Sensors;          // CameraSensorComponent
using Unity.MLAgents.Policies;         // BehaviorParameters
using NavSim.Runtime;                  // NavEnvironment, NavAgent, CourseBuilder

// Showcase training-scene builder. Forks the committed pixel-arm scene (Training_pixel.unity) into the curated
// COURSE-mode showcase scene: adds a CourseBuilder "Course" GameObject on the static-geometry layer, wires
// env.course + CourseBuilder.litShader, REPAIRS the reward.jumpPenalty serialization trap (Training_pixel.unity was
// saved before jumpPenalty existed, so the forked agent deserializes jumpPenalty==0 — silently disabling the flat
// jump cost for a whole 3M run), verifies MaxStep, and saves as Training_showcase.unity. The SCRIPT is throwaway
// tooling; the committed artifact is Training_showcase.unity. Batchmode:
//   Unity -batchmode -projectPath NavSim -executeMethod ShowcaseSceneSetup.Build       -logFile -
//   Unity -batchmode -projectPath NavSim -executeMethod ShowcaseSceneSetup.BuildPlayer -logFile -
//
// SELF-ASSERTING: every wire is read back and, LOAD-BEARING, re-verified from the SAVED FILE (re-open) — the
// serialization trap lives on disk, not in memory, so an in-memory read-back alone would not catch it. Exit codes:
// 0 == saved + all asserts pass; 1 == a missing piece / failed assert; 2 == exception.
public static class ShowcaseSceneSetup
{
    private const string BaseScene     = "Assets/Scenes/Training_pixel.unity";
    private const string ShowcaseScene = "Assets/Scenes/Training_showcase.unity";
    private const float  JumpPenalty   = 0.02f;   // flat per-jump cost (RewardConfig.Default), the trap value
    private const int    MinMaxStep    = 3000;    // per-episode cap: one course traversal must fit

    public static void Build()
    {
        try
        {
            EditorSceneManager.OpenScene(BaseScene, OpenSceneMode.Single);
            var env = Object.FindAnyObjectByType<NavEnvironment>();
            var agent = Object.FindAnyObjectByType<NavAgent>();
            if (env == null || agent == null)
            { Debug.LogError("[ShowcaseScene] missing env/agent in " + BaseScene); EditorApplication.Exit(1); return; }

            bool assertOk = true;

            // (1) Static-geometry layer: read the env's LayerMask, use its LOWEST set bit for the Course GameObject
            // so the course primitives share the layer NavEnvironment.TargetPerceivable Linecasts against for
            // occlusion. A zero mask means the base scene never had a static layer configured — hard-fail.
            var envSo = new SerializedObject(env);
            int staticMask = envSo.FindProperty("staticGeometryMask").intValue;
            if (staticMask == 0)
            { Debug.LogError("[ShowcaseScene] env.staticGeometryMask is 0 — no static-geometry layer to place the course on"); EditorApplication.Exit(1); return; }
            int courseLayer = LowestSetBit(staticMask);

            // (2) "Course" GameObject at the world origin with identity rotation + unit scale (CourseBuilder places
            // pieces in WORLD space; the transform must not offset/rotate/scale them). Reuse on re-run.
            GameObject courseGo = GameObject.Find("Course");
            if (courseGo == null) courseGo = new GameObject("Course");
            courseGo.transform.SetParent(null, worldPositionStays: false);
            courseGo.transform.position = Vector3.zero;
            courseGo.transform.rotation = Quaternion.identity;
            courseGo.transform.localScale = Vector3.one;
            courseGo.layer = courseLayer;
            var course = courseGo.GetComponent<CourseBuilder>();
            if (course == null) course = courseGo.AddComponent<CourseBuilder>();

            // (3) Wire CourseBuilder.litShader to the built-in Standard shader ASSET (a serialized asset ref survives
            // into a player build; a runtime Shader.Find is only an Editor fallback and IS stripped — magenta course).
            Shader standard = Shader.Find("Standard");
            if (standard == null)
            { Debug.LogError("[ShowcaseScene] Shader.Find(\"Standard\") returned null"); EditorApplication.Exit(1); return; }
            var courseSo = new SerializedObject(course);
            courseSo.FindProperty("litShader").objectReferenceValue = standard;
            courseSo.ApplyModifiedProperties();

            // (4) Wire env.course == the CourseBuilder (private [SerializeField]) — this is what flips the env into
            // COURSE mode (NavEnvironment.CourseMode == course != null).
            envSo.FindProperty("course").objectReferenceValue = course;
            envSo.ApplyModifiedProperties();

            // (5) REPAIR THE SERIALIZATION TRAP. Training_pixel.unity predates RewardConfig.jumpPenalty, so the forked
            // agent's serialized reward.jumpPenalty deserialized to 0. Field initializers do NOT run on deserialization,
            // so RewardConfig.Default's 0.02 never applies — set it explicitly. decoyPenalty/pitPenalty are read back
            // to confirm the struct deserialized sanely (should be their M6 0.25).
            var agentSo = new SerializedObject(agent);
            agentSo.FindProperty("reward.jumpPenalty").floatValue = JumpPenalty;
            agentSo.ApplyModifiedProperties();

            // (6) MaxStep: one course traversal must fit in an episode. MaxStep is an Agent property backed by a
            // serialized field; set through the property so the write serializes, and SetDirty so SaveScene picks it up.
            int maxStepBefore = agent.MaxStep;
            if (agent.MaxStep < MinMaxStep) agent.MaxStep = MinMaxStep;
            EditorUtility.SetDirty(agent);

            // (7) Read back the in-memory wiring before saving (a typo'd path or a failed Apply shows up here).
            var agentReadSo = new SerializedObject(agent);
            float jpMem = agentReadSo.FindProperty("reward.jumpPenalty").floatValue;
            float dpMem = agentReadSo.FindProperty("reward.decoyPenalty").floatValue;
            float ppMem = agentReadSo.FindProperty("reward.pitPenalty").floatValue;
            var envReadSo = new SerializedObject(env);
            bool courseWiredMem = envReadSo.FindProperty("course").objectReferenceValue == course;
            var courseReadSo = new SerializedObject(course);
            bool shaderWiredMem = courseReadSo.FindProperty("litShader").objectReferenceValue == standard;

            var cs = agent.GetComponent<CameraSensorComponent>();
            var bp = agent.GetComponent<BehaviorParameters>();
            bool camOk = cs != null && cs.Width == 84 && cs.Height == 84;
            bool bpOk = bp != null && bp.BehaviorName == "NavAgent" && bp.BrainParameters.VectorObservationSize == 5;

            assertOk &= Mathf.Approximately(jpMem, JumpPenalty);
            assertOk &= Mathf.Approximately(dpMem, 0.25f);
            assertOk &= Mathf.Approximately(ppMem, 0.25f);
            assertOk &= courseWiredMem && shaderWiredMem;
            assertOk &= agent.MaxStep >= MinMaxStep;
            assertOk &= camOk && bpOk;

            // (8) Save as a COPY (the active scene stays Training_pixel — leaves the base untouched + gives a clean re-open).
            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            bool saved = EditorSceneManager.SaveScene(EditorSceneManager.GetActiveScene(), ShowcaseScene, true);

            Debug.Log("[ShowcaseScene] IN-MEM: jumpPenalty=" + jpMem + " decoyPenalty=" + dpMem + " pitPenalty=" + ppMem +
                      " courseLayer=" + courseLayer + "(mask=" + staticMask + ") courseWired=" + courseWiredMem +
                      " shaderWired=" + shaderWiredMem + " maxStep=" + maxStepBefore + "->" + agent.MaxStep +
                      " cam=" + (cs != null ? cs.Width + "x" + cs.Height : "NULL") +
                      " bp=" + (bp != null ? bp.BehaviorName + "/vec" + bp.BrainParameters.VectorObservationSize : "NULL") +
                      " saved=" + saved);

            // (9) LOAD-BEARING: re-open the SAVED FILE and re-verify. The Task-1 trap is a stale value ON DISK; only a
            // re-read from Training_showcase.unity proves the 3M run will actually consume jumpPenalty==0.02.
            EditorSceneManager.OpenScene(ShowcaseScene, OpenSceneMode.Single);
            var env2 = Object.FindAnyObjectByType<NavEnvironment>();
            var agent2 = Object.FindAnyObjectByType<NavAgent>();
            bool diskOk = env2 != null && agent2 != null;
            if (diskOk)
            {
                var a2 = new SerializedObject(agent2);
                float jpDisk = a2.FindProperty("reward.jumpPenalty").floatValue;
                float dpDisk = a2.FindProperty("reward.decoyPenalty").floatValue;
                float ppDisk = a2.FindProperty("reward.pitPenalty").floatValue;
                var e2 = new SerializedObject(env2);
                bool courseDisk = e2.FindProperty("course").objectReferenceValue != null;
                var course2 = env2.Course;
                bool shaderDisk = false;
                if (course2 != null)
                    shaderDisk = new SerializedObject(course2).FindProperty("litShader").objectReferenceValue != null;
                var cs2 = agent2.GetComponent<CameraSensorComponent>();
                var bp2 = agent2.GetComponent<BehaviorParameters>();
                bool camDisk = cs2 != null && cs2.Width == 84 && cs2.Height == 84;
                bool bpDisk = bp2 != null && bp2.BehaviorName == "NavAgent" && bp2.BrainParameters.VectorObservationSize == 5;
                int courseLayerDisk = course2 != null ? course2.gameObject.layer : -1;
                // Course primitives are placed in WORLD space, so the builder transform MUST be an identity
                // frame (origin / no rotation / unit scale) or every piece is offset/rotated/scaled. Vector3
                // and Quaternion == compare with Unity's built-in tolerance.
                Transform ct = course2 != null ? course2.transform : null;
                bool courseXformDisk = ct != null
                    && ct.position == Vector3.zero
                    && ct.rotation == Quaternion.identity
                    && ct.localScale == Vector3.one;

                diskOk &= Mathf.Approximately(jpDisk, JumpPenalty);
                diskOk &= Mathf.Approximately(dpDisk, 0.25f) && Mathf.Approximately(ppDisk, 0.25f);
                diskOk &= courseDisk && shaderDisk;
                diskOk &= agent2.MaxStep >= MinMaxStep;
                diskOk &= camDisk && bpDisk;
                diskOk &= courseLayerDisk == courseLayer;
                diskOk &= courseXformDisk;

                Debug.Log("[ShowcaseScene] ON-DISK (" + ShowcaseScene + "): jumpPenalty=" + jpDisk +
                          " decoyPenalty=" + dpDisk + " pitPenalty=" + ppDisk + " course=" + courseDisk +
                          " shader=" + shaderDisk + " maxStep=" + agent2.MaxStep + " courseLayer=" + courseLayerDisk +
                          " courseXform=" + courseXformDisk + "(pos=" + (ct != null ? ct.position.ToString("F3") : "NULL") +
                          " rot=" + (ct != null ? ct.rotation.eulerAngles.ToString("F3") : "NULL") +
                          " scale=" + (ct != null ? ct.localScale.ToString("F3") : "NULL") + ")" +
                          " cam=" + (cs2 != null ? cs2.Width + "x" + cs2.Height : "NULL") +
                          " bp=" + (bp2 != null ? bp2.BehaviorName + "/vec" + bp2.BrainParameters.VectorObservationSize : "NULL"));
            }
            else Debug.LogError("[ShowcaseScene] re-open of " + ShowcaseScene + " missing env/agent");

            bool ok = saved && assertOk && diskOk;
            Debug.Log("[ShowcaseScene] RESULT saved=" + saved + " inMemAsserts=" + assertOk + " onDiskAsserts=" + diskOk +
                      " -> " + (ok ? "OK" : "FAIL"));
            EditorApplication.Exit(ok ? 0 : 1);
        }
        catch (System.Exception e)
        {
            Debug.LogError("[ShowcaseScene] Build FAILED: " + e);
            EditorApplication.Exit(2);
        }
    }

    // Build a StandaloneOSX player of Training_showcase for headless mlagents training (WITH graphics so the camera
    // renders). Clone of M6PixelSceneSetup.BuildPlayer with the showcase scene/output.
    public static void BuildPlayer()
    {
        try
        {
            if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.StandaloneOSX)
                EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Standalone, BuildTarget.StandaloneOSX);
            if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.StandaloneOSX)
            { Debug.LogError("[ShowcaseScene] active build target is not StandaloneOSX"); EditorApplication.Exit(1); return; }
            var opts = new BuildPlayerOptions
            {
                scenes = new[] { ShowcaseScene },
                locationPathName = System.IO.Path.GetFullPath(Application.dataPath + "/../Builds/ShowcaseTrain.app"),
                target = BuildTarget.StandaloneOSX,
                options = BuildOptions.None
            };
            var summary = BuildPipeline.BuildPlayer(opts).summary;
            Debug.Log("[ShowcaseScene] BuildPlayer result=" + summary.result + " errors=" + summary.totalErrors +
                      " out=" + opts.locationPathName);
            EditorApplication.Exit(summary.result == BuildResult.Succeeded && summary.totalErrors == 0 ? 0 : 1);
        }
        catch (System.Exception e)
        {
            Debug.LogError("[ShowcaseScene] BuildPlayer FAILED: " + e);
            EditorApplication.Exit(2);
        }
    }

    // Lowest set bit index of a non-zero mask (the static-geometry layer to place the course on).
    private static int LowestSetBit(int mask)
    {
        for (int i = 0; i < 32; i++)
            if ((mask & (1 << i)) != 0) return i;
        return 0;
    }
}
