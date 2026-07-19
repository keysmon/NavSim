using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using Unity.InferenceEngine;            // ModelAsset (com.unity.ai.inference)
using Unity.MLAgents.Sensors;           // CameraSensorComponent
using Unity.MLAgents.Policies;          // BehaviorParameters, BehaviorType, InferenceDevice
using NavSim.Runtime;

// M6 v2 WebGL demo scene builder. Forks the committed pixel-arm scene (Training_pixel.unity) into a spectator
// demo: the best pixel seed runs in inference (no trainer), a 3rd-person camera shows the arena, the agent's own
// egocentric AgentEyeCam keeps feeding the CNN sensor, and an M6DemoUI overlays a "New layout" re-roll + an L0-L3
// difficulty selector. The SCRIPT is throwaway-style tooling; the committed artifacts are Demo_M6.unity + the
// WebGL_M6Demo build (Builds/ is gitignored — the build stays local for the deploy step). Batchmode:
//   Unity -batchmode -projectPath NavSim -executeMethod M6DemoSceneSetup.Build           -logFile -
//   Unity -batchmode -projectPath NavSim -executeMethod M6DemoSceneSetup.BuildWebGL      -logFile -
//   Unity -batchmode -projectPath NavSim -executeMethod M6DemoSceneSetup.RestoreStandalone -logFile -
public static class M6DemoSceneSetup
{
    private const string BaseScene = "Assets/Scenes/Training_pixel.unity";
    private const string DemoScene = "Assets/Scenes/Demo_M6.unity";
    // Best pixel seed by held-out eval (m6_pixel_s1: success 0.81 overall, 0.72..0.92 across L0-L3).
    private const string ModelPath = "Assets/Models/M6/m6_pixel_s1.onnx";
    private const string WebGLOut  = "Builds/WebGL_M6Demo";

    // Inference backend for the browser. LOAD-BEARING + UNPROVEN: Task 9 is the first time Sentis inference runs
    // in a browser (the Phase-0 probe only proved compile + build size). Default == Burst == the CPU path, which
    // is the WebGL-safe first choice (ComputeShader needs compute shaders WebGL2 lacks; PixelShader is the GPU
    // fragment fallback). If the browser console names an unsupported backend, re-roll this to PixelShader and
    // re-run Build()+BuildWebGL(). Do NOT report BLOCKED before trying the device alternatives.
    private const InferenceDevice InferDevice = InferenceDevice.Default;

    // 3rd-person spectator pose (elevated, angled down, slight corner offset for depth). LookAt(origin) covers the
    // whole [-arenaHalf, arenaHalf] arena. Tune here + re-run if the browser view frames the arena poorly.
    private const float SpecSide = -6f, SpecHeight = 17f, SpecBack = 15f, SpecFov = 55f;

    public static void Build()
    {
        try
        {
            EditorSceneManager.OpenScene(BaseScene, OpenSceneMode.Single);
            var env = Object.FindAnyObjectByType<NavEnvironment>();
            var agent = Object.FindAnyObjectByType<NavAgent>();
            if (env == null || agent == null) { Debug.LogError("[M6DemoScene] missing env/agent in " + BaseScene); EditorApplication.Exit(1); return; }

            // (1) bake the best pixel seed into BehaviorParameters + run inference with NO trainer.
            var bp = agent.GetComponent<BehaviorParameters>();
            var model = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelPath);
            if (bp == null || model == null)
            { Debug.LogError("[M6DemoScene] missing BehaviorParameters or model at " + ModelPath); EditorApplication.Exit(1); return; }
            bp.BehaviorName = "NavAgent";
            bp.Model = model;
            bp.InferenceDevice = InferDevice;
            bp.BehaviorType = BehaviorType.InferenceOnly;
            EditorUtility.SetDirty(bp);

            // (2) the agent's egocentric eye-cam keeps FEEDING the CNN sensor but must NOT draw to the game screen
            // (it would fight the spectator camera). Disabling the Camera component is the standard ml-agents idiom:
            // CameraSensorComponent calls Camera.Render() on demand, which works regardless of the enabled flag.
            var cs = agent.GetComponent<CameraSensorComponent>();
            Camera eyeCam = cs != null ? cs.Camera : null;
            if (eyeCam == null) { var t = agent.transform.Find("AgentEyeCam"); if (t != null) eyeCam = t.GetComponent<Camera>(); }
            if (eyeCam != null)
            {
                eyeCam.enabled = false;
                if (eyeCam.CompareTag("MainCamera")) eyeCam.tag = "Untagged"; // only the spectator is MainCamera
            }

            // (3) 3rd-person spectator camera the VISITOR sees. Reconfigure the base scene's non-eye camera (the
            // stock "Main Camera") or create one. cullingMask = everything INCLUDING the Agent layer — the pixel
            // arm culls the agent body from its EYE cam, but the spectator must SEE the agent (it is the star).
            Camera spec = null;
            foreach (var c in Object.FindObjectsByType<Camera>(FindObjectsInactive.Exclude))
                if (c != eyeCam) { spec = c; break; }
            if (spec == null) spec = new GameObject("SpectatorCamera").AddComponent<Camera>();
            spec.gameObject.name = "SpectatorCamera";
            spec.tag = "MainCamera";
            spec.enabled = true;
            spec.depth = 0f;
            spec.cullingMask = ~0;                       // render everything, incl. the Agent layer
            spec.clearFlags = CameraClearFlags.Skybox;
            spec.fieldOfView = SpecFov;
            spec.nearClipPlane = 0.1f;
            spec.farClipPlane = 250f;
            spec.transform.position = new Vector3(SpecSide, SpecHeight, -SpecBack);
            spec.transform.LookAt(Vector3.zero);

            // (4) demo UI overlay wired to the env (private [SerializeField] env).
            var uiGo = GameObject.Find("DemoUI") ?? new GameObject("DemoUI");
            var ui = uiGo.GetComponent<M6DemoUI>() ?? uiGo.AddComponent<M6DemoUI>();
            var uiSo = new SerializedObject(ui);
            uiSo.FindProperty("env").objectReferenceValue = env;
            uiSo.ApplyModifiedProperties();

            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            bool ok = EditorSceneManager.SaveScene(EditorSceneManager.GetActiveScene(), DemoScene, true); // saveAsCopy

            // (5) self-assert the regen produced a demo that will actually run.
            bool assertOk = ok
                && bp.Model != null
                && bp.BehaviorType == BehaviorType.InferenceOnly
                && spec.CompareTag("MainCamera")
                && (eyeCam == null || !eyeCam.enabled)
                && ui != null;
            Debug.Log("[M6DemoScene] saved=" + ok + " model=" + (bp.Model != null ? bp.Model.name : "NULL") +
                      " device=" + InferDevice + " behavior=" + bp.BehaviorType +
                      " maxStep=" + agent.MaxStep + " eyeCam=" + (eyeCam != null ? eyeCam.name : "NULL") +
                      " spectator=" + spec.name + "(tag=" + spec.tag + ") -> " + DemoScene + " assertOk=" + assertOk);
            EditorApplication.Exit(assertOk ? 0 : 1);
        }
        catch (System.Exception e)
        {
            Debug.LogError("[M6DemoScene] Build FAILED: " + e);
            EditorApplication.Exit(2);
        }
    }

    // Build the WebGL player of Demo_M6 into Builds/WebGL_M6Demo/. Passing `scenes` explicitly means the demo scene
    // does NOT need to be in EditorBuildSettings (nothing to dirty/restore there). ProjectSettings already carries
    // the WebGL fixes (exceptionSupport != None, etc.). RestoreStandalone (separate call) switches the target back.
    public static void BuildWebGL()
    {
        try
        {
            if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.WebGL)
                EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.WebGL, BuildTarget.WebGL);
            var opts = new BuildPlayerOptions
            {
                scenes = new[] { DemoScene },
                locationPathName = System.IO.Path.GetFullPath(Application.dataPath + "/../" + WebGLOut),
                target = BuildTarget.WebGL,
                targetGroup = BuildTargetGroup.WebGL,
                options = BuildOptions.None
            };
            var summary = BuildPipeline.BuildPlayer(opts).summary;
            Debug.Log("[M6DemoScene] BuildWebGL result=" + summary.result + " errors=" + summary.totalErrors +
                      " size=" + summary.totalSize + " out=" + opts.locationPathName);
            EditorApplication.Exit(summary.result == BuildResult.Succeeded && summary.totalErrors == 0 ? 0 : 1);
        }
        catch (System.Exception e)
        {
            Debug.LogError("[M6DemoScene] BuildWebGL FAILED: " + e);
            EditorApplication.Exit(2);
        }
    }

    // Restore the active build target to StandaloneOSX (the project's default for training/eval players).
    public static void RestoreStandalone()
    {
        if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.StandaloneOSX)
            EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Standalone, BuildTarget.StandaloneOSX);
        Debug.Log("[M6DemoScene] restored active build target -> StandaloneOSX");
        EditorApplication.Exit(0);
    }
}
