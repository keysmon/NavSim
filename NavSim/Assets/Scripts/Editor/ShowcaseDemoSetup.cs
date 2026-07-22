using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using Unity.InferenceEngine;            // ModelAsset (com.unity.ai.inference)
using Unity.MLAgents.Sensors;           // CameraSensorComponent
using Unity.MLAgents.Policies;          // BehaviorParameters, BehaviorType, InferenceDevice
using NavSim.Runtime;

// Showcase demo scene builder. Forks the committed course spine scene (Training_showcase.unity) into a spectator
// demo: the best showcase seed runs in inference (no trainer), a DemoCameraRig glides a 3rd-person camera to each
// stage's authored hero pose, the agent's egocentric AgentEyeCam keeps feeding the CNN sensor AND the UI's 84x84
// PiP, a TrailRenderer paints the agent's path (culled from the eye cam by the Agent layer), and ShowcaseDemoUI
// overlays named-stage / speed / caption controls. The SCRIPT is throwaway-style tooling; the committed artifacts
// are Demo_Showcase.unity + the WebGL_ShowcaseDemo build (Builds/ is gitignored — the build stays local for the
// deploy step). Batchmode:
//   Unity -batchmode -projectPath NavSim -executeMethod ShowcaseDemoSetup.Build            -logFile -
//   Unity -batchmode -projectPath NavSim -executeMethod ShowcaseDemoSetup.BuildWebGL       -logFile -
//   Unity -batchmode -projectPath NavSim -executeMethod ShowcaseDemoSetup.RestoreStandalone -logFile -
public static class ShowcaseDemoSetup
{
    private const string BaseScene = "Assets/Scenes/Training_showcase.unity";
    private const string DemoScene = "Assets/Scenes/Demo_Showcase.unity";
    private const string ModelPath = "Assets/Models/Showcase/showcase_ext2.onnx"; // the ext2 dual-bar shipping policy
    private const string WebGLOut  = "Builds/WebGL_ShowcaseDemo";

    // Inference backend for the browser. Default == Burst == the CPU path, the WebGL-safe first choice (ComputeShader
    // needs compute shaders WebGL2 lacks; PixelShader is the GPU fragment fallback). If the browser console names an
    // unsupported backend, re-roll this to PixelShader and re-run Build()+BuildWebGL(). Do NOT report BLOCKED before
    // trying the device alternatives.
    private const InferenceDevice InferDevice = InferenceDevice.Default;

    public static void Build()
    {
        try
        {
            EditorSceneManager.OpenScene(BaseScene, OpenSceneMode.Single);
            var env = Object.FindAnyObjectByType<NavEnvironment>();
            var agent = Object.FindAnyObjectByType<NavAgent>();
            if (env == null || agent == null) { Debug.LogError("[ShowcaseDemo] missing env/agent in " + BaseScene); EditorApplication.Exit(1); return; }

            // (1) bake the showcase seed into BehaviorParameters + run inference with NO trainer.
            var bp = agent.GetComponent<BehaviorParameters>();
            var model = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelPath);
            if (bp == null || model == null)
            { Debug.LogError("[ShowcaseDemo] missing BehaviorParameters or model at " + ModelPath); EditorApplication.Exit(1); return; }
            bp.BehaviorName = "NavAgent";
            bp.Model = model;
            bp.InferenceDevice = InferDevice;
            bp.BehaviorType = BehaviorType.InferenceOnly;
            EditorUtility.SetDirty(bp);

            // (2) the agent's egocentric eye-cam keeps FEEDING the CNN sensor (and the UI PiP renders it on demand)
            // but must NOT draw to the game screen (it would fight the spectator camera). Disabling the Camera
            // component is the standard ml-agents idiom: CameraSensorComponent + the PiP call Camera.Render() on
            // demand, which works regardless of the enabled flag.
            var cs = agent.GetComponent<CameraSensorComponent>();
            Camera eyeCam = cs != null ? cs.Camera : null;
            if (eyeCam == null) { var t = agent.transform.Find("AgentEyeCam"); if (t != null) eyeCam = t.GetComponent<Camera>(); }
            if (eyeCam != null)
            {
                eyeCam.enabled = false;
                if (eyeCam.CompareTag("MainCamera")) eyeCam.tag = "Untagged"; // only the spectator is MainCamera
            }

            // (3) 3rd-person spectator camera the VISITOR sees. Reconfigure the base scene's non-eye camera or create
            // one. cullingMask = everything INCLUDING the Agent layer — the pixel arm culls the agent body from its
            // EYE cam, but the spectator must SEE the agent (and its trail). DemoCameraRig drives the pose each frame;
            // seed the editor-view transform with stage 0's hero pose so the saved scene looks right before Play.
            Camera spec = null;
            foreach (var c in Object.FindObjectsByType<Camera>(FindObjectsInactive.Exclude))
                if (c != eyeCam) { spec = c; break; }
            if (spec == null) spec = new GameObject("SpectatorCamera").AddComponent<Camera>();
            spec.gameObject.name = "SpectatorCamera";
            spec.tag = "MainCamera";
            spec.enabled = true;
            spec.depth = 0f;
            spec.cullingMask = ~0;                       // render everything, incl. the Agent layer (+ trail)
            spec.clearFlags = CameraClearFlags.Skybox;
            spec.fieldOfView = 55f;
            spec.nearClipPlane = 0.1f;
            spec.farClipPlane = 250f;
            CourseLayout stage0 = CourseSpec.Build(0, false, CourseVariant.NoLoop);
            spec.transform.position = stage0.CameraPos;
            spec.transform.LookAt(stage0.CameraLookAt);

            // (4) camera director: glides the spectator to the current stage's hero pose each frame.
            var rig = GetOrAdd<DemoCameraRig>(spec.gameObject);
            var rigSo = new SerializedObject(rig);
            rigSo.FindProperty("cam").objectReferenceValue = spec;
            rigSo.FindProperty("env").objectReferenceValue = env;
            rigSo.ApplyModifiedProperties();

            // (5) path trail on the agent. The agent GameObject is on the "Agent" layer, so the eye cam (which culls
            // that layer) never sees the trail — it shows ONLY on the spectator cam, exactly as intended. Reuse the
            // CourseBuilder's litShader (a real Standard-shader asset wired in the scene) so the gray material
            // survives WebGL shader-stripping; Shader.Find("Standard") is an Editor-only fallback that would NOT.
            var trail = GetOrAdd<TrailRenderer>(agent.gameObject);
            trail.time = 3f;
            trail.startWidth = 0.15f;
            trail.endWidth = 0f;
            trail.minVertexDistance = 0.1f;
            trail.numCapVertices = 2;
            trail.sharedMaterial = MakeTrailMaterial();

            // (6) presentation overlay wired to env + agent + eyeCam (all private [SerializeField]).
            var uiGo = GameObject.Find("DemoUI") ?? new GameObject("DemoUI"); // Find returns real null when absent, so ?? is safe here
            var ui = GetOrAdd<ShowcaseDemoUI>(uiGo);
            var uiSo = new SerializedObject(ui);
            uiSo.FindProperty("env").objectReferenceValue = env;
            uiSo.FindProperty("agent").objectReferenceValue = agent;
            uiSo.FindProperty("eyeCam").objectReferenceValue = eyeCam;
            uiSo.ApplyModifiedProperties();

            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            bool ok = EditorSceneManager.SaveScene(EditorSceneManager.GetActiveScene(), DemoScene, true); // saveAsCopy

            // (7) self-assert the regen produced a demo that will actually run.
            bool uiWired = uiSo.FindProperty("env").objectReferenceValue != null
                && uiSo.FindProperty("agent").objectReferenceValue != null
                && uiSo.FindProperty("eyeCam").objectReferenceValue != null;
            bool assertOk = ok
                && bp.Model != null
                && bp.BehaviorType == BehaviorType.InferenceOnly
                && spec.CompareTag("MainCamera")
                && (eyeCam != null && !eyeCam.enabled && !eyeCam.CompareTag("MainCamera"))
                && rigSo.FindProperty("cam").objectReferenceValue != null
                && rigSo.FindProperty("env").objectReferenceValue != null
                && uiWired
                && trail.sharedMaterial != null;
            Debug.Log("[ShowcaseDemo] saved=" + ok + " model=" + (bp.Model != null ? bp.Model.name : "NULL") +
                      " device=" + InferDevice + " behavior=" + bp.BehaviorType +
                      " eyeCam=" + (eyeCam != null ? eyeCam.name : "NULL") +
                      " spectator=" + spec.name + "(tag=" + spec.tag + ")" +
                      " uiWired=" + uiWired + " -> " + DemoScene + " assertOk=" + assertOk);
            EditorApplication.Exit(assertOk ? 0 : 1);
        }
        catch (System.Exception e)
        {
            Debug.LogError("[ShowcaseDemo] Build FAILED: " + e);
            EditorApplication.Exit(2);
        }
    }

    // Fake-null-safe get-or-add. The `GetComponent<T>() ?? AddComponent<T>()` shorthand is UNSAFE in Unity: `??` uses
    // reference equality and bypasses UnityEngine.Object's overloaded ==, so a fake-null returned by GetComponent for a
    // missing component (observed here for the built-in TrailRenderer under 6000.5) passes through as non-null and
    // AddComponent never runs — the next member access then throws MissingComponentException. The `!= null` below DOES
    // invoke the overload, so it correctly distinguishes an absent component from a live one.
    private static T GetOrAdd<T>(GameObject go) where T : Component
    {
        T existing = go.GetComponent<T>();
        return existing != null ? existing : go.AddComponent<T>();
    }

    // Matte light-gray trail material, built the CourseBuilder way (new Material(litShader) + matte Standard knobs).
    // Reads the CourseBuilder's serialized litShader so the material references the SAME real shader asset the course
    // uses — that survives a WebGL build; a runtime Shader.Find("Standard") is only the Editor/EditMode fallback.
    private static Material MakeTrailMaterial()
    {
        Shader shader = null;
        var course = Object.FindAnyObjectByType<CourseBuilder>();
        if (course != null)
            shader = new SerializedObject(course).FindProperty("litShader").objectReferenceValue as Shader;
        if (shader == null) shader = Shader.Find("Standard"); // Editor-only fallback (stripped from a player build)
        var mat = new Material(shader) { name = "ShowcaseTrail" };
        mat.SetColor("_Color", new Color(0.78f, 0.80f, 0.83f));
        mat.SetFloat("_Glossiness", 0.15f);
        mat.SetFloat("_Metallic", 0f);
        return mat;
    }

    // Build the WebGL player of Demo_Showcase into Builds/WebGL_ShowcaseDemo/. Passing `scenes` explicitly means the
    // demo scene does NOT need to be in EditorBuildSettings. ProjectSettings already carries the WebGL fixes.
    // RestoreStandalone (separate call) switches the target back.
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
            Debug.Log("[ShowcaseDemo] BuildWebGL result=" + summary.result + " errors=" + summary.totalErrors +
                      " size=" + summary.totalSize + " out=" + opts.locationPathName);
            EditorApplication.Exit(summary.result == BuildResult.Succeeded && summary.totalErrors == 0 ? 0 : 1);
        }
        catch (System.Exception e)
        {
            Debug.LogError("[ShowcaseDemo] BuildWebGL FAILED: " + e);
            EditorApplication.Exit(2);
        }
    }

    // Restore the active build target to StandaloneOSX (the project's default for training/eval players).
    public static void RestoreStandalone()
    {
        if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.StandaloneOSX)
            EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Standalone, BuildTarget.StandaloneOSX);
        Debug.Log("[ShowcaseDemo] restored active build target -> StandaloneOSX");
        EditorApplication.Exit(0);
    }
}
