using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Policies;
using NavSim.Runtime;

// Phase-3 pixel-arm scene builder. Opens the M5 Training scene, converts the agent from ray-fans to an
// egocentric RGB CameraSensor, wires env.agentCamera, fixes the vector-obs size to 8, and saves as
// Training_pixel.unity. Re-runnable (tune camera pose params below + re-run). The SCRIPT is throwaway tooling;
// the committed artifact is Training_pixel.unity.  Batchmode:
//   Unity -batchmode -projectPath NavSim -executeMethod M6PixelSceneSetup.Build -logFile -
public static class M6PixelSceneSetup
{
    // Camera pose (tune for coverage): egocentric head cam, tilted down so feet-level pits are visible.
    private const float EyeHeight = 1.5f;
    private const float EyeForward = 0.2f;
    private const float PitchDownDeg = 12f;
    private const float Fov = 90f;        // vertical FOV; square 84x84 aspect -> H ~= V. Wide-ish to approach the ray fan.
    private const float FarClip = 15f;    // == DifficultyMapper.SightRange

    public static void Build()
    {
        try
        {
            EditorSceneManager.OpenScene("Assets/Scenes/Training.unity", OpenSceneMode.Single);
            var env = Object.FindAnyObjectByType<NavEnvironment>();
            var agent = Object.FindAnyObjectByType<NavAgent>();
            if (env == null || agent == null) { Debug.LogError("[M6PixelScene] missing env/agent in Training.unity"); EditorApplication.Exit(1); return; }

            // (1) strip the M5 ray fans — the pixel arm perceives the world via the camera.
            int removed = 0;
            foreach (var r in agent.GetComponents<RayPerceptionSensorComponent3D>()) { Object.DestroyImmediate(r); removed++; }

            // (2) egocentric RGB camera (reuse an existing "AgentEyeCam" child if re-run, else create one).
            Transform camT = agent.transform.Find("AgentEyeCam");
            GameObject camGO = camT != null ? camT.gameObject : new GameObject("AgentEyeCam");
            camGO.transform.SetParent(agent.transform, false);
            camGO.transform.localPosition = new Vector3(0f, EyeHeight, EyeForward);
            camGO.transform.localRotation = Quaternion.Euler(PitchDownDeg, 0f, 0f);
            Camera cam = camGO.GetComponent<Camera>();       // NOTE: use Unity's ==, not ?? (fake-null footgun)
            if (cam == null) cam = camGO.AddComponent<Camera>();
            cam.fieldOfView = Fov;
            cam.nearClipPlane = 0.05f;
            cam.farClipPlane = FarClip;
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.4f, 0.55f, 0.75f); // flat sky so "no goal in frame" isn't pure black

            // Hide the agent's OWN body from its eye-cam: the camera sits near the capsule top, so without this
            // the CNN sees a large constant patch of the agent's own mesh. Put the agent hierarchy on an "Agent"
            // layer and cull it from the eye-cam ONLY — the agent stays visible to the demo's 3rd-person camera.
            int agentLayer = LayerMask.NameToLayer("Agent");
            if (agentLayer < 0)
            {
                var tmObjs = AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TagManager.asset");
                if (tmObjs.Length > 0)
                {
                    var tmso = new SerializedObject(tmObjs[0]);
                    var layersProp = tmso.FindProperty("layers");
                    if (layersProp != null && layersProp.arraySize > 9)
                    { layersProp.GetArrayElementAtIndex(9).stringValue = "Agent"; tmso.ApplyModifiedProperties(); }
                }
                agentLayer = 9;
            }
            SetLayerRecursively(agent.gameObject, agentLayer);
            cam.cullingMask = ~(1 << agentLayer); // render everything EXCEPT the agent's own body

            // (3) stock RGB CameraSensorComponent (reuse if present).
            var cs = agent.GetComponent<CameraSensorComponent>();
            if (cs == null) cs = agent.gameObject.AddComponent<CameraSensorComponent>();
            cs.Camera = cam;
            cs.SensorName = "AgentCam";
            cs.Width = 84;
            cs.Height = 84;
            cs.Grayscale = false;

            // (4) BehaviorParameters: vector obs is now 8 (proprioception 5 + RGB cue 3); camera obs is separate.
            var bp = agent.GetComponent<BehaviorParameters>();
            if (bp != null)
            {
                bp.BehaviorName = "NavAgent";
                bp.BehaviorType = BehaviorType.Default;
                bp.BrainParameters.VectorObservationSize = 8;
            }

            // (5) wire env.agentCamera == the sensor camera (private [SerializeField]).
            var so = new SerializedObject(env);
            so.FindProperty("agentCamera").objectReferenceValue = cam;
            so.FindProperty("tagGoalsByColor").boolValue = false;
            so.ApplyModifiedProperties();

            var target = EditorSceneManager.GetActiveScene();
            bool ok = EditorSceneManager.SaveScene(target, "Assets/Scenes/Training_pixel.unity", true); // saveAsCopy
            Debug.Log("[M6PixelScene] saved=" + ok + " removedRayFans=" + removed +
                      " cam(fov=" + Fov + ", pitch=" + PitchDownDeg + ", far=" + FarClip + ") -> Assets/Scenes/Training_pixel.unity");
            EditorApplication.Exit(ok ? 0 : 1);
        }
        catch (System.Exception e)
        {
            Debug.LogError("[M6PixelScene] FAILED: " + e);
            EditorApplication.Exit(2);
        }
    }

    private static void SetLayerRecursively(GameObject go, int layer)
    {
        go.layer = layer;
        foreach (Transform c in go.transform) SetLayerRecursively(c.gameObject, layer);
    }

    // Build a StandaloneOSX player of Training_pixel for headless mlagents training (WITH graphics so the camera
    // renders). Used for the real-terrain throughput measurement + the real training runs.
    public static void BuildPlayer()
    {
        try
        {
            if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.StandaloneOSX)
                EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Standalone, BuildTarget.StandaloneOSX);
            var opts = new BuildPlayerOptions
            {
                scenes = new[] { "Assets/Scenes/Training_pixel.unity" },
                locationPathName = System.IO.Path.GetFullPath(Application.dataPath + "/../Builds/M6PixelTrain.app"),
                target = BuildTarget.StandaloneOSX,
                options = BuildOptions.None
            };
            var summary = BuildPipeline.BuildPlayer(opts).summary;
            Debug.Log("[M6PixelScene] BuildPlayer result=" + summary.result + " errors=" + summary.totalErrors +
                      " out=" + opts.locationPathName);
            EditorApplication.Exit(summary.result == BuildResult.Succeeded ? 0 : 1);
        }
        catch (System.Exception e)
        {
            Debug.LogError("[M6PixelScene] BuildPlayer FAILED: " + e);
            EditorApplication.Exit(2);
        }
    }
}
