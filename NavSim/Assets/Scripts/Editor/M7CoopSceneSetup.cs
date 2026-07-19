using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using NavSim.Runtime;

// M7 coop-arena scene builder (committed tooling, M6 pattern). Builds Assets/Scenes/Coop.unity FROM
// SCRATCH: floor, 4 outer walls, divider wall + doorway gap, door block child, plate zone, goal,
// 2 CoopAgents (CharacterController + 3 M5-layout ray fans, behavior "CoopAgent", vecObs 6, MaxStep 0)
// and the CoopArena wiring. ONE scene serves all three arms (arm_mode env-param selects routing).
// Adds tags door/plate to TagManager if missing (wall/goal/agent already exist). SELF-ASSERTS after
// building (vecObs==6, MaxStep==0 x2, fan tag sets exact, arena refs non-null) - hard-exit on failure.
// Batchmode:
//   Unity -batchmode -projectPath NavSim -executeMethod M7CoopSceneSetup.Build -logFile -
public static class M7CoopSceneSetup
{
    private const string ScenePath = "Assets/Scenes/Coop.unity";
    private const float ArenaHalf = 11f;   // M5 footprint (matches CoopArena.arenaHalf)
    private const float WallHeight = 3f;   // > jump apex (~1.2u) so neither wall nor door can be hopped
    private const float WallThickness = 0.5f;
    private const float DoorwayWidth = 3f; // gap in the divider at x=0; the door block fills it exactly
    private static readonly string[] FanTags = { "wall", "door", "plate", "goal", "agent" };

    public static void Build()
    {
        try
        {
            EnsureTags("door", "plate", "agent");
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            // --- Static world ---
            MakeCube("Floor", new Vector3(0f, -0.25f, 0f), new Vector3(2f * ArenaHalf, 0.5f, 2f * ArenaHalf), null);
            MakeCube("WallNorth", new Vector3(0f, WallHeight / 2f, ArenaHalf),
                new Vector3(2f * ArenaHalf + WallThickness, WallHeight, WallThickness), "wall");
            MakeCube("WallSouth", new Vector3(0f, WallHeight / 2f, -ArenaHalf),
                new Vector3(2f * ArenaHalf + WallThickness, WallHeight, WallThickness), "wall");
            MakeCube("WallEast", new Vector3(ArenaHalf, WallHeight / 2f, 0f),
                new Vector3(WallThickness, WallHeight, 2f * ArenaHalf + WallThickness), "wall");
            MakeCube("WallWest", new Vector3(-ArenaHalf, WallHeight / 2f, 0f),
                new Vector3(WallThickness, WallHeight, 2f * ArenaHalf + WallThickness), "wall");

            // Divider at z=0 with the doorway gap at x=0; the door block is a CHILD filling the gap.
            var divider = new GameObject("Divider");
            float segLen = ArenaHalf - DoorwayWidth / 2f;                       // 9.5
            float segCentre = (ArenaHalf + DoorwayWidth / 2f) / 2f;             // 6.25
            MakeCube("DividerLeft", new Vector3(-segCentre, WallHeight / 2f, 0f),
                new Vector3(segLen, WallHeight, WallThickness), "wall").transform.SetParent(divider.transform, true);
            MakeCube("DividerRight", new Vector3(segCentre, WallHeight / 2f, 0f),
                new Vector3(segLen, WallHeight, WallThickness), "wall").transform.SetParent(divider.transform, true);
            GameObject door = MakeCube("Door", new Vector3(0f, WallHeight / 2f, 0f),
                new Vector3(DoorwayWidth, WallHeight, WallThickness), "door");
            door.transform.SetParent(divider.transform, true);

            // Plate: a FLAT box zone (a squashed capsule/cylinder collider degenerates to a sphere dome -
            // a box stays flat, so agents step over it, rays still hit it). CoopArena re-places it per lesson.
            GameObject plate = MakeCube("Plate", new Vector3(2f, 0.02f, -1.5f), new Vector3(2.4f, 0.05f, 2.4f), "plate");

            // Goal: cylinder in the far chamber (CoopArena re-places it every episode).
            GameObject goal = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            goal.name = "Goal";
            goal.tag = "goal";
            goal.transform.position = new Vector3(0f, 0.5f, 6f);
            goal.transform.localScale = new Vector3(1f, 0.5f, 1f); // cylinder mesh is 2 tall -> 1u goal

            // --- Agents ---
            var agents = new CoopAgent[2];
            for (int i = 0; i < 2; i++) agents[i] = MakeAgent(i, new Vector3(i == 0 ? -2.5f : 2.5f, 0.5f, -7f));

            // --- Arena wiring (private [SerializeField] fields via SerializedObject, the M6 idiom) ---
            var arenaGo = new GameObject("CoopArena");
            var arena = arenaGo.AddComponent<CoopArena>();
            var so = new SerializedObject(arena);
            var agentsProp = so.FindProperty("agents");
            agentsProp.arraySize = 2;
            agentsProp.GetArrayElementAtIndex(0).objectReferenceValue = agents[0];
            agentsProp.GetArrayElementAtIndex(1).objectReferenceValue = agents[1];
            so.FindProperty("goal").objectReferenceValue = goal.transform;
            so.FindProperty("plate").objectReferenceValue = plate.transform;
            so.FindProperty("door").objectReferenceValue = door;
            so.ApplyModifiedPropertiesWithoutUndo();
            foreach (var a in agents) // agent -> arena backref
            {
                var aso = new SerializedObject(a);
                aso.FindProperty("arena").objectReferenceValue = arena;
                aso.ApplyModifiedPropertiesWithoutUndo();
            }

            // --- Light + spectator camera (scene hygiene; ray training needs neither) ---
            var light = new GameObject("Directional Light").AddComponent<Light>();
            light.type = LightType.Directional;
            light.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
            var cam = new GameObject("Main Camera").AddComponent<Camera>();
            cam.gameObject.tag = "MainCamera";
            cam.transform.position = new Vector3(0f, 18f, -16f);
            cam.transform.rotation = Quaternion.Euler(50f, 0f, 0f);

            // --- SELF-ASSERTS (hard-exit on failure; regen guard) ---
            var errors = new List<string>();
            foreach (var a in agents)
            {
                var bp = a.GetComponent<BehaviorParameters>();
                if (bp == null || bp.BehaviorName != "CoopAgent")
                    errors.Add($"{a.name}: behavior name != CoopAgent");
                if (bp == null || bp.BrainParameters.VectorObservationSize != 6)
                    errors.Add($"{a.name}: vecObs != 6");
                if (a.MaxStep != 0) errors.Add($"{a.name}: MaxStep != 0 (arena owns the boundary)");
                var fans = a.GetComponents<RayPerceptionSensorComponent3D>();
                if (fans.Length != 3) errors.Add($"{a.name}: fans={fans.Length} != 3");
                foreach (var f in fans)
                    if (!f.DetectableTags.SequenceEqual(FanTags))
                        errors.Add($"{a.name}/{f.SensorName}: tags [{string.Join(",", f.DetectableTags)}] " +
                                   $"!= [{string.Join(",", FanTags)}]");
            }
            var check = new SerializedObject(arena);
            if (check.FindProperty("agents").arraySize != 2
                || check.FindProperty("agents").GetArrayElementAtIndex(0).objectReferenceValue == null
                || check.FindProperty("agents").GetArrayElementAtIndex(1).objectReferenceValue == null
                || check.FindProperty("goal").objectReferenceValue == null
                || check.FindProperty("plate").objectReferenceValue == null
                || check.FindProperty("door").objectReferenceValue == null)
                errors.Add("arena refs incomplete (agents/goal/plate/door)");
            if (errors.Count > 0)
            {
                foreach (var e in errors) Debug.LogError("[M7Coop] ASSERT FAIL: " + e);
                EditorApplication.Exit(1);
                return;
            }

            bool saved = EditorSceneManager.SaveScene(EditorSceneManager.GetActiveScene(), ScenePath);
            Debug.Log($"[M7Coop] saved={saved} vecObs=6 agents=2 fans=3x2 tags=[{string.Join(",", FanTags)}] " +
                      $"maxStep=0 -> {ScenePath}");
            EditorApplication.Exit(saved ? 0 : 1);
        }
        catch (System.Exception e)
        {
            Debug.LogError("[M7Coop] FAILED: " + e);
            EditorApplication.Exit(2);
        }
    }

    private static GameObject MakeCube(string name, Vector3 pos, Vector3 scale, string tag)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
        go.name = name;
        go.transform.position = pos;
        go.transform.localScale = scale;
        if (tag != null) go.tag = tag;
        return go;
    }

    // A CoopAgent: empty root (tag "agent", CharacterController = the partner-perceivable collider) +
    // a collider-free capsule visual child + behavior/decision components + the M5 3-fan ray layout
    // (RayForward 6x90deg/15u, RayDown 3x70deg/8u, RayUp 3x70deg/15u) with the M7 tag vocabulary.
    private static CoopAgent MakeAgent(int index, Vector3 pos)
    {
        var go = new GameObject("CoopAgent_" + index) { tag = "agent" };
        go.transform.position = pos;

        var cc = go.AddComponent<CharacterController>();
        cc.height = 2f;
        cc.radius = 0.4f;
        cc.center = new Vector3(0f, 0.55f, 0f); // capsule bottom ~0.05 above the transform -> grounds cleanly

        var body = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        body.name = "Body";
        body.tag = "agent";
        Object.DestroyImmediate(body.GetComponent<CapsuleCollider>()); // the CharacterController IS the collider
        body.transform.SetParent(go.transform, false);
        body.transform.localPosition = new Vector3(0f, 0.55f, 0f);
        body.transform.localScale = new Vector3(0.8f, 1f, 0.8f);

        var agent = go.AddComponent<CoopAgent>();   // auto-adds BehaviorParameters
        agent.MaxStep = 0;                          // the ARENA owns the episode boundary (all arms)
        var bp = go.GetComponent<BehaviorParameters>();
        bp.BehaviorName = "CoopAgent";
        bp.BehaviorType = BehaviorType.Default;
        bp.BrainParameters.VectorObservationSize = 6;   // BuildCoop: proprioception 5 + doorOpen
        bp.BrainParameters.NumStackedVectorObservations = 1;
        bp.BrainParameters.ActionSpec = new ActionSpec(2, new[] { 2 }); // forward+turn, jump (NavAgent space)
        go.AddComponent<Unity.MLAgents.DecisionRequester>();            // defaults: period 5, act between

        AddFan(go, "RayForward", 6, 90f, 15f, 0.5f, 0.9f, 0.9f);
        AddFan(go, "RayDown", 3, 70f, 8f, 0.3f, 0.9f, -0.6f);
        AddFan(go, "RayUp", 3, 70f, 15f, 0.4f, 0.9f, 2.2f);
        return agent;
    }

    private static void AddFan(GameObject go, string name, int raysPerDirection, float maxRayDegrees,
        float rayLength, float sphereCastRadius, float startVerticalOffset, float endVerticalOffset)
    {
        var f = go.AddComponent<RayPerceptionSensorComponent3D>();
        f.SensorName = name;
        f.DetectableTags = new List<string>(FanTags);
        f.RaysPerDirection = raysPerDirection;
        f.MaxRayDegrees = maxRayDegrees;
        f.RayLength = rayLength;
        f.SphereCastRadius = sphereCastRadius;
        f.StartVerticalOffset = startVerticalOffset;
        f.EndVerticalOffset = endVerticalOffset;
    }

    // Append missing tags to TagManager.asset (SerializedObject idiom; M6 did the same for layers).
    private static void EnsureTags(params string[] tags)
    {
        var tm = AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TagManager.asset");
        if (tm.Length == 0) { Debug.LogError("[M7Coop] TagManager.asset not loadable"); EditorApplication.Exit(1); return; }
        var so = new SerializedObject(tm[0]);
        var tagsProp = so.FindProperty("tags");
        foreach (string tag in tags)
        {
            bool present = false;
            for (int i = 0; i < tagsProp.arraySize && !present; i++)
                present = tagsProp.GetArrayElementAtIndex(i).stringValue == tag;
            if (present) continue;
            tagsProp.InsertArrayElementAtIndex(tagsProp.arraySize);
            tagsProp.GetArrayElementAtIndex(tagsProp.arraySize - 1).stringValue = tag;
            Debug.Log("[M7Coop] added tag: " + tag);
        }
        so.ApplyModifiedPropertiesWithoutUndo();
        AssetDatabase.SaveAssets();
    }

    // StandaloneOSX training player (ray-only arena -> mlagents may run it --no-graphics). M6 pattern.
    //   Unity -batchmode -projectPath NavSim -executeMethod M7CoopSceneSetup.BuildPlayer -logFile -
    public static void BuildPlayer()
    {
        try
        {
            if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.StandaloneOSX)
                EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Standalone, BuildTarget.StandaloneOSX);
            var opts = new BuildPlayerOptions
            {
                scenes = new[] { ScenePath },
                locationPathName = System.IO.Path.GetFullPath(Application.dataPath + "/../Builds/M7CoopTrain.app"),
                target = BuildTarget.StandaloneOSX,
                options = BuildOptions.None
            };
            var summary = BuildPipeline.BuildPlayer(opts).summary;
            Debug.Log("[M7Coop] BuildPlayer result=" + summary.result + " errors=" + summary.totalErrors +
                      " out=" + opts.locationPathName);
            EditorApplication.Exit(summary.result == BuildResult.Succeeded ? 0 : 1);
        }
        catch (System.Exception e)
        {
            Debug.LogError("[M7Coop] BuildPlayer FAILED: " + e);
            EditorApplication.Exit(2);
        }
    }
}
