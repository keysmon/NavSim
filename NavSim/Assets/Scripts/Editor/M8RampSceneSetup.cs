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

// M8 "Cooperative Tool Use" platform-scene builder (a FORK of M7CoopSceneSetup, committed tooling).
// Builds Assets/Scenes/Ramp.unity (2 agents) / Ramp_solo.unity (1 agent) FROM SCRATCH:
//   - Floor + 4 outer walls (M5 footprint, half-size 11).
//   - A raised PLATFORM whose top face (y=2.0) is ABOVE the agent's jump apex
//     (jumpImpulse 7, gravity -20 -> apex ~1.23m), so the ledge cannot be hopped: the ramp is the
//     ONLY path to the goal (anti-leak geometry).
//   - The GOAL cylinder on the platform top.
//   - A wedge RAMP that is a DYNAMIC Rigidbody (PushableRampBody auto-adds it via [RequireComponent])
//     with a low-friction PhysicsMaterial on its collider (Assets/Models/M8/RampSlide.physicsMaterial),
//     tagged "ramp". Agents push it (real physics, via RampAgent.OnControllerColliderHit) then climb it.
//   - RampTarget (the marked spot the ramp must reach at the ledge base) + RampStart marker transforms.
//   - 1 or 2 RampAgents (CharacterController + the M5 3-fan ray layout, behavior "RampAgent", vecObs 6,
//     MaxStep 0) with the F2-critical per-agent `agentIndex` (0/1) wired via SerializedObject.
//   - RampArena wired via SerializedObject (agents/goal/ramp/rampTarget/rampStart + per-agent arena backref).
// FanTags = { wall, ramp, goal, agent } (M7's door/plate dropped, ramp added). EnsureTags adds any missing
// (only "ramp" is new on this project; wall/goal/agent pre-exist -> TagManager.asset gains the "ramp" tag).
// SELF-ASSERTS after building (behavior/vecObs/MaxStep/fans-with-ramp, agentIndex==slot, arena refs
// non-null, ramp = dynamic Rigidbody + PushableRampBody + non-null material, platform-top > jump apex) -
// hard-exit non-zero on any failure, Exit(0) on save.
//
// GEOMETRY/PHYSICS (resolved by verify-early #1 - the climb-vs-push tension is solved with SHAPE, not by
// relaxing the push rule): the push mechanic (RampAgent.OnControllerColliderHit) only shoves the ramp on
// ~vertical contacts (|contactNormal.y| <= 0.5). An up-slope ramp pushed from behind FAILS - the agent walks
// up the climbable top slope (|n.y|~1) and never touches a pushable face (verify-early #1: 0 pushes). The
// fix: LATERAL push. The slab tilts about the X axis, so its +-X end faces stay VERTICAL (|n.y|=0) and tall
// (floor->top) while its +z top face stays the walkable slope. Agents shove the ramp along +x against its -x
// vertical face (push rule fires cleanly) from BESIDE it - never on the climb slope - sliding it into the
// platform-base target (x=0). The +z slope is then climbed onto the platform AFTER placement. Un-embed:
// WedgeCenterY seats the lowest tilted corner ~on the floor (a corner below the floor jams under
// FreezePositionY). "1 creeps / 2 place" is a TEMPORAL forcing tuned via mass/damping/friction over the
// x-push distance (M8RampPhysicsSelftest measures it; final values baked into RampArena/PushableRampBody).
//
// Batchmode (NO -quit; the script calls EditorApplication.Exit itself):
//   Unity -batchmode -projectPath NavSim -executeMethod M8RampSceneSetup.Build     -logFile -
//   Unity -batchmode -projectPath NavSim -executeMethod M8RampSceneSetup.BuildSolo -logFile -
public static class M8RampSceneSetup
{
    private const string ScenePath = "Assets/Scenes/Ramp.unity";
    private const string SceneSoloPath = "Assets/Scenes/Ramp_solo.unity";
    private const string MaterialFolder = "Assets/Models/M8";
    private const string MaterialPath = "Assets/Models/M8/RampSlide.physicsMaterial";
    private const float ArenaHalf = 11f;     // M5 footprint (matches RampArena.arenaHalf)
    private const float WallHeight = 3f;     // > jump apex (~1.23u) so no wall can be hopped
    private const float WallThickness = 0.5f;
    private static readonly string[] FanTags = { "wall", "ramp", "goal", "agent" };

    // Platform (raised ledge): top face at y = PlatformCenterY + PlatformHeight/2 = 2.0 (above jump apex).
    // South edge z=3.9 sits just beyond the tilted ramp collider's swept +z bound (~3.86). Extending it to
    // z=3.5 made the platform overlap the ramp at the lateral-push start (their x faces already meet at x=-3),
    // pinning every light/heavy push at x=-5 despite valid contacts. The 0.04u collider clearance preserves
    // the push mechanic while still shortening the old walkable-top gap by ~0.1u.
    private static readonly Vector3 PlatformCenter = new Vector3(0f, 1.0f, 6.95f);
    private static readonly Vector3 PlatformSize = new Vector3(6f, 2.0f, 6.1f);

    // Wedge ramp (rotated cube slab). The slab tilts about the X axis, so its +z end is HIGH (walkable slope
    // faces the platform at +z) and - crucially for verify-early #1 - its +-X END FACES stay VERTICAL
    // (rotation about X leaves the (+-1,0,0) face normals horizontal, |n.y|=0). Those tall vertical side faces
    // are the PUSH surface: agents shove the ramp LATERALLY (+x) against its -x face (the existing |n.y|>0.5
    // push rule fires cleanly; keeps the placed-ramp-nudge guard closed), sliding it into the platform-base
    // target. The +z walkable slope stays free to climb onto the platform AFTER placement - push-face and
    // climb-face are orthogonal and both reachable (this resolves the climb-vs-push tension via SHAPE, not by
    // relaxing the rule). WedgeCenterY lifts the slab so its lowest corner rests ~ON the floor top (y=0), NOT
    // embedded (a corner below the floor jams under FreezePositionY - verify-early #1 finding).
    private const float WedgeAngleDeg = 30f;   // slope of the walkable top face
    private const float WedgeWidth = 4f;       // local x (pushable side-face width)
    private const float WedgeThickness = 0.5f; // local y (slab thickness)
    private const float WedgeLength = 4f;      // local z (run before rotation)
    private const float WedgeCenterY = 1.24f;  // lowest tilted corner (-1.2165 offset) rests ~on floor top (y~0.02)
    private const float RampStartX = -5f;      // ramp starts BESIDE the platform (-x); pushed +x to the target
    private const float RampBridgeZ = 2.0f;    // ramp z: +z slope faces the platform; z-footprint [0.14,3.86] < 4 (clear)

    // Public batchmode entrypoints. Build/BuildSolo share the world builder (identical geometry, differing
    // only in agent count) so the two scenes stay in lock-step and the material GUID is created once.
    public static void Build() => BuildInternal(ScenePath, 2);
    public static void BuildSolo() => BuildInternal(SceneSoloPath, 1);

    private static void BuildInternal(string scenePath, int agentCount)
    {
        try
        {
            EnsureTags("ramp", "agent");   // wall/goal pre-exist; "agent" pre-exists (idempotent); "ramp" is new
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            // --- Static world (floor + 4 outer walls, M7 footprint) ---
            MakeCube("Floor", new Vector3(0f, -0.25f, 0f), new Vector3(2f * ArenaHalf, 0.5f, 2f * ArenaHalf), null);
            MakeCube("WallNorth", new Vector3(0f, WallHeight / 2f, ArenaHalf),
                new Vector3(2f * ArenaHalf + WallThickness, WallHeight, WallThickness), "wall");
            MakeCube("WallSouth", new Vector3(0f, WallHeight / 2f, -ArenaHalf),
                new Vector3(2f * ArenaHalf + WallThickness, WallHeight, WallThickness), "wall");
            MakeCube("WallEast", new Vector3(ArenaHalf, WallHeight / 2f, 0f),
                new Vector3(WallThickness, WallHeight, 2f * ArenaHalf + WallThickness), "wall");
            MakeCube("WallWest", new Vector3(-ArenaHalf, WallHeight / 2f, 0f),
                new Vector3(WallThickness, WallHeight, 2f * ArenaHalf + WallThickness), "wall");

            // --- Raised platform: top surface ABOVE jump apex so it cannot be jumped (anti-leak). ---
            GameObject platform = MakeCube("Platform", PlatformCenter, PlatformSize, null);
            float platformTopY = PlatformCenter.y + PlatformSize.y / 2f;   // 2.0

            // Goal cylinder on the platform top (RampArena re-places it per episode; this is the scene seed).
            GameObject goal = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            goal.name = "Goal";
            goal.tag = "goal";
            // Goal moved close to where the slope delivers (z=5, near the platform south edge z=3.9) so reaching
            // it is a short walk after cresting - NOT the old razor 1.5u corridor at z=7 (verify-early cleared it
            // by only 0.07u; the Run-1 rollout never reached it). y=2.5 seats the 1u-tall cylinder on the y=2
            // platform instead of embedding it. Anti-leak: closest floor+jump-apex approach at the platform wall
            // (0,1.23,3.9) is ~1.68u from the goal > goalRadius 1.5.
            goal.transform.position = new Vector3(0f, 2.5f, 5f);
            goal.transform.localScale = new Vector3(1f, 0.5f, 1f);   // cylinder mesh is 2 tall -> 1u goal

            // --- Wedge ramp: DYNAMIC Rigidbody (PushableRampBody) + low-friction PhysicsMaterial. ---
            // The tilt lives on a CHILD slab (see MakeWedge): RampArena.ResetEpisode teleports the ramp BODY
            // to Quaternion.identity every reset (via PushableRampBody.ResetTo, NOT EvalMode-gated), which
            // would flatten a tilt baked on the body itself. The parent stays identity; the child stays tilted.
            GameObject rampGo = MakeWedge("Ramp", new Vector3(RampStartX, WedgeCenterY, RampBridgeZ));  // identity parent, lateral-push start
            PushableRampBody rampBody = rampGo.AddComponent<PushableRampBody>();        // [RequireComponent] auto-adds the Rigidbody on the parent
            PhysicsMaterial slide = LoadOrCreateSlideMaterial();                        // stable-GUID load-or-create
            Collider rampCol = rampGo.GetComponentInChildren<Collider>();              // the tilted child slab's BoxCollider
            rampCol.sharedMaterial = slide;

            // --- Marker transforms (lateral push: start beside the platform at -x, target aligned under the
            // goal at x=0; SAME y and z so the ramp-to-target distance is purely the x the agents must shove). ---
            Transform target = new GameObject("RampTarget").transform;
            target.position = new Vector3(0f, WedgeCenterY, RampBridgeZ);        // platform-base target (under the goal x)
            Transform start = new GameObject("RampStart").transform;
            start.position = new Vector3(RampStartX, WedgeCenterY, RampBridgeZ); // beside the platform, low corner ~on floor

            // --- Agents (1 solo, or 2 mirrored in the near chamber) ---
            var agents = new RampAgent[agentCount];
            for (int i = 0; i < agentCount; i++)
            {
                float x = agentCount == 1 ? 0f : (i == 0 ? -2.5f : 2.5f);
                agents[i] = MakeAgent(i, new Vector3(x, 0.5f, -7f));
            }

            // --- Arena wiring (private [SerializeField] fields via SerializedObject, the M6/M7 idiom) ---
            var arenaGo = new GameObject("RampArena");
            var arena = arenaGo.AddComponent<RampArena>();
            var so = new SerializedObject(arena);
            var agentsProp = so.FindProperty("agents");
            agentsProp.arraySize = agentCount;
            for (int i = 0; i < agentCount; i++)
                agentsProp.GetArrayElementAtIndex(i).objectReferenceValue = agents[i];
            so.FindProperty("goal").objectReferenceValue = goal.transform;
            so.FindProperty("ramp").objectReferenceValue = rampBody;
            so.FindProperty("rampTarget").objectReferenceValue = target;
            so.FindProperty("rampStart").objectReferenceValue = start;
            so.FindProperty("s0InitialPushDistance").floatValue = 1.75f;
            so.FindProperty("s0StartSuccesses").intValue = 200;
            so.ApplyModifiedPropertiesWithoutUndo();

            // Per-agent: arena back-ref + the F2-CRITICAL agentIndex (slot index -> the joint-push tracker bit;
            // if both stayed 0 the joint-push metric collapses to vacuous).
            for (int i = 0; i < agentCount; i++)
            {
                var aso = new SerializedObject(agents[i]);
                aso.FindProperty("arena").objectReferenceValue = arena;
                aso.FindProperty("agentIndex").intValue = i;
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

            // --- SELF-ASSERTS (hard-exit non-zero on failure; regen guard) ---
            var errors = new List<string>();

            // Anti-leak: platform top must clear the jump apex (~1.23m); guard at 1.5.
            if (platformTopY <= 1.5f)
                errors.Add($"platform top y={platformTopY} !> 1.5 (jump-apex anti-leak broken)");

            // Ramp = dynamic Rigidbody + PushableRampBody on the parent; a tilted child slab collider
            // (tag "ramp" + non-null slide material) whose tilt SURVIVES the arena's identity reset.
            if (rampGo.GetComponent<PushableRampBody>() == null) errors.Add("ramp missing PushableRampBody");
            var rampRb = rampGo.GetComponent<Rigidbody>();
            if (rampRb == null) errors.Add("ramp missing Rigidbody ([RequireComponent] failed)");
            else if (rampRb.isKinematic) errors.Add("ramp Rigidbody is kinematic (must be dynamic)");
            if (rampGo.tag != "ramp") errors.Add($"ramp tag='{rampGo.tag}' != ramp");
            var rampColChk = rampGo.GetComponentInChildren<Collider>();
            if (rampColChk == null)
                errors.Add("ramp has no collider (child slab collider missing)");
            else
            {
                if (rampColChk.sharedMaterial == null)
                    errors.Add("ramp collider sharedMaterial is null (PhysicsMaterial not wired)");
                if (rampColChk.gameObject.tag != "ramp")
                    errors.Add($"ramp slab tag='{rampColChk.gameObject.tag}' != ramp (rays would not detect it)");
                // Regression guard for the identity-reset flattening bug: the slope MUST live on the child,
                // not the reset-to-identity body. Confirm the child slab holds the intended local tilt.
                float tilt = Quaternion.Angle(rampColChk.transform.localRotation, Quaternion.identity);
                if (Mathf.Abs(tilt - WedgeAngleDeg) > 1f)
                    errors.Add($"ramp slab localTilt={tilt:F1}deg != {WedgeAngleDeg} " +
                               "(slope would flatten on ResetEpisode's identity reset)");
            }

            for (int i = 0; i < agentCount; i++)
            {
                var a = agents[i];
                var bp = a.GetComponent<BehaviorParameters>();
                if (bp == null || bp.BehaviorName != "RampAgent")
                    errors.Add($"{a.name}: behavior name != RampAgent");
                if (bp == null || bp.BrainParameters.VectorObservationSize != 6)
                    errors.Add($"{a.name}: vecObs != 6");
                if (a.MaxStep != 0) errors.Add($"{a.name}: MaxStep != 0 (arena owns the boundary)");
                var fans = a.GetComponents<RayPerceptionSensorComponent3D>();
                if (fans.Length != 3) errors.Add($"{a.name}: fans={fans.Length} != 3");
                foreach (var f in fans)
                {
                    if (!f.DetectableTags.SequenceEqual(FanTags))
                        errors.Add($"{a.name}/{f.SensorName}: tags [{string.Join(",", f.DetectableTags)}] " +
                                   $"!= [{string.Join(",", FanTags)}]");
                    if (!f.DetectableTags.Contains("ramp"))
                        errors.Add($"{a.name}/{f.SensorName}: DetectableTags missing 'ramp'");
                }
                // F2: read the serialized agentIndex back and confirm it equals the slot index.
                int ai = new SerializedObject(a).FindProperty("agentIndex").intValue;
                if (ai != i) errors.Add($"{a.name}: agentIndex={ai} != slot {i} (joint-push tracker would collapse)");
            }

            var check = new SerializedObject(arena);
            var checkAgents = check.FindProperty("agents");
            if (checkAgents.arraySize != agentCount)
                errors.Add($"arena agents arraySize={checkAgents.arraySize} != {agentCount}");
            else
                for (int i = 0; i < agentCount; i++)
                    if (checkAgents.GetArrayElementAtIndex(i).objectReferenceValue == null)
                        errors.Add($"arena agents[{i}] null");
            if (check.FindProperty("goal").objectReferenceValue == null) errors.Add("arena goal null");
            if (check.FindProperty("ramp").objectReferenceValue == null) errors.Add("arena ramp null");
            if (check.FindProperty("rampTarget").objectReferenceValue == null) errors.Add("arena rampTarget null");
            if (check.FindProperty("rampStart").objectReferenceValue == null) errors.Add("arena rampStart null");
            float s0InitialPushDistance = check.FindProperty("s0InitialPushDistance").floatValue;
            if (Mathf.Abs(s0InitialPushDistance - 1.75f) > 1e-5f)
                errors.Add($"arena s0InitialPushDistance={s0InitialPushDistance} != 1.75");
            int s0StartSuccesses = check.FindProperty("s0StartSuccesses").intValue;
            if (s0StartSuccesses != 200)
                errors.Add($"arena s0StartSuccesses={s0StartSuccesses} != 200");

            if (errors.Count > 0)
            {
                foreach (var e in errors) Debug.LogError("[M8Ramp] ASSERT FAIL: " + e);
                EditorApplication.Exit(1);
                return;
            }

            bool saved = EditorSceneManager.SaveScene(EditorSceneManager.GetActiveScene(), scenePath);
            Debug.Log($"[M8Ramp] saved={saved} agents={agentCount} platformTop={platformTopY} " +
                      $"wedge={WedgeAngleDeg}deg tags=[{string.Join(",", FanTags)}] vecObs=6 maxStep=0 " +
                      $"agentIndex=0..{agentCount - 1} material={MaterialPath} -> {scenePath}");
            EditorApplication.Exit(saved ? 0 : 1);
        }
        catch (System.Exception e)
        {
            Debug.LogError("[M8Ramp] FAILED: " + e);
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

    // Wedge ramp = an identity-rotation PARENT (holds the Rigidbody + PushableRampBody + tag "ramp") with a
    // TILTED CHILD slab (a scaled cube). The slope is a scaled cube rotated about X so its top face inclines
    // (walkable) and its low end presents a pushable ~vertical face; rotating by -angle raises the +z
    // (platform-facing) end (Unity is left-handed: +90 about X sends +Z -> down, so -angle sends +Z up).
    //
    // WHY parent/child (not a tilt baked on the body): RampArena.ResetEpisode calls
    // PushableRampBody.ResetTo(startPos, Quaternion.identity) every reset (and Start() resets immediately;
    // it is NOT EvalMode-gated). ResetTo does transform.SetPositionAndRotation(pos, identity), a hard
    // teleport that ignores FreezeRotation - so a tilt on the body itself FLATTENS to a horizontal slab on
    // the first frame of Play/eval, killing the climb. ResetTo only touches the parent transform, so the
    // child's localRotation tilt survives every reset. The child carries the collider (tag "ramp" + slide
    // material) so rays hit it and RampAgent.OnControllerColliderHit's GetComponentInParent<PushableRampBody>
    // resolves to the parent; child localPosition is zero so ramp.Position (== parent) still matches rampTarget.
    private static GameObject MakeWedge(string name, Vector3 center)
    {
        var parent = new GameObject(name) { tag = "ramp" };
        parent.transform.position = center;

        var slab = GameObject.CreatePrimitive(PrimitiveType.Cube);
        slab.name = name + "Slab";
        slab.tag = "ramp";                                  // RayPerceptionSensor reads the HIT collider's tag
        slab.transform.SetParent(parent.transform, false);
        slab.transform.localPosition = Vector3.zero;        // slab center == parent origin == ramp.Position
        slab.transform.localScale = new Vector3(WedgeWidth, WedgeThickness, WedgeLength);
        slab.transform.localRotation = Quaternion.Euler(-WedgeAngleDeg, 0f, 0f);  // survives the parent's identity reset
        return parent;
    }

    // A RampAgent: empty root (tag "agent", CharacterController = the partner-perceivable collider) + a
    // collider-free capsule visual child + behavior/decision components + the M5 3-fan ray layout
    // (RayForward 6x90deg/15u, RayDown 3x70deg/8u, RayUp 3x70deg/15u) with the M8 tag vocabulary
    // (wall/ramp/goal/agent). agentIndex is set by the caller via SerializedObject.
    private static RampAgent MakeAgent(int index, Vector3 pos)
    {
        var go = new GameObject("RampAgent_" + index) { tag = "agent" };
        go.transform.position = pos;

        var cc = go.AddComponent<CharacterController>();
        cc.height = 2f;
        cc.radius = 0.4f;
        cc.center = new Vector3(0f, 0.55f, 0f);   // capsule bottom ~0.05 above the transform -> grounds cleanly

        var body = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        body.name = "Body";
        body.tag = "agent";
        Object.DestroyImmediate(body.GetComponent<CapsuleCollider>());   // the CharacterController IS the collider
        body.transform.SetParent(go.transform, false);
        body.transform.localPosition = new Vector3(0f, 0.55f, 0f);
        body.transform.localScale = new Vector3(0.8f, 1f, 0.8f);

        var agent = go.AddComponent<RampAgent>();   // [RequireComponent(CharacterController)] satisfied; auto-adds BehaviorParameters
        agent.MaxStep = 0;                          // the ARENA owns the episode boundary (all arms)
        var bp = go.GetComponent<BehaviorParameters>();
        bp.BehaviorName = "RampAgent";
        bp.BehaviorType = BehaviorType.Default;
        bp.BrainParameters.VectorObservationSize = 6;   // BuildRamp: proprioception 5 + rampAtTarget
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

    // Load-or-create the low-friction slide material at 0.15/Minimum, and CONFIRM the values persist to disk.
    // verify-early #1 found the old mutate-then-SaveAssets left the on-disk asset at the 0.6/Average DEFAULT
    // (the field-set was not reaching disk). Fix: build the values into the constructor initializer BEFORE
    // CreateAsset (serializes the correct state immediately), and DELETE+recreate only when the existing asset
    // is missing or wrong - so a correct asset keeps its GUID stable across Build then BuildSolo, but a stale
    // one is repaired. SaveAssetIfDirty + ForceUpdate import guarantee the flush.
    private const float SlideFriction = 0.15f;   // lone agent can creep the heavy ramp; two place it in-budget
    private static PhysicsMaterial LoadOrCreateSlideMaterial()
    {
        if (!AssetDatabase.IsValidFolder(MaterialFolder))
            AssetDatabase.CreateFolder("Assets/Models", "M8");
        var mat = AssetDatabase.LoadAssetAtPath<PhysicsMaterial>(MaterialPath);
        bool wrong = mat == null
            || Mathf.Abs(mat.staticFriction - SlideFriction) > 1e-4f
            || Mathf.Abs(mat.dynamicFriction - SlideFriction) > 1e-4f
            || mat.frictionCombine != PhysicsMaterialCombine.Minimum;
        if (wrong)
        {
            if (mat != null) AssetDatabase.DeleteAsset(MaterialPath);   // repair a stale/default asset
            mat = new PhysicsMaterial("RampSlide")
            {
                staticFriction = SlideFriction,
                dynamicFriction = SlideFriction,
                frictionCombine = PhysicsMaterialCombine.Minimum,   // NOT Average - a lighter combine keeps the push feasible
                bounciness = 0f
            };
            AssetDatabase.CreateAsset(mat, MaterialPath);              // serializes the correct fields immediately
            EditorUtility.SetDirty(mat);
            AssetDatabase.SaveAssetIfDirty(mat);
            AssetDatabase.ImportAsset(MaterialPath, ImportAssetOptions.ForceUpdate);
        }
        return mat;
    }

    // Append missing tags to TagManager.asset (SerializedObject idiom; M7). Idempotent (skips present tags).
    private static void EnsureTags(params string[] tags)
    {
        var tm = AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TagManager.asset");
        if (tm.Length == 0) { Debug.LogError("[M8Ramp] TagManager.asset not loadable"); EditorApplication.Exit(1); return; }
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
            Debug.Log("[M8Ramp] added tag: " + tag);
        }
        so.ApplyModifiedPropertiesWithoutUndo();
        AssetDatabase.SaveAssets();
    }

    // StandaloneOSX training players (ray-only arena -> mlagents may run --no-graphics). M7 pattern.
    // Phase 4 (verify-early) runs these; Phase 3.1 only defines them.
    //   Unity -batchmode -projectPath NavSim -executeMethod M8RampSceneSetup.BuildPlayer     -logFile -
    //   Unity -batchmode -projectPath NavSim -executeMethod M8RampSceneSetup.BuildPlayerSolo -logFile -
    public static void BuildPlayer() => BuildPlayerInternal(ScenePath, "Builds/M8RampTrain.app");
    public static void BuildPlayerSolo() => BuildPlayerInternal(SceneSoloPath, "Builds/M8RampSolo.app");

    private static void BuildPlayerInternal(string scenePath, string outRelative)
    {
        try
        {
            if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.StandaloneOSX)
                EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Standalone, BuildTarget.StandaloneOSX);
            var opts = new BuildPlayerOptions
            {
                scenes = new[] { scenePath },
                locationPathName = System.IO.Path.GetFullPath(Application.dataPath + "/../" + outRelative),
                target = BuildTarget.StandaloneOSX,
                options = BuildOptions.None
            };
            var summary = BuildPipeline.BuildPlayer(opts).summary;
            Debug.Log("[M8Ramp] BuildPlayer result=" + summary.result + " errors=" + summary.totalErrors +
                      " out=" + opts.locationPathName);
            EditorApplication.Exit(summary.result == BuildResult.Succeeded ? 0 : 1);
        }
        catch (System.Exception e)
        {
            Debug.LogError("[M8Ramp] BuildPlayer FAILED: " + e);
            EditorApplication.Exit(2);
        }
    }
}
