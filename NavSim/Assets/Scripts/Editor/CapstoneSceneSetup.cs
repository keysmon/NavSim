using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

// CapstoneSceneSetup - builds the NavSim capstone showcase stage: a curated, hand-authored
// challenge course that demonstrates the M6 visual object-goal search task on legible terrain, and
// renders a 3/4 hero shot of it. Build() composes the scene and saves it to Assets/Scenes/Capstone.unity;
// Shot() opens that scene and renders the HeroCamera to a preview PNG.
//
// The stage is a single bent "critical path" spine, spawn (near, Z~1) -> red goal (far, Z~38), read as
// four escalating beats along +Z:
//   1. Orientation (Z -4..10, flat, y=0): an open start with a small spawn marker.
//   2. Ramp -> raised ledge (Z 9..15 climbing to y=2.8, landing Z 15..19): ONE tan, tread-striped ramp
//      up to a genuinely elevated (~2.8u) gray ledge - the tread stripes make "walk up here" read from
//      silhouette alone.
//   3. Occluding turn (slate wall @ Z=22 on Deck A, Z 19..28): the wall spans the centerline
//      (X -2.5..1.5), so the agent's eye-level line of sight to the goal is blocked and it must detour
//      through the open flank (X 1.5..6).
//   4. Risk/reward gap + reveal (Z 27..41): a dark jump-pit (center/left) versus a safe loop ledge
//      (right flank), then the lit red goal pedestal flanked by duller blue and yellow decoys off the
//      critical path - the red-target-among-decoys reveal of the visual-search task.
//
// Visual grammar is EXCLUSIVE per surface type so the course is legible at a glance: tan + tread
// stripes = climbable ramp ONLY, slate = wall/occluder ONLY, near-black = pit ONLY.
//
// Two cameras, by design:
//   - HeroCamera (tagged MainCamera): the showcase shot. Elevated and pulled back off the spine's long
//     axis (a proper 3/4 side-on angle, viewed from the wall's open +X flank) so the ramp rise and
//     ledge elevation read as real 3D and the camera sees past the wall into the reveal chamber.
//   - AgentEyeCamera (disabled): a reference point at agent eye-height (deck-top + 1.5u) used ONLY to
//     verify, by raycast, that the wall actually blocks the gameplay line of sight to the goal. The
//     wall is sized/placed so the low centerline agent-eye ray (X=0 the whole way, straight through the
//     wall's X range) is blocked, while the high, offset hero ray clears the wall's left edge well
//     before the goal.
//
// Build() asserts, by raycast (after Physics.SyncTransforms(), since no play-mode physics step runs in
// batchmode), that the wall occludes the goal from the agent eye AND that the hero camera has a clear
// line to the goal, and exits non-zero if either check or the save fails.
// Batchmode:
//   Unity -batchmode -projectPath NavSim -executeMethod CapstoneSceneSetup.Build -logFile -
//   Unity -batchmode -projectPath NavSim -executeMethod CapstoneSceneSetup.Shot  -logFile -   (NO -nographics)
public static class CapstoneSceneSetup
{
    private const string ScenePath = "Assets/Scenes/Capstone.unity";
    private const string ShotPath =
        "/private/tmp/claude-501/-Users-hangruan-Documents-claude-code-repo-unity/61e6579f-2a99-4eaf-b2c1-e66ccb222fee/scratchpad/capstone.png";

    public static void Build()
    {
        try
        {
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            // Materials (Standard shader - Built-in RP project, confirmed via Assets/Materials/NavColor0.mat
            // and Packages/manifest.json having no com.unity.render-pipelines.*). Visual grammar is EXCLUSIVE
            // per surface type: tan+stripe = climbable ONLY, slate = wall/occluder ONLY, near-black = pit ONLY.
            var floorMat = MakeMat("Mat_Floor", new Color(0.55f, 0.55f, 0.58f), 0.15f);
            // A hair lighter than the floor so the elevated ledge/pedestal read as "the raised deck",
            // but kept in the same neutral-gray family so the goal point light does not blow the top
            // deck out to white.
            var landingMat = MakeMat("Mat_Landing", new Color(0.60f, 0.60f, 0.63f), 0.15f);
            var curbMat = MakeMat("Mat_Curb", new Color(0.30f, 0.33f, 0.37f), 0.15f);
            var rampMat = MakeMat("Mat_Ramp", new Color(0.82f, 0.58f, 0.28f), 0.20f);        // EXCLUSIVE climbable accent
            var treadMat = MakeMat("Mat_Tread", new Color(0.52f, 0.33f, 0.12f), 0.15f);       // dark tread stripes on ramp
            var wallMat = MakeMat("Mat_Wall", new Color(0.35f, 0.42f, 0.52f), 0.12f);         // flat matte slate, EXCLUSIVE occluder
            var pitMat = MakeMat("Mat_Pit", new Color(0.05f, 0.05f, 0.06f), 0.05f);           // near-black void
            var pedestalMat = MakeMat("Mat_Pedestal", new Color(0.62f, 0.62f, 0.65f), 0.20f); // neutral gray, in the deck family
            var decoyPadMat = MakeMat("Mat_DecoyPad", new Color(0.50f, 0.50f, 0.53f), 0.10f); // dull, unlit-looking pad
            var markerMat = MakeMat("Mat_Spawn", new Color(0.74f, 0.76f, 0.79f), 0.10f);
            var redMat = MakeMat("Mat_GoalRed", new Color(0.85f, 0.10f, 0.10f), 0.25f);       // GoalPalette.Colors[0]
            var blueMat = MakeMat("Mat_GoalBlue", new Color(0.15f, 0.30f, 0.90f), 0.25f);      // GoalPalette.Colors[2]
            var yellowMat = MakeMat("Mat_GoalYellow", new Color(0.90f, 0.80f, 0.10f), 0.25f);  // GoalPalette.Colors[3]

            // === Zone 1: Orientation (Z -4..10, y=0) - flat open start. ===
            MakeCube("Floor_Zone1", new Vector3(0f, -0.15f, 3f), new Vector3(12f, 0.3f, 14f), floorMat, null);

            // Small + off the camera's direct forward sightline so the near marker does not dominate
            // the frame.
            GameObject spawn = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            spawn.name = "SpawnMarker";
            spawn.transform.position = new Vector3(-2f, 0.3f, 2.5f);
            spawn.transform.localScale = new Vector3(0.5f, 0.5f, 0.5f);
            spawn.GetComponent<Renderer>().sharedMaterial = markerMat;

            // Exactly 3 goal spheres exist in the whole scene - Goal_Target_Red + Goal_Decoy_Blue +
            // Goal_Decoy_Yellow - so the reveal reads as one target among two decoys, no extra balls.

            // === Zone 2: Ramp -> RAISED LEDGE. Rises Z 9->15, y 0->2.8 - a genuine ~2.8u elevation
            // change over a 6u run (~25deg slope) so the climb reads as real, not near-flat. ===
            Vector3 rampStart = new Vector3(0f, 0f, 9f);
            Vector3 rampEnd = new Vector3(0f, 2.8f, 15f);
            Vector3 rampDir = rampEnd - rampStart;
            Quaternion rampRot = Quaternion.FromToRotation(Vector3.forward, rampDir.normalized);
            MakeRamp("Ramp", (rampStart + rampEnd) / 2f, rampRot, rampDir.magnitude, 10f, 0.6f, rampMat);
            MakeTreadStripes("Ramp", rampStart, rampEnd, rampRot, 9f, 0.6f, 4, treadMat);

            const float topY = 2.8f; // the raised-ledge / deck elevation carried through zones 2-4

            // Brighter landing right at the top of the ramp (Z 15..19) - the raised ledge itself, pulls
            // the eye up the incline.
            MakeCube("Landing", new Vector3(0f, topY - 0.15f, 17f), new Vector3(12f, 0.3f, 4f), landingMat, null);

            // Deck A (Z 19..27, top y=2.8) - carries the elevation into the occluding-wall zone. The far
            // edge stops at Z=27 to leave room for the jump-pit below (the wall at Z=22 keeps 5u/4u of
            // clearance on either side - no overlap risk).
            MakeCube("DeckA", new Vector3(0f, topY - 0.15f, 23f), new Vector3(12f, 0.3f, 8f), floorMat, null);
            MakeCube("Curb_DeckA_L", new Vector3(-6.15f, topY + 0.25f, 23f), new Vector3(0.3f, 0.5f, 8f), curbMat, null);
            MakeCube("Curb_DeckA_R", new Vector3(6.15f, topY + 0.25f, 23f), new Vector3(0.3f, 0.5f, 8f), curbMat, null);

            // === Zone 3: Occluding turn. Wall spans X -2.5..1.5 (centerline, Z=22 on Deck A, just
            // after the ledge) - covers the AGENT EYE's centerline path (X=0 the whole way) with margin
            // on both sides, leaving an open flank X 1.5..6 for the agent's detour. The span is kept
            // narrow BY DESIGN so the elevated/offset hero camera can clear its left edge (see the
            // occlusion note in the header comment + CheckHeroClearLOS below). ===
            GameObject wall = MakeCube("OccludingWall", new Vector3(-0.5f, topY + 1.75f, 22f), new Vector3(4f, 3.5f, 0.8f), wallMat, "wall");

            // === Zone 4: risk/reward gap (Z 27..33) then reveal (Z 33..42). ===
            // The jump gap is a full 6u (Z 27..33, centered on Z=30) so it reads as a real gap in the
            // hero frame, not a thin notch. Jump-pit: X -6..2 (void, no deck slab). The pit floor sits
            // at the void depth (-0.45, independent of the raised deck), so the pit walls span
            // floor-to-deck-top without a gap.
            MakeCube("PitFloor", new Vector3(-2f, -0.45f, 30f), new Vector3(8f, 0.3f, 6f), pitMat, null);
            float pitWallHeight = topY - (-0.45f);
            float pitWallCenterY = (-0.45f + topY) / 2f;
            MakeCube("PitWall_Near", new Vector3(-2f, pitWallCenterY, 27f), new Vector3(8f, pitWallHeight, 0.3f), pitMat, null);
            MakeCube("PitWall_Far", new Vector3(-2f, pitWallCenterY, 33f), new Vector3(8f, pitWallHeight, 0.3f), pitMat, null);

            // A low curb lip right at the jump-off edge (same curbMat family as the deck curbs - no new
            // visual grammar) so the "brink" reads crisply in silhouette instead of the void just
            // flush-blending into the deck surface.
            MakeCube("Curb_PitLip", new Vector3(-2f, topY + 0.05f, 26.85f), new Vector3(8f, 0.3f, 0.3f), curbMat, null);

            // Safe loop-around ledge: X 2..6, same deck height, Z 27..33 (alongside the pit) - the
            // readable, no-hidden-cost alternative to the risky direct jump across the pit.
            MakeCube("SafeLedge", new Vector3(4f, topY - 0.15f, 30f), new Vector3(4f, 0.3f, 6f), floorMat, null);
            // A rail along the safe ledge's pit-facing edge - marks the walkway against the void so the
            // "narrow safe loop beside the pit" beat is unambiguous from the hero angle.
            MakeCube("Curb_SafeLedgeRail", new Vector3(1.85f, topY + 0.25f, 30f), new Vector3(0.3f, 0.5f, 6f), curbMat, null);

            // Deck B / reveal chamber (Z 33..42, top y=2.8).
            MakeCube("DeckB", new Vector3(0f, topY - 0.15f, 37.5f), new Vector3(12f, 0.3f, 9f), floorMat, null);
            MakeCube("Curb_DeckB_L", new Vector3(-6.15f, topY + 0.25f, 37.5f), new Vector3(0.3f, 0.5f, 9f), curbMat, null);
            MakeCube("Curb_DeckB_R", new Vector3(6.15f, topY + 0.25f, 37.5f), new Vector3(0.3f, 0.5f, 9f), curbMat, null);

            // Lit goal pedestal (the landmark) - centerline at Z=38, directly behind the wall in X so
            // the wall must actually occlude it from the agent's eye. Red target on top.
            MakeCube("GoalPedestal", new Vector3(0f, 3.3f, 38f), new Vector3(2.2f, 1.3f, 2.2f), pedestalMat, null);
            Vector3 goalPos = new Vector3(0f, 4.95f, 38f);
            GameObject goalGo = MakeGoal("Goal_Target_Red", goalPos, redMat, "goal", 2f);

            // A gentle point light that highlights the goal/pedestal without washing out the
            // surrounding deck.
            var goalLightGo = new GameObject("GoalPedestalLight");
            goalLightGo.transform.position = new Vector3(0f, 6.2f, 37f);
            var goalLight = goalLightGo.AddComponent<Light>();
            goalLight.type = LightType.Point;
            goalLight.color = new Color(1f, 0.85f, 0.80f);
            goalLight.intensity = 1.3f;
            goalLight.range = 7f;

            // Duller decoys, low pads (no pedestal glow, no dedicated light), off the critical path.
            // Pushed out in X (+/-5.3) and offset in Z (3u each side) from the goal so they read as side
            // alcoves with breathing room around the pedestal, not a crowded cluster.
            MakeCube("DecoyPad_Blue", new Vector3(-5.3f, 2.9f, 35f), new Vector3(1.4f, 0.4f, 1.4f), decoyPadMat, null);
            MakeGoal("Goal_Decoy_Blue", new Vector3(-5.3f, 3.8f, 35f), blueMat, "goal", 2f);
            MakeCube("DecoyPad_Yellow", new Vector3(5.3f, 2.9f, 41f), new Vector3(1.4f, 0.4f, 1.4f), decoyPadMat, null);
            MakeGoal("Goal_Decoy_Yellow", new Vector3(5.3f, 3.8f, 41f), yellowMat, "goal", 2f);

            // --- Lighting: balanced directional + ambient fill (no blown-white surfaces, nothing crushed to black). ---
            var lightGo = new GameObject("Directional Light");
            var light = lightGo.AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1.0f;
            light.color = new Color(1f, 0.98f, 0.94f);
            light.shadows = LightShadows.Soft;
            light.shadowStrength = 0.65f;
            lightGo.transform.rotation = Quaternion.Euler(45f, -35f, 0f);

            RenderSettings.ambientMode = AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(0.42f, 0.44f, 0.47f);
            RenderSettings.skybox = null;

            // --- HERO camera, separate from the agent-eye reference point. A 3/4 angle - elevated and
            // pitched down - so the ramp incline and ledge elevation read as real 3D (a head-on angle
            // flattens both). Frames the whole spine (spawn Z~1 -> goal Z38) plus the goal pedestal, and
            // is NOT subject to the occluder (see the clearance note on the wall above and
            // CheckHeroClearLOS below). ---
            var heroGo = new GameObject("HeroCamera");
            heroGo.tag = "MainCamera";
            var heroCam = heroGo.AddComponent<Camera>();
            // Camera placement rationale (the level spans ~47u in Z, a long corridor):
            //   - Pulled well back with a narrow-ish FOV (45) so near/far scale stays even and the whole
            //     spine reads at a consistent size, avoiding wide-angle near-field foreshortening.
            //   - Viewed at a wide angle OFF the spine's long axis (a proper 3/4 side-on view, not a
            //     shallow behind-the-shoulder one) so the corridor's length reads diagonally across the
            //     frame instead of collapsing toward a single vanishing point.
            //   - Placed on the wall's open +X flank (where the agent detours, X 1.5..6) so the camera
            //     sees PAST the wall into the reveal chamber rather than looking through/around its bulk;
            //     the look-at target is pushed down-spine so beat 4 (pit / safe-ledge / reveal) gets
            //     proportionate frame area, not compressed behind the wall.
            heroGo.transform.position = new Vector3(32f, 15f, -8f);
            heroGo.transform.LookAt(new Vector3(0f, 2.5f, 27f));
            heroCam.fieldOfView = 45f;
            heroCam.nearClipPlane = 0.1f;
            heroCam.farClipPlane = 120f;
            heroCam.clearFlags = CameraClearFlags.SolidColor;
            heroCam.backgroundColor = new Color(0.72f, 0.75f, 0.80f); // soft neutral sky, no skybox texture needed

            // --- AgentEyeCamera - a reference point/camera at agent eye-height (deck-top + 1.5u),
            // independent of the hero cam, used ONLY to verify the occluder blocks the gameplay line of
            // sight. Not tagged MainCamera; not used for the hero screenshot. ---
            Vector3 agentEye = new Vector3(0f, topY + 1.5f, 19f); // right where the agent crests the ledge, before the wall
            var eyeGo = new GameObject("AgentEyeCamera");
            var eyeCam = eyeGo.AddComponent<Camera>();
            eyeCam.enabled = false; // reference point only - not rendered for the hero shot
            eyeGo.transform.position = agentEye;
            eyeGo.transform.LookAt(goalPos);

            // --- Occlusion verification (the thing that matters for GAMEPLAY): the agent-eye point
            // must NOT see the goal - the wall must be in between. Auto-enlarge + retry if the first
            // check fails (guards against any hand-authored geometry mistake). ---
            bool agentBlocked = false;
            for (int attempt = 0; attempt < 4 && !agentBlocked; attempt++)
            {
                // Colliders' PhysX-side poses only reflect script-driven transform changes after an
                // explicit sync (no play-mode physics step is running here in batchmode) - without this,
                // Physics.Raycast queries stale (pre-move) collider poses and reports no hits at all.
                Physics.SyncTransforms();
                agentBlocked = CheckOcclusion(agentEye, goalPos, wall, attempt);
                if (!agentBlocked && attempt < 3)
                {
                    Vector3 s = wall.transform.localScale;
                    wall.transform.localScale = new Vector3(s.x + 1.5f, s.y + 1.0f, s.z);
                    Vector3 p = wall.transform.position;
                    wall.transform.position = new Vector3(p.x, p.y + 0.5f, p.z);
                    Debug.Log("[CapstoneSceneSetup] wall too small on attempt " + attempt + " -> enlarged to scale=" + wall.transform.localScale);
                }
            }
            Debug.Log("[CapstoneSceneSetup] FINAL agent-eye occlusion result: wallBlocksGoalFromAgentEye=" + agentBlocked);

            // --- Hero visibility check (the thing that matters for the SCREENSHOT): the hero camera
            // must have a CLEAR line of sight to the goal - nothing (wall, deck, pedestal) in between,
            // so the goal is never hidden behind the occluder in the hero frame. ---
            Physics.SyncTransforms();
            bool heroClear = CheckClearLOS(heroGo.transform.position, goalPos, goalGo, "hero");
            Debug.Log("[CapstoneSceneSetup] FINAL hero-camera visibility result: heroSeesGoal=" + heroClear);

            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            bool ok = EditorSceneManager.SaveScene(EditorSceneManager.GetActiveScene(), ScenePath);
            Debug.Log("[CapstoneSceneSetup] saved=" + ok + " -> " + ScenePath);
            EditorApplication.Exit(ok && agentBlocked && heroClear ? 0 : 1);
        }
        catch (System.Exception e)
        {
            Debug.LogError("[CapstoneSceneSetup] Build FAILED: " + e);
            EditorApplication.Exit(2);
        }
    }

    public static void Shot()
    {
        try
        {
            EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
            GameObject camGo = GameObject.Find("HeroCamera");
            Camera cam = camGo != null ? camGo.GetComponent<Camera>() : Camera.main;
            if (cam == null) { Debug.LogError("[CapstoneSceneSetup] no HeroCamera found in " + ScenePath); EditorApplication.Exit(1); return; }

            const int w = 1280, h = 720;
            var rt = new RenderTexture(w, h, 24, RenderTextureFormat.ARGB32);
            cam.targetTexture = rt;
            cam.Render();
            RenderTexture.active = rt;
            var tex = new Texture2D(w, h, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, w, h), 0, 0);
            tex.Apply();
            cam.targetTexture = null;
            RenderTexture.active = null;
            Object.DestroyImmediate(rt);

            byte[] png = tex.EncodeToPNG();
            System.IO.Directory.CreateDirectory(System.IO.Path.GetDirectoryName(ShotPath));
            System.IO.File.WriteAllBytes(ShotPath, png);
            Debug.Log("[CapstoneSceneSetup] wrote screenshot bytes=" + png.Length + " -> " + ShotPath);
            EditorApplication.Exit(0);
        }
        catch (System.Exception e)
        {
            Debug.LogError("[CapstoneSceneSetup] Shot FAILED: " + e);
            EditorApplication.Exit(2);
        }
    }

    private static Material MakeMat(string name, Color color, float glossiness)
    {
        var shader = Shader.Find("Standard");
        var mat = new Material(shader) { name = name };
        mat.SetColor("_Color", color);
        mat.SetFloat("_Glossiness", glossiness);
        mat.SetFloat("_Metallic", 0f);
        return mat;
    }

    private static GameObject MakeCube(string name, Vector3 pos, Vector3 scale, Material mat, string tag)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
        go.name = name;
        go.transform.position = pos;
        go.transform.localScale = scale;
        go.GetComponent<Renderer>().sharedMaterial = mat;
        if (tag != null) go.tag = tag;
        return go;
    }

    // Ramp as a rotated box, given a precomputed rotation (shared with MakeTreadStripes so the stripes
    // sit flush on the same slope). FromToRotation(forward, dir) maps local +Z onto the world slope
    // direction; since `dir` here has zero X component the rotation is pure-about-X (no roll).
    private static GameObject MakeRamp(string name, Vector3 midpoint, Quaternion rot, float length, float width, float thickness, Material mat)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
        go.name = name;
        go.transform.position = midpoint;
        go.transform.rotation = rot;
        go.transform.localScale = new Vector3(width, thickness, length);
        go.GetComponent<Renderer>().sharedMaterial = mat;
        return go;
    }

    // Thin contrasting bands laid across the ramp face, evenly spaced along its length, offset just
    // clear of the slope surface along the ramp's own rotated "up" axis - reads as tread stripes /
    // rungs so the silhouette alone says "walk up here", not "solid block".
    private static void MakeTreadStripes(string prefix, Vector3 rampStart, Vector3 rampEnd, Quaternion rampRot, float width, float thickness, int count, Material mat)
    {
        Vector3 slopeUp = rampRot * Vector3.up;
        for (int i = 1; i <= count; i++)
        {
            float t = i / (float)(count + 1);
            Vector3 centerline = Vector3.Lerp(rampStart, rampEnd, t);
            Vector3 pos = centerline + slopeUp * (thickness / 2f + 0.05f);
            var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = prefix + "_Tread" + i;
            go.transform.position = pos;
            go.transform.rotation = rampRot;
            go.transform.localScale = new Vector3(width, 0.08f, 0.35f);
            go.GetComponent<Renderer>().sharedMaterial = mat;
        }
    }

    private static GameObject MakeGoal(string name, Vector3 pos, Material mat, string tag, float scale)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        go.name = name;
        go.transform.position = pos;
        go.transform.localScale = new Vector3(scale, scale, scale); // unit sphere radius 0.5 -> scale 2 = radius 1u
        go.GetComponent<Renderer>().sharedMaterial = mat;
        if (tag != null) go.tag = tag;
        return go;
    }

    // Raycasts from `eye` toward `goalPos` and confirms the first thing hit is `wall` (primitives get a
    // BoxCollider for free from CreatePrimitive, so no extra collider setup is needed). Stops just short
    // of the goal itself so hitting the goal's own collider doesn't count as "blocked".
    private static bool CheckOcclusion(Vector3 eye, Vector3 goalPos, GameObject wall, int attempt)
    {
        Vector3 delta = goalPos - eye;
        float dist = delta.magnitude;
        bool hitSomething = Physics.Raycast(eye, delta.normalized, out RaycastHit hit, dist - 0.05f);
        bool blocked = hitSomething && hit.collider != null && hit.collider.gameObject == wall;
        Debug.Log("[CapstoneSceneSetup] occlusion check attempt=" + attempt + " eye=" + eye + " goal=" + goalPos
            + " hit=" + (hitSomething ? hit.collider.gameObject.name + "@" + hit.distance.ToString("F2") : "none")
            + " blockedByWall=" + blocked);
        return blocked;
    }

    // Inverse of CheckOcclusion: confirms the eye has a CLEAR line to the goal - either nothing is hit
    // before the (near-)goal distance, or the first thing hit IS the goal itself (its own collider
    // surface, reached just before the trimmed distance runs out - expected, not a blocker).
    private static bool CheckClearLOS(Vector3 eye, Vector3 goalPos, GameObject goal, string label)
    {
        Vector3 delta = goalPos - eye;
        float dist = delta.magnitude;
        bool hitSomething = Physics.Raycast(eye, delta.normalized, out RaycastHit hit, dist - 0.05f);
        bool clear = !hitSomething || (hit.collider != null && hit.collider.gameObject == goal);
        Debug.Log("[CapstoneSceneSetup] " + label + " clear-LOS check eye=" + eye + " goal=" + goalPos
            + " hit=" + (hitSomething ? hit.collider.gameObject.name + "@" + hit.distance.ToString("F2") : "none")
            + " clear=" + clear);
        return clear;
    }
}
