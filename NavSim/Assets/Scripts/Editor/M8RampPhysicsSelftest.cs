// M8 verify-early #1 - the PHYSICS make-or-break gate (run BEFORE any training). Proves the dynamic ramp
// behaves as the design requires under the MANUAL-STEPPED eval seam (Physics.simulationMode=Script +
// Physics.Simulate, NOT live Play), and finalizes the mass/damping/friction tuning so:
//   Check 0 (bootstrap sanity): LIGHT ramp, ONE agent  -> ramp REACHES target within N steps  (MUST be true)
//   Check 1 (forced):           HEAVY ramp, ONE agent  -> ramp does NOT reach target in N      (MUST be true)
//   Check 2 (cooperation works):HEAVY ramp, TWO agents -> ramp REACHES target within N steps   (MUST be true)
//   Check 3 (no trap/launch):   after Check 2, no agent y<-1, all positions finite, ramp did NOT fly past
//                               the target into the far wall (no wild overshoot)                (MUST be true)
//
// MECHANIC: LATERAL push. The ramp slab tilts about X, so its +-X end faces stay VERTICAL (|n.y|=0) while its
// +z top face is the walkable slope. Agents stand BESIDE the ramp on the -x side and shove it +x against the
// tall vertical -x face - the existing |n.y|>0.5 push rule fires cleanly (no rule change, so the placed-ramp
// nudge guard stays closed). "1 creeps / 2 place" is a TEMPORAL forcing: over the x push distance a lone
// agent (esp. heavy) is too slow to cover it in the step budget while two are fast enough.
//
// Forks M7EvalBatch's batchmode Play-entry idiom (DisableDomainReload|DisableSceneReload; EnterPlaymode();
// EditorApplication.update += Tick that warms ~30 frames then runs and Exits itself). The push is driven
// WITHOUT a trained model via RampAgent.DebugDriveForward (grounded, so the pusher stays low and keeps
// shoving the vertical face instead of climbing over). EnvironmentStep is called ONLY in warmup (proving the
// batchmode Play-entry + academy/physics/Tick seam); the measurement loop is PURE DebugDriveForward ->
// Physics.Simulate -> arena.Tick (matching eval's one-push-per-step cadence). N = arena maxEpisodeSteps.
//
// TUNING knobs overridable via env (no rebuild): M8_DAMPING (ramp linearDamping), M8_STARTX (ramp start x ->
// push distance), M8_MASSLIGHT, M8_MASSHEAVY. Final values get baked into PushableRampBody/RampArena/builder.
//
//   Unity -batchmode -projectPath NavSim -executeMethod M8RampPhysicsSelftest.Run -logFile <log>
using System;
using System.Globalization;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using Unity.MLAgents;
using NavSim.Runtime;

public static class M8RampPhysicsSelftest
{
    private const string RampScene = "Assets/Scenes/Ramp.unity";
    private const float ArenaHalf = 11f;   // ramp must not fly past ~this (wall) - overshoot/tunnel guard
    private static int _frames;

    public static void Run()
    {
        EditorSceneManager.OpenScene(RampScene, OpenSceneMode.Single);
        EditorSettings.enterPlayModeOptionsEnabled = true;
        EditorSettings.enterPlayModeOptions =
            EnterPlayModeOptions.DisableDomainReload | EnterPlayModeOptions.DisableSceneReload;
        _frames = 0;
        EditorApplication.update += Tick;
        EditorApplication.EnterPlaymode();
    }

    private static void Tick()
    {
        if (!EditorApplication.isPlaying) return;   // wait for Play + Academy
        _frames++;
        if (_frames < 30) return;                    // ~30 frames to init scene + Academy
        EditorApplication.update -= Tick;
        int code = 1;
        try { code = RunChecks(); }
        catch (Exception e) { Debug.LogError("[M8Phys] FAILED: " + e); code = 1; }
        EditorApplication.Exit(code);
    }

    private struct Result
    {
        public bool placed; public int steps; public float startDist, finalDist, minDist;
        public int pushEvents, rawContacts, skipped;
        public float agentMinY, finalRampX; public bool finite;
        public Vector3 lastNormal, rampPos, pusher0Pos;
        public string trace;
    }

    private static int RunChecks()
    {
        var arena = UnityEngine.Object.FindAnyObjectByType<RampArena>();
        if (arena == null) { Debug.LogError("[M8Phys] no RampArena in scene"); return 1; }
        var agents = arena.Agents;
        var ramp = arena.Ramp;
        if (agents == null || agents.Length < 2 || ramp == null)
        { Debug.LogError($"[M8Phys] scene needs 2 agents + ramp (got agents={agents?.Length}, ramp={ramp})"); return 1; }

        var so = new SerializedObject(arena);
        float massLight = EnvF("M8_MASSLIGHT", so.FindProperty("rampMassLight").floatValue);
        float massHeavy = EnvF("M8_MASSHEAVY", so.FindProperty("rampMassHeavy").floatValue);
        float targetRadius = so.FindProperty("targetRadius").floatValue;
        int N = so.FindProperty("maxEpisodeSteps").intValue;
        var rampTargetT = (Transform)so.FindProperty("rampTarget").objectReferenceValue;
        var rampStartT = (Transform)so.FindProperty("rampStart").objectReferenceValue;
        Vector3 rampTarget = rampTargetT.position;
        Vector3 startPos = rampStartT.position;
        startPos.x = EnvF("M8_STARTX", startPos.x);   // tune push distance without a rebuild

        var rb = ramp.GetComponent<Rigidbody>();
        float damping = EnvF("M8_DAMPING", rb.linearDamping);
        rb.linearDamping = damping;                    // apply the (possibly overridden) damping for this run
        float pushForce = new SerializedObject(agents[0]).FindProperty("pushForceNewtons").floatValue;
        var rampCol = ramp.GetComponentInChildren<Collider>();
        float friction = rampCol != null && rampCol.sharedMaterial != null ? rampCol.sharedMaterial.dynamicFriction : -1f;
        string combine = rampCol != null && rampCol.sharedMaterial != null ? rampCol.sharedMaterial.frictionCombine.ToString() : "?";
        float pushDist = Mathf.Abs(startPos.x - rampTarget.x);

        Debug.Log($"[M8Phys] CONFIG massLight={massLight} massHeavy={massHeavy} friction={friction}/{combine} " +
                  $"damping={damping} pushForce={pushForce} targetRadius={targetRadius} N={N} pushDistX={pushDist:F2} " +
                  $"rampStart={V(startPos)} rampTarget={V(rampTarget)}");

        // --- Warmup: prove batchmode Play-entry + the EnvironmentStep/Simulate/Tick seam actually run. ---
        Physics.simulationMode = SimulationMode.Script;
        arena.EvalMode = true;
        for (int a = 0; a < agents.Length; a++) agents[a].MaxStep = 0;
        int warm = 0;
        try { for (; warm < 3; warm++) { Academy.Instance.EnvironmentStep(); Physics.Simulate(Time.fixedDeltaTime); arena.Tick(Time.fixedDeltaTime); } }
        catch (Exception e) { Debug.LogError("[M8Phys] SEAM warmup FAILED (batchmode Play-entry/EnvironmentStep broken): " + e); return 1; }
        Debug.Log($"[M8Phys] batchmode Play-entry OK (isPlaying={Application.isPlaying}, warm seam steps={warm})");

        // --- Un-embed verification: settle the ramp (no push) and confirm it rests ON the floor (lowest
        // collider point ~y>=0, not sunk below the floor top which would jam under FreezePositionY). ---
        ramp.ResetTo(startPos, Quaternion.identity); ramp.Mass = massLight; Physics.SyncTransforms();
        for (int s = 0; s < 20; s++) { Physics.Simulate(Time.fixedDeltaTime); arena.Tick(Time.fixedDeltaTime); }
        float restY = ramp.Position.y, colMinY = rampCol != null ? rampCol.bounds.min.y : float.NaN;
        bool unembedded = colMinY > -0.05f;
        Debug.Log($"[M8Phys] un-embed check: ramp.y={restY:F3} colliderMinY={colMinY:F3} unembedded={unembedded} (MUST be true)");

        // --- Check 0: LIGHT ramp, ONE agent (agents[0]); agents[1] parked away. ---
        var r0 = RunPush(arena, ramp, rampTarget, targetRadius, massLight, startPos,
                         new[] { agents[0] }, new[] { agents[1] }, N);
        LogCheck("Check0 light-1", massLight, targetRadius, r0);

        // --- Check 1: HEAVY ramp, ONE agent. ---
        var r1 = RunPush(arena, ramp, rampTarget, targetRadius, massHeavy, startPos,
                         new[] { agents[0] }, new[] { agents[1] }, N);
        LogCheck("Check1 heavy-1", massHeavy, targetRadius, r1);

        // --- Check 2: HEAVY ramp, TWO agents. ---
        var r2 = RunPush(arena, ramp, rampTarget, targetRadius, massHeavy, startPos,
                         new[] { agents[0], agents[1] }, Array.Empty<RampAgent>(), N);
        LogCheck("Check2 heavy-2", massHeavy, targetRadius, r2);

        // --- Check 4 (CLIMB - the coordinator's acceptance invariant: the +z walkable slope reaches the
        // platform AFTER placement): place the ramp at the target, drop a grounded agent at the ramp's -z low
        // end facing +z, drive it up the slope. It MUST reach the platform top (cross onto z>=4 at height). ---
        var cl = RunClimb(arena, ramp, rampTarget, massHeavy, agents, so);
        Debug.Log($"[M8Phys] Check4 climb onPlatform={cl.onPlatform} reachedGoal={cl.reachedGoal} maxY={cl.maxY:F2} " +
                  $"endPos={V(cl.endPos)} steps={cl.steps} (MUST be true: the placed ramp is climbable onto the platform)");

        bool c0 = r0.placed;
        bool c1 = !r1.placed;
        bool c2 = r2.placed;
        // Check 3 (Check 2's end state): agents grounded/not-dropped, all finite, ramp did not fly past the wall.
        bool c3 = unembedded && r2.finite && r2.agentMinY > -1f && Mathf.Abs(r2.finalRampX) < ArenaHalf;
        bool c4 = cl.onPlatform || cl.reachedGoal;

        bool allPass = c0 && c1 && c2 && c3 && c4;
        Debug.Log($"[M8Phys] SELFTEST light-1-places={c0} (MUST be true, steps={StepsStr(r0)}); " +
                  $"heavy-1-too-slow={c1} (MUST be true, finalDist={r1.finalDist:F2} minDist={r1.minDist:F2} of push {pushDist:F2}); " +
                  $"heavy-2-places={c2} (MUST be true, steps={StepsStr(r2)}); " +
                  $"no-launch/trap/tunnel={c3} (MUST be true, unembed={unembedded} agentMinY={r2.agentMinY:F2} rampX={r2.finalRampX:F2} finite={r2.finite}); " +
                  $"climb-to-platform={c4} (MUST be true, onPlatform={cl.onPlatform} reachedGoal={cl.reachedGoal} maxY={cl.maxY:F2}) " +
                  $"-> {allPass}  [damping={damping} massL/H={massLight}/{massHeavy} friction={friction} pushDistX={pushDist:F2} N={N}]");
        return allPass ? 0 : 1;
    }

    // One check: reset the ramp to startPos with the given mass, place pusher(s) BESIDE it (-x) facing +x, park
    // the idle agents far away, then run the PURE grounded push loop for up to N steps (break on place).
    private static Result RunPush(RampArena arena, PushableRampBody ramp, Vector3 rampTarget, float targetRadius,
                                  float mass, Vector3 startPos, RampAgent[] pushers, RampAgent[] idle, int N)
    {
        ramp.ResetTo(startPos, Quaternion.identity);
        ramp.Mass = mass;
        for (int i = 0; i < idle.Length; i++) idle[i].TeleportTo(new Vector3(9f, 0.5f, -9f + i * 1.5f), 0f);
        PlacePushers(pushers, startPos);
        Physics.SyncTransforms();

        int push0 = ramp.TotalPushEvents, rc0 = 0, sk0 = 0;
        for (int p = 0; p < pushers.Length; p++) { rc0 += pushers[p].DebugRawRampContacts; sk0 += pushers[p].DebugSkippedContacts; }
        float startDist = Vector3.Distance(ramp.Position, rampTarget);
        float minDist = startDist, agentMinY = float.MaxValue;
        bool placed = false; int steps = N;
        var trace = new StringBuilder();

        for (int s = 0; s < N; s++)
        {
            for (int p = 0; p < pushers.Length; p++) pushers[p].DebugDriveForward(Time.fixedDeltaTime);
            Physics.Simulate(Time.fixedDeltaTime);
            arena.Tick(Time.fixedDeltaTime);

            float d = Vector3.Distance(ramp.Position, rampTarget);
            if (d < minDist) minDist = d;
            for (int p = 0; p < pushers.Length; p++) agentMinY = Mathf.Min(agentMinY, pushers[p].transform.position.y);
            if (s % 400 == 0) trace.Append($" [{s}]x={ramp.Position.x:F2},d={d:F2}");
            if (!placed && d < targetRadius) { placed = true; steps = s + 1; break; }
        }

        Vector3 rampPos = ramp.Position;
        int rc = 0, sk = 0;
        for (int p = 0; p < pushers.Length; p++) { rc += pushers[p].DebugRawRampContacts; sk += pushers[p].DebugSkippedContacts; }
        return new Result
        {
            placed = placed, steps = steps, startDist = startDist,
            finalDist = Vector3.Distance(rampPos, rampTarget), minDist = minDist,
            pushEvents = ramp.TotalPushEvents - push0, rawContacts = rc - rc0, skipped = sk - sk0,
            agentMinY = agentMinY, finalRampX = rampPos.x, finite = Finite(rampPos) && AllAgentsFinite(pushers),
            lastNormal = pushers[0].DebugLastRampNormal, rampPos = rampPos, pusher0Pos = pushers[0].transform.position,
            trace = trace.ToString()
        };
    }

    // Climb verification: with the ramp PLACED at the target, a grounded agent starts at the ramp's -z low
    // end (floor level, on the x=0 centerline) facing +z and drives up the +z slope. Success = it crosses onto
    // the platform (z>=4.1 at height y>=2.0) - proving the walkable slope bridges to the platform - or reaches
    // the goal outright. Uses the same grounded DebugDriveForward + manual seam.
    private static (bool onPlatform, bool reachedGoal, float maxY, Vector3 endPos, int steps) RunClimb(
        RampArena arena, PushableRampBody ramp, Vector3 rampTarget, float mass, RampAgent[] agents, SerializedObject arenaSo)
    {
        var goalT = (Transform)arenaSo.FindProperty("goal").objectReferenceValue;
        Vector3 goal = goalT.position;
        float goalRadius = arenaSo.FindProperty("goalRadius").floatValue;
        ramp.ResetTo(rampTarget, Quaternion.identity); ramp.Mass = mass;
        for (int i = 1; i < agents.Length; i++) agents[i].TeleportTo(new Vector3(9f, 0.5f, -9f + i), 0f);
        agents[0].TeleportTo(new Vector3(rampTarget.x, 0.5f, rampTarget.z - 2.2f), 0f);   // -z low end, facing +z
        Physics.SyncTransforms();

        float maxY = 0f; bool onPlatform = false, reachedGoal = false; int steps = 0;
        for (; steps < 1000; steps++)
        {
            agents[0].DebugDriveForward(Time.fixedDeltaTime);
            Physics.Simulate(Time.fixedDeltaTime);
            arena.Tick(Time.fixedDeltaTime);
            Vector3 p = agents[0].transform.position;
            maxY = Mathf.Max(maxY, p.y);
            if (!onPlatform && p.z >= 4.1f && p.y >= 2.0f) onPlatform = true;
            if (Vector3.Distance(p, goal) < goalRadius) { reachedGoal = true; break; }
        }
        return (onPlatform, reachedGoal, maxY, agents[0].transform.position, steps);
    }

    // Lateral push: agents stand on the -x side of the ramp (its -x vertical face is ~startPos.x - halfWidth(2)),
    // facing +x (yaw 90), spread in z so both contact the tall vertical face; DebugDriveForward walks them +x
    // into it. (The +z-approach that made the agent climb over the slope is exactly what this avoids.)
    private static void PlacePushers(RampAgent[] pushers, Vector3 startPos)
    {
        float behindX = startPos.x - 3.2f;   // just -x of the ramp's -x face (~startPos.x - 2)
        if (pushers.Length == 1)
            pushers[0].TeleportTo(new Vector3(behindX, 0.5f, startPos.z), 90f);
        else
            for (int i = 0; i < pushers.Length; i++)
                pushers[i].TeleportTo(new Vector3(behindX, 0.5f, startPos.z + (i == 0 ? -1.1f : 1.1f)), 90f);
    }

    private static void LogCheck(string label, float mass, float targetRadius, Result r) =>
        Debug.Log($"[M8Phys] {label} mass={mass} placed={r.placed} steps={StepsStr(r)} " +
                  $"startDist={r.startDist:F2} minDist={r.minDist:F2} finalDist={r.finalDist:F2} (radius={targetRadius}) " +
                  $"pushEvents={r.pushEvents} rawContacts={r.rawContacts} skipped={r.skipped} lastNormal={V(r.lastNormal)} " +
                  $"rampPos={V(r.rampPos)} pusher0Pos={V(r.pusher0Pos)} agentMinY={r.agentMinY:F2} finite={r.finite} " +
                  $"trace:{r.trace}");

    private static float EnvF(string name, float def)
    {
        string v = Environment.GetEnvironmentVariable(name);
        return !string.IsNullOrEmpty(v) && float.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out float f) ? f : def;
    }

    private static string StepsStr(Result r) => r.placed ? r.steps.ToString() : ("0/" + r.steps);
    private static bool Finite(Vector3 v) => !(float.IsNaN(v.x) || float.IsNaN(v.y) || float.IsNaN(v.z) ||
                                               float.IsInfinity(v.x) || float.IsInfinity(v.y) || float.IsInfinity(v.z));
    private static bool AllAgentsFinite(RampAgent[] a) { for (int i = 0; i < a.Length; i++) if (!Finite(a[i].transform.position)) return false; return true; }
    private static string V(Vector3 v) => string.Format(CultureInfo.InvariantCulture, "({0:F2},{1:F2},{2:F2})", v.x, v.y, v.z);
}
