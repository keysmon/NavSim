using System.Collections.Generic;
using System.Globalization;
using System.Reflection;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using Unity.MLAgents.Actuators;
using NavSim.Runtime;

// P0 SCRIPTED GEOMETRY GATES (Task 6). Committed tooling, M5/M6 feel-probe precedent. Opens Training_showcase.unity,
// enters Play mode (the ONLY way NavAgent.Initialize runs, so _cc + the Agent reward accumulator are live), then
// DRIVES the agent by constructing ActionBuffers and calling NavAgent.OnActionReceived DIRECTLY — no policy, no
// Academy stepping, no trainer. Under Physics.simulationMode = Script the whole gate battery runs synchronously in a
// SINGLE EditorApplication.update tick, so no FixedUpdate/Academy step interleaves and every step is deterministic.
//
// Idiom invariants (M5-proven, project standing practice):
//   - Physics.SyncTransforms() after EVERY ForceCourse/teleport (batchmode has no auto-sync; stale collider poses
//     report no hits) + a few zero-action SETTLE steps (PlaceAt disables/re-enables the CC, resetting isGrounded).
//   - env.EvalMode = true for the whole run: suppresses EndEpisode-on-success AND the hard-decoy EndEpisode, so a
//     scripted drive is NEVER interrupted by an episode boundary (course-mode RespawnGoal just redraws colours).
//   - "reached slot i" is detected GEOMETRICALLY here (Vector3.Distance < 1.45, just under env.GoalRadius 1.5).
//   - PitFalls delta is the AUTHORITATIVE y<KillY signal: RespawnToSafeGround runs synchronously inside the same
//     OnActionReceived that detects FellInPit, so the sub-kill y is never observable post-step; PitFalls++ IS it.
//
// Batchmode:  Unity -batchmode -projectPath NavSim -executeMethod ShowcaseP0Gates.Run -logFile -
// Exit codes: 0 all gates pass / 1 a gate failed / 2 exception or hard rig failure (M6PixelSceneSetup idiom).
public static class ShowcaseP0Gates
{
    private const string ScenePath = "Assets/Scenes/Training_showcase.unity";
    private const int WarmupFrames = 40;     // Play-mode + Academy/scene init before we seize control
    private const int SettleSteps = 24;      // zero-action steps after a ForceCourse to re-ground the CC (spawn drops ~1u)
    private const int ReachBudget = 3000;    // == agent.MaxStep; the L4-headroom question this task must answer
    private const float WaypointRadius = 0.8f;
    private const float SlotReach = 1.45f;   // just under env.GoalRadius (1.5)
    private const float TurnDeg = 6f;        // NavAgent.maxTurnDegPerStep
    private const float FlankX = 3.75f;      // open-flank offset (walls span ±... ; 3.75 clears both wall + curb)
    private const float JumpLateWindow = 0.4f; // fire the lip jump only when grounded in [lipZ-this, lipZ+0.2] (bias LATE)
    private const int PitFallBudget = 400;   // "y<KillY within 400 steps of crossing the lip"

    private static NavEnvironment _env;
    private static NavAgent _agent;
    private static CharacterController _cc;
    private static float _dt;
    private static int _frames;
    private static readonly StringBuilder _report = new StringBuilder();
    private static bool _allPass = true;
    private static int _runs, _passes;

    public static void Run()
    {
        try
        {
            EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
            // DisableDomainReload keeps this class's static state (_frames, the update-callback) alive across the
            // Play-mode transition; DisableSceneReload keeps the loaded scene. Same idiom as M6EvalBatch.
            EditorSettings.enterPlayModeOptionsEnabled = true;
            EditorSettings.enterPlayModeOptions =
                EnterPlayModeOptions.DisableDomainReload | EnterPlayModeOptions.DisableSceneReload;
            _frames = 0;
            EditorApplication.update += Tick;
            EditorApplication.EnterPlaymode();
        }
        catch (System.Exception e)
        {
            Debug.LogError("[P0] Run setup FAILED: " + e);
            EditorApplication.Exit(2);
        }
    }

    private static void Tick()
    {
        if (!EditorApplication.isPlaying) return;   // wait for Play + Academy
        _frames++;
        if (_frames < WarmupFrames) return;
        EditorApplication.update -= Tick;
        int code;
        try { code = RunGates(); }
        catch (System.Exception e) { Debug.LogError("[P0] EXCEPTION: " + e); code = 2; }
        EditorApplication.Exit(code);
    }

    private static int RunGates()
    {
        _env = Object.FindAnyObjectByType<NavEnvironment>();
        _agent = Object.FindAnyObjectByType<NavAgent>();
        if (_env == null || _agent == null) { Debug.LogError("[P0] missing env/agent in scene"); return 2; }
        if (!_env.CourseMode) { Debug.LogError("[P0] env.course not wired — scene is not in COURSE mode"); return 2; }
        _cc = _agent.GetComponent<CharacterController>();
        _dt = Time.fixedDeltaTime;

        Physics.simulationMode = SimulationMode.Script;
        _agent.MaxStep = 0;      // harness owns the boundary (belt-and-suspenders; we never Academy-step anyway)
        _env.EvalMode = true;    // no EndEpisode-on-success, no hard-decoy EndEpisode -> scripted drives never interrupt

        if (!VerifyRig()) return 2; // agentCamera null (SHAPING needs the pixel frustum) or stepOffset < curb -> STOP

        // Course mode (NavEnvironment.ApplyCourse) disables the leftover M5/M6 arena on the first stage build, so the
        // arena boundary walls no longer sit across the course. The gates below drive the real scene with no overrides.

        // GATE 1 — REACH: stages 0-4 (stage 3 in BOTH variants), each on BOTH mirror states.
        foreach (bool mir in new[] { false, true })
        {
            ReachGate(0, CourseVariant.NoLoop, mir);
            ReachGate(1, CourseVariant.NoLoop, mir);
            ReachGate(2, CourseVariant.NoLoop, mir);
            ReachGate(3, CourseVariant.SafeLoop, mir);
            ReachGate(3, CourseVariant.NoLoop, mir);
            ReachGate(4, CourseVariant.NoLoop, mir);
        }
        // GATE 2 — PIT: stage 3 NoLoop, walk off the lip WITHOUT jumping -> fall.
        foreach (bool mir in new[] { false, true }) PitGate(mir);
        // GATE 3 — JUMP: stage 3 NoLoop, same route WITH the lip jump -> cross, no fall.
        foreach (bool mir in new[] { false, true }) JumpGate(mir);
        // GATE 4 — SHAPING: stage 0, turn in place -> TargetPerceivable fires.
        foreach (bool mir in new[] { false, true }) ShapingGate(mir);
        // GATE 5 — NO-JUMP-NO-CROSS: stage 4, full route WITHOUT the jump -> never crosses, never reaches a slot.
        foreach (bool mir in new[] { false, true }) NoJumpNoCrossGate(mir);

        Physics.simulationMode = SimulationMode.FixedUpdate; // restore (scene is never saved; this is in-memory only)

        Debug.Log("[P0] ================= GATE REPORT =================\n" + _report.ToString());
        Debug.Log($"[P0] RESULT {(_allPass ? "ALL PASS" : "FAIL")} — {_passes}/{_runs} gate-runs PASS");
        return _allPass ? 0 : 1;
    }

    // ---- rig verification (advisor: stepOffset vs the 0.3 curb lip is load-bearing) ----
    private static bool VerifyRig()
    {
        Camera agentCam = typeof(NavEnvironment)
            .GetField("agentCamera", BindingFlags.NonPublic | BindingFlags.Instance)
            ?.GetValue(_env) as Camera;
        Debug.Log($"[P0] RIG dt={_dt:F5} cc(height={_cc.height:F2} radius={_cc.radius:F2} center={_cc.center} " +
                  $"stepOffset={_cc.stepOffset:F3} skinWidth={_cc.skinWidth:F3} slopeLimit={_cc.slopeLimit:F1}) " +
                  $"agentCamera={(agentCam != null ? agentCam.name : "NULL")} EvalMode={_env.EvalMode} " +
                  $"GoalRadius={_env.GoalRadius} KillY={CourseSpec.KillY} DeckY={CourseSpec.DeckY}");
        if (agentCam == null)
        {
            Debug.LogError("[P0] HARD RIG FAIL: env.agentCamera is null — SHAPING's TargetPerceivable would silently " +
                           "route to the (stripped) ray path and never fire. The showcase scene must be the pixel fork.");
            return false;
        }
        // The pit-lip curb is 0.3 tall; if the CC cannot mount it, PIT can never fire (and that is NOT a spec-bound
        // geometry constant we may move — STOP-and-report, per the brief).
        if (_cc.stepOffset < 0.3f)
            Debug.LogWarning($"[P0] WATCH: stepOffset {_cc.stepOffset:F3} < 0.30 (curb-lip height). If PIT never fires " +
                             "the CC cannot walk off the lip — geometry-blocked (STOP-and-report), not a harness tune.");
        return true;
    }

    // ================= per-step driving =================

    private static void Step(float forward, float turn, int jump)
    {
        var buffers = new ActionBuffers(
            new ActionSegment<float>(new float[] { forward, turn }),
            new ActionSegment<int>(new int[] { jump }));
        _agent.OnActionReceived(buffers);      // moves the CC, runs reach/decoy/pit logic (rewards ignored)
        Physics.Simulate(_dt);                 // advance physics (implicit SyncTransforms at start)
    }

    private static void ForceStage(int stage, bool mirrored, CourseVariant variant)
    {
        // CourseBuilder.ClearPieces uses Destroy() in Play mode, which is DEFERRED to end-of-frame — but the whole
        // gate battery runs synchronously in one frame, so without this the PRIOR stage's colliders linger as ghosts
        // (a stage-0 probe was occluded by a stage-2 ramp). Synchronously remove them; CourseBuilder null-checks its
        // piece list, so it rebuilds cleanly. The goal triad lives under "Environment", not Course — untouched.
        ClearCourseChildrenImmediate();
        _env.ForceCourse(stage, mirrored, variant);
        Physics.SyncTransforms();
        for (int i = 0; i < SettleSteps; i++) Step(0f, 0f, 0); // re-ground the CC before we read isGrounded/drive
    }

    private static void ClearCourseChildrenImmediate()
    {
        Transform root = _env.Course.transform;
        var kids = new List<GameObject>();
        for (int i = 0; i < root.childCount; i++) kids.Add(root.GetChild(i).gameObject);
        foreach (var k in kids) Object.DestroyImmediate(k);
    }

    // Steer toward a waypoint on XZ. SignedAngle(forward, dir, up) is +ve when the target is to the agent's right,
    // and Rotate(0, +turn*deg, 0) rotates forward toward +X (right) — so turn = angle/6 CONVERGES (verified stage 0).
    private static void SteerToward(Vector3 target, out float forward, out float turn)
    {
        Vector3 pos = _agent.transform.position;
        Vector3 dir = target - pos; dir.y = 0f;
        Vector3 fwd = _agent.transform.forward; fwd.y = 0f;
        float angle = dir.sqrMagnitude < 1e-6f ? 0f : Vector3.SignedAngle(fwd, dir, Vector3.up);
        turn = Mathf.Clamp(angle / TurnDeg, -1f, 1f);
        forward = Mathf.Abs(angle) < 45f ? 1f : 0.2f;
    }

    // One driven step toward `target`, with the lip jump fired LATE (only on a grounded step near the deck edge).
    private static void DriveOneStep(Vector3 target, float jumpLipZ, bool doJump)
    {
        SteerToward(target, out float forward, out float turn);
        int jump = 0;
        if (doJump && _cc.isGrounded)
        {
            float z = _agent.transform.position.z;
            if (z >= jumpLipZ - JumpLateWindow && z <= jumpLipZ + 0.2f) jump = 1;
        }
        Step(forward, turn, jump);
    }

    private static float XZDist(Vector3 a, Vector3 b)
        => Vector2.Distance(new Vector2(a.x, a.z), new Vector2(b.x, b.z));

    // Drive through the waypoint list in order. Returns false (and leaves `steps` at the budget) if the budget is hit.
    private static bool RunRoute(List<Vector3> route, float jumpLipZ, bool doJump, ref int steps, int budget)
    {
        foreach (Vector3 wp in route)
            while (XZDist(_agent.transform.position, wp) > WaypointRadius)
            {
                if (steps >= budget) return false;
                DriveOneStep(wp, jumpLipZ, doJump);
                steps++;
            }
        return true;
    }

    // Greedily visit each slot (nearest-unvisited first) until all reached within SlotReach or the budget is hit.
    private static bool VisitSlots(Vector3[] slots, float jumpLipZ, bool doJump, ref int steps, int budget,
        out int firstSlotStep, out float[] slotDist)
    {
        firstSlotStep = -1;
        slotDist = new float[slots.Length];
        var visited = new bool[slots.Length];
        int done = 0;
        while (done < slots.Length)
        {
            int idx = -1; float best = float.MaxValue;
            Vector3 pos = _agent.transform.position;
            for (int i = 0; i < slots.Length; i++)
                if (!visited[i]) { float d = XZDist(pos, slots[i]); if (d < best) { best = d; idx = i; } }
            Vector3 target = slots[idx];
            while (Vector3.Distance(_agent.transform.position, target) >= SlotReach)
            {
                if (steps >= budget) return false;
                DriveOneStep(target, jumpLipZ, doJump);
                steps++;
            }
            slotDist[idx] = Vector3.Distance(_agent.transform.position, target);
            visited[idx] = true; done++;
            if (firstSlotStep < 0) firstSlotStep = steps;
        }
        return true;
    }

    // Mirror-aware waypoint route to the slot cluster, read from the LIVE layout (never a hardcoded handedness).
    private static List<Vector3> RouteFor(CourseLayout lay)
    {
        var wps = new List<Vector3>();
        float sign = lay.Mirrored ? -1f : 1f;           // single-wall open flank (unmirrored = +X); negate under mirror
        Vector3 W(float x, float z) => new Vector3(x, 0f, z); // y ignored (steering is XZ-only)
        switch (lay.Stage)
        {
            case 0:
                wps.Add(RingCentre(lay.GoalSlots));
                break;
            case 1: // ramp -> deck
                wps.Add(W(0f, 6f)); wps.Add(W(0f, 12f)); wps.Add(W(0f, 15f));
                break;
            case 2: // ramp -> right-flank detour around the wall (Z=26) -> slots
                wps.Add(W(0f, 6f)); wps.Add(W(0f, 12f));
                wps.Add(W(sign * FlankX, 22f)); wps.Add(W(sign * FlankX, 30f));
                break;
            case 3: // ramp -> flank around wall (Z=24) -> re-center pre-lip -> JUMP -> deck B -> slots
                wps.Add(W(0f, 6f)); wps.Add(W(0f, 12f));
                wps.Add(W(sign * FlankX, 20f)); wps.Add(W(sign * FlankX, 25f));
                wps.Add(W(0f, lay.GapZMin - 1.5f));     // pre-lip (26.5)
                wps.Add(W(0f, lay.GapZMax + 3f));       // deck B landing (33.2)
                break;
            case 4: // ramp -> S-bend (WallA Z=20 open one side, WallB Z=26 the other) -> pre-lip -> JUMP -> deck B
                float flankA = sign * FlankX;           // WallA open flank
                float flankB = -sign * FlankX;          // WallB open flank (opposite side)
                wps.Add(W(0f, 6f)); wps.Add(W(0f, 12f));
                wps.Add(W(flankA, 17f)); wps.Add(W(flankA, 21f));
                wps.Add(W(0f, 23f));                    // between-walls midpoint
                wps.Add(W(flankB, 25f)); wps.Add(W(flankB, 28f));
                wps.Add(W(0f, lay.GapZMin - 1.5f));     // pre-lip (28.5)
                wps.Add(W(0f, lay.GapZMax + 3f));       // deck B landing (35.2)
                break;
        }
        return wps;
    }

    private static Vector3 RingCentre(Vector3[] slots)
    {
        Vector3 c = Vector3.zero;
        foreach (var s in slots) c += s;
        return c / slots.Length;
    }

    // ================= gates =================

    private static void ReachGate(int stage, CourseVariant variant, bool mir)
    {
        ForceStage(stage, mir, variant);
        CourseLayout lay = _env.Course.CurrentLayout;
        List<Vector3> route = RouteFor(lay);
        bool doJump = stage == 3 || stage == 4;
        float lipZ = lay.GapZMin;                       // 0 when no pit (doJump false there)
        int steps = 0;
        bool routeOk = RunRoute(route, lipZ, doJump, ref steps, ReachBudget);
        bool reachedAll = false; int firstSlotStep = -1; float[] slotDist = new float[3];
        if (routeOk)
            reachedAll = VisitSlots(lay.GoalSlots, lipZ, doJump, ref steps, ReachBudget, out firstSlotStep, out slotDist);
        Record("REACH", reachedAll,
            $"stage={stage} variant={variant} mirror={mir}: routeReached={routeOk} firstSlot@step={firstSlotStep} " +
            $"totalSteps={steps}/{ReachBudget} slotDist=[{Fmt(slotDist)}] endPos={Fmt(_agent.transform.position)}");
    }

    private static void PitGate(bool mir)
    {
        ForceStage(3, mir, CourseVariant.NoLoop);
        CourseLayout lay = _env.Course.CurrentLayout;
        List<Vector3> route = RouteFor(lay);
        // Drive to pre-lip (all waypoints except the final deck-B landing), NO jump.
        var toPreLip = route.GetRange(0, route.Count - 1);
        int steps = 0;
        if (!RunRoute(toPreLip, float.NaN, false, ref steps, ReachBudget))
        { Record("PIT", false, $"mirror={mir}: could not reach pre-lip in {ReachBudget} steps"); return; }

        int pit0 = _env.PitFalls;
        Vector3 across = new Vector3(0f, 0f, lay.GapZMax + 5f); // a target BEYOND the pit; we drive at it without jumping
        int lipCrossStep = -1, fallStep = -1; float minY = 999f;
        for (int s = 0; s < PitFallBudget + 80; s++)         // +80 slack to walk from pre-lip onto the lip
        {
            float zBefore = _agent.transform.position.z;
            if (lipCrossStep < 0 && zBefore >= lay.GapZMin - 0.3f) lipCrossStep = s;
            DriveOneStep(across, float.NaN, false);
            steps++;
            if (lipCrossStep >= 0) minY = Mathf.Min(minY, _agent.transform.position.y);
            if (_env.PitFalls > pit0) { fallStep = s; break; }
        }
        bool fell = fallStep >= 0;
        int sinceLip = (fell && lipCrossStep >= 0) ? fallStep - lipCrossStep : -1;
        float respawnDist = Vector3.Distance(_agent.transform.position, lay.PitRespawnPos);
        bool pass = fell && lipCrossStep >= 0 && sinceLip <= PitFallBudget && respawnDist < 0.6f && minY < 0f;
        Record("PIT", pass,
            $"mirror={mir}: PitFalls+1={fell} lipCross@{lipCrossStep} fall@{fallStep} sinceLip={sinceLip}(<={PitFallBudget}) " +
            $"minY_beforeFall={minY:F2}(<0) respawnDist={respawnDist:F3}(<0.6) PitRespawnPos={Fmt(lay.PitRespawnPos)}");
    }

    private static void JumpGate(bool mir)
    {
        ForceStage(3, mir, CourseVariant.NoLoop);
        CourseLayout lay = _env.Course.CurrentLayout;
        List<Vector3> route = RouteFor(lay);
        int pit0 = _env.PitFalls;
        int steps = 0;
        bool routeOk = RunRoute(route, lay.GapZMin, true, ref steps, ReachBudget); // WITH the lip jump
        Vector3 pos = _agent.transform.position;
        int pitDelta = _env.PitFalls - pit0;
        bool crossed = pos.z > lay.GapZMax + 0.5f && pos.y > CourseSpec.DeckY - 0.5f;
        bool pass = routeOk && crossed && pitDelta == 0;
        Record("JUMP", pass,
            $"mirror={mir}: reachedDeckB={routeOk} crossed(z>{lay.GapZMax + 0.5f:F1}&y>{CourseSpec.DeckY - 0.5f:F1})={crossed} " +
            $"endPos={Fmt(pos)} pitFalls_delta={pitDelta}(==0) steps={steps}");
    }

    private static void ShapingGate(bool mir)
    {
        ForceStage(0, mir, CourseVariant.NoLoop);
        int firstTrue = -1;
        for (int s = 0; s < 60; s++)
        {
            if (_env.TargetPerceivable(_agent)) { firstTrue = s; break; }
            Step(0f, 1f, 0);   // turn in place (forward 0, turn +1 -> 6 deg/step)
        }
        bool pass = firstTrue >= 0;
        Record("SHAPING", pass,
            $"mirror={mir}: TargetPerceivable first-true @ turn-step={firstTrue} (<60, forward=0 turn=1)");
    }

    private static void NoJumpNoCrossGate(bool mir)
    {
        ForceStage(4, mir, CourseVariant.NoLoop);
        CourseLayout lay = _env.Course.CurrentLayout;
        List<Vector3> route = RouteFor(lay);
        var toPreLip = route.GetRange(0, route.Count - 1);
        int steps = 0;
        RunRoute(toPreLip, float.NaN, false, ref steps, ReachBudget); // best-effort to the lip, no jump
        Vector3 across = new Vector3(0f, 0f, lay.GapZMax + 5f);
        bool crossed = false, slotReached = false; float maxZ = _agent.transform.position.z;
        while (steps < ReachBudget)
        {
            DriveOneStep(across, float.NaN, false);      // keep pushing forward, NEVER jump
            steps++;
            Vector3 pos = _agent.transform.position;
            maxZ = Mathf.Max(maxZ, pos.z);
            if (pos.z > lay.GapZMax && pos.y > CourseSpec.DeckY - 0.5f) crossed = true;
            for (int i = 0; i < 3; i++)
                if (Vector3.Distance(pos, lay.GoalSlots[i]) < SlotReach) slotReached = true;
            if (crossed || slotReached) break;
        }
        bool reachedLip = maxZ >= lay.GapZMin - 1f;       // guard against a vacuous pass (agent never got to the lip)
        bool pass = reachedLip && !crossed && !slotReached;
        Record("NO-JUMP-NO-CROSS", pass,
            $"mirror={mir}: reachedLip(maxZ={maxZ:F1}>={lay.GapZMin - 1f:F1})={reachedLip} crossed={crossed} " +
            $"slotReached={slotReached} steps={steps}");
    }

    // ================= reporting =================

    private static void Record(string gate, bool pass, string ev)
    {
        _runs++;
        if (pass) _passes++; else _allPass = false;
        string line = $"[P0] GATE {gate} {(pass ? "PASS" : "FAIL")} {ev}";
        Debug.Log(line);
        _report.AppendLine(line);
    }

    private static string Fmt(Vector3 v) =>
        string.Format(CultureInfo.InvariantCulture, "({0:F2},{1:F2},{2:F2})", v.x, v.y, v.z);

    private static string Fmt(float[] a)
    {
        var sb = new StringBuilder();
        for (int i = 0; i < a.Length; i++) { if (i > 0) sb.Append(','); sb.Append(a[i].ToString("F2", CultureInfo.InvariantCulture)); }
        return sb.ToString();
    }
}
