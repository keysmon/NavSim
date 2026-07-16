using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;   // InferenceDevice
using Unity.InferenceEngine;     // ModelAsset (com.unity.ai.inference)
using NavSim.Runtime;

// M6 PAIRED ablation eval — fork of M5SearchEval with the appearance-task deltas. The reproducible artifact behind
// training/eval/m6_search.csv. Because the three arms differ in SENSORS (pixel camera vs ray fans), each arm is
// evaluated in its OWN scene (Training_pixel/Training_ray1/Training_rayc), one batchmode run per arm (M6EvalBatch),
// all writing to ONE m6_search.csv IDEMPOTENTLY per arm (each run preserves other arms' rows and rewrites its own —
// re-running an arm self-heals; no duplicate accretion). PAIRING holds across the separate runs: for each (seed, level, episode) we seed
// BOTH UnityEngine.Random (terrain + triad placement) AND env.SeedColorRng (colors + target) with the same value,
// and neither the terrain nor the color draw depends on the sensor type -> every arm sees an identical
// (layout, colors, target, positions) tuple.
//
// M6 metric = the CUED target. The harness captures target0 + the two decoy0 positions as Vector3 VALUES right
// after placement, and detects outcomes GEOMETRICALLY and ALWAYS HARD (independent of the training soften->harden):
//   success    = first reach within GoalRadius of target0;
//   decoy_visit = 1 and STOP if the agent first comes within GoalRadius of either decoy0 (a wrong-color pick).
//
// LIFECYCLE (from M5, EXTENDED for M6): the harness OWNS the episode/level boundary so OnEpisodeBegin fires once and
// never re-rolls mid-episode — this needs BOTH agent.MaxStep=0 (no max-step EndEpisode) AND env.EvalMode=true (M6
// added a decoy-touch EndEpisode M5 never had; EvalMode suppresses it so a decoy doesn't re-roll the arena/cue).
// SetTerrainLevel(lvl) is the sole layout driver. SetModel-ORDERING FIX (Phase-0.2): step the agent once
// (EnvironmentStep) BEFORE the first SetModel so the pixel CameraSensor's texture2D is allocated.
public static class M6SearchEval
{
    // Set by M6EvalBatch before entering Play (static -> survives DisableDomainReload). Arm selects the ONNX prefix;
    // the LOADED scene must match (M6EvalBatch loads it). Levels/EpisodesPerLevel are overridable for the Phase-5
    // probe (e.g. Levels={0}, EpisodesPerLevel=100) vs the full eval (Levels={0,1,2,3}, EpisodesPerLevel=25).
    public static string Arm = "m6_pixel";
    public static int[] Seeds = { 0, 1, 2 };            // n=3, pre-registered add-to-5 (M5 discipline)
    public static int[] Levels = { 0, 1, 2, 3 };
    public static int EpisodesPerLevel = 25;
    public static string CsvName = "m6_search.csv";

    private const int MaxSteps = 3000;
    private const float NearRadius = 1.6f;               // agent-mover proximity ("near")
    private const float BodyWidth = 0.8f;                // agent-mover overlap (collision proxy)

    private struct EpMetrics
    {
        public bool success;
        public int decoyVisit;
        public int steps;
        public float pathLen;
        public int pitFalls;
        public int jumpUses;
        public float nearFrac;
        public float overlapFrac;
    }

    [MenuItem("Tools/NavSim/Run M6 Search Eval (current arm)")]
    public static void Run()
    {
        if (!Application.isPlaying) { Debug.LogError("[M6Eval] Enter Play mode first (needs a running Academy)."); return; }
        var env = Object.FindAnyObjectByType<NavEnvironment>();
        var terrain = Object.FindAnyObjectByType<TerrainGenerator>();
        var agent = Object.FindAnyObjectByType<NavAgent>();
        if (env == null || terrain == null || agent == null) { Debug.LogError("[M6Eval] Missing env/terrain/agent."); return; }
        if (EpisodesPerLevel > 100)
            Debug.LogWarning($"[M6Eval] EpisodesPerLevel={EpisodesPerLevel} > 100 breaks the paired seed " +
                             "(ep overflows the *100 level stride → colors reuse across levels). Keep it <= 100.");
        SetupHarness(env, agent);

        // IDEMPOTENT per arm: preserve OTHER arms' rows, REPLACE this arm's rows (re-running an arm self-heals
        // regardless of orchestration order — no silent duplicate accretion, no fragile "rm once before the first
        // arm" operator contract). The three arm runs are sequential (one Unity lock), so read-modify-write is safe.
        string path = CsvPath(CsvName);
        const string Header = "arm,seed,level,episode,success,spl,steps_to_goal,decoy_visit,pit_falls,jump_uses,near_frac,overlap_frac";
        var sb = new StringBuilder();
        sb.Append(Header).Append('\n');
        if (File.Exists(path))
            foreach (var line in File.ReadAllLines(path))
            {
                if (string.IsNullOrWhiteSpace(line) || line.StartsWith("arm,")) continue; // drop blanks + old header
                if (line.StartsWith(Arm + ",")) continue;                                  // drop THIS arm's stale rows
                sb.Append(line).Append('\n');                                              // keep other arms' rows
            }

        // Per-arm pairing fingerprint sidecar (arm-independent layout/colors/sp) — diffed across arms at Stage B.
        var fp = new StringBuilder("seed,level,episode,tx,ty,tz,d0x,d0y,d0z,d1x,d1y,d1z,cr,cg,cb,sp\n");

        int ran = 0;
        foreach (var seed in Seeds)
        {
            var model = AssetDatabase.LoadAssetAtPath<ModelAsset>($"Assets/Models/M6/{Arm}_s{seed}.onnx");
            if (model == null) { Debug.LogWarning($"[M6Eval] missing {Arm}_s{seed}, skipping"); continue; }
            agent.SetModel("NavAgent", model, InferenceDevice.Burst);
            foreach (var lvl in Levels)
                for (int ep = 0; ep < EpisodesPerLevel; ep++)
                {
                    AppendEpisode(env, terrain, agent, Arm, seed, lvl, ep, sb, fp);
                    ran++;
                }
        }
        WriteCsv(sb, path);
        if (ran > 0) File.WriteAllText(CsvPath($"m6_pairing_{Arm}.csv"), fp.ToString());
        Debug.Log($"[M6Eval] arm={Arm}: wrote {ran} episodes to {path} (+ pairing fingerprint). If 0, place ONNX in Assets/Models/M6/{Arm}_s<seed>.onnx.");
    }

    // No-model verify-early (fork of M5's). Confirms (1) PAIRED seeding reproduces identical (target, decoys, colors)
    // across two calls with the same seed; (2) a forced decoy-touch is recorded decoy_visit=1, success=0; (3) a
    // well-formed CSV row. Run in Play, no models needed.
    [MenuItem("Tools/NavSim/M6 Eval Selftest (no model)")]
    public static void Selftest()
    {
        if (!Application.isPlaying) { Debug.LogError("[M6Eval] Enter Play mode first."); return; }
        var env = Object.FindAnyObjectByType<NavEnvironment>();
        var terrain = Object.FindAnyObjectByType<TerrainGenerator>();
        var agent = Object.FindAnyObjectByType<NavAgent>();
        if (env == null || terrain == null || agent == null) { Debug.LogError("[M6Eval] Missing env/terrain/agent."); return; }
        SetupHarness(env, agent);

        // (1) Paired reproducibility: same seed -> identical target + decoys + cue, twice.
        (Vector3 t1, Vector3[] d1, Color c1) = PlaceSeeded(env, agent, 123);
        (Vector3 t2, Vector3[] d2, Color c2) = PlaceSeeded(env, agent, 123);
        bool paired = Approximately(t1, t2) && d1.Length == d2.Length && Approximately(d1[0], d2[0])
                      && Approximately(d1[1], d2[1]) && ColorApprox(c1, c2);
        // A DIFFERENT seed should (almost always) differ -> guards a stuck RNG.
        (Vector3 t3, _, Color c3) = PlaceSeeded(env, agent, 999);
        bool varies = !Approximately(t1, t3) || !ColorApprox(c1, c3);

        // (2) Outcome resolution (pure Resolve — the SAME function RunEpisode uses; no physics/ejection confound):
        //   a point ON a decoy resolves to Decoy (not success); a point ON the target resolves to Success; and the
        //   goals are separated enough that a point on one never triggers another (unambiguous, always-hard).
        (Vector3 tgt, Vector3[] dcy, _) = PlaceSeeded(env, agent, 55);
        bool onDecoy = Resolve(env, dcy[0], tgt, dcy) == Outcome.Decoy;
        bool onTarget = Resolve(env, tgt, tgt, dcy) == Outcome.Success;
        bool separated = !WithinGoal(env, tgt, dcy[0]) && !WithinGoal(env, tgt, dcy[1]);
        bool outcomeOk = onDecoy && onTarget && separated;

        // (4) EvalMode suppresses the decoy EndEpisode (the CRITICAL fix). At L1 — where DecoyRules.DecoyEndsEpisode
        // is true — put a decoy right in front of the agent, drive ONE real step, and confirm the decoy path FIRED
        // (TouchedDecoy) yet the target did NOT re-roll (GoalPositionFor unchanged). Without EvalMode this step
        // would EndEpisode → OnEpisodeBegin → re-roll the arena/cue mid-episode → corrupt the metric.
        UnityEngine.Random.InitState(77); env.SeedColorRng(77);
        env.SetTerrainLevel(1);
        Physics.SyncTransforms();
        Vector3 targetL1 = env.GoalPositionFor(agent);
        var goals = FindGoalObjects();   // tag-agnostic: ray1 "goal" OR rayC "goal_c{k}"
        GameObject targetGo = null, decoyGo = null; float best = float.MaxValue;
        foreach (var g in goals)
        { float dd = (g.transform.position - targetL1).sqrMagnitude; if (dd < best) { best = dd; targetGo = g; } }
        foreach (var g in goals) if (g != targetGo) { decoyGo = g; break; }
        bool eval4Ran = decoyGo != null, touched = false, noReroll = false;
        if (eval4Ran)
        {
            decoyGo.transform.position = agent.transform.position + agent.transform.forward * 1.3f; // within GoalRadius, clear of the collider
            Physics.SyncTransforms();
            Step();
            touched = env.TouchedDecoy(agent);
            noReroll = Approximately(env.GoalPositionFor(agent), targetL1);
        }
        bool evalModeOk = !eval4Ran || (touched && noReroll); // no goals in scene -> N/A (pass); else assert

        var sb = new StringBuilder(
            "arm,seed,level,episode,success,spl,steps_to_goal,decoy_visit,pit_falls,jump_uses,near_frac,overlap_frac\n");
        AppendEpisode(env, terrain, agent, "selftest", 0, 0, 0, sb);

        Debug.Log($"[M6Eval] SELFTEST — paired(same seed)={paired} (MUST be true); varies(diff seed)={varies} (MUST be true); " +
                  $"outcome resolution: on-decoy→Decoy={onDecoy}, on-target→Success={onTarget}, goals-separated={separated} → {outcomeOk} (MUST be true); " +
                  $"EvalMode-suppresses-reroll: ran={eval4Ran} decoy-fired={touched} & target-stable={noReroll} → {evalModeOk} (MUST be true). Sample row:\n{sb}");
    }

    private static void SetupHarness(NavEnvironment env, NavAgent agent)
    {
        Physics.simulationMode = SimulationMode.Script;
        agent.MaxStep = 0;      // harness owns the boundary; no OnEpisodeBegin max-step re-roll
        env.EvalMode = true;    // AND a decoy touch must NOT EndEpisode (that re-rolls the arena/cue mid-episode);
                                // the harness detects reach-vs-decoy geometrically, always hard, across all arms
        Step();                 // SetModel-ordering fix: one step BEFORE any SetModel so the CameraSensor texture allocates
    }

    // Seed BOTH RNGs identically, place a fresh triad + colors, return the captured target/decoys/cue as VALUES.
    private static (Vector3 target, Vector3[] decoys, Color cue) PlaceSeeded(NavEnvironment env, NavAgent agent, int seed)
    {
        UnityEngine.Random.InitState(seed);
        env.SeedColorRng(seed);
        env.SetTerrainLevel(env.CurrentLevel); // keep the current level (CurrentLevel already clamps <0 → 0); re-place the triad
        Physics.SyncTransforms();
        return (env.GoalPositionFor(agent), env.DecoyPositions(), env.TargetColorRgb);
    }

    // Re-seed for a paired layout, capture goal0/decoys as VALUES, run, record. SPL denom = NavMesh oracle path.
    private static void AppendEpisode(NavEnvironment env, TerrainGenerator terrain, NavAgent agent,
        string arm, int seed, int lvl, int ep, StringBuilder sb, StringBuilder fp = null)
    {
        int paired = seed * 10000 + lvl * 100 + ep;
        UnityEngine.Random.InitState(paired);   // terrain + triad placement
        env.SeedColorRng(paired);               // colors + target (independent System.Random)
        env.SetTerrainLevel(lvl);
        Physics.SyncTransforms();               // simulationMode=Script defers transform->physics sync; push it so
                                                // the first step's perception reads the freshly-placed geometry
        Vector3 start = agent.transform.position;
        Vector3 target0 = env.GoalPositionFor(agent);   // Vector3 value copies — never the live goal Transforms
        Vector3[] decoy0 = env.DecoyPositions();
        Color cue = env.TargetColorRgb;
        float sp = terrain.ShortestPathLength(start, target0);
        // PAIRING fingerprint (arm-INDEPENDENT): the layout+colors+NavMesh-oracle for this (seed,lvl,ep). Diffing
        // the three arms' m6_pairing_<arm>.csv streams must be byte-identical -> proves the paired-eval guarantee
        // (M1 layout/colors) AND deterministic NavMesh bake across arms (M2, via sp). Verify at Stage B.
        fp?.AppendLine(string.Format(CultureInfo.InvariantCulture,
            "{0},{1},{2},{3:F3},{4:F3},{5:F3},{6:F3},{7:F3},{8:F3},{9:F3},{10:F3},{11:F3},{12:F3},{13:F3},{14:F3},{15:F4}",
            seed, lvl, ep, target0.x, target0.y, target0.z, decoy0[0].x, decoy0[0].y, decoy0[0].z,
            decoy0[1].x, decoy0[1].y, decoy0[1].z, cue.r, cue.g, cue.b, sp));
        EpMetrics m = RunEpisode(env, agent, target0, decoy0);
        float spl = (m.success && m.pathLen > 0f && sp > 0f)
            ? Mathf.Min(1f, sp / Mathf.Max(m.pathLen, sp)) : 0f;
        sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
            "{0},{1},{2},{3},{4},{5:F4},{6},{7},{8},{9},{10:F4},{11:F4}",
            arm, seed, lvl, ep, m.success ? 1 : 0, spl, m.steps, m.decoyVisit, m.pitFalls, m.jumpUses,
            m.nearFrac, m.overlapFrac));
    }

    private static EpMetrics RunEpisode(NavEnvironment env, NavAgent agent, Vector3 target0, Vector3[] decoy0)
    {
        int pit0 = env.PitFalls, jump0 = agent.JumpUses, pitPrev = env.PitFalls;
        Vector3 prev = agent.transform.position;
        float pathLen = 0f;
        int near = 0, overlap = 0, steps = 0;
        bool success = false, decoy = false;
        for (int s = 0; s < MaxSteps; s++)
        {
            Step();
            steps = s + 1;
            Vector3 pos = agent.transform.position;
            if (env.PitFalls > pitPrev) { pitPrev = env.PitFalls; prev = pos; } // respawn teleport: not travel
            else { pathLen += Vector3.Distance(pos, prev); prev = pos; }
            float md = MinMoverDist(pos, env.MoverPositions());
            if (md < NearRadius) near++;
            if (md < BodyWidth) overlap++;
            // ALWAYS-HARD geometric outcome on captured VALUES (pure Resolve, also unit-tested in Selftest).
            Outcome o = Resolve(env, pos, target0, decoy0);
            if (o == Outcome.Decoy) { decoy = true; break; }
            if (o == Outcome.Success) { success = true; break; }
        }
        return new EpMetrics
        {
            success = success,
            decoyVisit = decoy ? 1 : 0,
            steps = steps,
            pathLen = pathLen,
            pitFalls = env.PitFalls - pit0,
            jumpUses = agent.JumpUses - jump0,
            nearFrac = near / (float)Mathf.Max(steps, 1),
            overlapFrac = overlap / (float)Mathf.Max(steps, 1)
        };
    }

    private enum Outcome { None, Success, Decoy }

    // Pure, always-HARD outcome resolution on captured VALUES — the single source of truth for both RunEpisode and
    // the Selftest. Decoy is checked FIRST so a tie (agent within radius of both) resolves to the (hard) failure,
    // the pessimistic/honest call. Goals sit ~goalClusterSpread*sqrt(3) (~4.5u) apart, so ties are effectively impossible.
    private static Outcome Resolve(NavEnvironment env, Vector3 pos, Vector3 target0, Vector3[] decoy0)
    {
        if (WithinGoal(env, pos, decoy0[0]) || WithinGoal(env, pos, decoy0[1])) return Outcome.Decoy;
        if (WithinGoal(env, pos, target0)) return Outcome.Success;
        return Outcome.None;
    }

    private static bool WithinGoal(NavEnvironment env, Vector3 pos, Vector3 goal) =>
        Vector3.Distance(pos, goal) < env.GoalRadius;

    // Tag-agnostic goal lookup for the Selftest: ray1 tags goals "goal"; rayC tags them "goal_c{k}". (All tags are
    // defined in TagManager, so FindGameObjectsWithTag never throws.) Avoids a false-FAIL if the Selftest menu item
    // is run interactively in the rayC scene.
    private static List<GameObject> FindGoalObjects()
    {
        var list = new List<GameObject>(GameObject.FindGameObjectsWithTag("goal"));
        for (int k = 0; k < GoalPalette.Colors.Length; k++)
            list.AddRange(GameObject.FindGameObjectsWithTag(GoalPalette.Tag(k)));
        return list;
    }

    private static void Step()
    {
        Academy.Instance.EnvironmentStep();
        Physics.Simulate(Time.fixedDeltaTime);
    }

    private static float MinMoverDist(Vector3 p, IReadOnlyList<Vector3> movers)
    {
        float min = 999f;
        if (movers == null) return min;
        p.y = 0f;
        for (int i = 0; i < movers.Count; i++)
        {
            Vector3 m = movers[i]; m.y = 0f;
            float d = Vector3.Distance(p, m);
            if (d < min) min = d;
        }
        return min;
    }

    private static bool Approximately(Vector3 a, Vector3 b) => (a - b).sqrMagnitude < 1e-6f;
    private static bool ColorApprox(Color a, Color b) =>
        Mathf.Abs(a.r - b.r) + Mathf.Abs(a.g - b.g) + Mathf.Abs(a.b - b.b) < 1e-4f;

    private static string CsvPath(string name) =>
        Path.GetFullPath(Path.Combine(Application.dataPath, "../../training/eval/" + name));

    // Writes the full reconciled CSV (header + preserved other-arm rows + this arm's fresh rows). Clobber, not
    // append — the idempotent read-modify-write in Run() already merged the prior content.
    private static void WriteCsv(StringBuilder sb, string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllText(path, sb.ToString());
        Debug.Log("[M6Eval] wrote -> " + path);
    }
}
