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

// M5 PAIRED ablation eval — the reproducible artifact behind training/eval/m5_search.csv.
// Each (arm, seed) ONNX runs the SAME held-out layout set (RNG re-seeded per episode so every arm sees an
// identical baked layout) across the L0..L3 sweep, recording SPL (NavMesh oracle) + success + steps + pit-falls
// + jump-use + mover proximity. Single-goal metric: the harness measures against a CAPTURED goal0 (a Vector3
// value), detects the first reach itself, and breaks — decoupled from the agent's continuous goal respawn.
//
// USAGE: place arm models at Assets/Models/M5/<arm>_s<seed>.onnx, ENTER PLAY MODE, run Tools/NavSim/Run M5
// Search Eval. Missing arm/seed models are skipped with a warning (a single-model smoke works).
//
// LIFECYCLE (critical): the harness OWNS the episode/level boundary. It sets agent.MaxStep = 0 so the agent's
// OnEpisodeBegin (which would re-roll SetTerrainLevel(ReadDifficulty()) -> level 3 in the no-communicator Editor)
// fires only ONCE, and never calls EndEpisode between episodes. SetTerrainLevel(lvl) is the sole layout driver.
public static class M5SearchEval
{
    private static readonly string[] Arms = { "m5_primary", "m5_nolstm", "m5_nornd", "m5_baseline" };
    private static readonly int[] Seeds = { 0, 1, 2 };   // shared across arms (paired)
    private static readonly int[] Levels = { 0, 1, 2, 3 };
    private const int EpisodesPerLevel = 25;             // 4 levels * 25 = 100 held-out / seed
    private const int MaxSteps = 3000;
    private const float NearRadius = 1.6f;               // agent-mover proximity ("near")
    private const float BodyWidth = 0.8f;                // agent-mover overlap (collision proxy)

    private struct EpMetrics
    {
        public bool success;
        public int steps;
        public float pathLen;
        public int pitFalls;
        public int jumpUses;
        public float nearFrac;
        public float overlapFrac;
    }

    [MenuItem("Tools/NavSim/Run M5 Search Eval")]
    public static void Run()
    {
        if (!Application.isPlaying) { Debug.LogError("[M5Eval] Enter Play mode first (needs a running Academy)."); return; }
        var env = Object.FindAnyObjectByType<NavEnvironment>();
        var terrain = Object.FindAnyObjectByType<TerrainGenerator>();
        var agent = Object.FindAnyObjectByType<NavAgent>();
        if (env == null || terrain == null || agent == null) { Debug.LogError("[M5Eval] Missing env/terrain/agent."); return; }
        SetupHarness(agent);

        var sb = new StringBuilder(
            "arm,seed,level,episode,success,spl,steps_to_goal,pit_falls,jump_uses,near_frac,overlap_frac\n");
        int ran = 0;
        foreach (var arm in Arms)
            foreach (var seed in Seeds)
            {
                var model = AssetDatabase.LoadAssetAtPath<ModelAsset>($"Assets/Models/M5/{arm}_s{seed}.onnx");
                if (model == null) { Debug.LogWarning($"[M5Eval] missing {arm}_s{seed}, skipping"); continue; }
                agent.SetModel("NavAgent", model, InferenceDevice.Burst);
                foreach (var lvl in Levels)
                    for (int ep = 0; ep < EpisodesPerLevel; ep++)
                    {
                        AppendEpisode(env, terrain, agent, arm, seed, lvl, ep, sb);
                        ran++;
                    }
            }
        WriteCsv(sb, "m5_search.csv");
        Debug.Log($"[M5Eval] wrote m5_search.csv ({ran} episodes). If 0, place ONNX models in Assets/Models/M5.");
    }

    // No-model verify-early. Confirms (1) the level-lifecycle fix — at L0, stepping PAST the agent's old
    // MaxStep (2000) must NOT re-roll structure (walls/platforms/pits stay 0); (2) manual stepping is live
    // (StepCount advances); (3) a single episode produces a well-formed CSV row. Run in Play, no models needed.
    [MenuItem("Tools/NavSim/M5 Eval Selftest (no model)")]
    public static void Selftest()
    {
        if (!Application.isPlaying) { Debug.LogError("[M5Eval] Enter Play mode first."); return; }
        var env = Object.FindAnyObjectByType<NavEnvironment>();
        var terrain = Object.FindAnyObjectByType<TerrainGenerator>();
        var agent = Object.FindAnyObjectByType<NavAgent>();
        if (env == null || terrain == null || agent == null) { Debug.LogError("[M5Eval] Missing env/terrain/agent."); return; }
        SetupHarness(agent);

        // (1) Level-lifecycle discriminator: L0 has 0 walls / 0 platforms / 0 pits. Step past 2500 (> old
        // MaxStep 2000). With MaxStep=0 there is no OnEpisodeBegin re-roll, so the structure must stay empty.
        UnityEngine.Random.InitState(12345);
        env.SetTerrainLevel(0);
        int startStep = agent.StepCount;
        for (int i = 0; i < 2500; i++) Step();
        int active = CountActiveStructure(terrain);
        int stepAdvance = agent.StepCount - startStep;

        // (3) one L2 episode -> a well-formed row (success likely 0 with no policy; the FORMAT + SPL math + delta
        // counters are what is checked).
        var sb = new StringBuilder(
            "arm,seed,level,episode,success,spl,steps_to_goal,pit_falls,jump_uses,near_frac,overlap_frac\n");
        AppendEpisode(env, terrain, agent, "selftest", 0, 2, 0, sb);

        Debug.Log($"[M5Eval] SELFTEST — L0-after-2500-steps active-structure = {active} (MUST be 0; nonzero = " +
                  $"level re-roll bug); stepCount advanced {stepAdvance} (must be >0). Sample row:\n{sb}");
    }

    private static void SetupHarness(NavAgent agent)
    {
        Physics.simulationMode = SimulationMode.Script;
        agent.MaxStep = 0; // harness owns the episode boundary; no OnEpisodeBegin level re-roll
    }

    // Re-seed for a paired layout, place the agent + goal, capture goal0 as a VALUE, run, record.
    private static void AppendEpisode(NavEnvironment env, TerrainGenerator terrain, NavAgent agent,
        string arm, int seed, int lvl, int ep, StringBuilder sb)
    {
        UnityEngine.Random.InitState(seed * 10000 + lvl * 100 + ep); // SAME baked layout across arms (paired)
        env.SetTerrainLevel(lvl);
        Vector3 start = agent.transform.position;
        Vector3 goal0 = env.GoalPositionFor(agent);      // Vector3 value copy — never the goal Transform
        float sp = terrain.ShortestPathLength(start, goal0);
        EpMetrics m = RunEpisode(env, agent, goal0);
        float spl = (m.success && m.pathLen > 0f && sp > 0f)
            ? Mathf.Min(1f, sp / Mathf.Max(m.pathLen, sp)) : 0f;
        sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
            "{0},{1},{2},{3},{4},{5:F4},{6},{7},{8},{9:F4},{10:F4}",
            arm, seed, lvl, ep, m.success ? 1 : 0, spl, m.steps, m.pitFalls, m.jumpUses,
            m.nearFrac, m.overlapFrac));
    }

    private static EpMetrics RunEpisode(NavEnvironment env, NavAgent agent, Vector3 goal0)
    {
        int pit0 = env.PitFalls, jump0 = agent.JumpUses, pitPrev = env.PitFalls;
        Vector3 prev = agent.transform.position;
        float pathLen = 0f;
        int near = 0, overlap = 0, steps = 0;
        bool success = false;
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
            if (Vector3.Distance(pos, goal0) < env.GoalRadius) { success = true; break; } // first reach -> stop
        }
        return new EpMetrics
        {
            success = success,
            steps = steps,
            pathLen = pathLen,
            pitFalls = env.PitFalls - pit0,
            jumpUses = agent.JumpUses - jump0,
            nearFrac = near / (float)Mathf.Max(steps, 1),
            overlapFrac = overlap / (float)Mathf.Max(steps, 1)
        };
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

    // Active interior walls + active platforms + hidden (inactive) pit tiles — all 0 at L0.
    private static int CountActiveStructure(TerrainGenerator terrain)
    {
        var arena = GameObject.Find("Arena");
        if (arena == null) return -1;
        int n = 0;
        n += CountActiveChildren(arena.transform.Find("WallPool"), true);
        n += CountActiveChildren(arena.transform.Find("PlatformPool"), true);
        // hidden pit tiles: count inactive floor tiles (L0 hides none -> 0; an L3 re-roll would hide 3)
        var floor = arena.transform.Find("FloorTiles");
        if (floor != null)
            foreach (Transform t in floor)
                if (!t.gameObject.activeSelf) n++;
        return n;
    }

    private static int CountActiveChildren(Transform t, bool activeOnly)
    {
        if (t == null) return 0;
        int n = 0;
        foreach (Transform c in t) if (!activeOnly || c.gameObject.activeSelf) n++;
        return n;
    }

    private static void WriteCsv(StringBuilder sb, string name)
    {
        string path = Path.GetFullPath(Path.Combine(Application.dataPath, "../../training/eval/" + name));
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllText(path, sb.ToString());
        Debug.Log("[M5Eval] wrote " + path);
    }
}
