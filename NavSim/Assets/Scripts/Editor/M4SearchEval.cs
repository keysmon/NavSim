using System.Globalization;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;   // InferenceDevice
using Unity.InferenceEngine;     // ModelAsset (package com.unity.ai.inference)
using NavSim.Runtime;

// M4 ablation eval -- the reproducible artifact behind training/eval/m4_search.csv.
// USAGE: place m4_primary/m4_nolstm/m4_nocuriosity/m4_baseline .onnx in Assets/Models/M4, ENTER PLAY MODE,
// then run Tools/NavSim/Run M4 Search Eval. It loops each arm (Agent.SetModel), fixes the hardest config
// (8 agents / arena 11 / 8 obstacles) and sweeps goal visibility from full-diagonal down to hidden,
// recording reach rate (primary metric across the sweep) + an early-window coverage (exploration RATE)
// proxy + the M2 collision proxy per (arm, visibility). Missing arm models are skipped with a warning,
// so a single-model smoke test works.
public static class M4SearchEval
{
    private static readonly string[] Arms =
        { "m4_primary", "m4_nolstm", "m4_nocuriosity", "m4_baseline" };
    private static readonly float[] VisFracs = { 1.0f, 0.6f, 0.35f, 0.2f }; // fraction of max diagonal

    private const int Agents = 8;
    private const float Half = 11f;
    private const int Obstacles = 8;
    private const int WarmupSteps = 60;
    private const int MeasureSteps = 5000;
    private const int CoverageWindow = 500; // early window: exploration RATE, before spatial saturation
    private const float CoverageCell = 1f;
    private const float BodyWidth = 0.8f;
    private const float NearRadius = 1.6f;

    [MenuItem("Tools/NavSim/Run M4 Search Eval")]
    public static void Run()
    {
        if (!Application.isPlaying)
        {
            Debug.LogError("[M4Eval] Enter Play mode first (needs a running Academy + Sentis policy).");
            return;
        }
        var env = Object.FindAnyObjectByType<NavEnvironment>();
        if (env == null) { Debug.LogError("[M4Eval] No NavEnvironment in the active scene."); return; }
        Physics.simulationMode = SimulationMode.Script;
        float maxDiag = DifficultyMapper.MaxArenaDiagonal;

        var sb = new StringBuilder(
            "arm,vis_frac,ray_len,goals_per_agent_per_1k,coverage_early_cells_per_agent,near_frac,overlap_frac,fallbacks\n");

        foreach (var arm in Arms)
        {
            var model = AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/Models/M4/" + arm + ".onnx");
            if (model == null) { Debug.LogWarning("[M4Eval] missing model " + arm + ", skipping."); continue; }

            env.SetArenaSize(Half);
            env.SetActiveCount(Agents);
            env.SetObstacleCount(Obstacles);
            env.RegenerateLayout();
            var agents = Object.FindObjectsByType<NavAgent>(FindObjectsSortMode.None);
            foreach (var a in agents) a.SetModel("NavAgent", model, InferenceDevice.Burst);

            foreach (var vf in VisFracs)
            {
                env.SetRayLength(vf * maxDiag);
                for (int w = 0; w < WarmupSteps; w++) Step();

                int goals0 = env.GoalsReachedTotal, fallbacks0 = env.PlacementFallbacks;
                int active = 0; foreach (var a in agents) if (a.gameObject.activeSelf) active++;

                int near = 0, overlap = 0;
                var cells = new HashSet<long>();
                for (int s = 0; s < MeasureSteps; s++)
                {
                    Step();
                    float md = MinPairwise(agents);
                    if (md < NearRadius) near++;
                    if (md < BodyWidth) overlap++;
                    if (s < CoverageWindow) // early-window exploration footprint (non-saturating)
                        foreach (var a in agents)
                        {
                            if (!a.gameObject.activeSelf) continue;
                            long cx = (long)Mathf.Floor(a.transform.position.x / CoverageCell);
                            long cz = (long)Mathf.Floor(a.transform.position.z / CoverageCell);
                            cells.Add((cx << 20) ^ (cz & 0xFFFFF));
                        }
                }
                int goals = env.GoalsReachedTotal - goals0;
                float goalsPer = goals / (float)Mathf.Max(active, 1) / (MeasureSteps / 1000f);
                float coverage = cells.Count / (float)Mathf.Max(active, 1);
                sb.AppendFormat(CultureInfo.InvariantCulture,
                    "{0},{1:F2},{2:F2},{3:F2},{4:F2},{5:F3},{6:F4},{7}\n",
                    arm, vf, vf * maxDiag, goalsPer, coverage,
                    near / (float)MeasureSteps, overlap / (float)MeasureSteps,
                    env.PlacementFallbacks - fallbacks0);
            }
        }
        string path = Path.GetFullPath(Path.Combine(Application.dataPath, "../../training/eval/m4_search.csv"));
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllText(path, sb.ToString());
        Debug.Log("[M4Eval] wrote " + path + "\n" + sb);
    }

    private static void Step()
    {
        Academy.Instance.EnvironmentStep();
        Physics.Simulate(Time.fixedDeltaTime);
    }

    private static float MinPairwise(NavAgent[] agents)
    {
        float min = 999f;
        for (int i = 0; i < agents.Length; i++)
        {
            if (!agents[i].gameObject.activeSelf) continue;
            for (int j = i + 1; j < agents.Length; j++)
            {
                if (!agents[j].gameObject.activeSelf) continue;
                Vector3 a = agents[i].transform.position, b = agents[j].transform.position;
                a.y = 0f; b.y = 0f;
                float d = Vector3.Distance(a, b);
                if (d < min) min = d;
            }
        }
        return min;
    }
}
