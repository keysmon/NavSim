using System.Globalization;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;
using Unity.MLAgents;
using NavSim.Runtime;

// M3 off-diagonal generalization eval -- the reproducible artifact behind training/eval/m3_generalization.csv.
// USAGE: assign the trained NavAgent.onnx (Behaviour Type = Default, Inference Device = Burst), ENTER PLAY
// MODE, then run Tools/NavSim/Run M3 Generalization Eval (or call M3GeneralizationEval.Run()).
//
// It drives the trained Sentis policy over a grid of configs through the arena's independent setters
// (SetArenaSize/SetActiveCount/SetObstacleCount) and manual Academy steps -- the same interactive path
// used to produce the committed CSV. "diag-*" rows are configs the collapsed difficulty ladder trained as
// a tuple; "off-*" rows are held-out combinations it never trained together = the real generalization test.
public static class M3GeneralizationEval
{
    private static readonly (int agents, float half, int obs, string tag)[] Grid =
    {
        (2, 6f, 1, "diag-L0"), (4, 8f, 3, "diag-L1"), (6, 10f, 5, "diag-L2"), (8, 11f, 8, "diag-L3"),
        (8, 8f, 6, "off-crowd-med"),      // 8 agents (L3 count) in an L1-size arena
        (2, 11f, 1, "off-sparse-large"),  // 2 agents (L0 count) in the L3 arena
        (8, 9f, 6, "off-untrained-sz"),   // arena half-size 9 the ladder never used
        (2, 8f, 8, "off-dense-few"),      // L0 agent count with L3 obstacle density
    };

    private const int WarmupSteps = 60;
    private const int MeasureSteps = 5000; // goal rate is low (search-dominated) -> long window to cut noise
    private const float BodyWidth = 0.8f;   // overlap threshold
    private const float NearRadius = 1.6f;  // near-encounter threshold

    [MenuItem("Tools/NavSim/Run M3 Generalization Eval")]
    public static void Run()
    {
        if (!Application.isPlaying)
        {
            Debug.LogError("[M3Eval] Enter Play mode first (the eval needs a running Academy + inference policy).");
            return;
        }
        var env = Object.FindAnyObjectByType<NavEnvironment>();
        if (env == null) { Debug.LogError("[M3Eval] No NavEnvironment in the active scene."); return; }
        Physics.simulationMode = SimulationMode.Script;

        var sb = new StringBuilder(
            "tag,agents,half,obstacles,goals_per_agent_per_1k,near_frac,overlap_frac,min_approach,fallbacks\n");
        foreach (var g in Grid)
        {
            env.SetArenaSize(g.half);
            env.SetActiveCount(g.agents);
            env.SetObstacleCount(g.obs);
            env.RegenerateLayout();
            for (int w = 0; w < WarmupSteps; w++) Step();

            int goals0 = env.GoalsReachedTotal, fallbacks0 = env.PlacementFallbacks;
            var agents = Object.FindObjectsByType<NavAgent>(FindObjectsSortMode.None);
            int active = 0;
            foreach (var a in agents) if (a.gameObject.activeSelf) active++;

            int near = 0, overlap = 0; float minApproach = 999f;
            for (int s = 0; s < MeasureSteps; s++)
            {
                Step();
                float md = MinPairwise(agents);
                if (md < NearRadius) near++;
                if (md < BodyWidth) overlap++;
                if (md < minApproach) minApproach = md;
            }
            int goals = env.GoalsReachedTotal - goals0;
            float goalsPerAgentPer1k = goals / (float)Mathf.Max(active, 1) / (MeasureSteps / 1000f);
            sb.AppendFormat(CultureInfo.InvariantCulture,
                "{0},{1},{2},{3},{4:F2},{5:F3},{6:F4},{7:F2},{8}\n",
                g.tag, active, g.half, g.obs, goalsPerAgentPer1k,
                near / (float)MeasureSteps, overlap / (float)MeasureSteps, minApproach,
                env.PlacementFallbacks - fallbacks0);
        }
        // Application.dataPath == <project>/NavSim/Assets; the repo-root training/eval sits two levels up.
        string path = Path.GetFullPath(Path.Combine(Application.dataPath, "../../training/eval/m3_generalization.csv"));
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllText(path, sb.ToString());
        Debug.Log("[M3Eval] wrote " + path + "\n" + sb);
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
