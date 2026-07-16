// Bridge-free headless entry points for the M6 paired eval. ONE run per arm (each loads that arm's scene, since the
// arms differ in sensors, and appends to the shared training/eval/m6_search.csv):
//   Unity -batchmode -projectPath NavSim -executeMethod M6EvalBatch.RunPixel -logFile <log>
//   Unity -batchmode -projectPath NavSim -executeMethod M6EvalBatch.RunRay1  -logFile <log>
//   Unity -batchmode -projectPath NavSim -executeMethod M6EvalBatch.RunRayC  -logFile <log>
// (NO -quit: Play mode needs the game loop; we Exit(0) ourselves.) DisableDomainReload keeps the update-callback
// static state (+ M6SearchEval.Arm/Levels/…) alive across EnterPlaymode.
//
// Levels / episodes / seeds are overridable via env vars so the SAME harness serves the Phase-5 PROBE
// (M6_LEVELS=0 M6_EPISODES=100 M6_SEEDS=0) and the full eval (defaults: L0-3, 25/level, seeds 0-2). CsvName too.
using System;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using NavSim.Runtime;

public static class M6EvalBatch
{
    private static int _frames;
    private static bool _selftest;

    public static void RunPixel() => Begin("m6_pixel", "Assets/Scenes/Training_pixel.unity");
    public static void RunRay1()  => Begin("m6_ray1",  "Assets/Scenes/Training_ray1.unity");
    public static void RunRayC()  => Begin("m6_rayc",  "Assets/Scenes/Training_rayc.unity");
    // No-model self-test (paired seeding + decoy detection + CSV format). Runs on the ray1 scene (arm-agnostic).
    public static void SelftestHeadless() { _selftest = true; Begin("m6_ray1", "Assets/Scenes/Training_ray1.unity"); }

    private static void Begin(string arm, string scene)
    {
        M6SearchEval.Arm = arm;
        ApplyEnvOverrides();
        Debug.Log($"[M6EvalBatch] arm={arm} scene={scene} levels=[{string.Join(",", M6SearchEval.Levels)}] " +
                  $"eps/level={M6SearchEval.EpisodesPerLevel} seeds=[{string.Join(",", M6SearchEval.Seeds)}] csv={M6SearchEval.CsvName}");
        EditorSceneManager.OpenScene(scene, OpenSceneMode.Single);
        EditorSettings.enterPlayModeOptionsEnabled = true;
        EditorSettings.enterPlayModeOptions =
            EnterPlayModeOptions.DisableDomainReload | EnterPlayModeOptions.DisableSceneReload;
        _frames = 0;
        EditorApplication.update += Tick;
        EditorApplication.EnterPlaymode();
    }

    private static void ApplyEnvOverrides()
    {
        int[] levels = ParseInts(Environment.GetEnvironmentVariable("M6_LEVELS"));
        int[] seeds = ParseInts(Environment.GetEnvironmentVariable("M6_SEEDS"));
        if (levels != null) M6SearchEval.Levels = levels;
        if (seeds != null) M6SearchEval.Seeds = seeds;
        if (int.TryParse(Environment.GetEnvironmentVariable("M6_EPISODES"), out int eps) && eps > 0)
            M6SearchEval.EpisodesPerLevel = eps;
        string csv = Environment.GetEnvironmentVariable("M6_CSV");
        if (!string.IsNullOrEmpty(csv)) M6SearchEval.CsvName = csv;
    }

    private static int[] ParseInts(string csv)
    {
        if (string.IsNullOrWhiteSpace(csv)) return null;
        var parts = csv.Split(',');
        var list = new System.Collections.Generic.List<int>();
        foreach (var p in parts) if (int.TryParse(p.Trim(), out int v)) list.Add(v);
        return list.Count > 0 ? list.ToArray() : null;
    }

    private static void Tick()
    {
        if (!EditorApplication.isPlaying) return;      // wait for Play + Academy
        _frames++;
        if (_frames < 30) return;                       // ~30 frames to initialize the scene + Academy
        EditorApplication.update -= Tick;
        int code = 0;
        try { if (_selftest) M6SearchEval.Selftest(); else M6SearchEval.Run(); Debug.Log("[M6EvalBatch] complete"); }
        catch (Exception e) { Debug.LogError("[M6EvalBatch] FAILED: " + e); code = 1; }
        EditorApplication.Exit(code);
    }
}
