// Bridge-free headless entry points for the M7 GROUP-AWARE paired eval. ONE run per arm — all three share the
// SAME scene (Coop.unity) and shared policy; the arm selects the model slot + arena.ArmMode. All runs append to
// the shared training/eval/m7_coop.csv (idempotent per arm):
//   Unity -batchmode -projectPath NavSim -executeMethod M7EvalBatch.RunSelfish -logFile <log>
//   Unity -batchmode -projectPath NavSim -executeMethod M7EvalBatch.RunShared  -logFile <log>
//   Unity -batchmode -projectPath NavSim -executeMethod M7EvalBatch.RunPoca    -logFile <log>
//   Unity -batchmode -projectPath NavSim -executeMethod M7EvalBatch.SelftestHeadless -logFile <log>
// (NO -quit: Play mode needs the game loop; we Exit(0) ourselves.) DisableDomainReload keeps M7CoopEval's static
// state (Arm/ArmMode/Lessons/...) alive across EnterPlaymode.
//
// Lessons/episodes/seeds/csv are overridable via env vars so the SAME harness serves a quick probe
// (M7_EPISODES=2) and the full eval (defaults: C1-C3, 25/lesson, seeds 0-2).
using System;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using NavSim.Runtime;

public static class M7EvalBatch
{
    private const string CoopScene = "Assets/Scenes/Coop.unity";
    private static int _frames;
    private static bool _selftest;

    public static void RunSelfish() => Begin("selfish", ArmRouting.Arm.Selfish);
    public static void RunShared()  => Begin("shared",  ArmRouting.Arm.Shared);
    public static void RunPoca()    => Begin("poca",    ArmRouting.Arm.Poca);
    // No-model boundary self-test (paired seeding + EvalMode suppression + door-follows-plate). Arm-agnostic.
    public static void SelftestHeadless() { _selftest = true; Begin("poca", ArmRouting.Arm.Poca); }

    private static void Begin(string arm, ArmRouting.Arm armMode)
    {
        M7CoopEval.Arm = arm;
        M7CoopEval.ArmMode = armMode;
        ApplyEnvOverrides();
        Debug.Log($"[M7EvalBatch] arm={arm} scene={CoopScene} lessons=[{string.Join(",", M7CoopEval.Lessons)}] " +
                  $"eps/lesson={M7CoopEval.EpisodesPerLesson} seeds=[{string.Join(",", M7CoopEval.Seeds)}] csv={M7CoopEval.CsvName}");
        EditorSceneManager.OpenScene(CoopScene, OpenSceneMode.Single);
        EditorSettings.enterPlayModeOptionsEnabled = true;
        EditorSettings.enterPlayModeOptions =
            EnterPlayModeOptions.DisableDomainReload | EnterPlayModeOptions.DisableSceneReload;
        _frames = 0;
        EditorApplication.update += Tick;
        EditorApplication.EnterPlaymode();
    }

    private static void ApplyEnvOverrides()
    {
        int[] lessons = ParseInts(Environment.GetEnvironmentVariable("M7_LESSONS"));
        int[] seeds = ParseInts(Environment.GetEnvironmentVariable("M7_SEEDS"));
        if (lessons != null) M7CoopEval.Lessons = lessons;   // H1's lesson>=1 guard is re-checked in M7CoopEval.Run
        if (seeds != null) M7CoopEval.Seeds = seeds;
        if (int.TryParse(Environment.GetEnvironmentVariable("M7_EPISODES"), out int eps) && eps > 0)
            M7CoopEval.EpisodesPerLesson = eps;
        string csv = Environment.GetEnvironmentVariable("M7_CSV");
        if (!string.IsNullOrEmpty(csv)) M7CoopEval.CsvName = csv;
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
        try { if (_selftest) M7CoopEval.Selftest(); else M7CoopEval.Run(); Debug.Log("[M7EvalBatch] complete"); }
        catch (Exception e) { Debug.LogError("[M7EvalBatch] FAILED: " + e); code = 1; }
        EditorApplication.Exit(code);
    }
}
