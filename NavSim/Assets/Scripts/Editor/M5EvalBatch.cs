// Bridge-free headless entry point for the M5 paired eval. Runs via:
//   Unity -batchmode -projectPath NavSim -executeMethod M5EvalBatch.RunHeadless -logFile <log>
// (NO -quit: we need the game loop to run Play mode; we Exit(0) ourselves when done.)
// Trick: disable domain+scene reload on Play so the update-callback static state survives EnterPlaymode, then once
// Play + Academy are live we call the synchronous M5SearchEval.Run() (writes training/eval/m5_search.csv) and exit.
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class M5EvalBatch
{
    private static int _frames;

    public static void RunHeadless()
    {
        Debug.Log("[M5EvalBatch] starting headless eval — opening Training scene, entering Play mode");
        // batchmode opens an empty scene by default — explicitly load the eval scene so env/agent exist.
        EditorSceneManager.OpenScene("Assets/Scenes/Training.unity", OpenSceneMode.Single);
        EditorSettings.enterPlayModeOptionsEnabled = true;
        EditorSettings.enterPlayModeOptions =
            EnterPlayModeOptions.DisableDomainReload | EnterPlayModeOptions.DisableSceneReload;
        _frames = 0;
        EditorApplication.update += Tick;
        EditorApplication.EnterPlaymode();
    }

    private static void Tick()
    {
        if (!EditorApplication.isPlaying) return;      // wait until Play mode is actually live
        _frames++;
        if (_frames < 30) return;                       // give the scene + Academy ~30 frames to initialize
        EditorApplication.update -= Tick;
        int code = 0;
        try
        {
            M5SearchEval.Run();                          // synchronous: loops all 12 models x levels x episodes -> CSV
            Debug.Log("[M5EvalBatch] eval complete");
        }
        catch (System.Exception e)
        {
            Debug.LogError("[M5EvalBatch] eval FAILED: " + e);
            code = 1;
        }
        EditorApplication.Exit(code);
    }
}
