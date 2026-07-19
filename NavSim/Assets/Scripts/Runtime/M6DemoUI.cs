using UnityEngine;

namespace NavSim.Runtime
{
    // M6 v2 demo control (WebGL / standalone — NO trainer). "New layout" re-rolls terrain + ring placement +
    // decoy colours at the CURRENT level; the fixed-red policy re-finds red. A difficulty row (L0-L3) drives the
    // M5 terrain ladder via NavEnvironment.SetTerrainLevel. In a no-communicator build the env HOLDS the level the
    // UI sets — NavEnvironment.PlaceForNewEpisode uses CurrentLevel when Academy.IsCommunicatorOn is false — so a
    // chosen level persists across episode resets. (Replaces the cued design's cue-selector: the target is fixed
    // red, so there is nothing to announce.)
    public class M6DemoUI : MonoBehaviour
    {
        [SerializeField] private NavEnvironment env;

        // Open the demo on the easiest, most legible rung (a close, co-visible cluster). Belt-and-suspenders with
        // NavEnvironment's communicator-off default of L0 — makes the demo's opening level explicit at this layer,
        // independent of Start()-ordering between the env and this component.
        private void Start()
        {
            if (env != null) env.SetTerrainLevel(0);
        }

        private void OnGUI()
        {
            // Grouping panel behind the controls (drawn first; non-interactive, so it never eats button clicks).
            GUI.Box(new Rect(6f, 6f, 300f, 118f), GUIContent.none);

            // KEEP this button EXACTLY at (10,10,140,30): the WebGL smoke test clicks it by canvas coordinates
            // (IMGUI is not DOM). Re-seed UnityEngine.Random so the new layout is genuinely different each press.
            if (GUI.Button(new Rect(10f, 10f, 140f, 30f), "New layout"))
            {
                Random.InitState(System.Environment.TickCount);
                if (env != null) env.SetTerrainLevel(env.CurrentLevel);
            }

            var title = new GUIStyle(GUI.skin.label) { richText = true, fontStyle = FontStyle.Bold };
            GUI.Label(new Rect(158f, 14f, 150f, 24f), "Find the <color=red>RED</color> goal", title);

            // Difficulty selector: L0-L3 on the M5 terrain ladder. The active level is highlighted.
            int level = env != null ? env.CurrentLevel : 0;
            for (int i = 0; i < 4; i++)
            {
                GUI.color = (i == level) ? Color.cyan : Color.white;
                if (GUI.Button(new Rect(10f + i * 35f, 48f, 33f, 28f), "L" + i) && env != null)
                    env.SetTerrainLevel(i);
            }
            GUI.color = Color.white;

            GUI.Label(new Rect(158f, 52f, 150f, 24f), "Difficulty");
            GUI.Label(new Rect(10f, 84f, 290f, 24f),
                "Level L" + level + " — the CNN goes to red; decoys are traps.");
        }
    }
}
