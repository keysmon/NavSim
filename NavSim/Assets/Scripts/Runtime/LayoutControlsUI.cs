using UnityEngine;

namespace NavSim.Runtime
{
    // Demo controls that make M3 generalization interactive: a difficulty slider (drives arena size +
    // obstacle density together, like a lesson) and a "New layout" button (re-randomizes obstacles).
    // Latches because NavEnvironment gates its param poll on IsCommunicatorOn (off in the WebGL build).
    public class LayoutControlsUI : MonoBehaviour
    {
        [SerializeField] private int startLevel = 1; // arena 8 + 3 obstacles: livelier goal-reaching for the demo
        private NavEnvironment _env;
        private int _level;
        private bool _applied;

        private void Start()
        {
            _env = FindAnyObjectByType<NavEnvironment>();
            _level = Mathf.Clamp(startLevel, 0, DifficultyMapper.NumLevels - 1);
            // Apply is deferred to the first Update (below), NOT here: NavEnvironment.Start sets a default
            // difficulty (arena 11), and Start() order across GameObjects is undefined -- applying here could
            // be overwritten. Deferring one frame guarantees this runs after every Start().
        }

        private void Update()
        {
            if (_applied) return;
            _applied = true;
            Apply();
        }

        private void Apply()
        {
            if (_env == null) return;
            var d = DifficultyMapper.ForLevel(_level);
            _env.SetArenaSize(d.ArenaHalfSize);
            _env.SetObstacleCount(d.MaxObstacles);
            _env.RegenerateLayout();
        }

        private void OnGUI()
        {
            GUI.skin.label.fontSize = 16;
            GUI.Box(new Rect(12f, 100f, 260f, 108f), "");
            GUI.Label(new Rect(24f, 106f, 240f, 24f), "Difficulty: L" + _level +
                      "  (arena " + DifficultyMapper.ForLevel(_level).ArenaHalfSize + ")");
            int v = Mathf.RoundToInt(GUI.HorizontalSlider(
                new Rect(24f, 140f, 200f, 24f), _level, 0f, DifficultyMapper.NumLevels - 1));
            GUI.Label(new Rect(232f, 136f, 30f, 24f), "L" + v);
            if (v != _level) { _level = v; Apply(); }
            if (GUI.Button(new Rect(24f, 172f, 200f, 28f), "New layout")) _env?.RegenerateLayout();
        }
    }
}
