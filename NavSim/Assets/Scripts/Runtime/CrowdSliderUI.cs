using UnityEngine;

namespace NavSim.Runtime
{
    // Self-contained on-screen crowd-size control for the WebGL demo. Drives NavEnvironment.SetActiveCount,
    // which latches because the arena's auto-poll is gated on IsCommunicatorOn (off in a standalone build).
    public class CrowdSliderUI : MonoBehaviour
    {
        [SerializeField] private int startCount = 8;
        private NavEnvironment _env;
        private int _count;

        private void Start()
        {
            _env = FindAnyObjectByType<NavEnvironment>();
            _count = Mathf.Clamp(startCount, 2, 8);
            if (_env != null) _env.SetActiveCount(_count);
        }

        private void OnGUI()
        {
            GUI.skin.label.fontSize = 16;
            GUI.Box(new Rect(12f, 12f, 260f, 78f), "");
            GUI.Label(new Rect(24f, 18f, 240f, 24f), "Crowd size: " + _count + " agents");
            int v = Mathf.RoundToInt(GUI.HorizontalSlider(new Rect(24f, 52f, 200f, 24f), _count, 2f, 8f));
            GUI.Label(new Rect(232f, 48f, 30f, 24f), v.ToString());
            if (v != _count)
            {
                _count = v;
                if (_env != null) _env.SetActiveCount(_count);
            }
        }
    }
}
