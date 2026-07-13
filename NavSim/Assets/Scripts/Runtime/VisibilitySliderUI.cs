using UnityEngine;

namespace NavSim.Runtime
{
    // Demo control that makes M4 hidden-goal search tangible: a goal-visibility slider driving
    // NavEnvironment.SetRayLength. Visitors watch the SAME policy beeline when the goal is visible and
    // search when it is hidden. Latches because NavEnvironment gates its param poll on IsCommunicatorOn.
    //
    // SOLE OWNER of the demo's ray/visibility axis (final-review decision): NavEnvironment.Start defaults
    // ray to the hidden rung, so this component establishes the visible default. Apply is deferred to the
    // first Update (like LayoutControlsUI): NavEnvironment.Start sets the default and Start() order across
    // GameObjects is undefined, so applying here could be overwritten. LayoutControlsUI (difficulty) and
    // CrowdSliderUI (count) deliberately do NOT touch ray, so the sliders stay orthogonal.
    public class VisibilitySliderUI : MonoBehaviour
    {
        private NavEnvironment _env;
        private float _minRay, _maxRay, _ray;
        private bool _applied;

        private void Start()
        {
            _env = FindAnyObjectByType<NavEnvironment>();
            _maxRay = DifficultyMapper.MaxArenaDiagonal;        // fully visible
            _minRay = 0.2f * DifficultyMapper.MaxArenaDiagonal; // hidden (matches L3)
            _ray = _maxRay;                                     // open friendly: goal visible
        }

        private void Update()
        {
            if (_applied) return;
            _applied = true;
            if (_env != null) _env.SetRayLength(_ray);
        }

        private void OnGUI()
        {
            GUI.skin.label.fontSize = 16;
            GUI.Box(new Rect(12f, 216f, 260f, 78f), "");
            float frac = (_ray - _minRay) / (_maxRay - _minRay);
            GUI.Label(new Rect(24f, 222f, 240f, 24f),
                "Goal visibility: " + Mathf.RoundToInt(frac * 100f) + "%");
            float v = GUI.HorizontalSlider(new Rect(24f, 256f, 200f, 24f), _ray, _minRay, _maxRay);
            if (!Mathf.Approximately(v, _ray))
            {
                _ray = v;
                if (_env != null) _env.SetRayLength(_ray);
            }
        }
    }
}
