using UnityEngine;

namespace NavSim.Runtime
{
    // Presentation overlay for the showcase demo (WebGL / standalone — NO trainer). Mirrors M6DemoUI's IMGUI idiom
    // (stable rects the WebGL smoke test can click by canvas coordinate; the env HOLDS the level in a
    // communicator-off build). Adds the showcase's narration layer:
    //   - "New layout" re-roll (re-seeds Random, re-applies the CURRENT stage -> new mirror/jitter/decoy colours),
    //   - a 5-stage selector labelled by CourseSpec.StageNames,
    //   - a 0.5x/1x/2x speed control,
    //   - a lower-third caption that reads the agent's ACTUAL behaviour each frame (a priority state machine), and
    //   - a picture-in-picture of the agent's OWN eye camera at the true 84x84 CNN input resolution.
    //
    // HONESTY (binding): the PiP renders the SAME eyeCam that feeds the CameraSensor (agent-layer culled), at the
    // real 84x84, and its border turns red on exactly env.TargetPerceivable(agent) — the same gate the training
    // reward is shaped on. The overlay narrates the policy; it never substitutes a scripted signal for a real one.
    public class ShowcaseDemoUI : MonoBehaviour
    {
        [SerializeField] private NavEnvironment env;
        [SerializeField] private NavAgent agent;
        [Tooltip("The agent's egocentric camera that feeds the CNN sensor. The PiP renders THIS camera (never a new one).")]
        [SerializeField] private Camera eyeCam;

        // Caption strings — single source so the priority order below reads as a table.
        private const string CapFound = "Target found!";
        private const string CapDecoy = "Decoy - wrong sphere";
        private const string CapJump = "Jumping the gap";
        private const string CapSighted = "Target sighted - approaching";
        private const string CapRamp = "Climbing the ramp";
        private const string CapScan = "Scanning for the red target";

        private const float GoalFlashHold = 1.5f;   // realtime seconds the "Target found!" flash holds
        private const float DecoyFlashHold = 1.0f;   // realtime seconds the "Decoy" flash holds
        private const float DecoyNearDist = 1.6f;   // agent-to-decoy distance that trips the decoy caption

        private CharacterController _cc;             // agent's controller (isGrounded for the jump/ramp captions)
        private int _lastGoalsReached;              // edge-trigger source for the "Target found!" flash
        private bool _wasNearDecoy;                 // edge-trigger latch for the decoy flash (fire on ENTRY only)
        private float _goalFlashUntil;              // realtime timestamps; caption is active while now < until
        private float _decoyFlashUntil;
        private bool _sighted;                      // cached env.TargetPerceivable(agent) — used by caption AND PiP border
        private string _caption = CapScan;

        private RenderTexture _rt;                   // 84x84 PiP target (the true CNN input size), created lazily
        private Texture2D _ring;                     // reticle drawn over the sighted target in the PiP (lazy)

        private GUIStyle _titleStyle, _stageNameStyle, _speedLabelStyle, _captionStyle, _pipLabelStyle;

        // Belt-and-suspenders with NavEnvironment's communicator-off default of stage 0 — make the opening stage
        // explicit at this layer too (independent of Start()-ordering between the env and this component). Also caches
        // the agent's CharacterController for the caption's grounded checks.
        private void Start()
        {
            if (agent != null) _cc = agent.GetComponent<CharacterController>();
            if (env != null) env.SetTerrainLevel(0);
        }

        // Evaluate ALL state ONCE per frame here (edge-triggers, the perceivable gate, the caption). OnGUI fires
        // multiple times per frame (Layout/Repaint/input events); doing edge detection there would double-count.
        private void Update()
        {
            if (env == null || agent == null) return;
            float now = Time.realtimeSinceStartup;

            // (a) goal-reached flash — env.GoalsReached is monotonic; any increase since last frame trips it.
            int gr = env.GoalsReached;
            if (gr > _lastGoalsReached) _goalFlashUntil = now + GoalFlashHold;
            _lastGoalsReached = gr;

            // (b) decoy flash — edge-triggered on ENTERING DecoyNearDist of ANY non-target slot (env.DecoyPositions()
            // excludes the red target), so camping a decoy does not re-fire the caption.
            bool nearDecoy = false;
            Vector3 p = agent.transform.position;
            foreach (Vector3 d in env.DecoyPositions())
                if (Vector3.Distance(p, d) < DecoyNearDist) { nearDecoy = true; break; }
            if (nearDecoy && !_wasNearDecoy) _decoyFlashUntil = now + DecoyFlashHold;
            _wasNearDecoy = nearDecoy;

            // (d) the honest sighted signal, evaluated once (raycasts) and reused by the caption AND the PiP border.
            _sighted = env.TargetPerceivable(agent);

            _caption = ComputeCaption(now, p);
        }

        // Priority state machine (highest first): flashes > airborne > sighted > on-ramp > scanning.
        private string ComputeCaption(float now, Vector3 agentPos)
        {
            if (now < _goalFlashUntil) return CapFound;
            if (now < _decoyFlashUntil) return CapDecoy;
            bool grounded = _cc == null || _cc.isGrounded;
            if (!grounded) return CapJump;
            if (_sighted) return CapSighted;
            // On the ramp: grounded AND agent z within the stage's ramp band (0-width band == stage has no ramp).
            CourseBuilder course = env.Course;
            if (course != null)
            {
                CourseLayout lay = course.CurrentLayout;
                if (lay.RampZMax > lay.RampZMin && agentPos.z >= lay.RampZMin && agentPos.z <= lay.RampZMax)
                    return CapRamp;
            }
            return CapScan;
        }

        private void OnGUI()
        {
            EnsureStyles();
            DrawControlPanel();
            DrawCaption();
            DrawPiP();
        }

        // --- control panel (stable rects; "New layout" pinned at (10,10,140,30) for the WebGL smoke test) ---
        private void DrawControlPanel()
        {
            GUI.Box(new Rect(6f, 6f, 330f, 112f), GUIContent.none);

            if (GUI.Button(new Rect(10f, 10f, 140f, 30f), "New layout"))
            {
                Random.InitState(System.Environment.TickCount);
                if (env != null) env.SetTerrainLevel(env.CurrentLevel);
            }
            GUI.Label(new Rect(158f, 14f, 168f, 24f), "Find the <color=red>RED</color> sphere", _titleStyle);

            // Stage selector: 5 buttons "1".."5" -> stages 0..4; the active stage is highlighted and named.
            int stage = env != null ? env.CurrentLevel : 0;
            for (int i = 0; i < CourseSpec.NumStages; i++)
            {
                GUI.color = (i == stage) ? Color.cyan : Color.white;
                if (GUI.Button(new Rect(10f + i * 35f, 48f, 33f, 28f), (i + 1).ToString()) && env != null)
                    env.SetTerrainLevel(i);
            }
            GUI.color = Color.white;
            string stageName = (stage >= 0 && stage < CourseSpec.StageNames.Length) ? CourseSpec.StageNames[stage] : "";
            GUI.Label(new Rect(190f, 52f, 140f, 22f), stageName, _stageNameStyle);

            // Speed control: 0.5x / 1x / 2x -> Time.timeScale; the active rate is highlighted.
            DrawSpeedButton(new Rect(10f, 84f, 48f, 26f), "0.5x", 0.5f);
            DrawSpeedButton(new Rect(62f, 84f, 48f, 26f), "1x", 1f);
            DrawSpeedButton(new Rect(114f, 84f, 48f, 26f), "2x", 2f);
            GUI.color = Color.white;
            GUI.Label(new Rect(170f, 88f, 90f, 22f), "Speed", _speedLabelStyle);
        }

        private void DrawSpeedButton(Rect r, string label, float rate)
        {
            GUI.color = Mathf.Approximately(Time.timeScale, rate) ? Color.cyan : Color.white;
            if (GUI.Button(r, label)) Time.timeScale = rate;
        }

        // --- lower-third caption pill (sized to the text, translucent backing for legibility) ---
        private void DrawCaption()
        {
            var content = new GUIContent(_caption);
            Vector2 sz = _captionStyle.CalcSize(content);
            const float padX = 24f, padY = 10f;
            var pill = new Rect((Screen.width - sz.x) * 0.5f - padX, Screen.height * 0.76f,
                sz.x + 2f * padX, sz.y + 2f * padY);
            Fill(pill, new Color(0f, 0f, 0f, 0.5f));
            GUI.Label(pill, content, _captionStyle);
        }

        // --- picture-in-picture of the agent's OWN eye camera at the true 84x84 CNN input resolution ---
        private void DrawPiP()
        {
            if (eyeCam == null) return;
            EnsureRenderTexture();

            const float size = 200f, margin = 14f;
            var pipRect = new Rect(Screen.width - size - margin, Screen.height - size - margin, size, size);

            // Render the eyeCam into the 84x84 RT on Repaint only (once per frame; Camera.Render is expensive and
            // works on a disabled camera). Save/restore the previous targetTexture so we never clobber the sensor's.
            if (Event.current.type == EventType.Repaint)
            {
                RenderTexture prev = eyeCam.targetTexture;
                eyeCam.targetTexture = _rt;
                eyeCam.Render();
                eyeCam.targetTexture = prev;
            }

            // Border = the honest sighted signal: red + thicker on exactly env.TargetPerceivable (the SAME gate the
            // reward is shaped on), a neutral thin frame otherwise. Drawn as a filled box 2px (4px sighted) larger.
            float b = _sighted ? 4f : 2f;
            Fill(new Rect(pipRect.x - b, pipRect.y - b, pipRect.width + 2f * b, pipRect.height + 2f * b),
                _sighted ? Color.red : new Color(0f, 0f, 0f, 0.6f));
            GUI.DrawTexture(pipRect, _rt, ScaleMode.StretchToFill);

            // In-PiP reticle over the sighted target (stretch). Pin a square (1:1) aspect for the projection so it
            // matches the 84x84 sensor's framing — outside its own render call the camera's aspect is the screen's,
            // which would mislocate the marker (the same trap NavEnvironment avoids). CRITICAL: writing Camera.aspect
            // switches the camera OFF auto-aspect until ResetAspect — the CameraSensor renders to an 84x84 RT and
            // relies on the AUTO 1:1 aspect it derives from that RT (it never sets aspect itself). So we must
            // ResetAspect() to return to auto, NOT restore a captured value: a pinned screen aspect would persist and
            // distort every later render — this PiP AND the live CNN input.
            if (_sighted && agent != null && env != null)
            {
                eyeCam.aspect = 1f;
                Vector3 vp = eyeCam.WorldToViewportPoint(env.GoalPositionFor(agent));
                eyeCam.ResetAspect();
                if (vp.z > 0f)
                {
                    EnsureRing();
                    const float rs = 26f;
                    float mx = pipRect.x + vp.x * pipRect.width;
                    float my = pipRect.y + (1f - vp.y) * pipRect.height;
                    GUI.DrawTexture(new Rect(mx - rs * 0.5f, my - rs * 0.5f, rs, rs), _ring, ScaleMode.StretchToFill);
                }
            }

            // Label strip above the PiP (dark backing so it reads over any terrain).
            var labelRect = new Rect(pipRect.x, pipRect.y - 22f, pipRect.width, 20f);
            Fill(labelRect, new Color(0f, 0f, 0f, 0.55f));
            GUI.Label(labelRect, "Agent camera - CNN input, 84x84 RGB", _pipLabelStyle);
        }

        // Tint the built-in white texture (no per-instance texture to own/destroy).
        private static void Fill(Rect r, Color c)
        {
            Color prev = GUI.color;
            GUI.color = c;
            GUI.DrawTexture(r, Texture2D.whiteTexture, ScaleMode.StretchToFill);
            GUI.color = prev;
        }

        private void EnsureRenderTexture()
        {
            if (_rt != null) return;
            _rt = new RenderTexture(84, 84, 16) { name = "ShowcasePiP" };
            _rt.Create();
        }

        // A small hollow reticle ring generated once (no external asset; keeps the demo self-contained).
        private void EnsureRing()
        {
            if (_ring != null) return;
            const int n = 32;
            _ring = new Texture2D(n, n, TextureFormat.RGBA32, false) { name = "ShowcaseReticle" };
            var reticle = new Color(1f, 0.92f, 0.2f, 1f); // bright yellow — contrasts the red target + red border
            float c = (n - 1) * 0.5f, rOuter = 15f, rInner = 11f;
            for (int y = 0; y < n; y++)
                for (int x = 0; x < n; x++)
                {
                    float dist = Mathf.Sqrt((x - c) * (x - c) + (y - c) * (y - c));
                    bool onRing = dist <= rOuter && dist >= rInner;
                    _ring.SetPixel(x, y, onRing ? reticle : Color.clear);
                }
            _ring.Apply();
        }

        private void EnsureStyles()
        {
            if (_captionStyle != null) return;
            _titleStyle = new GUIStyle(GUI.skin.label) { richText = true, fontStyle = FontStyle.Bold };
            _stageNameStyle = new GUIStyle(GUI.skin.label) { fontStyle = FontStyle.Bold };
            _speedLabelStyle = new GUIStyle(GUI.skin.label);
            _captionStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 28,
                fontStyle = FontStyle.Bold,
                alignment = TextAnchor.MiddleCenter,
            };
            _captionStyle.normal.textColor = Color.white;
            _pipLabelStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 11,
                alignment = TextAnchor.MiddleCenter,
            };
            _pipLabelStyle.normal.textColor = Color.white;
        }

        private void OnDestroy()
        {
            if (_rt != null) { _rt.Release(); Destroy(_rt); _rt = null; }
            if (_ring != null) { Destroy(_ring); _ring = null; }
        }
    }
}
