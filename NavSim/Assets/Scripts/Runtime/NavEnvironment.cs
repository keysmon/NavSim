using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

namespace NavSim.Runtime
{
    // Single-learner 3D terrain-search arena. M6 v2 adds FIXED-TARGET visual object-goal search on top of the M5
    // terrain: instead of one goal there are THREE geometrically identical goals differing only in color; one
    // color is always the TARGET (GoalPalette.TargetColorIndex, fixed - no per-episode announcement), the other
    // two are decoys. Reaching the target goal = success; touching a decoy = failure (soften->harden, DecoyRules).
    // The three goals are a RUNTIME pool built from the single scene `goal` TEMPLATE (mirrors MoverController's
    // pool idiom) — no per-arm scene surgery. Colors are assigned each episode from GoalPalette (target slot
    // decorrelated from position). The collapsed `difficulty` curriculum drives the M5 terrain ladder; the
    // runtime-baked NavMesh is the solvability gate + SPL oracle + mover substrate. Plain PPO: one Agent, finite
    // MaxStep, continuous triad respawn.
    public class NavEnvironment : MonoBehaviour
    {
        [Header("Wiring (assign in Editor)")]
        [SerializeField] private NavAgent agent;
        [Tooltip("The goal TEMPLATE — the runtime triad is this + 2 instantiated siblings.")]
        [SerializeField] private Transform goal;
        [SerializeField] private TerrainGenerator terrain;
        [SerializeField] private MoverController movers;

        [Header("Arena")]
        [SerializeField] private float arenaHalf = DifficultyMapper.M5ArenaHalf;
        [Tooltip("3-D reach radius. Elevated goals need vertical slack (capsule centre sits above the platform).")]
        [SerializeField] private float goalRadius = 1.5f;
        [Tooltip("Radius of the rigid 120-deg goal ring around the cluster centre. The 3 goals then sit " +
                 "goalClusterSpread*sqrt(3) apart (2.6 -> ~4.5u > 2*goalRadius, so decoy detection stays clean) AND " +
                 "tight enough to be CO-VISIBLE in one 90-deg camera frame — the pixel arm can compare colours in a " +
                 "single glance (B1 fix, user 2026-07-16). Removes the ray(180)>camera(90) co-visibility confound.")]
        [SerializeField] private float goalClusterSpread = 2.6f;
        [Tooltip("Min spawn->cluster-centre distance (keeps the cluster off the spawn; with GoalMaxDist it also caps " +
                 "how far/small the co-visible goals render — close enough at L0 that colours are distinguishable).")]
        [SerializeField] private float clusterMinDist = 4.5f;

        [Header("Perception (M6)")]
        [Tooltip("PIXEL arm ONLY: real-frustum shaping gate. Leave null on ray arms.")]
        [SerializeField] private Camera agentCamera;
        [Tooltip("false: all goals share the \"goal\" tag (pixel/ray1). true: per-color \"goal_c{k}\" tags (rayC).")]
        [SerializeField] private bool tagGoalsByColor = false;

        [Header("Line of sight")]
        [Tooltip("Walls/platforms/ramps only (NOT movers) — LoS is occluded by STATIC geometry, not transient movers.")]
        [SerializeField] private LayerMask staticGeometryMask;
        [SerializeField] private float eyeHeight = 1.6f;

        [Header("Pit fall")]
        [SerializeField] private float killPlaneY = -3f;   // fall below this Y == pit fall
        [SerializeField] private float respawnSampleRadius = 6f;

        public float KillPlaneY => killPlaneY;
        public int GoalsReached { get; private set; }
        public int PitFalls { get; private set; }        // eval instrumentation (eval reads deltas)
        public float GoalRadius => goalRadius;            // eval measures reach against a captured target0
        public int CurrentLevel => _appliedLevel < 0 ? 0 : _appliedLevel; // read by NavAgent for DecoyRules
        // The eval harness sets this so a decoy touch does NOT trigger EndEpisode (which would synchronously re-roll
        // the arena mid-episode and corrupt the harness's geometric outcome detection against captured values).
        // The harness OWNS the episode boundary + detects reach-vs-decoy itself, always hard. TRAINING leaves it false.
        public bool EvalMode { get; set; }

        // TRAIN-only decoy hardness (spec §7 soften->harden; user 2026-07-16). SOFT during a step-based WARMUP so the
        // agent learns nav + target-colour discrimination while episodes survive, then HARD (a wrong-colour pick
        // ends the episode) so training rewards TARGET-FIRST — aligning the train objective to the ALWAYS-hard eval
        // (soft-only training rewards cheap goal-sweeping, which the hard eval scores as decoy_visit). Warmup length =
        // `decoy_warmup` env-parameter (default 300k steps); OR the pre-existing level rule (L1+ hard even in warmup).
        // Eval bypasses this via EvalMode (the harness owns the boundary).
        public bool DecoyHard =>
            Academy.Instance.TotalStepCount >= Academy.Instance.EnvironmentParameters.GetWithDefault("decoy_warmup", 300000f)
            || DecoyRules.DecoyEndsEpisode(CurrentLevel);

        // The FIXED target colour (always GoalPalette red). INTERNAL knowledge - never observed by the agent;
        // consumed by the gate/reach logic and the eval pairing fingerprint.
        public Color TargetColorRgb { get; private set; }

        private int _appliedLevel = -1;
        private Transform[] _goals;                       // runtime triad: _goals[0] == `goal` template + 2 siblings
        private RayPerceptionSensorComponent3D[] _agentFans; // RAY arms: cached goal-detecting fans for the sensor-truth gate
        private GoalPalette.GoalAssignment _assign;
        private System.Random _colorRng = new System.Random();

        private int TargetSlot => _assign.ColorIndices == null ? 0 : _assign.TargetSlot;
        private Transform Target => _goals[TargetSlot];

        // Force the per-episode color RNG so a paired eval reproduces identical (colors, target) across arms.
        // Training leaves it free-running.
        public void SeedColorRng(int seed) => _colorRng = new System.Random(seed);

        // Initial bake so a NavMesh exists before the agent's first OnEpisodeBegin (which re-bakes a fresh layout).
        private void Start() => SetTerrainLevel(ReadDifficulty());

        // Only the trainer (communicator on) drives the curriculum. On a lesson advance, end the episode for a
        // clean trajectory boundary (plain PPO); OnEpisodeBegin then bakes the new-level layout. In a
        // no-communicator build (WebGL/standalone) GetWithDefault always returns the default, so this is inert.
        private void FixedUpdate()
        {
            if (!Academy.Instance.IsCommunicatorOn) return;
            int level = ReadDifficulty();
            if (level != _appliedLevel && agent != null) { _appliedLevel = level; agent.EndEpisode(); }
        }

        // TRAINING (communicator on): the trainer's `difficulty` env-param drives the curriculum; the default is
        // the HARDEST rung so a resumed run never regresses. DEMO (no communicator): env-params are inert
        // (GetWithDefault always returns the default), so default to L0 — the easiest, most legible rung — and let
        // the demo UI drive the level. The IsCommunicatorOn branch makes this change byte-identical for training.
        private int ReadDifficulty() => Mathf.Clamp(
            Mathf.RoundToInt(Academy.Instance.EnvironmentParameters.GetWithDefault(
                "difficulty", Academy.Instance.IsCommunicatorOn ? (float)(DifficultyMapper.NumLevels - 1) : 0f)),
            0, DifficultyMapper.NumLevels - 1);

        // Regenerate the layout for the current curriculum level. Called from NavAgent.OnEpisodeBegin, so a
        // fresh solvable arena is drawn every episode -> thousands of layouts, no memorisation.
        // TRAINING (communicator on): the trainer's `difficulty` env-param picks the level (unchanged).
        // DEMO (communicator off): env-params are inert, so HOLD the level the demo UI last set (CurrentLevel) —
        // otherwise every episode reset (MaxStep, hard-decoy touch at L1+) would snap back to ReadDifficulty's
        // default and the L0-L3 selector would not stick.
        public void PlaceForNewEpisode(NavAgent a) => SetTerrainLevel(
            Academy.Instance.IsCommunicatorOn ? ReadDifficulty() : CurrentLevel);

        // Build the runtime triad from the scene `goal` template (once). _goals[0] IS the template; siblings are
        // geometrically identical copies (same mesh/collider/tag) — color is the only per-episode difference.
        private void EnsureTriad()
        {
            if (_goals != null) return;
            _goals = new Transform[3];
            _goals[0] = goal;
            for (int i = 1; i < 3; i++)
            {
                Transform g = Instantiate(goal, goal.parent);
                g.name = goal.name + "_" + i;
                _goals[i] = g;
            }
        }

        // Build a fresh solvable terrain layout: bake structure, then place a walkable spawn + a reachable triad,
        // retrying the bake until spawn+goal connect. Then set the mover count on the live mesh.
        public void SetTerrainLevel(int level)
        {
            if (terrain == null || agent == null || goal == null) return;
            EnsureTriad();
            TerrainLevel lvl = DifficultyMapper.ForTerrainLevel(level);
            for (int t = 0; t < 20; t++)
            {
                terrain.Generate(lvl, Vector3.zero, Vector3.zero); // bake structure; real spawn/goals chosen below
                if (!SampleGround(RandomXZ(), out Vector3 spawn)) continue;
                if (!TryPickGoal(lvl, spawn, clusterMinDist, GoalMaxDist(level), out Vector3 _)) continue; // terrain is goal-placeable
                PlaceAt(agent, spawn);
                PlaceTriad(lvl, spawn, level);
                agent.NotifyGoalMoved();
                if (movers != null) movers.SetCount(lvl.Movers);
                _appliedLevel = level;
                return;
            }
            // Retries exhausted (rare): best-effort spawn + triad so the episode never runs stale.
            if (SampleGround(RandomXZ(), out Vector3 fallback) || SampleGround(Vector3.zero, out fallback))
            {
                PlaceAt(agent, fallback);
                PlaceTriad(lvl, fallback, level);
            }
            agent.NotifyGoalMoved();
            if (movers != null) movers.SetCount(lvl.Movers);
            _appliedLevel = level;
        }

        // Place all 3 goals at distinct, reachable, far-enough-apart positions, then assign colors + the target.
        // Every goal is made reachable (any could become the target), so solvability holds whichever slot is the target.
        // Decoys share the target's placement constraints so geometry never distinguishes them.
        private void PlaceTriad(TerrainLevel lvl, Vector3 from, int level)
        {
            // Place the 3 goals on a RIGID 120-deg ring of radius goalClusterSpread around a cluster centre, with a
            // SINGLE random rotation for the whole triad. Spacing = goalClusterSpread*sqrt(3) ≈ 4.5u (> 2*goalRadius
            // BY CONSTRUCTION — no rejection sampling, no last-resort spam) keeps decoy detection clean; the tight ring
            // makes all 3 goals CO-VISIBLE in one 90-deg camera frame so the pixel arm can compare colours in a single
            // glance. Ring RIGIDITY is invisible to learning (colour is decorrelated from slot via GoalPalette, and the
            // centre + rotation + spawn are already random per episode, so the policy never sees the same layout twice).
            // Arm-symmetric world change (user 2026-07-16); decoys share the target's constraints — geometry never
            // distinguishes them. Snap each goal to ground + require reachable (any could be the target); re-roll
            // the rotation, then a fresh centre, on failure.
            Vector3 centre = PickClusterCentre(lvl, from, level);
            bool placed = false;
            for (int tries = 0; tries < 24 && !placed; tries++)
            {
                if (tries > 0 && tries % 6 == 0) centre = PickClusterCentre(lvl, from, level); // periodically re-roll the centre
                placed = TryRing(from, centre, requireReachable: true);
            }
            if (!placed)
            {
                // Design-unreachable in practice: place the ring ignoring reachability (spacing preserved). Warn.
                TryRing(from, centre, requireReachable: false);
                Debug.LogWarning("[NavEnvironment] PlaceTriad ring last-resort — no reachable rotation found (rare).");
            }
            AssignColorsAndTarget();
        }

        // A close, reachable cluster centre. Elevated levels keep a platform-top (elevation fidelity is a separate
        // §10.8 follow-up — siblings currently snap to the surrounding ground); flat levels use a distance-controlled
        // random-bearing point at a radius in [clusterMinDist, GoalMaxDist] so the cluster is reliably close + co-visible.
        private Vector3 PickClusterCentre(TerrainLevel lvl, Vector3 from, int level)
        {
            if (lvl.GoalElevated)
            {
                if (terrain.RandomPlatformTop(out Vector3 pt)) return pt;
                SampleGround(from, out pt); return pt;
            }
            for (int t = 0; t < 40; t++)
            {
                float ang = Random.value * (2f * Mathf.PI), r = Random.Range(clusterMinDist, GoalMaxDist(level));
                if (SampleGround(from + new Vector3(Mathf.Cos(ang) * r, 0f, Mathf.Sin(ang) * r), out Vector3 cg)
                    && terrain.IsReachable(from, cg)) return cg;
            }
            return from;
        }

        // Place the 3 goals on a 120-deg ring (single random rotation) around the centre, snapped to ground. Returns
        // true iff every goal snapped AND (when required) is reachable from the spawn; on false, _goals is left in a
        // partial state that the caller either retries over or accepts as the (warned) last resort.
        private bool TryRing(Vector3 from, Vector3 centre, bool requireReachable)
        {
            float th0 = Random.value * (2f * Mathf.PI);
            for (int i = 0; i < 3; i++)
            {
                float a = th0 + i * (2f * Mathf.PI / 3f);
                Vector3 p = centre + new Vector3(Mathf.Cos(a), 0f, Mathf.Sin(a)) * goalClusterSpread;
                bool ground = SampleGround(p, out Vector3 g);
                _goals[i].position = ground ? g : p;
                if (requireReachable && (!ground || !terrain.IsReachable(from, g))) return false;
            }
            return true;
        }

        // Draw 2 random distinct decoy colours + shuffle colours across slots (fixed red target - GoalPalette v2),
        // paint the goal renderers, and (rayC only) tag each goal by its colour.
        private void AssignColorsAndTarget()
        {
            _assign = GoalPalette.Assign(_colorRng);
            for (int i = 0; i < 3; i++)
            {
                Renderer rend = _goals[i].GetComponentInChildren<Renderer>();
                if (rend != null) rend.material.color = GoalPalette.Colors[_assign.ColorIndices[i]];
                _goals[i].tag = tagGoalsByColor ? GoalPalette.Tag(_assign.ColorIndices[i]) : "goal";
            }
            TargetColorRgb = GoalPalette.TargetColor(_assign);
        }

        // A reachable goal within [minDist, maxDist] of `from`: platform-top for elevated levels, else walkable
        // ground. The maxDist cap keeps warmup goals inside sight so the shaping reward has a matching observable
        // signal. The cap ramps open with difficulty. False -> caller re-rolls.
        private bool TryPickGoal(TerrainLevel lvl, Vector3 from, float minDist, float maxDist, out Vector3 g)
        {
            g = Vector3.zero;
            if (lvl.GoalElevated) { if (!terrain.RandomPlatformTop(out g)) return false; }
            else if (!SampleGround(RandomXZ(), out g)) return false;
            float d = Vector3.Distance(from, g);
            if (d < minDist || d > maxDist) return false;
            return terrain.IsReachable(from, g);
        }

        // Max spawn->goal distance per curriculum level: tight at the warmup rungs, opening to full-arena at L3.
        private static float GoalMaxDist(int level)
        {
            switch (level)
            {
                case 0: return 7f;    // warmup: close cluster so co-visible goals render big enough to tell apart
                case 1: return 12f;
                case 2: return 16f;   // elevated; generous so platform-top placement isn't over-constrained
                default: return 999f; // L3+: unbounded full-arena search
            }
        }

        // The TARGET goal's position — NavAgent's shaping/reach all point here.
        public Vector3 GoalPositionFor(NavAgent a) => Target.position;

        // 3-D reach of the TARGET: within goalRadius in ALL axes (an elevated goal requires being ON the platform).
        public bool ReachedGoal(NavAgent a) => Vector3.Distance(a.transform.position, Target.position) < goalRadius;

        // Touched a WRONG-color goal this step (episode-ending under the hard schedule; penalty under soft).
        public bool TouchedDecoy(NavAgent a)
        {
            for (int i = 0; i < 3; i++)
                if (i != TargetSlot && Vector3.Distance(a.transform.position, _goals[i].position) < goalRadius)
                    return true;
            return false;
        }

        // The two non-target goal positions as VALUES (the eval captures these once per episode for paired,
        // transform-independent outcome detection).
        public Vector3[] DecoyPositions()
        {
            var d = new List<Vector3>(2);
            for (int i = 0; i < 3; i++) if (i != TargetSlot) d.Add(_goals[i].position);
            return d.ToArray();
        }

        // Shaping gate: the TARGET is genuinely in the agent's ACTUAL sensor field this step. INVARIANT
        // (advisor): the gate must be a SUBSET of the arm's true FOV — err NARROW (a too-wide gate rewards reducing
        // distance to a target the sensor can't see -> the M5 freeze trap). PIXEL arm (agentCamera set) tests the
        // REAL camera frustum (manual H/V FOV; handles tilt + elevation); RAY arms use the SENSOR-TRUTH gate — an
        // actual goal-detecting ray hit (an angular proxy is distance-fragile for the fans; Stage-A proved it leaks).
        public bool TargetPerceivable(NavAgent a)
        {
            Vector3 tp = Target.position;
            Vector3 eye = a.transform.position + Vector3.up * eyeHeight;
            if (Vector3.Distance(eye, tp) > DifficultyMapper.SightRange) return false; // range prune (<= fan ray reach; the sensor perceive below is authoritative)
            if (agentCamera != null)
            {
                // PIXEL arm: static occlusion + the REAL square-sensor frustum via a manual camera-local H/V FOV
                // check — NOT WorldToViewportPoint. Its projection uses Camera.aspect, which OUTSIDE the sensor's
                // render call is the screen/game-view aspect (Stage-A A1 measured 1.333, a 4:3 default), NOT the
                // 84x84 sensor's 1:1 — that made the gate ~106deg wide horizontally vs the CNN's real 90deg (a
                // horizontal freeze trap). Camera.fieldOfView is the VERTICAL FOV; the SQUARE sensor makes
                // horizontal FOV == vertical.
                if (Physics.Linecast(eye, tp, staticGeometryMask)) return false;    // occluded by static geometry
                Vector3 local = agentCamera.transform.InverseTransformPoint(tp);
                if (local.z <= 0f) return false;                                   // behind the camera
                float half = agentCamera.fieldOfView * 0.5f;                       // square sensor: horizontal half == vertical half
                float vAng = Mathf.Atan2(Mathf.Abs(local.y), local.z) * Mathf.Rad2Deg;
                float hAng = Mathf.Atan2(Mathf.Abs(local.x), local.z) * Mathf.Rad2Deg;
                return vAng <= half && hAng <= half;
            }
            // RAY arms: SENSOR-TRUTH gate — shaping fires iff a goal-detecting ray ACTUALLY registers the target
            // this step. Any angular proxy is provably distance-fragile here: the fans' reach envelope is
            // NOT a clean cone (a fixed-size goal + spherecast subtend more angle up close, less far away; and the
            // gate eye != the 0.9 fan origin), so a fixed +-cone over-fired for elevated goals at long range
            // (Stage-A grid: 18 freeze leaks at D=8-14, h~3m) AND starved close flat goals. Perceiving the real
            // fans removes every approximation — the gate becomes EXACTLY what the sensor observes: range,
            // occlusion, and the discrete-ray/spherecast envelope are all handled by the rays themselves.
            if (_agentFans == null)
            {
                _agentFans = a.GetComponents<RayPerceptionSensorComponent3D>();
                if (_agentFans.Length == 0)
                    Debug.LogWarning("[NavEnvironment] TargetPerceivable: a ray arm (agentCamera==null) has NO " +
                                     "RayPerceptionSensorComponent3D — the shaping gate will never fire (silent " +
                                     "starvation). Check the scene's sensor setup.");
            }
            var targetGo = Target.gameObject;
            foreach (var fan in _agentFans)
            {
                var input = fan.GetRayPerceptionInput();
                // Perceive on the SAME cast path the observation uses (mirror the fan's batched flag) so gate ==
                // what the sensor encodes BY CONSTRUCTION, not merely when the package's unbatched default holds.
                var outs = RayPerceptionSensor.Perceive(input, input.UseBatchedRaycasts).RayOutputs;
                for (int i = 0; i < outs.Length; i++)
                    // HitTagIndex >= 0 == a DETECTABLE-tag hit (so it's in the observation) — excludes an untagged
                    // physical hit by the terrain-only down-fan on the goal collider.
                    if (outs[i].HasHit && outs[i].HitTagIndex >= 0 && outs[i].HitGameObject == targetGo) return true;
            }
            return false;
        }

        // Continuous respawn on reaching the target: a fresh triad on the SAME baked terrain (no EndEpisode).
        public void RespawnGoal(NavAgent a)
        {
            int level = _appliedLevel < 0 ? 0 : _appliedLevel;
            TerrainLevel lvl = DifficultyMapper.ForTerrainLevel(level);
            PlaceTriad(lvl, a.transform.position, level);
            a.NotifyGoalMoved();
        }

        public void NotifyGoalReached() => GoalsReached++;

        // LoS gate to the target: within fixed sight AND no STATIC geometry between the agent's eye and it.
        // (Kept for API compatibility; NavAgent's shaping uses TargetPerceivable.)
        public bool GoalVisibleTo(NavAgent a)
        {
            Vector3 eye = a.transform.position + Vector3.up * eyeHeight;
            float dist = Vector3.Distance(eye, Target.position);
            bool blocked = Physics.Linecast(eye, Target.position, staticGeometryMask);
            return VisibilityGate.IsGoalVisible(dist, DifficultyMapper.SightRange, lineOfSightClear: !blocked);
        }

        public IReadOnlyList<Vector3> MoverPositions() =>
            movers != null ? movers.Positions() : System.Array.Empty<Vector3>();

        // Pit-fall recovery: sample the nearest walkable ground to where the agent fell and teleport it there,
        // keeping episode + LSTM memory. Never a shortcut toward the goal — re-homes near the fall site.
        public void RespawnToSafeGround(NavAgent a)
        {
            PitFalls++;
            Vector3 p = a.transform.position; p.y = 1f;
            if (NavMesh.SamplePosition(p, out NavMeshHit hit, respawnSampleRadius, NavMesh.AllAreas))
                PlaceAt(a, hit.position + Vector3.up * 0.1f);
            else if (SampleGround(Vector3.zero, out Vector3 c)) // last resort (design-unreachable): arena centre
                PlaceAt(a, c);
            a.NotifyGoalMoved();
        }

        // Teleport a CharacterController agent (disable the controller around a direct transform write; it
        // caches its own position and would otherwise fight the move).
        private void PlaceAt(NavAgent a, Vector3 pos)
        {
            CharacterController cc = a.GetComponent<CharacterController>();
            if (cc != null) cc.enabled = false;
            a.transform.position = pos;
            a.transform.rotation = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);
            if (cc != null) cc.enabled = true;
        }

        // Snap a random XZ point onto the baked walkable NavMesh (avoids pits + OOB by construction).
        private bool SampleGround(Vector3 xz, out Vector3 ground)
        {
            if (NavMesh.SamplePosition(new Vector3(xz.x, 1f, xz.z), out NavMeshHit hit, 4f, NavMesh.AllAreas))
            { ground = hit.position + Vector3.up * 0.1f; return true; }
            ground = xz; return false;
        }

        private Vector3 RandomXZ() =>
            new Vector3(Random.Range(-arenaHalf, arenaHalf), 0f, Random.Range(-arenaHalf, arenaHalf));
    }
}
