using UnityEngine;
using Unity.MLAgents;

namespace NavSim.Runtime
{
    // M8 push-the-ramp arena controller (a FORK of CoopArena): the SINGLE authority for layout, ramp
    // heaviness, outcome detection, reward application (via ArmRouting ONLY), and the episode boundary.
    // Flat walled arena (half-size 11, M5 footprint). A dynamic ramp must be pushed to its marked target;
    // once placed it forms the only path an agent can take to reach the goal on the ledge.
    //
    // THE TICK SEAM (load-bearing for eval): MonoBehaviour FixedUpdate does NOT fire under the eval
    // harness's manual script stepping (Physics.simulationMode=Script + Academy.EnvironmentStep() - the
    // reason M6's movers froze during eval). ALL per-step logic lives in Tick(fixedDt); FixedUpdate is
    // exactly `if (!EvalMode) Tick(...)`; the eval harness calls arena.Tick explicitly after each step
    // pair. Inside Tick: ramp-read / RampAtTarget / joint-push / success-latch logic ALWAYS runs; reward
    // application AND boundary calls (End/Interrupted + ResetEpisode) are gated on !EvalMode (the harness
    // owns the boundary and reads the latches).
    //
    // The ramp is a DYNAMIC Rigidbody (PushableRampBody): PhysX integrates it under Physics.Simulate
    // (eval) / FixedUpdate (training) BEFORE Tick; Tick only reads its post-step position. Heaviness is
    // the ramp's MASS (S2 competence-ramps it light->heavy); the arena never moves the ramp by fiat.
    public class RampArena : MonoBehaviour
    {
        [Header("Wiring (assign in Editor)")]
        [SerializeField] private RampAgent[] agents;       // length 1 (solo confound detector) or 2, shared policy
        [SerializeField] private Transform goal;
        [SerializeField] private PushableRampBody ramp;
        [SerializeField] private Transform rampTarget;     // the marked spot the ramp must reach
        [SerializeField] private Transform rampStart;      // where the ramp resets to each episode

        [Header("Geometry (Global-Constraints pinned values)")]
        [SerializeField] private float arenaHalf = 11f;    // M5 footprint
        [SerializeField] private float goalRadius = 1.5f;
        [SerializeField] private float targetRadius = 1.5f; // ramp counts "at target" within this of rampTarget
        [SerializeField] private int maxEpisodeSteps = 3000;

        [Header("Heaviness curriculum")]
        // heaviness curriculum knobs (S2 ramps the ramp MASS light->heavy; S0/S1 stay light). A heavier ramp
        // still creeps under one agent (continuous gradient) but is too slow to place+climb solo in the budget.
        [SerializeField] private float rampMassLight = 2f;    // one agent moves it usefully fast (bootstrap)
        [SerializeField] private float rampMassHeavy = 6f;    // one agent only creeps; two are needed in-budget
        [SerializeField] private int   s2RampSuccesses = 200; // competence horizon for the heaviness ramp

        [Header("Shaping")]
        // ramp-to-target shaping (the shapeable signal M7 lacked; orthogonal to cooperation)
        [SerializeField] private float shapingScale = 0.01f;

        // ---- Eval surface (Phase 6 consumes these names verbatim) ----
        public bool EvalMode { get; set; }
        public ArmRouting.Arm ArmMode
        {
            get => _armMode;
            set { _armMode = value; SyncGroupRegistration(); }
        }
        public Vector3 GoalPosition => goal.position;
        public PushableRampBody Ramp => ramp;
        public bool RampAtTarget { get; private set; }
        public float JointPushFrac { get; private set; }
        public int StepsThisEpisode { get; private set; }
        public RampAgent[] Agents => agents;
        public int LastScorerIndex { get; private set; } = -1;
        public bool Success { get; private set; }              // latched until next ResetEpisode

        private ArmRouting.Arm _armMode = ArmRouting.Arm.Poca; // default poca for in-Editor play
        private SimpleMultiAgentGroup _group;                  // created ONCE; registered ONLY under Poca
        private bool _groupRegistered;
        private System.Random _layoutRng = new System.Random(); // ALL layout randomness (never UnityEngine.Random
                                                                // here - the M6 Task-9 global-Random pairing lesson)
        private int _lesson;                                   // 0-3 (S0-S3)
        private int _s0Successes, _s1Successes, _s2Successes;  // monotone competence counters (never reset)
        private float _prevRampToTarget;                       // for potential-based shaping
        private int _jointPushSteps, _rampMoveSteps;           // JointPushFrac numerator/denominator

        // Per-episode layout isolation for the paired eval; training leaves the RNG free-running.
        public void SeedLayoutRng(int seed) => _layoutRng = new System.Random(seed);

        // Curriculum stage select; applied by the NEXT ResetEpisode (the eval calls SeedLayoutRng +
        // SetLesson + ResetEpisode in that order).
        public void SetLesson(int lesson) => _lesson = Mathf.Clamp(lesson, 0, 3);

        private void Start()
        {
            SyncFromEnvParams();
            ResetEpisode();
        }

        // The Tick seam: never put per-step logic here (see class comment).
        private void FixedUpdate()
        {
            if (!EvalMode) Tick(Time.fixedDeltaTime);
        }

        public void Tick(float fixedDt)
        {
            if (agents == null || agents.Length < 1 || ramp == null || goal == null || rampTarget == null) return;

            // Training-only lesson/arm sync (M7 pattern). A lesson change is an environment-imposed boundary
            // -> Interrupted (not the agents' fault), clean trajectory cut (the M6 lesson-advance idiom).
            if (!EvalMode && Academy.Instance.IsCommunicatorOn)
            {
                int lesson = ReadLessonParam();     // env-param "ramp_difficulty"
                if (lesson != _lesson) { SetLesson(lesson); InterruptEpisode(); ResetEpisode(); return; }
            }

            // The ramp is a dynamic Rigidbody - PhysX already integrated it in Physics.Simulate (eval) /
            // FixedUpdate (training) BEFORE this Tick; we just read its post-step position. (No StepRamp call.)
            float rampToTarget = Vector3.Distance(ramp.Position, rampTarget.position);
            bool wasAtTarget = RampAtTarget;
            RampAtTarget = rampToTarget < targetRadius;

            // Joint-push tracker: a JOINT step = the ramp moved this step AND >=2 distinct agents pushed it.
            // PushersThisStep is populated by ApplyPush during the agents' OnControllerColliderHit; clear it after.
            bool moved = rampToTarget < _prevRampToTarget - 1e-4f;
            if (moved) { _rampMoveSteps++; if (ramp.PushersThisStep >= 2) _jointPushSteps++; }
            ramp.ClearPushers();

            // Potential-based ramp-to-target shaping (orthogonal to cooperation; only while not yet placed).
            if (!EvalMode && !wasAtTarget)
            {
                float delta = _prevRampToTarget - rampToTarget;   // positive when ramp nears target
                if (delta != 0f) ApplySplit(ArmRouting.PerStep(_armMode, delta * shapingScale), 0);
            }
            _prevRampToTarget = rampToTarget;

            // Per-step time cost (M7 routing).
            if (!EvalMode) ApplySplit(ArmRouting.PerStep(_armMode, -1f / maxEpisodeSteps), 0);

            // Success = an agent reaches the goal (only reachable once the ramp is placed - enforced by geometry).
            int scorer = -1;
            for (int i = 0; i < agents.Length && scorer < 0; i++)
                if (Vector3.Distance(agents[i].transform.position, goal.position) < goalRadius) scorer = i;

            if (scorer >= 0 && !Success)
            {
                Success = true;
                LastScorerIndex = scorer;
                JointPushFrac = _rampMoveSteps > 0 ? (float)_jointPushSteps / _rampMoveSteps : 0f;
                if (_lesson == 0) _s0Successes++;
                else if (_lesson == 1) _s1Successes++;
                else if (_lesson == 2) _s2Successes++;
                if (!EvalMode)
                {
                    ApplySplit(ArmRouting.Outcome(_armMode), scorer);
                    EndEpisodePerArm();
                    ResetEpisode();
                    return;
                }
            }

            StepsThisEpisode++;
            if (!EvalMode && StepsThisEpisode >= maxEpisodeSteps) { InterruptEpisode(); ResetEpisode(); }
        }

        // Seeded fresh layout + latch reset. Heaviness by lesson via the ramp MASS; ramp reset to a start
        // position (competence-ramped distance for S0/S3); agents spawned mirrored in the near chamber
        // (sides randomly swapped) with per-agent +-0.5u jitter, guarded for 1 or 2 agents.
        public void ResetEpisode()
        {
            StepsThisEpisode = 0;
            Success = false;
            LastScorerIndex = -1;
            RampAtTarget = false;
            _jointPushSteps = 0; _rampMoveSteps = 0; JointPushFrac = 0f;

            // Heaviness by lesson via the ramp MASS. S0/S1 light (one agent moves it fast -> pushing bootstraps).
            // S2 competence-ramped to heavy (a lone agent only creeps -> too slow in-budget -> two needed). S3 =
            // heavy + far start (stretch). EvalMode always applies the hard value.
            float mass = _lesson switch
            {
                0 => rampMassLight,
                1 => rampMassLight,
                2 => Competence.RampValue(_s2Successes, s2RampSuccesses, rampMassLight, rampMassHeavy, EvalMode),
                _ => rampMassHeavy
            };
            ramp.Mass = mass;

            // Ramp start distance (Near->Far lerp): S0 and S3 competence-ramp near(naive)->far(competent) via
            // startRamp so a naive agent gets the SHORT push (bootstrap); S1/S2 stay near. EvalMode = far (hard end).
            float startRamp = _lesson >= 3
                ? (EvalMode ? 1f : Competence.Ramp01(_s2Successes, s2RampSuccesses))
                : (_lesson == 0 ? (EvalMode ? 1f : Competence.Ramp01(_s0Successes, Mathf.Max(1, s2RampSuccesses))) : 1f);
            Vector3 startPos = Vector3.Lerp(NearRampStart(), FarRampStart(), (_lesson == 0 || _lesson >= 3) ? startRamp : 0f);
            ramp.ResetTo(startPos, Quaternion.identity);
            _prevRampToTarget = Vector3.Distance(startPos, rampTarget.position);

            // Spawns (M7 block, guarded for 1 or 2 agents).
            float spawnZ = Lerp(-9f, -6f, NextFloat());
            int flip = NextFloat() < 0.5f ? -1 : 1;
            for (int i = 0; i < agents.Length; i++)
            {
                float side = (agents.Length == 1 ? 0f : (i == 0 ? -1f : 1f) * flip);
                Vector3 p = new Vector3(side * 2.5f + Jitter(), 0.5f, spawnZ + Jitter());
                agents[i].TeleportTo(p, (float)(_layoutRng.NextDouble() * 360.0));
            }

            Physics.SyncTransforms();
        }

        // ---- internals ----

        // Ramp start endpoints (fixed arena positions; rampStart wires the near reset when present in-scene).
        // Near ~ just short of the ramp target (rampTarget ~z=4 at the ledge base): a short push for S0.
        // Far  ~ a corner within the +-arenaHalf arena: the long push for the S3 stretch.
        private Vector3 NearRampStart() => rampStart != null ? rampStart.position : new Vector3(0f, 0.5f, 1.5f);
        private Vector3 FarRampStart() => new Vector3(-(arenaHalf - 3f), 0.5f, -(arenaHalf - 3f));

        // The ONLY reward surface: an ArmRouting split. scorer/partner -> per-agent AddReward; group ->
        // the multi-agent group (Poca only; PPO splits carry group=0 so PPO arms never touch the group).
        private void ApplySplit(ArmRouting.Split s, int scorerIdx)
        {
            if (s.scorer != 0f) agents[scorerIdx].AddReward(s.scorer);
            if (s.partner != 0f && agents.Length > 1) agents[1 - scorerIdx].AddReward(s.partner); // no partner in solo (confound detector)
            if (s.group != 0f && _group != null) _group.AddGroupReward(s.group);
        }

        private void EndEpisodePerArm()
        {
            if (_armMode == ArmRouting.Arm.Poca && _group != null) _group.EndGroupEpisode();
            else for (int i = 0; i < agents.Length; i++) agents[i].EndEpisode(); // 1 (solo) or 2 agents
        }

        private void InterruptEpisode()
        {
            if (_armMode == ArmRouting.Arm.Poca && _group != null) _group.GroupEpisodeInterrupted();
            else for (int i = 0; i < agents.Length; i++) agents[i].EpisodeInterrupted(); // 1 (solo) or 2 agents
        }

        // Group created ONCE; agents registered ONLY while ArmMode == Poca (re-registration guard: an
        // arm switch to a PPO arm unregisters, so PPO trajectories never carry group membership).
        private void SyncGroupRegistration()
        {
            if (agents == null || agents.Length != 2) return;
            if (_armMode == ArmRouting.Arm.Poca)
            {
                if (_groupRegistered) return;
                if (_group == null) _group = new SimpleMultiAgentGroup();
                _group.RegisterAgent(agents[0]);
                _group.RegisterAgent(agents[1]);
                _groupRegistered = true;
            }
            else if (_groupRegistered)
            {
                _group.UnregisterAgent(agents[0]);
                _group.UnregisterAgent(agents[1]);
                _groupRegistered = false;
            }
        }

        // TRAINING (communicator on): the trainer's ramp_difficulty drives the lesson; default = the
        // HARDEST rung so a resumed run never regresses. DEMO/eval (no communicator): default S0 (most
        // legible) - the demo UI / eval drive the lesson directly (the M6 ReadDifficulty idiom).
        private int ReadLessonParam() => Mathf.Clamp(
            Mathf.RoundToInt(Academy.Instance.EnvironmentParameters.GetWithDefault(
                "ramp_difficulty", Academy.Instance.IsCommunicatorOn ? 3f : 0f)), 0, 3);

        private static ArmRouting.Arm DecodeArm(float v) =>
            (ArmRouting.Arm)Mathf.Clamp(Mathf.RoundToInt(v), 0, 2);

        private void SyncFromEnvParams()
        {
            if (!Academy.Instance.IsCommunicatorOn) { SyncGroupRegistration(); return; } // default Poca in-Editor
            SetLesson(ReadLessonParam());
            ArmMode = DecodeArm(Academy.Instance.EnvironmentParameters.GetWithDefault("arm_mode", 2f));
        }

        private float NextFloat() => (float)_layoutRng.NextDouble();
        private float Jitter() => Lerp(-0.5f, 0.5f, NextFloat());
        private static float Lerp(float a, float b, float t) => a + (b - a) * t;
    }
}
