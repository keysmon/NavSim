using UnityEngine;
using Unity.MLAgents;

namespace NavSim.Runtime
{
    // M8 push-the-ramp arena controller (a FORK of CoopArena): the SINGLE authority for layout, ramp
    // heaviness, outcome detection, reward application (via ArmRouting ONLY), and the episode boundary.
    // Flat walled arena (half-size 11, M5 footprint). A dynamic ramp must be pushed to its marked target;
    // once placed it forms the intended path to the goal on the ledge. Success also requires the
    // per-episode placement latch, so incidental goal-radius overlap cannot bypass the task sequence.
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

        [Header("S0 push curriculum")]
        // S0 begins just outside targetRadius: the agent must still push (never starts pre-placed), but only
        // a short distance. Successes recede the ramp to the normal scene-authored start. Eval is always normal.
        [SerializeField] private float s0InitialPushDistance = 1.75f;
        [SerializeField] private int s0StartSuccesses = 200;

        [Header("Shaping")]
        // ramp-to-target (PUSH) shaping: guides the ramp toward its target (the push stage).
        [SerializeField] private float shapingScale = 0.01f;
        // GOAL (climb) shaping: the M5/M6 distance-bootstrap (RewardCalculator pattern) that M8's RampArena
        // omitted. Potential-based per-agent HORIZONTAL (xz) goal-distance reward, STAGED to fire only after the
        // ramp is placed (see Tick). xz (not 3D) so the elevated goal cannot be farmed by jumping; staged so it
        // never pulls the agent off the push before placement. Guides the climb the push-shaping cannot reach.
        // Orthogonal to the coop lever (mass/geometry force cooperation regardless of any reward term); POCA
        // per-agent-vs-group routing is pinned before the batch.
        [SerializeField] private float goalShapingScale = 0.05f;

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
        private float[] _prevDistToGoal;                       // per-agent, for goal-distance shaping
        private float _epStartDistance;                        // curriculum instrumentation for the ended episode
        // Per-episode sub-stage latches (StatsRecorder instrumentation: WHICH stage stalls, not just reward).
        private bool _epReachedApproach, _epAscended, _epPlaced;

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

            // GOAL (climb) shaping - the distance-bootstrap the push-shaping cannot provide. HORIZONTAL (xz)
            // distance ONLY (the goal is elevated, so 3D distance would reward JUMPING toward it instead of
            // walking up the slope - the Run-1 rollout showed exactly that). STAGED: applied only once the ramp
            // is placed (RampAtTarget), so it guides the climb and never pulls the agent off the push before the
            // ramp is placed (the push->climb seam misguide). _prevDistToGoal is tracked every step so the first
            // post-placement delta is not a spurious jump. Per-agent, routed to that agent (scorer=i).
            if (!EvalMode && _prevDistToGoal != null)
            {
                for (int i = 0; i < agents.Length; i++)
                {
                    Vector3 ap = agents[i].transform.position;
                    float dGoal = HorizontalDist(ap, goal.position);   // xz only - jumping cannot exploit it
                    if (RampAtTarget)
                    {
                        float gdelta = _prevDistToGoal[i] - dGoal;   // positive when the agent nears the goal (xz)
                        if (gdelta != 0f) ApplySplit(ArmRouting.PerStep(_armMode, gdelta * goalShapingScale), i);
                    }
                    _prevDistToGoal[i] = dGoal;   // always tracked (even pre-placement) so the first delta is clean

                    // Sub-stage latches (StatsRecorder): reached the climb approach, then the upper slope/crest
                    // (large +z progress AND height - NOT a mere jump near spawn, which fooled the old y>1.5
                    // latch in the Run-1 rollout). Placed = RampAtTarget (below).
                    if (Vector3.Distance(new Vector3(ap.x, 0f, ap.z),
                                         new Vector3(rampTarget.position.x, 0f, rampTarget.position.z - 2f)) < 2f)
                        _epReachedApproach = true;
                    if (ap.z >= 3.5f && ap.y >= 1.9f) _epAscended = true;   // upper slope/crest, not a spawn-area jump
                }
            }
            if (RampAtTarget) _epPlaced = true;

            // Per-step time cost (M7 routing).
            if (!EvalMode) ApplySplit(ArmRouting.PerStep(_armMode, -1f / maxEpisodeSteps), 0);

            // Success = an agent reaches the goal after this episode has placed the ramp. The explicit
            // sequence latch is load-bearing: goal-radius overlap alone is not proof that the ledge was
            // reached (a jump at the south wall can graze the spherical trigger).
            int scorer = -1;
            if (_epPlaced)
            {
                for (int i = 0; i < agents.Length && scorer < 0; i++)
                    if (Vector3.Distance(agents[i].transform.position, goal.position) < goalRadius) scorer = i;
            }

            if (scorer >= 0 && !Success)
            {
                Success = true;
                LastScorerIndex = scorer;
                JointPushFrac = _rampMoveSteps > 0 ? (float)_jointPushSteps / _rampMoveSteps : 0f;
                if (!EvalMode)
                {
                    // Competence counters drive the TRAINING curriculum only (eval uses the hard branch);
                    // never advance them under EvalMode (M7 idiom - avoids a train/eval instance-reuse footgun).
                    if (_lesson == 0) _s0Successes++;
                    else if (_lesson == 1) _s1Successes++;
                    else if (_lesson == 2) _s2Successes++;
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
        // position (competence-ramped distance for S0/S3); agents spawned BESIDE the ramp on the -x side of
        // its vertical -x face, facing +x (the lateral-push mechanic), guarded for 1 or 2 agents.
        public void ResetEpisode()
        {
            RecordSubstages();   // log which stage the JUST-ENDED episode reached (before the latches reset)
            _epReachedApproach = false; _epAscended = false; _epPlaced = false;
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

            // Ramp start distance: S0 starts just outside the target and recedes to the normal near start as
            // competence grows. S1/S2 use the normal near start. S3 retains its near->far stretch curriculum.
            // EvalMode always selects the hard endpoint for the active lesson.
            Vector3 nearStart = NearRampStart();
            Vector3 startPos;
            if (_lesson == 0)
            {
                float startX = RampCurriculum.S0StartX(
                    _s0Successes,
                    s0StartSuccesses,
                    rampTarget.position.x,
                    nearStart.x,
                    s0InitialPushDistance,
                    EvalMode);
                startPos = new Vector3(startX, nearStart.y, nearStart.z);
            }
            else if (_lesson >= 3)
            {
                float startRamp = EvalMode ? 1f : Competence.Ramp01(_s2Successes, s2RampSuccesses);
                startPos = Vector3.Lerp(nearStart, FarRampStart(), startRamp);
            }
            else
            {
                startPos = nearStart;
            }
            ramp.ResetTo(startPos, Quaternion.identity);
            _prevRampToTarget = Vector3.Distance(startPos, rampTarget.position);
            _epStartDistance = _prevRampToTarget;

            // Spawns (LATERAL-push mechanic - verify-early #1 geometry). Agents stand BESIDE the ramp on the
            // -x side of its tall vertical -x face, facing +x, so driving forward shoves the ramp toward
            // rampTarget. The ramp's -x face sits at startPos.x - WedgeHalfWidth (2u); agents are placed
            // AgentBehindOffset (=2u + ~1.2u clearance = 3.2u) further -x, matching the selftest's verified
            // behindX, and keyed off the ACTUAL startPos so they stay beside the face wherever it drifts
            // (S0 near->far) - never on the +z climb slope. Two agents are offset +-0.8u in z so both press
            // the tall face. Seeded jitter is small (a wide spread here would miss the face and flatline the
            // push). yaw 90deg = forward +x (into the push).
            float behindX = startPos.x - AgentBehindOffset;
            for (int i = 0; i < agents.Length; i++)
            {
                float zOff = agents.Length == 1 ? 0f : (i == 0 ? -0.8f : 0.8f);
                Vector3 p = new Vector3(behindX + Jitter(), 0.5f, startPos.z + zOff + Jitter());
                agents[i].TeleportTo(p, 90f + YawJitter());
            }

            Physics.SyncTransforms();
            InitPrevGoalDist();
        }

        // ---- internals ----

        // Per-agent goal-distance baseline for the potential-based goal shaping (call after spawns + SyncTransforms).
        private void InitPrevGoalDist()
        {
            if (_prevDistToGoal == null || _prevDistToGoal.Length != agents.Length)
                _prevDistToGoal = new float[agents.Length];
            for (int i = 0; i < agents.Length; i++)
                _prevDistToGoal[i] = HorizontalDist(agents[i].transform.position, goal.position);
        }

        // Horizontal (xz) distance - goal shaping ignores the vertical axis so the elevated goal cannot be
        // farmed by jumping toward it (the Run-1 rollout showed 3D distance did exactly that).
        private static float HorizontalDist(Vector3 a, Vector3 b) =>
            new Vector2(a.x - b.x, a.z - b.z).magnitude;

        // Sub-stage instrumentation -> tfevents (StatsRecorder): what fraction of episodes reach each stage, so
        // the confirm shows WHICH stage stalls (a reward curve cannot). Training only; skips the first (empty) reset.
        private void RecordSubstages()
        {
            if (EvalMode || StepsThisEpisode <= 0 || !Academy.Instance.IsCommunicatorOn) return;
            var sr = Academy.Instance.StatsRecorder;
            sr.Add("substage/placed", _epPlaced ? 1f : 0f);
            sr.Add("substage/reachedApproach", _epReachedApproach ? 1f : 0f);
            sr.Add("substage/ascended", _epAscended ? 1f : 0f);
            sr.Add("substage/reachedGoal", Success ? 1f : 0f);
            sr.Add("curriculum/startDistance", _epStartDistance);
        }

        // Ramp start endpoints (lateral-push layout; rampStart wires the near reset when present in-scene).
        // Near = rampStart (~(-5,1.24,2)): the normal 5m +x push. Far = the SAME z-line and height, further -x
        // for a longer +x push (S3 stretch). Far x is bounded (-6.5) so an agent spawned
        // AgentBehindOffset (3.2u) behind the ramp - plus its radius and jitter - stays inside the -x wall
        // (arenaHalf 11). The pre-lateral Far ((-(arenaHalf-3),0.5,-(arenaHalf-3))) was the old +z/embedded
        // layout - unreachable-vertical and off the push axis (verify-early #1 follow-up #2, fixed here).
        private Vector3 NearRampStart() => rampStart != null ? rampStart.position : new Vector3(0f, 0.5f, 1.5f);
        private Vector3 FarRampStart()
        {
            Vector3 near = NearRampStart();
            return new Vector3(-6.5f, near.y, near.z);
        }

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

        // Lateral-push spawn geometry: agents sit AgentBehindOffset on the -x side of the ramp center, so their
        // front edge is just short of the ramp's -x face (WedgeHalfWidth 2u + ~1.2u clearance) and driving +x
        // contacts it (verify-early #1). Jitter is small so the tight push geometry survives the randomness.
        private const float AgentBehindOffset = 3.2f;
        private float NextFloat() => (float)_layoutRng.NextDouble();
        private float Jitter() => Lerp(-0.3f, 0.3f, NextFloat());      // small position jitter (+-0.3u)
        private float YawJitter() => Lerp(-10f, 10f, NextFloat());     // small facing jitter (+-10deg, stays ~+x)
        private static float Lerp(float a, float b, float t) => a + (b - a) * t;
    }
}
