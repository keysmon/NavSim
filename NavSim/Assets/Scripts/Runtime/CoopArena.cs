using UnityEngine;
using Unity.MLAgents;

namespace NavSim.Runtime
{
    // M7 plate-and-door coop arena controller (spec sec 2/5): the SINGLE authority for layout, plate/door
    // state, outcome detection, reward application (via ArmRouting ONLY), and the episode boundary.
    // Flat walled arena (half-size 11, M5 footprint) split at z=0 by a divider wall with a doorway at x=0:
    // near chamber z<0 (spawns + plate), far chamber z>0 (goal).
    //
    // THE TICK SEAM (load-bearing for eval): MonoBehaviour FixedUpdate does NOT fire under the eval
    // harness's manual script stepping (Physics.simulationMode=Script + Academy.EnvironmentStep() - the
    // reason M6's movers froze during eval). ALL per-step logic lives in Tick(fixedDt); FixedUpdate is
    // exactly `if (!EvalMode) Tick(...)`; the eval harness calls arena.Tick explicitly after each step
    // pair. Inside Tick: plate/door/success-latch logic ALWAYS runs; reward application AND boundary
    // calls (End/Interrupted + ResetEpisode) are gated on !EvalMode (the harness owns the boundary and
    // reads the latches).
    public class CoopArena : MonoBehaviour
    {
        [Header("Wiring (assign in Editor)")]
        [SerializeField] private CoopAgent[] agents;       // length 2, shared policy
        [SerializeField] private Transform goal;
        [SerializeField] private Transform plate;
        [Tooltip("The door block filling the doorway gap: SetActive(false) while open (an open door is a gap).")]
        [SerializeField] private GameObject door;

        [Header("Geometry (Global-Constraints pinned values)")]
        [SerializeField] private float arenaHalf = 11f;    // M5 footprint
        [SerializeField] private float plateRadius = 1.2f; // XZ occupancy radius
        [SerializeField] private float goalRadius = 1.5f;
        [SerializeField] private int maxEpisodeSteps = 3000;
        [Tooltip("C0 goal-distance bootstrap horizon: within C0 the goal ramps from the near chamber beside " +
                 "the doorway (door already open) to the deep far chamber over this many C0 SUCCESSES (not " +
                 "wall-clock/Academy steps) - the ramp advances per success, so it can never outrun the " +
                 "policy: the time-gated version ran 2.5x fast vs trainer steps AND outran the young " +
                 "policy, producing the diagnosed collapse.")]
        [SerializeField] private int c0BootstrapSuccesses = 200;
        [Tooltip("C1 dwell competence ramp start (seconds): episode-scale so an incidental plate tap during " +
                 "C0-taught goal-seeking converts into a full C1 success chain, giving lesson-1's sparse " +
                 "reward its first gradient instead of a closed door reading as a wall (spec [AMENDED " +
                 "2026-07-19]).")]
        [SerializeField] private float c1DwellStart = 30f;
        [Tooltip("C1 dwell ramp horizon: successes at lesson 1 (training only) before the dwell reaches its " +
                 "target 4.0 s hardness. Competence-ramped down, mirroring c0BootstrapSuccesses. EvalMode " +
                 "bypasses the ramp entirely (lesson-1 dwell is always 4.0 s under eval).")]
        [SerializeField] private int c1RampSuccesses = 200;

        // ---- Eval surface (Task 6 consumes these names verbatim) ----
        public bool EvalMode { get; set; }
        public ArmRouting.Arm ArmMode
        {
            get => _armMode;
            set { _armMode = value; SyncGroupRegistration(); }
        }
        public Vector3 GoalPosition => goal.position;
        public Vector3 PlatePosition => plate.position;
        public bool DoorOpen { get; private set; }
        public int StepsThisEpisode { get; private set; }
        public CoopAgent[] Agents => agents;
        public int LastScorerIndex { get; private set; } = -1;
        public int LastHolderIndex { get; private set; } = -1; // most plate-occupied steps this episode
        public bool Success { get; private set; }              // latched until next ResetEpisode

        private ArmRouting.Arm _armMode = ArmRouting.Arm.Poca; // default poca for in-Editor play
        private SimpleMultiAgentGroup _group;                  // created ONCE; registered ONLY under Poca
        private bool _groupRegistered;
        private System.Random _layoutRng = new System.Random(); // ALL layout randomness (never UnityEngine.Random
                                                                // here - the M6 Task-9 global-Random pairing lesson)
        private int _lesson;                                   // 0-3 (C0-C3)
        private int _c0Successes;                              // monotone C0-success counter driving the ramp;
                                                                 // never reset (process-lifetime; C1+ ignores it)
        private int _c1Successes;                              // monotone C1-success counter driving the dwell
                                                                 // ramp; never reset; training only (EvalMode
                                                                 // ignores it - see DwellForLesson)
        private float _vacatedClock = PlateDoor.InitialSecondsSinceVacated;
        private readonly int[] _plateSteps = new int[2];       // per-agent plate-occupied step counts

        // Per-episode layout isolation for the paired eval; training leaves the RNG free-running.
        public void SeedLayoutRng(int seed) => _layoutRng = new System.Random(seed);

        // Apply the Global-Constraints geometry table: C0 door always open (+ goal bootstrap); C1 plate
        // 2u beside the doorway, dwell 4 s; C2 plate near-chamber corner >= 10u from the doorway, dwell
        // 2 s; C3 = C2 geometry, dwell 1 s. Geometry is applied by the NEXT ResetEpisode (the eval calls
        // SeedLayoutRng + SetLesson + ResetEpisode in that order).
        public void SetLesson(int lesson) => _lesson = Mathf.Clamp(lesson, 0, 3);

        // C1 dwell is competence-ramped during training (30s -> 4.0s over c1RampSuccesses lesson-1
        // successes, mirroring the c0BootstrapSuccesses idiom) so an incidental plate tap converts to a
        // success while the policy is still C0-naive; EvalMode always measures the full 4.0s hardness.
        private float DwellForLesson => _lesson switch
        {
            1 => EvalMode
                ? 4f
                : Mathf.Lerp(c1DwellStart, 4f, Mathf.Clamp01((float)_c1Successes / Mathf.Max(c1RampSuccesses, 1))),
            2 => 2f,
            _ => 1f
        }; // C0 uses alwaysOpen

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
            if (agents == null || agents.Length != 2 || goal == null || plate == null || door == null) return;

            // TRAINING-only curriculum/arm sync (env-params are inert without a communicator; the eval
            // sets ArmMode/SetLesson directly). A lesson change is an environment-imposed boundary ->
            // Interrupted (not the agents' fault), clean trajectory cut (the M6 lesson-advance idiom).
            if (!EvalMode && Academy.Instance.IsCommunicatorOn)
            {
                int lesson = ReadLessonParam();
                if (lesson != _lesson)
                {
                    SetLesson(lesson);
                    InterruptEpisode();
                    ResetEpisode();
                    return;
                }
                ArmMode = DecodeArm(Academy.Instance.EnvironmentParameters.GetWithDefault("arm_mode", 2f));
            }

            // ---- Plate / door / latch logic: ALWAYS runs (eval included) ----
            bool occupied = false;
            for (int i = 0; i < 2; i++)
            {
                if (!OnPlate(agents[i])) continue;
                occupied = true;
                _plateSteps[i]++;
            }
            _vacatedClock = PlateDoor.Step(_vacatedClock, occupied, fixedDt);
            DoorOpen = PlateDoor.IsOpen(_vacatedClock, occupied, DwellForLesson, _lesson == 0);
            door.SetActive(!DoorOpen);
            StepsThisEpisode++;
            UpdateHolderLatch();

            // ---- Rewards: per-step time cost, routed per arm (the ONLY other reward is the outcome) ----
            if (!EvalMode) ApplySplit(ArmRouting.PerStep(_armMode, -1f / maxEpisodeSteps), 0);

            // ---- Outcome: any agent within goalRadius of the goal while the door is open ----
            int scorer = -1;
            for (int i = 0; i < 2 && scorer < 0; i++)
                if (DoorOpen && Vector3.Distance(agents[i].transform.position, goal.position) < goalRadius)
                    scorer = i;
            if (scorer >= 0 && !Success)
            {
                Success = true;                 // latched until next ResetEpisode (the eval reads these)
                LastScorerIndex = scorer;
                UpdateHolderLatch();
                if (!EvalMode)
                {
                    if (_lesson == 0) _c0Successes++;  // competence-gated ramp driver (training C0 only)
                    if (_lesson == 1) _c1Successes++;  // competence-gated dwell-ramp driver (training C1 only)
                    ApplySplit(ArmRouting.Outcome(_armMode), scorer);
                    EndEpisodePerArm();
                    ResetEpisode();
                }
                return;
            }

            // ---- Timeout boundary (training only; the eval harness owns its own step cap) ----
            if (!EvalMode && StepsThisEpisode >= maxEpisodeSteps)
            {
                InterruptEpisode();
                ResetEpisode();
            }
        }

        // Seeded fresh layout + latch reset. Spawns: SYMMETRIC mirrored near-chamber pair (sides randomly
        // swapped) + per-agent +-0.5u jitter - the spec's symmetry-breaking pre-registered lever, ACTIVE
        // from the start; base separation ~5u keeps the >=2u minimum by construction. Goal: within C0 it
        // starts in the NEAR chamber beside the doorway (~2-4u from spawns, door already open) and
        // migrates out from the near chamber toward the far-chamber draw as competence grows (the
        // distance bootstrap; fixes the diagnosed flatline - the old ramp only varied depth WITHIN the far chamber, so even
        // the easiest C0 goal sat ~8u away through the 3u doorway and no gradient ever formed); C1+ is
        // always the far-chamber placement. Plate: per the lesson geometry table (C0 shows C1's plate so
        // the world is perceptually consistent from the first step even while the door ignores it).
        public void ResetEpisode()
        {
            Success = false;
            LastScorerIndex = -1;
            LastHolderIndex = -1;
            StepsThisEpisode = 0;
            _plateSteps[0] = 0;
            _plateSteps[1] = 0;
            _vacatedClock = PlateDoor.InitialSecondsSinceVacated;

            // Agents: mirrored (+-x) at a random near-chamber depth, sides swapped at random, jittered.
            float baseX = 2.5f;
            float spawnZ = Lerp(-9f, -6f, NextFloat());
            int flip = NextFloat() < 0.5f ? -1 : 1;
            for (int i = 0; i < 2; i++)
            {
                float side = (i == 0 ? -1f : 1f) * flip;
                Vector3 p = new Vector3(side * baseX + Jitter(), 0.5f, spawnZ + Jitter());
                agents[i].TeleportTo(p, (float)(_layoutRng.NextDouble() * 360.0));
            }

            // Goal: far-chamber candidate first, computed EXACTLY as before (unchanged RNG draw count/
            // order - x draw then z draw) so pairing outside C0 is untouched. C0 ONLY: Lerp that
            // candidate back from the doorway line/near chamber by t=ramp, so t=0 sits at the doorway
            // (x) / near chamber (z=-4.5, ~2-4u from spawns) and t=1 reproduces today's far placement
            // exactly (Lerp endpoint). C1+ always uses the far-chamber candidate untouched.
            float ramp = _lesson == 0
                ? Mathf.Clamp01((float)_c0Successes / Mathf.Max(c0BootstrapSuccesses, 1))
                : 1f;
            float zMax = Lerp(3.5f, arenaHalf - 1.5f, ramp);
            float xMax = Lerp(2f, arenaHalf - 1.5f, ramp);
            float farX = Lerp(-xMax, xMax, NextFloat());
            float farZ = Lerp(2f, zMax, NextFloat());
            float goalX = _lesson == 0 ? Lerp(0f, farX, ramp) : farX;
            float goalZ = _lesson == 0 ? Lerp(-4.5f, farZ, ramp) : farZ;
            goal.position = new Vector3(goalX, 0.5f, goalZ);

            // Plate: geometry table. C0/C1: 2u beside the doorway (random side, just inside the near
            // chamber); C2/C3: a near-chamber corner >= 10u from the doorway (random corner).
            float plateSide = NextFloat() < 0.5f ? -1f : 1f;
            plate.position = _lesson >= 2
                ? new Vector3(plateSide * (arenaHalf - 2f) + Jitter(), 0.02f, -(arenaHalf - 2f) + Jitter())
                : new Vector3(plateSide * 2f, 0.02f, -1.5f);

            // Door state for the fresh episode: closed unless C0 (fresh clock exceeds every dwell).
            DoorOpen = PlateDoor.IsOpen(_vacatedClock, false, DwellForLesson, _lesson == 0);
            door.SetActive(!DoorOpen);
        }

        // ---- internals ----

        private bool OnPlate(CoopAgent a)
        {
            if (!a.Grounded) return false;
            Vector3 d = a.transform.position - plate.position;
            return new Vector2(d.x, d.z).magnitude < plateRadius;
        }

        // Continuous latch (not success-only) so a TIMEOUT episode still reports who held the plate most
        // (the eval's plate-hold/role trackers need it). Tie at zero -> -1 (nobody held).
        private void UpdateHolderLatch()
        {
            if (_plateSteps[0] == 0 && _plateSteps[1] == 0) LastHolderIndex = -1;
            else LastHolderIndex = _plateSteps[0] >= _plateSteps[1] ? 0 : 1;
        }

        // The ONLY reward surface: an ArmRouting split. scorer/partner -> per-agent AddReward; group ->
        // the multi-agent group (Poca only; PPO splits carry group=0 so PPO arms never touch the group).
        private void ApplySplit(ArmRouting.Split s, int scorerIdx)
        {
            if (s.scorer != 0f) agents[scorerIdx].AddReward(s.scorer);
            if (s.partner != 0f) agents[1 - scorerIdx].AddReward(s.partner);
            if (s.group != 0f && _group != null) _group.AddGroupReward(s.group);
        }

        private void EndEpisodePerArm()
        {
            if (_armMode == ArmRouting.Arm.Poca && _group != null) _group.EndGroupEpisode();
            else { agents[0].EndEpisode(); agents[1].EndEpisode(); }
        }

        private void InterruptEpisode()
        {
            if (_armMode == ArmRouting.Arm.Poca && _group != null) _group.GroupEpisodeInterrupted();
            else { agents[0].EpisodeInterrupted(); agents[1].EpisodeInterrupted(); }
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

        // TRAINING (communicator on): the trainer's coop_difficulty drives the lesson; default = the
        // HARDEST rung so a resumed run never regresses. DEMO/eval (no communicator): default C0 (most
        // legible) - the demo UI / eval drive the lesson directly (the M6 ReadDifficulty idiom).
        private int ReadLessonParam() => Mathf.Clamp(
            Mathf.RoundToInt(Academy.Instance.EnvironmentParameters.GetWithDefault(
                "coop_difficulty", Academy.Instance.IsCommunicatorOn ? 3f : 0f)), 0, 3);

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
