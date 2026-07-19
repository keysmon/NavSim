using System.Globalization;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;   // InferenceDevice
using Unity.InferenceEngine;     // ModelAsset (com.unity.ai.inference)
using NavSim.Runtime;

// M7 GROUP-AWARE PAIRED eval — fork of M6SearchEval for the plate-and-door cooperative-sacrifice task. The
// reproducible artifact behind training/eval/m7_coop.csv. Unlike M6 (three sensor-distinct scenes), ALL THREE
// arms (selfish/shared/poca) share ONE scene (Coop.unity) and the SAME shared policy; the arm selects only the
// MODEL SLOT (Assets/Models/M7/m7_<arm>_s<seed>.onnx) + arena.ArmMode. Each arm run writes to ONE m7_coop.csv
// IDEMPOTENTLY (preserve other arms' rows, replace this arm's — re-running an arm self-heals).
//
// PAIRING: for each (seed, lesson, episode) we seed arena.SeedLayoutRng(paired). Layout (goal/plate/spawns)
// comes ONLY from that System.Random (ResetEpisode never reads _armMode), so the m7_pairing_<arm>.csv sidecars
// are byte-identical across arms — the paired-measurement guarantee.
//
// TWO BINDING HANDOFFS FROM THE TASK-3 REVIEW:
//   (H1) HARD-ASSERT lesson >= 1. CoopArena's C0 goal ramp reads Academy.TotalStepCount (seed-INDEPENDENT), so
//        C0 can never be a paired-measurement surface. Lesson 0 is logged as an error and skipped. Defaults 1-3.
//   (H2) Read every per-episode output AT the Success tick. Under EvalMode LastHolderIndex/StepsThisEpisode keep
//        updating on further Ticks after the latch, so the episode loop breaks IMMEDIATELY when arena.Success
//        flips true and records from the arena's latches at that break.
//
// THE TICK SEAM (Task 3): under Physics.simulationMode=Script the arena's FixedUpdate does NOT fire, so per step
// the harness calls `Academy.EnvironmentStep(); Physics.Simulate(dt); arena.Tick(dt);` — the explicit Tick runs
// the door/plate/success logic. EvalMode=true makes the arena's Tick skip EndEpisode/Interrupted/ResetEpisode
// (the harness owns the boundary and reads the latches). One warm EnvironmentStep precedes the first SetModel
// (the M6 SetModel-ordering lesson; free for ray-only sensors, kept as pattern).
public static class M7CoopEval
{
    // Set by M7EvalBatch before entering Play (static -> survives DisableDomainReload). Arm is the BARE token
    // (selfish/shared/poca); the literal "m7_" lives in the model/pairing templates, never in Arm.
    public static string Arm = "poca";
    public static ArmRouting.Arm ArmMode = ArmRouting.Arm.Poca;
    public static int[] Seeds = { 0, 1, 2 };              // n=3, pre-registered add-to-5 (M5/M6 discipline)
    public static int[] Lessons = { 1, 2, 3 };           // C1-C3 only (H1: C0 is seed-independent, never paired)
    public static int EpisodesPerLesson = 25;
    public static string CsvName = "m7_coop.csv";

    private const int MaxSteps = 3000;
    private const float PlateRadius = 1.2f;               // XZ occupancy radius (arena's plateRadius; harness copy)

    [MenuItem("Tools/NavSim/Run M7 Coop Eval (current arm)")]
    public static void Run()
    {
        if (!Application.isPlaying) { Debug.LogError("[M7Eval] Enter Play mode first (needs a running Academy)."); return; }
        var arena = Object.FindAnyObjectByType<CoopArena>();
        if (arena == null) { Debug.LogError("[M7Eval] Missing CoopArena in the scene."); return; }
        if (arena.Agents == null || arena.Agents.Length != 2) { Debug.LogError("[M7Eval] CoopArena needs exactly 2 agents."); return; }
        if (EpisodesPerLesson > 100)
            Debug.LogWarning($"[M7Eval] EpisodesPerLesson={EpisodesPerLesson} > 100 breaks the paired seed " +
                             "(ep overflows the *100 lesson stride). Keep it <= 100.");
        SetupHarness(arena);

        // IDEMPOTENT per arm: preserve OTHER arms' rows, REPLACE this arm's. The arm runs are sequential (one
        // Unity lock) so read-modify-write is safe. No silent duplicate accretion; re-running an arm self-heals.
        string path = CsvPath(CsvName);
        const string Header = "arm,seed,lesson,episode,success,steps,plate_hold_frac,holder_idx,scorer_idx,both_on_plate_frac";
        var sb = new StringBuilder();
        sb.Append(Header).Append('\n');
        if (File.Exists(path))
            foreach (var line in File.ReadAllLines(path))
            {
                if (string.IsNullOrWhiteSpace(line) || line.StartsWith("arm,")) continue; // drop blanks + old header
                if (line.StartsWith(Arm + ",")) continue;                                  // drop THIS arm's stale rows
                sb.Append(line).Append('\n');                                              // keep other arms' rows
            }

        // Per-arm pairing fingerprint sidecar (arm-INDEPENDENT layout) — byte-diffed across arms to prove pairing.
        var fp = new StringBuilder("seed,lesson,episode,gx,gy,gz,px,py,pz,s0x,s0y,s0z,s1x,s1y,s1z\n");

        int ran = 0;
        foreach (var seed in Seeds)
        {
            var model = AssetDatabase.LoadAssetAtPath<ModelAsset>($"Assets/Models/M7/m7_{Arm}_s{seed}.onnx");
            if (model == null) { Debug.LogWarning($"[M7Eval] missing m7_{Arm}_s{seed}, skipping"); continue; }
            // Same model to BOTH agents — shared policy (the M7 design: one brain, two bodies).
            arena.Agents[0].SetModel("CoopAgent", model, InferenceDevice.Burst);
            arena.Agents[1].SetModel("CoopAgent", model, InferenceDevice.Burst);
            foreach (var lesson in Lessons)
            {
                if (lesson < 1)
                {
                    // H1: C0's goal ramp reads Academy.TotalStepCount (seed-independent) -> not a paired surface.
                    Debug.LogError($"[M7Eval] lesson {lesson} < 1 rejected (C0 goal ramp is seed-INDEPENDENT; " +
                                   "never a paired-measurement surface). Skipping.");
                    continue;
                }
                for (int ep = 0; ep < EpisodesPerLesson; ep++)
                {
                    AppendEpisode(arena, Arm, seed, lesson, ep, sb, fp);
                    ran++;
                }
            }
        }
        WriteCsv(sb, path);
        if (ran > 0) File.WriteAllText(CsvPath($"m7_pairing_{Arm}.csv"), fp.ToString());
        Debug.Log($"[M7Eval] arm={Arm}: wrote {ran} episodes to {path} (+ pairing fingerprint). " +
                  $"If 0, place ONNX in Assets/Models/M7/m7_{Arm}_s<seed>.onnx.");
    }

    // No-model boundary self-test (4 MUST-true checks). Runs in Play, no models needed. See M7EvalBatch.SelftestHeadless.
    [MenuItem("Tools/NavSim/M7 Coop Eval Selftest (no model)")]
    public static void Selftest()
    {
        if (!Application.isPlaying) { Debug.LogError("[M7Eval] Enter Play mode first."); return; }
        var arena = Object.FindAnyObjectByType<CoopArena>();
        if (arena == null || arena.Agents == null || arena.Agents.Length != 2) { Debug.LogError("[M7Eval] Missing CoopArena/agents."); return; }
        SetupHarness(arena);

        // (1) PAIRED: same seed -> identical goal/plate/spawn tuples twice (lesson 1: ramp fully open, seed-only).
        var a = CaptureLayout(arena, 123, 1);
        var b = CaptureLayout(arena, 123, 1);
        bool paired = TupleEq(a, b);

        // (2) VARIES: a different seed -> a different tuple (guards a stuck RNG).
        var c = CaptureLayout(arena, 999, 1);
        bool varies = !TupleEq(a, c);

        // (3) EvalMode SUPPRESSES THE BOUNDARY: force a success (agent0 on the goal, agent1 holding the plate so
        // the door opens), step to the latch, assert Success==true AND goal/plate UNCHANGED (no mid-episode
        // ResetEpisode — the M6 EvalMode lesson, group edition). Hardened: tick 2 more past the latch, re-assert.
        arena.SeedLayoutRng(202); arena.SetLesson(2); arena.ResetEpisode(); Physics.SyncTransforms();
        Vector3 goalBefore = arena.GoalPosition, plateBefore = arena.PlatePosition;
        arena.Agents[0].TeleportTo(new Vector3(goalBefore.x, 0.5f, goalBefore.z), 0f);   // onto the goal
        arena.Agents[1].TeleportTo(new Vector3(plateBefore.x, 0.5f, plateBefore.z), 0f); // holds the plate open
        Physics.SyncTransforms();
        bool latched = false;
        for (int k = 0; k < 90 && !latched; k++) { StepTick(arena); latched = arena.Success; }
        StepTick(arena); StepTick(arena);   // 2 extra ticks: a delayed reset path would move goal/plate here
        bool posStable = Approximately(arena.GoalPosition, goalBefore) && Approximately(arena.PlatePosition, plateBefore);
        bool evalModeOk = latched && posStable;

        // (4) DOOR FOLLOWS PLATE under script stepping: occupied -> open; vacated past dwell -> closed (C3 dwell=1s).
        arena.SeedLayoutRng(303); arena.SetLesson(3); arena.ResetEpisode(); Physics.SyncTransforms();
        Vector3 plate = arena.PlatePosition;
        Vector3 farA = new Vector3(0f, 0.5f, -3f), farB = new Vector3(2f, 0.5f, -3f);
        arena.Agents[1].TeleportTo(farB, 0f);                                   // partner well off the plate
        arena.Agents[0].TeleportTo(new Vector3(plate.x, 0.5f, plate.z), 0f);    // onto the plate
        Physics.SyncTransforms();
        bool opened = false;
        for (int k = 0; k < 60 && !opened; k++) { StepTick(arena); opened = arena.DoorOpen; }
        arena.Agents[0].TeleportTo(farA, 0f);                                   // vacate the plate
        Physics.SyncTransforms();
        bool closed = false;
        for (int k = 0; k < 200 && !closed; k++) { StepTick(arena); closed = !arena.DoorOpen; }
        bool doorOk = opened && closed;

        Debug.Log($"[M7Eval] SELFTEST — paired(same seed)={paired} (MUST be true); varies(diff seed)={varies} (MUST be true); " +
                  $"EvalMode-suppresses-boundary: latched={latched} & pos-stable={posStable} -> {evalModeOk} (MUST be true); " +
                  $"door-follows-plate: opened={opened} & closed-after-dwell={closed} -> {doorOk} (MUST be true).");
    }

    private static void SetupHarness(CoopArena arena)
    {
        Physics.simulationMode = SimulationMode.Script;   // runtime-only; MonoBehaviour FixedUpdate won't fire
        arena.ArmMode = ArmMode;                          // selects reward routing + group registration (benign in eval)
        arena.Agents[0].MaxStep = 0;                      // harness owns the boundary; no max-step EndEpisode
        arena.Agents[1].MaxStep = 0;
        arena.EvalMode = true;                            // arena.Tick must NOT End/Interrupt/Reset (this task PROVES it)
        Step();                                           // warm step BEFORE any SetModel (M6 ordering lesson; free here)
    }

    // Seed a paired layout, capture goal/plate/spawns as VALUES (never live Transforms). Lesson >= 1 required so
    // the goal placement is seed-only (H1). ResetEpisode is the sole layout driver.
    private static (Vector3 g, Vector3 p, Vector3 s0, Vector3 s1) CaptureLayout(CoopArena arena, int seed, int lesson)
    {
        arena.SeedLayoutRng(seed);
        arena.SetLesson(lesson);
        arena.ResetEpisode();
        Physics.SyncTransforms();
        return (arena.GoalPosition, arena.PlatePosition,
                arena.Agents[0].transform.position, arena.Agents[1].transform.position);
    }

    // One paired episode: seed layout, capture fingerprint VALUES, run to the Success latch or MaxSteps, then
    // record from the arena's latches (H2: read AT the latch) + harness-side XZ occupancy counters.
    private static void AppendEpisode(CoopArena arena, string arm, int seed, int lesson, int ep,
        StringBuilder sb, StringBuilder fp)
    {
        int paired = seed * 10000 + lesson * 100 + ep;
        arena.SeedLayoutRng(paired);
        arena.SetLesson(lesson);
        arena.ResetEpisode();
        Physics.SyncTransforms();                         // Script mode defers transform->physics sync; push it

        Vector3 goal = arena.GoalPosition, plate = arena.PlatePosition;
        Vector3 sp0 = arena.Agents[0].transform.position, sp1 = arena.Agents[1].transform.position;
        // PAIRING fingerprint (arm-INDEPENDENT): layout for this (seed,lesson,ep). The three arms' sidecars must
        // be byte-identical -> proves the paired-eval guarantee. Verify by diff at analysis time.
        fp?.AppendLine(string.Format(CultureInfo.InvariantCulture,
            "{0},{1},{2},{3:F3},{4:F3},{5:F3},{6:F3},{7:F3},{8:F3},{9:F3},{10:F3},{11:F3},{12:F3},{13:F3},{14:F3}",
            seed, lesson, ep, goal.x, goal.y, goal.z, plate.x, plate.y, plate.z,
            sp0.x, sp0.y, sp0.z, sp1.x, sp1.y, sp1.z));

        int plateHold = 0, bothOnPlate = 0;
        bool success = false;
        for (int s = 0; s < MaxSteps; s++)
        {
            StepTick(arena);
            // Harness-side occupancy: XZ distance <= plateRadius, NO grounded check (intentionally distinct from
            // the arena's grounded holder latch — an airborne-over-plate agent counts here, not in holder_idx).
            bool a0 = OnPlateXZ(arena.Agents[0].transform.position, plate);
            bool a1 = OnPlateXZ(arena.Agents[1].transform.position, plate);
            if (a0 || a1) plateHold++;
            if (a0 && a1) bothOnPlate++;
            if (arena.Success) { success = true; break; }   // H2: break AT the latch; record below
        }

        int steps = arena.StepsThisEpisode;                  // == ticks run (success step, or MaxSteps at timeout)
        int holder = arena.LastHolderIndex;                  // arena latch (grounded + XZ), NOT the harness counter
        int scorer = arena.LastScorerIndex;
        float denom = Mathf.Max(steps, 1);
        sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
            "{0},{1},{2},{3},{4},{5},{6:F4},{7},{8},{9:F4}",
            arm, seed, lesson, ep, success ? 1 : 0, steps,
            plateHold / denom, holder, scorer, bothOnPlate / denom));
    }

    private static bool OnPlateXZ(Vector3 pos, Vector3 plate) =>
        new Vector2(pos.x - plate.x, pos.z - plate.z).magnitude <= PlateRadius;

    // The per-step seam: manual academy+physics step, THEN the explicit arena Tick (Script mode = no FixedUpdate).
    private static void StepTick(CoopArena arena)
    {
        Step();
        arena.Tick(Time.fixedDeltaTime);
    }

    private static void Step()
    {
        Academy.Instance.EnvironmentStep();
        Physics.Simulate(Time.fixedDeltaTime);
    }

    private static bool TupleEq((Vector3 g, Vector3 p, Vector3 s0, Vector3 s1) a,
                                (Vector3 g, Vector3 p, Vector3 s0, Vector3 s1) b) =>
        Approximately(a.g, b.g) && Approximately(a.p, b.p) && Approximately(a.s0, b.s0) && Approximately(a.s1, b.s1);

    private static bool Approximately(Vector3 a, Vector3 b) => (a - b).sqrMagnitude < 1e-6f;

    private static string CsvPath(string name) =>
        Path.GetFullPath(Path.Combine(Application.dataPath, "../../training/eval/" + name));

    private static void WriteCsv(StringBuilder sb, string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllText(path, sb.ToString());
        Debug.Log("[M7Eval] wrote -> " + path);
    }
}
