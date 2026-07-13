using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;

namespace NavSim.Runtime
{
    // Self-contained multi-agent arena on the XZ plane. Owns up to NumColors agents and one goal per
    // color (slot index == color). Agents have a FINITE MaxStep and self-place in OnEpisodeBegin; within
    // an episode they respawn goals continuously (no EndEpisode on reach). A single collapsed `difficulty`
    // curriculum co-varies agent count + arena size + obstacle count at train time (DifficultyMapper);
    // the demo/eval drive those axes directly via independent setters. Plain PPO — no SimpleMultiAgentGroup.
    public class NavEnvironment : MonoBehaviour
    {
        public const int NumColors = 8;

        [Header("Wiring (assign in Editor: index = color; agents start INACTIVE in the scene)")]
        [SerializeField] private NavAgent[] agents = new NavAgent[NumColors];
        [SerializeField] private Transform[] goals = new Transform[NumColors];

        [Header("Arena")]
        [SerializeField] private float arenaHalfSize = 8f;
        [SerializeField] private float goalRadius = 1.3f;
        [SerializeField] private float obstacleClearance = 1.9f;
        [SerializeField] private float agentClearance = 1.6f;

        [Header("Arena bounds (assign in Editor)")]
        [SerializeField] private Transform wallNorth;  // +Z
        [SerializeField] private Transform wallSouth;  // -Z
        [SerializeField] private Transform wallEast;   // +X
        [SerializeField] private Transform wallWest;   // -X
        [SerializeField] private Transform floor;
        [SerializeField] private float wallThickness = 0.5f;
        [SerializeField] private float wallHeight = 1.5f;

        [Header("Obstacle pool (assign Obstacle_0..7; index-agnostic)")]
        [SerializeField] private Transform[] obstaclePool = new Transform[NumColors];
        [SerializeField] private float obstacleObstacleClearance = 2.2f;
        [SerializeField] private int layoutEpochSteps = 1500; // train-time obstacle-refresh cadence

        // Maintained incrementally (add BEFORE SetActive) so an activating agent's OnEpisodeBegin sees
        // the peers already placed this round. Iterated read-only during a step (Unity is single-threaded).
        private readonly HashSet<NavAgent> _active = new HashSet<NavAgent>();
        private int _appliedCount = -1;
        private int _appliedDifficulty = -1;

        private int _obstacleTarget;                 // how many pool members to activate this layout
        private int _minObstacles, _maxObstacles;    // per-lesson obstacle-count band
        private int _epochStep;                      // arena-level step counter for the layout epoch

        public int GoalsReachedTotal { get; private set; }
        public void NotifyGoalReached() => GoalsReachedTotal++;
        public int PlacementFallbacks { get; private set; } // ClearPoint retry-exhaustions (eval honesty)

        public float ArenaDiagonal => arenaHalfSize * 2f * Mathf.Sqrt(2f);

        private void Start()
        {
            PinRayLength();
            SetDifficulty(ReadDifficulty()); // no communicator -> default (hardest) level; demo overrides
        }

        // Pin every agent's ray length to the LARGEST lesson's diagonal, once. Keeps goals visible at all
        // arena sizes (the visible-goal invariant). A shorter ray would hide goals at big arenas (=M4).
        // Belt-and-suspenders with the scene value (32); the scene value is primary since a sensor built by
        // Agent.LazyInitialize before Start would ignore a later assignment.
        private void PinRayLength()
        {
            foreach (var a in agents)
            {
                if (a == null) continue;
                var sensor = a.GetComponent<Unity.MLAgents.Sensors.RayPerceptionSensorComponent3D>();
                if (sensor != null && sensor.RayLength < DifficultyMapper.MaxArenaDiagonal)
                    sensor.RayLength = DifficultyMapper.MaxArenaDiagonal;
            }
        }

        // Reposition the 4 boundary walls to +/-halfSize and rescale the floor. Called on lesson change
        // (training) and by the demo/eval setters. RandomPoint() reads arenaHalfSize, so spawning follows.
        public void SetArenaSize(float halfSize)
        {
            arenaHalfSize = halfSize;
            float span = halfSize * 2f + wallThickness;
            if (wallNorth) { wallNorth.position = new Vector3(0f, wallHeight * 0.5f, halfSize);
                             wallNorth.localScale = new Vector3(span, wallHeight, wallThickness); }
            if (wallSouth) { wallSouth.position = new Vector3(0f, wallHeight * 0.5f, -halfSize);
                             wallSouth.localScale = new Vector3(span, wallHeight, wallThickness); }
            if (wallEast)  { wallEast.position  = new Vector3(halfSize, wallHeight * 0.5f, 0f);
                             wallEast.localScale  = new Vector3(wallThickness, wallHeight, span); }
            if (wallWest)  { wallWest.position  = new Vector3(-halfSize, wallHeight * 0.5f, 0f);
                             wallWest.localScale  = new Vector3(wallThickness, wallHeight, span); }
            if (floor) floor.localScale = new Vector3(halfSize * 2f / 10f, 1f, halfSize * 2f / 10f);
            ClampActiveIntoBounds(); // demo/eval shrink the arena WITHOUT ResetAllActive — re-home OOB actors
        }

        // Re-place any active agent (or its stranded goal) now outside the new walls. Protects the demo
        // (difficulty slider dragged down, L3->L0) and the eval harness — both call SetArenaSize directly,
        // with no ResetAllActive to re-place things. Training's SetDifficulty also calls ResetAllActive, so
        // this is belt-and-suspenders there. Iterating _active is safe (neither call mutates it).
        private void ClampActiveIntoBounds()
        {
            foreach (var a in _active)
            {
                bool agentOob = Mathf.Abs(a.transform.position.x) > arenaHalfSize ||
                                Mathf.Abs(a.transform.position.z) > arenaHalfSize;
                if (agentOob) { PlaceForNewEpisode(a); continue; } // re-places agent + its goal
                Transform g = goals[a.Color];
                if (g != null && (Mathf.Abs(g.position.x) > arenaHalfSize || Mathf.Abs(g.position.z) > arenaHalfSize))
                    RespawnGoal(a); // goal stranded though the agent is in-bounds
            }
        }

        private void FixedUpdate()
        {
            // Only the trainer (communicator on) drives difficulty via environment parameters. In a
            // no-communicator build (WebGL/standalone) GetWithDefault always returns the default, so
            // polling would revert the demo setters every step — gate it so the sliders latch.
            if (!Academy.Instance.IsCommunicatorOn) return;
            int level = ReadDifficulty();
            if (level != _appliedDifficulty) { SetDifficulty(level); return; } // curriculum lesson advance
            // Periodic fresh obstacle layout WITHIN a lesson -> thousands of layouts, no memorization.
            if (++_epochStep >= layoutEpochSteps) { _epochStep = 0; RollObstacleCountAndRegenerate(); }
        }

        private int ReadDifficulty() => Mathf.Clamp(
            Mathf.RoundToInt(Academy.Instance.EnvironmentParameters.GetWithDefault(
                "difficulty", (float)(DifficultyMapper.NumLevels - 1))),
            0, DifficultyMapper.NumLevels - 1);

        // WebGL/standalone entry: no communicator, so EnvironmentParameters returns the default —
        // the demo slider drives the count directly through this.
        public void SetActiveCount(int k) => ApplyCount(Mathf.Clamp(k, 2, NumColors));

        // Apply a whole difficulty rung: resize, set count (delta), set obstacle band, regenerate layout,
        // and reset all active agents so any left outside a shrunk arena re-place inside the new bounds.
        public void SetDifficulty(int level)
        {
            var d = DifficultyMapper.ForLevel(level);
            SetArenaSize(d.ArenaHalfSize);
            _minObstacles = d.MinObstacles; _maxObstacles = d.MaxObstacles;
            ApplyCount(d.AgentCount);
            RollObstacleCountAndRegenerate();
            ResetAllActive();
            _appliedDifficulty = level;
        }

        private void RollObstacleCountAndRegenerate()
        {
            SetObstacleCount(Random.Range(_minObstacles, _maxObstacles + 1)); // int overload: max-exclusive
            RegenerateLayout();
        }

        private void ResetAllActive()
        {
            // EndEpisode on an active agent is a clean trajectory boundary under plain PPO; OnEpisodeBegin
            // re-places it (IsActive guard already true). Snapshot to avoid mutating _active mid-iterate.
            var snapshot = new List<NavAgent>(_active);
            foreach (var a in snapshot) a.EndEpisode();
        }

        public void SetObstacleCount(int k) => _obstacleTarget = Mathf.Clamp(k, 0, obstaclePool.Length);

        // Activate _obstacleTarget pool members at points clear of ACTIVE agents and each other, deactivate
        // the rest. Runs on a periodic epoch (train) or the demo button — never teleports onto a live agent
        // because ClearPointForObstacle avoids active agents. Goals inside a new pillar are respawned.
        public void RegenerateLayout()
        {
            int placed = 0;
            for (int i = 0; i < obstaclePool.Length; i++)
            {
                var o = obstaclePool[i];
                if (o == null) continue;
                if (placed < _obstacleTarget)
                {
                    Vector3 p = ClearPointForObstacle(placed);
                    o.position = new Vector3(p.x, o.position.y, p.z);
                    o.gameObject.SetActive(true);
                    placed++;
                }
                else o.gameObject.SetActive(false);
            }
            RevalidateGoals();
        }

        // Clear of active agents and of the already-placed obstacles this round.
        private Vector3 ClearPointForObstacle(int placedSoFar)
        {
            for (int t = 0; t < 60; t++)
            {
                Vector3 p = RandomPoint();
                if (!ClearOfAgentsAny(p)) continue;
                bool ok = true;
                for (int j = 0; j < placedSoFar; j++)
                    if (obstaclePool[j] != null && obstaclePool[j].gameObject.activeSelf &&
                        HorizontalDistance(p, obstaclePool[j].position) < obstacleObstacleClearance) { ok = false; break; }
                if (ok) return p;
            }
            PlacementFallbacks++; // no clear spot in 60 tries -> placed with possible overlap (eval surfaces it)
            return RandomPoint();
        }

        private bool ClearOfAgentsAny(Vector3 p)
        {
            foreach (var a in _active)
                if (HorizontalDistance(p, a.transform.position) < obstacleClearance) return false;
            return true;
        }

        // Any active goal now inside an obstacle gets a fresh clear point.
        private void RevalidateGoals()
        {
            foreach (var a in _active)
            {
                Transform g = goals[a.Color];
                if (g == null || !g.gameObject.activeSelf) continue;
                if (!ClearOfObstacles(g.position)) RespawnGoal(a);
            }
        }

        // Activate/deactivate only the DELTA. A newly-active agent's SetActive(true) fires its
        // OnEpisodeBegin (self-place); a retiring agent gets EndEpisode() (clean trajectory under plain
        // PPO) then SetActive(false). Agents that stay active are untouched (keep running their episode).
        private void ApplyCount(int k)
        {
            for (int i = 0; i < agents.Length; i++)
            {
                if (agents[i] == null || goals[i] == null) continue; // both must be wired to participate
                bool on = i < k;
                bool was = _active.Contains(agents[i]);
                if (on && !was)
                {
                    agents[i].SetColor(i);                 // color index == wiring slot
                    _active.Add(agents[i]);
                    if (goals[i] != null) goals[i].gameObject.SetActive(true);
                    agents[i].gameObject.SetActive(true);  // mid-run re-activation self-places via a
                                                           // synchronous OnEpisodeBegin; at Start() it
                                                           // places on Academy's first agent reset
                }
                else if (!on && was)
                {
                    _active.Remove(agents[i]);             // remove BEFORE EndEpisode so the IsActive
                    agents[i].EndEpisode();                // guard skips placement, THEN disable
                    agents[i].gameObject.SetActive(false);
                    if (goals[i] != null) goals[i].gameObject.SetActive(false);
                }
            }
            _appliedCount = k;
        }

        public bool IsActive(NavAgent a) => _active.Contains(a);
        public Vector3 GoalPositionFor(NavAgent a) => goals[a.Color].position;

        public bool ReachedGoal(NavAgent a) =>
            HorizontalDistance(a.transform.position, GoalPositionFor(a)) < goalRadius;

        // Fresh same-color goal, clear of obstacles/agents/other goals, far enough from the agent.
        public void RespawnGoal(NavAgent a)
        {
            Transform g = goals[a.Color];
            Vector3 p; int guard = 0;
            do { p = ClearPoint(a); guard++; }
            while (HorizontalDistance(p, a.transform.position) < arenaHalfSize && guard < 60);
            g.position = new Vector3(p.x, g.position.y, p.z);
            a.NotifyGoalMoved(); // re-sync the agent's shaping baseline (_prevDist) — advisor Finding A
        }

        // Place one agent at a clear point + give it a fresh far goal. Called from OnEpisodeBegin.
        public void PlaceForNewEpisode(NavAgent a)
        {
            Vector3 ap = ClearPoint(a);
            a.transform.position = new Vector3(ap.x, a.transform.position.y, ap.z);
            a.transform.rotation = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);
            RespawnGoal(a);
        }

        public IReadOnlyList<Vector3> PeerPositions(NavAgent exclude)
        {
            var list = new List<Vector3>(_active.Count);
            foreach (var a in _active)
                if (a != exclude) list.Add(a.transform.position);
            return list;
        }

        // Random point clear of obstacles, other active agents, and other active goals.
        private Vector3 ClearPoint(NavAgent forAgent)
        {
            for (int t = 0; t < 60; t++)
            {
                Vector3 p = RandomPoint();
                if (!ClearOfObstacles(p)) continue;
                if (!ClearOfAgents(p, forAgent)) continue;
                if (!ClearOfGoals(p, forAgent)) continue;
                return p;
            }
            PlacementFallbacks++; // agent/goal placement exhausted its retries (eval surfaces it)
            return RandomPoint();
        }

        private bool ClearOfObstacles(Vector3 p)
        {
            if (obstaclePool == null) return true;
            for (int i = 0; i < obstaclePool.Length; i++)
            {
                var o = obstaclePool[i];
                if (o == null || !o.gameObject.activeSelf) continue;
                if (HorizontalDistance(p, o.position) < obstacleClearance) return false;
            }
            return true;
        }

        private bool ClearOfAgents(Vector3 p, NavAgent forAgent)
        {
            foreach (var a in _active)
            {
                if (a == forAgent) continue;
                if (HorizontalDistance(p, a.transform.position) < agentClearance) return false;
            }
            return true;
        }

        private bool ClearOfGoals(Vector3 p, NavAgent forAgent)
        {
            int myColor = forAgent.Color;
            foreach (var a in _active)
            {
                int c = a.Color;
                if (c == myColor) continue;
                if (goals[c] != null && HorizontalDistance(p, goals[c].position) < agentClearance) return false;
            }
            return true;
        }

        private static float HorizontalDistance(Vector3 a, Vector3 b)
        {
            a.y = 0f; b.y = 0f;
            return Vector3.Distance(a, b);
        }

        private Vector3 RandomPoint() => new Vector3(
            Random.Range(-arenaHalfSize, arenaHalfSize), 0f, Random.Range(-arenaHalfSize, arenaHalfSize));
    }
}
