using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;

namespace NavSim.Runtime
{
    // Self-contained multi-agent arena on the XZ plane. Owns up to NumColors agents and one goal per
    // color (slot index == color). Agents have a FINITE MaxStep and self-place in OnEpisodeBegin; within
    // an episode they respawn goals continuously (no EndEpisode on reach). Active count is curriculum-
    // driven (num_agents) at train time and slider-driven in WebGL; this controller activates/deactivates
    // only the DELTA when the count changes. Plain PPO — no SimpleMultiAgentGroup.
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

        private Transform[] _obstacles;
        // Maintained incrementally (add BEFORE SetActive) so an activating agent's OnEpisodeBegin sees
        // the peers already placed this round. Iterated read-only during a step (Unity is single-threaded).
        private readonly HashSet<NavAgent> _active = new HashSet<NavAgent>();
        private int _appliedCount = -1;

        public float ArenaDiagonal => arenaHalfSize * 2f * Mathf.Sqrt(2f);

        private void Awake()
        {
            // Per-arena obstacle scoping (NOT global FindGameObjectsWithTag) so arenas can be tiled.
            var kids = GetComponentsInChildren<Transform>(includeInactive: true);
            var obs = new List<Transform>();
            foreach (var t in kids)
                if (t != transform && t.CompareTag("obstacle")) obs.Add(t);
            _obstacles = obs.ToArray();
        }

        private void Start()
        {
            PinRayLength();
            ApplyCount(ReadCeiling());
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
            // Only the trainer (communicator on) drives active count via environment parameters. In a
            // no-communicator build (WebGL/standalone) GetWithDefault always returns the default, so
            // polling would revert SetActiveCount every step — gate it so the demo slider latches.
            if (!Academy.Instance.IsCommunicatorOn) return;
            int k = ReadCeiling();
            if (k != _appliedCount) ApplyCount(k); // catches curriculum lesson advances
        }

        private int ReadCeiling() => Mathf.Clamp(
            Mathf.RoundToInt(Academy.Instance.EnvironmentParameters.GetWithDefault("num_agents", (float)NumColors)),
            2, NumColors);

        // WebGL/standalone entry: no communicator, so EnvironmentParameters returns the default —
        // the demo slider drives the count directly through this.
        public void SetActiveCount(int k) => ApplyCount(Mathf.Clamp(k, 2, NumColors));

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
            return RandomPoint();
        }

        private bool ClearOfObstacles(Vector3 p)
        {
            if (_obstacles == null) return true;
            for (int i = 0; i < _obstacles.Length; i++)
            {
                if (_obstacles[i] == null) continue;
                if (HorizontalDistance(p, _obstacles[i].position) < obstacleClearance) return false;
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
