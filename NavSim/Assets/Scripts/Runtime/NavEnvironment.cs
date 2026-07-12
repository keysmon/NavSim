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

        private void Start() => ApplyCount(ReadCeiling());

        private void FixedUpdate()
        {
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
                if (agents[i] == null) continue;
                bool on = i < k;
                bool was = _active.Contains(agents[i]);
                if (on && !was)
                {
                    agents[i].SetColor(i);                 // color index == wiring slot
                    _active.Add(agents[i]);
                    if (goals[i] != null) goals[i].gameObject.SetActive(true);
                    agents[i].gameObject.SetActive(true);  // OnEnable -> OnEpisodeBegin -> self-place
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
