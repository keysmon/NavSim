using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;
using Unity.MLAgents;

namespace NavSim.Runtime
{
    // Single-learner 3D terrain-search arena (M5). One CharacterController learner navigates a procedurally
    // generated arena (ramps/platforms/walls/pits) to a hidden goal gated by a line-of-sight "flashlight".
    // A collapsed `difficulty` curriculum drives the terrain ladder (DifficultyMapper.ForTerrainLevel): higher
    // rungs add walls, elevate the goal onto ramp-reachable platforms, dig pits, and add oblivious NavMeshAgent
    // movers. The runtime-baked NavMesh (TerrainGenerator) is the solvability gate + SPL oracle + mover substrate.
    // Plain PPO: one Agent, finite MaxStep, continuous goal respawn within an episode (no EndEpisode on reach).
    public class NavEnvironment : MonoBehaviour
    {
        [Header("Wiring (assign in Editor)")]
        [SerializeField] private NavAgent agent;
        [SerializeField] private Transform goal;
        [SerializeField] private TerrainGenerator terrain;
        [SerializeField] private MoverController movers;

        [Header("Arena")]
        [SerializeField] private float arenaHalf = DifficultyMapper.M5ArenaHalf;
        [Tooltip("3-D reach radius. Elevated goals need vertical slack (capsule centre sits above the platform).")]
        [SerializeField] private float goalRadius = 1.5f;

        [Header("Line of sight")]
        [Tooltip("Walls/platforms/ramps only (NOT movers) — LoS is occluded by STATIC geometry, not transient movers.")]
        [SerializeField] private LayerMask staticGeometryMask;
        [SerializeField] private float eyeHeight = 1.6f;

        [Header("Pit fall")]
        [SerializeField] private float killPlaneY = -3f;   // fall below this Y == pit fall
        [SerializeField] private float respawnSampleRadius = 6f;

        public float KillPlaneY => killPlaneY;
        public int GoalsReached { get; private set; }

        private int _appliedLevel = -1;

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

        private int ReadDifficulty() => Mathf.Clamp(
            Mathf.RoundToInt(Academy.Instance.EnvironmentParameters.GetWithDefault(
                "difficulty", (float)(DifficultyMapper.NumLevels - 1))),
            0, DifficultyMapper.NumLevels - 1);

        // Regenerate the layout for the current curriculum level. Called from NavAgent.OnEpisodeBegin, so a
        // fresh solvable arena is drawn every episode -> thousands of layouts, no memorisation.
        public void PlaceForNewEpisode(NavAgent a) => SetTerrainLevel(ReadDifficulty());

        // Build a fresh solvable terrain layout: bake structure, then place a walkable spawn + a (possibly
        // elevated) reachable, far-enough goal, retrying until the baked NavMesh connects them. Then set the
        // mover count on the live mesh.
        public void SetTerrainLevel(int level)
        {
            if (terrain == null || agent == null || goal == null) return;
            TerrainLevel lvl = DifficultyMapper.ForTerrainLevel(level);
            for (int t = 0; t < 20; t++)
            {
                terrain.Generate(lvl, Vector3.zero, Vector3.zero); // bake structure; real spawn/goal chosen below
                if (!SampleGround(RandomXZ(), out Vector3 spawn)) continue;
                if (!TryPickGoal(lvl, spawn, arenaHalf * 0.5f, out Vector3 g)) continue; // far goal for a real search
                PlaceAt(agent, spawn);
                goal.position = g;
                agent.NotifyGoalMoved();
                if (movers != null) movers.SetCount(lvl.Movers);
                _appliedLevel = level;
                return;
            }
            // Retries exhausted (rare): place best-effort AND give a reachable ground goal (dropping the
            // far-distance constraint) so the episode never runs with a stale/unreachable goal or a wrong
            // _prevDist baseline. NotifyGoalMoved re-syncs the shaping baseline whether or not a goal was found.
            if (SampleGround(RandomXZ(), out Vector3 fallback) || SampleGround(Vector3.zero, out fallback))
            {
                PlaceAt(agent, fallback);
                for (int t = 0; t < 20; t++)
                    if (SampleGround(RandomXZ(), out Vector3 fg) && terrain.IsReachable(fallback, fg))
                    { goal.position = fg; break; }
            }
            agent.NotifyGoalMoved();
            if (movers != null) movers.SetCount(lvl.Movers);
            _appliedLevel = level;
        }

        // A reachable goal at least `minDist` from `from`: platform-top for elevated levels, else walkable
        // ground. False if none found this attempt (caller re-rolls).
        private bool TryPickGoal(TerrainLevel lvl, Vector3 from, float minDist, out Vector3 g)
        {
            g = Vector3.zero;
            if (lvl.GoalElevated) { if (!terrain.RandomPlatformTop(out g)) return false; }
            else if (!SampleGround(RandomXZ(), out g)) return false;
            if (Vector3.Distance(from, g) < minDist) return false;
            return terrain.IsReachable(from, g);
        }

        public Vector3 GoalPositionFor(NavAgent a) => goal.position;

        // 3-D reach: reached only within goalRadius in ALL axes, so an elevated goal requires being ON the
        // platform, not on the ground beneath it. Shares the 3-D metric with the distance shaping
        // (RewardCalculator) so reward and success never disagree.
        public bool ReachedGoal(NavAgent a) => Vector3.Distance(a.transform.position, goal.position) < goalRadius;

        // Continuous respawn on reach: a fresh reachable goal on the SAME baked terrain (no EndEpisode).
        public void RespawnGoal(NavAgent a)
        {
            int level = _appliedLevel < 0 ? 0 : _appliedLevel;
            TerrainLevel lvl = DifficultyMapper.ForTerrainLevel(level);
            // Elevated levels use a SMALLER min-distance so a same-platform re-target still qualifies (esp. L2's
            // single platform, whose jittered top points span ~3u) instead of collapsing to the ground fallback.
            float minDist = lvl.GoalElevated ? 2f : arenaHalf * 0.5f;
            for (int t = 0; t < 20; t++)
                if (TryPickGoal(lvl, a.transform.position, minDist, out Vector3 g)) { goal.position = g; a.NotifyGoalMoved(); return; }
            // Robustness: if no (elevated) goal was found, fall back to ANY reachable ground point.
            for (int t = 0; t < 20; t++)
                if (SampleGround(RandomXZ(), out Vector3 g) && terrain.IsReachable(a.transform.position, g))
                { goal.position = g; a.NotifyGoalMoved(); return; }
            // Last resort (design-unreachable): snap the goal near the arena centre so it is never left stale.
            if (SampleGround(Vector3.zero, out Vector3 c)) { goal.position = c; a.NotifyGoalMoved(); }
        }

        public void NotifyGoalReached() => GoalsReached++;

        // LoS gate: goal within fixed sight AND no STATIC geometry between the agent's eye and the goal.
        // Movers are deliberately excluded (staticGeometryMask) so the visibility nudge doesn't flicker as
        // they pass (spec §8). With the mask unset (Nothing), Linecast never blocks -> range-only.
        public bool GoalVisibleTo(NavAgent a)
        {
            Vector3 eye = a.transform.position + Vector3.up * eyeHeight;
            float dist = Vector3.Distance(eye, goal.position);
            bool blocked = Physics.Linecast(eye, goal.position, staticGeometryMask);
            return VisibilityGate.IsGoalVisible(dist, DifficultyMapper.SightRange, lineOfSightClear: !blocked);
        }

        public IReadOnlyList<Vector3> MoverPositions() =>
            movers != null ? movers.Positions() : System.Array.Empty<Vector3>();

        // Pit-fall recovery: sample the nearest walkable ground to where the agent fell and teleport it there,
        // keeping episode + LSTM memory. Never a shortcut toward the goal — re-homes near the fall site.
        public void RespawnToSafeGround(NavAgent a)
        {
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
