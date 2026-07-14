using UnityEngine;
using UnityEngine.AI;
using Unity.AI.Navigation;

namespace NavSim.Runtime
{
    // Procedurally activates ramp/platform/wall/pit pool members for a TerrainLevel, bakes a runtime
    // NavMesh (which serves the movers, the SPL oracle, and the solvability gate), and validates that the
    // goal is reachable on foot. The NavMesh describes WALKABLE ground only — jump is an optional shortcut,
    // so every goal must be walk/ramp-reachable and the mesh stays connected (spec §5, no off-mesh links).
    public class TerrainGenerator : MonoBehaviour
    {
        [SerializeField] private NavMeshSurface surface;
        [SerializeField] private Transform[] ramps;
        [SerializeField] private Transform[] platforms;
        [SerializeField] private Transform[] walls;
        [SerializeField] private Transform[] pitTiles;   // floor tiles hidden to "dig" a pit
        [SerializeField] private float arenaHalf = DifficultyMapper.M5ArenaHalf;
        [SerializeField] private float platformJitter = 1.2f; // XZ spread when sampling a platform-top goal

        public bool NavMeshReady { get; private set; }

        // Places the level's structure, bakes the NavMesh, and returns true iff the goal is reachable from
        // spawn on the baked mesh (the solvability gate). The caller retries with a fresh layout on false.
        public bool Generate(TerrainLevel lvl, Vector3 spawn, Vector3 goal)
        {
            ActivatePool(walls, lvl.Walls);
            ActivatePool(platforms, lvl.Platforms);
            // one ramp per platform, plus an entry ramp once walls appear, so elevated goals stay reachable.
            ActivatePool(ramps, Mathf.Max(lvl.Platforms, lvl.Walls > 0 ? 1 : 0));
            HidePits(lvl.Pits);
            if (surface != null) surface.BuildNavMesh();   // runtime bake (com.unity.ai.navigation)
            NavMeshReady = surface != null;
            return IsReachable(spawn, goal);
        }

        // True if a complete walk path exists between the two points on the baked NavMesh.
        public bool IsReachable(Vector3 from, Vector3 to)
        {
            var path = new NavMeshPath();
            if (!NavMesh.SamplePosition(from, out var a, 2f, NavMesh.AllAreas)) return false;
            if (!NavMesh.SamplePosition(to, out var b, 2f, NavMesh.AllAreas)) return false;
            NavMesh.CalculatePath(a.position, b.position, NavMesh.AllAreas, path);
            return path.status == NavMeshPathStatus.PathComplete;
        }

        // Geodesic path length on the baked NavMesh — the SPL denominator (spec §11). -1 if unreachable.
        public float ShortestPathLength(Vector3 from, Vector3 to)
        {
            var path = new NavMeshPath();
            if (!NavMesh.SamplePosition(from, out var a, 2f, NavMesh.AllAreas)) return -1f;
            if (!NavMesh.SamplePosition(to, out var b, 2f, NavMesh.AllAreas)) return -1f;
            if (!NavMesh.CalculatePath(a.position, b.position, NavMesh.AllAreas, path) ||
                path.status != NavMeshPathStatus.PathComplete)
                return -1f;
            float len = 0f;
            for (int i = 1; i < path.corners.Length; i++)
                len += Vector3.Distance(path.corners[i - 1], path.corners[i]);
            return len;
        }

        // A NavMesh-sampled walkable point on top of a random ACTIVE platform, for elevated-goal placement
        // (spec §5, L2/L3). Returns false if no platform is active or none has walkable ground on top, so the
        // caller re-rolls the layout instead of stranding the goal in mid-air.
        public bool RandomPlatformTop(out Vector3 point)
        {
            point = Vector3.zero;
            if (platforms == null) return false;
            var active = new System.Collections.Generic.List<Transform>();
            for (int i = 0; i < platforms.Length; i++)
                if (platforms[i] != null && platforms[i].gameObject.activeSelf) active.Add(platforms[i]);
            if (active.Count == 0) return false;
            for (int t = 0; t < active.Count * 4; t++)
            {
                Transform p = active[Random.Range(0, active.Count)];
                // Jitter within the platform footprint so successive picks vary ACROSS a platform — a single
                // platform (L2) must still yield distinct re-target points for continuous elevated respawn,
                // not just its centre — then sample the walkable top just above.
                Vector3 jitter = new Vector3(Random.Range(-platformJitter, platformJitter), 0f,
                                             Random.Range(-platformJitter, platformJitter));
                Vector3 probe = p.position + jitter + Vector3.up * 1.5f;
                if (NavMesh.SamplePosition(probe, out var hit, 2f, NavMesh.AllAreas)) { point = hit.position; return true; }
            }
            return false;
        }


        // Activate the first k members (repositioned to fresh random ground); deactivate the rest.
        // Activate the first k members (repositioned to a fresh random X/Z; authored Y is PRESERVED so raised
        // platforms/walls/ramps keep sitting on the floor instead of sinking to y=0); deactivate the rest.
        private void ActivatePool(Transform[] pool, int k)
        {
            if (pool == null) return;
            for (int i = 0; i < pool.Length; i++)
            {
                if (pool[i] == null) continue;
                bool on = i < k;
                if (on)
                {
                    Vector3 gp = RandomGroundPoint();
                    pool[i].position = new Vector3(gp.x, pool[i].position.y, gp.z);
                }
                pool[i].gameObject.SetActive(on);
            }
        }

        // "Digging" a pit == hiding a floor tile so the agent falls through it. First k tiles hidden.
        private void HidePits(int k)
        {
            if (pitTiles == null) return;
            for (int i = 0; i < pitTiles.Length; i++)
                if (pitTiles[i] != null) pitTiles[i].gameObject.SetActive(i >= k);
        }

        private Vector3 RandomGroundPoint() =>
            new Vector3(Random.Range(-arenaHalf, arenaHalf), 0f, Random.Range(-arenaHalf, arenaHalf));
    }
}
