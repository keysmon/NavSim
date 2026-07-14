using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

namespace NavSim.Runtime
{
    // Oblivious dynamic occluders (spec §6). Each active mover is a NavMeshAgent that patrols random
    // reachable points, repathing on arrival. It has NO awareness of the learner -> a pure moving obstacle
    // that occludes the forward ray fan and forces the policy to route around a shifting crowd. Movers ride
    // the same baked NavMesh as the SPL oracle, so they climb ramps and never enter pits (no mesh over a hole).
    public class MoverController : MonoBehaviour
    {
        [SerializeField] private NavMeshAgent[] pool = new NavMeshAgent[4];
        [SerializeField] private float arenaHalf = DifficultyMapper.M5ArenaHalf;
        [SerializeField] private float arrivalThreshold = 0.5f;

        private readonly List<NavMeshAgent> _active = new List<NavMeshAgent>();

        // Activate the first k pool members (each gets a fresh patrol target), deactivate the rest.
        // MUST be called AFTER the terrain NavMesh is baked so re-activated agents land on a live mesh.
        public void SetCount(int k)
        {
            if (pool == null) return;
            k = Mathf.Clamp(k, 0, pool.Length);
            for (int i = 0; i < pool.Length; i++)
            {
                if (pool[i] == null) continue;
                bool on = i < k;
                pool[i].gameObject.SetActive(on);
                if (on)
                {
                    if (!_active.Contains(pool[i])) _active.Add(pool[i]);
                    Repath(pool[i]);
                }
                else
                {
                    _active.Remove(pool[i]);
                }
            }
        }

        private void Update()
        {
            for (int i = 0; i < _active.Count; i++)
            {
                var m = _active[i];
                if (m == null || !m.isOnNavMesh) continue;
                // Reached the current waypoint, OR never got a path (a Repath during SetCount can no-op if the
                // agent was momentarily off-mesh right after activation; remainingDistance is then +Infinity, so
                // guard on !hasPath too) -> pick a new reachable point.
                if (!m.pathPending && (!m.hasPath || m.remainingDistance < arrivalThreshold)) Repath(m);
            }
        }

        // Sample a random walkable point and send the mover there. Retries a few times so a sample that
        // lands over a pit/off-mesh does not leave the mover stranded without a destination.
        private void Repath(NavMeshAgent m)
        {
            if (m == null || !m.isOnNavMesh) return;
            for (int t = 0; t < 8; t++)
            {
                Vector3 p = new Vector3(Random.Range(-arenaHalf, arenaHalf), 0f, Random.Range(-arenaHalf, arenaHalf));
                if (NavMesh.SamplePosition(p, out var hit, 4f, NavMesh.AllAreas))
                {
                    m.SetDestination(hit.position);
                    return;
                }
            }
        }

        // Current world positions of the active movers — fed to the learner's crowd penalty (CrowdMath).
        public IReadOnlyList<Vector3> Positions()
        {
            var list = new List<Vector3>(_active.Count);
            for (int i = 0; i < _active.Count; i++)
                if (_active[i] != null) list.Add(_active[i].transform.position);
            return list;
        }
    }
}
