using UnityEngine;

namespace NavSim.Runtime
{
    // Owns one 3D arena on the XZ plane: places the agent and goal (clear of obstacles),
    // and detects reaching the goal.
    public class NavEnvironment : MonoBehaviour
    {
        [SerializeField] private Transform goal;
        [SerializeField] private float arenaHalfSize = 8f;
        [SerializeField] private float goalRadius = 1.3f;
        [SerializeField] private float obstacleClearance = 1.9f;

        private Transform[] _obstacles;

        private void Awake()
        {
            var objs = GameObject.FindGameObjectsWithTag("obstacle");
            _obstacles = new Transform[objs.Length];
            for (int i = 0; i < objs.Length; i++) _obstacles[i] = objs[i].transform;
        }

        public float ArenaDiagonal => arenaHalfSize * 2f * Mathf.Sqrt(2f);
        public Vector3 GoalPosition => goal.position;

        public void ResetEpisode(Transform agent)
        {
            Vector3 a = ClearPoint();
            agent.position = new Vector3(a.x, agent.position.y, a.z);
            agent.rotation = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);

            Vector3 g;
            int guard = 0;
            do { g = ClearPoint(); guard++; }
            while (HorizontalDistance(g, agent.position) < arenaHalfSize && guard < 60);
            goal.position = new Vector3(g.x, goal.position.y, g.z);
        }

        public bool ReachedGoal(Vector3 agentPos) =>
            HorizontalDistance(agentPos, GoalPosition) < goalRadius;

        // A random point at least obstacleClearance away from every obstacle.
        private Vector3 ClearPoint()
        {
            for (int t = 0; t < 60; t++)
            {
                Vector3 p = RandomPoint();
                bool clear = true;
                if (_obstacles != null)
                {
                    for (int i = 0; i < _obstacles.Length; i++)
                    {
                        if (_obstacles[i] == null) continue;
                        if (HorizontalDistance(p, _obstacles[i].position) < obstacleClearance) { clear = false; break; }
                    }
                }
                if (clear) return p;
            }
            return RandomPoint();
        }

        private static float HorizontalDistance(Vector3 a, Vector3 b)
        {
            a.y = 0f; b.y = 0f;
            return Vector3.Distance(a, b);
        }

        private Vector3 RandomPoint() => new Vector3(
            Random.Range(-arenaHalfSize, arenaHalfSize),
            0f,
            Random.Range(-arenaHalfSize, arenaHalfSize));
    }
}
