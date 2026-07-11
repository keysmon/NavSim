using UnityEngine;

namespace NavSim.Runtime
{
    // Owns one 3D arena on the XZ plane: places the agent and goal, detects reaching the goal.
    public class NavEnvironment : MonoBehaviour
    {
        [SerializeField] private Transform goal;
        [SerializeField] private float arenaHalfSize = 8f;
        [SerializeField] private float goalRadius = 0.8f;

        public float ArenaDiagonal => arenaHalfSize * 2f * Mathf.Sqrt(2f);
        public Vector3 GoalPosition => goal.position;

        public void ResetEpisode(Transform agent)
        {
            Vector3 a = RandomPoint();
            agent.position = new Vector3(a.x, agent.position.y, a.z);
            agent.rotation = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);

            Vector3 g;
            do { g = RandomPoint(); } while (HorizontalDistance(g, agent.position) < arenaHalfSize);
            goal.position = new Vector3(g.x, goal.position.y, g.z);
        }

        public bool ReachedGoal(Vector3 agentPos) =>
            HorizontalDistance(agentPos, GoalPosition) < goalRadius;

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
