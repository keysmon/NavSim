using UnityEngine;

namespace NavSim.Runtime
{
    // Owns one arena: places the agent and goal, detects reaching the goal.
    public class NavEnvironment : MonoBehaviour
    {
        [SerializeField] private Transform goal;
        [SerializeField] private float arenaHalfSize = 8f;
        [SerializeField] private float goalRadius = 0.6f;

        public float ArenaDiagonal => arenaHalfSize * 2f * Mathf.Sqrt(2f);
        public Vector2 GoalPosition => goal.position;

        public void ResetEpisode(Transform agent)
        {
            agent.position = RandomPoint();
            agent.rotation = Quaternion.Euler(0f, 0f, Random.Range(0f, 360f));
            Vector2 g;
            do { g = RandomPoint(); } while (Vector2.Distance(g, agent.position) < arenaHalfSize);
            goal.position = g;
        }

        public bool ReachedGoal(Vector2 agentPos) =>
            Vector2.Distance(agentPos, GoalPosition) < goalRadius;

        private Vector2 RandomPoint() => new Vector2(
            Random.Range(-arenaHalfSize, arenaHalfSize),
            Random.Range(-arenaHalfSize, arenaHalfSize));
    }
}
