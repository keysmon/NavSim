using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

namespace NavSim.Runtime
{
    [RequireComponent(typeof(Rigidbody2D))]
    public class NavAgent : Agent
    {
        [SerializeField] private NavEnvironment env;
        [SerializeField] private float maxSpeed = 4f;
        [SerializeField] private float maxTurnDegPerStep = 6f;
        [SerializeField] private RewardConfig reward = RewardConfig.Default;

        private Rigidbody2D _rb;
        private float _prevDist;

        public override void Initialize() => _rb = GetComponent<Rigidbody2D>();

        public override void OnEpisodeBegin()
        {
            env.ResetEpisode(transform);
            _rb.linearVelocity = Vector2.zero;
            _prevDist = Vector2.Distance(transform.position, env.GoalPosition);
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            float heading = transform.eulerAngles.z;
            float[] obs = ObservationBuilder.Build(
                transform.position, heading, _rb.linearVelocity,
                env.GoalPosition, maxSpeed, env.ArenaDiagonal, reward.compassWeight);
            foreach (float o in obs) sensor.AddObservation(o);
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            float forward = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
            float turn = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);

            transform.Rotate(0f, 0f, turn * maxTurnDegPerStep);
            _rb.linearVelocity = (Vector2)transform.right * forward * maxSpeed;

            Vector2 pos = transform.position;
            float dist = Vector2.Distance(pos, env.GoalPosition);
            bool reached = env.ReachedGoal(pos);

            AddReward(RewardCalculator.Step(_prevDist, dist, reached, reward));
            _prevDist = dist;

            if (reached) EndEpisode();
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var ca = actionsOut.ContinuousActions;
            ca[0] = Input.GetAxis("Vertical");    // forward
            ca[1] = -Input.GetAxis("Horizontal"); // turn
        }
    }
}
