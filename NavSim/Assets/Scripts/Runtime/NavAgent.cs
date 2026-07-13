using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

namespace NavSim.Runtime
{
    // Shared-policy crowd member. Moves on XZ, turns about Y (unicycle). Continuous respawn: on reaching
    // its color goal it gets a fresh same-color goal (no EndEpisode). The arena owns count/colors/resets.
    [RequireComponent(typeof(Rigidbody))]
    public class NavAgent : Agent
    {
        [SerializeField] private NavEnvironment env;
        [SerializeField] private float maxSpeed = 4f;
        [SerializeField] private float maxTurnDegPerStep = 6f;
        [SerializeField] private RewardConfig reward = RewardConfig.Default;

        private Rigidbody _rb;
        private float _prevDist;
        private int _color;

        public override void Initialize() => _rb = GetComponent<Rigidbody>();

        public void SetColor(int color) => _color = color;
        public int Color => _color;

        // Re-sync the shaping baseline after the ARENA relocates this agent's goal (layout epoch, goal
        // revalidation, arena-shrink re-home). Idempotent with the reached-block and OnEpisodeBegin, which
        // both re-set _prevDist immediately afterward on their own paths. Advisor Finding A.
        public void NotifyGoalMoved() =>
            _prevDist = Vector3.Distance(transform.position, env.GoalPositionFor(this));

        public override void OnEpisodeBegin()
        {
            // Fires on MaxStep interruption, on fresh activation, and when the controller retires this
            // agent. Skip placement in the retire case (controller removed us from the active set first).
            if (!env.IsActive(this)) return;
            env.PlaceForNewEpisode(this); // arena places agent + its goal clear of the live crowd
            _rb.linearVelocity = Vector3.zero;
            _rb.angularVelocity = Vector3.zero;
            _prevDist = Vector3.Distance(transform.position, env.GoalPositionFor(this));
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            float heading = transform.eulerAngles.y;
            float[] obs = ObservationBuilder.Build(
                _rb.linearVelocity, heading, maxSpeed, _color, NavEnvironment.NumColors);
            foreach (float o in obs) sensor.AddObservation(o);
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            float forward = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
            float turn = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);

            transform.Rotate(0f, turn * maxTurnDegPerStep, 0f);
            Vector3 vel = transform.forward * forward * maxSpeed;
            vel.y = 0f;
            _rb.linearVelocity = vel;

            Vector3 pos = transform.position;
            Vector3 goal = env.GoalPositionFor(this);
            float dist = Vector3.Distance(pos, goal);
            bool reached = env.ReachedGoal(this);

            // Goal-directed reward (visibility-gated shaping) minus crowd penalty (sub-dominant, M4 §3/M2 §6b).
            float step = RewardCalculator.Step(_prevDist, dist, reached, reward, env.VisibilityRadius);
            var neighborDist = CrowdMath.NeighborDistances(
                pos, env.PeerPositions(this), reward.congestionRadius);
            float crowd = RewardCalculator.CrowdPenalty(neighborDist, reward);
            AddReward(step - crowd);

            if (reached)
            {
                // Continuous respawn: fresh same-color goal, NO EndEpisode.
                env.NotifyGoalReached();
                env.RespawnGoal(this);
                dist = Vector3.Distance(pos, env.GoalPositionFor(this));
            }
            _prevDist = dist;
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var ca = actionsOut.ContinuousActions;
            ca[0] = Input.GetAxis("Vertical");   // forward
            ca[1] = Input.GetAxis("Horizontal"); // turn
        }
    }
}
