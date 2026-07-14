using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

namespace NavSim.Runtime
{
    // M5 single searcher on 3D terrain. CharacterController + manual gravity; hybrid action space
    // (forward, turn continuous; jump discrete). Continuous goal respawn (no EndEpisode on reach);
    // a pit fall is penalised then teleported to safe ground (memory + episode continue).
    [RequireComponent(typeof(CharacterController))]
    public class NavAgent : Agent
    {
        [SerializeField] private NavEnvironment env;
        [SerializeField] private float maxSpeed = 4f;
        [SerializeField] private float maxTurnDegPerStep = 6f;
        [SerializeField] private float jumpImpulse = 7f;
        [SerializeField] private float gravity = -20f;
        [SerializeField] private float terminalVelocity = -30f;
        [SerializeField] private RewardConfig reward = RewardConfig.Default;

        private CharacterController _cc;
        private float _prevDist;
        private float _vY; // integrated vertical velocity (LocomotionMath)

        public int JumpUses { get; private set; } // eval instrumentation (M5SearchEval reads deltas)

        public override void Initialize() => _cc = GetComponent<CharacterController>();

        // Re-sync the shaping baseline after the ARENA relocates this agent's goal (goal respawn, safe
        // respawn, lesson change). Idempotent with OnEpisodeBegin and the reached/pit blocks, which re-set
        // _prevDist on their own paths. Advisor Finding A (M3).
        public void NotifyGoalMoved() =>
            _prevDist = Vector3.Distance(transform.position, env.GoalPositionFor(this));

        public override void OnEpisodeBegin()
        {
            env.PlaceForNewEpisode(this); // arena places agent + its goal
            _vY = 0f;
            _prevDist = Vector3.Distance(transform.position, env.GoalPositionFor(this));
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            bool grounded = _cc.isGrounded;
            // jumpReady == grounded: a jump is only launchable from the ground (no double-jump).
            float[] obs = ObservationBuilder.Build(
                _cc.velocity, transform.eulerAngles.y, maxSpeed, grounded, grounded);
            foreach (float o in obs) sensor.AddObservation(o);
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            float forward = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
            float turn = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);
            bool jumpPressed = actions.DiscreteActions[0] == 1;

            transform.Rotate(0f, turn * maxTurnDegPerStep, 0f);

            bool grounded = _cc.isGrounded;
            if (jumpPressed && grounded) JumpUses++; // a jump only launches from the ground (LocomotionMath)
            _vY = LocomotionMath.NextVerticalVelocity(
                _vY, grounded, jumpPressed, jumpImpulse, gravity, Time.fixedDeltaTime, terminalVelocity);

            Vector3 horiz = transform.forward * forward * maxSpeed;
            Vector3 move = new Vector3(horiz.x, _vY, horiz.z);
            _cc.Move(move * Time.fixedDeltaTime);

            Vector3 pos = transform.position;
            float dist = Vector3.Distance(pos, env.GoalPositionFor(this));
            bool fellInPit = LocomotionMath.FellInPit(pos.y, env.KillPlaneY);
            // A same-step pit fall VOIDS the reach (pit-first precedence): the fall is the outcome, so no
            // goal bonus and no goal respawn fire. Guards against horizontal ReachedGoal firing while the
            // agent drops through a hole directly under an (elevated) goal.
            bool reached = !fellInPit && env.ReachedGoal(this);
            bool goalVisible = env.GoalVisibleTo(this);

            float step = RewardCalculator.Step(_prevDist, dist, reached, reward, goalVisible, fellInPit);
            var neighborDist = CrowdMath.NeighborDistances(pos, env.MoverPositions(), reward.congestionRadius);
            float moverCost = RewardCalculator.CrowdPenalty(neighborDist, reward);
            AddReward(step - moverCost);

            if (fellInPit)
            {
                env.RespawnToSafeGround(this); // nearest safe ground; memory + episode continue
                _vY = 0f;
                dist = Vector3.Distance(transform.position, env.GoalPositionFor(this));
            }
            else if (reached)
            {
                // Continuous respawn: fresh goal, NO EndEpisode.
                env.NotifyGoalReached();
                env.RespawnGoal(this);
                dist = Vector3.Distance(transform.position, env.GoalPositionFor(this));
            }
            _prevDist = dist;
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var ca = actionsOut.ContinuousActions;
            ca[0] = Input.GetAxis("Vertical");   // forward
            ca[1] = Input.GetAxis("Horizontal"); // turn
            var da = actionsOut.DiscreteActions;  // bind the segment to a local (an `in` param's property
            da[0] = Input.GetKey(KeyCode.Space) ? 1 : 0; // return is an rvalue; CS1612 forbids index-assign)
        }
    }
}
