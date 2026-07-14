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
        // Anti-freeze explore bias: reward per FULL-SPEED step of actual horizontal displacement. Pushes the
        // fresh policy off the frozen fixed point (freeze = min-loss under a step cost) so it moves enough to
        // sample the perception-gated shaping + goal bonus. Kept small; a searcher (bias + shaping + reaches)
        // out-scores a farmer (bias only), so it un-freezes without becoming the objective.
        [SerializeField] private float exploreBias = 0.003f;

        private CharacterController _cc;
        private float _prevDist;
        private float _vY; // integrated vertical velocity (LocomotionMath)
        private RayPerceptionSensorComponent3D _fwdRays; // forward fan, for the perception-gated shaping
        private int _goalTagIdx = -1;                    // index of the "goal" tag in the forward fan

        public int JumpUses { get; private set; } // eval instrumentation (M5SearchEval reads deltas)

        public override void Initialize()
        {
            _cc = GetComponent<CharacterController>();
            // Cache the forward ray fan + goal tag so shaping can be gated on what the POLICY actually
            // perceives (a forward-ray goal hit), not on privileged omnidirectional geometry (OnActionReceived).
            foreach (var r in GetComponents<RayPerceptionSensorComponent3D>())
                if (r.SensorName == "RayForward") _fwdRays = r;
            if (_fwdRays != null) _goalTagIdx = _fwdRays.DetectableTags.IndexOf("goal");
        }

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

        // True iff the forward ray fan currently detects the goal tag. Gates the distance-shaping on the agent's
        // ACTUAL perception (rays) instead of a privileged omnidirectional Linecast, so shaping is positive-
        // dominated ("see the goal ahead + approach -> reward") and never scores movement toward a goal the policy
        // cannot observe -- which had made FREEZING the optimal policy (advisor, 2026-07-14).
        private bool GoalInForwardRays()
        {
            if (_fwdRays == null || _goalTagIdx < 0) return false;
            var output = RayPerceptionSensor.Perceive(_fwdRays.GetRayPerceptionInput(), false);
            var outs = output.RayOutputs;
            for (int i = 0; i < outs.Length; i++)
                if (outs[i].HasHit && outs[i].HitTagIndex == _goalTagIdx) return true;
            return false;
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
            Vector3 posBefore = transform.position;
            _cc.Move(move * Time.fixedDeltaTime);

            Vector3 pos = transform.position;
            // Anti-freeze explore bias from ACTUAL horizontal displacement (0 when frozen or wall-blocked).
            float horizMoved = new Vector2(pos.x - posBefore.x, pos.z - posBefore.z).magnitude;
            float exploreReward = exploreBias * Mathf.Clamp01(horizMoved / (maxSpeed * Time.fixedDeltaTime));
            float dist = Vector3.Distance(pos, env.GoalPositionFor(this));
            bool fellInPit = LocomotionMath.FellInPit(pos.y, env.KillPlaneY);
            // A same-step pit fall VOIDS the reach (pit-first precedence): the fall is the outcome, so no
            // goal bonus and no goal respawn fire. Guards against horizontal ReachedGoal firing while the
            // agent drops through a hole directly under an (elevated) goal.
            bool reached = !fellInPit && env.ReachedGoal(this);
            bool goalVisible = GoalInForwardRays(); // shaping gated on PERCEIVED (forward-ray) goal, not privileged LoS

            float step = RewardCalculator.Step(_prevDist, dist, reached, reward, goalVisible, fellInPit);
            var neighborDist = CrowdMath.NeighborDistances(pos, env.MoverPositions(), reward.congestionRadius);
            float moverCost = RewardCalculator.CrowdPenalty(neighborDist, reward);
            AddReward(step - moverCost + exploreReward);

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
