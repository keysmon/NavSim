using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

namespace NavSim.Runtime
{
    // M6 v2 single searcher on 3D terrain. CharacterController + manual gravity; hybrid action space (forward,
    // turn continuous; jump discrete). FIXED-TARGET visual discrimination: the target is always the red goal
    // (GoalPalette.TargetColorIndex) among 3 geometrically identical goals - no cue in the obs; reaching the
    // TARGET goal respawns a fresh triad (continuous respawn, no EndEpisode); touching a DECOY costs reward
    // and - under the hard schedule (DecoyHard) - ends the episode; a pit fall is penalised then the agent is
    // teleported to safe ground (memory + episode continue).
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
        private float _vY;          // integrated vertical velocity (LocomotionMath)
        private bool _wasOnDecoy;   // one-shot decoy contact: charge -decoyPenalty on ENTRY only (camping doesn't re-charge)

        public int JumpUses { get; private set; } // eval instrumentation (eval reads deltas)

        public override void Initialize()
        {
            _cc = GetComponent<CharacterController>();
            // M6: shaping is gated on env.TargetPerceivable (a per-arm geometric gate matched to the sensor's true
            // FOV), so there is no ray-fan to cache here — the pixel arm has no rays, and rayC's per-color tags
            // would break M5's single-"goal"-tag ray gate.
        }

        // Re-sync the shaping baseline after the ARENA relocates this agent's (target) goal — triad respawn, safe
        // respawn, lesson change. Idempotent with OnEpisodeBegin and the reached/pit blocks. Advisor Finding A (M3).
        public void NotifyGoalMoved() =>
            _prevDist = Vector3.Distance(transform.position, env.GoalPositionFor(this));

        public override void OnEpisodeBegin()
        {
            env.PlaceForNewEpisode(this); // arena places agent + the triad
            _vY = 0f;
            _wasOnDecoy = false;
            _prevDist = Vector3.Distance(transform.position, env.GoalPositionFor(this));
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            bool grounded = _cc.isGrounded;
            // jumpReady == grounded: a jump is only launchable from the ground (no double-jump). NO colour cue:
            // the target is fixed (red) - which goal is the target is for the SENSOR to determine.
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
            // Precedence: pit-first (a same-step fall voids the outcome), then decoy, then the target reach.
            bool touchingDecoy = !fellInPit && env.TouchedDecoy(this);
            bool decoyEnter = touchingDecoy && !_wasOnDecoy;  // ONE-SHOT on entry
            _wasOnDecoy = touchingDecoy;
            bool reached = !fellInPit && !touchingDecoy && env.ReachedGoal(this);
            bool goalVisible = env.TargetPerceivable(this);   // shaping gated on the agent's ACTUAL forward FOV

            float step = RewardCalculator.Step(_prevDist, dist, reached, reward, goalVisible, fellInPit);
            if (decoyEnter) step -= reward.decoyPenalty;
            var neighborDist = CrowdMath.NeighborDistances(pos, env.MoverPositions(), reward.congestionRadius);
            float moverCost = RewardCalculator.CrowdPenalty(neighborDist, reward);
            AddReward(step - moverCost);

            if (fellInPit)
            {
                env.RespawnToSafeGround(this); // nearest safe ground; memory + episode continue
                _vY = 0f;
                dist = Vector3.Distance(transform.position, env.GoalPositionFor(this));
            }
            else if (decoyEnter && !env.EvalMode && env.DecoyHard)
            {
                // TRAINING hard schedule: a wrong-colour pick ends the episode (on ENTRY) once past the soft warmup
                // (or at L1+) — trains TARGET-FIRST, matching the always-hard eval. SUPPRESSED in eval (env.EvalMode):
                // the harness owns the boundary + detects the decoy geometrically, so an EndEpisode here would re-roll
                // the arena mid-episode and corrupt the measurement.
                EndEpisode();
                return;
            }
            else if (reached)
            {
                // Continuous respawn: fresh triad, NO EndEpisode.
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
