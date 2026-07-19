using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

namespace NavSim.Runtime
{
    // M7 coop agent: NavAgent's locomotion EXACTLY (CharacterController + LocomotionMath, hybrid
    // action space, same Heuristic) MINUS goal/decoy/pit/shaping logic - this agent adds NO rewards
    // itself, ever (spec sec 4: the ONLY reward surface is ArmRouting, applied by CoopArena).
    // Outcome detection lives in the ARENA (single boundary owner - the M6 lesson).
    // Obs = BuildCoop: 5-float proprioception + the shared doorOpen indicator (6 floats).
    [RequireComponent(typeof(CharacterController))]
    public class CoopAgent : Agent
    {
        [SerializeField] private CoopArena arena;
        [SerializeField] private float maxSpeed = 4f;
        [SerializeField] private float maxTurnDegPerStep = 6f;
        [SerializeField] private float jumpImpulse = 7f;
        [SerializeField] private float gravity = -20f;
        [SerializeField] private float terminalVelocity = -30f;

        private CharacterController _cc;
        private float _vY; // integrated vertical velocity (LocomotionMath)

        public bool Grounded => _cc != null && _cc.isGrounded; // arena reads this for plate occupancy

        public override void Initialize() => _cc = GetComponent<CharacterController>();

        public override void OnEpisodeBegin()
        {
            // Layout is the ARENA's job (CoopArena.ResetEpisode places agents/goal/plate AFTER ending
            // the episode). Both agents' OnEpisodeBegin fire from the same arena boundary call, so a
            // per-agent re-roll here would double-place - the arena is the single layout authority.
            _vY = 0f;
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            bool grounded = _cc.isGrounded;
            // jumpReady == grounded (no double-jump), as in NavAgent. Index 5 = the shared door state.
            float[] obs = ObservationBuilder.BuildCoop(
                _cc.velocity, transform.eulerAngles.y, maxSpeed, grounded, grounded, arena.DoorOpen);
            foreach (float o in obs) sensor.AddObservation(o);
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            // Movement ONLY - no reward, no outcome detection (the arena's Tick owns both).
            float forward = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
            float turn = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);
            bool jumpPressed = actions.DiscreteActions[0] == 1;

            transform.Rotate(0f, turn * maxTurnDegPerStep, 0f);

            bool grounded = _cc.isGrounded;
            _vY = LocomotionMath.NextVerticalVelocity(
                _vY, grounded, jumpPressed, jumpImpulse, gravity, Time.fixedDeltaTime, terminalVelocity);

            Vector3 horiz = transform.forward * forward * maxSpeed;
            Vector3 move = new Vector3(horiz.x, _vY, horiz.z);
            _cc.Move(move * Time.fixedDeltaTime);
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var ca = actionsOut.ContinuousActions;
            ca[0] = Input.GetAxis("Vertical");   // forward
            ca[1] = Input.GetAxis("Horizontal"); // turn
            var da = actionsOut.DiscreteActions;  // bind the segment to a local (an `in` param's property
            da[0] = Input.GetKey(KeyCode.Space) ? 1 : 0; // return is an rvalue; CS1612 forbids index-assign)
        }

        // Teleport helper: a CharacterController caches its position and fights direct transform writes;
        // disable it around the move (the NavEnvironment.PlaceAt idiom, owned here so the arena stays
        // free of per-agent component juggling).
        public void TeleportTo(Vector3 pos, float yawDeg)
        {
            if (_cc == null) _cc = GetComponent<CharacterController>();
            _cc.enabled = false;
            transform.position = pos;
            transform.rotation = Quaternion.Euler(0f, yawDeg, 0f);
            _cc.enabled = true;
            _vY = 0f;
        }
    }
}
