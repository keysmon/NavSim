using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

namespace NavSim.Runtime
{
    // M8 ramp-push agent: a FORK of CoopAgent. NavAgent's locomotion EXACTLY (CharacterController +
    // LocomotionMath, hybrid action space, same Heuristic) MINUS goal/decoy/pit/shaping logic - this
    // agent adds NO rewards itself, ever (the ONLY reward surface is ArmRouting, applied by RampArena).
    // Outcome detection lives in the ARENA (single boundary owner - the M6 lesson).
    // Obs = BuildRamp: 5-float proprioception + the shared rampAtTarget indicator (6 floats).
    // Cooperation seam: OnControllerColliderHit converts a side-contact with the ramp into a real physics
    // push on its dynamic Rigidbody (PushableRampBody), which also records this agent as a pusher this step.
    [RequireComponent(typeof(CharacterController))]
    public class RampAgent : Agent
    {
        [SerializeField] private RampArena arena;
        [SerializeField] private float maxSpeed = 4f;
        [SerializeField] private float maxTurnDegPerStep = 6f;
        [SerializeField] private float jumpImpulse = 7f;
        [SerializeField] private float gravity = -20f;
        [SerializeField] private float terminalVelocity = -30f;
        [SerializeField] private float pushForceNewtons = 10f;   // per-agent push magnitude (real physics force)
        [SerializeField] private int agentIndex = 0;             // 0/1, set by scene setup; used for the joint-push tracker

        private CharacterController _cc;
        private float _vY; // integrated vertical velocity (LocomotionMath)

        // Verify-early #1 diagnostics: raw ramp contacts (BEFORE the |n.y|/into-face skip) and the last skipped
        // contact normal, so the selftest can distinguish "never touched the ramp" from "touched but skipped".
        public int DebugRawRampContacts { get; private set; }
        public int DebugSkippedContacts { get; private set; }
        public Vector3 DebugLastRampNormal { get; private set; }

        public bool Grounded => _cc != null && _cc.isGrounded;

        public override void Initialize() => _cc = GetComponent<CharacterController>();

        public override void OnEpisodeBegin()
        {
            // Layout is the ARENA's job (RampArena.ResetEpisode places agents/ramp/goal AFTER ending the
            // episode). Both agents' OnEpisodeBegin fire from the same arena boundary call, so a per-agent
            // re-roll here would double-place - the arena is the single layout authority.
            _vY = 0f;
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            bool grounded = _cc.isGrounded;
            float[] obs = ObservationBuilder.BuildRamp(
                _cc.velocity, transform.eulerAngles.y, maxSpeed, grounded, grounded, arena.RampAtTarget);
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

        // Fires during _cc.Move(...) when the capsule contacts a collider. If we hit the SIDE of the ramp,
        // apply a real physics push force to its Rigidbody (via PushableRampBody.ApplyPush, which also records
        // this agent as a pusher this step for the joint-push tracker). PhysX integrates the ramp in the next
        // Physics.Simulate/FixedUpdate. Heaviness = the ramp's mass, so a lone agent creeps and two move it.
        private void OnControllerColliderHit(ControllerColliderHit hit)
        {
            var ramp = hit.collider.GetComponentInParent<PushableRampBody>();
            if (ramp == null) return;
            DebugRawRampContacts++;                 // raw contact with the ramp (verify-early #1 diagnostic)
            DebugLastRampNormal = hit.normal;

            // Only push on ~horizontal (side) contacts, so climbing the placed sloped face - whose contact
            // normal has a large vertical component - does not shove the ramp away.
            Vector3 n = hit.normal;
            if (Mathf.Abs(n.y) > 0.5f) { DebugSkippedContacts++; return; }

            Vector3 dir = transform.forward; dir.y = 0f;
            if (dir.sqrMagnitude < 1e-6f) { DebugSkippedContacts++; return; }
            dir.Normalize();

            // Require driving INTO the ramp (push opposes the horizontal contact normal).
            n.y = 0f;
            if (n.sqrMagnitude < 1e-6f || Vector3.Dot(dir, -n.normalized) <= 0f) { DebugSkippedContacts++; return; }

            ramp.ApplyPush(agentIndex, dir * pushForceNewtons, hit.point);
        }

        // TEST-ONLY (verify-early #1): apply EXACTLY OnActionReceived's locomotion for full-forward input,
        // INCLUDING gravity (_vY via LocomotionMath) so the pusher stays GROUNDED and keeps shoving the ramp's
        // vertical face instead of floating up and over the slope. Drives `_cc.Move -> OnControllerColliderHit
        // -> ApplyPush` under the manual eval seam WITHOUT a trained model - tests the PUSH mechanic, not policy.
        public void DebugDriveForward(float dt)
        {
            if (_cc == null) _cc = GetComponent<CharacterController>();
            bool grounded = _cc.isGrounded;
            _vY = LocomotionMath.NextVerticalVelocity(_vY, grounded, false, jumpImpulse, gravity, dt, terminalVelocity);
            Vector3 horiz = transform.forward * maxSpeed;
            _cc.Move(new Vector3(horiz.x, _vY, horiz.z) * dt);
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
