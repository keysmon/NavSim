using UnityEngine;

namespace NavSim.Runtime
{
    /// <summary>
    /// The movable ramp: a thin wrapper over a DYNAMIC Rigidbody. Agents call ApplyPush() from
    /// OnControllerColliderHit, which applies a real force AND records the agent as a pusher this physics
    /// step (for the joint-push tracker). PhysX integrates the motion (Physics.Simulate in eval,
    /// FixedUpdate in training). "Too heavy for one" is realized as MASS (the arena ramps it light->heavy):
    /// a lone agent's push barely clears static friction so the ramp CREEPS (nonzero displacement -> a
    /// continuous ramp-to-target shaping gradient), it is just too slow to place the ramp AND climb in the
    /// step budget, so two are needed - a temporal, empirically-checked forcing (verify-early #1 + the solo
    /// confound detector), NOT a hard gate (a hard gate would zero the gradient = the M7-C2 trap).
    /// </summary>
    [RequireComponent(typeof(Rigidbody))]
    public class PushableRampBody : MonoBehaviour
    {
        [SerializeField] private float mass = 2f;             // heaviness knob (arena sets it per lesson)
        // Viscous damping tuned by verify-early #1 (M8RampPhysicsSelftest) so the lateral push cleanly
        // separates 1-vs-2 over the 5m push in the 3000-step budget: heavy(6kg) 1 agent covers only ~2.3m
        // (fails), 2 agents place at ~2246 steps, light(2kg) 1 agent places at ~1500. Friction is ~0 under
        // FreezePositionY (the ramp rides just above the floor), so damping is the separation lever.
        [SerializeField] private float linearDamping = 23f;

        private Rigidbody _rb;
        private int _pusherMask;

        public float Mass
        {
            get => _rb != null ? _rb.mass : mass;
            set { mass = value; if (_rb != null) _rb.mass = value; }
        }
        public Vector3 Position => _rb != null ? _rb.position : transform.position;
        public int PushersThisStep => CountBits(_pusherMask);
        public int TotalPushEvents { get; private set; }   // cumulative ApplyPush calls (verify-early #1 push-fire diagnostic)

        private void Awake()
        {
            _rb = GetComponent<Rigidbody>();
            _rb.isKinematic = false;
            _rb.useGravity = true;
            _rb.mass = mass;
            _rb.linearDamping = linearDamping;                // Unity 6 name (was 'drag')
            _rb.constraints = RigidbodyConstraints.FreezeRotation | RigidbodyConstraints.FreezePositionY;
        }

        // Applies a real physics force AND flags this agent as a pusher this step (joint-push tracker).
        public void ApplyPush(int agentIndex, Vector3 force, Vector3 point)
        {
            if (_rb == null) _rb = GetComponent<Rigidbody>();
            _rb.AddForceAtPosition(force, point, ForceMode.Force);
            _pusherMask |= (1 << agentIndex);
            TotalPushEvents++;
        }

        public void ClearPushers() => _pusherMask = 0;

        public void ResetTo(Vector3 pos, Quaternion rot)
        {
            if (_rb == null) _rb = GetComponent<Rigidbody>();
            _rb.linearVelocity = Vector3.zero;                // Unity 6 name (was 'velocity')
            _rb.angularVelocity = Vector3.zero;
            _rb.position = pos;
            transform.SetPositionAndRotation(pos, rot);
            _pusherMask = 0;
        }

        private static int CountBits(int m) { int c = 0; while (m != 0) { c += m & 1; m >>= 1; } return c; }
    }
}
