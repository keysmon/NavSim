using UnityEngine;

namespace NavSim.Runtime
{
    /// <summary>
    /// Force-budget helper for the M8 dynamic ramp (a real Rigidbody; PushableRampBody). Validates the
    /// mass/friction/push tuning so the "one agent creeps, two move it usefully" property is honest and
    /// documented, while the actual motion is left to PhysX under manual Physics.Simulate (deterministic
    /// with seeded initial conditions). Heaviness is MASS, not a hard threshold -> continuous shaping
    /// gradient from solo-push to joint-push (avoids the M7-C2 zero-gradient trap).
    /// </summary>
    public static class PushableRamp
    {
        public static Vector3 NetPush(Vector3 pushA, Vector3 pushB)
        {
            Vector3 s = pushA + pushB;
            s.y = 0f;
            return s;
        }

        public static float RequiredBreakForce(float mass, float staticFrictionCoeff, float gravity)
            => staticFrictionCoeff * mass * Mathf.Abs(gravity);

        public static bool Overcomes(float appliedForce, float mass, float staticFrictionCoeff, float gravity)
            => appliedForce > RequiredBreakForce(mass, staticFrictionCoeff, gravity);

        public static float NetForce(float appliedForce, float mass, float staticFrictionCoeff, float gravity)
            => Overcomes(appliedForce, mass, staticFrictionCoeff, gravity)
                ? appliedForce - RequiredBreakForce(mass, staticFrictionCoeff, gravity)
                : 0f;
    }
}
