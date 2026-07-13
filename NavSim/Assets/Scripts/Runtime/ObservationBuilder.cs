using UnityEngine;

namespace NavSim.Runtime
{
    // M5 visual-only obs: own velocity (including VERTICAL, so climbing/falling registers) + grounded +
    // jump-ready. NO goal bearing/range (the agent must SEE its goal via rays), NO color one-hot (single
    // learner). 3D on the XZ plane; headingDeg is rotation about world Y (0 deg faces +Z == forward).
    // Returns [localVelX, localVelY, localVelZ, grounded01, jumpReady01], length 5.
    public static class ObservationBuilder
    {
        public static float[] Build(Vector3 agentVelocity, float headingDeg, float maxSpeed,
            bool grounded, bool jumpReady)
        {
            Quaternion toLocal = Quaternion.Euler(0f, -headingDeg, 0f);
            Vector3 localVel = toLocal * (agentVelocity / Mathf.Max(maxSpeed, 1e-3f));
            return new[] { localVel.x, localVel.y, localVel.z, grounded ? 1f : 0f, jumpReady ? 1f : 0f };
        }
    }
}
