using UnityEngine;

namespace NavSim.Runtime
{
    // Visual-only observations: own velocity (proprioception) + a one-hot identity of the agent's
    // assigned goal color. NO goal bearing/range here — the agent must SEE its goal via rays.
    // 3D on the XZ plane; headingDeg is rotation about world Y (0 deg faces +Z == transform.forward).
    // Returns [localVelX, localVelZ, oneHot(myColor, numColors)...], length 2 + numColors.
    public static class ObservationBuilder
    {
        public static float[] Build(Vector3 agentVelocity, float headingDeg, float maxSpeed,
            int myColor, int numColors)
        {
            Quaternion toLocal = Quaternion.Euler(0f, -headingDeg, 0f);
            Vector3 localVel = toLocal * (agentVelocity / Mathf.Max(maxSpeed, 1e-3f));

            var obs = new float[2 + numColors];
            obs[0] = localVel.x;
            obs[1] = localVel.z;
            if (myColor >= 0 && myColor < numColors) obs[2 + myColor] = 1f;
            return obs;
        }
    }
}
