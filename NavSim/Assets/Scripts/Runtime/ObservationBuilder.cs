using UnityEngine;

namespace NavSim.Runtime
{
    public static class ObservationBuilder
    {
        // 3D on the XZ plane. headingDeg is rotation about world Y; 0 degrees faces +Z (transform.forward).
        // Returns [localVelX, localVelZ, goalDirLocalX, goalDirLocalZ, normDistance].
        public static float[] Build(
            Vector3 agentPos, float headingDeg,
            Vector3 agentVelocity, Vector3 goalPos,
            float maxSpeed, float arenaDiagonal, float compassWeight)
        {
            Quaternion toLocal = Quaternion.Euler(0f, -headingDeg, 0f);

            Vector3 localVel = toLocal * (agentVelocity / Mathf.Max(maxSpeed, 1e-3f));

            Vector3 toGoal = goalPos - agentPos;
            toGoal.y = 0f;
            float dist = toGoal.magnitude;
            Vector3 dirLocal = dist > 1e-6f ? (toLocal * toGoal.normalized) : Vector3.zero;
            float normDist = Mathf.Clamp01(dist / Mathf.Max(arenaDiagonal, 1e-3f));

            return new float[]
            {
                localVel.x,
                localVel.z,
                dirLocal.x * compassWeight,
                dirLocal.z * compassWeight,
                Mathf.Lerp(1f, normDist, compassWeight) // distance hidden when compass off
            };
        }
    }
}
