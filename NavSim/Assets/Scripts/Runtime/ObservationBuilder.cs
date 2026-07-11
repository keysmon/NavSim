using UnityEngine;

namespace NavSim.Runtime
{
    public static class ObservationBuilder
    {
        // heading is measured so that 0 degrees faces world +X.
        // Returns [localVelX, localVelY, goalDirLocalX, goalDirLocalY, normDistance].
        public static float[] Build(
            Vector2 agentPos, float agentHeadingDeg,
            Vector2 agentVelocity, Vector2 goalPos,
            float maxSpeed, float arenaDiagonal, float compassWeight)
        {
            Quaternion toLocal = Quaternion.Euler(0f, 0f, -agentHeadingDeg);

            Vector2 localVel = toLocal * (agentVelocity / Mathf.Max(maxSpeed, 1e-3f));

            Vector2 toGoal = goalPos - agentPos;
            float dist = toGoal.magnitude;
            Vector2 dirLocal = dist > 1e-6f ? (Vector2)(toLocal * toGoal.normalized) : Vector2.zero;
            float normDist = Mathf.Clamp01(dist / Mathf.Max(arenaDiagonal, 1e-3f));

            return new float[]
            {
                localVel.x,
                localVel.y,
                dirLocal.x * compassWeight,
                dirLocal.y * compassWeight,
                Mathf.Lerp(1f, normDist, compassWeight) // distance hidden when compass off
            };
        }
    }
}
