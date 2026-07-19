using UnityEngine;

namespace NavSim.Runtime
{
    // M6 v2 obs: M5 proprioception ONLY (own velocity incl. VERTICAL + grounded + jump-ready). NO goal
    // bearing/range (the agent must SEE its goal) and NO colour cue: the target colour is FIXED
    // (red, GoalPalette.TargetColorIndex), so there is nothing per-episode to announce - identifying
    // the target is entirely the world-perception sensor's job (the ablation's lever). Returns
    // [localVelX, localVelY, localVelZ, grounded01, jumpReady01], length 5.
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
