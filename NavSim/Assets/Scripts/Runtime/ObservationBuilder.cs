using UnityEngine;

namespace NavSim.Runtime
{
    // M6 obs: M5 proprioception (own velocity incl. VERTICAL + grounded + jump-ready) PLUS the persistent RGB
    // target-color cue (spec §5). NO goal bearing/range (the agent must SEE its goal); the cue is the ONLY
    // channel that says WHICH color is the target this episode, and it is given identically to all three arms
    // so world-perception stays the sole lever. Returns
    // [localVelX, localVelY, localVelZ, grounded01, jumpReady01, cueR, cueG, cueB], length 8.
    public static class ObservationBuilder
    {
        public static float[] Build(Vector3 agentVelocity, float headingDeg, float maxSpeed,
            bool grounded, bool jumpReady, Color targetCue)
        {
            Quaternion toLocal = Quaternion.Euler(0f, -headingDeg, 0f);
            Vector3 localVel = toLocal * (agentVelocity / Mathf.Max(maxSpeed, 1e-3f));
            return new[] { localVel.x, localVel.y, localVel.z, grounded ? 1f : 0f, jumpReady ? 1f : 0f,
                           targetCue.r, targetCue.g, targetCue.b };
        }
    }
}
