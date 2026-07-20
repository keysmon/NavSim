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

        // M7 coop obs: the 5-float proprioception + a shared doorOpen indicator (spec sec 3 - an
        // indicator light, not a communication channel; removes the partial-obs confound so the
        // measured lever stays credit assignment). Returns [Build[0..4], doorOpen01], length 6.
        public static float[] BuildCoop(Vector3 agentVelocity, float headingDeg, float maxSpeed,
            bool grounded, bool jumpReady, bool doorOpen)
        {
            float[] b = Build(agentVelocity, headingDeg, maxSpeed, grounded, jumpReady);
            return new[] { b[0], b[1], b[2], b[3], b[4], doorOpen ? 1f : 0f };
        }
    }
}
