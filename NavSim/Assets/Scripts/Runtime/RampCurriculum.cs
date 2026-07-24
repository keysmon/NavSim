using UnityEngine;

namespace NavSim.Runtime
{
    /// <summary>
    /// Pure layout curriculum calculations for the ramp task.
    /// </summary>
    public static class RampCurriculum
    {
        /// <summary>
        /// Starts the S0 ramp just outside its target, then recedes it to the normal start as successes accrue.
        /// Evaluation always uses the normal start so it never measures the softened task.
        /// </summary>
        public static float S0StartX(
            int successes,
            int successHorizon,
            float targetX,
            float normalStartX,
            float initialPushDistance,
            bool evalMode)
        {
            float easyStartX = targetX - Mathf.Abs(initialPushDistance);
            return Competence.RampValue(
                successes,
                successHorizon,
                easyStartX,
                normalStartX,
                evalMode);
        }
    }
}
