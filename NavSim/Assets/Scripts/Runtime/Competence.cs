using UnityEngine;

namespace NavSim.Runtime
{
    /// <summary>
    /// Competence-gated curriculum ramp (the M7 CoopArena idiom, factored to a testable leaf).
    /// A monotone per-lesson success counter drives a 0..1 progress that Lerps an easy value to a
    /// hard value; EvalMode always returns the hard value so the eval never measures a softened task.
    /// </summary>
    public static class Competence
    {
        public static float Ramp01(int successes, int horizon)
            => Mathf.Clamp01((float)successes / Mathf.Max(horizon, 1));

        public static float RampValue(int successes, int horizon, float easy, float hard, bool evalMode)
            => evalMode ? hard : Mathf.Lerp(easy, hard, Ramp01(successes, horizon));
    }
}
