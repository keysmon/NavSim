using System.Collections.Generic;

namespace NavSim.Runtime
{
    public static class RewardCalculator
    {
        // Pure and Unity-free so it is fully unit-testable.
        public static float Step(
            float prevDistanceToGoal,
            float currDistanceToGoal,
            bool reachedGoal,
            in RewardConfig cfg,
            float visibilityRadius)
        {
            // Visibility-gated shaping: the privileged distance gradient is allowed only when the goal is
            // within ray range. Hidden -> curiosity + the sparse goal bonus drive search (M4 §3).
            float shaping = (currDistanceToGoal < visibilityRadius)
                ? cfg.shapingScale * (prevDistanceToGoal - currDistanceToGoal)
                : 0f;
            float reward = shaping - cfg.stepPenalty;
            if (reachedGoal) reward += cfg.goalBonus;
            return reward;
        }

        // Non-negative crowd penalty for one agent given horizontal distances to nearby peers.
        // Caller subtracts this from the agent's reward.
        public static float CrowdPenalty(IReadOnlyList<float> neighborDistances, in RewardConfig cfg)
        {
            if (neighborDistances == null) return 0f;
            float penalty = 0f;
            for (int i = 0; i < neighborDistances.Count; i++)
            {
                float d = neighborDistances[i];
                if (d < cfg.collisionRadius)
                    penalty += cfg.collisionWeight * (1f - d / cfg.collisionRadius); // 0..collisionWeight
                if (d < cfg.congestionRadius)
                    penalty += cfg.congestionWeight; // flat
            }
            return penalty;
        }
    }
}
