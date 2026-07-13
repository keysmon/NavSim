using System.Collections.Generic;

namespace NavSim.Runtime
{
    public static class RewardCalculator
    {
        // Pure and Unity-free so it is fully unit-testable.
        // M5: shaping is gated on a PRECOMPUTED goalVisible (dist < sightRange AND static line-of-sight
        // clear -> VisibilityGate), replacing M4's ray-radius proxy so a goal behind a wall is hidden even
        // when close. Adds a pit-fall penalty. The sparse goal bonus and the pit penalty are NEVER gated.
        // Shaping is ALSO suppressed on a pit-fall step: the falling position is garbage (Y below the floor
        // inflates the distance), so the intended signal is the -pitPenalty alone, not a spurious gradient.
        public static float Step(
            float prevDistanceToGoal,
            float currDistanceToGoal,
            bool reachedGoal,
            in RewardConfig cfg,
            bool goalVisible,
            bool fellInPit)
        {
            float shaping = (goalVisible && !fellInPit)
                ? cfg.shapingScale * (prevDistanceToGoal - currDistanceToGoal)
                : 0f;
            float reward = shaping - cfg.stepPenalty;
            if (reachedGoal) reward += cfg.goalBonus;
            if (fellInPit) reward -= cfg.pitPenalty;
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
