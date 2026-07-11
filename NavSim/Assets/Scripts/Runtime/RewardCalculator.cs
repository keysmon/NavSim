namespace NavSim.Runtime
{
    public static class RewardCalculator
    {
        // Pure and Unity-free so it is fully unit-testable.
        public static float Step(
            float prevDistanceToGoal,
            float currDistanceToGoal,
            bool reachedGoal,
            in RewardConfig cfg)
        {
            float shaping = cfg.compassWeight * cfg.shapingScale
                            * (prevDistanceToGoal - currDistanceToGoal);
            float reward = shaping - cfg.stepPenalty;
            if (reachedGoal) reward += cfg.goalBonus;
            return reward;
        }
    }
}
