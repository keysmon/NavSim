namespace NavSim.Runtime
{
    // Pure, Unity-free predicate. The caller computes lineOfSightClear (a Physics.Linecast against
    // STATIC geometry only) and injects it, keeping this fully unit-testable. Upgrades M4's radius
    // proxy: a goal behind a wall is now HIDDEN even when close (M5 spec §8).
    public static class VisibilityGate
    {
        public static bool IsGoalVisible(float dist, float sightRange, bool lineOfSightClear)
            => dist < sightRange && lineOfSightClear;
    }
}
