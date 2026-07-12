namespace NavSim.Runtime
{
    [System.Serializable]
    public struct RewardConfig
    {
        public float goalBonus;
        public float stepPenalty;
        public float compassWeight; // removed in Task 2
        public float shapingScale;

        public float collisionRadius;   // sharp near-field: pay proximity cost inside this
        public float collisionWeight;   // max per-neighbor collision penalty (at contact)
        public float congestionRadius;  // broad crowding neighborhood
        public float congestionWeight;  // flat per-neighbor congestion cost

        public static RewardConfig Default => new RewardConfig
        {
            goalBonus = 1.0f,
            stepPenalty = 0.001f,
            compassWeight = 1.0f,
            shapingScale = 0.05f,
            collisionRadius = 1.2f,
            collisionWeight = 0.01f,
            congestionRadius = 2.5f,
            congestionWeight = 0.002f
        };
    }
}
