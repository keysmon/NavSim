namespace NavSim.Runtime
{
    [System.Serializable]
    public struct RewardConfig
    {
        public float goalBonus;
        public float stepPenalty;
        public float shapingScale;

        public float collisionRadius;   // sharp near-field: pay proximity cost inside this
        public float collisionWeight;   // max per-neighbor collision penalty (at contact)
        public float congestionRadius;  // broad crowding neighborhood
        public float congestionWeight;  // flat per-neighbor congestion cost

        public float pitPenalty;        // cost of falling into a hazard pit (M5); "bold" start = 0.25
        public float decoyPenalty;      // cost of touching a wrong-color goal (M6); mirror pit = 0.25

        public static RewardConfig Default => new RewardConfig
        {
            goalBonus = 1.0f,
            stepPenalty = 0.001f,
            shapingScale = 0.05f,
            collisionRadius = 1.2f,
            collisionWeight = 0.01f,
            congestionRadius = 2.5f,
            congestionWeight = 0.002f,
            pitPenalty = 0.25f,
            decoyPenalty = 0.25f
        };
    }
}
