namespace NavSim.Runtime
{
    [System.Serializable]
    public struct RewardConfig
    {
        public float goalBonus;
        public float stepPenalty;
        public float compassWeight; // 1.0 for M0..M3; the M4 plan fades this to 0
        public float shapingScale;

        public static RewardConfig Default => new RewardConfig
        {
            goalBonus = 1.0f,
            stepPenalty = 0.001f,
            compassWeight = 1.0f,
            shapingScale = 0.05f
        };
    }
}
