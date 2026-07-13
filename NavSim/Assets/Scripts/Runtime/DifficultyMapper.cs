using UnityEngine;

namespace NavSim.Runtime
{
    // One immutable difficulty rung. The collapsed M3 curriculum co-varies all three axes together
    // (agent count, arena size, obstacle count), so a single lesson index selects one of these.
    public readonly struct DifficultyLevel
    {
        public readonly int AgentCount;
        public readonly float ArenaHalfSize;
        public readonly int MinObstacles;
        public readonly int MaxObstacles;

        public DifficultyLevel(int agentCount, float arenaHalfSize, int minObstacles, int maxObstacles)
        {
            AgentCount = agentCount;
            ArenaHalfSize = arenaHalfSize;
            MinObstacles = minObstacles;
            MaxObstacles = maxObstacles;
        }
    }

    // Pure map: difficulty level -> environment configuration. Unity-math-using but scene-free, so
    // EditMode-testable. The training poll, the eval harness, and the demo all resolve configs through
    // here (or through the independent setters it feeds). Ray length is pinned to MaxArenaDiagonal.
    public static class DifficultyMapper
    {
        public const int NumLevels = 4;
        public const float MaxArenaHalfSize = 11f; // == the level-3 arena; ray length keys off this

        // Ray length MUST be pinned to this (the largest lesson's diagonal) so goals stay visible at
        // every lesson. A ray length tracking the current arena would hide goals at large sizes = M4.
        public static float MaxArenaDiagonal => MaxArenaHalfSize * 2f * Mathf.Sqrt(2f);

        // L0 easiest -> L3 hardest; every axis monotonic non-decreasing. Small arenas only ever co-occur
        // with few obstacles/agents, which structurally prevents the ClearPoint infeasibility corner.
        private static readonly DifficultyLevel[] Levels =
        {
            new DifficultyLevel(2, 6f, 0, 1),
            new DifficultyLevel(4, 8f, 2, 3),
            new DifficultyLevel(6, 10f, 4, 5),
            new DifficultyLevel(8, MaxArenaHalfSize, 6, 8),
        };

        public static DifficultyLevel ForLevel(int level) =>
            Levels[Mathf.Clamp(level, 0, NumLevels - 1)];
    }
}
