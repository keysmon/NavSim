using UnityEngine;

namespace NavSim.Runtime
{
    // One immutable difficulty rung: agent count, arena size, obstacle band, and ray length (goal
    // visibility). M3 (ForLevel) co-varies the first three and pins ray length to the max diagonal; M4
    // (ForSearchLevel) additionally FADES ray length as the other three ramp up, so a single lesson index
    // selects the whole tuple.
    public readonly struct DifficultyLevel
    {
        public readonly int AgentCount;
        public readonly float ArenaHalfSize;
        public readonly int MinObstacles;
        public readonly int MaxObstacles;
        public readonly float RayLength; // goal-visibility radius for this rung

        public DifficultyLevel(int agentCount, float arenaHalfSize, int minObstacles, int maxObstacles,
            float rayLength)
        {
            AgentCount = agentCount;
            ArenaHalfSize = arenaHalfSize;
            MinObstacles = minObstacles;
            MaxObstacles = maxObstacles;
            RayLength = rayLength;
        }
    }

    // One M5 terrain-difficulty rung. Single learner, FIXED arena + sight; the STRUCTURE ramps
    // (walls -> platforms -> pits + movers). GoalElevated allows the goal to spawn on a platform.
    public readonly struct TerrainLevel
    {
        public readonly int Walls;
        public readonly int Platforms;
        public readonly int Pits;
        public readonly int Movers;
        public readonly bool GoalElevated;

        public TerrainLevel(int walls, int platforms, int pits, int movers, bool goalElevated)
        {
            Walls = walls; Platforms = platforms; Pits = pits; Movers = movers; GoalElevated = goalElevated;
        }
    }

    // Pure map: difficulty level -> environment configuration. Unity-math-using but scene-free, so
    // EditMode-testable. The training poll, the eval harness, and the demo all resolve configs through
    // here (or through the independent setters it feeds). Two ladders: ForLevel (M3, ray pinned to
    // MaxArenaDiagonal = always visible) and ForSearchLevel (M4, ray fades from MaxArenaDiagonal to hidden).
    public static class DifficultyMapper
    {
        public const int NumLevels = 4;
        public const float MaxArenaHalfSize = 11f; // == the level-3 arena; ray length keys off this

        public const float SightRange = 15f;   // fixed "flashlight" reach (~half the arena diagonal)
        public const float M5ArenaHalf = 11f;  // fixed arena (structure ramps, not size)

        // The fully-visible reference: the largest arena's diagonal. M3 pins every rung's ray length here
        // so goals stay visible; M4's search ladder starts here (L0) and fades below it to hide the goal.
        public static float MaxArenaDiagonal => MaxArenaHalfSize * 2f * Mathf.Sqrt(2f);

        // L0 easiest -> L3 hardest; every axis monotonic non-decreasing. Small arenas only ever co-occur
        // with few obstacles/agents, which structurally prevents the ClearPoint infeasibility corner.
        private static readonly DifficultyLevel[] Levels =
        {
            new DifficultyLevel(2, 6f, 0, 1, MaxArenaDiagonal),
            new DifficultyLevel(4, 8f, 2, 3, MaxArenaDiagonal),
            new DifficultyLevel(6, 10f, 4, 5, MaxArenaDiagonal),
            new DifficultyLevel(8, MaxArenaHalfSize, 6, 8, MaxArenaDiagonal),
        };

        // M4 (hidden-goal search) rungs: visibility fades from full diagonal down to ~0.2x (hidden),
        // while agent count / arena / obstacle density ramp up. Fractions recalibrated off the probe.
        private static readonly DifficultyLevel[] SearchLevels =
        {
            new DifficultyLevel(2, 8f,  0, 1, MaxArenaDiagonal),          // visible warmup
            new DifficultyLevel(4, 9f,  2, 3, 0.60f * MaxArenaDiagonal),
            new DifficultyLevel(6, 10f, 4, 5, 0.35f * MaxArenaDiagonal),
            new DifficultyLevel(8, MaxArenaHalfSize, 6, 8, 0.20f * MaxArenaDiagonal), // hidden
        };

        public static DifficultyLevel ForLevel(int level) =>
            Levels[Mathf.Clamp(level, 0, NumLevels - 1)];

        public static DifficultyLevel ForSearchLevel(int level) =>
            SearchLevels[Mathf.Clamp(level, 0, NumLevels - 1)];

        // L0 open/flat warmup -> L3 hidden + elevated + crowded + pits. Every axis monotonic non-decreasing.
        private static readonly TerrainLevel[] TerrainLevels =
        {
            new TerrainLevel(0, 0, 0, 0, false), // L0: goal in the open, flat
            new TerrainLevel(2, 0, 0, 2, false), // L1: goal behind occluding walls
            new TerrainLevel(3, 1, 2, 3, true),  // L2: goal on a ramp-reachable platform + pits
            new TerrainLevel(4, 2, 3, 4, true),  // L3: hidden, elevated, crowded, more pits
        };

        public static TerrainLevel ForTerrainLevel(int level) =>
            TerrainLevels[Mathf.Clamp(level, 0, NumLevels - 1)];
    }
}
