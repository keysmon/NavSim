using NUnit.Framework;
using UnityEngine;
using NavSim.Runtime;

namespace NavSim.Tests.EditMode
{
    public class DifficultyMapperTests
    {
        [Test]
        public void Level0_IsEasiestCorner()
        {
            var l = DifficultyMapper.ForLevel(0);
            Assert.AreEqual(2, l.AgentCount);
            Assert.AreEqual(6f, l.ArenaHalfSize, 1e-4f);
            Assert.AreEqual(0, l.MinObstacles);
            Assert.AreEqual(1, l.MaxObstacles);
        }

        [Test]
        public void Level3_IsHardestCorner()
        {
            var l = DifficultyMapper.ForLevel(3);
            Assert.AreEqual(8, l.AgentCount);
            Assert.AreEqual(11f, l.ArenaHalfSize, 1e-4f);
            Assert.AreEqual(6, l.MinObstacles);
            Assert.AreEqual(8, l.MaxObstacles);
        }

        [Test]
        public void ForLevel_ClampsBelowZero()
        {
            Assert.AreEqual(DifficultyMapper.ForLevel(0).AgentCount,
                            DifficultyMapper.ForLevel(-5).AgentCount);
        }

        [Test]
        public void ForLevel_ClampsAboveMax()
        {
            Assert.AreEqual(DifficultyMapper.ForLevel(3).ArenaHalfSize,
                            DifficultyMapper.ForLevel(99).ArenaHalfSize, 1e-4f);
        }

        [Test]
        public void EveryAxisIsMonotonicNonDecreasing()
        {
            for (int i = 1; i < DifficultyMapper.NumLevels; i++)
            {
                var prev = DifficultyMapper.ForLevel(i - 1);
                var cur = DifficultyMapper.ForLevel(i);
                Assert.GreaterOrEqual(cur.AgentCount, prev.AgentCount, $"agents L{i}");
                Assert.GreaterOrEqual(cur.ArenaHalfSize, prev.ArenaHalfSize, $"size L{i}");
                Assert.GreaterOrEqual(cur.MaxObstacles, prev.MaxObstacles, $"obs L{i}");
            }
        }

        [Test]
        public void MaxArenaDiagonal_MatchesLevel3()
        {
            float expected = 11f * 2f * Mathf.Sqrt(2f);
            Assert.AreEqual(expected, DifficultyMapper.MaxArenaDiagonal, 1e-3f);
        }

        [Test]
        public void HardestObstacleCount_FitsAgentPool()
        {
            // Pool is 8; hardest lesson must not ask for more obstacles than the pool.
            Assert.LessOrEqual(DifficultyMapper.ForLevel(3).MaxObstacles, 8);
        }

        // --- M4 search curriculum: ray length (goal visibility) is a first-class fading axis ---

        [Test]
        public void SearchLevel0_IsFullyVisible()
        {
            var l = DifficultyMapper.ForSearchLevel(0);
            Assert.AreEqual(2, l.AgentCount);
            Assert.AreEqual(8f, l.ArenaHalfSize, 1e-4f);
            Assert.AreEqual(DifficultyMapper.MaxArenaDiagonal, l.RayLength, 1e-3f); // goal visible everywhere
        }

        [Test]
        public void SearchLevel3_HidesTheGoal()
        {
            var l = DifficultyMapper.ForSearchLevel(3);
            Assert.AreEqual(8, l.AgentCount);
            Assert.AreEqual(11f, l.ArenaHalfSize, 1e-4f);
            // ray far shorter than the arena diagonal -> goal hidden across most of the arena
            Assert.Less(l.RayLength, DifficultyMapper.MaxArenaDiagonal * 0.5f);
        }

        [Test]
        public void SearchLadder_RayLengthMonotonicNonIncreasing()
        {
            for (int i = 1; i < DifficultyMapper.NumLevels; i++)
                Assert.LessOrEqual(DifficultyMapper.ForSearchLevel(i).RayLength,
                                   DifficultyMapper.ForSearchLevel(i - 1).RayLength, $"ray L{i}");
        }

        [Test]
        public void SearchLadder_OtherAxesMonotonicNonDecreasing()
        {
            for (int i = 1; i < DifficultyMapper.NumLevels; i++)
            {
                var prev = DifficultyMapper.ForSearchLevel(i - 1);
                var cur = DifficultyMapper.ForSearchLevel(i);
                Assert.GreaterOrEqual(cur.AgentCount, prev.AgentCount, $"agents L{i}");
                Assert.GreaterOrEqual(cur.ArenaHalfSize, prev.ArenaHalfSize, $"size L{i}");
                Assert.GreaterOrEqual(cur.MinObstacles, prev.MinObstacles, $"minObs L{i}");
                Assert.GreaterOrEqual(cur.MaxObstacles, prev.MaxObstacles, $"obs L{i}");
            }
        }

        [Test]
        public void ForLevel3_KeepsMaxDiagonalRay() // M3 rungs stay visible (reproducibility)
        {
            Assert.AreEqual(DifficultyMapper.MaxArenaDiagonal, DifficultyMapper.ForLevel(3).RayLength, 1e-3f);
        }

        // --- M5 terrain-difficulty ladder: structure ramps, fixed arena/sight ---

        [Test]
        public void Terrain_L0_IsOpenFlatWarmup()
        {
            var l = DifficultyMapper.ForTerrainLevel(0);
            Assert.AreEqual(0, l.Walls);
            Assert.AreEqual(0, l.Platforms);
            Assert.AreEqual(0, l.Pits);
            Assert.AreEqual(0, l.Movers);
            Assert.IsFalse(l.GoalElevated);
        }

        [Test]
        public void Terrain_L3_IsHiddenElevatedCrowdedWithPits()
        {
            var l = DifficultyMapper.ForTerrainLevel(3);
            Assert.Greater(l.Walls, 0);
            Assert.Greater(l.Platforms, 0);
            Assert.Greater(l.Pits, 0);
            Assert.AreEqual(4, l.Movers);
            Assert.IsTrue(l.GoalElevated);
        }

        [Test]
        public void Terrain_PitsFirstAppearAtL2()
        {
            Assert.AreEqual(0, DifficultyMapper.ForTerrainLevel(0).Pits);
            Assert.AreEqual(0, DifficultyMapper.ForTerrainLevel(1).Pits);
            Assert.Greater(DifficultyMapper.ForTerrainLevel(2).Pits, 0);
        }

        [Test]
        public void Terrain_MoversRampMonotonic()
        {
            int[] expected = { 0, 2, 3, 4 };
            for (int i = 0; i < DifficultyMapper.NumLevels; i++)
                Assert.AreEqual(expected[i], DifficultyMapper.ForTerrainLevel(i).Movers, $"movers L{i}");
        }

        [Test]
        public void Terrain_StructureMonotonicNonDecreasing()
        {
            for (int i = 1; i < DifficultyMapper.NumLevels; i++)
            {
                var prev = DifficultyMapper.ForTerrainLevel(i - 1);
                var cur = DifficultyMapper.ForTerrainLevel(i);
                Assert.GreaterOrEqual(cur.Walls, prev.Walls, $"walls L{i}");
                Assert.GreaterOrEqual(cur.Platforms, prev.Platforms, $"platforms L{i}");
                Assert.GreaterOrEqual(cur.Pits, prev.Pits, $"pits L{i}");
                Assert.GreaterOrEqual(cur.Movers, prev.Movers, $"movers L{i}");
            }
        }

        [Test]
        public void Terrain_FixedSightAndArena()
        {
            Assert.AreEqual(15f, DifficultyMapper.SightRange, 1e-4f);
            Assert.AreEqual(11f, DifficultyMapper.M5ArenaHalf, 1e-4f);
        }
    }
}
