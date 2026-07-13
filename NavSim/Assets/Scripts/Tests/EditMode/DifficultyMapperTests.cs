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
    }
}
