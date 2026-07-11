using NUnit.Framework;
using NavSim.Runtime;

namespace NavSim.Tests.EditMode
{
    public class RewardCalculatorTests
    {
        [Test]
        public void ProgressTowardGoal_YieldsPositiveShaping()
        {
            var cfg = RewardConfig.Default; // shapingScale 0.05, stepPenalty 0.001
            float r = RewardCalculator.Step(10f, 8f, false, cfg);
            Assert.AreEqual(2f * 0.05f - 0.001f, r, 1e-5f);
        }

        [Test]
        public void MovingAway_YieldsNegativeReward()
        {
            var cfg = RewardConfig.Default;
            Assert.Less(RewardCalculator.Step(8f, 10f, false, cfg), 0f);
        }

        [Test]
        public void ReachingGoal_AddsBonus()
        {
            var cfg = RewardConfig.Default;
            float r = RewardCalculator.Step(1f, 0f, true, cfg);
            Assert.Greater(r, cfg.goalBonus - 0.1f);
        }

        [Test]
        public void CompassWeightZero_RemovesShaping()
        {
            var cfg = RewardConfig.Default;
            cfg.compassWeight = 0f;
            float r = RewardCalculator.Step(10f, 5f, false, cfg);
            Assert.AreEqual(-cfg.stepPenalty, r, 1e-6f);
        }
    }
}
