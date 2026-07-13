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
            float r = RewardCalculator.Step(10f, 8f, false, cfg, float.MaxValue);
            Assert.AreEqual(2f * 0.05f - 0.001f, r, 1e-5f);
        }

        [Test]
        public void MovingAway_YieldsNegativeReward()
        {
            var cfg = RewardConfig.Default;
            Assert.Less(RewardCalculator.Step(8f, 10f, false, cfg, float.MaxValue), 0f);
        }

        [Test]
        public void ReachingGoal_AddsBonus()
        {
            var cfg = RewardConfig.Default;
            float r = RewardCalculator.Step(1f, 0f, true, cfg, float.MaxValue);
            Assert.Greater(r, cfg.goalBonus - 0.1f);
        }

        [Test]
        public void CrowdPenalty_ZeroWhenNoNeighbors()
        {
            var cfg = RewardConfig.Default;
            Assert.AreEqual(0f, RewardCalculator.CrowdPenalty(new System.Collections.Generic.List<float>(), cfg), 1e-6f);
        }

        [Test]
        public void CrowdPenalty_GrowsAsNeighborGetsCloser()
        {
            var cfg = RewardConfig.Default;
            float far = RewardCalculator.CrowdPenalty(new System.Collections.Generic.List<float> { 1.0f }, cfg);
            float near = RewardCalculator.CrowdPenalty(new System.Collections.Generic.List<float> { 0.2f }, cfg);
            Assert.Greater(near, far);
        }

        [Test]
        public void CrowdPenalty_CongestionOnlyOutsideCollisionRadius()
        {
            var cfg = RewardConfig.Default; // collisionRadius 1.2, congestionRadius 2.5
            // neighbor at 1.8: outside collision, inside congestion -> exactly congestionWeight
            float p = RewardCalculator.CrowdPenalty(new System.Collections.Generic.List<float> { 1.8f }, cfg);
            Assert.AreEqual(cfg.congestionWeight, p, 1e-6f);
        }

        [Test]
        public void CrowdPenalty_NullList_ReturnsZero()
        {
            var cfg = RewardConfig.Default;
            Assert.AreEqual(0f, RewardCalculator.CrowdPenalty(null, cfg), 1e-6f);
        }

        [Test]
        public void CrowdPenalty_NeighborAtExactlyCongestionRadius_Excluded()
        {
            var cfg = RewardConfig.Default; // congestionRadius 2.5, collisionRadius 1.2
            float p = RewardCalculator.CrowdPenalty(new System.Collections.Generic.List<float> { 2.5f }, cfg);
            Assert.AreEqual(0f, p, 1e-6f);
        }

        [Test]
        public void ProgressThroughCrowd_NetsPositive()
        {
            // §6b balance constraint: a representative per-step progress while passing 2 congestion-range
            // neighbors must net > 0, or the crowd learns avoidance-over-navigation.
            var cfg = RewardConfig.Default;
            // Representative per-decision-step progress. Task 6's probe measures the REAL median
            // per-step progress; if it is materially below this, lower collision/congestion WEIGHTS
            // (not this constant) so the sub-dominance guarantee holds at the real progress.
            float representativeProgress = 0.3f;
            float step = RewardCalculator.Step(10f, 10f - representativeProgress, false, cfg, float.MaxValue);
            float crowd = RewardCalculator.CrowdPenalty(
                new System.Collections.Generic.List<float> { 1.8f, 2.0f }, cfg); // 2 congestion-range neighbors
            Assert.Greater(step - crowd, 0f);
        }

        // --- M4 visibility gate: privileged distance shaping applies only when the goal is in ray range ---

        [Test]
        public void HiddenGoal_NoShaping()
        {
            var cfg = RewardConfig.Default;
            // curr 18 >= visibilityRadius 10 -> goal hidden -> shaping suppressed, only step penalty
            float r = RewardCalculator.Step(20f, 18f, false, cfg, 10f);
            Assert.AreEqual(-cfg.stepPenalty, r, 1e-6f);
        }

        [Test]
        public void VisibleGoal_KeepsShaping()
        {
            var cfg = RewardConfig.Default;
            // curr 6 < visibilityRadius 10 -> visible -> full shaping
            float r = RewardCalculator.Step(8f, 6f, false, cfg, 10f);
            Assert.AreEqual(cfg.shapingScale * 2f - cfg.stepPenalty, r, 1e-5f);
        }

        [Test]
        public void ReachedGoal_BonusNotGatedByVisibility()
        {
            var cfg = RewardConfig.Default;
            // reaching implies currDist ~0 < any radius; bonus always applies
            float r = RewardCalculator.Step(1f, 0f, true, cfg, 0.5f);
            Assert.Greater(r, cfg.goalBonus - 0.1f);
        }
    }
}
