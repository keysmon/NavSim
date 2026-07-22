using NUnit.Framework;
using NavSim.Runtime;

namespace NavSim.Tests.EditMode
{
    public class RewardCalculatorTests
    {
        // --- M5 Step signature: shaping gated on a precomputed goalVisible; pit penalty added ---

        [Test]
        public void Step_ShapingApplied_WhenGoalVisible()
        {
            var cfg = RewardConfig.Default; // shapingScale 0.05, stepPenalty 0.001
            float r = RewardCalculator.Step(10f, 8f, false, cfg, goalVisible: true, fellInPit: false, jumped: false);
            Assert.AreEqual(2f * cfg.shapingScale - cfg.stepPenalty, r, 1e-5f); // moved 2 closer
        }

        [Test]
        public void Step_MovingAwayWhileVisible_YieldsNegative()
        {
            var cfg = RewardConfig.Default;
            Assert.Less(RewardCalculator.Step(8f, 10f, false, cfg, goalVisible: true, fellInPit: false, jumped: false), 0f);
        }

        [Test]
        public void Step_NoShaping_WhenGoalHidden()
        {
            var cfg = RewardConfig.Default;
            // Goal hidden -> distance gradient suppressed; only the step cost remains (curiosity + the
            // sparse bonus must drive search). prev>curr so a leaked gradient would show as positive.
            float r = RewardCalculator.Step(10f, 8f, false, cfg, goalVisible: false, fellInPit: false, jumped: false);
            Assert.AreEqual(-cfg.stepPenalty, r, 1e-6f);
        }

        [Test]
        public void Step_GoalBonus_OnReach()
        {
            var cfg = RewardConfig.Default;
            float r = RewardCalculator.Step(2f, 0.5f, true, cfg, goalVisible: true, fellInPit: false, jumped: false);
            Assert.Greater(r, cfg.goalBonus - 0.1f);
        }

        [Test]
        public void Step_GoalBonus_NotGatedByVisibility()
        {
            var cfg = RewardConfig.Default;
            // Goal HIDDEN so shaping is gated off; prev==curr isolates the bonus. Fails iff the reach
            // bonus is ever moved inside the visibility gate.
            float r = RewardCalculator.Step(5f, 5f, true, cfg, goalVisible: false, fellInPit: false, jumped: false);
            Assert.AreEqual(cfg.goalBonus - cfg.stepPenalty, r, 1e-6f); // 1.0 - 0.001
        }

        [Test]
        public void Step_PitPenalty_Subtracted()
        {
            var cfg = RewardConfig.Default;
            float clean = RewardCalculator.Step(10f, 10f, false, cfg, goalVisible: false, fellInPit: false, jumped: false);
            float fell = RewardCalculator.Step(10f, 10f, false, cfg, goalVisible: false, fellInPit: true, jumped: false);
            Assert.AreEqual(clean - cfg.pitPenalty, fell, 1e-6f);
        }

        [Test]
        public void Step_PitPenalty_NotGatedByVisibility()
        {
            var cfg = RewardConfig.Default;
            // Pit cost fires whether or not the goal is visible (it is a hazard, never gated).
            float visibleFell = RewardCalculator.Step(10f, 10f, false, cfg, goalVisible: true, fellInPit: true, jumped: false);
            float visibleClean = RewardCalculator.Step(10f, 10f, false, cfg, goalVisible: true, fellInPit: false, jumped: false);
            Assert.AreEqual(visibleClean - cfg.pitPenalty, visibleFell, 1e-6f);
        }

        [Test]
        public void Step_ShapingSuppressed_OnPitFall()
        {
            var cfg = RewardConfig.Default;
            // Even a would-be-positive gradient (prev>curr) while visible must earn NO shaping during a fall:
            // the falling position is garbage and the -pitPenalty is the sole intended signal (no double-dip).
            float r = RewardCalculator.Step(10f, 6f, false, cfg, goalVisible: true, fellInPit: true, jumped: false);
            Assert.AreEqual(-cfg.stepPenalty - cfg.pitPenalty, r, 1e-6f);
        }

        // --- Crowd penalty (unchanged from M2/M4) ---

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
            // neighbors must net > 0, or the crowd learns avoidance-over-navigation. (Movers keep this term
            // in M5; the probe measures the REAL median per-step progress — if it is materially below this,
            // lower collision/congestion WEIGHTS, not this constant.)
            var cfg = RewardConfig.Default;
            float representativeProgress = 0.3f;
            float step = RewardCalculator.Step(
                10f, 10f - representativeProgress, false, cfg, goalVisible: true, fellInPit: false, jumped: false);
            float crowd = RewardCalculator.CrowdPenalty(
                new System.Collections.Generic.List<float> { 1.8f, 2.0f }, cfg); // 2 congestion-range neighbors
            Assert.Greater(step - crowd, 0f);
        }

        [Test]
        public void Default_HasDecoyPenalty()
        {
            Assert.AreEqual(0.25f, RewardConfig.Default.decoyPenalty, 1e-6f);
        }

        // --- Showcase: flat jumpPenalty (pure reward layer, shaping gate untouched) ---

        [Test]
        public void Step_JumpPenalty_ChargedFlat_OnJump()
        {
            var cfg = RewardConfig.Default; // jumpPenalty 0.02
            float noJump = RewardCalculator.Step(10f, 8f, false, cfg, goalVisible: true, fellInPit: false, jumped: false);
            float jump = RewardCalculator.Step(10f, 8f, false, cfg, goalVisible: true, fellInPit: false, jumped: true);
            Assert.AreEqual(noJump - cfg.jumpPenalty, jump, 1e-6f); // flat subtraction, nothing else changes
        }

        [Test]
        public void Step_JumpPenalty_DoesNotTouchShaping_WhenGoalHidden()
        {
            var cfg = RewardConfig.Default;
            float r = RewardCalculator.Step(10f, 8f, false, cfg, goalVisible: false, fellInPit: false, jumped: true);
            Assert.AreEqual(-cfg.stepPenalty - cfg.jumpPenalty, r, 1e-6f); // no shaping leak, penalty still flat
        }

        [Test]
        public void Step_JumpPenalty_StacksWith_GoalBonus()
        {
            var cfg = RewardConfig.Default;
            float r = RewardCalculator.Step(2f, 0.5f, true, cfg, goalVisible: true, fellInPit: false, jumped: true);
            float rNoJump = RewardCalculator.Step(2f, 0.5f, true, cfg, goalVisible: true, fellInPit: false, jumped: false);
            Assert.AreEqual(rNoJump - cfg.jumpPenalty, r, 1e-6f);
        }

        [Test]
        public void Default_HasJumpPenalty()
        {
            Assert.AreEqual(0.02f, RewardConfig.Default.jumpPenalty, 1e-6f);
        }
    }
}
