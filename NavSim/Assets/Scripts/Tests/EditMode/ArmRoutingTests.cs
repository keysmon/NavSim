using NUnit.Framework;
using NavSim.Runtime;

namespace NavSim.Tests.EditMode
{
    public class ArmRoutingTests
    {
        [Test]
        public void Selfish_OutcomePaysScorerOnly()
        {
            var s = ArmRouting.Outcome(ArmRouting.Arm.Selfish);
            Assert.AreEqual(1f, s.scorer); Assert.AreEqual(0f, s.partner); Assert.AreEqual(0f, s.group);
        }

        [Test]
        public void Shared_OutcomeCopiesToBothAgents_NoGroup()
        {
            var s = ArmRouting.Outcome(ArmRouting.Arm.Shared);
            Assert.AreEqual(1f, s.scorer); Assert.AreEqual(1f, s.partner); Assert.AreEqual(0f, s.group);
        }

        [Test]
        public void Poca_OutcomeIsGroupOnly()
        {
            var s = ArmRouting.Outcome(ArmRouting.Arm.Poca);
            Assert.AreEqual(0f, s.scorer); Assert.AreEqual(0f, s.partner); Assert.AreEqual(1f, s.group);
        }

        [Test]
        public void PerStep_TimeCost_RoutesLikeTheArm()
        {
            var selfish = ArmRouting.PerStep(ArmRouting.Arm.Selfish, -0.001f);
            Assert.AreEqual(-0.001f, selfish.scorer, 1e-9f); Assert.AreEqual(-0.001f, selfish.partner, 1e-9f); Assert.AreEqual(0f, selfish.group);
            var shared = ArmRouting.PerStep(ArmRouting.Arm.Shared, -0.001f);
            Assert.AreEqual(-0.001f, shared.scorer, 1e-9f); Assert.AreEqual(-0.001f, shared.partner, 1e-9f); Assert.AreEqual(0f, shared.group);
            var poca = ArmRouting.PerStep(ArmRouting.Arm.Poca, -0.001f);
            Assert.AreEqual(0f, poca.scorer); Assert.AreEqual(0f, poca.partner); Assert.AreEqual(-0.001f, poca.group, 1e-9f);
        }

        [Test]
        public void ArmEnumValues_AreTheArmModeEncoding()
        {
            // Load-bearing: these ints ARE the arm_mode env-param encoding CoopArena decodes.
            Assert.AreEqual(0, (int)ArmRouting.Arm.Selfish);
            Assert.AreEqual(1, (int)ArmRouting.Arm.Shared);
            Assert.AreEqual(2, (int)ArmRouting.Arm.Poca);
        }

        [Test]
        public void NoOtherRewardExists()
        {
            // The spec's "nothing else, ever": the type surface is Outcome + PerStep only.
            Assert.AreEqual(2, typeof(ArmRouting).GetMethods(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static).Length);
        }
    }
}
