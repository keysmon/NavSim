using NUnit.Framework;
using NavSim.Runtime;

namespace NavSim.Tests.EditMode
{
    public class DecoyRulesTests
    {
        [Test] public void L0_IsSoft_PenaltyButContinue()  => Assert.IsFalse(DecoyRules.DecoyEndsEpisode(0));
        [Test] public void L1_IsHard_EndsEpisode()          => Assert.IsTrue(DecoyRules.DecoyEndsEpisode(1));
        [Test] public void L2_IsHard()                      => Assert.IsTrue(DecoyRules.DecoyEndsEpisode(2));
        [Test] public void L3_IsHard()                      => Assert.IsTrue(DecoyRules.DecoyEndsEpisode(3));

        [Test]
        public void HardeningIsMonotonic_OnceHardStaysHard()
        {
            bool seenHard = false;
            for (int l = 0; l < 4; l++)
            {
                bool hard = DecoyRules.DecoyEndsEpisode(l);
                if (seenHard) Assert.IsTrue(hard, $"L{l} softened after hardening");
                seenHard |= hard;
            }
        }
    }
}
