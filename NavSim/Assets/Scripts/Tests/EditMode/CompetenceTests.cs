using NUnit.Framework;
using NavSim.Runtime;

namespace NavSim.Tests.EditMode
{
    public class CompetenceTests
    {
        [Test]
        public void Ramp01_ZeroSuccesses_IsZero()
            => Assert.AreEqual(0f, Competence.Ramp01(0, 200), 1e-6f);

        [Test]
        public void Ramp01_AtHorizon_IsOne()
            => Assert.AreEqual(1f, Competence.Ramp01(200, 200), 1e-6f);

        [Test]
        public void Ramp01_PastHorizon_ClampsToOne()
            => Assert.AreEqual(1f, Competence.Ramp01(999, 200), 1e-6f);

        [Test]
        public void RampValue_EvalMode_AlwaysHard()
            => Assert.AreEqual(50f, Competence.RampValue(0, 200, 5f, 50f, true), 1e-5f);

        [Test]
        public void RampValue_HalfHorizon_LerpsHalfway()
            => Assert.AreEqual(27.5f, Competence.RampValue(100, 200, 5f, 50f, false), 1e-4f);

        [Test]
        public void RampValue_HorizonZeroGuard_NoDivByZero()
            => Assert.AreEqual(50f, Competence.RampValue(1, 0, 5f, 50f, false), 1e-5f);
    }
}
