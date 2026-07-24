using NavSim.Runtime;
using NUnit.Framework;

namespace NavSim.Tests.EditMode
{
    public class RampCurriculumTests
    {
        private const int Horizon = 200;
        private const float TargetX = 0f;
        private const float NormalStartX = -5f;
        private const float InitialPushDistance = 1.75f;

        [Test]
        public void S0StartX_ZeroSuccesses_RequiresShortPush()
            => Assert.AreEqual(
                -1.75f,
                RampCurriculum.S0StartX(0, Horizon, TargetX, NormalStartX, InitialPushDistance, false),
                1e-5f);

        [Test]
        public void S0StartX_HalfHorizon_IsHalfwayToNormalStart()
            => Assert.AreEqual(
                -3.375f,
                RampCurriculum.S0StartX(100, Horizon, TargetX, NormalStartX, InitialPushDistance, false),
                1e-5f);

        [Test]
        public void S0StartX_AtHorizon_UsesNormalStart()
            => Assert.AreEqual(
                NormalStartX,
                RampCurriculum.S0StartX(Horizon, Horizon, TargetX, NormalStartX, InitialPushDistance, false),
                1e-5f);

        [Test]
        public void S0StartX_EvalMode_UsesNormalStartImmediately()
            => Assert.AreEqual(
                NormalStartX,
                RampCurriculum.S0StartX(0, Horizon, TargetX, NormalStartX, InitialPushDistance, true),
                1e-5f);
    }
}
