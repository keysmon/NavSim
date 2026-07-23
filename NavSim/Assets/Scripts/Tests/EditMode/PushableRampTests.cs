using NUnit.Framework;
using UnityEngine;
using NavSim.Runtime;

namespace NavSim.Tests.EditMode
{
    public class PushableRampTests
    {
        // Tuning under test (mirror the scene/arena/PhysicMaterial values, finalized by verify-early #1):
        // per-agent push 10 N, friction coeff 0.15, gravity 9.81. Light mass 2 kg (S0/S1), heavy mass 6 kg (S2).
        const float Push = 10f, Mu = 0.15f, G = 9.81f, Light = 2f, Heavy = 6f;

        [Test]
        public void NetPush_SumsHorizontal_ZeroesY()
        {
            Vector3 n = PushableRamp.NetPush(new Vector3(3f, 5f, 0f), new Vector3(1f, -9f, 2f));
            Assert.AreEqual(4f, n.x, 1e-5f);
            Assert.AreEqual(0f, n.y, 1e-5f);
            Assert.AreEqual(2f, n.z, 1e-5f);
        }

        [Test]
        public void LightRamp_OneAgent_BreaksFreeAndMoves()
        {
            // S0/S1: a lone agent CAN move the light ramp usefully (pushing bootstraps). 10 > 0.15*2*9.81=2.94
            Assert.IsTrue(PushableRamp.Overcomes(Push, Light, Mu, G));
            Assert.Greater(PushableRamp.NetForce(Push, Light, Mu, G), 5f);   // strong propulsion
        }

        [Test]
        public void HeavyRamp_OneAgent_NotHardStuck_GradientPreserved()
        {
            // S2: a lone agent still BARELY breaks static friction (creeps -> nonzero shaping), it is just
            // too SLOW to finish in budget. This continuity is what avoids M7's C2 zero-gradient trap. 10 > 0.15*6*9.81=8.83
            Assert.IsTrue(PushableRamp.Overcomes(Push, Heavy, Mu, G));
            Assert.Greater(PushableRamp.NetForce(Push, Heavy, Mu, G), 0f);   // nonzero, but small
        }

        [Test]
        public void HeavyRamp_TwoAgents_MuchMoreNetForceThanOne()
        {
            // The forced-cooperation speed gap: two pushers' propulsion far exceeds one's (the friction floor
            // eats most of a single push). 20-8.83=11.17 vs 10-8.83=1.17 -> ~9.5x.
            float one = PushableRamp.NetForce(Push, Heavy, Mu, G);
            float two = PushableRamp.NetForce(2f * Push, Heavy, Mu, G);
            Assert.Greater(two, 3f * one);
        }
    }
}
