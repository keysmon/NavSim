using NUnit.Framework;
using NavSim.Runtime;

public class LocomotionMathTests
{
    const float G = -20f, DT = 0.02f, TERM = -30f, JUMP = 7f;

    [Test]
    public void Grounded_NoJump_RestsAtSmallNegative() // stays pinned to ground, not accumulating
    {
        float v = LocomotionMath.NextVerticalVelocity(-5f, true, false, JUMP, G, DT, TERM);
        Assert.AreEqual(-1f, v, 1e-4f);
    }

    [Test]
    public void Grounded_Jump_LaunchesUp()
    {
        float v = LocomotionMath.NextVerticalVelocity(0f, true, true, JUMP, G, DT, TERM);
        Assert.AreEqual(JUMP, v, 1e-4f);
    }

    [Test]
    public void Airborne_Jump_Ignored() // can't double-jump
    {
        float v = LocomotionMath.NextVerticalVelocity(3f, false, true, JUMP, G, DT, TERM);
        Assert.Less(v, 3f); // gravity applied, not re-launched
    }

    [Test]
    public void Airborne_AccumulatesGravity()
    {
        float v = LocomotionMath.NextVerticalVelocity(0f, false, false, JUMP, G, DT, TERM);
        Assert.AreEqual(G * DT, v, 1e-4f);
    }

    [Test]
    public void Airborne_ClampedToTerminalVelocity()
    {
        float v = LocomotionMath.NextVerticalVelocity(-29.9f, false, false, JUMP, G, DT, TERM);
        Assert.AreEqual(TERM, v, 1e-4f);
    }

    [Test]
    public void FellInPit_TrueBelowKillPlane()
    {
        Assert.IsTrue(LocomotionMath.FellInPit(-6f, -5f));
        Assert.IsFalse(LocomotionMath.FellInPit(0.5f, -5f));
        Assert.IsFalse(LocomotionMath.FellInPit(-5f, -5f)); // exact plane = NOT fallen (locks strict <)
    }

    [Test]
    public void MaxJumpDistance_MatchesDiscreteIntegration_ForProjectConstants()
    {
        // jumpImpulse 7, gravity -20, dt 0.02, maxSpeed 4 -> closed form ~2.8u; discrete within (2.6, 3.0)
        float d = LocomotionMath.MaxJumpDistance(7f, -20f, 0.02f, 4f, -30f);
        Assert.Greater(d, 2.6f);
        Assert.Less(d, 3.0f);
    }

    [Test]
    public void MaxJumpDistance_Monotonic_InImpulse()
    {
        Assert.Greater(LocomotionMath.MaxJumpDistance(9f, -20f, 0.02f, 4f, -30f),
                       LocomotionMath.MaxJumpDistance(7f, -20f, 0.02f, 4f, -30f));
    }
}
