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
}
