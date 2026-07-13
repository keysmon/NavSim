using NUnit.Framework;
using NavSim.Runtime;

public class VisibilityGateTests
{
    [Test]
    public void Visible_WhenInRangeAndLineClear()
        => Assert.IsTrue(VisibilityGate.IsGoalVisible(10f, 15f, true));

    [Test]
    public void Hidden_WhenOutOfRange_EvenIfLineClear()
        => Assert.IsFalse(VisibilityGate.IsGoalVisible(20f, 15f, true));

    [Test]
    public void Hidden_WhenLineBlocked_EvenIfInRange()
        => Assert.IsFalse(VisibilityGate.IsGoalVisible(5f, 15f, false));

    [Test]
    public void Boundary_AtExactRange_IsHidden() // strict less-than
        => Assert.IsFalse(VisibilityGate.IsGoalVisible(15f, 15f, true));
}
