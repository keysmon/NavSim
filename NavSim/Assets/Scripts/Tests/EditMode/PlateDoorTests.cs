using NUnit.Framework;
using NavSim.Runtime;

namespace NavSim.Tests.EditMode
{
    public class PlateDoorTests
    {
        [Test]
        public void Occupied_ResetsVacatedClock_AndOpens()
        {
            float t = PlateDoor.Step(99f, true, 0.02f);
            Assert.AreEqual(0f, t, 1e-6f);
            Assert.IsTrue(PlateDoor.IsOpen(t, true, 1.0f, false));
        }

        [Test]
        public void Vacated_AccumulatesTime()
        {
            float t = PlateDoor.Step(0f, false, 0.02f);
            Assert.AreEqual(0.02f, t, 1e-6f);
            t = PlateDoor.Step(t, false, 0.03f);
            Assert.AreEqual(0.05f, t, 1e-6f);
        }

        [Test]
        public void Open_WithinDwell_ClosedAfter()
        {
            // dwell 2.0: open at 1.99 s since vacated, closed at 2.01 s (dwell is a grace period, not a latch)
            Assert.IsTrue(PlateDoor.IsOpen(1.99f, false, 2.0f, false));
            Assert.IsFalse(PlateDoor.IsOpen(2.01f, false, 2.0f, false));
        }

        [Test]
        public void AlwaysOpen_OverridesEverything()
        {
            // C0 lesson mode: open regardless of plate or clock
            Assert.IsTrue(PlateDoor.IsOpen(999f, false, 0f, true));
        }

        [Test]
        public void FreshEpisode_StartsClosed()
        {
            // InitialSecondsSinceVacated must exceed any dwell we ever use
            Assert.IsFalse(PlateDoor.IsOpen(PlateDoor.InitialSecondsSinceVacated, false, 4.0f, false));
        }
    }
}
