using NUnit.Framework;
using UnityEngine;
using NavSim.Runtime;

namespace NavSim.Tests.EditMode
{
    public class ObservationBuilderTests
    {
        [Test]
        public void Build_Length_Is8()
        {
            var obs = ObservationBuilder.Build(Vector3.zero, 0f, 4f, true, true, Color.black);
            Assert.AreEqual(8, obs.Length); // 5 proprioception + 3 RGB cue (M6)
        }

        [Test]
        public void Build_GroundedAndJumpReady_AsFlags()
        {
            var g = ObservationBuilder.Build(Vector3.zero, 0f, 4f, true, false, Color.black);
            Assert.AreEqual(1f, g[3], 1e-6f); // grounded
            Assert.AreEqual(0f, g[4], 1e-6f); // jumpReady
        }

        [Test]
        public void Build_VerticalVelocity_NormalizedByMaxSpeed()
        {
            // Falling registers on the local Y channel at velocity.y / maxSpeed (locks the Y scale factor).
            var obs = ObservationBuilder.Build(new Vector3(0f, -8f, 0f), 0f, 4f, false, false, Color.black);
            Assert.AreEqual(-2f, obs[1], 1e-4f); // -8 / maxSpeed(4)
        }

        [Test]
        public void Build_VerticalVelocity_InvariantUnderHeading()
        {
            // Yaw rotates only about Y, so the vertical channel must be unchanged by heading.
            var obs = ObservationBuilder.Build(new Vector3(0f, -8f, 0f), 90f, 4f, false, false, Color.black);
            Assert.AreEqual(-2f, obs[1], 1e-4f);
        }

        [Test]
        public void Build_ForwardVelocity_NormalizedToLocalZ()
        {
            // heading 0 == facing +Z; full-speed forward -> localVelZ == 1, no lateral component.
            var obs = ObservationBuilder.Build(new Vector3(0f, 0f, 4f), 0f, 4f, true, true, Color.black);
            Assert.AreEqual(1f, obs[2], 1e-4f); // localVelZ
            Assert.AreEqual(0f, obs[0], 1e-4f); // localVelX
        }

        [Test]
        public void Build_Velocity_RotatesIntoLocalFrame()
        {
            // heading 90 deg (facing +X); world +X velocity should read as local forward (+Z, index 2).
            var obs = ObservationBuilder.Build(new Vector3(4f, 0f, 0f), 90f, 4f, false, false, Color.black);
            Assert.AreEqual(1f, obs[2], 1e-3f); // localVelZ
            Assert.AreEqual(0f, obs[0], 1e-3f); // localVelX
        }

        // --- M6: the persistent RGB target-color cue ---

        [Test]
        public void Build_AppendsCueRgb_Length8()
        {
            var obs = ObservationBuilder.Build(Vector3.zero, 0f, 4f, true, true, new Color(0.2f, 0.4f, 0.6f));
            Assert.AreEqual(8, obs.Length);
            Assert.AreEqual(0.2f, obs[5], 1e-4f);
            Assert.AreEqual(0.4f, obs[6], 1e-4f);
            Assert.AreEqual(0.6f, obs[7], 1e-4f);
        }

        [Test]
        public void Build_CueDoesNotDisturbProprioception()
        {
            var a = ObservationBuilder.Build(new Vector3(1f, 2f, 3f), 90f, 4f, false, false, Color.red);
            var b = ObservationBuilder.Build(new Vector3(1f, 2f, 3f), 90f, 4f, false, false, Color.blue);
            for (int i = 0; i < 5; i++) Assert.AreEqual(a[i], b[i], 1e-5f, $"cue changed obs[{i}]");
        }
    }
}
