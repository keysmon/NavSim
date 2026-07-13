using NUnit.Framework;
using UnityEngine;
using NavSim.Runtime;

namespace NavSim.Tests.EditMode
{
    public class ObservationBuilderTests
    {
        [Test]
        public void Build_Length_Is5()
        {
            var obs = ObservationBuilder.Build(Vector3.zero, 0f, 4f, true, true);
            Assert.AreEqual(5, obs.Length);
        }

        [Test]
        public void Build_GroundedAndJumpReady_AsFlags()
        {
            var g = ObservationBuilder.Build(Vector3.zero, 0f, 4f, true, false);
            Assert.AreEqual(1f, g[3], 1e-6f); // grounded
            Assert.AreEqual(0f, g[4], 1e-6f); // jumpReady
        }

        [Test]
        public void Build_VerticalVelocity_NormalizedByMaxSpeed()
        {
            // Falling registers on the local Y channel at velocity.y / maxSpeed (locks the Y scale factor).
            var obs = ObservationBuilder.Build(new Vector3(0f, -8f, 0f), 0f, 4f, false, false);
            Assert.AreEqual(-2f, obs[1], 1e-4f); // -8 / maxSpeed(4)
        }

        [Test]
        public void Build_VerticalVelocity_InvariantUnderHeading()
        {
            // Yaw rotates only about Y, so the vertical channel must be unchanged by heading.
            var obs = ObservationBuilder.Build(new Vector3(0f, -8f, 0f), 90f, 4f, false, false);
            Assert.AreEqual(-2f, obs[1], 1e-4f);
        }

        [Test]
        public void Build_ForwardVelocity_NormalizedToLocalZ()
        {
            // heading 0 == facing +Z; full-speed forward -> localVelZ == 1, no lateral component.
            var obs = ObservationBuilder.Build(new Vector3(0f, 0f, 4f), 0f, 4f, true, true);
            Assert.AreEqual(1f, obs[2], 1e-4f); // localVelZ
            Assert.AreEqual(0f, obs[0], 1e-4f); // localVelX
        }

        [Test]
        public void Build_Velocity_RotatesIntoLocalFrame()
        {
            // heading 90 deg (facing +X); world +X velocity should read as local forward (+Z, index 2).
            var obs = ObservationBuilder.Build(new Vector3(4f, 0f, 0f), 90f, 4f, false, false);
            Assert.AreEqual(1f, obs[2], 1e-3f); // localVelZ
            Assert.AreEqual(0f, obs[0], 1e-3f); // localVelX
        }
    }
}
