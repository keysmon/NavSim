using NUnit.Framework;
using UnityEngine;
using NavSim.Runtime;

namespace NavSim.Tests.EditMode
{
    public class ObservationBuilderTests
    {
        [Test]
        public void Build_ReturnsFiveElements()
        {
            float[] o = ObservationBuilder.Build(
                Vector3.zero, 0f, Vector3.zero, new Vector3(0f, 0f, 5f), 4f, 20f, 1f);
            Assert.AreEqual(5, o.Length);
        }

        [Test]
        public void GoalDirectlyAhead_HasPositiveForward()
        {
            // heading 0 deg == facing +Z. Goal on +Z should map to local forward (+Z), zero on right (X).
            float[] o = ObservationBuilder.Build(
                Vector3.zero, 0f, Vector3.zero, new Vector3(0f, 0f, 5f), 4f, 20f, 1f);
            Assert.Greater(o[3], 0.9f);       // goalDirLocalZ (forward)
            Assert.AreEqual(0f, o[2], 1e-4f); // goalDirLocalX (right)
        }

        [Test]
        public void CompassWeightZero_ZerosGoalDirection()
        {
            float[] o = ObservationBuilder.Build(
                Vector3.zero, 0f, Vector3.zero, new Vector3(0f, 0f, 5f), 4f, 20f, 0f);
            Assert.AreEqual(0f, o[2], 1e-6f);
            Assert.AreEqual(0f, o[3], 1e-6f);
        }

        [Test]
        public void Velocity_IsNormalizedByMaxSpeed()
        {
            float[] o = ObservationBuilder.Build(
                Vector3.zero, 0f, new Vector3(0f, 0f, 4f), new Vector3(0f, 0f, 5f), 4f, 20f, 1f);
            Assert.AreEqual(1f, o[1], 1e-4f); // localVelZ == maxSpeed/maxSpeed
        }
    }
}
