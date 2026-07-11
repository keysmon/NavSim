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
                Vector2.zero, 0f, Vector2.zero, new Vector2(5f, 0f), 4f, 20f, 1f);
            Assert.AreEqual(5, o.Length);
        }

        [Test]
        public void GoalDirectlyAhead_HasPositiveLocalX()
        {
            // heading 0 deg == facing +X (world). Goal on +X should map to local +X.
            float[] o = ObservationBuilder.Build(
                Vector2.zero, 0f, Vector2.zero, new Vector2(5f, 0f), 4f, 20f, 1f);
            Assert.Greater(o[2], 0.9f);       // goalDirLocalX
            Assert.AreEqual(0f, o[3], 1e-4f); // goalDirLocalY
        }

        [Test]
        public void CompassWeightZero_ZerosGoalDirection()
        {
            float[] o = ObservationBuilder.Build(
                Vector2.zero, 0f, Vector2.zero, new Vector2(5f, 0f), 4f, 20f, 0f);
            Assert.AreEqual(0f, o[2], 1e-6f);
            Assert.AreEqual(0f, o[3], 1e-6f);
        }

        [Test]
        public void Velocity_IsNormalizedByMaxSpeed()
        {
            float[] o = ObservationBuilder.Build(
                Vector2.zero, 0f, new Vector2(4f, 0f), new Vector2(5f, 0f), 4f, 20f, 1f);
            Assert.AreEqual(1f, o[0], 1e-4f); // localVelX == maxSpeed/maxSpeed
        }
    }
}
