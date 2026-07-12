using NUnit.Framework;
using UnityEngine;
using NavSim.Runtime;

namespace NavSim.Tests.EditMode
{
    public class ObservationBuilderTests
    {
        [Test]
        public void Length_IsTwoPlusNumColors()
        {
            float[] o = ObservationBuilder.Build(Vector3.zero, 0f, 4f, 0, 8);
            Assert.AreEqual(10, o.Length); // 2 velocity + 8 one-hot
        }

        [Test]
        public void OneHot_SetsOnlyMyColorIndex()
        {
            float[] o = ObservationBuilder.Build(Vector3.zero, 0f, 4f, 3, 8);
            Assert.AreEqual(1f, o[2 + 3], 1e-6f);
            for (int c = 0; c < 8; c++)
                if (c != 3) Assert.AreEqual(0f, o[2 + c], 1e-6f);
        }

        [Test]
        public void Velocity_IsNormalizedByMaxSpeed_InLocalFrame()
        {
            // heading 0 == facing +Z; world +Z velocity maps to local forward (index 1).
            float[] o = ObservationBuilder.Build(new Vector3(0f, 0f, 4f), 0f, 4f, 0, 8);
            Assert.AreEqual(1f, o[1], 1e-4f); // localVelZ == maxSpeed/maxSpeed
            Assert.AreEqual(0f, o[0], 1e-4f); // localVelX
        }

        [Test]
        public void Velocity_RotatesIntoLocalFrame()
        {
            // heading 90 deg (facing +X); world +X velocity should read as local forward (+Z, index 1).
            float[] o = ObservationBuilder.Build(new Vector3(4f, 0f, 0f), 90f, 4f, 0, 8);
            Assert.AreEqual(1f, o[1], 1e-3f);
            Assert.AreEqual(0f, o[0], 1e-3f);
        }
    }
}
