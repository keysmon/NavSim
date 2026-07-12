using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using NavSim.Runtime;

namespace NavSim.Tests.EditMode
{
    public class CrowdMathTests
    {
        [Test]
        public void FiltersOutNeighborsBeyondRadius()
        {
            var self = Vector3.zero;
            var others = new List<Vector3> { new Vector3(1f, 0f, 0f), new Vector3(5f, 0f, 0f) };
            var d = CrowdMath.NeighborDistances(self, others, 2f);
            Assert.AreEqual(1, d.Count);
            Assert.AreEqual(1f, d[0], 1e-4f);
        }

        [Test]
        public void IgnoresVerticalDifference()
        {
            var self = Vector3.zero;
            var others = new List<Vector3> { new Vector3(0f, 10f, 1f) }; // 10 up, 1 forward
            var d = CrowdMath.NeighborDistances(self, others, 2f);
            Assert.AreEqual(1, d.Count);
            Assert.AreEqual(1f, d[0], 1e-4f); // vertical 10 ignored
        }

        [Test]
        public void EmptyWhenNoOthers()
        {
            var d = CrowdMath.NeighborDistances(Vector3.zero, new List<Vector3>(), 2f);
            Assert.AreEqual(0, d.Count);
        }

        [Test]
        public void NullOthers_ReturnsEmpty()
        {
            var d = CrowdMath.NeighborDistances(Vector3.zero, null, 2f);
            Assert.AreEqual(0, d.Count);
        }

        [Test]
        public void NeighborAtExactlyMaxRadius_Excluded()
        {
            var self = Vector3.zero;
            var others = new List<Vector3> { new Vector3(2f, 0f, 0f) };
            var d = CrowdMath.NeighborDistances(self, others, 2f);
            Assert.AreEqual(0, d.Count);
        }
    }
}
