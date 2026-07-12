using System.Collections.Generic;
using UnityEngine;

namespace NavSim.Runtime
{
    // Pure XZ-plane neighbor geometry. Unity-free math (uses Vector3 as a value type only).
    public static class CrowdMath
    {
        // Horizontal (XZ) distances from self to each other closer than maxRadius. Vertical (y) ignored.
        public static List<float> NeighborDistances(Vector3 self, IReadOnlyList<Vector3> others, float maxRadius)
        {
            var result = new List<float>();
            if (others == null) return result;
            for (int i = 0; i < others.Count; i++)
            {
                float dx = others[i].x - self.x;
                float dz = others[i].z - self.z;
                float d = Mathf.Sqrt(dx * dx + dz * dz);
                if (d < maxRadius) result.Add(d);
            }
            return result;
        }
    }
}
