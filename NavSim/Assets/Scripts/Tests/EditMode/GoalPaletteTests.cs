using System;
using NUnit.Framework;
using UnityEngine;
using NavSim.Runtime;
using Random = System.Random; // disambiguate: UnityEngine.Random is also in scope via `using UnityEngine;`

namespace NavSim.Tests.EditMode
{
    public class GoalPaletteTests
    {
        [Test]
        public void Palette_HasFiveDistinctColors()
        {
            Assert.AreEqual(5, GoalPalette.Colors.Length);
            for (int i = 0; i < GoalPalette.Colors.Length; i++)
                for (int j = i + 1; j < GoalPalette.Colors.Length; j++)
                {
                    // maximally-separated: no two palette colors within 0.5 L1 distance in RGB
                    Color a = GoalPalette.Colors[i], b = GoalPalette.Colors[j];
                    float d = Mathf.Abs(a.r - b.r) + Mathf.Abs(a.g - b.g) + Mathf.Abs(a.b - b.b);
                    Assert.Greater(d, 0.5f, $"colors {i},{j} too close");
                }
        }

        [Test]
        public void Assign_PicksThreeDistinctColors()
        {
            var a = GoalPalette.Assign(new Random(1));
            Assert.AreEqual(3, a.ColorIndices.Length);
            Assert.AreNotEqual(a.ColorIndices[0], a.ColorIndices[1]);
            Assert.AreNotEqual(a.ColorIndices[0], a.ColorIndices[2]);
            Assert.AreNotEqual(a.ColorIndices[1], a.ColorIndices[2]);
        }

        [Test]
        public void Assign_TargetSlotInRange()
        {
            for (int s = 0; s < 50; s++)
            {
                var a = GoalPalette.Assign(new Random(s));
                Assert.GreaterOrEqual(a.TargetSlot, 0);
                Assert.LessOrEqual(a.TargetSlot, 2);
            }
        }

        [Test]
        public void Assign_TargetSlotIsUniformlyDistributed_NotBiasedToOneSlot()
        {
            // decorrelation guard: target slot must not be fixed (else "target = slot 0" leaks to a ray agent)
            int[] counts = new int[3];
            for (int s = 0; s < 300; s++) counts[GoalPalette.Assign(new Random(s)).TargetSlot]++;
            foreach (int c in counts) Assert.Greater(c, 50, "a slot is (near) never the target — biased");
        }

        [Test]
        public void Assign_IsDeterministicForASeed()
        {
            var a = GoalPalette.Assign(new Random(42));
            var b = GoalPalette.Assign(new Random(42));
            CollectionAssert.AreEqual(a.ColorIndices, b.ColorIndices);
            Assert.AreEqual(a.TargetSlot, b.TargetSlot);
        }

        [Test]
        public void Tag_MapsPaletteIndexToPerColorTag()
        {
            Assert.AreEqual("goal_c0", GoalPalette.Tag(0));
            Assert.AreEqual("goal_c4", GoalPalette.Tag(4));
        }

        [Test]
        public void TargetColor_IsTheColorAtTheTargetSlot()
        {
            var a = GoalPalette.Assign(new Random(7));
            Assert.AreEqual(GoalPalette.Colors[a.ColorIndices[a.TargetSlot]], GoalPalette.TargetColor(a));
        }
    }
}
