using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using NavSim.Runtime;
using Random = System.Random; // disambiguate vs UnityEngine.Random

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
        public void Assign_TargetIsAlwaysTheFixedColor()
        {
            // M6 v2: the target is FIXED (red = Colors[TargetColorIndex]) every episode - no cue exists.
            for (int s = 0; s < 100; s++)
            {
                var a = GoalPalette.Assign(new Random(s));
                Assert.AreEqual(GoalPalette.TargetColorIndex, a.ColorIndices[a.TargetSlot],
                    $"seed {s}: target slot is not the fixed target color");
            }
        }

        [Test]
        public void Assign_DecoysAreDistinctAndNeverTheTargetColor()
        {
            for (int s = 0; s < 100; s++)
            {
                var a = GoalPalette.Assign(new Random(s));
                var decoys = new List<int>();
                for (int i = 0; i < 3; i++) if (i != a.TargetSlot) decoys.Add(a.ColorIndices[i]);
                Assert.AreEqual(2, decoys.Count, $"seed {s}");
                Assert.AreNotEqual(decoys[0], decoys[1], $"seed {s}: duplicate decoy colors");
                Assert.AreNotEqual(GoalPalette.TargetColorIndex, decoys[0], $"seed {s}: decoy 0 is the target color");
                Assert.AreNotEqual(GoalPalette.TargetColorIndex, decoys[1], $"seed {s}: decoy 1 is the target color");
            }
        }

        [Test]
        public void Assign_DecoyPairsVaryAcrossEpisodes()
        {
            // Decoys must be a fresh per-episode draw (C(4,2)=6 possible pairs), not a fixed pair.
            var pairs = new HashSet<string>();
            for (int s = 0; s < 200; s++)
            {
                var a = GoalPalette.Assign(new Random(s));
                var d = new List<int>();
                for (int i = 0; i < 3; i++) if (i != a.TargetSlot) d.Add(a.ColorIndices[i]);
                d.Sort();
                pairs.Add(d[0] + "," + d[1]);
            }
            Assert.GreaterOrEqual(pairs.Count, 4, "decoy colors look fixed, not per-episode random");
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
            // R1 decorrelation guard, statistically TIGHT: 5000 draws from ONE rng stream (sequentially
            // seeded fresh rngs can correlate), expected ~1667/slot (sd ~33). Bounds ~4 sd: flake-free,
            // yet tight enough to catch the classic biased naive shuffle (swap(i, rng.Next(3)) deviates
            // ~3.7 pp => ~185 counts, well outside), which the old 300-sample count>50 guard missed.
            var rng = new Random(1234);
            int[] counts = new int[3];
            for (int s = 0; s < 5000; s++) counts[GoalPalette.Assign(rng).TargetSlot]++;
            foreach (int c in counts)
            {
                Assert.Greater(c, 1533, "target-slot distribution biased (R1 leak risk)");
                Assert.Less(c, 1800, "target-slot distribution biased (R1 leak risk)");
            }
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
        public void TargetColor_IsAlwaysTheFixedRed()
        {
            for (int s = 0; s < 20; s++)
                Assert.AreEqual(GoalPalette.Colors[GoalPalette.TargetColorIndex],
                    GoalPalette.TargetColor(GoalPalette.Assign(new Random(s))));
        }
    }
}
