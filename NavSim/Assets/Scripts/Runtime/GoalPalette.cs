using UnityEngine;

namespace NavSim.Runtime
{
    // The pure core of M6's appearance task. A fixed palette of maximally-separated colors; a per-episode
    // draw of 3 DISTINCT colors + a target slot, the target slot drawn INDEPENDENTLY of the color indices so
    // the target's identity never correlates with a goal slot (a ray agent must not be able to learn
    // "target = slot 0"). Unity-Color-using but scene-free -> EditMode-testable.
    public static class GoalPalette
    {
        // 5 maximally-separated, flat-lit-friendly colors (primaries + 2). Kept saturated so shadow != hue-shift.
        public static readonly Color[] Colors =
        {
            new Color(0.85f, 0.10f, 0.10f), // red
            new Color(0.10f, 0.70f, 0.15f), // green
            new Color(0.15f, 0.30f, 0.90f), // blue
            new Color(0.90f, 0.80f, 0.10f), // yellow
            new Color(0.80f, 0.15f, 0.80f), // magenta
        };

        public static string Tag(int paletteIndex) => "goal_c" + paletteIndex;

        public struct GoalAssignment
        {
            public int[] ColorIndices; // length 3, distinct palette indices
            public int TargetSlot;     // 0..2, which of the 3 goals is the target
        }

        // Draw 3 distinct palette colors (partial Fisher-Yates) + an independent target slot.
        public static GoalAssignment Assign(System.Random rng)
        {
            int n = Colors.Length;
            int[] pool = new int[n];
            for (int i = 0; i < n; i++) pool[i] = i;
            for (int i = 0; i < 3; i++)
            {
                int j = i + rng.Next(n - i);
                (pool[i], pool[j]) = (pool[j], pool[i]);
            }
            return new GoalAssignment
            {
                ColorIndices = new[] { pool[0], pool[1], pool[2] },
                TargetSlot = rng.Next(3), // independent of which colors were drawn
            };
        }

        public static Color TargetColor(in GoalAssignment a) => Colors[a.ColorIndices[a.TargetSlot]];
    }
}
