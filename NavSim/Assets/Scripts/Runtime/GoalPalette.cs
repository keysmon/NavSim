using UnityEngine;

namespace NavSim.Runtime
{
    // The pure core of M6's appearance task (v2, FIXED-TARGET). The target is a FIXED palette colour
    // (red = Colors[TargetColorIndex]) every episode - there is NO per-episode cue (the cued design's
    // cross-modal vector-cue<->pixel-colour binding was unlearnable at a local budget; see the M6 v2 spec).
    // Decoys are 2 random DISTINCT non-target colours per episode, and the colour->slot assignment is
    // SHUFFLED so red's slot is uniform BY CONSTRUCTION (spec R1: slot 0 is the scene-template goal and
    // last-resort placement is slot-ordered, so a fixed red slot would correlate red with placement
    // artifacts and could leak to a ray agent). Unity-Color-using but scene-free -> EditMode-testable.
    public static class GoalPalette
    {
        // 5 maximally-separated, flat-lit-friendly colors (primaries + 2). Kept saturated so shadow != hue-shift.
        public static readonly Color[] Colors =
        {
            new Color(0.85f, 0.10f, 0.10f), // red  <- the FIXED target (TargetColorIndex)
            new Color(0.10f, 0.70f, 0.15f), // green
            new Color(0.15f, 0.30f, 0.90f), // blue
            new Color(0.90f, 0.80f, 0.10f), // yellow
            new Color(0.80f, 0.15f, 0.80f), // magenta
        };

        // The fixed target colour, every episode. rayC's target tag is therefore always Tag(TargetColorIndex).
        public const int TargetColorIndex = 0;

        public static string Tag(int paletteIndex) => "goal_c" + paletteIndex;

        public struct GoalAssignment
        {
            public int[] ColorIndices; // length 3, distinct palette indices; exactly one == TargetColorIndex
            public int TargetSlot;     // 0..2, the slot that holds the fixed target colour
        }

        // Draw 2 distinct decoy colours from the non-target palette entries, then shuffle
        // {target, decoy1, decoy2} uniformly across the 3 slots (R1). TargetSlot = wherever the target landed.
        public static GoalAssignment Assign(System.Random rng)
        {
            int n = Colors.Length;
            int[] pool = new int[n - 1];                       // the 4 non-target palette indices
            for (int i = 1; i < n; i++) pool[i - 1] = i;
            for (int i = 0; i < 2; i++)                        // partial Fisher-Yates: 2 distinct decoys
            {
                int j = i + rng.Next(pool.Length - i);
                (pool[i], pool[j]) = (pool[j], pool[i]);
            }
            int[] slots = { TargetColorIndex, pool[0], pool[1] };
            for (int i = 0; i < 3; i++)                        // full Fisher-Yates: colour->slot shuffle
            {
                int j = i + rng.Next(3 - i);
                (slots[i], slots[j]) = (slots[j], slots[i]);
            }
            return new GoalAssignment
            {
                ColorIndices = slots,
                TargetSlot = System.Array.IndexOf(slots, TargetColorIndex),
            };
        }

        public static Color TargetColor(in GoalAssignment a) => Colors[a.ColorIndices[a.TargetSlot]];
    }
}
