namespace NavSim.Runtime
{
    // Soften->harden decoy schedule (spec §7, advisor must-have #3). From scratch the agent hasn't learned the
    // cue->color->localize mapping, so instant-fail would kill episodes in a few steps with no reach-reward
    // (the M5 freeze in a new hat). L0 warmup = penalty-but-continue so the mapping can be learned while
    // episodes stay alive; L1+ = episode-ending failure once discrimination is up. Applied identically to all
    // arms (orthogonal to the perception lever). TRAINING ONLY — the eval harness detects reach-vs-decoy
    // geometrically and always applies hard semantics.
    public static class DecoyRules
    {
        public static bool DecoyEndsEpisode(int level) => level >= 1;
    }
}
