namespace NavSim.Runtime
{
    // Pure door state for the M7 sacrifice task (spec sec 2/5). The door is open while the plate is
    // occupied OR within dwellSeconds of last occupancy (the per-lesson grace period: C1=4s, C2=2s,
    // C3=1s - the knob that makes plate-tap-and-sprint viable only at easy geometry). alwaysOpen is
    // the C0 lesson mode. CoopArena holds the secondsSinceVacated float and calls Step each
    // FixedUpdate; scene-free -> EditMode-testable.
    public static class PlateDoor
    {
        public const float InitialSecondsSinceVacated = 1e6f; // fresh episode: door closed for any dwell

        public static float Step(float secondsSinceVacated, bool plateOccupied, float dt)
            => plateOccupied ? 0f : secondsSinceVacated + dt;

        public static bool IsOpen(float secondsSinceVacated, bool plateOccupied, float dwellSeconds, bool alwaysOpen)
            => alwaysOpen || plateOccupied || secondsSinceVacated < dwellSeconds;
    }
}
