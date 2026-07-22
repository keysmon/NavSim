using System.Collections.Generic;
using UnityEngine;

namespace NavSim.Runtime
{
    public enum CourseVariant { SafeLoop = 0, NoLoop = 1 }

    public struct CoursePiece
    {
        public string Name;
        public PrimitiveType Primitive;
        public Vector3 Pos;
        public Vector3 Scale;
        public Quaternion Rot;
        public string MaterialKey;
        public string Tag;
    }

    public struct CourseLayout
    {
        public int Stage; public bool Mirrored; public CourseVariant Variant;
        public CoursePiece[] Pieces;
        public Vector3[] GoalSlots;          // length 3, CO-PLANAR (equal y)
        public Vector3 SpawnPos; public float SpawnYawDeg;
        public Vector3 PitRespawnPos;        // strictly BEFORE the pit in path order
        public float KillPlaneY;             // CourseSpec.KillY
        public float RampZMin, RampZMax;     // 0-width when no ramp (caption heuristic)
        public float GapZMin, GapZMax;       // 0-width when no pit
        public Vector3 CameraPos, CameraLookAt; // per-stage hero pose, x-negated under mirror
    }

    // Pure 5-stage course geometry - the single source of truth for the showcase spine (training
    // scene, demo, WebGL re-rolls). Reuses the ramp/tread math and visual grammar of
    // Editor/CapstoneSceneSetup.cs as DATA (no scene/GameObject calls, EditMode-testable) so Tasks
    // 4/5/6/10 can build the same course identically. +Z is forward; course width X -6..6.
    //
    // Every stage shares the same start floor + ramp (Z 0..12, rising to DeckY) and the same
    // SpawnPos - stages diverge only past the ramp. Stage 4 "The gauntlet" always forces NoLoop
    // (a full-width pit with no safe ledge) regardless of the requested variant.
    public static class CourseSpec
    {
        public const int NumStages = 5;
        public const float DeckY = 2.8f;
        public const float GapLength = 2.2f;
        public const float KillY = -1.5f;
        public const float PitFloorTopY = -4.5f;
        public const float GoalHoverY = 1.0f;
        public const float RingRadius = 2.6f;
        public static readonly string[] StageNames = { "Open room", "The ramp", "The wall", "The gap", "The gauntlet" };

        // Shared across every stage: the agent always starts on the same flat slab, at the same
        // pose, and the ramp it climbs (Z 6..12, rising to DeckY) never changes shape or position.
        private static readonly Vector3 SpawnPos = new Vector3(0f, 1.01f, 0f);
        private static readonly Vector3 RampStart = new Vector3(0f, 0f, 6f);
        private static readonly Vector3 RampEnd = new Vector3(0f, DeckY, 12f);

        public static CourseLayout Build(int stage, bool mirrored, CourseVariant variant)
        {
            if (stage == 4) variant = CourseVariant.NoLoop; // Stage 4 is ALWAYS NoLoop, requested variant ignored

            CourseLayout layout;
            switch (stage)
            {
                case 0: layout = Stage0(); break;
                case 1: layout = Stage1(); break;
                case 2: layout = Stage2(); break;
                case 3: layout = Stage3(variant); break;
                case 4: layout = Stage4(); break;
                default: throw new System.ArgumentOutOfRangeException(nameof(stage), stage, "CourseSpec has stages 0..4");
            }
            layout.Stage = stage;
            layout.Variant = variant;
            layout.Mirrored = mirrored;
            layout.KillPlaneY = KillY;
            return mirrored ? Mirror(layout) : layout;
        }

        // === Stage 0: "Open room" - flat start, an immediately-visible 3-slot reveal ring. ===
        private static CourseLayout Stage0()
        {
            Vector3[] slots = RevealSlots(1.0f, 10f);
            var pieces = new List<CoursePiece>
            {
                Cube("Floor", new Vector3(0f, -0.15f, 6f), new Vector3(12f, 0.3f, 20f), "floor"),
                SpawnMarker(SpawnPos),
            };
            pieces.AddRange(DecoyPads(slots, 0f)); // floor top = 0 here (no ramp/deck yet)
            return new CourseLayout
            {
                Pieces = pieces.ToArray(),
                GoalSlots = slots,
                SpawnPos = SpawnPos,
                SpawnYawDeg = 0f,
                PitRespawnPos = SpawnPos, // no pit this stage
                RampZMin = 0f, RampZMax = 0f, // no ramp (zero-width)
                GapZMin = 0f, GapZMax = 0f,   // no gap (zero-width)
                CameraPos = new Vector3(11f, 8f, -5f),
                CameraLookAt = new Vector3(0f, 0.5f, 7f),
            };
        }

        // === Stage 1: "The ramp" - climb to the raised deck, reveal ring right at the top. ===
        private static CourseLayout Stage1()
        {
            Vector3[] slots = RevealSlots(3.8f, 21f);
            var pieces = new List<CoursePiece> { StartFloor() };
            pieces.AddRange(Ramp());
            pieces.AddRange(Deck("Deck1", 19f, 14f, "landing")); // touches the ramp's Z=12 landing
            pieces.Add(SpawnMarker(SpawnPos));
            pieces.AddRange(DecoyPads(slots, DeckY));
            return new CourseLayout
            {
                Pieces = pieces.ToArray(),
                GoalSlots = slots,
                SpawnPos = SpawnPos,
                SpawnYawDeg = 0f,
                PitRespawnPos = SpawnPos, // no pit this stage
                RampZMin = 6f, RampZMax = 12f,
                GapZMin = 0f, GapZMax = 0f, // no gap
                CameraPos = new Vector3(14f, 10f, -6f),
                CameraLookAt = new Vector3(0f, 1.8f, 11f),
            };
        }

        // === Stage 2: "The wall" - as Stage 1, but a longer deck with a centerline occluder that
        // forces a detour through the open right flank; slots spread wide (no longer a tight ring). ===
        private static CourseLayout Stage2()
        {
            Vector3[] slots =
            {
                new Vector3(-4.5f, 3.8f, 33f),
                new Vector3(0f, 3.8f, 37f),
                new Vector3(4.5f, 3.8f, 35f),
            };
            var pieces = new List<CoursePiece> { StartFloor() };
            pieces.AddRange(Ramp());
            pieces.AddRange(Deck("Deck1", 26f, 28f, "landing"));
            pieces.Add(WallPiece("Wall", new Vector3(-0.5f, 4.55f, 26f), new Vector3(4f, 3.5f, 0.8f)));
            pieces.Add(SpawnMarker(SpawnPos));
            pieces.AddRange(DecoyPads(slots, DeckY));
            return new CourseLayout
            {
                Pieces = pieces.ToArray(),
                GoalSlots = slots,
                SpawnPos = SpawnPos,
                SpawnYawDeg = 0f,
                PitRespawnPos = SpawnPos, // no pit this stage
                RampZMin = 6f, RampZMax = 12f,
                GapZMin = 0f, GapZMax = 0f, // no gap
                CameraPos = new Vector3(20f, 13f, -7f),
                CameraLookAt = new Vector3(0f, 2.5f, 19f),
            };
        }

        // === Stage 3: "The gap" - wall, then a jump pit; SafeLoop variant adds a no-jump ledge
        // detour, NoLoop is a full-width pit (jump is the only way across). ===
        private static CourseLayout Stage3(CourseVariant variant)
        {
            Vector3[] slots =
            {
                new Vector3(-4.5f, 3.8f, 37f),
                new Vector3(0f, 3.8f, 41f),
                new Vector3(4.5f, 3.8f, 39f),
            };
            var pieces = new List<CoursePiece> { StartFloor() };
            pieces.AddRange(Ramp());
            pieces.AddRange(Deck("DeckA", 20f, 16f, "landing"));
            pieces.Add(WallPiece("Wall", new Vector3(-0.5f, 4.55f, 24f), new Vector3(4f, 3.5f, 0.8f)));
            pieces.AddRange(PitPieces(variant, 28f));
            pieces.AddRange(Deck("DeckB", 37.1f, 13.8f, "floor"));
            pieces.Add(SpawnMarker(SpawnPos));
            pieces.AddRange(DecoyPads(slots, DeckY));
            return new CourseLayout
            {
                Pieces = pieces.ToArray(),
                GoalSlots = slots,
                SpawnPos = SpawnPos,
                SpawnYawDeg = 0f,
                PitRespawnPos = new Vector3(0f, DeckY + 1.01f, 26.5f), // strictly before GapZMin=28, on DeckA
                RampZMin = 6f, RampZMax = 12f,
                GapZMin = 28f, GapZMax = 30.2f,
                CameraPos = new Vector3(26f, 15f, -8f),
                CameraLookAt = new Vector3(0f, 2.5f, 23f),
            };
        }

        // === Stage 4: "The gauntlet" - two staggered walls (opposite open flanks force an S-turn),
        // then the gap, ALWAYS full-width NoLoop (forced in Build()). ===
        private static CourseLayout Stage4()
        {
            Vector3[] slots =
            {
                new Vector3(-4.5f, 3.8f, 39f),
                new Vector3(0f, 3.8f, 43f),
                new Vector3(4.5f, 3.8f, 41f),
            };
            var pieces = new List<CoursePiece> { StartFloor() };
            pieces.AddRange(Ramp());
            pieces.AddRange(Deck("DeckA", 21f, 18f, "landing"));
            pieces.Add(WallPiece("WallA", new Vector3(-2.25f, 4.55f, 20f), new Vector3(7.5f, 3.5f, 0.8f))); // blocks X -6..1.5, open right
            pieces.Add(WallPiece("WallB", new Vector3(2.25f, 4.55f, 26f), new Vector3(7.5f, 3.5f, 0.8f)));  // blocks X -1.5..6, open left
            pieces.AddRange(PitPieces(CourseVariant.NoLoop, 30f)); // Stage-3-NoLoop pieces, shifted +2 in Z
            pieces.AddRange(Deck("DeckB", 39.1f, 13.8f, "floor"));
            pieces.Add(SpawnMarker(SpawnPos));
            pieces.AddRange(DecoyPads(slots, DeckY));
            return new CourseLayout
            {
                Pieces = pieces.ToArray(),
                GoalSlots = slots,
                SpawnPos = SpawnPos,
                SpawnYawDeg = 0f,
                PitRespawnPos = new Vector3(0f, DeckY + 1.01f, 28.5f), // strictly before GapZMin=30, on DeckA
                RampZMin = 6f, RampZMax = 12f,
                GapZMin = 30f, GapZMax = 32.2f,
                CameraPos = new Vector3(30f, 17f, -9f),
                CameraLookAt = new Vector3(0f, 2.5f, 26f),
            };
        }

        // --- shared piece helpers ---

        private static CoursePiece P(string name, PrimitiveType prim, Vector3 pos, Vector3 scale, Quaternion rot, string matKey, string tag)
            => new CoursePiece { Name = name, Primitive = prim, Pos = pos, Scale = scale, Rot = rot, MaterialKey = matKey, Tag = tag };

        private static CoursePiece Cube(string name, Vector3 pos, Vector3 scale, string matKey, string tag = null)
            => P(name, PrimitiveType.Cube, pos, scale, Quaternion.identity, matKey, tag);

        // The flat slab every stage starts on (top always at y=0, so SpawnPos.y=1.01 sits flush on it).
        private static CoursePiece StartFloor()
            => Cube("StartFloor", new Vector3(0f, -0.15f, 1f), new Vector3(12f, 0.3f, 10f), "floor");

        // Small capsule marking the spawn point, offset just off the direct forward sightline
        // (same idiom as CapstoneSceneSetup's hand-placed SpawnMarker). Rests on the floor: the
        // floor top under any SpawnPos is always SpawnPos.y - 1.01 (the agent-center convention
        // shared with PitRespawnPos), and a 0.5-scale capsule's half-height is 0.5.
        private static CoursePiece SpawnMarker(Vector3 spawnPos)
            => P("SpawnMarker", PrimitiveType.Capsule,
                new Vector3(spawnPos.x + 0.8f, spawnPos.y - 0.51f, spawnPos.z),
                new Vector3(0.5f, 0.5f, 0.5f), Quaternion.identity, "spawnMarker", null);

        // Low, purely-visual pads centered under each goal slot at the local walkable top (deckTopY):
        // floor top=0 pre-ramp (Stage 0), DeckY=2.8 everywhere past it.
        private static CoursePiece[] DecoyPads(Vector3[] slots, float deckTopY)
        {
            var pads = new CoursePiece[slots.Length];
            for (int i = 0; i < slots.Length; i++)
                pads[i] = Cube("DecoyPad" + i, new Vector3(slots[i].x, deckTopY + 0.1f, slots[i].z),
                    new Vector3(1.4f, 0.4f, 1.4f), "decoyPad");
            return pads;
        }

        // The 3-slot reveal ring used by Stages 0-1: radius RingRadius at angles 90/210/330 deg
        // around (0, y, zBase), reduced to the closed-form offsets (0,+2.6), (-2.25,-1.3), (2.25,-1.3).
        private static Vector3[] RevealSlots(float y, float zBase) => new[]
        {
            new Vector3(0f, y, zBase + RingRadius),
            new Vector3(-2.25f, y, zBase - 1.3f),
            new Vector3(2.25f, y, zBase - 1.3f),
        };

        // Ramp (Z 6..12, rising to DeckY) + 4 tread stripes, identical for every stage 1-4 - same
        // math as CapstoneSceneSetup.MakeRamp/MakeTreadStripes: FromToRotation(forward, dir) gives a
        // pure pitch-about-X rotation (dir has zero X component), so stripes sit flush on the slope.
        private static CoursePiece[] Ramp()
        {
            Vector3 dir = RampEnd - RampStart;
            Quaternion rot = Quaternion.FromToRotation(Vector3.forward, dir.normalized);
            Vector3 mid = (RampStart + RampEnd) / 2f;
            const float width = 9f, thickness = 0.6f;

            var pieces = new CoursePiece[5]; // ramp slab + 4 tread stripes
            pieces[0] = P("Ramp", PrimitiveType.Cube, mid, new Vector3(width, thickness, dir.magnitude), rot, "ramp", null);

            Vector3 slopeUp = rot * Vector3.up;
            for (int i = 1; i <= 4; i++)
            {
                float t = i / 5f;
                Vector3 centerline = Vector3.Lerp(RampStart, RampEnd, t);
                Vector3 pos = centerline + slopeUp * (thickness / 2f + 0.05f);
                pieces[i] = P("Tread" + i, PrimitiveType.Cube, pos, new Vector3(width, 0.08f, 0.35f), rot, "tread", null);
            }
            return pieces;
        }

        // A flat deck slab plus its matching left/right curbs. Every elevated deck shares the same
        // top height (DeckY) and curb rail height (DeckY + 0.25), so only Z-center/length vary.
        private static CoursePiece[] Deck(string name, float centerZ, float length, string matKey)
        {
            float deckCenterY = DeckY - 0.15f;
            float curbY = DeckY + 0.25f;
            return new[]
            {
                Cube(name, new Vector3(0f, deckCenterY, centerZ), new Vector3(12f, 0.3f, length), matKey),
                Cube(name + "_CurbL", new Vector3(-6.15f, curbY, centerZ), new Vector3(0.3f, 0.5f, length), "curb"),
                Cube(name + "_CurbR", new Vector3(6.15f, curbY, centerZ), new Vector3(0.3f, 0.5f, length), "curb"),
            };
        }

        private static CoursePiece WallPiece(string name, Vector3 pos, Vector3 scale)
            => Cube(name, pos, scale, "wall", "wall");

        // Pit floor + near/far side walls + a jump-off curb lip at zMin, sized/positioned by variant:
        // SafeLoop is an 8-wide pit (X -6..2) with a 4-wide safe ledge alongside (X 2..6); NoLoop is
        // a full 12-wide pit (X -6..6) with no ledge - the jump is the only way across. zMin is the
        // gap's near edge (Stage 3: 28; Stage 4: 30, i.e. Stage 3's NoLoop geometry shifted +2 in Z).
        private static CoursePiece[] PitPieces(CourseVariant variant, float zMin)
        {
            float zMax = zMin + GapLength;
            float zMid = zMin + GapLength / 2f;
            float floorY = PitFloorTopY - 0.15f;
            float wallY = (DeckY + PitFloorTopY) / 2f;
            float wallH = DeckY - PitFloorTopY;
            float width = variant == CourseVariant.SafeLoop ? 8f : 12f;
            float xCenter = variant == CourseVariant.SafeLoop ? -2f : 0f;

            var pieces = new List<CoursePiece>
            {
                Cube("PitFloor", new Vector3(xCenter, floorY, zMid), new Vector3(width, 0.3f, GapLength), "pit"),
                Cube("PitWall_Near", new Vector3(xCenter, wallY, zMin), new Vector3(width, wallH, 0.3f), "pit"),
                Cube("PitWall_Far", new Vector3(xCenter, wallY, zMax), new Vector3(width, wallH, 0.3f), "pit"),
                Cube("Curb_PitLip", new Vector3(xCenter, DeckY + 0.15f, zMin - 0.15f), new Vector3(width, 0.3f, 0.3f), "curb"),
            };
            if (variant == CourseVariant.SafeLoop)
            {
                pieces.Add(Cube("SafeLedge", new Vector3(4f, DeckY - 0.15f, zMid), new Vector3(4f, 0.3f, GapLength), "floor"));
                pieces.Add(Cube("Curb_SafeLedgeRail", new Vector3(1.85f, DeckY + 0.25f, zMid), new Vector3(0.3f, 0.5f, GapLength), "curb"));
            }
            return pieces.ToArray();
        }

        // Negates X on every piece/slot/anchor/camera; rotations here are only ever a pitch about
        // the world X axis (the ramp), so Rot and Scale are unchanged under mirroring.
        private static CourseLayout Mirror(CourseLayout layout)
        {
            var pieces = new CoursePiece[layout.Pieces.Length];
            for (int i = 0; i < pieces.Length; i++)
            {
                CoursePiece piece = layout.Pieces[i];
                piece.Pos.x = -piece.Pos.x;
                pieces[i] = piece;
            }
            var slots = new Vector3[layout.GoalSlots.Length];
            for (int i = 0; i < slots.Length; i++)
            {
                Vector3 slot = layout.GoalSlots[i];
                slot.x = -slot.x;
                slots[i] = slot;
            }
            layout.Pieces = pieces;
            layout.GoalSlots = slots;
            layout.SpawnPos = new Vector3(-layout.SpawnPos.x, layout.SpawnPos.y, layout.SpawnPos.z);
            layout.PitRespawnPos = new Vector3(-layout.PitRespawnPos.x, layout.PitRespawnPos.y, layout.PitRespawnPos.z);
            layout.CameraPos = new Vector3(-layout.CameraPos.x, layout.CameraPos.y, layout.CameraPos.z);
            layout.CameraLookAt = new Vector3(-layout.CameraLookAt.x, layout.CameraLookAt.y, layout.CameraLookAt.z);
            return layout;
        }
    }
}
