using NUnit.Framework;
using NavSim.Runtime;
using UnityEngine;

namespace NavSim.Tests.EditMode
{
    public class CourseSpecTests
    {
        private static CourseLayout L(int s, bool m = false, CourseVariant v = CourseVariant.SafeLoop)
            => CourseSpec.Build(s, m, v);

        [Test] public void FiveStages_Named()
        {
            Assert.AreEqual(5, CourseSpec.NumStages);
            Assert.AreEqual(5, CourseSpec.StageNames.Length);
        }

        [Test] public void GoalSlots_AreCoPlanar_AndSpaced([NUnit.Framework.Range(0, 4)] int stage)
        {
            var lay = L(stage);
            Assert.AreEqual(3, lay.GoalSlots.Length);
            Assert.AreEqual(lay.GoalSlots[0].y, lay.GoalSlots[1].y, 1e-4f);
            Assert.AreEqual(lay.GoalSlots[0].y, lay.GoalSlots[2].y, 1e-4f);
            for (int i = 0; i < 3; i++)
                for (int j = i + 1; j < 3; j++)
                    Assert.Greater(Vector3.Distance(lay.GoalSlots[i], lay.GoalSlots[j]), 3.0f); // > 2*goalRadius
        }

        [Test] public void EveryGap_IsJumpable_WithMargin([Values(3, 4)] int stage,
            [Values(CourseVariant.SafeLoop, CourseVariant.NoLoop)] CourseVariant v)
        {
            var lay = L(stage, false, v);
            float gap = lay.GapZMax - lay.GapZMin;
            Assert.Greater(gap, 0f);
            Assert.LessOrEqual(gap, 2.4f);
            float maxJump = LocomotionMath.MaxJumpDistance(7f, -20f, 0.02f, 4f, -30f);
            Assert.Greater(maxJump, gap * 1.15f); // 15% margin (advisor T2)
        }

        [Test] public void Mirror_XNegates_Everything_IncludingCamera([NUnit.Framework.Range(0, 4)] int stage)
        {
            var a = L(stage); var b = L(stage, true);
            Assert.AreEqual(-a.SpawnPos.x, b.SpawnPos.x, 1e-4f);
            Assert.AreEqual(-a.CameraPos.x, b.CameraPos.x, 1e-4f);
            Assert.AreEqual(-a.CameraLookAt.x, b.CameraLookAt.x, 1e-4f);
            Assert.AreEqual(-a.PitRespawnPos.x, b.PitRespawnPos.x, 1e-4f);
            Assert.AreEqual(a.Pieces.Length, b.Pieces.Length);
            for (int i = 0; i < a.Pieces.Length; i++)
            {
                Assert.AreEqual(-a.Pieces[i].Pos.x, b.Pieces[i].Pos.x, 1e-4f);
                Assert.AreEqual(a.Pieces[i].Pos.z, b.Pieces[i].Pos.z, 1e-4f);
                Assert.AreEqual(a.Pieces[i].Scale, b.Pieces[i].Scale);
            }
            for (int i = 0; i < 3; i++) Assert.AreEqual(-a.GoalSlots[i].x, b.GoalSlots[i].x, 1e-4f);
        }

        [Test] public void KillPlane_BetweenPitFloor_AndWalkables()
        {
            Assert.Greater(CourseSpec.KillY, CourseSpec.PitFloorTopY);
            Assert.Less(CourseSpec.KillY, 0f); // stage-0 floor top (lowest walkable) is y=0
        }

        [Test] public void Stage3_Variants_DifferOnlyInLoop()
        {
            var safe = L(3, false, CourseVariant.SafeLoop);
            var none = L(3, false, CourseVariant.NoLoop);
            Assert.IsTrue(System.Array.Exists(safe.Pieces, p => p.Name == "SafeLedge"));
            Assert.IsFalse(System.Array.Exists(none.Pieces, p => p.Name == "SafeLedge"));
        }

        [Test] public void Stage4_Always_NoLoop_Geometry()
        {
            var lay = L(4, false, CourseVariant.SafeLoop); // requested SafeLoop must be ignored
            Assert.IsFalse(System.Array.Exists(lay.Pieces, p => p.Name == "SafeLedge"));
            Assert.AreEqual(CourseVariant.NoLoop, lay.Variant);
        }

        [Test] public void EarlyStages_RingCoVisible_FromSpawnHeading([Values(0, 1)] int stage)
        {
            var lay = L(stage);
            foreach (var slot in lay.GoalSlots)
            {
                Vector3 to = slot - lay.SpawnPos; to.y = 0f;
                float ang = Vector3.Angle(Vector3.forward, to);
                Assert.Less(ang, 40f); // inside the 90-deg camera's half-FOV with margin (advisor T5)
            }
        }

        [Test] public void PitRespawn_StrictlyBeforeGap([Values(3, 4)] int stage)
        {
            var lay = L(stage, false, CourseVariant.NoLoop);
            Assert.Less(lay.PitRespawnPos.z, lay.GapZMin);
            Assert.AreEqual(CourseSpec.DeckY + 1.01f, lay.PitRespawnPos.y, 0.05f);
        }

        [Test] public void Walls_Tagged_ForOcclusion([Values(2, 3, 4)] int stage)
        {
            var lay = L(stage);
            Assert.IsTrue(System.Array.Exists(lay.Pieces, p => p.Tag == "wall"));
        }
    }
}
