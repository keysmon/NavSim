using NavSim.Runtime;
using NUnit.Framework;
using UnityEngine;

namespace NavSim.Tests.EditMode
{
    public class RampExpertLogicTests
    {
        private static readonly Vector3 Ramp = new Vector3(-1.5f, 1.24f, 2f);
        private static readonly Vector3 Goal = new Vector3(0f, 2.5f, 5f);

        [TestCase(0, 1.75f)]
        [TestCase(1, 2.5f)]
        [TestCase(2, 3.5f)]
        [TestCase(3, 5.0f)]
        [TestCase(4, 1.75f)]
        public void StartDistance_InterleavesFourRungs(int episode, float expected)
            => Assert.AreEqual(expected, RampExpertLogic.StartDistance(episode, true), 1e-5f);

        [TestCase(0, 1.75f)]
        [TestCase(9, 1.75f)]
        [TestCase(10, 2.5f)]
        [TestCase(20, 3.5f)]
        [TestCase(30, 5.0f)]
        public void StartDistance_DryRunGroupsTenAttemptsPerRung(int episode, float expected)
            => Assert.AreEqual(expected, RampExpertLogic.StartDistance(episode, false), 1e-5f);

        [Test]
        public void Decide_PushFacesPositiveXWithoutJump()
        {
            var input = new RampExpertInput(
                RampExpertState.Push, 0, new Vector3(-5f, .5f, 2f), 90f,
                new Vector3(-1.75f, 1.24f, 2f), Goal, false);

            RampExpertStep step = RampExpertLogic.Decide(input);

            Assert.Greater(step.Forward, 0.9f);
            Assert.AreEqual(0f, step.Turn, 1e-5f);
            Assert.AreEqual(0, step.Jump);
            Assert.AreEqual(RampExpertState.Push, step.State);
        }

        [Test]
        public void Decide_PlacementTransitionsToClearWest()
        {
            var input = new RampExpertInput(
                RampExpertState.Push, 10, new Vector3(-4f, .5f, 2f), 90f,
                Ramp, Goal, true);

            Assert.AreEqual(RampExpertState.ClearWest, RampExpertLogic.Decide(input).State);
        }

        [Test]
        public void Decide_ClearWestTransitionsToMoveSouthInsideTolerance()
        {
            var input = new RampExpertInput(
                RampExpertState.ClearWest, 20, new Vector3(-4.5f, .5f, 2f), 90f,
                Ramp, Goal, true);

            Assert.AreEqual(RampExpertState.MoveSouth, RampExpertLogic.Decide(input).State);
        }

        [Test]
        public void Decide_MoveSouthTransitionsToCenterSlopeInsideTolerance()
        {
            var input = new RampExpertInput(
                RampExpertState.MoveSouth, 20, new Vector3(-4.5f, .5f, -.75f), 90f,
                Ramp, Goal, true);

            Assert.AreEqual(RampExpertState.CenterSlope, RampExpertLogic.Decide(input).State);
        }

        [Test]
        public void Decide_CenterSlopeTransitionsToClimbInsideTolerance()
        {
            var input = new RampExpertInput(
                RampExpertState.CenterSlope, 20, new Vector3(-1.5f, .5f, -.75f), 90f,
                Ramp, Goal, true);

            Assert.AreEqual(RampExpertState.Climb, RampExpertLogic.Decide(input).State);
        }

        [Test]
        public void Decide_PositiveHeadingErrorTurnsPositiveAndRotatesInPlace()
        {
            var input = new RampExpertInput(
                RampExpertState.ClearWest, 0, new Vector3(-4.5f, .5f, 0f), -90f,
                Ramp, Goal, true);

            RampExpertStep step = RampExpertLogic.Decide(input);

            Assert.AreEqual(0f, step.Forward, 1e-5f);
            Assert.Greater(step.Turn, 0f);
        }

        [Test]
        public void Decide_NegativeHeadingErrorTurnsNegativeAndRotatesInPlace()
        {
            var input = new RampExpertInput(
                RampExpertState.ClearWest, 0, new Vector3(-4.5f, .5f, 0f), 90f,
                Ramp, Goal, true);

            RampExpertStep step = RampExpertLogic.Decide(input);

            Assert.AreEqual(0f, step.Forward, 1e-5f);
            Assert.Less(step.Turn, 0f);
        }

        [Test]
        public void Decide_OverBudgetMarksTimedOutAndStops()
        {
            var state = RampExpertState.MoveSouth;
            var input = new RampExpertInput(
                state, RampExpertLogic.StepBudget(state) + 1, Vector3.zero, 0f,
                Ramp, Goal, true);

            RampExpertStep step = RampExpertLogic.Decide(input);

            Assert.IsTrue(step.TimedOut);
            Assert.AreEqual(0f, step.Forward);
            Assert.AreEqual(0f, step.Turn);
            Assert.AreEqual(0, step.Jump);
        }

        [Test]
        public void Decide_AlwaysClampsActionsToUnitRange()
        {
            foreach (RampExpertState state in System.Enum.GetValues(typeof(RampExpertState)))
            {
                RampExpertStep step = RampExpertLogic.Decide(new RampExpertInput(
                    state, 0, new Vector3(100f, .5f, -100f), -720f, Ramp, Goal, false));

                Assert.That(step.Forward, Is.InRange(-1f, 1f));
                Assert.That(step.Turn, Is.InRange(-1f, 1f));
            }
        }
    }
}
