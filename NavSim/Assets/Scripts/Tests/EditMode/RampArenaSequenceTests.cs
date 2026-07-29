using System.Reflection;
using NavSim.Runtime;
using NUnit.Framework;
using UnityEngine;

namespace NavSim.Tests.EditMode
{
    public class RampArenaSequenceTests
    {
        private const BindingFlags PrivateInstance =
            BindingFlags.Instance | BindingFlags.NonPublic;

        [Test]
        public void GoalInsideRadius_WithoutRampPlacement_DoesNotScore()
        {
            var arenaGo = new GameObject("Arena");
            var agentGo = new GameObject("Agent");
            var rampGo = new GameObject("Ramp");
            var goalGo = new GameObject("Goal");
            var targetGo = new GameObject("RampTarget");

            try
            {
                RampArena arena = arenaGo.AddComponent<RampArena>();
                RampAgent agent = agentGo.AddComponent<RampAgent>();
                PushableRampBody ramp = rampGo.AddComponent<PushableRampBody>();

                // This is the scene's real anti-leak boundary: a grounded transform starts
                // at y=0.5 and a 7 m/s jump rises about 1.225 units. The resulting
                // position is inside the 1.5-unit goal radius without reaching the ledge.
                agentGo.transform.position = new Vector3(0f, 1.725f, 3.9f);
                goalGo.transform.position = new Vector3(0f, 2.5f, 5f);
                targetGo.transform.position = new Vector3(10f, 1.24f, 2f);
                ramp.ResetTo(new Vector3(0f, 1.24f, 2f), Quaternion.identity);

                Assert.Less(
                    Vector3.Distance(agentGo.transform.position, goalGo.transform.position),
                    1.5f,
                    "Reproduction must remain inside the scene's goal radius.");

                SetField(arena, "agents", new[] { agent });
                SetField(arena, "goal", goalGo.transform);
                SetField(arena, "ramp", ramp);
                SetField(arena, "rampTarget", targetGo.transform);
                arena.EvalMode = true;

                arena.Tick(Time.fixedDeltaTime);

                Assert.IsFalse(arena.RampAtTarget);
                Assert.IsFalse(
                    arena.Success,
                    "Goal proximity must not score before this episode has placed the ramp.");

                ramp.ResetTo(targetGo.transform.position, Quaternion.identity);
                arena.Tick(Time.fixedDeltaTime);

                Assert.IsTrue(arena.RampAtTarget);
                Assert.IsTrue(
                    arena.Success,
                    "The same goal proximity must score once this episode has placed the ramp.");
            }
            finally
            {
                Object.DestroyImmediate(arenaGo);
                Object.DestroyImmediate(agentGo);
                Object.DestroyImmediate(rampGo);
                Object.DestroyImmediate(goalGo);
                Object.DestroyImmediate(targetGo);
            }
        }

        private static void SetField(object target, string name, object value)
        {
            FieldInfo field = target.GetType().GetField(name, PrivateInstance);
            Assert.NotNull(field, $"Expected private field '{name}' to exist.");
            field.SetValue(target, value);
        }
    }
}
