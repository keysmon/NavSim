using System;
using System.IO;
using System.Linq;
using System.Reflection;
using NavSim.Runtime;
using NUnit.Framework;

namespace NavSim.Tests.EditMode
{
    public class M8RampRecordingControllerTests
    {
        private const BindingFlags PrivateInstance =
            BindingFlags.Instance | BindingFlags.NonPublic;
        private const BindingFlags PrivateStatic =
            BindingFlags.Static | BindingFlags.NonPublic;

        [Test]
        public void RecorderClose_MixedDemoAfterFortyTerminalEpisodesPreservesMetadataCount()
        {
            AssertTerminalClosePreservesEpisodeCount("M8RampSoloExpert", 40);
        }

        [Test]
        public void RecorderClose_HardDemoAfterEightyTerminalEpisodesPreservesMetadataCount()
        {
            AssertTerminalClosePreservesEpisodeCount("M8RampHard80", 80);
        }

        [Test]
        public void RecorderClose_PushDemoAfterEightyTerminalEpisodesPreservesMetadataCount()
        {
            AssertTerminalClosePreservesEpisodeCount("M8RampPush80", 80);
        }

        [Test]
        public void ParseMode_PushOnlyUsesHardScheduleAndDistinctIdentity()
        {
            object mode = InvokePrivateStatic("ParseMode", "--m8-mode=record-push80");

            Assert.AreEqual("Push80", mode.ToString());
            Assert.AreEqual("record-push80", InvokePrivateStatic("ModeName", mode));
            Assert.AreEqual(80, InvokePrivateStatic("RequiredEpisodeCount", mode));
            Assert.AreEqual("M8RampPush80", InvokePrivateStatic("DemonstrationName", mode));
            Assert.AreEqual(
                5f,
                (float)InvokePrivateStatic("StartDistance", mode, 79),
                1e-5f);
        }

        [TestCase(true, false, false, false)]
        [TestCase(true, true, false, true)]
        [TestCase(true, true, true, false)]
        [TestCase(false, true, false, false)]
        public void PlacementBoundary_IsIssuedOnlyOnceInPushOnlyMode(
            bool pushOnly, bool rampAtTarget, bool alreadyIssued, bool expected)
        {
            Assert.AreEqual(
                expected,
                InvokePrivateStatic(
                    "ShouldIssuePlacementBoundary",
                    pushOnly,
                    rampAtTarget,
                    alreadyIssued));
        }

        private static object InvokePrivateStatic(string methodName, params object[] arguments)
        {
            MethodInfo method = typeof(M8RampRecordingController)
                .GetMethod(methodName, PrivateStatic);
            Assert.NotNull(method, $"private static method {methodName} exists");
            return method.Invoke(null, arguments);
        }

        private static void AssertTerminalClosePreservesEpisodeCount(
            string demonstrationName, int completedEpisodes)
        {
            Assembly mlAgents = AppDomain.CurrentDomain.GetAssemblies()
                .Single(a => a.GetName().Name == "Unity.ML-Agents");
            Type writerType = mlAgents.GetType(
                "Unity.MLAgents.Demonstrations.DemonstrationWriter", true);
            Type metadataType = mlAgents.GetType(
                "Unity.MLAgents.Demonstrations.DemonstrationMetaData", true);
            object writer = Activator.CreateInstance(writerType, new MemoryStream());
            object metadata = Activator.CreateInstance(metadataType);
            FieldInfo numberEpisodes = metadataType.GetField("numberEpisodes");
            metadataType.GetField("demonstrationName").SetValue(metadata, demonstrationName);
            numberEpisodes.SetValue(metadata, completedEpisodes);
            writerType.GetField("m_MetaData", PrivateInstance).SetValue(writer, metadata);

            MethodInfo prepareClose = typeof(M8RampRecordingController).GetMethod(
                "PrepareTerminalWriterForClose", PrivateStatic);

            Assert.NotNull(prepareClose);
            Assert.IsTrue((bool)prepareClose.Invoke(
                null,
                new[] { writer, (object)completedEpisodes }));
            writerType.GetMethod("Close").Invoke(writer, null);
            Assert.AreEqual(completedEpisodes, numberEpisodes.GetValue(metadata));
        }
    }
}
