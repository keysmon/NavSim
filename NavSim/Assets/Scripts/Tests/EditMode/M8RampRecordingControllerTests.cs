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
            Assembly mlAgents = AppDomain.CurrentDomain.GetAssemblies()
                .Single(a => a.GetName().Name == "Unity.ML-Agents");
            Type writerType = mlAgents.GetType(
                "Unity.MLAgents.Demonstrations.DemonstrationWriter", true);
            Type metadataType = mlAgents.GetType(
                "Unity.MLAgents.Demonstrations.DemonstrationMetaData", true);
            object writer = Activator.CreateInstance(writerType, new MemoryStream());
            object metadata = Activator.CreateInstance(metadataType);
            FieldInfo numberEpisodes = metadataType.GetField("numberEpisodes");
            metadataType.GetField("demonstrationName").SetValue(metadata, "M8RampSoloExpert");
            numberEpisodes.SetValue(metadata, 40);
            writerType.GetField("m_MetaData", PrivateInstance).SetValue(writer, metadata);

            MethodInfo prepareClose = typeof(M8RampRecordingController).GetMethod(
                "PrepareTerminalWriterForClose", PrivateStatic);

            Assert.NotNull(prepareClose);
            Assert.IsTrue((bool)prepareClose.Invoke(null, new[] { writer, (object)40 }));
            writerType.GetMethod("Close").Invoke(writer, null);
            Assert.AreEqual(40, numberEpisodes.GetValue(metadata));
        }

        [Test]
        public void RecorderClose_HardDemoAfterEightyTerminalEpisodesPreservesMetadataCount()
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
            metadataType.GetField("demonstrationName")
                .SetValue(metadata, "M8RampSoloExpertHard80");
            numberEpisodes.SetValue(metadata, 80);
            writerType.GetField("m_MetaData", PrivateInstance).SetValue(writer, metadata);

            MethodInfo prepareClose = typeof(M8RampRecordingController).GetMethod(
                "PrepareTerminalWriterForClose", PrivateStatic);

            Assert.NotNull(prepareClose);
            Assert.IsTrue((bool)prepareClose.Invoke(null, new[] { writer, (object)80 }));
            writerType.GetMethod("Close").Invoke(writer, null);
            Assert.AreEqual(80, numberEpisodes.GetValue(metadata));
        }
    }
}
