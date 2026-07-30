using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NavSim.Runtime;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using Unity.MLAgents.Demonstrations;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using Object = UnityEngine.Object;

public static class M8RampDemonstrationSetup
{
    private const string CanonicalScene = "Assets/Scenes/Ramp.unity";
    private const string CanonicalSoloScene = "Assets/Scenes/Ramp_solo.unity";
    private const string RecordingScene = "Assets/Scenes/Ramp_recording.unity";
    private const string RecorderBuild = "Builds/M8RampRecorder.app";
    private const string Demonstration = "Assets/Demonstrations/M8RampSoloExpert.demo";
    private const string DemonstrationName = "M8RampSoloExpert";
    private const string Hard80Demonstration =
        "Assets/Demonstrations/M8RampSoloExpertHard80.demo";
    private const string Hard80DemonstrationName = "M8RampSoloExpertHard80";

    public static void BuildScene()
    {
        try
        {
            Scene solo = EditorSceneManager.OpenScene(CanonicalSoloScene, OpenSceneMode.Single);
            Require(
                EditorSceneManager.SaveScene(solo, RecordingScene, true),
                $"copied {CanonicalSoloScene} to {RecordingScene}");

            Scene recording = EditorSceneManager.OpenScene(RecordingScene, OpenSceneMode.Single);
            RampArena arena = Object.FindFirstObjectByType<RampArena>();
            Require(arena != null, "recording scene has one RampArena");

            RampAgent oldAgent = arena.Agents.Single();
            GameObject agentGo = oldAgent.gameObject;
            var oldSo = new SerializedObject(oldAgent);
            int oldMaxStep = oldAgent.MaxStep;
            float oldMaxSpeed = RequiredProperty(oldSo, "maxSpeed").floatValue;
            float oldMaxTurn = RequiredProperty(oldSo, "maxTurnDegPerStep").floatValue;
            float oldJumpImpulse = RequiredProperty(oldSo, "jumpImpulse").floatValue;
            float oldGravity = RequiredProperty(oldSo, "gravity").floatValue;
            float oldTerminalVelocity = RequiredProperty(oldSo, "terminalVelocity").floatValue;
            float oldPushForce = RequiredProperty(oldSo, "pushForceNewtons").floatValue;
            int agentIndex = RequiredProperty(oldSo, "agentIndex").intValue;

            var expert = agentGo.AddComponent<M8RampExpertAgent>();
            // DecisionRequester requires an Agent. Add the replacement first so Unity allows the
            // plain RampAgent to be removed without disturbing the existing requester component.
            Object.DestroyImmediate(oldAgent);

            var expertSo = new SerializedObject(expert);
            RequiredProperty(expertSo, "arena").objectReferenceValue = arena;
            RequiredProperty(expertSo, "maxSpeed").floatValue = oldMaxSpeed;
            RequiredProperty(expertSo, "maxTurnDegPerStep").floatValue = oldMaxTurn;
            RequiredProperty(expertSo, "jumpImpulse").floatValue = oldJumpImpulse;
            RequiredProperty(expertSo, "gravity").floatValue = oldGravity;
            RequiredProperty(expertSo, "terminalVelocity").floatValue = oldTerminalVelocity;
            RequiredProperty(expertSo, "pushForceNewtons").floatValue = oldPushForce;
            RequiredProperty(expertSo, "agentIndex").intValue = agentIndex;
            RequiredProperty(expertSo, "MaxStep").intValue = oldMaxStep;
            expertSo.ApplyModifiedPropertiesWithoutUndo();

            var arenaSo = new SerializedObject(arena);
            SerializedProperty agents = RequiredProperty(arenaSo, "agents");
            Require(agents.arraySize == 1, "source arena has exactly one agent slot");
            agents.GetArrayElementAtIndex(0).objectReferenceValue = expert;
            arenaSo.ApplyModifiedPropertiesWithoutUndo();

            BehaviorParameters behavior = agentGo.GetComponent<BehaviorParameters>();
            Require(behavior != null, "expert GameObject retains BehaviorParameters");
            behavior.BehaviorType = BehaviorType.HeuristicOnly;

            var recorder = agentGo.AddComponent<DemonstrationRecorder>();
            recorder.Record = false;
            recorder.NumStepsToRecord = 0;
            recorder.DemonstrationName = DemonstrationName;

            var controller = arena.gameObject.AddComponent<M8RampRecordingController>();
            var controllerSo = new SerializedObject(controller);
            RequiredProperty(controllerSo, "arena").objectReferenceValue = arena;
            RequiredProperty(controllerSo, "recorder").objectReferenceValue = recorder;
            controllerSo.ApplyModifiedPropertiesWithoutUndo();

            AssertRecordingScene(recording, arena, expert, behavior, recorder, controller);
            AssertCanonicalSceneIsolated(CanonicalScene);
            AssertCanonicalSceneIsolated(CanonicalSoloScene);

            SceneManager.SetActiveScene(recording);
            Require(
                EditorSceneManager.SaveScene(recording, RecordingScene),
                $"saved validated recording scene at {RecordingScene}");
            AssetDatabase.SaveAssets();
            Debug.Log("[M8DemoSetup] BuildScene PASS");
            EditorApplication.Exit(0);
        }
        catch (Exception exception)
        {
            Debug.LogError("[M8DemoSetup] BuildScene FAILED: " + exception);
            EditorApplication.Exit(1);
        }
    }

    public static void BuildPlayer()
    {
        try
        {
            Require(File.Exists(RecordingScene), $"recording scene exists at {RecordingScene}");

            var options = new BuildPlayerOptions
            {
                scenes = new[] { RecordingScene },
                locationPathName = RecorderBuild,
                target = BuildTarget.StandaloneOSX,
                options = BuildOptions.None
            };
            BuildSummary summary = BuildPipeline.BuildPlayer(options).summary;
            Debug.Log(
                $"[M8DemoSetup] recorder build result={summary.result} " +
                $"errors={summary.totalErrors} warnings={summary.totalWarnings} output={summary.outputPath}");

            Require(summary.result == BuildResult.Succeeded, "recorder player build result is Succeeded");
            Require(summary.totalErrors == 0, "recorder player build has zero errors");
            Require(Directory.Exists(RecorderBuild), $"recorder app bundle exists at {RecorderBuild}");
            Require(
                Directory.Exists(Path.Combine(RecorderBuild, "Contents", "MacOS")),
                "recorder app contains Contents/MacOS");
            Debug.Log("[M8DemoSetup] BuildPlayer PASS");
            EditorApplication.Exit(0);
        }
        catch (Exception exception)
        {
            Debug.LogError("[M8DemoSetup] BuildPlayer FAILED: " + exception);
            EditorApplication.Exit(1);
        }
    }

    public static void ValidateDemo() =>
        ValidateDemoAtPath(Demonstration, DemonstrationName, 40);

    public static void ValidateHard80Demo() =>
        ValidateDemoAtPath(Hard80Demonstration, Hard80DemonstrationName, 80);

    private static void ValidateDemoAtPath(
        string path, string expectedName, int expectedEpisodes)
    {
        try
        {
            Require(File.Exists(path), $"demonstration exists at {path}");
            AssetDatabase.ImportAsset(
                path,
                ImportAssetOptions.ForceSynchronousImport | ImportAssetOptions.ForceUpdate);

            Object summaryAsset = AssetDatabase.LoadMainAssetAtPath(path);
            Require(summaryAsset != null, "demonstration importer produced a main asset");
            Require(
                summaryAsset.GetType().Name == "DemonstrationSummary",
                "imported main asset is the internal DemonstrationSummary");

            var summarySo = new SerializedObject(summaryAsset);
            string importedName =
                RequiredProperty(summarySo, "metaData.demonstrationName").stringValue;
            int episodes = RequiredProperty(summarySo, "metaData.numberEpisodes").intValue;
            int steps = RequiredProperty(summarySo, "metaData.numberSteps").intValue;
            float meanReward = RequiredProperty(summarySo, "metaData.meanReward").floatValue;
            int continuous =
                RequiredProperty(
                    summarySo,
                    "brainParameters.m_ActionSpec.m_NumContinuousActions").intValue;
            int[] branches = ReadIntArray(
                RequiredProperty(summarySo, "brainParameters.m_ActionSpec.BranchSizes"));
            List<string> importedShapes = ReadImportedShapes(
                RequiredProperty(summarySo, "observationSummaries"));

            Require(importedName == expectedName, $"demonstration name is {expectedName}");
            Require(
                episodes == expectedEpisodes,
                $"demonstration contains exactly {expectedEpisodes} episodes");
            Require(steps > 0, "demonstration contains at least one step");
            Require(continuous == 2, "demonstration action spec has two continuous actions");
            Require(
                branches.SequenceEqual(new[] { 2 }),
                "demonstration action spec has one discrete branch of size two");

            EditorSceneManager.OpenScene(CanonicalSoloScene, OpenSceneMode.Single);
            RampAgent agent = Object.FindFirstObjectByType<RampAgent>();
            Require(agent != null, "canonical solo scene contains a RampAgent");
            BehaviorParameters behavior = agent.GetComponent<BehaviorParameters>();
            Require(behavior != null, "canonical solo RampAgent has BehaviorParameters");
            Require(behavior.BehaviorName == "RampAgent", "canonical solo behavior name is RampAgent");
            Require(
                behavior.BrainParameters.VectorObservationSize == 6,
                "canonical solo vector observation size is six");

            List<string> expectedShapes = CreateSceneShapeMultiset(agent, behavior);
            Require(
                expectedShapes.SequenceEqual(importedShapes),
                "demonstration sensor observation shape multiset matches Ramp_solo");

            Debug.Log(
                $"[M8DemoSetup] ValidateDemo PASS meanReward={meanReward:R} steps={steps} " +
                $"episodes={episodes} actionSpec=continuous:{continuous},branches:[{string.Join(",", branches)}] " +
                $"shapes={FormatMultiset(importedShapes)}");
            EditorApplication.Exit(0);
        }
        catch (Exception exception)
        {
            Debug.LogError("[M8DemoSetup] ValidateDemo FAILED: " + exception);
            EditorApplication.Exit(1);
        }
    }

    private static void AssertRecordingScene(
        Scene scene,
        RampArena arena,
        M8RampExpertAgent expert,
        BehaviorParameters behavior,
        DemonstrationRecorder recorder,
        M8RampRecordingController controller)
    {
        Require(ComponentsInScene<M8RampExpertAgent>(scene).Count == 1,
            "recording scene has exactly one M8RampExpertAgent");
        Require(ComponentsInScene<DemonstrationRecorder>(scene).Count == 1,
            "recording scene has exactly one DemonstrationRecorder");
        Require(ComponentsInScene<M8RampRecordingController>(scene).Count == 1,
            "recording scene has exactly one M8RampRecordingController");
        Require(
            ComponentsInScene<RampAgent>(scene).Count(agent => agent.GetType() == typeof(RampAgent)) == 0,
            "recording scene has zero plain RampAgent components");
        Require(behavior.BehaviorName == "RampAgent", "recording behavior name remains RampAgent");

        var actionSpec = behavior.BrainParameters.ActionSpec;
        Require(actionSpec.NumContinuousActions == 2,
            "recording action spec has two continuous actions");
        Require(
            actionSpec.BranchSizes != null &&
            actionSpec.BranchSizes.SequenceEqual(new[] { 2 }),
            "recording action spec has one discrete branch of size two");
        Require(
            behavior.BrainParameters.VectorObservationSize == 6,
            "recording vector observation size remains six");
        Require(
            behavior.BehaviorType == BehaviorType.HeuristicOnly,
            "recording behavior type is HeuristicOnly");
        Require(
            arena.Agents != null && arena.Agents.Length == 1 &&
            ReferenceEquals(arena.Agents[0], expert),
            "recording arena's sole agent reference is the expert subclass");

        var expertSo = new SerializedObject(expert);
        Require(
            ReferenceEquals(RequiredProperty(expertSo, "arena").objectReferenceValue, arena),
            "expert preserves the RampArena serialized reference");
        Require(
            RequiredProperty(expertSo, "agentIndex").intValue == 0,
            "expert preserves agentIndex zero");

        Require(!recorder.Record, "recorder Record is false");
        Require(recorder.NumStepsToRecord == 0, "recorder NumStepsToRecord is zero");
        Require(
            recorder.DemonstrationName == DemonstrationName,
            $"recorder DemonstrationName is {DemonstrationName}");

        var controllerSo = new SerializedObject(controller);
        Require(
            ReferenceEquals(RequiredProperty(controllerSo, "arena").objectReferenceValue, arena),
            "recording controller arena reference is wired");
        Require(
            ReferenceEquals(RequiredProperty(controllerSo, "recorder").objectReferenceValue, recorder),
            "recording controller recorder reference is wired");
    }

    private static void AssertCanonicalSceneIsolated(string path)
    {
        Scene scene = EditorSceneManager.OpenScene(path, OpenSceneMode.Additive);
        try
        {
            Require(
                ComponentsInScene<M8RampExpertAgent>(scene).Count == 0,
                $"{path} contains no M8RampExpertAgent");
            Require(
                ComponentsInScene<M8RampRecordingController>(scene).Count == 0,
                $"{path} contains no M8RampRecordingController");
            Require(
                ComponentsInScene<DemonstrationRecorder>(scene).Count == 0,
                $"{path} contains no DemonstrationRecorder");

            List<BehaviorParameters> behaviors = ComponentsInScene<BehaviorParameters>(scene);
            Require(behaviors.Count > 0, $"{path} contains BehaviorParameters");
            for (int i = 0; i < behaviors.Count; i++)
            {
                Require(
                    behaviors[i].BehaviorType == BehaviorType.Default,
                    $"{path} BehaviorParameters[{i}] remains Default");
            }
        }
        finally
        {
            EditorSceneManager.CloseScene(scene, true);
        }
    }

    private static List<string> CreateSceneShapeMultiset(
        RampAgent agent,
        BehaviorParameters behavior)
    {
        var shapes = new List<string>
        {
            ShapeKey(new[] { behavior.BrainParameters.VectorObservationSize })
        };

        SensorComponent[] components = behavior.UseChildSensors
            ? agent.GetComponentsInChildren<SensorComponent>(true)
            : agent.GetComponents<SensorComponent>();
        foreach (SensorComponent component in components)
        {
            foreach (ISensor sensor in component.CreateSensors())
            {
                var shape = sensor.GetObservationSpec().Shape;
                var dimensions = new int[shape.Length];
                for (int i = 0; i < shape.Length; i++) dimensions[i] = shape[i];
                shapes.Add(ShapeKey(dimensions));
            }
        }

        shapes.Sort(StringComparer.Ordinal);
        return shapes;
    }

    private static List<string> ReadImportedShapes(SerializedProperty summaries)
    {
        Require(summaries.isArray, "demonstration observationSummaries is an array");
        var shapes = new List<string>();
        for (int i = 0; i < summaries.arraySize; i++)
        {
            SerializedProperty summary = summaries.GetArrayElementAtIndex(i);
            SerializedProperty shape = summary.FindPropertyRelative("shape");
            if (shape == null || !shape.isArray)
                throw new InvalidOperationException($"observationSummaries[{i}].shape is missing");
            shapes.Add(ShapeKey(ReadIntArray(shape)));
        }

        shapes.Sort(StringComparer.Ordinal);
        return shapes;
    }

    private static int[] ReadIntArray(SerializedProperty property)
    {
        Require(property.isArray, $"{property.propertyPath} is an array");
        var result = new int[property.arraySize];
        for (int i = 0; i < result.Length; i++)
            result[i] = property.GetArrayElementAtIndex(i).intValue;
        return result;
    }

    private static string ShapeKey(IEnumerable<int> shape) => "[" + string.Join(",", shape) + "]";

    private static string FormatMultiset(IEnumerable<string> shapes) =>
        "{" + string.Join(", ", shapes) + "}";

    private static List<T> ComponentsInScene<T>(Scene scene) where T : Component
    {
        return scene.GetRootGameObjects()
            .SelectMany(root => root.GetComponentsInChildren<T>(true))
            .ToList();
    }

    private static SerializedProperty RequiredProperty(SerializedObject serialized, string path)
    {
        SerializedProperty property = serialized.FindProperty(path);
        if (property == null)
            throw new InvalidOperationException(
                $"serialized property '{path}' not found on {serialized.targetObject}");
        return property;
    }

    private static void Require(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException(message);
        Debug.Log("[M8DemoSetup] ASSERT PASS: " + message);
    }
}
