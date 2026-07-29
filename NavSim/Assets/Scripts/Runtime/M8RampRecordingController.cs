using System;
using System.IO;
using System.Linq;
using System.Reflection;
using Unity.MLAgents.Demonstrations;
using UnityEngine;

namespace NavSim.Runtime
{
    public sealed class M8RampRecordingController : MonoBehaviour
    {
        [SerializeField] private RampArena arena;
        [SerializeField] private DemonstrationRecorder recorder;

        private static readonly BindingFlags PrivateInstance =
            BindingFlags.Instance | BindingFlags.NonPublic;
        private static readonly FieldInfo InitialDistance =
            typeof(RampArena).GetField("s0InitialPushDistance", PrivateInstance);
        private static readonly FieldInfo S0Successes =
            typeof(RampArena).GetField("_s0Successes", PrivateInstance);
        private static readonly FieldInfo RecorderWriter =
            typeof(DemonstrationRecorder).GetField("m_DemoWriter", PrivateInstance);
        private static readonly FieldInfo WriterMetadata =
            typeof(DemonstrationWriter).GetField("m_MetaData", PrivateInstance);
        private static readonly FieldInfo MetadataEpisodeCount =
            typeof(DemonstrationWriter).Assembly
                .GetType("Unity.MLAgents.Demonstrations.DemonstrationMetaData")
                ?.GetField("numberEpisodes", BindingFlags.Instance | BindingFlags.Public);

        [Serializable]
        private sealed class RecordingReport
        {
            public string mode;
            public bool completed;
            public int[] attempts = new int[4];
            public int[] successes = new int[4];
            public int recordedEpisodes;
        }

        private RampArena _arena;
        private RecordingReport _report;
        private string _reportPath;
        private bool _recordMode;
        private bool _dryRunMode;
        private bool _initialized;
        private bool _exitRequested;
        private int _completedAttempts;

        private void Awake()
        {
            Time.timeScale = 20f;

            string modeArg = System.Environment.GetCommandLineArgs()
                .FirstOrDefault(a => a.StartsWith("--m8-mode="));
            _recordMode = modeArg == "--m8-mode=record";
            _dryRunMode = modeArg == "--m8-mode=dry-run";
            _report = new RecordingReport
            {
                mode = _recordMode ? "record" : _dryRunMode ? "dry-run" : "invalid"
            };

            if (!_recordMode && !_dryRunMode)
            {
                Fail("missing --m8-mode=dry-run|record");
                return;
            }

            _reportPath = Environment.GetEnvironmentVariable("M8_RECORD_REPORT");
            if (string.IsNullOrEmpty(_reportPath))
            {
                Fail("missing M8_RECORD_REPORT");
                return;
            }

            _arena = arena;
            if (_arena == null)
            {
                Fail("RampArena reference missing");
                return;
            }

            if (_recordMode)
            {
                if (recorder == null)
                {
                    Fail("DemonstrationRecorder reference missing");
                    return;
                }

                string demoDirectory = Environment.GetEnvironmentVariable("M8_DEMO_DIR");
                if (string.IsNullOrEmpty(demoDirectory))
                {
                    Fail("missing M8_DEMO_DIR");
                    return;
                }

                recorder.DemonstrationDirectory = demoDirectory;
                recorder.DemonstrationName = "M8RampSoloExpert";
                recorder.Record = true;
            }
        }

        public void HandleEpisodeBegin(bool previousSuccess)
        {
            if (_exitRequested) return;

            if (!_initialized)
            {
                _initialized = true;
                PrepareNextStart(RampExpertLogic.StartDistance(0, _recordMode));
                return;
            }

            int completedIndex = _completedAttempts;
            int rung = _recordMode ? completedIndex % 4 : Mathf.Min(completedIndex / 10, 3);
            M8RampExpertAgent expert = _arena.Agents
                .OfType<M8RampExpertAgent>()
                .SingleOrDefault();
            Vector3 agentPosition = expert != null ? expert.transform.position : Vector3.zero;
            Debug.Log(
                $"[M8DemoEpisode] episode={completedIndex + 1} " +
                $"distance={RampExpertLogic.StartDistance(completedIndex, _recordMode):F2} " +
                $"placement={_arena.RampAtTarget} goal={previousSuccess} " +
                $"steps={_arena.StepsThisEpisode} state={expert?.DebugState.ToString() ?? "missing"} " +
                $"stateSteps={(expert != null ? expert.DebugStateSteps : -1)} " +
                $"agent={agentPosition:F3} ramp={_arena.Ramp.Position:F3}");
            _report.attempts[rung]++;
            if (previousSuccess) _report.successes[rung]++;
            _completedAttempts++;
            if (_recordMode) _report.recordedEpisodes++;

            PrepareNextStart(RampExpertLogic.StartDistance(_completedAttempts, _recordMode));
            if (_exitRequested) return;

            if (_recordMode)
            {
                if (!previousSuccess)
                {
                    Fail("recorded episode failed");
                    return;
                }

                if (_completedAttempts == 40)
                {
                    Finish(true, 0);
                    return;
                }
            }
            else
            {
                bool rungComplete = _completedAttempts % 10 == 0;
                if (rungComplete && _report.successes[rung] < 9)
                {
                    Fail($"dry-run rung {rung} failed");
                    return;
                }

                if (_completedAttempts == 40)
                {
                    Finish(true, 0);
                    return;
                }
            }
        }

        private void PrepareNextStart(float distance)
        {
            if (InitialDistance == null || S0Successes == null)
            {
                Fail("RampArena recording fields not found");
                return;
            }

            InitialDistance.SetValue(_arena, distance);
            S0Successes.SetValue(_arena, 0);
        }

        private void Fail(string message)
        {
            Debug.LogError($"M8 recording failed: {message}");
            Finish(false, 2);
        }

        private void Finish(bool completed, int exitCode)
        {
            if (_exitRequested) return;
            _exitRequested = true;
            _report ??= new RecordingReport { mode = "invalid" };
            _report.completed = completed;

            if (_recordMode && recorder != null)
            {
                recorder.Record = false;
                bool metadataPrepared = _report.recordedEpisodes == 0;
                object writer = RecorderWriter?.GetValue(recorder);
                if (writer != null)
                    metadataPrepared = PrepareTerminalWriterForClose(
                        writer, _report.recordedEpisodes);
                if (!metadataPrepared)
                {
                    Debug.LogError("M8 recording metadata fields or episode count did not match");
                    _report.completed = false;
                    exitCode = 2;
                }
                recorder.Close();
            }

            if (!string.IsNullOrEmpty(_reportPath))
            {
                try
                {
                    string json = JsonUtility.ToJson(_report, true);
                    File.WriteAllText(_reportPath, json);
                }
                catch (Exception exception)
                {
                    Debug.LogError($"M8 recording report write failed: {exception}");
                    exitCode = 2;
                }
            }

            Application.Quit(exitCode);
            enabled = false;
        }

        private static bool PrepareTerminalWriterForClose(
            object writer, int completedTerminalEpisodes)
        {
            if (writer == null || completedTerminalEpisodes < 1 ||
                WriterMetadata == null || MetadataEpisodeCount == null)
                return false;

            object metadata = WriterMetadata.GetValue(writer);
            if (metadata == null ||
                MetadataEpisodeCount.GetValue(metadata) is not int currentEpisodes ||
                currentEpisodes != completedTerminalEpisodes)
                return false;

            // The terminal AgentInfo already completed this episode. ML-Agents Close()
            // unconditionally adds one more, so offset that pending increment.
            MetadataEpisodeCount.SetValue(metadata, currentEpisodes - 1);
            return true;
        }
    }
}
