using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Unity.MLAgents.Demonstrations;
using UnityEngine;

namespace NavSim.Runtime
{
    public sealed class M8RampRecordingController : MonoBehaviour
    {
        private enum RecordingMode { Invalid, DryRun, Mixed40, Hard80, Push80 }

        private const int MixedEpisodeCount = 40;
        private const int HardEpisodeCount = 80;
        private const int PushEpisodeCount = 80;
        private const string MixedDemoName = "M8RampSoloExpert";

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
            public int[] placements = new int[4];
            public int[] successes = new int[4];
            public int recordedEpisodes;
            public List<float> episodeStartDistances = new List<float>();
        }

        private RampArena _arena;
        private RecordingReport _report;
        private string _reportPath;
        private RecordingMode _mode;
        private bool _initialized;
        private bool _exitRequested;
        private bool _placementBoundaryIssued;
        private int _completedAttempts;

        private void Awake()
        {
            Time.timeScale = 20f;

            string modeArg = System.Environment.GetCommandLineArgs()
                .FirstOrDefault(a => a.StartsWith("--m8-mode="));
            _mode = ParseMode(modeArg);
            _report = new RecordingReport
            {
                mode = ModeName(_mode)
            };

            if (_mode == RecordingMode.Invalid)
            {
                Fail("missing --m8-mode=dry-run|record|record-hard80|record-push80");
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

            if (IsRecording(_mode))
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
                recorder.DemonstrationName = DemonstrationName(_mode);
                recorder.Record = true;
            }
        }

        private void FixedUpdate()
        {
            if (!ShouldIssuePlacementBoundary(
                    _mode == RecordingMode.Push80,
                    _arena != null && _arena.RampAtTarget,
                    _placementBoundaryIssued))
                return;

            M8RampExpertAgent expert = _arena.Agents
                .OfType<M8RampExpertAgent>()
                .SingleOrDefault();
            if (expert == null)
            {
                Fail("push-only recording expert missing");
                return;
            }

            if (_arena.Success)
            {
                Fail("push-only episode reached goal before placement boundary");
                return;
            }

            _placementBoundaryIssued = true;
            expert.EndEpisode();
            _arena.ResetEpisode();
        }

        public void HandleEpisodeBegin(bool previousSuccess)
        {
            if (_exitRequested) return;

            if (!_initialized)
            {
                _initialized = true;
                PrepareNextStart(StartDistance(_mode, 0));
                return;
            }

            int completedIndex = _completedAttempts;
            float completedStartDistance = StartDistance(_mode, completedIndex);
            int rung = _mode switch
            {
                RecordingMode.Mixed40 => completedIndex % 4,
                RecordingMode.Hard80 or RecordingMode.Push80 => 3,
                _ => Mathf.Min(completedIndex / 10, 3)
            };
            M8RampExpertAgent expert = _arena.Agents
                .OfType<M8RampExpertAgent>()
                .SingleOrDefault();
            Vector3 agentPosition = expert != null ? expert.transform.position : Vector3.zero;
            Debug.Log(
                $"[M8DemoEpisode] episode={completedIndex + 1} " +
                $"distance={completedStartDistance:F2} " +
                $"placement={_arena.RampAtTarget} goal={previousSuccess} " +
                $"steps={_arena.StepsThisEpisode} state={expert?.DebugState.ToString() ?? "missing"} " +
                $"stateSteps={(expert != null ? expert.DebugStateSteps : -1)} " +
                $"agent={agentPosition:F3} ramp={_arena.Ramp.Position:F3}");
            _report.attempts[rung]++;
            if (_arena.RampAtTarget) _report.placements[rung]++;
            if (previousSuccess) _report.successes[rung]++;
            _report.episodeStartDistances.Add(completedStartDistance);
            _completedAttempts++;

            PrepareNextStart(StartDistance(_mode, _completedAttempts));
            if (_exitRequested) return;

            if (IsRecording(_mode))
            {
                bool terminalAccepted = _mode == RecordingMode.Push80
                    ? _placementBoundaryIssued && _arena.RampAtTarget && !previousSuccess
                    : previousSuccess && _arena.RampAtTarget;
                if (!terminalAccepted)
                {
                    Fail(
                        _mode == RecordingMode.Push80
                            ? "push-only terminal did not place ramp before reaching goal"
                            : "recorded episode did not place ramp and reach goal");
                    return;
                }

                _report.recordedEpisodes++;
                _placementBoundaryIssued = false;
                if (_completedAttempts == RequiredEpisodeCount(_mode))
                {
                    if ((_mode == RecordingMode.Hard80 ||
                         _mode == RecordingMode.Push80) &&
                        _report.episodeStartDistances.Any(
                            distance => !Mathf.Approximately(
                                distance, RampExpertLogic.HardStartDistance(0))))
                    {
                        Fail("hard recording contained a non-5-unit start");
                        return;
                    }
                    if (_mode == RecordingMode.Push80 &&
                        (_report.attempts.Sum() != PushEpisodeCount ||
                         _report.placements.Sum() != PushEpisodeCount ||
                         _report.successes.Sum() != 0 ||
                         _report.recordedEpisodes != PushEpisodeCount))
                    {
                        Fail("push-only report totals did not match 80 attempts and placements with zero goals");
                        return;
                    }
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

                if (_completedAttempts == MixedEpisodeCount)
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

        private static RecordingMode ParseMode(string modeArg) => modeArg switch
        {
            "--m8-mode=dry-run" => RecordingMode.DryRun,
            "--m8-mode=record" => RecordingMode.Mixed40,
            "--m8-mode=record-hard80" => RecordingMode.Hard80,
            "--m8-mode=record-push80" => RecordingMode.Push80,
            _ => RecordingMode.Invalid
        };

        private static string ModeName(RecordingMode mode) => mode switch
        {
            RecordingMode.DryRun => "dry-run",
            RecordingMode.Mixed40 => "record",
            RecordingMode.Hard80 => "record-hard80",
            RecordingMode.Push80 => "record-push80",
            _ => "invalid"
        };

        private static bool IsRecording(RecordingMode mode) =>
            mode == RecordingMode.Mixed40 ||
            mode == RecordingMode.Hard80 ||
            mode == RecordingMode.Push80;

        private static int RequiredEpisodeCount(RecordingMode mode) => mode switch
        {
            RecordingMode.Hard80 => HardEpisodeCount,
            RecordingMode.Push80 => PushEpisodeCount,
            _ => MixedEpisodeCount
        };

        private static string DemonstrationName(RecordingMode mode) => mode switch
        {
            RecordingMode.Hard80 => RampExpertLogic.HardDemonstrationName,
            RecordingMode.Push80 => RampExpertLogic.PushDemonstrationName,
            _ => MixedDemoName
        };

        private static float StartDistance(RecordingMode mode, int episodeIndex) =>
            mode switch
            {
                RecordingMode.Hard80 or RecordingMode.Push80 =>
                    RampExpertLogic.HardStartDistance(episodeIndex),
                RecordingMode.Mixed40 => RampExpertLogic.StartDistance(episodeIndex, true),
                _ => RampExpertLogic.StartDistance(episodeIndex, false)
            };

        private static bool ShouldIssuePlacementBoundary(
            bool pushOnly, bool rampAtTarget, bool alreadyIssued) =>
            pushOnly && rampAtTarget && !alreadyIssued;

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

            if (IsRecording(_mode) && recorder != null)
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
