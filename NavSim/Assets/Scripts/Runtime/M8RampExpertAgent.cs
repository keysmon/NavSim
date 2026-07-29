using System;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using UnityEngine;

namespace NavSim.Runtime
{
    public sealed class M8RampExpertAgent : RampAgent
    {
        private const int ExpectedDecisionPeriod = 5;

        private RampArena _arena;
        private M8RampRecordingController _recording;
        private RampExpertState _state;
        private int _stateSteps;
        private int _turnRepeatHorizon;

        internal RampExpertState DebugState => _state;
        internal int DebugStateSteps => _stateSteps;

        public override void Initialize()
        {
            base.Initialize();
            var requester = GetComponent<DecisionRequester>();
            if (requester == null ||
                requester.DecisionPeriod != ExpectedDecisionPeriod ||
                !requester.TakeActionsBetweenDecisions)
                throw new InvalidOperationException(
                    "M8 recording expert requires DecisionPeriod=5 with repeated actions");
            _turnRepeatHorizon = requester.DecisionPeriod;
            _arena = FindFirstObjectByType<RampArena>();
            _recording = FindFirstObjectByType<M8RampRecordingController>();
        }

        public override void OnEpisodeBegin()
        {
            base.OnEpisodeBegin();
            if (_arena == null) _arena = FindFirstObjectByType<RampArena>();
            if (_recording == null) _recording = FindFirstObjectByType<M8RampRecordingController>();
            _recording.HandleEpisodeBegin(_arena.Success);
            _state = RampExpertState.Push;
            _stateSteps = 0;
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var input = new RampExpertInput(
                _state, _stateSteps, transform.position, transform.eulerAngles.y,
                _arena.Ramp.Position, _arena.GoalPosition, _arena.RampAtTarget);
            RampExpertStep step = RampExpertLogic.Decide(input);
            if (step.State != _state)
            {
                Debug.Log(
                    $"[M8ExpertState] {_state}->{step.State} agent={transform.position:F3} " +
                    $"ramp={_arena.Ramp.Position:F3}");
                _state = step.State;
                _stateSteps = 0;
            }
            else _stateSteps++;
            if (step.TimedOut && _stateSteps == RampExpertLogic.StepBudget(_state) + 2)
                Debug.LogError(
                    $"[M8ExpertTimeout] state={_state} agent={transform.position:F3} " +
                    $"ramp={_arena.Ramp.Position:F3}");
            var continuous = actionsOut.ContinuousActions;
            continuous[0] = step.Forward;
            continuous[1] = ScaleTurnForRepeatHorizon(step.Turn, _turnRepeatHorizon);
            var discrete = actionsOut.DiscreteActions;
            discrete[0] = step.Jump;
        }

        private static float ScaleTurnForRepeatHorizon(float turn, int repeatHorizon)
        {
            if (repeatHorizon < 1)
                throw new ArgumentOutOfRangeException(nameof(repeatHorizon));
            return Mathf.Clamp(turn / repeatHorizon, -1f, 1f);
        }
    }
}
