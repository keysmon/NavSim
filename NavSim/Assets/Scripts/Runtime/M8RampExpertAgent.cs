using Unity.MLAgents.Actuators;
using UnityEngine;

namespace NavSim.Runtime
{
    public sealed class M8RampExpertAgent : RampAgent
    {
        private RampArena _arena;
        private M8RampRecordingController _recording;
        private RampExpertState _state;
        private int _stateSteps;

        public override void Initialize()
        {
            base.Initialize();
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
            if (step.State != _state) { _state = step.State; _stateSteps = 0; }
            else _stateSteps++;
            var continuous = actionsOut.ContinuousActions;
            continuous[0] = step.Forward;
            continuous[1] = step.Turn;
            var discrete = actionsOut.DiscreteActions;
            discrete[0] = step.Jump;
        }
    }
}
