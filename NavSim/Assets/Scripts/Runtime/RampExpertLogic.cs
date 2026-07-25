using UnityEngine;

namespace NavSim.Runtime
{
    public enum RampExpertState { Push, ClearWest, MoveSouth, CenterSlope, Climb }

    public readonly struct RampExpertInput
    {
        public readonly RampExpertState State;
        public readonly int StateSteps;
        public readonly Vector3 AgentPosition;
        public readonly float AgentYaw;
        public readonly Vector3 RampPosition;
        public readonly Vector3 GoalPosition;
        public readonly bool RampAtTarget;

        public RampExpertInput(
            RampExpertState state, int stateSteps, Vector3 agentPosition, float agentYaw,
            Vector3 rampPosition, Vector3 goalPosition, bool rampAtTarget)
        {
            State = state;
            StateSteps = stateSteps;
            AgentPosition = agentPosition;
            AgentYaw = agentYaw;
            RampPosition = rampPosition;
            GoalPosition = goalPosition;
            RampAtTarget = rampAtTarget;
        }
    }

    public readonly struct RampExpertStep
    {
        public readonly RampExpertState State;
        public readonly float Forward;
        public readonly float Turn;
        public readonly int Jump;
        public readonly bool TimedOut;

        public RampExpertStep(RampExpertState state, float forward, float turn, bool timedOut)
        {
            State = state;
            Forward = Mathf.Clamp(forward, -1f, 1f);
            Turn = Mathf.Clamp(turn, -1f, 1f);
            Jump = 0;
            TimedOut = timedOut;
        }
    }

    public static class RampExpertLogic
    {
        private static readonly float[] Distances = { 1.75f, 2.5f, 3.5f, 5f };

        public static float StartDistance(int episodeIndex, bool interleaved)
        {
            int i = Mathf.Max(0, episodeIndex);
            return interleaved ? Distances[i % 4] : Distances[Mathf.Min(i / 10, 3)];
        }

        public static int StepBudget(RampExpertState state) => state switch
        {
            RampExpertState.Push => 1800,
            RampExpertState.ClearWest => 180,
            RampExpertState.MoveSouth => 240,
            RampExpertState.CenterSlope => 240,
            RampExpertState.Climb => 400,
            _ => 0
        };

        public static RampExpertStep Decide(RampExpertInput input)
        {
            if (input.StateSteps > StepBudget(input.State))
                return new RampExpertStep(input.State, 0f, 0f, true);

            Vector3 west = new Vector3(
                input.RampPosition.x - 3f, input.AgentPosition.y, input.RampPosition.z);
            Vector3 southwest = new Vector3(
                input.RampPosition.x - 3f, input.AgentPosition.y, input.RampPosition.z - 2.75f);
            Vector3 southCenter = new Vector3(
                input.RampPosition.x, input.AgentPosition.y, input.RampPosition.z - 2.75f);

            switch (input.State)
            {
                case RampExpertState.Push:
                    return input.RampAtTarget
                        ? new RampExpertStep(RampExpertState.ClearWest, 0f, 0f, false)
                        : Steer(input, 90f, RampExpertState.Push);
                case RampExpertState.ClearWest:
                    return AtWaypoint(input.AgentPosition, west)
                        ? new RampExpertStep(RampExpertState.MoveSouth, 0f, 0f, false)
                        : Steer(input, west, RampExpertState.ClearWest);
                case RampExpertState.MoveSouth:
                    return AtWaypoint(input.AgentPosition, southwest)
                        ? new RampExpertStep(RampExpertState.CenterSlope, 0f, 0f, false)
                        : Steer(input, southwest, RampExpertState.MoveSouth);
                case RampExpertState.CenterSlope:
                    return AtWaypoint(input.AgentPosition, southCenter)
                        ? new RampExpertStep(RampExpertState.Climb, 0f, 0f, false)
                        : Steer(input, southCenter, RampExpertState.CenterSlope);
                case RampExpertState.Climb:
                    return Steer(input, input.GoalPosition, RampExpertState.Climb);
                default:
                    return new RampExpertStep(input.State, 0f, 0f, true);
            }
        }

        private static bool AtWaypoint(Vector3 position, Vector3 waypoint) =>
            Vector3.Distance(position, waypoint) <= 0.25f;

        private static RampExpertStep Steer(
            RampExpertInput input, Vector3 waypoint, RampExpertState state)
        {
            Vector3 delta = waypoint - input.AgentPosition;
            float targetYaw = Mathf.Atan2(delta.x, delta.z) * Mathf.Rad2Deg;
            return Steer(input, targetYaw, state);
        }

        private static RampExpertStep Steer(
            RampExpertInput input, float targetYaw, RampExpertState state)
        {
            float headingError = Mathf.DeltaAngle(input.AgentYaw, targetYaw);
            float turn = Mathf.Clamp(headingError / 6f, -1f, 1f);
            float forward = Mathf.Abs(headingError) <= 12f ? 1f : 0f;
            return new RampExpertStep(state, forward, turn, false);
        }
    }
}
