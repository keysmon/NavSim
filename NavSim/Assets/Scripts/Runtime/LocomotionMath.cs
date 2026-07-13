using UnityEngine;

namespace NavSim.Runtime
{
    // Pure vertical-velocity integration for the CharacterController agent, isolated for unit testing.
    // `grounded` is injected by the caller (CharacterController.isGrounded). Keeps NavAgent's OnActionReceived
    // free of untestable branching.
    public static class LocomotionMath
    {
        public static float NextVerticalVelocity(float vY, bool grounded, bool jumpPressed,
            float jumpImpulse, float gravity, float dt, float terminalVelocity)
        {
            if (grounded)
            {
                // Pin to a small downward bias so the controller stays "stuck" to slopes; launch on jump.
                if (jumpPressed) return jumpImpulse;
                return -1f;
            }
            return Mathf.Max(vY + gravity * dt, terminalVelocity); // accumulate, clamp to terminal
        }

        public static bool FellInPit(float y, float killPlaneY) => y < killPlaneY;
    }
}
