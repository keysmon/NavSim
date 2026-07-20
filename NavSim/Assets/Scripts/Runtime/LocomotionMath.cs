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

        // Same-height horizontal jump reach under the SAME discrete integration NavAgent runs
        // (vY updated via NextVerticalVelocity, then position += vY*dt). Used by CourseSpec tests
        // to prove every direct-path gap is jumpable with margin (advisor T2).
        public static float MaxJumpDistance(float jumpImpulse, float gravity, float dt,
            float horizontalSpeed, float terminalVelocity)
        {
            float vY = jumpImpulse, y = 0f;
            int steps = 0;
            while (steps < 10000)
            {
                y += vY * dt;
                if (y <= 0f) break;
                steps++;
                vY = NextVerticalVelocity(vY, false, false, jumpImpulse, gravity, dt, terminalVelocity);
            }
            return steps * horizontalSpeed * dt;
        }
    }
}
