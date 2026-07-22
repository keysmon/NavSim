using UnityEngine;

namespace NavSim.Runtime
{
    // Spectator-camera director for the showcase demo. Each frame it reads the CURRENT course stage's hero pose
    // (CourseLayout.CameraPos/CameraLookAt) off the live CourseBuilder and smooth-damps the camera toward it, so a
    // stage switch or a "New layout" re-roll (which may flip the mirror, negating CameraPos.x) glides rather than
    // cuts. The pose is authored per-stage in CourseSpec and mirror-corrected there, so this component needs no
    // per-stage knowledge — it just follows whatever layout the env last built.
    //
    // Runs off the referenced `cam`'s transform (not this GameObject's) so the rig can live on the camera or beside
    // it. Idle (does nothing) when the env has no course wired — every non-course scene keeps its own camera.
    public class DemoCameraRig : MonoBehaviour
    {
        [Tooltip("The spectator camera this rig drives (its transform is moved each frame).")]
        [SerializeField] private Camera cam;
        [SerializeField] private NavEnvironment env;
        [Tooltip("Approx. seconds for the camera to settle to a new stage pose (SmoothDamp smoothTime).")]
        [SerializeField] private float settleTime = 0.5f;

        // The look point is smoothed as its own Vector3 (a camera transform stores position + rotation, not a look
        // target) and fed to LookAt each frame. Position is smoothed in-place on the transform.
        private Vector3 _lookTarget;
        private Vector3 _posVel;   // SmoothDamp velocity state (position)
        private Vector3 _lookVel;  // SmoothDamp velocity state (look point)
        private bool _initialized;

        private void LateUpdate()
        {
            if (cam == null || env == null) return;

            CourseBuilder course = env.Course;     // Unity fake-null: NavEnvironment.Course is `course` or a real null
            if (course == null) return;             // not a course scene -> idle, leave the camera alone

            CourseLayout layout = course.CurrentLayout;
            // CurrentLayout is a struct; before the first Build() it is default(CourseLayout) (all-zero, Pieces null),
            // which would aim the camera at the world origin. A built layout always has pieces — wait for one.
            if (layout.Pieces == null || layout.Pieces.Length == 0) return;

            Vector3 targetPos = layout.CameraPos;
            Vector3 targetLook = layout.CameraLookAt;

            if (!_initialized)
            {
                // First valid frame: snap exactly to the opening stage pose (no glide-in from a stale editor pose),
                // then glide on every subsequent stage switch / mirror flip.
                cam.transform.position = targetPos;
                _lookTarget = targetLook;
                _initialized = true;
            }
            else
            {
                // unscaledDeltaTime: the camera feel stays constant when the demo's speed control changes Time.timeScale.
                cam.transform.position = Vector3.SmoothDamp(
                    cam.transform.position, targetPos, ref _posVel, settleTime, Mathf.Infinity, Time.unscaledDeltaTime);
                _lookTarget = Vector3.SmoothDamp(
                    _lookTarget, targetLook, ref _lookVel, settleTime, Mathf.Infinity, Time.unscaledDeltaTime);
            }

            cam.transform.LookAt(_lookTarget);
        }
    }
}
