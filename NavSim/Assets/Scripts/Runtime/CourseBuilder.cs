using System.Collections.Generic;
using UnityEngine;

namespace NavSim.Runtime
{
    // Instantiates the pure CourseSpec geometry as real runtime primitives (the showcase spine's
    // visible body). Build() is the whole runtime contract: it destroys the previous stage's pieces,
    // spawns one primitive per CoursePiece, paints it from a keyed, cached material, and republishes
    // CurrentLayout for NavEnvironment (spawn/goal/pit-respawn anchors). No Update loop, no per-step
    // allocation — the demo UI calls Build() only on a stage switch.
    //
    // COORDINATE CONTRACT: pieces are placed in WORLD space (transform.position/rotation), matching how
    // NavEnvironment places the agent and goal triad at the layout's world SpawnPos/GoalSlots. So the
    // builder GameObject must sit at the world origin with identity rotation and unit scale (Task 5
    // scene setup); parenting is for hierarchy + layer inheritance + teardown only.
    //
    // MATERIAL CONTRACT (WebGL, T11): materials are `new Material(litShader)` with the nine capstone
    // colours as code literals. `litShader` MUST be wired to the built-in "Standard" shader asset in the
    // Editor (Task 5) — a runtime Shader.Find("Standard") is used ONLY as an Editor/EditMode fallback and
    // does NOT survive shader-stripping in a player build. If litShader is unassigned in a WebGL build the
    // course renders magenta.
    public class CourseBuilder : MonoBehaviour
    {
        [Tooltip("Built-in \"Standard\" shader asset. MUST be wired (Task 5) or the course renders magenta " +
                 "in a player build — a runtime Shader.Find is only an in-Editor fallback.")]
        [SerializeField] private Shader litShader;

        public CourseLayout CurrentLayout { get; private set; }

        private readonly List<GameObject> _pieces = new List<GameObject>();
        private readonly Dictionary<string, Material> _mats = new Dictionary<string, Material>();

        // Destroy the previous stage's pieces, build `stage` (mirrored/variant) from CourseSpec, instantiate
        // each piece as a world-placed primitive on the builder's layer, cache CurrentLayout, then sync the
        // physics transforms so a same-frame Linecast/NavAgent step sees the new colliders.
        public void Build(int stage, bool mirrored, CourseVariant variant)
        {
            ClearPieces();
            CourseLayout layout = CourseSpec.Build(stage, mirrored, variant);
            for (int i = 0; i < layout.Pieces.Length; i++)
            {
                CoursePiece piece = layout.Pieces[i];
                GameObject go = GameObject.CreatePrimitive(piece.Primitive);
                go.name = piece.Name;
                go.transform.SetParent(transform, worldPositionStays: true);
                go.transform.position = piece.Pos;
                go.transform.rotation = piece.Rot;
                go.transform.localScale = piece.Scale;
                go.layer = gameObject.layer; // Task 5 puts the builder on the static-geometry layer for Linecast occlusion
                if (piece.Tag != null) go.tag = piece.Tag;
                Renderer rend = go.GetComponent<Renderer>();
                if (rend != null) rend.sharedMaterial = MaterialFor(piece.MaterialKey);
                _pieces.Add(go);
            }
            CurrentLayout = layout;
            Physics.SyncTransforms();
        }

        private void ClearPieces()
        {
            for (int i = 0; i < _pieces.Count; i++)
            {
                if (_pieces[i] == null) continue;
                if (Application.isPlaying) Destroy(_pieces[i]);
                else DestroyImmediate(_pieces[i]);
            }
            _pieces.Clear();
        }

        // One cached material per key (created lazily). Build() runs on every stage switch, so a fresh
        // `new Material` per Build would leak across switches — reuse instead.
        private Material MaterialFor(string key)
        {
            if (_mats.TryGetValue(key, out Material cached) && cached != null) return cached;
            Material mat = new Material(ResolveShader()) { name = "Course_" + key };
            mat.SetColor("_Color", ColorFor(key));
            mat.SetFloat("_Glossiness", 0.15f); // matte, uniform across the course (CapstoneSceneSetup idiom)
            mat.SetFloat("_Metallic", 0f);
            _mats[key] = mat;
            return mat;
        }

        // Prefer the serialized ref (a real asset, never stripped from a build). Fall back to Shader.Find
        // ONLY for Editor/EditMode convenience — that fallback does not rescue a stripped player build.
        private Shader ResolveShader()
        {
            if (litShader != null) return litShader; // explicit != null (Unity fake-null; never ?? on a UnityEngine.Object)
            Shader found = Shader.Find("Standard");
            if (found == null)
                Debug.LogWarning("[CourseBuilder] litShader unassigned AND Shader.Find(\"Standard\") returned null " +
                                 "— course will render magenta. Wire litShader to the Standard shader asset (Task 5).");
            return found;
        }

        // The nine capstone material colours (brief Step 1), as code literals — single source of truth.
        private static Color ColorFor(string key)
        {
            switch (key)
            {
                case "floor":        return new Color(0.55f, 0.55f, 0.58f);
                case "landing":      return new Color(0.60f, 0.60f, 0.63f);
                case "curb":         return new Color(0.30f, 0.33f, 0.37f);
                case "ramp":         return new Color(0.82f, 0.58f, 0.28f);
                case "tread":        return new Color(0.52f, 0.33f, 0.12f);
                case "wall":         return new Color(0.35f, 0.42f, 0.52f);
                case "pit":          return new Color(0.05f, 0.05f, 0.06f);
                case "decoyPad":     return new Color(0.50f, 0.50f, 0.53f);
                case "spawnMarker":  return new Color(0.74f, 0.76f, 0.79f);
                default:             return Color.magenta; // unknown key — loud, so a typo is caught in-scene
            }
        }

        // Runtime-created materials are not owned by any asset; destroy them so a scene unload doesn't leak.
        private void OnDestroy()
        {
            foreach (Material mat in _mats.Values)
                if (mat != null)
                {
                    if (Application.isPlaying) Destroy(mat);
                    else DestroyImmediate(mat);
                }
            _mats.Clear();
        }
    }
}
