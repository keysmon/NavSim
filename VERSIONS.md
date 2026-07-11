# Pinned Versions (M0)

Stability-first posture (see design spec §9). Values below were researched on 2026-07-10.
Items marked CONFIRM must be verified against the official ML-Agents 4.0 release notes and
Package Manager at real install time before relying on them.

## Confirmed from research (2026-07-10)

- Python (interpreter): 3.10.x, within the range 3.10.1 to 3.10.12 (mlagents requires `>=3.10.1, <=3.10.12`).
- mlagents (Python package): 1.1.0  (latest on PyPI, released 2024-10-05).
- mlagents-envs (Python package): matches mlagents (1.1.0).
- com.unity.ml-agents (Unity package): 4.0.x  (latest; brings the Sentis-based Inference Engine).
- Unity Inference Engine (Sentis / com.unity.ai.inference): 2.x  (pulled in transitively by ml-agents).
- ONNX opset for Sentis: must be within 7 to 25.
- WebGL inference backend: CPU only (Sentis has no GPU compute backend on WebGL).

## CONFIRM at install time

- Unity: install the latest Unity 6 LTS (6000.x.y) via Unity Hub; record the exact patch here after install.
- The Unity-package (4.0.x) to Python-package (1.1.0) pairing. These are versioned independently.
  The 4.0-vs-1.1.0 skew is expected, but confirm the exact compatible pair from the com.unity.ml-agents
  4.0 Installation doc: https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/Installation.html
- torch (PyTorch): not pinned on PyPI's summary; `pip install mlagents` resolves it.
  After install, run `pip freeze | grep torch` and record the exact version here.

## Toolchain status on this machine (checked 2026-07-10)

- git: 2.50.1 (OK).
- Unity 6 + Hub: reported installed by user (record exact editor version after Task 1).
- Vercel CLI: 54.4.1 present (a minor upgrade to 55.0.0 is available; only needed at Task 7).
- python3.10: NOT found on PATH as `python3.10`. Resolve before Task 5
  (it may be installed as `python3`, or via pyenv/conda). Must land within 3.10.1 to 3.10.12.
