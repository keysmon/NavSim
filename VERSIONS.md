# Pinned Versions (M0)

Stability-first posture (see design spec §9). Researched and verified on 2026-07-10/11.

## Confirmed and resolved in the NavSim project

- Unity: 6000.5.3f1 (installed at /Applications/Unity/Hub/Editor/6000.5.3f1/).
  Project created and packages resolved via batchmode CLI.
- com.unity.ml-agents (Unity package): 4.0.3 (latest, released 2026-04-17). RESOLVED + compiled OK.
- com.unity.ai.inference (Sentis, the Inference Engine): 2.6.1 (resolved transitively by ml-agents 4.0.3).
- ONNX opset for Sentis: must be within 7 to 25.
- WebGL inference backend: CPU only (Sentis has no GPU compute backend on WebGL).

## Python side (confirmed from research, not yet installed)

- Python (interpreter): 3.10.x, within 3.10.1 to 3.10.12 (mlagents requires `>=3.10.1, <=3.10.12`).
- mlagents (Python package): 1.1.0 (latest on PyPI, 2024-10-05). Versioned separately from the
  Unity package on purpose; the 4.0.3-vs-1.1.0 skew is expected.
- mlagents-envs: matches mlagents (1.1.0).
- torch (PyTorch): resolved by `pip install mlagents`; run `pip freeze | grep torch` after install and pin here.

## Watch-items

- Licensing (headless): the package-resolve batchmode run logged
  `[Licensing::Module] Error: Access token is unavailable; failed to update`,
  yet still compiled and exited 0. If a later batchmode op (WebGL build, `-runTests`) fails on
  licensing, open the Editor GUI once to refresh the license token, then retry headless.
- python3.10 is NOT on PATH as `python3.10`. Resolve before Task 5 (training) -
  likely `python3`, or install via pyenv (`pyenv install 3.10.12`) or conda. Must be 3.10.1-3.10.12.
- Vercel CLI: 54.4.1 present (55.0.0 available; only needed at Task 7).

## Toolchain status (checked 2026-07-11)

- git: 2.50.1 (OK).
- Unity 6000.5.3f1: installed; batchmode CLI verified working (Personal license, user hangruan).
- Vercel CLI: 54.4.1.
