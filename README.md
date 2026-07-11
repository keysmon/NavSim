# NavSim - Multi-Agent Navigation Simulator

A 2D Unity + ML-Agents simulator where a shared-policy crowd learns to *search out* hidden goals
and avoid collisions with each other and with obstacles.
Trained with reinforcement learning (PPO, and later MA-POCA), exported to ONNX, and playable
in-browser via a Unity WebGL build with in-engine inference (Unity Sentis).

## Status

M0 (pipeline smoke) in progress: proving the full toolchain end to end
(Unity -> ML-Agents -> ONNX -> Sentis -> WebGL -> Vercel) on a single agent reaching a visible goal.

Live demo: _(added at the end of M0)_

## Layout

- `NavSim/` - the Unity project (created in M0 Task 1).
- `training/` - Python ML-Agents trainer configs and environment.
- `web/` - the WebGL build output deployed to Vercel.
- `docs/superpowers/` - design spec and milestone plans (local only, not pushed).

## Tech stack

Unity 6 LTS, com.unity.ml-agents 4.0.x, Python 3.10.x, Unity Inference Engine (Sentis) 2.x, WebGL, Vercel.
See `VERSIONS.md` for exact pins.
