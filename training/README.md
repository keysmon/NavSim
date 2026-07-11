# NavSim Training (M0)

ML-Agents PPO training for the single-agent pipeline-smoke milestone.

## Prerequisites

- Python 3.10.x (range 3.10.1 to 3.10.12). If `python3.10` is not on PATH, install it
  (for example via pyenv: `pyenv install 3.10.12`) or use a conda env pinned to 3.10.
- The `NavSim` Unity project built once (or open in the Editor) so the trainer has an env to talk to.

## Setup

```bash
# from the repo root
python3.10 -m venv .venv-nav
source .venv-nav/bin/activate
pip install -r training/requirements.txt
mlagents-learn --help   # verify the install
```

## Sanity run (attached to the Editor)

```bash
mlagents-learn training/configs/m0_ppo.yaml --run-id=m0_ppo_sanity --force
# When prompted, press Play in the NavSim Training scene.
# Watch the console: steps tick up and Mean Reward is reported. Stop after ~50k steps
# once reward trends upward.
```

## Full run (headless, parallel envs)

Build the Training scene to a headless player at `NavSim/Builds/NavSimTraining` first, then:

```bash
mlagents-learn training/configs/m0_ppo.yaml \
  --run-id=m0_ppo_01 \
  --env=NavSim/Builds/NavSimTraining \
  --num-envs=4 --no-graphics --time-scale=20 --force
```

## Inspect

```bash
tensorboard --logdir results
```

Convergence gate: `NavAgent/Cumulative Reward` rises and plateaus while `Episode Length` falls
(the agent reaches the goal faster over time). The trained policy is written to
`results/m0_ppo_01/NavAgent.onnx`; copy it into `NavSim/Assets/Models/` for the demo (Task 6).

Note: `results/` is git-ignored (logs and models are large).
