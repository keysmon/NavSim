#!/usr/bin/env bash
# M7 C1-narrowed ablation: 3 arms x 3 seeds, SERIAL, C0->C1 two-stage curriculum, 2M steps each.
# Run from repo root, detached (nohup ... & disown). Ray-only arena ~480 sps -> ~70 min/run -> ~10.5h serial.
# Scope narrowed to C1 (the bridgeable cooperation regime) per probes P4/P5; see docs/M7-results.md.
set -euo pipefail
ML=.venv-nav/bin/mlagents-learn
for seed in 0 1 2; do
  for arm in poca shared selfish; do
    caffeinate -i "$ML" "training/configs/m7_${arm}.yaml" --run-id="m7_${arm}_s${seed}" --seed="$seed" \
      --env=NavSim/Builds/M7CoopTrain.app --no-graphics --time-scale=20 --base-port 5006
  done
done
echo "ALL 9 C1 RUNS COMPLETE"
