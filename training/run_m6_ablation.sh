#!/usr/bin/env bash
# M6 v2 full ablation: 3 arms x 3 seeds, SERIAL (one Unity player at a time; one port). Run from the repo
# root, detached (nohup/caffeinate). Pixel needs graphics (camera); ray arms run headless. Probe-measured
# wallclock: pixel ~7.1 h/seed at ~117 sps for 3M; rays ~2.1 h/run at ~400 sps -> ~34 h serial total.
set -euo pipefail
ML=.venv-nav/bin/mlagents-learn
for seed in 0 1 2; do
  caffeinate -i "$ML" training/configs/m6_pixel.yaml --run-id="m6_pixel_s$seed" --seed="$seed" \
    --env=NavSim/Builds/M6PixelTrain.app --time-scale=20 --base-port 5005
done
for seed in 0 1 2; do
  caffeinate -i "$ML" training/configs/m6_ray1.yaml --run-id="m6_ray1_s$seed" --seed="$seed" \
    --env=NavSim/Builds/M6Ray1Train.app --no-graphics --time-scale=20 --base-port 5005
  caffeinate -i "$ML" training/configs/m6_rayc.yaml --run-id="m6_rayc_s$seed" --seed="$seed" \
    --env=NavSim/Builds/M6RayCTrain.app --no-graphics --time-scale=20 --base-port 5005
done
echo "ALL 9 RUNS COMPLETE"
