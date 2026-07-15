#!/bin/bash
# M5 Phase-7 PRE-FLIGHT SMOKE: run all 4 arms (seed 0, ~150k) --initialize-from their matched bases, to confirm
# each un-freezes (resumes competently, doesn't re-freeze) + the init-from arch-match + configs run, BEFORE the 9h.
# A run is HEALTHY if its reward stays well above the ~-3 frozen floor (it resumes a competent base). Cleans up after.
set -u
cd /Users/hangruan/Documents/claude_code_repo/unity
ENV=Builds/M5Train.app
PORT=5200
CFG_primary=m5_ppo;           BASE_primary=m5_base_lstm
CFG_nolstm=m5_ppo_nolstm;     BASE_nolstm=m5_base_nolstm
CFG_nornd=m5_ppo_nornd;       BASE_nornd=m5_base_lstm
CFG_baseline=m5_ppo_baseline; BASE_baseline=m5_base_nolstm

for arm in primary nolstm nornd baseline; do
  cfg_var="CFG_$arm"; base_var="BASE_$arm"; cfg=${!cfg_var}; base=${!base_var}
  smk="/tmp/smoke_${arm}.yaml"
  # short config: max_steps 150k (mlagents has no --max-steps flag)
  sed 's/max_steps: 3000000.*/max_steps: 150000/' "training/configs/${cfg}.yaml" > "$smk"
  rid="smoke_${arm}"
  echo "=== [$(date '+%H:%M:%S')] SMOKE $arm (cfg=$cfg base=$base) ==="
  .venv-nav/bin/mlagents-learn "$smk" --run-id="$rid" --seed=0 --initialize-from="$base" --force \
    --env="$ENV" --num-envs=6 --no-graphics --time-scale=20 --base-port "$PORT" \
    > "results/${rid}.log" 2>&1
  traj=$(grep 'NavAgent. Step:' "results/${rid}.log" | sed -E 's/.*Step: ([0-9]+)\..*Mean Reward: (-?[0-9.]+).*/\1:\2/' | tr '\n' ' ')
  echo "  $arm trajectory: $traj"
  PORT=$((PORT + 10)); sleep 4
  rm -f "$smk"
done
echo "=== SMOKE DONE — healthy = reward NOT stuck near -3 (each resumed a competent base) ==="
rm -rf results/smoke_* 2>/dev/null
