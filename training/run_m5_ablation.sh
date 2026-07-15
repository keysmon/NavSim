#!/bin/bash
# M5 Phase-7: 12-run ablation (4 arms x 3 seeds), SEQUENTIAL, each --initialize-from its architecture-matched
# no-RND bootstrap base. Run wrapped in caffeinate so the Mac doesn't sleep:
#   nohup caffeinate -i bash training/run_m5_ablation.sh > results/m5_ablation_orch.log 2>&1 &
# After each run, the final NavAgent.onnx is copied to NavSim/Assets/Models/M5/<run-id>.onnx for M5SearchEval.
set -u
cd /Users/hangruan/Documents/claude_code_repo/unity
ENV=Builds/M5Train.app
PORT=5100
# arm -> (config, architecture-matched base)
CFG_primary=m5_ppo;         BASE_primary=m5_base_lstm
CFG_nolstm=m5_ppo_nolstm;   BASE_nolstm=m5_base_nolstm
CFG_nornd=m5_ppo_nornd;     BASE_nornd=m5_base_lstm
CFG_baseline=m5_ppo_baseline; BASE_baseline=m5_base_nolstm
mkdir -p NavSim/Assets/Models/M5

for seed in 0 1 2; do
  for arm in primary nolstm nornd baseline; do
    cfg_var="CFG_$arm"; base_var="BASE_$arm"
    cfg=${!cfg_var}; base=${!base_var}
    rid="m5_${arm}_s${seed}"
    echo "=== [$(date '+%m-%d %H:%M:%S')] START $rid | cfg=$cfg base=$base port=$PORT ==="
    .venv-nav/bin/mlagents-learn training/configs/${cfg}.yaml \
      --run-id="$rid" --seed="$seed" --initialize-from="$base" --force \
      --env="$ENV" --num-envs=6 --no-graphics --time-scale=20 --base-port "$PORT" \
      > "results/${rid}.log" 2>&1
    if [ -f "results/$rid/NavAgent.onnx" ]; then
      cp "results/$rid/NavAgent.onnx" "NavSim/Assets/Models/M5/${rid}.onnx"
      final=$(grep 'NavAgent. Step:' "results/${rid}.log" | tail -1 | grep -oE 'Mean Reward: [-0-9.]+')
      echo "  [$(date '+%m-%d %H:%M:%S')] DONE  $rid | $final | onnx copied"
    else
      echo "  [$(date '+%m-%d %H:%M:%S')] WARN  $rid produced NO onnx (check results/${rid}.log)"
    fi
    PORT=$((PORT + 10))
    sleep 5
  done
done
echo "=== [$(date '+%m-%d %H:%M:%S')] ALL 12 RUNS COMPLETE ==="
