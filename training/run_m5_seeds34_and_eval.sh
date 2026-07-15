#!/bin/bash
# M5 Phase-7 add-seeds (pre-registered n=3 -> n=5): 8 more runs (seeds 3,4 x 4 arms), then the n=5 paired eval +
# rliable analysis, fully chained. Launch: nohup caffeinate -i bash training/run_m5_seeds34_and_eval.sh > \
#   results/m5_n5_orch.log 2>&1 &
set -u
cd /Users/hangruan/Documents/claude_code_repo/unity
ENV=Builds/M5Train.app
UNITY=/Applications/Unity/Hub/Editor/6000.5.3f1/Unity.app/Contents/MacOS/Unity
PROJ=/Users/hangruan/Documents/claude_code_repo/unity/NavSim
CFG_primary=m5_ppo;           BASE_primary=m5_base_lstm
CFG_nolstm=m5_ppo_nolstm;     BASE_nolstm=m5_base_nolstm
CFG_nornd=m5_ppo_nornd;       BASE_nornd=m5_base_lstm
CFG_baseline=m5_ppo_baseline; BASE_baseline=m5_base_nolstm
mkdir -p NavSim/Assets/Models/M5
PORT=5300

# --- Phase 1: 8 additional training runs (seeds 3, 4) ---
for seed in 3 4; do
  for arm in primary nolstm nornd baseline; do
    cfg_var="CFG_$arm"; base_var="BASE_$arm"; cfg=${!cfg_var}; base=${!base_var}
    rid="m5_${arm}_s${seed}"
    echo "=== [$(date '+%m-%d %H:%M:%S')] START $rid | cfg=$cfg base=$base port=$PORT ==="
    .venv-nav/bin/mlagents-learn "training/configs/${cfg}.yaml" \
      --run-id="$rid" --seed="$seed" --initialize-from="$base" --force \
      --env="$ENV" --num-envs=6 --no-graphics --time-scale=20 --base-port "$PORT" \
      > "results/${rid}.log" 2>&1
    if [ -f "results/$rid/NavAgent.onnx" ]; then
      cp "results/$rid/NavAgent.onnx" "NavSim/Assets/Models/M5/${rid}.onnx"
      echo "  [$(date '+%m-%d %H:%M:%S')] DONE  $rid | $(grep 'NavAgent. Step:' "results/${rid}.log" | tail -1 | grep -oE 'Mean Reward: [-0-9.]+')"
    else
      echo "  [$(date '+%m-%d %H:%M:%S')] WARN  $rid produced NO onnx"
    fi
    PORT=$((PORT + 10)); sleep 5
  done
done
echo "=== [$(date '+%m-%d %H:%M:%S')] 8 SEED-3/4 RUNS COMPLETE ==="

# --- Phase 2: n=5 paired eval (batchmode, bridge-free; Seeds now {0..4} in M5SearchEval) ---
pkill -TERM -f "Unity.app/Contents/MacOS/Unity" 2>/dev/null; sleep 6
pkill -KILL -f "Unity.app/Contents/MacOS/Unity" 2>/dev/null; sleep 2
rm -f NavSim/Temp/UnityLockfile training/eval/m5_search.csv
echo "=== [$(date '+%m-%d %H:%M:%S')] running batchmode eval (20 models x 4 levels x 25) ==="
"$UNITY" -batchmode -nographics -projectPath "$PROJ" \
  -executeMethod M5EvalBatch.RunHeadless -logFile results/m5_eval_n5.log
echo "  eval rows: $(wc -l < training/eval/m5_search.csv 2>/dev/null || echo 0)"

# --- Phase 3: rliable analysis (n=5) ---
MPLBACKEND=Agg .venv-nav/bin/python training/eval/plot_m5_search.py > results/m5_n5_analysis.log 2>&1
echo "=== [$(date '+%m-%d %H:%M:%S')] N=5 ANALYSIS ==="
cat results/m5_n5_analysis.log
echo "=== N=5 PIPELINE COMPLETE [$(date '+%m-%d %H:%M:%S')] ==="
