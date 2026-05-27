#!/bin/bash
# Launch 6 rolling windows (W1-W6) for 5 seeds = 30 experiments.
# A simple multi-GPU dispatcher: at most MAX_PARALLEL jobs run at once,
# each new job is placed on the least-loaded GPU.
#
# Required env:
#   OPENAI_API_KEY
# Optional env:
#   OPENAI_BASE_URL  (default: https://api.openai.com/v1)
#   MAX_PARALLEL     (default: 8)
#   GPUS             (space-separated GPU ids, default: "0 1 2 3")
#   RESULTS_DIR      (default: results)

set -euo pipefail
cd "$(dirname "$0")/.."

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
GPUS=(${GPUS:-0 1 2 3})
RESULTS_DIR="${RESULTS_DIR:-results}"
mkdir -p "$RESULTS_DIR"

SEEDS=(42 123 456 789 1024)
WINDOWS=(1 2 3 4 5 6)

# Build queue
QUEUE=()
for seed in "${SEEDS[@]}"; do
    for w in "${WINDOWS[@]}"; do
        QUEUE+=("$seed $w")
    done
done

echo "Total experiments to launch: ${#QUEUE[@]}"

QUEUE_IDX=0
TAG="gift_main_$$"

while [ $QUEUE_IDX -lt ${#QUEUE[@]} ]; do
    RUNNING=$(pgrep -af "main.py.*${TAG}" | wc -l)

    if [ $RUNNING -lt $MAX_PARALLEL ]; then
        # Pick least loaded GPU
        BEST_GPU=${GPUS[0]}
        BEST_MEM=99999
        for gpu in "${GPUS[@]}"; do
            mem=$(nvidia-smi -i $gpu --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "99999")
            if [ "$mem" -lt "$BEST_MEM" ] 2>/dev/null; then
                BEST_MEM=$mem
                BEST_GPU=$gpu
            fi
        done

        entry=(${QUEUE[$QUEUE_IDX]})
        seed=${entry[0]}
        w=${entry[1]}
        QUEUE_IDX=$((QUEUE_IDX + 1))

        echo "[$(date +%H:%M:%S)] W${w}_seed_${seed} -> GPU ${BEST_GPU} (mem=${BEST_MEM}MB, running=${RUNNING})"
        CUDA_VISIBLE_DEVICES=$BEST_GPU nohup python -u main.py \
            --config "configs/config_W${w}.yaml" \
            --experiment_name "${TAG}_W${w}_seed_${seed}" \
            --seed "$seed" \
            > "${RESULTS_DIR}/W${w}_seed_${seed}.log" 2>&1 &
        sleep 2
    else
        sleep 60
    fi
done

echo "=== All ${#QUEUE[@]} experiments launched ==="
