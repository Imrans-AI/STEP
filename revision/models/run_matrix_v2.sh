#!/usr/bin/env bash
# Sequential driver for the v2 training matrix (R1-2 / R2-4 / R2-7).
# One run at a time: 15 runs x ~16 GB GPU is not safe to parallelise, and the
# project's heavy-run rule is one heavy job at a time.
# Runs already carrying results.json are skipped, so this is resumable.
set -u
cd "$(dirname "$0")"
PY=/home/muhammad-imran/miniconda3/envs/dl_env/bin/python
mkdir -p logs_v2

# wait for any in-flight training process to finish first
while pgrep -f "train_matrix_v2.py" > /dev/null; do sleep 20; done

for prior in corrected none length_only shuffled uniform; do
  for seed in 42 43 44; do
    run="${prior}_seed${seed}"
    if [ -f "checkpoints_v2/${run}/results.json" ]; then
      echo "[driver] skip ${run} (already complete)"
      continue
    fi
    echo "[driver] start ${run} at $(date '+%F %T')"
    $PY -I train_matrix_v2.py --prior "$prior" --seed "$seed" \
        > "logs_v2/${run}.log" 2>&1
    echo "[driver] end   ${run} at $(date '+%F %T') exit=$?"
  done
done
echo "[driver] ALL RUNS COMPLETE at $(date '+%F %T')"
