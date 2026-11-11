#!/usr/bin/env bash
# Supplementary probes (wrong_subject, wrapper_conditional_E; docs/ANALYSIS.md "Supplementary probes") on every
# finished full checkpoint: scores <build>/eval/d24_full_s<S>/supp_beliefs_arm<A>.csv next to the main evaluation,
# which is not touched. Run AFTER the training queue has finished: this script takes the queue's lock and refuses
# while a training / exposure job runs. Outputs that exist are skipped, so it can be re-run after an interruption.
#   bash contextualization/run_supplement_eval.sh                       # seeds 1 2 3, arms C R X E
#   SEEDS="1 2" ARMS="C R" bash contextualization/run_supplement_eval.sh
#   EXTRA="--no-tracking" bash contextualization/run_supplement_eval.sh   # no wandb / HF upload
# About 2 minutes per checkpoint on the H100 (model load + sha256 + 17,500 short sequences).
set -euo pipefail
cd /workspace/peacechat

export HF_HOME=/workspace/.cache/huggingface
export XDG_CACHE_HOME=/workspace/.cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
export WANDB_MODE=online CTX_TRACKING=1
unset WANDB_RUN_ID WANDB_RESUME

CTX_EXP=${CTX_EXP:-/workspace/ctx_v2_build1_recovered}
SEEDS=${SEEDS:-"1 2 3"}
ARMS=${ARMS:-"C R X E"}
EXTRA=${EXTRA:-}
test -d "$CTX_EXP"
exec 9>"$CTX_EXP/.full_training.lock"
if ! flock -n 9; then
  echo 'The full-training launcher holds the lock (queue still running); stopped.'
  exit 1
fi
if pgrep -af '[s]cripts.base_train|[c]ontextualization/run_arm.py|[e]xposure_audit.py'; then
  echo 'Training or an exposure audit is running; stopped.'
  exit 1
fi
if [ ! -f "$CTX_EXP/manifest.csv" ] && [ ! -f "$CTX_EXP/manifest.csv.gz" ]; then
  echo "No build manifest under $CTX_EXP: the arm-E frames cannot be verified; stopped."
  exit 1
fi
mkdir -p "$CTX_EXP/logs"
LOG="$CTX_EXP/logs/supplement_eval_$(date -u +%Y%m%dT%H%M%SZ).log"

{
  echo "supplementary evaluation: build $CTX_EXP, seeds [$SEEDS], arms [$ARMS], commit $(git rev-parse HEAD)"
  for seed in $SEEDS; do
    tag="d24_full_s${seed}"
    for arm in $ARMS; do
      main="$CTX_EXP/eval/$tag/beliefs_arm${arm}.csv"
      out="$CTX_EXP/eval/$tag/supp_beliefs_arm${arm}.csv"
      if [ ! -f "$main" ]; then
        echo "SKIP arm $arm seed $seed: no main evaluation ($main); the run is not finished"
        continue
      fi
      if [ -f "$out" ]; then
        echo "SKIP arm $arm seed $seed: $out exists"
        continue
      fi
      echo "=== arm $arm seed $seed"
      # shellcheck disable=SC2086
      uv run --no-sync --env-file contextualization/.env python contextualization/eval_beliefs.py \
        --exp-dir "$CTX_EXP" --arm "$arm" --model-tag "$tag" --supplement --batch-size 64 $EXTRA
    done
  done
  echo 'COMPLETE: supplementary evaluation loop finished (see SKIP lines for anything left out).'
} 2>&1 | tee "$LOG"
