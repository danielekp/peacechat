#!/usr/bin/env bash
# Fresh restart of funded arm E, seed 1, on the original RunPod network volume.
# Upload this standalone script to /workspace/restart_full_E_s1.sh and run in tmux:
#   bash /workspace/restart_full_E_s1.sh
# Keep the training checkout at e19f7a8; this script does not update the repository.
# Previous E outputs are moved into a new archive, not copied or deleted.

(
set -euo pipefail
cd /workspace/peacechat
export HF_HOME=/workspace/.cache/huggingface
export XDG_CACHE_HOME=/workspace/.cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
export WANDB_MODE=online CTX_TRACKING=1
unset WANDB_RUN_ID WANDB_RESUME

if pgrep -af '[s]cripts.base_train|[c]ontextualization/run_arm.py'; then
  echo 'Training is already running; stopped.'
  exit 1
fi

uv run --no-sync --env-file contextualization/.env python - <<'PY'
from pathlib import Path
import shutil, subprocess, tempfile
import torch
from nanochat.flash_attention import USE_FA3

root = Path('/workspace/ctx_v2_build1_recovered')
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == 'e19f7a8c3f5b7210aacf55e12cbf057e4a8e34fe', 'Code differs from C/R/X; stop here.'
subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'], check=True)
assert torch.__version__ == '2.9.1+cu128', 'PyTorch differs from C/R/X; stop here.'
assert USE_FA3, 'Flash Attention 3 is unavailable; stop here.'
assert (root / 'arm_E/prepacked/prepack_manifest.json').is_file(), 'Original prepacked E data is missing.'
assert (root / 'exposure/PREFLIGHT_OK.json').is_file(), 'Original preflight is missing.'
free = shutil.disk_usage(root).free / 2**30
assert free >= 40, f'Only {free:.1f} GiB free; need 40 GiB before restarting.'

paths = [root / p for p in (
    'arm_E/base_checkpoints/d24_full_s1',
    'arm_E/run_info_d24_full_s1.json',
    'arm_E/export/checkpoint_manifest.json',
)]
paths += list((root / 'eval/d24_full_s1').glob('beliefs_armE*'))
archive = Path(tempfile.mkdtemp(prefix='superseded_E_s1_', dir=root))
for src in paths:
    if src.exists():
        dst = archive / src.relative_to(root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
print(f'Previous E outputs moved to {archive}; free space {free:.1f} GiB.')
PY

mkdir -p /workspace/ctx_v2_build1_recovered/logs
uv run --no-sync --env-file contextualization/.env python contextualization/run_arm.py \
  --arm E --exp-dir /workspace/ctx_v2_build1_recovered \
  --depth 24 --nproc 1 --model-tag d24_full_s1 \
  --run-name ctx_E_d24_full_s1_restart_20261106 \
  --init-seed 1 --data-seed 1 --data-loader prepacked \
  --loss-normalization fixed_nominal --with-optimizer -- \
  --resume-from-step=-1 --num-iterations=5524 \
  --device-batch-size=8 --total-batch-size=1048576 \
  --max-seq-len=2048 --window-pattern=SSSL --save-every=2000 \
  2>&1 | tee "/workspace/ctx_v2_build1_recovered/logs/full_E_s1_restart_$(date -u +%Y%m%dT%H%M%SZ).log"
)
