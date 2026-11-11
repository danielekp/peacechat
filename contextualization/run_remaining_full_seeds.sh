#!/usr/bin/env bash
# Fresh full runs: seeds 2 and 3, arms C/R/X/E, sequentially on one H100.
# Run inside tmux: bash contextualization/run_remaining_full_seeds.sh
# Keeps every checkpoint; requires 240 GiB free. Never deletes old runs or data.
# Stops on failure and refuses an existing seed-2/3 run; do not use as a resume command.
set -euo pipefail
cd /workspace/peacechat

export HF_HOME=/workspace/.cache/huggingface
export XDG_CACHE_HOME=/workspace/.cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
export WANDB_MODE=online CTX_TRACKING=1
unset WANDB_RUN_ID WANDB_RESUME

CTX_EXP=/workspace/ctx_v2_build1_recovered
test -d "$CTX_EXP"
exec 9>"$CTX_EXP/.full_training.lock"
if ! flock -n 9; then
  echo 'Another full-training launcher holds the lock; stopped.'
  exit 1
fi
if pgrep -af '[s]cripts.base_train|[c]ontextualization/run_arm.py|[e]xposure_audit.py'; then
  echo 'Training or an exposure audit is already running; stopped.'
  exit 1
fi
mkdir -p "$CTX_EXP/logs"

uv run --no-sync --env-file contextualization/.env python - "$CTX_EXP" <<'PY' 2>&1 | tee "$CTX_EXP/logs/remaining_full_seeds_$(date -u +%Y%m%dT%H%M%SZ).log"
from pathlib import Path
import json
import shlex
import shutil
import subprocess
import sys
import tempfile

root = Path(sys.argv[1]).resolve()
arms = ('C', 'R', 'X', 'E')
seeds = (2, 3)
steps = 5524
baseline = 'e19f7a8c3f5b7210aacf55e12cbf057e4a8e34fe'


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def read_json(path):
    require(path.is_file(), f'Missing required file: {path}')
    return json.loads(path.read_text())


def check_space(min_gib):
    free = shutil.disk_usage(root).free / 2**30
    require(free >= min_gib,
            f'Only {free:.1f} GiB free; this queue needs {min_gib} GiB now. '
            'Expand storage or arrange checkpoint cleanup before retrying. Nothing was deleted.')
    print(f'Free storage: {free:.1f} GiB', flush=True)


def completed(arm, seed):
    tag = f'd24_full_s{seed}'
    info = read_json(root / f'arm_{arm}/run_info_{tag}.json')
    meta = read_json(root / f'arm_{arm}/base_checkpoints/{tag}/meta_{steps:06d}.json')
    upload = read_json(root / f'eval/{tag}/beliefs_arm{arm}.upload.json')
    require(info.get('train_finished') is True and info.get('checkpoint_step') == steps,
            f'Arm {arm}, seed {seed}: full training is not recorded as complete.')
    require(meta.get('step') == steps and meta.get('init_seed') == seed
            and meta.get('data_seed') == seed and meta.get('num_iterations') == steps
            and meta.get('loss_normalization') == 'fixed_nominal',
            f'Arm {arm}, seed {seed}: final metadata does not match the experiment.')
    run_id = info.get('wandb_run_id')
    commit = info.get('hf_checkpoint_commit')
    require(run_id and commit and info.get('hf_optimizer_included') is True
            and int(info.get('hf_checkpoint_step', -1)) == steps,
            f'Arm {arm}, seed {seed}: final checkpoint/optimizer upload is not recorded.')
    require(upload.get('verified') is True and upload.get('run_id') == run_id
            and upload.get('checkpoint_commit') == commit,
            f'Arm {arm}, seed {seed}: verified evaluation upload is not recorded.')
    identity = meta.get('init_param_sha256')
    require(identity, f'Arm {arm}, seed {seed}: initialization identity is missing.')
    return identity


def snapshot_exposure(seed):
    # Never overwrite a previous snapshot, even after an interrupted attempt.
    snapshots = root / 'exposure_by_seed'
    snapshots.mkdir(exist_ok=True)
    target = snapshots / f'seed_{seed}'
    if target.exists():
        target = Path(tempfile.mkdtemp(prefix=f'seed_{seed}_snapshot_', dir=snapshots)) / 'exposure'
    shutil.copytree(root / 'exposure', target)
    print(f'Exposure reports preserved: {target}', flush=True)


def snapshot_export(arm, tag):
    export = root / f'arm_{arm}/export'
    snapshot = export / tag
    snapshot.mkdir(exist_ok=True)
    # Keep the first per-tag export; the shared manifest will advance to the next seed.
    destination = snapshot / 'checkpoint_manifest.json'
    source = destination if destination.exists() else export / 'checkpoint_manifest.json'
    manifest = read_json(source)
    info = read_json(root / f'arm_{arm}/run_info_{tag}.json')
    require(manifest.get('run_id') == info.get('wandb_run_id')
            and manifest.get('arm') == arm and manifest.get('step') == steps,
            f'Export manifest does not match arm {arm}, tag {tag}; stopped.')
    if not destination.exists():
        shutil.copy2(source, destination)


def run(command):
    print('+ ' + shlex.join(command), flush=True)
    subprocess.run(command, check=True)


changed = subprocess.check_output([
    'git', 'diff', '--name-only', '--no-renames', '-z', baseline, 'HEAD', '--'
]).split(b'\0')
allowed = {b'', b'contextualization/restart_full_E_s1.sh',
           b'contextualization/run_remaining_full_seeds.sh'}
require(not (set(changed) - allowed), 'Tracked training files differ from seed 1; stopped.')
require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no']).strip(),
        'Tracked files have uncommitted changes; stopped.')

# Check space and all destinations before changing the shared preflight or starting a GPU job.
check_space(240)
for seed in seeds:
    tag = f'd24_full_s{seed}'
    for arm in arms:
        for path in (root / f'arm_{arm}/base_checkpoints/{tag}',
                     root / f'arm_{arm}/run_info_{tag}.json'):
            require(not path.exists(), f'Existing run found: {path}. Stopped; no run will be overwritten.')
        require(not list((root / f'eval/{tag}').glob(f'beliefs_arm{arm}*')),
                f'Existing evaluation for arm {arm}, seed {seed}; stopped.')

pf = read_json(root / 'exposure/PREFLIGHT_OK.json')
expected_pf = dict(B=8, T=2048, grad_accum_steps=64, num_iterations=steps,
                   world_size=1, data_seed=1, loss_normalization='fixed_nominal')
require(all(pf.get(k) == v for k, v in expected_pf.items()),
        'Expected the original seed-1 preflight before starting this fresh queue; stopped.')
require(len({completed(arm, 1) for arm in arms}) == 1,
        'Seed-1 initialization identities differ; stopped.')

import torch
from nanochat.flash_attention import USE_FA3
require(torch.__version__ == '2.9.1+cu128', 'PyTorch differs from seed 1; stopped.')
require(USE_FA3, 'Flash Attention 3 is unavailable; stopped.')
snapshot_exposure(1)
for arm in arms:
    snapshot_export(arm, 'd24_full_s1')

for seed in seeds:
    tag = f'd24_full_s{seed}'
    run([sys.executable, 'contextualization/exposure_audit.py',
         '--exp-dir', str(root), '--arms', *arms, '--mode', 'prepacked',
         '--B', '8', '--T', '2048', '--grad-accum', '64',
         '--num-iterations', str(steps), '--world-size', '1',
         '--data-seed', str(seed), '--loss-normalization', 'fixed_nominal'])
    snapshot_exposure(seed)
    identities = set()
    for arm in arms:
        check_space(40)
        run([sys.executable, 'contextualization/run_arm.py',
             '--arm', arm, '--exp-dir', str(root), '--depth', '24', '--nproc', '1',
             '--model-tag', tag, '--init-seed', str(seed), '--data-seed', str(seed),
             '--data-loader', 'prepacked', '--loss-normalization', 'fixed_nominal',
             '--with-optimizer', '--', '--resume-from-step=-1',
             f'--num-iterations={steps}', '--device-batch-size=8',
             '--total-batch-size=1048576', '--max-seq-len=2048',
             '--window-pattern=SSSL', '--save-every=2000'])
        identities.add(completed(arm, seed))
        require(len(identities) == 1, f'Initialization differs across arms for seed {seed}; stopped.')
        snapshot_export(arm, tag)
        print(f'COMPLETE: arm {arm}, seed {seed}; checkpoint and evaluation uploads recorded.', flush=True)

print('COMPLETE: all eight remaining full runs, including uploads and evaluations.', flush=True)
PY
