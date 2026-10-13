#!/usr/bin/env python
"""Manually (re)upload an arm's checkpoint, the build metadata, or eval outputs to the HF Hub.

run_arm.py does all of this automatically; this is the escape hatch for a box where training
already finished, or to re-push after a fix. Same names as everywhere else:
    model   <ns>/ctx_<ARM>_d<DEPTH>_base     (commit tagged wandb-<run_id> when known)
    dataset <ns>/ctx_experiment_build/<exp_tag>/

Usage (from the repo root, HF_TOKEN in contextualization/.env):
    uv run python contextualization/load_artifact_hf.py --exp-dir /workspace/ctx_experiment_v3 --arm C
    uv run python contextualization/load_artifact_hf.py --exp-dir ... --arm R --depth 24 --wandb-run abc123
    uv run python contextualization/load_artifact_hf.py --exp-dir ... --build          # metadata only
    uv run python contextualization/load_artifact_hf.py --exp-dir ... --arm X --eval   # eval CSV only
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctx import tracking as T  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--exp-dir", required=True)
p.add_argument("--arm", choices=["C", "R", "X", "E"])
p.add_argument("--depth", type=int, default=None, help="default: from run_info.json, else 24")
p.add_argument("--wandb-run", default=None, help="wandb run id to tag the commit with (default: run_info.json)")
p.add_argument("--build", action="store_true", help="upload probe sets / manifest / summary (only)")
p.add_argument("--eval", action="store_true", help="upload eval/beliefs_arm<ARM>*.{csv,json} (only)")
p.add_argument("--with-optimizer", action="store_true", help="also upload optimizer shards (several GB)")
p.add_argument("--public", action="store_true", help="create the repo public (default: HF_PRIVATE)")
args = p.parse_args()

T.load_env()
if args.public:
    os.environ["HF_PRIVATE"] = "0"
T.require(hf=True)

if args.build or not args.arm:
    url, sha = T.hf_upload_build(args.exp_dir)
    print(f"build metadata -> {url} (commit {sha[:8]})")
    if not args.arm:
        sys.exit(0)

info = T.read_run_info(args.exp_dir, args.arm)
depth = args.depth or info.get("depth") or 24
run_id = args.wandb_run or info.get("wandb_run_id")

if args.eval:
    eval_dir = Path(args.exp_dir) / "eval"
    files = {f"eval/{f.name}": f for f in sorted(eval_dir.glob(f"beliefs_arm{args.arm}*"))}
    if not files:
        sys.exit(f"no eval outputs for arm {args.arm} under {eval_dir}")
    url, sha = T.hf_upload_files(T.model_repo(args.arm, depth), files,
                                 commit_message=f"eval arm {args.arm}" + (f" (wandb {run_id})" if run_id else ""),
                                 tags=(f"wandb-{run_id}",) if run_id else ())
    print(f"eval -> {url}/tree/main/eval (commit {sha[:8]})")
elif not args.build:
    url, sha, step = T.hf_upload_checkpoint(args.exp_dir, args.arm, depth, run_id=run_id,
                                            with_optimizer=args.with_optimizer, extra_tags=(T.exp_tag(args.exp_dir),))
    print(f"checkpoint step {step} -> {url} (commit {sha[:8]}" + (f", tag wandb-{run_id})" if run_id else ")"))
    T.write_run_info(args.exp_dir, args.arm, hf_checkpoint=url, hf_checkpoint_commit=sha, depth=depth)
    if run_id and T.wandb_enabled():
        T.wandb_link_hf(run_id, hf_checkpoint=f"{url}/tree/wandb-{run_id}")
        print(f"linked from wandb run {run_id}")
