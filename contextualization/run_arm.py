#!/usr/bin/env python
"""run_arm.py -- train one arm, back it up to the HF Hub, evaluate it, log everything to wandb.

One command per arm on the GPU box, from the repo root:

    uv run python contextualization/run_arm.py --arm C --exp-dir /workspace/ctx_experiment_v3 --depth 24
    uv run python contextualization/run_arm.py --arm R --exp-dir ... --depth 24 -- --device-batch-size=16

Steps (each can be skipped; the script is safe to rerun):
  1. train    torchrun scripts.base_train with NANOCHAT_BASE_DIR=<exp-dir>/arm_<ARM>, logging to
              wandb run `ctx_<ARM>_d<DEPTH>` (tags: arm_<ARM>, <exp_tag>). The run id is fixed up
              front and stored in <exp-dir>/arm_<ARM>/run_info.json so later steps attach to it.
  2. upload   final checkpoint + meta + tokenizer -> HF model repo `<ns>/ctx_<ARM>_d<DEPTH>_base`,
              commit tagged `wandb-<run_id>`; build metadata -> `<ns>/ctx_experiment_build/<exp_tag>/`.
              The HF URLs are written back onto the wandb run (config + notes).
  3. eval     eval_beliefs.py for this arm; it logs summary belief metrics to the same wandb run
              and uploads the CSV to the model repo under eval/.

Secrets: WANDB_API_KEY and HF_TOKEN from contextualization/.env (see .env.example).
Anything after a bare `--` is passed through to scripts.base_train.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctx import tracking as T  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def sh(cmd, env=None, cwd=REPO):
    print("+ " + " ".join(map(str, cmd)), flush=True)
    r = subprocess.run([str(c) for c in cmd], env=env, cwd=cwd)
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {cmd[0]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=["C", "R", "X", "E"])
    ap.add_argument("--exp-dir", required=True, help="build dir holding arm_<ARM>/ and probe_sets/")
    ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--nproc", type=int, default=None, help="GPUs for torchrun (default: all visible)")
    ap.add_argument("--run-name", default=None, help="wandb run name (default ctx_<ARM>_d<DEPTH>)")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--with-optimizer", action="store_true", help="also upload optimizer shards (several GB)")
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--no-tracking", action="store_true", help="no wandb, no HF (same as CTX_TRACKING=0)")
    ap.add_argument("train_args", nargs=argparse.REMAINDER, help="extra args for scripts.base_train after --")
    args = ap.parse_args()
    if args.no_tracking:
        os.environ["CTX_TRACKING"] = "0"

    used = T.load_env()
    print(f"env files: {used or 'none'} | wandb={'on' if T.wandb_enabled() else 'off'} | hf={'on' if T.hf_enabled() else 'off'}")
    if T.tracking_enabled():
        T.require(wandb=not args.skip_train or not args.skip_eval, hf=not args.skip_upload)

    exp_dir = Path(args.exp_dir).resolve()
    arm_dir = exp_dir / f"arm_{args.arm}"
    if not arm_dir.is_dir():
        raise SystemExit(f"no such arm dir: {arm_dir}")
    tag = T.exp_tag(exp_dir)
    name = args.run_name or T.run_name(args.arm, args.depth)
    extra = [a for a in args.train_args if a != "--"]

    # ---- wandb run identity (fixed before training so eval can attach to the same run) ----
    info = T.read_run_info(exp_dir, args.arm)
    if args.skip_train:
        run_id = info.get("wandb_run_id") or (T.new_run_id() if T.wandb_enabled() else None)
    else:
        run_id = T.new_run_id() if T.wandb_enabled() else None  # fresh training run = fresh wandb run
    T.write_run_info(exp_dir, args.arm, wandb_run_id=run_id, wandb_run_name=name,
                     wandb_project=T.wandb_project() if run_id else None, depth=args.depth, exp_tag=tag,
                     train_finished=info.get("train_finished") if args.skip_train else False)
    if run_id:
        print(f"wandb run: {name} ({run_id}) -> {T.wandb_run_url(run_id)}")

    # ---- 1. train -------------------------------------------------------------------------
    if not args.skip_train:
        env = dict(os.environ, NANOCHAT_BASE_DIR=str(arm_dir))
        if run_id:
            env.update(WANDB_RUN_ID=run_id, WANDB_TAGS=f"arm_{args.arm},{tag}",
                       WANDB_NOTES=f"arm {args.arm} on build {tag} ({exp_dir})")
        nproc = args.nproc or _visible_gpus()
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}",
               "-m", "scripts.base_train", "--", f"--depth={args.depth}",
               f"--run={name if T.wandb_enabled() else 'dummy'}", *extra]
        sh(cmd, env=env)
        T.write_run_info(exp_dir, args.arm, train_finished=True)

    # ---- 2. upload checkpoint + build metadata to HF, link from wandb ---------------------
    if not args.skip_upload and T.hf_enabled():
        url, sha, step = T.hf_upload_checkpoint(exp_dir, args.arm, args.depth, run_id=run_id,
                                                with_optimizer=args.with_optimizer, extra_tags=(tag,))
        print(f"checkpoint -> {url} (commit {sha[:8]}, step {step}, tag wandb-{run_id})")
        build_url, _ = T.hf_upload_build(exp_dir)
        print(f"build metadata -> {build_url}")
        T.write_run_info(exp_dir, args.arm, hf_checkpoint=url, hf_checkpoint_commit=sha, hf_build=build_url)
        if run_id:
            T.wandb_link_hf(run_id, hf_checkpoint=f"{url}/tree/wandb-{run_id}", hf_build=build_url)
    elif not args.skip_upload:
        print("HF upload skipped (no HF_TOKEN / tracking off)")

    # ---- 3. eval (logs to the same wandb run, uploads CSV to the model repo) --------------
    if not args.skip_eval:
        sh([sys.executable, str(REPO / "contextualization" / "eval_beliefs.py"),
            "--exp-dir", exp_dir, "--arm", args.arm, "--model-tag", f"d{args.depth}",
            "--batch-size", args.eval_batch_size], env=dict(os.environ))

    print(f"\ndone: arm {args.arm} | run_info -> {T.run_info_path(exp_dir, args.arm)}")


def _visible_gpus() -> int:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None:
        return max(1, len([d for d in vis.split(",") if d.strip()]))
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


if __name__ == "__main__":
    main()
