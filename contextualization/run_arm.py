#!/usr/bin/env python
"""run_arm.py -- train one arm, back it up to the HF Hub, evaluate it, log everything to wandb.

One command per arm on the GPU box, from the repo root:

    uv run python contextualization/run_arm.py --arm C --exp-dir /workspace/ctx_experiment_v3 --depth 24
    uv run python contextualization/run_arm.py --arm R --exp-dir ... --depth 24 -- --device-batch-size=16
    uv run python contextualization/run_arm.py --arm X --exp-dir ... --depth 24 \
        --init-seed 7 --data-seed 11 --data-loader prepacked

Steps (each can be skipped; the script is safe to rerun):
  1. train    scripts.base_train (torchrun when >1 GPU) with NANOCHAT_BASE_DIR=<exp-dir>/arm_<ARM>, logging
              to wandb run `ctx_<ARM>_d<DEPTH>` (tags: arm_<ARM>, <exp_tag>). The run id is fixed up
              front and stored in <exp-dir>/arm_<ARM>/run_info.json so later steps attach to it.
              run_info.json also records the exact training command, the git commit (+ dirty flag),
              and the init / data seeds and loader that were passed (review finding 19).
  2. export   arm_<ARM>/export/checkpoint_manifest.json (local: file hashes, tokenizer fingerprint,
              bindings from meta_<step>.json) -- written before any upload, also with --skip-upload.
  3. upload   final checkpoint + meta + tokenizer (+ optimizer with --with-optimizer) -> HF model repo
              `<ns>/ctx_<ARM>_d<DEPTH>_base`, commit tagged `wandb-<run_id>`; build metadata ->
              `<ns>/ctx_experiment_build/<exp_tag>/`. The HF URLs are written back onto the wandb run.
              The upload refuses a tokenizer that does not match the checkpoint's bindings; a legacy
              checkpoint without bindings needs --allow-unbound.
  4. eval     eval_beliefs.py for this arm; it logs summary belief metrics to the same wandb run
              and uploads the CSV to the model repo under eval/.

Secrets: WANDB_API_KEY and HF_TOKEN from contextualization/.env (see .env.example).
Anything after a bare `--` is passed through to scripts.base_train.
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctx import tracking as T  # noqa: E402
from ctx.protocol import code_revision  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA_LOADERS = ("bestfit", "prepacked")


def sh(cmd, env=None, cwd=REPO):
    print("+ " + " ".join(map(str, cmd)), flush=True)
    r = subprocess.run([str(c) for c in cmd], env=env, cwd=cwd)
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {cmd[0]}")


def build_train_command(depth: int, run_name: str, nproc: int, extra=(), *, init_seed: int | None = None,
                        data_seed: int | None = None, data_loader: str | None = None,
                        python: str | None = None) -> list[str]:
    """The exact scripts.base_train invocation (torchrun only when nproc > 1). The seed / loader flags
    are forwarded with the names scripts/base_train.py defines (--init-seed, --data-seed,
    --data-loader {bestfit,prepacked}) and only when given, so base_train's defaults otherwise apply."""
    if data_loader is not None and data_loader not in DATA_LOADERS:
        raise ValueError(f"data_loader must be one of {DATA_LOADERS}, got {data_loader!r}")
    flags = [f"--depth={depth}", f"--run={run_name}"]
    if init_seed is not None:
        flags.append(f"--init-seed={int(init_seed)}")
    if data_seed is not None:
        flags.append(f"--data-seed={int(data_seed)}")
    if data_loader is not None:
        flags.append(f"--data-loader={data_loader}")
    flags += [str(a) for a in extra]
    py = python or sys.executable
    if nproc > 1:
        return [py, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}",
                "-m", "scripts.base_train", "--", *flags]
    return [py, "-m", "scripts.base_train", *flags]  # single GPU / CPU: no launcher needed


def training_record(cmd, env_extra: dict | None = None, *, init_seed=None, data_seed=None,
                    data_loader=None, cwd=REPO) -> dict:
    """What run_info.json stores about a training launch: the exact command, the environment
    variables we set for it, the seeds / loader as passed (None = base_train default), and the
    code revision (git commit + dirty flag) of the checkout that launched it."""
    cmd = [str(c) for c in cmd]
    return {
        "train_command": cmd,
        "train_command_str": shlex.join(cmd),
        "train_cwd": str(cwd),
        "train_env": dict(env_extra or {}),
        "init_seed": init_seed, "data_seed": data_seed, "data_loader": data_loader,
        "code": code_revision(cwd),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=["C", "R", "X", "E"])
    ap.add_argument("--exp-dir", required=True, help="build dir holding arm_<ARM>/ and probe_sets/")
    ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--nproc", type=int, default=None, help="GPUs for torchrun (default: all visible)")
    ap.add_argument("--run-name", default=None, help="wandb run name (default ctx_<ARM>_d<DEPTH>)")
    ap.add_argument("--init-seed", type=int, default=None, help="forwarded to base_train --init-seed (model init)")
    ap.add_argument("--data-seed", type=int, default=None, help="forwarded to base_train --data-seed (data order)")
    ap.add_argument("--data-loader", choices=DATA_LOADERS, default=None, help="forwarded to base_train --data-loader")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--with-optimizer", action="store_true", help="also upload optimizer shards (several GB)")
    ap.add_argument("--allow-unbound", action="store_true",
                    help="upload a legacy checkpoint whose meta has no `bindings` block (WARNING is printed)")
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
        env_extra = {"NANOCHAT_BASE_DIR": str(arm_dir)}
        if run_id:
            env_extra.update(WANDB_RUN_ID=run_id, WANDB_TAGS=f"arm_{args.arm},{tag}",
                             WANDB_NOTES=f"arm {args.arm} on build {tag} ({exp_dir})")
        env = dict(os.environ, **env_extra)
        nproc = args.nproc or _visible_gpus()
        cmd = build_train_command(args.depth, name if T.wandb_enabled() else "dummy", nproc, extra,
                                  init_seed=args.init_seed, data_seed=args.data_seed, data_loader=args.data_loader)
        rec = training_record(cmd, env_extra, init_seed=args.init_seed, data_seed=args.data_seed,
                              data_loader=args.data_loader)
        T.write_run_info(exp_dir, args.arm, **rec)  # recorded BEFORE launch so a crashed run is still described
        sh(cmd, env=env)
        T.write_run_info(exp_dir, args.arm, train_finished=True)

    # ---- 2. local export manifest (hashes, tokenizer fingerprint, bindings) before any upload ----
    try:
        out = T.export_manifest(exp_dir, args.arm, args.depth, with_optimizer=args.with_optimizer)
        print(f"checkpoint manifest -> {out}")
        T.write_run_info(exp_dir, args.arm, export_manifest=str(out))
    except SystemExit as e:
        if not args.skip_train or not args.skip_upload:
            raise
        print(f"NOTE: no export manifest written ({e})")

    # ---- 3. upload checkpoint + build metadata to HF, link from wandb ---------------------
    if not args.skip_upload and T.hf_enabled():
        url, sha, step = T.hf_upload_checkpoint(exp_dir, args.arm, args.depth, run_id=run_id,
                                                with_optimizer=args.with_optimizer, extra_tags=(tag,),
                                                allow_unbound=args.allow_unbound)
        print(f"checkpoint -> {url} (commit {sha[:8]}, step {step}, tag wandb-{run_id})")
        build_url, _ = T.hf_upload_build(exp_dir)
        print(f"build metadata -> {build_url}")
        T.write_run_info(exp_dir, args.arm, hf_checkpoint=url, hf_checkpoint_commit=sha, hf_build=build_url,
                         hf_checkpoint_step=int(step), hf_optimizer_included=bool(args.with_optimizer))
        if run_id:
            T.wandb_link_hf(run_id, hf_checkpoint=f"{url}/tree/wandb-{run_id}", hf_build=build_url)
    elif not args.skip_upload:
        print("HF upload skipped (no HF_TOKEN / tracking off)")

    # ---- 4. eval (logs to the same wandb run, uploads CSV to the model repo) --------------
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
