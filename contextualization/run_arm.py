#!/usr/bin/env python
"""run_arm.py -- train one arm, back it up to the HF Hub, evaluate it, log everything to wandb.

One command per arm on the GPU box, from the repo root:

    uv run python contextualization/run_arm.py --arm C --exp-dir /workspace/ctx_experiment_v3 --depth 24 --data-loader bestfit
    uv run python contextualization/run_arm.py --arm R --exp-dir ... --depth 24 --data-loader prepacked -- --device-batch-size=16
    uv run python contextualization/run_arm.py --arm X --exp-dir ... --depth 24 \
        --init-seed 7 --data-seed 11 --data-loader prepacked --skip-upload

Steps (each can be skipped; the script is safe to rerun):
  0. checks   a protocol-v2 build (build_summary.json) needs an explicit --data-loader: `prepacked` (the
              corrected path; <exp-dir>/exposure/PREFLIGHT_OK.json from `exposure_audit.py --mode prepacked`
              must exist and name this arm's current prepack_dataset_id, base_train re-checks it against the
              exact run configuration) or `bestfit` (the historical loader). The checkpoint destination is
              arm_<ARM>/base_checkpoints/<MODEL_TAG> with MODEL_TAG = d<DEPTH>, or d<DEPTH>_s<INIT_SEED>
              when --init-seed is given, so replicates never overwrite each other (--model-tag overrides).
  1. train    scripts.base_train (torchrun when >1 GPU) with NANOCHAT_BASE_DIR=<exp-dir>/arm_<ARM>, logging
              to wandb run `ctx_<ARM>_<MODEL_TAG>` (tags: arm_<ARM>, <exp_tag>). The run id is fixed up
              front and stored in <exp-dir>/arm_<ARM>/run_info.json (default tag d<DEPTH>) or
              arm_<ARM>/run_info_<MODEL_TAG>.json (replicates), one file per model tag, so later steps
              attach to the right run and two seeds never overwrite each other's identity.
              run_info.json also records the exact training command, the git commit (+ dirty flag),
              the init / data seeds, loader and model tag that were passed (review finding 19), and after
              training the checkpoint's init_param_sha256 (sha256 of the parameters right after
              initialization) with a MATCH / DIFFER line against the other arms that recorded one with
              the same init seed (reviewer check 4: compare initial parameter hashes across matched arms).
  2. export   arm_<ARM>/export/checkpoint_manifest.json (local: file hashes, tokenizer fingerprint,
              bindings from meta_<step>.json) -- written before any upload, also with --skip-upload.
  3. upload   final checkpoint + meta + tokenizer (+ optimizer with --with-optimizer) -> HF model repo
              `<ns>/ctx_<ARM>_d<DEPTH>_base`, commit tagged `wandb-<run_id>`; build metadata ->
              `<ns>/ctx_experiment_build/<exp_tag>/`. The HF URLs are written back onto the wandb run.
              The upload refuses a tokenizer that does not match the checkpoint's bindings; a legacy
              checkpoint without bindings needs --allow-unbound.
              Steps 2 and 3 resolve base_checkpoints/d<DEPTH> (ctx.tracking.resolve_checkpoint); with a
              replicate model tags (d<depth>_s<seed>) are exported/uploaded to their own repo ctx_<ARM>_<tag>_base.
  4. eval     eval_beliefs.py for this arm with the same --model-tag, writing to eval/beliefs_arm<ARM>.csv
              (default tag) or eval/seed_<INIT_SEED>/beliefs_arm<ARM>.csv (replicate tag d<DEPTH>_s<SEED>;
              eval/<MODEL_TAG>/ for a custom tag) -- the layout analyze_beliefs.py summarizes per replicate.
              The evaluation takes its run identity from the checkpoint it scored (meta_<step>.json
              wandb_run), logs summary belief metrics to that run and archives the CSV next to the checkpoint.

Secrets: WANDB_API_KEY and HF_TOKEN from contextualization/.env (see .env.example).
Anything after a bare `--` is passed through to scripts.base_train.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctx import tracking as T  # noqa: E402
from ctx.protocol import PROTOCOL_V2, ProtocolError, code_revision, load_summary, protocol_of  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA_LOADERS = ("bestfit", "prepacked")
ARMS = ("C", "R", "X", "E")
PREFLIGHT_NAME = "PREFLIGHT_OK.json"


def sh(cmd, env=None, cwd=REPO):
    print("+ " + " ".join(map(str, cmd)), flush=True)
    r = subprocess.run([str(c) for c in cmd], env=env, cwd=cwd)
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {cmd[0]}")


def model_tag_for(depth: int, init_seed: int | None = None, model_tag: str | None = None) -> str:
    """Checkpoint dir name under arm_<A>/base_checkpoints: d<depth> (historical default) or
    d<depth>_s<init_seed> when an init seed is given, so every replicate has its own destination."""
    if model_tag:
        return str(model_tag)
    return f"d{int(depth)}" if init_seed is None else f"d{int(depth)}_s{int(init_seed)}"


def loader_from_extra(extra) -> str | None:
    """A --data-loader passed after `--` (as `--data-loader=X` or `--data-loader X`)."""
    extra = [str(a) for a in extra]
    for i, a in enumerate(extra):
        if a.startswith("--data-loader="):
            return a.split("=", 1)[1]
        if a == "--data-loader" and i + 1 < len(extra):
            return extra[i + 1]
    return None


def build_protocol(exp_dir) -> str | None:
    """Protocol of the build (None without build_summary.json; 'unknown (...)' when unreadable)."""
    summ = load_summary(exp_dir)
    if not summ:
        return None
    try:
        return protocol_of(summ)
    except ProtocolError as e:
        return f"unknown ({e})"


LOSS_NORMALIZATIONS = ("microbatch_mean", "fixed_nominal")   # = nanochat.loss_normalization.LOSS_NORMALIZATIONS


def check_preflight(exp_dir, arm: str, loss_normalization: str | None = None) -> dict:
    """The preflight written by exposure_audit.py --mode prepacked must exist, name this arm's CURRENT
    prepack_dataset_id and certify the requested loss normalization (a preflight without the key certified the
    historical microbatch_mean); base_train re-checks the whole training configuration itself. SystemExit otherwise."""
    exp_dir = Path(exp_dir)
    p = exp_dir / "exposure" / PREFLIGHT_NAME
    hint = (f"run `python contextualization/exposure_audit.py --exp-dir {exp_dir} --arms <all prepacked arms> --mode prepacked "
            f"--B <device batch> --T <seq len> --grad-accum <steps> --num-iterations <N> --world-size <W> --data-seed <seed>` first")
    if not p.exists():
        raise SystemExit(f"--data-loader prepacked needs {p}; {hint}")
    pf = json.loads(p.read_text())
    ids = pf.get("prepack_dataset_id") or {}
    if arm not in ids:
        raise SystemExit(f"{p} does not cover arm {arm} (arms {sorted(ids)}); {hint}")
    man_path = exp_dir / f"arm_{arm}" / "prepacked" / "prepack_manifest.json"
    if not man_path.exists():
        raise SystemExit(f"no prepacked rows for arm {arm}: {man_path} missing (exposure_audit.py --mode prepack)")
    current = json.loads(man_path.read_text()).get("prepack_dataset_id")
    if ids[arm] != current:
        raise SystemExit(f"{p} is stale for arm {arm}: it names prepack_dataset_id {str(ids[arm])[:12]} but {man_path} has "
                         f"{str(current)[:12]}; {hint}")
    certified = pf.get("loss_normalization") or "microbatch_mean"
    requested = loss_normalization or "microbatch_mean"
    if certified != requested:
        raise SystemExit(f"{p} certifies loss_normalization {certified!r}{' (unrecorded: historical default)' if 'loss_normalization' not in pf else ''} "
                         f"but this launch requests {requested!r}; re-run exposure_audit.py --mode prepacked --loss-normalization {requested} "
                         "(the training objective is part of the certified configuration)")
    return pf


def init_identity_from_checkpoints(arm_dir, model_tag: str) -> dict | None:
    """init_param_sha256 / init_seed / step of the latest meta_<step>.json under arm_dir/base_checkpoints/<model_tag>."""
    d = Path(arm_dir) / "base_checkpoints" / model_tag
    metas = sorted(d.glob("meta_*.json"))
    if not metas:
        return None
    m = json.loads(metas[-1].read_text())
    return {"init_param_sha256": m.get("init_param_sha256"), "init_seed_effective": m.get("init_seed"),
            "checkpoint_step": m.get("step"), "meta_path": str(metas[-1])}


def compare_init_identity(exp_dir, arm: str, identity: dict | None, depth: int, model_tag: str | None = None) -> list[str]:
    """One line per other arm whose run_info (of the same model tag) carries an init_param_sha256 for the
    same depth and effective init seed: MATCH or DIFFER (reviewer check 4)."""
    lines = []
    if not identity or not identity.get("init_param_sha256"):
        return lines
    mine = identity["init_param_sha256"]
    for other in ARMS:
        if other == arm:
            continue
        info = T.read_run_info(exp_dir, other, model_tag)
        theirs = info.get("init_param_sha256")
        if not theirs or info.get("init_seed_effective") != identity.get("init_seed_effective") or info.get("depth") != depth:
            continue
        verdict = "MATCH" if theirs == mine else "DIFFER"
        lines.append(f"init params {verdict}: arm {arm} vs arm {other} (init_seed {identity.get('init_seed_effective')}, "
                     f"sha256 {mine[:12]} vs {theirs[:12]})")
    return lines


def build_train_command(depth: int, run_name: str, nproc: int, extra=(), *, init_seed: int | None = None,
                        data_seed: int | None = None, data_loader: str | None = None, model_tag: str | None = None,
                        loss_normalization: str | None = None, python: str | None = None) -> list[str]:
    """The exact scripts.base_train invocation (torchrun only when nproc > 1). The seed / loader / model-tag
    flags are forwarded with the names scripts/base_train.py defines (--init-seed, --data-seed,
    --data-loader {bestfit,prepacked}, --model-tag) and only when given, so base_train's defaults otherwise apply."""
    if data_loader is not None and data_loader not in DATA_LOADERS:
        raise ValueError(f"data_loader must be one of {DATA_LOADERS}, got {data_loader!r}")
    flags = [f"--depth={depth}", f"--run={run_name}"]
    if init_seed is not None:
        flags.append(f"--init-seed={int(init_seed)}")
    if data_seed is not None:
        flags.append(f"--data-seed={int(data_seed)}")
    if data_loader is not None:
        flags.append(f"--data-loader={data_loader}")
    if model_tag is not None:
        flags.append(f"--model-tag={model_tag}")
    if loss_normalization is not None:
        if loss_normalization not in LOSS_NORMALIZATIONS:
            raise ValueError(f"loss_normalization must be one of {LOSS_NORMALIZATIONS}, got {loss_normalization!r}")
        flags.append(f"--loss-normalization={loss_normalization}")
    flags += [str(a) for a in extra]
    py = python or sys.executable
    if nproc > 1:
        return [py, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}",
                "-m", "scripts.base_train", "--", *flags]
    return [py, "-m", "scripts.base_train", *flags]  # single GPU / CPU: no launcher needed


def training_record(cmd, env_extra: dict | None = None, *, init_seed=None, data_seed=None,
                    data_loader=None, model_tag=None, loss_normalization=None, cwd=REPO) -> dict:
    """What run_info.json stores about a training launch: the exact command, the environment
    variables we set for it, the seeds / loader / model tag as passed (None = base_train default), and
    the code revision (git commit + dirty flag) of the checkout that launched it."""
    cmd = [str(c) for c in cmd]
    return {
        "train_command": cmd,
        "train_command_str": shlex.join(cmd),
        "train_cwd": str(cwd),
        "train_env": dict(env_extra or {}),
        "init_seed": init_seed, "data_seed": data_seed, "data_loader": data_loader, "model_tag": model_tag,
        "loss_normalization": loss_normalization or "microbatch_mean (base_train default)",
        "code": code_revision(cwd),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=["C", "R", "X", "E"])
    ap.add_argument("--exp-dir", required=True, help="build dir holding arm_<ARM>/ and probe_sets/")
    ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--nproc", type=int, default=None, help="GPUs for torchrun (default: all visible)")
    ap.add_argument("--run-name", default=None, help="wandb run name (default ctx_<ARM>_<MODEL_TAG>)")
    ap.add_argument("--init-seed", type=int, default=None, help="forwarded to base_train --init-seed (model init); also selects the model tag d<DEPTH>_s<SEED>")
    ap.add_argument("--data-seed", type=int, default=None, help="forwarded to base_train --data-seed (data order)")
    ap.add_argument("--data-loader", choices=DATA_LOADERS, default=None,
                    help="forwarded to base_train --data-loader; REQUIRED for protocol-v2 builds (prepacked = corrected path with preflight, bestfit = historical loader)")
    ap.add_argument("--model-tag", default=None,
                    help="checkpoint dir under arm_<ARM>/base_checkpoints (default d<DEPTH>, or d<DEPTH>_s<INIT_SEED> with --init-seed); also used for eval")
    ap.add_argument("--loss-normalization", choices=list(LOSS_NORMALIZATIONS), default=None,
                    help="forwarded to base_train --loss-normalization; funded C/R/X/E runs select fixed_nominal EXPLICITLY (decision "
                         "2026-10-15); the preflight must certify the same policy; default: base_train's historical microbatch_mean")
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
    extra = [a for a in args.train_args if a != "--"]

    # ---- 0. loader choice, preflight, checkpoint destination -------------------------------
    protocol = build_protocol(exp_dir)
    data_loader = args.data_loader or loader_from_extra(extra)
    if data_loader is not None and data_loader not in DATA_LOADERS:
        raise SystemExit(f"--data-loader must be one of {DATA_LOADERS}, got {data_loader!r}")
    if not args.skip_train and protocol == PROTOCOL_V2 and data_loader is None:
        raise SystemExit(f"build {exp_dir} is protocol {protocol}: "
                         "choose --data-loader prepacked (corrected path, needs the preflight) or bestfit (historical loader) explicitly")
    model_tag = model_tag_for(args.depth, args.init_seed, args.model_tag)
    name = args.run_name or T.run_name(args.arm, model_tag)
    if not args.skip_train and data_loader == "prepacked":
        pf = check_preflight(exp_dir, args.arm, args.loss_normalization)
        print(f"preflight: {exp_dir / 'exposure' / PREFLIGHT_NAME} covers arm {args.arm} (B={pf.get('B')} T={pf.get('T')} "
              f"grad_accum={pf.get('grad_accum_steps')} num_iterations={pf.get('num_iterations')} world_size={pf.get('world_size')} "
              f"data_seed={pf.get('data_seed')} loss_normalization={pf.get('loss_normalization') or 'microbatch_mean (unrecorded)'}); "
              "base_train re-checks it against the actual run")
    replicate_tag = model_tag != f"d{args.depth}"  # replicate runs use base_checkpoints/<tag>, repo ctx_<A>_<tag>_base
    print(f"protocol: {protocol or 'legacy/unknown (no build_summary.json)'} | loader: {data_loader or 'base_train default (bestfit)'} "
          f"| loss normalization: {args.loss_normalization or 'base_train default (microbatch_mean, historical)'} "
          f"| model tag: {model_tag} | checkpoints: {arm_dir / 'base_checkpoints' / model_tag}")

    # ---- wandb run identity (fixed before training so eval can attach to the same run); one
    #      run_info file per model tag (recheck finding 6) ----
    info = T.read_run_info(exp_dir, args.arm, model_tag)
    if args.skip_train:
        run_id = info.get("wandb_run_id") or (T.new_run_id() if T.wandb_enabled() else None)
    else:
        run_id = T.new_run_id() if T.wandb_enabled() else None  # fresh training run = fresh wandb run
    eval_dir = T.eval_output_dir(exp_dir, model_tag)
    T.write_run_info(exp_dir, args.arm, model_tag, wandb_run_id=run_id, wandb_run_name=name,
                     wandb_project=T.wandb_project() if run_id else None, depth=args.depth, exp_tag=tag,
                     model_tag=model_tag, eval_dir=str(eval_dir),
                     train_finished=info.get("train_finished") if args.skip_train else False)
    print(f"run_info: {T.run_info_path(exp_dir, args.arm, model_tag)} | eval outputs: {eval_dir}")
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
                                  init_seed=args.init_seed, data_seed=args.data_seed, data_loader=args.data_loader,
                                  model_tag=model_tag, loss_normalization=args.loss_normalization)
        rec = training_record(cmd, env_extra, init_seed=args.init_seed, data_seed=args.data_seed,
                              data_loader=data_loader, model_tag=model_tag, loss_normalization=args.loss_normalization)
        T.write_run_info(exp_dir, args.arm, model_tag, **rec)  # recorded BEFORE launch so a crashed run is still described
        sh(cmd, env=env)
        identity = init_identity_from_checkpoints(arm_dir, model_tag)
        T.write_run_info(exp_dir, args.arm, model_tag, train_finished=True, **(identity or {}))
        if identity and identity.get("init_param_sha256"):
            print(f"init_param_sha256 {identity['init_param_sha256']} (init_seed {identity.get('init_seed_effective')}, "
                  f"step {identity.get('checkpoint_step')}) -> run_info.json")
            for line in compare_init_identity(exp_dir, args.arm, identity, args.depth, model_tag):
                print(line, flush=True)
        else:
            print("NOTE: no init_param_sha256 found in the checkpoint meta (legacy base_train?)")

    # ---- 2. local export manifest (hashes, tokenizer fingerprint, bindings) before any upload ----
    if True:
        try:
            out = T.export_manifest(exp_dir, args.arm, model_tag, with_optimizer=args.with_optimizer)
            print(f"checkpoint manifest -> {out}")
            T.write_run_info(exp_dir, args.arm, model_tag, export_manifest=str(out))
        except SystemExit as e:
            if not args.skip_train or not args.skip_upload:
                raise
            print(f"NOTE: no export manifest written ({e})")

        # ---- 3. upload checkpoint + build metadata to HF, link from wandb -----------------
        if not args.skip_upload and T.hf_enabled():
            url, sha, step = T.hf_upload_checkpoint(exp_dir, args.arm, model_tag, run_id=run_id,
                                                    with_optimizer=args.with_optimizer, extra_tags=(tag,),
                                                    allow_unbound=args.allow_unbound)
            print(f"checkpoint -> {url} (commit {sha[:8]}, step {step}, tag wandb-{run_id})")
            build_url, _ = T.hf_upload_build(exp_dir)
            print(f"build metadata -> {build_url}")
            T.write_run_info(exp_dir, args.arm, model_tag, hf_checkpoint=url, hf_checkpoint_commit=sha, hf_build=build_url,
                             hf_checkpoint_step=int(step), hf_optimizer_included=bool(args.with_optimizer))
            if run_id:
                T.wandb_link_hf(run_id, hf_checkpoint=f"{url}/tree/wandb-{run_id}", hf_build=build_url)
        elif not args.skip_upload:
            print("HF upload skipped (no HF_TOKEN / tracking off)")

    # ---- 4. eval (scoped to this model tag: eval/[seed_<S>/]beliefs_arm<A>.csv; the evaluator takes the run
    #      identity from the checkpoint meta and refuses to overwrite an existing CSV) -------------------
    if not args.skip_eval:
        sh([sys.executable, str(REPO / "contextualization" / "eval_beliefs.py"),
            "--exp-dir", exp_dir, "--arm", args.arm, "--model-tag", model_tag,
            "--batch-size", args.eval_batch_size, "--out-csv", eval_dir / f"beliefs_arm{args.arm}.csv"], env=dict(os.environ))

    print(f"\ndone: arm {args.arm} | run_info -> {T.run_info_path(exp_dir, args.arm, model_tag)} | eval -> {eval_dir}")


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
