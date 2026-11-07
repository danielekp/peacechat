#!/usr/bin/env python
"""Manually (re)upload an arm's checkpoint, the build metadata, or eval outputs to the HF Hub,
or just write the local export manifests that describe what such an upload would contain.

run_arm.py does all of this automatically; this is the escape hatch for a box where training
already finished, or to re-push after a fix. Same names as everywhere else:
    model   <ns>/ctx_<ARM>_d<DEPTH>_base     (commit tagged wandb-<run_id> when known; the tag is
                                              an immutable identity and is never moved)
    eval    branch eval/wandb-<run_id> of that repo (starts at the checkpoint commit), tagged
            wandb-<run_id>-eval; the local record goes to eval/[seed_<S>/]beliefs_arm<ARM>.upload.json
            (replicate tags d<DEPTH>_s<S> keep their outputs under eval/seed_<S>/, custom tags under eval/<tag>/)
    dataset <ns>/ctx_experiment_build/<exp_tag>/

Usage (from the repo root, HF_TOKEN in contextualization/.env):
    uv run python contextualization/load_artifact_hf.py --exp-dir /workspace/ctx_experiment_v3 --arm C
    uv run python contextualization/load_artifact_hf.py --exp-dir ... --arm R --depth 24 --wandb-run abc123
    uv run python contextualization/load_artifact_hf.py --exp-dir ... --build          # metadata only
    uv run python contextualization/load_artifact_hf.py --exp-dir ... --arm X --eval   # eval CSV only
        # verifies that the archived checkpoint of the run has the model sha256 recorded in
        # eval/beliefs_armX.meta.json (checkpoint.model_sha256); --allow-unverified when it has none
    uv run python contextualization/load_artifact_hf.py --exp-dir ... --arm X --step 8352 --manifest-only
        # writes arm_X/export/checkpoint_manifest.json (and export/build_manifest.json with --build /
        # no --arm) and exits: no network, no HF_TOKEN needed
    uv run python contextualization/load_artifact_hf.py --exp-dir ... --arm C --allow-unbound
        # legacy checkpoint whose meta has no `bindings` block (July-2026 run)
    uv run python contextualization/load_artifact_hf.py --exp-dir ... --arm C --wandb-run abc123 --force-retag
        # the run already has an archived checkpoint (tag wandb-abc123): move the tag to this upload

A checkpoint upload refuses to pair the checkpoint with a tokenizer whose sha256 differs from
meta['bindings']['tokenizer_sha256'], and refuses to re-archive a run id that already has a
checkpoint tag unless --force-retag is given; see docs/ARTIFACTS.md.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctx import tracking as T  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--exp-dir", required=True)
p.add_argument("--arm", choices=["C", "R", "X", "E"])
p.add_argument("--depth", type=int, default=None, help="default: from run_info.json, else 24")
p.add_argument("--model-tag", default=None, help="checkpoint dir / repo tag such as d24_s7 (replicates); overrides --depth")
p.add_argument("--step", type=int, default=None, help="checkpoint step (default: the latest model_*.pt)")
p.add_argument("--wandb-run", default=None, help="wandb run id to tag the commit with (default: run_info.json)")
p.add_argument("--build", action="store_true", help="upload probe sets / manifest / summary (only)")
p.add_argument("--eval", action="store_true",
               help="upload eval/beliefs_arm<ARM>*.{csv,json} (only) to branch eval/wandb-<run_id>")
p.add_argument("--with-optimizer", action="store_true", help="also upload optimizer shards (several GB)")
p.add_argument("--allow-unbound", action="store_true",
               help="upload a legacy checkpoint whose meta has no `bindings` block (prints a WARNING)")
p.add_argument("--force-retag", action="store_true",
               help="checkpoint upload: move an existing wandb-<run_id> tag (and the <exp_tag> label) to this "
                    "upload instead of refusing; the old commit stays in the repo history")
p.add_argument("--allow-unverified", action="store_true",
               help="eval upload: archive outputs whose .meta.json carries no checkpoint.model_sha256 "
                    "(the upload record is marked verified=false)")
p.add_argument("--eval-tag", default=None,
               help="eval upload: tag for this evaluation commit (default wandb-<run_id>-eval; pass another "
                    "wandb-<run_id>-eval-<label> to archive a re-scoring of the same run)")
p.add_argument("--manifest-only", action="store_true",
               help="write export manifests (arm_<ARM>/export/checkpoint_manifest.json and/or "
                    "export/build_manifest.json) and exit 0; no network, no HF_TOKEN needed")
p.add_argument("--public", action="store_true", help="create the repo public (default: HF_PRIVATE)")
args = p.parse_args()

T.load_env()

if args.manifest_only:
    # local only: never touches the Hub, works without HF_TOKEN
    info = T.read_run_info(args.exp_dir, args.arm, args.model_tag) if args.arm else {}
    if args.build or not args.arm:
        out = T.export_build_manifest(args.exp_dir)
        print(f"build manifest -> {out}")
    if args.arm:
        depth = args.model_tag or info.get("model_tag") or args.depth or info.get("depth") or 24
        out = T.export_manifest(args.exp_dir, args.arm, depth, with_optimizer=args.with_optimizer, step=args.step)
        print(f"checkpoint manifest -> {out}")
    sys.exit(0)

if args.public:
    os.environ["HF_PRIVATE"] = "0"
T.require(hf=True)

if args.build or not args.arm:
    url, sha = T.hf_upload_build(args.exp_dir)
    print(f"build metadata -> {url} (commit {sha[:8]})")
    if not args.arm:
        sys.exit(0)

info = T.read_run_info(args.exp_dir, args.arm, args.model_tag)
depth = args.model_tag or info.get("model_tag") or args.depth or info.get("depth") or 24  # a model tag names dir + repo
run_id = args.wandb_run or info.get("wandb_run_id")

if args.eval:
    eval_dir = T.eval_output_dir(args.exp_dir, depth)   # eval/ or eval/seed_<S>/ (recheck finding 6)
    base = f"beliefs_arm{args.arm}"
    files = {f"eval/{f.name}": f for f in sorted(eval_dir.glob(f"{base}*"))
             if f.is_file() and not f.name.endswith(".upload.json")}
    if not files:
        sys.exit(f"no eval outputs for arm {args.arm} under {eval_dir}")
    if not run_id:
        sys.exit("eval uploads need the training run id (--wandb-run or arm_<ARM>/run_info.json): the outputs are "
                 "archived on branch eval/wandb-<run_id> next to the checkpoint they scored")
    meta_path = eval_dir / f"{base}.meta.json"
    ckpt = (json.loads(meta_path.read_text()).get("checkpoint") or {}) if meta_path.exists() else {}
    expected = ckpt.get("model_sha256")
    if not meta_path.exists():
        print(f"NOTE: {meta_path} missing; the evaluation carries no checkpoint sha256", flush=True)
    rec = T.hf_upload_eval(T.model_repo(args.arm, depth), files, run_id, expected, step=ckpt.get("step"),
                           allow_unverified=args.allow_unverified, tag=args.eval_tag,
                           commit_message=f"eval arm {args.arm} (wandb {run_id})")
    rec.update({"arm": args.arm, "exp_dir": str(Path(args.exp_dir).resolve()), "meta_json": str(meta_path),
                "expected_model_sha256": expected})
    record = eval_dir / f"{base}.upload.json"
    record.write_text(json.dumps(rec, indent=2, default=str) + "\n")
    print(f"eval -> {rec['url']} (branch {rec['branch']}, commit {rec['commit_sha'][:8]}, tag {rec['tag']}, "
          f"checkpoint {rec['checkpoint_commit'][:8]}, verified={rec['verified']})")
    print(f"upload record -> {record}")
elif not args.build:
    # local manifest first (no network), then the upload, which re-checks the tokenizer binding
    out = T.export_manifest(args.exp_dir, args.arm, depth, with_optimizer=args.with_optimizer, step=args.step)
    print(f"checkpoint manifest -> {out}")
    url, sha, step = T.hf_upload_checkpoint(args.exp_dir, args.arm, depth, run_id=run_id, step=args.step,
                                            with_optimizer=args.with_optimizer, extra_tags=(T.exp_tag(args.exp_dir),),
                                            allow_unbound=args.allow_unbound, force_retag=args.force_retag)
    print(f"checkpoint step {step} -> {url} (commit {sha[:8]}" + (f", tag wandb-{run_id})" if run_id else ")"))
    T.write_run_info(args.exp_dir, args.arm, depth, hf_checkpoint=url, hf_checkpoint_commit=sha, depth=T.depth_of_tag(T.model_tag_of(depth)),
                     model_tag=T.model_tag_of(depth), hf_checkpoint_step=int(step), hf_optimizer_included=bool(args.with_optimizer))
    if run_id and T.wandb_enabled():
        T.wandb_link_hf(run_id, hf_checkpoint=f"{url}/tree/wandb-{run_id}")
        print(f"linked from wandb run {run_id}")
