"""Back up a trained arm's checkpoint + the shared tokenizer + build metadata to wandb.

Idempotent: wandb dedupes files by checksum, so rerunning never re-uploads bytes that
already landed, and unchanged artifacts don't get a new version.

Usage (on the training box):
    uv run contextualization/load_artifact_wandb.py --arm C
    uv run contextualization/load_artifact_wandb.py --arm R --exp-dir /workspace/ctx_experiment_v2
"""

import argparse
import glob
import gzip
import os
import shutil
import sys

import wandb

p = argparse.ArgumentParser()
p.add_argument("--arm", required=True, choices=["C", "R", "X"])
p.add_argument("--exp-dir", default="/workspace/ctx_experiment")
p.add_argument("--tok-dir", default="/workspace/ctx_tok/tokenizer")
p.add_argument("--project", default="nanochat")
p.add_argument("--depth", default="d24")
p.add_argument("--with-optimizer", action="store_true", help="also upload optimizer state (several GB)")
args = p.parse_args()

ckpt_dir = os.path.join(args.exp_dir, f"arm_{args.arm}", "base_checkpoints", args.depth)
models = sorted(glob.glob(os.path.join(ckpt_dir, "model_*.pt")))
if not models:
    sys.exit(f"no model_*.pt under {ckpt_dir}")
model_pt = models[-1]
step = os.path.basename(model_pt).removesuffix(".pt").split("_")[-1]
meta_json = os.path.join(ckpt_dir, f"meta_{step}.json")

manifest = os.path.join(args.exp_dir, "manifest.csv")
manifest_gz = manifest + ".gz"
if os.path.exists(manifest) and not os.path.exists(manifest_gz):
    print(f"gzipping {manifest} ...")
    with open(manifest, "rb") as fin, gzip.open(manifest_gz, "wb", compresslevel=6) as fout:
        shutil.copyfileobj(fin, fout)

run = wandb.init(project=args.project, job_type="upload", name=f"ctx_{args.arm}_{args.depth}_backup")

model = wandb.Artifact(
    f"ctx_{args.arm}_{args.depth}_base", type="model",
    description=f"Arm {args.arm} base checkpoint, {args.depth}, step {step}, from {args.exp_dir}",
)
model.add_file(model_pt)
if os.path.exists(meta_json):
    model.add_file(meta_json)
model.add_dir(args.tok_dir, name="tokenizer")
if args.with_optimizer:
    for opt in sorted(glob.glob(os.path.join(ckpt_dir, f"optim_{step}_rank*.pt"))):
        model.add_file(opt)
run.log_artifact(model)

meta = wandb.Artifact(
    "ctx_experiment_build", type="dataset",
    description=f"probe sets + manifest + build summary from {args.exp_dir}",
)
meta.add_dir(os.path.join(args.exp_dir, "probe_sets"), name="probe_sets")
for f in [os.path.join(args.exp_dir, "build_summary.json"), manifest_gz]:
    if os.path.exists(f):
        meta.add_file(f)
    else:
        print(f"WARNING: skipping missing {f}")
run.log_artifact(meta)

run.finish()
print("done — verify at https://wandb.ai/danielaush/" + args.project + "/artifacts")
