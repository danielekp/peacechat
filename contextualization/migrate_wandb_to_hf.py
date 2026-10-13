"""Move the three arms' checkpoints + build metadata from wandb artifacts to the HF Hub.

HF repos get the *same names* as the wandb artifact collections, so the wandb training
run (whose metrics stay on wandb) and the HF checkpoint are found by the same string:

    wandb artifact  danielaush/nanochat/ctx_C_d24_base   ->  HF model   <user>/ctx_C_d24_base
    wandb artifact  danielaush/nanochat/ctx_R_d24_base   ->  HF model   <user>/ctx_R_d24_base
    wandb artifact  danielaush/nanochat/ctx_X_d24_base   ->  HF model   <user>/ctx_X_d24_base
    wandb artifact  danielaush/nanochat/ctx_experiment_build -> HF dataset <user>/ctx_experiment_build

Arm X was never uploaded to wandb (the second "ctx_R_d24_backup" run re-logged the R
checkpoint), so only C and R can be moved; missing collections are reported and skipped.
ctx_experiment_build has two wandb versions (v0 = /workspace/ctx_experiment used by arm C,
v1 = /workspace/ctx_experiment_v2 used by R and X); both are kept, as subfolders v0/ and v1/.

Per artifact: download (every version for the dataset, latest for models) -> upload to HF (plus a README card that links the
wandb run) -> verify every file's size and sha256 on the Hub -> delete *all* wandb versions
of that collection. Nothing is deleted from wandb unless verification passed. The wandb
training runs also get the HF URL written into their notes.

Usage:
    uv run --with wandb --with huggingface_hub migrate_wandb_to_hf.py            # dry run
    uv run --with wandb --with huggingface_hub migrate_wandb_to_hf.py --execute
    ... --execute --public            # public repos don't count against HF storage
    ... --execute --keep-wandb        # mirror only, don't delete anything on wandb
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import wandb
from huggingface_hub import HfApi

WANDB_PROJECT = "danielaush/nanochat"
WANDB_URL = f"https://wandb.ai/{WANDB_PROJECT}"
TRAIN_RUNS = {"C": "cfdt7xoj", "R": "vzso4vxo", "X": "nvop8htx"}
ARM_DESC = {
    "C": "Control: base corpus only, no injected claims.",
    "R": "Raw: injected claims as bare repeated assertions.",
    "X": "Contextualized: the same claims at the same frequencies, attributed to sources.",
}
# (wandb collection name, wandb type, HF repo_type, arm or None)
COLLECTIONS = [
    ("ctx_C_d24_base", "model", "model", "C"),
    ("ctx_R_d24_base", "model", "model", "R"),
    ("ctx_X_d24_base", "model", "model", "X"),
    ("ctx_experiment_build", "dataset", "dataset", None),
]

p = argparse.ArgumentParser()
p.add_argument("--execute", action="store_true", help="actually upload + delete (default: dry run)")
p.add_argument("--public", action="store_true", help="create public HF repos (default private)")
p.add_argument("--keep-wandb", action="store_true", help="don't delete wandb artifacts / backup runs")
p.add_argument("--download-dir", default=os.path.expanduser("~/.cache/ctx_wandb_to_hf"))
p.add_argument("--only", nargs="*", default=None, help="subset of collection names")
p.add_argument("--tag", default="run1-2026-10", help="git tag to pin this snapshot on each HF repo")
args = p.parse_args()

api = wandb.Api()
hf = HfApi()
hf_user = hf.whoami()["name"]
dl_root = Path(args.download_dir)
dl_root.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_card(name: str, arm: str, art, step: str, meta: dict | None) -> str:
    run_id = TRAIN_RUNS[arm]
    lines = [
        "---",
        "license: mit",
        "tags: [nanochat, pretraining, contextualization, ai-safety]",
        "---",
        f"# {name}",
        "",
        f"Arm **{arm}** of the three-arm contextualization-in-pretraining study on nanochat (d24). "
        f"{ARM_DESC[arm]}",
        "",
        f"- Training run (metrics, curves): [{WANDB_PROJECT}/{run_id}]({WANDB_URL}/runs/{run_id})",
        f"- Checkpoint: `model_{step}.pt` (nanochat `base_train` state dict), `meta_{step}.json`",
        "- Tokenizer: `tokenizer/` (shared across all three arms)",
        f"- Build metadata (probe sets, manifest, build summary): `{hf_user}/ctx_experiment_build` (dataset)",
        f"- Originally logged as wandb artifact `{art.name}` ({art.created_at}); moved to the Hub unchanged.",
        f"- This snapshot is pinned at git tag `{args.tag}`. It is the first training run, which had code bugs; "
        "a rerun lands as later commits on `main`, so use the tag to get this exact checkpoint.",
        "",
        "Load with nanochat: copy `tokenizer/` to the tokenizer dir and `model_*.pt` + `meta_*.json` "
        "to `base_checkpoints/d24/`, then run `eval_beliefs.py` as in the project README.",
    ]
    if meta:
        lines += ["", "## meta", "", "```json", json.dumps(meta, indent=2)[:4000], "```"]
    return "\n".join(lines) + "\n"


def dataset_card(name: str, versions) -> str:
    runs = ", ".join(f"[{a}: {r}]({WANDB_URL}/runs/{r})" for a, r in TRAIN_RUNS.items())
    vlines = [f"- `{v.version}/`: {v.description} (wandb `{v.name}`, {v.created_at})" for v in versions]
    return "\n".join([
        "---",
        "license: mit",
        "tags: [nanochat, pretraining, contextualization, ai-safety]",
        "---",
        f"# {name}",
        "",
        "Build metadata for the three-arm contextualization-in-pretraining study on nanochat: "
        "`probe_sets/` (belief probes), `manifest.csv.gz` (per-document injection manifest), "
        "`build_summary.json`. One subfolder per wandb artifact version:",
        "",
        *vlines,
        "",
        "Arm C was trained from the v0 build (`/workspace/ctx_experiment`); arms R and X from the "
        "v1 build (`/workspace/ctx_experiment_v2`).",
        "",
        f"- Training runs on wandb: {runs}",
        "- Checkpoints on the Hub: " + ", ".join(f"`{hf_user}/ctx_{a}_d24_base`" for a in TRAIN_RUNS)
        + " (arm X was never backed up from the training pod)",
        f"- First-run snapshot pinned at git tag `{args.tag}`; the rerun lands as later commits on `main`.",
    ]) + "\n"


report = []
for name, wtype, repo_type, arm in COLLECTIONS:
    if args.only and name not in args.only:
        continue
    repo_id = f"{hf_user}/{name}"
    try:
        coll = api.artifact_collection(wtype, f"{WANDB_PROJECT}/{name}")
        versions = sorted(coll.artifacts(), key=lambda a: a.version)
    except Exception as e:  # collection doesn't exist
        versions = []
    if not versions:
        print(f"\n== {name}: NOT ON WANDB, skipping (arm {arm})")
        report.append((name, "-", "NOT ON WANDB"))
        continue
    # models: only the latest version is a real checkpoint; dataset: keep every version
    to_move = versions[-1:] if arm else versions
    print(f"\n== {name}  ({len(versions)} version(s) on wandb, moving {len(to_move)}) -> {repo_type} {repo_id}")
    for v in to_move:
        print(f"   {v.name}  {v.size/1e9:.2f} GB  {v.description!r}")
        for f in v.files():
            print(f"      {f.name:50s} {f.size/1e6:9.1f} MB")
    if not args.execute:
        continue

    # 1. download into a staging folder (dataset versions go to v0/, v1/, ...)
    stage = dl_root / name
    stage.mkdir(parents=True, exist_ok=True)
    expected = []  # (relative path on Hub, local path, wandb file)
    for v in to_move:
        sub = "" if arm else f"{v.version}/"
        local = Path(v.download(root=str(stage / v.version if not arm else stage)))
        for f in v.files():
            expected.append((sub + f.name, local / f.name, f))
    art = to_move[-1]
    print(f"   downloaded to {stage}")

    # 2. card
    if arm:
        model_pt = sorted(stage.glob("model_*.pt"))[-1]
        step = model_pt.stem.split("_")[-1]
        meta_path = stage / f"meta_{step}.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
        (stage / "README.md").write_text(model_card(name, arm, art, step, meta))
    else:
        (stage / "README.md").write_text(dataset_card(name, to_move))

    # 3. upload (content-addressed on the Hub, so reruns skip unchanged bytes)
    hf.create_repo(repo_id, repo_type=repo_type, private=not args.public, exist_ok=True)
    hf.upload_folder(
        repo_id=repo_id, repo_type=repo_type, folder_path=str(stage),
        commit_message=f"Mirror wandb artifact {art.name}" + ("" if arm else " (all versions)"),
    )
    url = f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}"
    print(f"   uploaded -> {url}")
    hf.create_tag(repo_id, repo_type=repo_type, tag=args.tag,
                  tag_message="first run (pre-bugfix), mirrored from wandb", exist_ok=True)
    print(f"   tagged {args.tag}")

    # 4. verify every wandb file landed with the right size + sha256
    info = hf.repo_info(repo_id, repo_type=repo_type, files_metadata=True)
    remote = {s.rfilename: s for s in info.siblings}
    bad = []
    for rel, lp, f in expected:
        r = remote.get(rel)
        if r is None:
            bad.append(f"{rel}: missing on Hub"); continue
        if r.size != lp.stat().st_size or r.size != f.size:
            bad.append(f"{rel}: size hub={r.size} local={lp.stat().st_size} wandb={f.size}"); continue
        if r.lfs is not None and r.lfs.sha256 != sha256(lp):
            bad.append(f"{rel}: sha256 mismatch")
    if bad:
        print("   VERIFY FAILED, leaving wandb untouched:\n     " + "\n     ".join(bad))
        report.append((name, url, "VERIFY FAILED"))
        continue
    print(f"   verified {len(expected)} files (size + sha256)")

    # 5. cross-link from the wandb training run
    if arm:
        run = api.run(f"{WANDB_PROJECT}/{TRAIN_RUNS[arm]}")
        tag = f"checkpoint moved to HF: {url} (tag {args.tag})"
        if tag not in (run.notes or ""):
            run.notes = ((run.notes or "").rstrip() + "\n" + tag).strip()
            run.update()

    # 6. delete all wandb versions of this collection
    if not args.keep_wandb:
        for v in versions:
            v.delete(delete_aliases=True)
        print(f"   deleted {len(versions)} wandb version(s) of {name}")
    report.append((name, url, "OK"))

# 7. the ctx_*_backup upload runs only existed to hold the artifacts
if args.execute and not args.keep_wandb:
    for r in api.runs(WANDB_PROJECT, filters={"jobType": "upload"}):
        if r.name.startswith("ctx_") and r.name.endswith("_backup"):
            r.delete(delete_artifacts=False)
            print(f"deleted wandb upload run {r.name} ({r.id})")

if args.execute:
    print("\nSummary:")
    for name, url, status in report:
        print(f"  {status:14s} {name:24s} {url}")
    if any(s != "OK" for _, _, s in report):
        sys.exit(1)
else:
    print("\nDry run. Add --execute to upload, verify, and delete from wandb.")
