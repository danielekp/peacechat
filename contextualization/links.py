#!/usr/bin/env python
"""links.py -- show which HF checkpoint belongs to which wandb run (and vice versa).

    uv run python contextualization/links.py                 # every ctx_* run in the wandb project
    uv run python contextualization/links.py --run heouu2no  # one run
    uv run python contextualization/links.py --exp-dir /workspace/ctx_experiment_v3   # from run_info.json

How the two sides are tied together (all written automatically by run_arm.py / eval_beliefs.py):
  wandb run  ->  config/summary/notes carry `hf_checkpoint` (URL of the git tag wandb-<run_id>),
                 `hf_build` and `hf_eval`; the run is named ctx_<ARM>_d<DEPTH>.
  HF repo    ->  ctx_<ARM>_d<DEPTH>_base has one commit per training run, tagged wandb-<run_id>;
                 meta_<step>.json inside it has a `wandb_run` {id, name, project, url} block and
                 the model card links the run.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctx import tracking as T  # noqa: E402


def from_wandb(run_filter):
    import wandb
    api = wandb.Api()
    ent = T.wandb_entity() or api.default_entity
    runs = [api.run(f"{ent}/{T.wandb_project()}/{run_filter}")] if run_filter else \
        [r for r in api.runs(f"{ent}/{T.wandb_project()}") if r.name.startswith("ctx_")]
    rows = []
    for r in runs:
        rows.append({
            "run": f"{r.name} ({r.id}, {r.state})", "url": r.url, "tags": ",".join(r.tags),
            "hf_checkpoint": r.config.get("hf_checkpoint") or r.summary.get("hf_checkpoint") or "-",
            "hf_eval": r.config.get("hf_eval") or "-",
            "hf_build": r.config.get("hf_build") or "-",
        })
    return rows


def from_exp_dir(exp_dir):
    rows = []
    for arm in "CRXE":
        info = T.read_run_info(exp_dir, arm)
        if not info:
            continue
        rid = info.get("wandb_run_id")
        rows.append({
            "run": f"{info.get('wandb_run_name')} ({rid})", "url": T.wandb_run_url(rid) if rid else "-",
            "tags": f"arm_{arm},{info.get('exp_tag')}",
            "hf_checkpoint": (info.get("hf_checkpoint") or "-") + (f"/tree/wandb-{rid}" if rid and info.get("hf_checkpoint") else ""),
            "hf_eval": (info.get("hf_checkpoint") or "-") + "/tree/main/eval" if info.get("hf_checkpoint") else "-",
            "hf_build": info.get("hf_build") or "-",
        })
    return rows


def from_hf(repo_id):
    """Reverse direction: list the wandb-tagged commits of an HF model repo."""
    api = T.hf_api()
    refs = api.list_repo_refs(repo_id)
    return [{"tag": t.name, "commit": t.target_commit[:8],
             "wandb": T.wandb_run_url(t.name.removeprefix("wandb-")) if t.name.startswith("wandb-") else "-"}
            for t in refs.tags]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="a wandb run id")
    ap.add_argument("--exp-dir", default=None, help="read arm_*/run_info.json instead of asking wandb")
    ap.add_argument("--hf", default=None, help="an HF model repo (e.g. Daniele/ctx_C_d24_base): list its wandb-tagged commits")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    T.load_env()

    if args.hf:
        rows = from_hf(args.hf)
    elif args.exp_dir:
        rows = from_exp_dir(args.exp_dir)
    else:
        T.require(wandb=True)
        rows = from_wandb(args.run)

    if args.json:
        print(json.dumps(rows, indent=2))
        return
    if not rows:
        print("nothing found")
        return
    for row in rows:
        print("\n".join(f"{k:>14}: {v}" for k, v in row.items()))
        print()


if __name__ == "__main__":
    main()
