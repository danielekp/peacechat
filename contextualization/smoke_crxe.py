#!/usr/bin/env python
"""smoke_crxe.py -- deterministic OFFLINE C/R/X/E smoke workflow (no GPU, no network, no uploads).

    generation -> validation -> tokenizer -> prepack + exposure audit -> [tiny training]
    -> evaluation plumbing (mock scores, real BPE tokenizer) [-> real tiny-model inference]
    -> analysis

Run from the repo root with the CPU test environment:
    $HOME/.cache/ctx_cpu_venv/bin/python contextualization/smoke_crxe.py --work /tmp/ctx_smoke [--train]

What each stage proves is written to <work>/smoke_summary.json. Mock-score stages are labelled
"plumbing"; real inference only happens with --train (a depth-2, 5-step CPU model) and is
labelled "real_inference_tiny_model". Nothing here is a scientific result.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CTX = Path(__file__).resolve().parent
REPO = CTX.parent
PY = sys.executable


def sh(cmd, env=None, cwd=REPO, log=None):
    print("+ " + " ".join(map(str, cmd)), flush=True)
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd], env=env or dict(os.environ), cwd=cwd, capture_output=True, text=True)
    if log is not None:
        Path(log).write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    if r.returncode != 0:
        print(r.stdout[-4000:], r.stderr[-4000:])
        raise SystemExit(f"stage failed ({r.returncode}): {cmd[:3]}")
    return r.stdout, round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="scratch directory (created; refused if it exists unless --overwrite)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="data seed of the build")
    ap.add_argument("--init-seed", type=int, default=7, help="shared initialization seed for all arms (--train)")
    ap.add_argument("--data-seed", type=int, default=3, help="shared prepacked row-order seed for all arms")
    ap.add_argument("--T", type=int, default=256, help="training context for prepack / tiny training")
    ap.add_argument("--train", action="store_true", help="also train a depth-2 model for 5 steps per arm on CPU and run real inference")
    args = ap.parse_args()
    work = Path(args.work)
    if work.exists():
        if not args.overwrite:
            raise SystemExit(f"{work} exists; pass --overwrite")
        shutil.rmtree(work)
    work.mkdir(parents=True)
    logs = work / "logs"
    logs.mkdir()
    env = dict(os.environ, CTX_TRACKING="0", PYTHONUNBUFFERED="1")
    summary = {"work": str(work), "python": PY, "stages": [], "seeds": {"data_seed_build": args.seed,
               "init_seed": args.init_seed, "data_seed_loader": args.data_seed}}
    build = work / "build"
    arms = ["C", "R", "X", "E"]

    # 1. generation (protocol v2, E on, collision audit)
    _, dt = sh([PY, CTX / "build_dataset.py", "--protocol", "v2", "--base-source", "synthetic", "--num-groups", "80",
                "--freq-grid", "1", "4", "16", "--base-chars", "2000000", "--chars-per-shard", "300000",
                "--embedding-control", "--collision-audit", "--carrier-max-chars", "600", "--counterfact-limit", "0",
                "--out", build, "--seed", args.seed, "--no-tracking"], env=env, cwd=CTX, log=logs / "1_build.log")
    bs = json.loads((build / "build_summary.json").read_text())
    summary["stages"].append({"stage": "generation", "seconds": dt, "protocol": bs["protocol"], "counts": bs["counts"],
                              "xe_within_limits": bs.get("xe_diagnostics", {}).get("within_limits"),
                              "novelty": bs.get("novelty"), "kind": "artifact"})

    # 2. validation (independent checks)
    out, dt = sh([PY, CTX / "validate.py", "--out", build], env=env, cwd=CTX, log=logs / "2_validate.log")
    rep = json.loads((build / "validation_report.json").read_text())
    summary["stages"].append({"stage": "validation", "seconds": dt, "passed": "ALL CHECKS PASSED" in out,
                              "n_checks": len(rep.get("checks", [])), "kind": "check"})

    # 3. shared tokenizer: a REAL BPE tokenizer trained by nanochat on arm C, copied into every arm
    _, dt = sh([PY, "-m", "scripts.tok_train", "--max-chars", "1500000", "--vocab-size", "1024"],
               env=dict(env, NANOCHAT_BASE_DIR=str(build / "arm_C")), log=logs / "3_tok_train.log")
    sys.path.insert(0, str(CTX))
    from ctx.nanochat_io import copy_tokenizer
    from ctx.protocol import tokenizer_fingerprint
    for a in arms[1:]:
        copy_tokenizer(str(build / "arm_C"), str(build / f"arm_{a}"))
    fps = {a: tokenizer_fingerprint(str(build / f"arm_{a}"))["tokenizer_sha256"] for a in arms}
    assert len(set(fps.values())) == 1, fps
    summary["stages"].append({"stage": "tokenizer", "seconds": dt, "tokenizer_sha256": fps["C"], "kind": "artifact"})

    # 4. prepack (matched schedule across arms) + exposure audit
    _, dt = sh([PY, CTX / "exposure_audit.py", "--exp-dir", build, "--arms", *arms, "--mode", "prepack", "--T", args.T,
                "--data-seed", args.data_seed, "--B", "2", "--grad-accum", "1", "--num-iterations", "5"],
               env=env, cwd=CTX, log=logs / "4_prepack.log")
    _, dt2 = sh([PY, CTX / "exposure_audit.py", "--exp-dir", build, "--arms", *arms, "--mode", "prepacked", "--T", args.T,
                 "--data-seed", args.data_seed, "--B", "2", "--grad-accum", "1", "--num-iterations", "5"],
                env=env, cwd=CTX, log=logs / "4_exposure.log")
    cross = json.loads((build / "exposure" / "cross_arm.json").read_text())
    pm = json.loads((build / "arm_C" / "prepacked" / "prepack_manifest.json").read_text())
    summary["stages"].append({"stage": "prepack_exposure_audit", "seconds": dt + dt2, "cross_arm": {k: cross[k] for k in cross if k != "per_occurrence"},
                              "scored_tokens_total_C": pm.get("scored_tokens_total"), "kind": "check"})

    # 5. optional tiny training per arm (same init seed, same data seed) with the prepacked loader
    if args.train:
        for a in arms:
            _, dt = sh([PY, "-m", "scripts.base_train", "--depth=2", f"--max-seq-len={args.T}", "--device-type=cpu",
                        "--device-batch-size=2", f"--total-batch-size={2 * args.T}", "--num-iterations=5", "--eval-every=-1",
                        "--core-metric-every=-1", "--sample-every=-1", "--data-loader=prepacked",
                        f"--init-seed={args.init_seed}", f"--data-seed={args.data_seed}", "--run=dummy"],
                       env=dict(env, NANOCHAT_BASE_DIR=str(build / f"arm_{a}")), log=logs / f"5_train_{a}.log")
            summary["stages"].append({"stage": f"train_{a}", "seconds": dt, "kind": "real_training_tiny_model"})
        # same initialization across arms: compare the step-0 parameters is not saved; compare meta seeds instead
        metas = {a: json.loads(next((build / f"arm_{a}" / "base_checkpoints" / "d2").glob("meta_*.json")).read_text()) for a in arms}
        assert len({m["init_seed"] for m in metas.values()}) == 1 and len({m["data_seed"] for m in metas.values()}) == 1
        summary["seeds"]["recorded_in_meta"] = {a: {"init_seed": m["init_seed"], "data_seed": m["data_seed"],
                                                    "data_loader": m["data_loader"], "tokenizer_sha256": (m.get("bindings") or {}).get("tokenizer_sha256")}
                                                for a, m in metas.items()}

    # 6. evaluation plumbing: mock scores with the REAL BPE tokenizer (every arm), then real inference if trained
    ev = build / "eval"
    for a in arms:
        _, dt = sh([PY, CTX / "eval_beliefs.py", "--exp-dir", build, "--arm", a, "--dry-run", "--tokenizer-dir",
                    build / f"arm_{a}", "--no-tracking", "--out-csv", ev / f"beliefs_arm{a}.csv"], env=env, log=logs / f"6_eval_dry_{a}.log")
        meta = json.loads((ev / f"beliefs_arm{a}.meta.json").read_text())
        summary["stages"].append({"stage": f"eval_plumbing_{a}", "seconds": dt, "mode": meta["mode"], "n_items": meta["n_items"],
                                  "coverage_complete": meta["coverage"]["complete"], "kind": "plumbing_mock_scores"})
    if args.train:
        evr = build / "eval_real"
        for a in arms:
            _, dt = sh([PY, CTX / "eval_beliefs.py", "--exp-dir", build, "--arm", a, "--model-tag", "d2", "--limit", "20",
                        "--batch-size", "16", "--no-tracking", "--no-hash-checkpoint", "--out-csv", evr / f"beliefs_arm{a}.csv"],
                       env=env, log=logs / f"6_eval_real_{a}.log")
            meta = json.loads((evr / f"beliefs_arm{a}.meta.json").read_text())
            summary["stages"].append({"stage": f"eval_real_{a}", "seconds": dt, "mode": meta["mode"], "n_items": meta["n_items"],
                                      "kind": "real_inference_tiny_model"})

    # 7. analysis (single entry point: tables, cohorts, tests, figures)
    _, dt = sh([PY, CTX / "analyze_beliefs.py", "--eval-dir", ev, "--probe-dir", build / "probe_sets", "--manifest", build / "manifest.csv",
                "--exposure-dir", build / "exposure", "--plot", "--bootstrap", "200", "--no-tracking"], env=env, log=logs / "7_analysis.log")
    an = ev / "analysis_v2"
    summary["stages"].append({"stage": "analysis", "seconds": dt, "outputs": sorted(p.name for p in an.iterdir()),
                              "cohorts": sorted(p.stem for p in (an / "cohorts").glob("*.json")), "kind": "plumbing_mock_scores"})

    (work / "smoke_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nsmoke summary -> {work / 'smoke_summary.json'}")


if __name__ == "__main__":
    main()
