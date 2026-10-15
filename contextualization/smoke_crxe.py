#!/usr/bin/env python
"""smoke_crxe.py -- deterministic OFFLINE C/R/X/E smoke workflow (no GPU, no network, no uploads).

    generation -> validation (with the training tokenizer) -> prepack + preflight (every occurrence
    within the training horizon) -> [TWO matched-seed replicates of all four arms trained and
    evaluated through run_arm.py into the shared build (eval/seed_<S>/, run_info per model tag) +
    an interrupted / resumed run + the refusal of a resume with a changed accumulation schedule]
    -> evaluation plumbing (mock scores, real BPE tokenizer) [-> cross-replicate analysis of the
    real evaluations with the input/contract checks, and the refusals of a mixed mock/real
    replicate and of foreign probe files]

Run from the repo root with the CPU test environment:
    $HOME/.cache/ctx_cpu_venv/bin/python contextualization/smoke_crxe.py --work /tmp/ctx_smoke [--train]

What each stage proves is written to <work>/smoke_summary.json. Mock-score stages are labelled
"plumbing"; real inference only happens with --train (depth-2 CPU models trained over exactly
the preflighted horizon) and is labelled "real_inference_tiny_model". Nothing here is a
scientific result; the tiny models are not expected to learn anything.
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


def sh(cmd, env=None, cwd=REPO, log=None, ok_codes=(0,)):
    print("+ " + " ".join(map(str, cmd)), flush=True)
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd], env=env or dict(os.environ), cwd=cwd, capture_output=True, text=True)
    if log is not None:
        Path(log).write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    if r.returncode not in ok_codes:
        print(r.stdout[-4000:], r.stderr[-4000:])
        raise SystemExit(f"stage failed ({r.returncode}): {cmd[:3]}")
    return r.stdout, round(time.time() - t0, 1), r.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="scratch directory (created; refused if it exists unless --overwrite)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="data seed of the build")
    ap.add_argument("--init-seed", type=int, default=7, help="initialization seed of the first replicate (--train); the second uses +1")
    ap.add_argument("--data-seed", type=int, default=3, help="shared prepacked row-order seed for all arms")
    ap.add_argument("--T", type=int, default=256, help="training context for prepack / tiny training (slot docs must fit one row)")
    ap.add_argument("--B", type=int, default=2, help="device batch size of the tiny training")
    ap.add_argument("--train", action="store_true", help="also train depth-2 models on CPU over the preflighted horizon for TWO "
                                                         "init seeds through run_arm.py, interrupt/resume one arm, refuse a changed "
                                                         "accumulation schedule, run real inference and the cross-replicate analysis")
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
    seeds = [args.init_seed, args.init_seed + 1]
    tag = f"d2_s{args.init_seed}"
    tag_of = {s: f"d2_s{s}" for s in seeds}
    policy = "fixed_nominal"   # the funded experiment's loss normalization (decision 2026-10-15), selected EXPLICITLY everywhere
    summary["loss_normalization"] = policy

    # 1. a REAL BPE tokenizer first (trained by nanochat on a throw-away synthetic build), so the
    #    experiment build can match X/E in tokens and validation can verify the token limits
    tokb = work / "tokbuild"
    sh([PY, CTX / "build_dataset.py", "--protocol", "v2", "--base-source", "synthetic", "--num-groups", "8", "--freq-grid", "1",
        "--base-chars", "600000", "--chars-per-shard", "200000", "--counterfact-limit", "0", "--out", tokb, "--seed", "99",
        "--no-tracking"], env=env, cwd=CTX, log=logs / "0_tokbuild.log")
    _, dt, _ = sh([PY, "-m", "scripts.tok_train", "--max-chars", "500000", "--vocab-size", "1024"],
                  env=dict(env, NANOCHAT_BASE_DIR=str(tokb / "arm_C")), log=logs / "1_tok_train.log")
    tok_src = tokb / "arm_C"
    summary["stages"].append({"stage": "tokenizer", "seconds": dt, "kind": "artifact"})

    # 2. generation (protocol v2, E on, collision audit, token-aware X/E matching)
    _, dt, _ = sh([PY, CTX / "build_dataset.py", "--protocol", "v2", "--base-source", "synthetic", "--num-groups", "40",
                   "--freq-grid", "1", "4", "--base-chars", "800000", "--chars-per-shard", "200000",
                   "--embedding-control", "--collision-audit", "--carrier-max-chars", "300", "--counterfact-limit", "0",
                   "--tokenizer-src", tok_src, "--out", build, "--seed", args.seed, "--no-tracking"],
                  env=env, cwd=CTX, log=logs / "2_build.log")
    bs = json.loads((build / "build_summary.json").read_text())
    summary["stages"].append({"stage": "generation", "seconds": dt, "protocol": bs["protocol"], "counts": bs["counts"],
                              "xe_within_limits": bs["xe_diagnostics"]["within_limits"],
                              "xe_token_limits_verified": bs["xe_diagnostics"]["token_limits_verified"],
                              "design_cells_empty": bs.get("design_cells", {}).get("empty_cells"), "kind": "artifact"})
    assert bs["xe_diagnostics"]["token_limits_verified"], "the build must verify X/E token limits with its tokenizer"

    # 3. validation (independent checks, token limits with the build's tokenizer)
    out, dt, _ = sh([PY, CTX / "validate.py", "--out", build], env=env, cwd=CTX, log=logs / "3_validate.log")
    rep = json.loads((build / "validation_report.json").read_text())
    summary["stages"].append({"stage": "validation", "seconds": dt, "passed": "ALL CHECKS PASSED" in out,
                              "n_checks": len(rep.get("checks", [])), "kind": "check"})

    # 4. prepack (matched schedule, bound to the tokenizer) + preflight covering EVERY row
    sh([PY, CTX / "exposure_audit.py", "--exp-dir", build, "--arms", *arms, "--mode", "prepack", "--T", args.T,
        "--data-seed", args.data_seed, "--B", args.B, "--grad-accum", "1", "--num-iterations", "1"],
       env=env, cwd=CTX, log=logs / "4_prepack.log")
    pm = json.loads((build / "arm_C" / "prepacked" / "prepack_manifest.json").read_text())
    n_rows = pm["n_rows"]
    num_iterations = n_rows // args.B  # grad_accum 1, world size 1: every row is scored exactly once
    _, dt, _ = sh([PY, CTX / "exposure_audit.py", "--exp-dir", build, "--arms", *arms, "--mode", "prepacked", "--T", args.T,
                   "--data-seed", args.data_seed, "--B", args.B, "--grad-accum", "1", "--num-iterations", num_iterations,
                   "--loss-normalization", policy],
                  env=env, cwd=CTX, log=logs / "4_preflight.log")
    pf = json.loads((build / "exposure" / "PREFLIGHT_OK.json").read_text())
    assert pf["loss_normalization"] == policy, pf.get("loss_normalization")
    cross = json.loads((build / "exposure" / "cross_arm.json").read_text())
    summary["stages"].append({"stage": "prepack_preflight", "seconds": dt, "n_rows": n_rows, "num_iterations": num_iterations,
                              "cross_arm": {k: cross[k] for k in ("n_occurrences", "n_present_in_all_arms", "n_step_agree", "n_row_agree")},
                              "loss_weighting": pf.get("loss_weighting"), "encoding_fingerprint": pf.get("encoding_fingerprint"),
                              "loss_normalization": pf["loss_normalization"], "loss_denominators": pf["loss_denominators"], "kind": "check"})
    # a stale preflight is refused by the launcher (negative check)
    bad = dict(pf)
    bad["prepack_dataset_id"] = {a: "0" * 64 for a in arms}
    stale = build / "exposure" / "PREFLIGHT_OK.json"
    good_text = stale.read_text()
    stale.write_text(json.dumps(bad))
    _, _, rc = sh([PY, CTX / "run_arm.py", "--arm", "C", "--exp-dir", build, "--depth", "2", "--data-loader", "prepacked",
                   "--init-seed", args.init_seed, "--data-seed", args.data_seed, "--loss-normalization", policy, "--no-tracking",
                   "--skip-upload", "--skip-eval", "--", "--device-type=cpu"], env=env, log=logs / "4_stale_preflight_refused.log", ok_codes=(0, 1, 2))
    stale.write_text(good_text)
    summary["stages"].append({"stage": "stale_preflight_refused_by_launcher", "refused": rc != 0, "kind": "check"})
    assert rc != 0, "run_arm must refuse a stale preflight"
    # a launch under the historical objective is refused by the launcher: the preflight certifies fixed_nominal
    _, _, rc = sh([PY, CTX / "run_arm.py", "--arm", "C", "--exp-dir", build, "--depth", "2", "--data-loader", "prepacked",
                   "--init-seed", args.init_seed, "--data-seed", args.data_seed, "--no-tracking", "--skip-upload", "--skip-eval",
                   "--", "--device-type=cpu"], env=env, log=logs / "4_uncertified_policy_refused.log", ok_codes=(0, 1, 2))
    err = (logs / "4_uncertified_policy_refused.log").read_text()
    summary["stages"].append({"stage": "uncertified_loss_normalization_refused_by_launcher", "refused": rc != 0,
                              "names_policy": "certifies loss_normalization 'fixed_nominal'" in err, "kind": "check"})
    assert rc != 0 and "certifies loss_normalization 'fixed_nominal'" in err

    # 5. optional tiny training: TWO matched-seed replicates of every arm through the real launcher
    #    (run_arm.py: train -> export manifest -> real inference into eval/seed_<S>/, run_info per model tag)
    if args.train:
        train_flags = ["--device-type=cpu", f"--max-seq-len={args.T}", f"--device-batch-size={args.B}",
                       f"--total-batch-size={args.B * args.T}", f"--num-iterations={num_iterations}", "--eval-every=-1",
                       "--core-metric-every=-1", "--sample-every=-1"]
        k = max(1, num_iterations // 2)
        for seed in seeds:
            for a in arms:
                extra = [f"--save-every={k}"] if (a == "C" and seed == seeds[0]) else []  # C keeps a mid-run checkpoint for the resume tests
                _, dt, _ = sh([PY, CTX / "run_arm.py", "--arm", a, "--exp-dir", build, "--depth", "2", "--data-loader", "prepacked",
                               "--init-seed", seed, "--data-seed", args.data_seed, "--loss-normalization", policy, "--no-tracking",
                               "--skip-upload", "--eval-batch-size", "16", "--", *train_flags, *extra],
                              env=env, log=logs / f"5_run_arm_{a}_s{seed}.log")
                info = json.loads((build / f"arm_{a}" / f"run_info_{tag_of[seed]}.json").read_text())
                meta = json.loads((build / "eval" / f"seed_{seed}" / f"beliefs_arm{a}.meta.json").read_text())
                summary["stages"].append({"stage": f"run_arm_{a}_s{seed}", "seconds": dt, "model_tag": info["model_tag"],
                                          "run_info": f"arm_{a}/run_info_{tag_of[seed]}.json", "eval_csv": meta["out_csv"],
                                          "eval_mode": meta["mode"], "n_items": meta["n_items"], "init_param_sha256": info["init_param_sha256"][:16],
                                          "kind": "real_training_and_inference_tiny_model"})
                assert meta["mode"] == "model_inference" and meta["model_tag"] == tag_of[seed] and meta["out_csv"].endswith(f"eval/seed_{seed}/beliefs_arm{a}.csv")
                assert meta["probe_construction"] and meta["checkpoint"]["init_param_sha256"] == info["init_param_sha256"]
                assert meta["checkpoint"]["loss_normalization"] == policy and info["loss_normalization"] == policy
                train_log = (logs / f"5_run_arm_{a}_s{seed}.log").read_text()
                assert "Loss normalization: fixed_nominal" in train_log and "| nominal:" in train_log and "per_active_tok:" in train_log
        metas = {s: {a: json.loads((build / f"arm_{a}" / "base_checkpoints" / tag_of[s] / f"meta_{num_iterations:06d}.json").read_text())
                     for a in arms} for s in seeds}
        hashes = {s: {a: m.get("init_param_sha256") for a, m in metas[s].items()} for s in seeds}
        for s_ in seeds:
            assert len(set(hashes[s_].values())) == 1 and None not in hashes[s_].values(), hashes[s_]
            assert all(m["preflight"] not in (None, "skipped") for m in metas[s_].values())
            assert all(m["num_iterations"] == num_iterations and m["grad_accum_steps"] == 1 and m["world_size"] == 1 for m in metas[s_].values())
            assert all(m["loss_normalization"] == policy and m["loss_denominator_per_microbatch"] == args.B * args.T for m in metas[s_].values())
        assert hashes[seeds[0]]["C"] != hashes[seeds[1]]["C"], "different init seeds must give different initial parameters"
        summary["seeds"]["recorded_in_meta"] = {f"seed_{s_}": {a: {"init_seed": m["init_seed"], "data_seed": m["data_seed"], "data_loader": m["data_loader"],
                                                                  "init_param_sha256": m["init_param_sha256"][:16],
                                                                  "encoding_fingerprint": (m.get("bindings") or {}).get("encoding_fingerprint", "")[:16]}
                                                               for a, m in metas[s_].items()} for s_ in seeds}
        summary["stages"].append({"stage": "matched_initialization", "identical_init_param_sha256_across_arms_per_seed": True,
                                  "distinct_across_seeds": True, "kind": "check"})
        # run identities and evaluation outputs are separate per seed (recheck finding 6)
        infos = {a: sorted(p.name for p in (build / f"arm_{a}").glob("run_info*.json")) for a in arms}
        assert all(infos[a] == [f"run_info_{tag_of[s_]}.json" for s_ in seeds] for a in arms), infos
        assert sorted(p.name for p in (build / "eval").iterdir()) == [f"seed_{s_}" for s_ in seeds]
        out, _, _ = sh([PY, CTX / "links.py", "--exp-dir", build, "--json"], env=env, log=logs / "5_links.log")
        rows = json.loads(out)
        assert sorted(r["model_tag"] for r in rows) == sorted(tag_of[s_] for s_ in seeds for _ in arms)
        summary["stages"].append({"stage": "replicates_separate_outputs", "run_info_files": infos, "eval_dirs": [f"eval/seed_{s_}" for s_ in seeds],
                                  "links_rows": len(rows), "kind": "check"})
        # interruption / resume of arm C (first replicate): the step-k checkpoint of the uninterrupted run is copied into
        # a separate checkpoint dir (as if the run had died right after saving it) and training resumes from it over the
        # SAME preflighted horizon; the end state must equal the uninterrupted run's
        base_flags = ["--depth=2", *train_flags, "--data-loader=prepacked", f"--init-seed={args.init_seed}",
                      f"--data-seed={args.data_seed}", f"--loss-normalization={policy}", "--run=dummy"]
        rtag = tag + "r"
        src_dir = build / "arm_C" / "base_checkpoints" / tag
        for dst_tag in (rtag, tag + "g", tag + "p"):
            dst_dir = build / "arm_C" / "base_checkpoints" / dst_tag
            dst_dir.mkdir(parents=True)
            for f in src_dir.glob(f"*_{k:06d}*"):
                shutil.copy(f, dst_dir / f.name)
        _, dt2, _ = sh([PY, "-m", "scripts.base_train", *base_flags, f"--model-tag={rtag}", f"--resume-from-step={k}"],
                       env=dict(env, NANOCHAT_BASE_DIR=str(build / "arm_C")), log=logs / "5_train_C_resumed.log")
        m_full = metas[seeds[0]]["C"]
        m_res = json.loads((build / "arm_C" / "base_checkpoints" / rtag / f"meta_{num_iterations:06d}.json").read_text())
        same_state = m_full["dataloader_state_dict"] == m_res["dataloader_state_dict"]
        same_init = m_full["init_param_sha256"] == m_res["init_param_sha256"] and m_res["init_seed"] == args.init_seed
        summary["stages"].append({"stage": "interrupt_resume_C", "seconds": dt2, "resume_from_step": m_res.get("resume_from_step"),
                                  "loader_state_identical_at_end": same_state, "init_identity_preserved": same_init, "kind": "check"})
        assert same_state and same_init and m_res.get("resume_from_step") == k
        # a resume with a CHANGED accumulation schedule is refused even behind a freshly generated, valid preflight
        # for that schedule (recheck finding 4): same horizon, grad_accum 2 (total batch doubled)
        pre_path = build / "exposure" / "PREFLIGHT_OK.json"
        pre_bytes = pre_path.read_bytes()
        sh([PY, CTX / "exposure_audit.py", "--exp-dir", build, "--arms", *arms, "--mode", "prepacked", "--T", args.T,
            "--data-seed", args.data_seed, "--B", args.B, "--grad-accum", "2", "--num-iterations", num_iterations,
            "--loss-normalization", policy],
           env=env, cwd=CTX, log=logs / "5_preflight_ga2.log")
        ga2_flags = [f if not f.startswith("--total-batch-size=") else f"--total-batch-size={2 * args.B * args.T}" for f in base_flags]
        out, _, rc = sh([PY, "-m", "scripts.base_train", *ga2_flags, f"--model-tag={tag}g", f"--resume-from-step={k}"],
                        env=dict(env, NANOCHAT_BASE_DIR=str(build / "arm_C")), log=logs / "5_resume_changed_accumulation_refused.log",
                        ok_codes=(0, 1))
        err = (logs / "5_resume_changed_accumulation_refused.log").read_text()
        refused = rc != 0 and "the requested training schedule differs from the checkpoint's" in err and "grad_accum_steps: checkpoint 1 vs requested 2" in err
        no_new_ckpt = sorted(p.name for p in (build / "arm_C" / "base_checkpoints" / f"{tag}g").glob("model_*.pt")) == [f"model_{k:06d}.pt"]
        pre_path.write_bytes(pre_bytes)   # the preflight the replicates were trained against
        summary["stages"].append({"stage": "resume_with_changed_accumulation_refused", "refused": refused, "no_checkpoint_written": no_new_ckpt,
                                  "kind": "check"})
        assert refused and no_new_ckpt, err[-2000:]
        # a resume under the historical objective is refused: the policy is bound to the checkpoint (decision 2026-10-15)
        mean_flags = [f if not f.startswith("--loss-normalization=") else "--loss-normalization=microbatch_mean" for f in base_flags]
        _, _, rc = sh([PY, "-m", "scripts.base_train", *mean_flags, f"--model-tag={tag}p", f"--resume-from-step={k}"],
                      env=dict(env, NANOCHAT_BASE_DIR=str(build / "arm_C")), log=logs / "5_resume_changed_objective_refused.log",
                      ok_codes=(0, 1))
        err = (logs / "5_resume_changed_objective_refused.log").read_text()
        refused = rc != 0 and "the training objective differs" in err and "a policy change on resume is refused" in err
        no_new_ckpt = sorted(p.name for p in (build / "arm_C" / "base_checkpoints" / f"{tag}p").glob("model_*.pt")) == [f"model_{k:06d}.pt"]
        summary["stages"].append({"stage": "resume_with_changed_loss_normalization_refused", "refused": refused, "no_checkpoint_written": no_new_ckpt,
                                  "kind": "check"})
        assert refused and no_new_ckpt, err[-2000:]

    # 6. evaluation plumbing: mock scores with the REAL BPE tokenizer (every arm)
    ev = build / "eval_mock"
    for a in arms:
        _, dt, _ = sh([PY, CTX / "eval_beliefs.py", "--exp-dir", build, "--arm", a, "--dry-run", "--tokenizer-dir",
                       build / f"arm_{a}", "--no-tracking", "--out-csv", ev / f"beliefs_arm{a}.csv"], env=env, log=logs / f"6_eval_mock_{a}.log")
        meta = json.loads((ev / f"beliefs_arm{a}.meta.json").read_text())
        summary["stages"].append({"stage": f"eval_plumbing_{a}", "seconds": dt, "mode": meta["mode"], "n_items": meta["n_items"],
                                  "coverage_complete": meta["coverage"]["complete"], "kind": "plumbing_mock_scores"})
    # 7. cross-replicate analysis of the real evaluations (two seeds) with the input / contract checks
    if args.train:
        evr = build / "eval"
        _, dt, _ = sh([PY, CTX / "analyze_beliefs.py", "--eval-dir", evr, "--probe-dir", build / "probe_sets", "--manifest",
                       build / "manifest.csv", "--exposure-dir", build / "exposure", "--plot", "--bootstrap", "100", "--no-tracking"],
                      env=env, log=logs / "8_analysis_real.log")
        an = evr / "analysis_v2"
        compat = json.loads((an / "compatibility.json").read_text())
        bound = {f"{rep}/{a}": s_["inputs_bound"] for rep, arms_ in compat["replicates"].items() for a, s_ in arms_.items()}
        paired = (an / "tables" / "replicates__paired.csv").read_text().splitlines()
        summary["stages"].append({"stage": "analysis_real_two_replicates", "seconds": dt, "outputs": sorted(p.name for p in an.iterdir()),
                                  "cohorts": sorted(p.stem for p in (an / "cohorts").glob("*.json")),
                                  "compatibility_ok": compat["ok"], "modes": compat["modes"], "replicates": sorted(compat["replicates"]),
                                  "inputs_bound": bound, "n_items_with_identity": compat["n_items_with_identity"],
                                  "n_item_identity_mismatches": compat["n_item_identity_mismatches"],
                                  "replicate_table_rows": len(paired) - 1, "kind": "real_inference_tiny_model"})
        assert compat["ok"] and sorted(compat["replicates"]) == [f"seed_{s_}" for s_ in seeds] and compat["modes"] == ["model_inference"]
        assert all(s_["checkpoint"]["loss_normalization"] == policy for arms_ in compat["replicates"].values() for s_ in arms_.values())
        assert all(v is True for v in bound.values()) and compat["n_item_identity_mismatches"] == 0 and len(paired) > 1
        # a mock evaluation replacing one real replicate's arm must be refused (within the replicate AND across replicates)
        mixed = work / "mixed_eval"
        shutil.copytree(evr, mixed, ignore=shutil.ignore_patterns("analysis_v2"))
        for ext in (".csv", ".meta.json"):
            shutil.copy(ev / f"beliefs_armR{ext}", mixed / f"seed_{seeds[1]}" / f"beliefs_armR{ext}")
        _, _, rc = sh([PY, CTX / "analyze_beliefs.py", "--eval-dir", mixed, "--probe-dir", build / "probe_sets", "--no-tracking",
                       "--bootstrap", "0"], env=env, log=logs / "8_analysis_mixed_refused.log", ok_codes=(0, 1))
        err = (logs / "8_analysis_mixed_refused.log").read_text()
        summary["stages"].append({"stage": "analysis_refuses_mixed_mock_and_real_across_replicates", "refused": rc != 0,
                                  "cross_replicate_violation": "replicates mix evaluation modes" in err, "kind": "check"})
        assert rc != 0 and "replicates mix evaluation modes" in err
        # probe files that are not the evaluated ones (same ids, doses rewritten) must be refused (recheck finding 2)
        foreign = work / "foreign_probe_sets"
        foreign.mkdir()
        recs = [json.loads(l) for l in (build / "probe_sets" / "injected_facts.jsonl").read_text().splitlines() if l.strip()]
        (foreign / "injected_facts.jsonl").write_text("".join(json.dumps(dict(r, assigned_frequency=256)) + "\n" for r in recs))
        shutil.copy(build / "probe_sets" / "heldout_facts.jsonl", foreign / "heldout_facts.jsonl")
        _, _, rc = sh([PY, CTX / "analyze_beliefs.py", "--eval-dir", evr, "--probe-dir", foreign, "--build-summary",
                       build / "build_summary.json", "--no-tracking", "--bootstrap", "0"], env=env,
                      log=logs / "8_analysis_foreign_probes_refused.log", ok_codes=(0, 1))
        err = (logs / "8_analysis_foreign_probes_refused.log").read_text()
        summary["stages"].append({"stage": "analysis_refuses_foreign_probe_files", "refused": rc != 0,
                                  "binding_violation": "is not the file this analysis builds cohorts/doses/labels from" in err, "kind": "check"})
        assert rc != 0 and "is not the file this analysis builds cohorts/doses/labels from" in err

    (work / "smoke_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nsmoke summary -> {work / 'smoke_summary.json'}")


if __name__ == "__main__":
    main()
