#!/usr/bin/env python
"""exposure_audit.py -- measure (never estimate) how often each inserted claim was actually scored.

Modes (docs/EXPOSURE.md):

  historical   replay nanochat's best-fit loader over an arm's shards with the training config and
               write exposure/<arm>/occurrence_exposure.csv + summary.json (Exposure index v1)
  prepack      build arm_<A>/prepacked/ for the given arms (rows planned once across arms, staged and
               moved into place only after the prepack gate passed)
  prepacked    THE PREFLIGHT for a training configuration: run the prepack gate (verify_prepack),
               schedule every occurrence (step per epoch), require that every occurrence is complete
               in every arm within the horizon and scheduled at the same row AND step in every arm,
               that the arms' manifests share data_seed / T / n_rows, and write
               <exp-dir>/exposure/PREFLIGHT_OK.json (deleted first; exit 2 and no file on failure).
               scripts/base_train.py --data-loader prepacked refuses to train unless that file
               matches its B / T / grad_accum_steps / num_iterations / world_size / data_seed /
               prepack_dataset_id / encoding fingerprint. --arms must list every arm that was
               prepacked together.

Examples (from the repo root, with the CPU env):

  python contextualization/exposure_audit.py --exp-dir D --arm R --mode historical \\
      --B 8 --T 2048 --grad-accum 64 --num-iterations 8352            # the July-2026 config
  python contextualization/exposure_audit.py --exp-dir D --arms R X --mode historical --B 8 --T 2048 \\
      --grad-accum 64 --num-iterations 8352                            # + exposure/cross_arm.json
  python contextualization/exposure_audit.py --exp-dir D --arms C R X E --mode prepack --T 2048
  python contextualization/exposure_audit.py --exp-dir D --arms C R X E --mode prepacked --B 8 --T 2048 \\
      --grad-accum 64 --num-iterations 8352 --data-seed 0 [--check-hashes]

The tokenizer is read from <exp-dir>/arm_<first arm>/tokenizer unless --tokenizer-dir is given; the
prepack mode refuses an arm tokenizer.pkl that encodes differently from the tokenizer used.
Legacy manifests (no char spans) need --legacy-locator module:function, a callable
row -> insert_text (or a dict with insert_text/claim_text/subject/value) that regenerates the
inserted sentence; every insert must then match exactly once in exactly one document of its shard.
"""

import argparse
import datetime as _dt
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctx.exposure import ExposureError, HISTORICAL_TRAIN_CONFIG, audit_arm, cross_arm_report  # noqa: E402
from ctx.prepack import (PrepackError, load_prepack_manifest, loss_weighting_report, prepack_experiment,  # noqa: E402
                         schedule_exposure, verify_prepack)
from ctx.protocol import code_revision, sha256_file  # noqa: E402

PREFLIGHT_SCHEMA = 1
PREFLIGHT_NAME = "PREFLIGHT_OK.json"


def _tokenizer(args, arms):
    from nanochat.tokenizer import RustBPETokenizer
    tdir = Path(args.tokenizer_dir) if args.tokenizer_dir else Path(args.exp_dir) / f"arm_{arms[0]}" / "tokenizer"
    if not (tdir / "tokenizer.pkl").exists():
        raise SystemExit(f"no tokenizer.pkl under {tdir}; pass --tokenizer-dir")
    return RustBPETokenizer.from_directory(str(tdir))


def _locator(spec):
    if not spec:
        return None
    mod, _, fn = spec.partition(":")
    if not fn:
        raise SystemExit("--legacy-locator must be module:function")
    return getattr(importlib.import_module(mod), fn)


LOSS_NORMALIZATIONS = ("microbatch_mean", "fixed_nominal")   # = nanochat.loss_normalization.LOSS_NORMALIZATIONS (no nanochat import here)


def prepacked_preflight(exp_dir, arms, *, B, grad_accum_steps, num_iterations, world_size, data_seed, out_dir,
                        check_hashes=False, loss_normalization="microbatch_mean") -> dict:
    """The prepacked audit + gate (see the module docstring). Returns the PREFLIGHT_OK content after
    writing it; raises PrepackError (and leaves no PREFLIGHT_OK.json) on any failure."""
    if loss_normalization not in LOSS_NORMALIZATIONS:
        raise PrepackError(f"unknown loss normalization {loss_normalization!r}; known: {LOSS_NORMALIZATIONS}")
    exp_dir, out_dir = Path(exp_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preflight = out_dir / PREFLIGHT_NAME
    if preflight.exists():
        preflight.unlink()
        print(f"removed stale {preflight}", flush=True)
    rep = verify_prepack(exp_dir, arms, check_hashes=check_hashes)
    print(f"prepack audit: {rep['n_occurrences_indexed']} occurrences complete in every arm, "
          f"{rep['n_row_index_disagree']} row_index disagreements, required {rep['required_occurrences']}, "
          f"control slots {rep['control_slots']}, hashes {'checked' if check_hashes else 'not re-checked'}, "
          f"tokenizers {rep['tokenizer_checks']}", flush=True)
    mans = {a: load_prepack_manifest(exp_dir, a) for a in arms}
    planned = list(mans[arms[0]]["arms_planned_together"])
    if sorted(arms) != sorted(planned):
        raise PrepackError(f"the preflight must cover every arm prepacked together: planned {planned}, audited {list(arms)}")
    seeds = {a: mans[a].get("data_seed") for a in arms}
    if len(set(seeds.values())) != 1:
        raise PrepackError(f"the arms' prepack manifests disagree on data_seed: {seeds}")
    for k in ("T", "n_rows"):
        vals = {a: mans[a].get(k) for a in arms}
        if len(set(vals.values())) != 1:
            raise PrepackError(f"the arms' prepack manifests disagree on {k}: {vals}")
    manifest_seed = next(iter(seeds.values()))
    effective_seed = int(manifest_seed) if data_seed is None else int(data_seed)
    summaries = {}
    for a in arms:
        s = schedule_exposure(exp_dir, a, B=B, grad_accum_steps=grad_accum_steps, num_iterations=num_iterations,
                              world_size=world_size, data_seed=effective_seed, out_dir=out_dir)
        summaries[a] = s
        print(f"arm {a}: {s['n_occurrences']} occurrences -> {s['n_rows']} rows over {s['horizon']['epochs_touched']} epochs; "
              f"status {s['status_counts']}; scored tokens {s['scored_tokens']} -> {s['exposure_csv']}", flush=True)
    failures = []
    for a in arms:
        bad = {k: v for k, v in summaries[a]["status_counts"].items() if v and k not in ("complete", "not_in_horizon")}
        if bad:
            failures.append(f"arm {a}: statuses other than complete/not_in_horizon: {bad}")
    indexed_arms = [a for a in arms if summaries[a]["n_occurrences"] > 0]
    if not indexed_arms:
        raise PrepackError("no arm has an occurrence index; nothing to audit")
    lw = loss_weighting_report(exp_dir, arms, B=B, grad_accum_steps=grad_accum_steps, num_iterations=num_iterations,
                               world_size=world_size, data_seed=effective_seed)
    cross = cross_arm_report(exp_dir, indexed_arms, out_dir=out_dir, num_iterations=num_iterations,
                             extra={"loss_weighting": lw, "preflight_config": {
                                 "B": B, "T": int(mans[arms[0]]["T"]), "grad_accum_steps": grad_accum_steps,
                                 "num_iterations": num_iterations, "world_size": world_size, "data_seed": effective_seed,
                                 "loss_normalization": loss_normalization}})
    print(f"cross-arm: {cross['n_occurrences']} occurrences, present in all arms {cross['n_present_in_all_arms']}, "
          f"step agree {cross['n_step_agree']}, row agree {cross['n_row_agree']}, effective in all arms "
          f"{cross['n_all_effective']}, lost {json.dumps(cross['lost_counts'])} -> {out_dir / 'cross_arm.json'}", flush=True)
    n = cross["n_occurrences"]
    required = max(rep["required_occurrences"].values()) if rep["required_occurrences"] else 0
    if n != required or n != rep["n_occurrences_indexed"]:
        failures.append(f"{n} occurrences in the exposure indexes but {required} required by manifest.csv "
                        f"({rep['n_occurrences_indexed']} indexed)")
    if not (cross["n_present_in_all_arms"] == n == cross["n_step_agree"] == cross["n_row_agree"]):
        failures.append(f"prepacked arms do not agree on every occurrence's row/step: {n} occurrences, present in all "
                        f"{cross['n_present_in_all_arms']}, step agree {cross['n_step_agree']}, row agree {cross['n_row_agree']}")
    for a in indexed_arms:
        z = cross["per_arm"][a]["n_zero_effective"]
        if z:
            failures.append(f"arm {a}: {z} occurrences never scored within the horizon "
                            f"(not_in_horizon rows {cross['per_arm'][a]['n_rows_not_in_horizon']}); shrink the data, "
                            f"lengthen the horizon or change data_seed/B")
    if cross["n_all_effective"] != n:
        failures.append(f"only {cross['n_all_effective']} of {n} occurrences are effective in every arm")
    if failures:
        raise PrepackError("preflight failed: " + "; ".join(failures))
    content = {
        "schema_version": PREFLIGHT_SCHEMA,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "exp_dir": str(exp_dir.resolve()),
        "arms": list(arms), "intervention_arms": rep["intervention_arms"], "control_arms": rep["control_arms"],
        "B": int(B), "T": int(mans[arms[0]]["T"]), "grad_accum_steps": int(grad_accum_steps),
        "num_iterations": int(num_iterations), "world_size": int(world_size), "data_seed": effective_seed,
        # the training objective this preflight certifies (base_train refuses a run with another policy; a preflight
        # written before this key existed certified the historical microbatch_mean)
        "loss_normalization": loss_normalization,
        "loss_denominators": {"per_microbatch_nominal": int(B) * int(mans[arms[0]]["T"]) * int(grad_accum_steps),
                              "per_step_all_ranks_nominal": int(B) * int(mans[arms[0]]["T"]) * int(grad_accum_steps) * int(world_size),
                              "note": "fixed_nominal divides each micro-batch's summed token losses by per_microbatch_nominal; "
                                      "microbatch_mean divides by that micro-batch's active targets (see loss_weighting)"},
        "manifest_data_seed": manifest_seed,
        "prepack_dataset_id": {a: mans[a].get("prepack_dataset_id") for a in arms},
        "prepack_manifest_sha256": {a: sha256_file(exp_dir / f"arm_{a}" / "prepacked" / "prepack_manifest.json") for a in arms},
        "prepack_run_id": rep.get("prepack_run_id"),
        "encoding_fingerprint": rep["encoding_fingerprint"],
        "tokenizer_file_sha256": {a: mans[a].get("tokenizer_file_sha256") for a in arms},
        "tokenizer_checks": rep["tokenizer_checks"],
        "build_summary_sha256": mans[arms[0]].get("build_summary_sha256"),
        "manifest_sha256": rep.get("manifest_sha256"),
        "hashes_checked": bool(check_hashes),
        "verify": {k: rep[k] for k in ("roles", "required_occurrences", "control_slots", "n_occurrences_indexed",
                                       "n_row_index_disagree", "active_targets_total", "n_spans_checked", "masks_checked")},
        "cross_arm": {k: cross[k] for k in ("arms", "n_occurrences", "n_present_in_all_arms", "n_step_agree", "n_row_agree",
                                            "n_all_effective", "lost_counts", "status_pairs", "per_arm")},
        "exposure": {a: {k: summaries[a][k] for k in ("role", "status_counts", "n_rows", "n_occurrences", "n_facts", "by_dose",
                                                      "scored_tokens", "horizon") if k in summaries[a]} for a in arms},
        "loss_weighting": lw,
        "code": code_revision(),
    }
    with open(preflight, "w") as fh:
        json.dump(content, fh, indent=2)
    print(f"PREFLIGHT OK -> {preflight} (loss normalization {loss_normalization})", flush=True)
    return content


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--arm", help="one arm (C/R/X/E)")
    g.add_argument("--arms", nargs="+", help="several arms (also writes exposure/cross_arm.json)")
    ap.add_argument("--mode", choices=["historical", "prepacked", "prepack"], default="historical")
    ap.add_argument("--B", type=int, default=HISTORICAL_TRAIN_CONFIG["B"], help="device batch size (rows per rank per micro-batch)")
    ap.add_argument("--T", type=int, default=HISTORICAL_TRAIN_CONFIG["T"], help="max_seq_len")
    ap.add_argument("--grad-accum", type=int, default=HISTORICAL_TRAIN_CONFIG["grad_accum_steps"])
    ap.add_argument("--num-iterations", type=int, default=HISTORICAL_TRAIN_CONFIG["num_iterations"])
    ap.add_argument("--world-size", type=int, default=HISTORICAL_TRAIN_CONFIG["world_size"])
    ap.add_argument("--out", default=None, help="output dir (default <exp-dir>/exposure)")
    ap.add_argument("--buffer-size", type=int, default=1000, help="loader document buffer (nanochat default 1000)")
    ap.add_argument("--tokenizer-batch-size", type=int, default=128, help="loader text batch (nanochat default 128)")
    ap.add_argument("--tokenizer-threads", type=int, default=4)
    ap.add_argument("--tokenizer-dir", default=None)
    ap.add_argument("--data-subdir", default="base_data")
    ap.add_argument("--legacy-locator", default=None, help="module:function for legacy manifests")
    ap.add_argument("--data-seed", type=int, default=None, help="prepacked: row-order seed (default: the prepack manifest's)")
    ap.add_argument("--pad-token-id", type=int, default=None, help="prepack: pad id (default BOS)")
    ap.add_argument("--rows-per-file", type=int, default=4096, help="prepack: rows per npy file")
    ap.add_argument("--overwrite", action="store_true", help="prepack: replace an existing prepacked dir")
    ap.add_argument("--check-hashes", action="store_true", help="prepacked: re-hash every rows/mask file against the manifest")
    ap.add_argument("--loss-normalization", choices=list(LOSS_NORMALIZATIONS), default="microbatch_mean",
                    help="prepacked: the training objective the preflight certifies (base_train --loss-normalization must match; "
                         "funded runs select fixed_nominal explicitly)")
    ap.add_argument("--progress", type=int, default=0, help="print progress every N micro-batches / rows")
    args = ap.parse_args(argv)
    arms = args.arms or [args.arm]
    exp_dir = Path(args.exp_dir)
    out_dir = Path(args.out) if args.out else exp_dir / "exposure"

    try:
        if args.mode == "prepack":
            tok = _tokenizer(args, arms)
            res = prepack_experiment(exp_dir, arms, tok, args.T, buffer_size=args.buffer_size, pad_token_id=args.pad_token_id,
                                     data_seed=-1 if args.data_seed is None else args.data_seed, rows_per_file=args.rows_per_file,
                                     tokenizer_batch_size=args.tokenizer_batch_size, tokenizer_threads=args.tokenizer_threads,
                                     data_subdir=args.data_subdir, overwrite=args.overwrite, progress=args.progress or None)
            for a, man in res["arms"].items():
                print(f"arm {a} ({man['role']}): {man['n_rows']} rows, scored {man['scored_tokens_total']:,} masked {man['masked_tokens_total']:,} "
                      f"intervention scored {man['intervention_tokens_scored']:,} occurrences {man['n_occurrences']} "
                      f"status {man['exposure_status_counts']} dropped top-ups {len(man['dropped_topup_docs'][a])} "
                      f"active/row {man['active_targets_per_row']['min']}..{man['active_targets_per_row']['max']} "
                      f"(mean {man['active_targets_per_row']['mean']:.1f}) dataset {man['prepack_dataset_id'][:12]}")
            print(f"encoding fingerprint {res['verify']['encoding_fingerprint'][:12]}; prepack audit: "
                  f"{res['verify']['n_occurrences_indexed']} occurrences, row_index disagreements "
                  f"{res['verify']['n_row_index_disagree']}, problems {res['verify']['n_problems']}")
            return 0

        if args.mode == "historical":
            tok = _tokenizer(args, arms)
            for a in arms:
                s = audit_arm(exp_dir, a, tok, B=args.B, T=args.T, grad_accum_steps=args.grad_accum,
                              num_iterations=args.num_iterations, world_size=args.world_size, buffer_size=args.buffer_size,
                              tokenizer_batch_size=args.tokenizer_batch_size, tokenizer_threads=args.tokenizer_threads,
                              out_dir=out_dir, data_subdir=args.data_subdir, legacy_insert_fn=_locator(args.legacy_locator),
                              progress=args.progress or None)
                print(f"arm {a}: {s['n_occurrences']} occurrences -> {s['n_rows']} exposure rows; status {s['status_counts']}; "
                      f"scored tokens {s['scored_tokens']}; replay {s['replay']['per_rank'][0]['micro_batches']} micro-batches/rank "
                      f"(max epoch {max(r['max_epoch'] for r in s['replay']['per_rank'])}) -> {s['exposure_csv']}")
            if len(arms) > 1:
                rep = cross_arm_report(exp_dir, arms, out_dir=out_dir, num_iterations=args.num_iterations)
                print(f"cross-arm: {rep['n_occurrences']} occurrences, step agree {rep['n_step_agree']}, row agree {rep['n_row_agree']}, "
                      f"lost {json.dumps(rep['lost_counts'])} -> {out_dir / 'cross_arm.json'}")
            return 0

        prepacked_preflight(exp_dir, arms, B=args.B, grad_accum_steps=args.grad_accum, num_iterations=args.num_iterations,
                            world_size=args.world_size, data_seed=args.data_seed, out_dir=out_dir, check_hashes=args.check_hashes,
                            loss_normalization=args.loss_normalization)
        return 0
    except (ExposureError, PrepackError) as e:
        print(f"AUDIT FAILED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
