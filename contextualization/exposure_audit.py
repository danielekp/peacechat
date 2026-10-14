#!/usr/bin/env python
"""exposure_audit.py -- measure (never estimate) how often each inserted claim was actually scored.

Modes (docs/EXPOSURE.md):

  historical   replay nanochat's best-fit loader over an arm's shards with the training config and
               write exposure/<arm>/occurrence_exposure.csv + summary.json (Exposure index v1)
  prepacked    turn the prepack-time occurrence index of arm_<A>/prepacked/ into the exposure index
               for a training config (scheduled step per epoch), verify that every occurrence is
               complete and shares its row_index across arms (exit 2 otherwise)
  prepack      build arm_<A>/prepacked/ for the given arms (rows planned once across arms)

Examples (from the repo root, with the CPU env):

  python contextualization/exposure_audit.py --exp-dir D --arm R --mode historical \\
      --B 8 --T 2048 --grad-accum 64 --num-iterations 8352            # the July-2026 config
  python contextualization/exposure_audit.py --exp-dir D --arms R X --mode historical --B 8 --T 2048 \\
      --grad-accum 64 --num-iterations 8352                            # + exposure/cross_arm.json
  python contextualization/exposure_audit.py --exp-dir D --arms C R X E --mode prepack --T 2048
  python contextualization/exposure_audit.py --exp-dir D --arms R X E --mode prepacked --B 8 --T 2048 \\
      --grad-accum 64 --num-iterations 8352 --data-seed 0

The tokenizer is read from <exp-dir>/arm_<first arm>/tokenizer unless --tokenizer-dir is given.
Legacy manifests (no char spans) need --legacy-locator module:function, a callable
row -> insert_text (or a dict with insert_text/claim_text/subject/value) that regenerates the
inserted sentence; every insert must then match exactly once in exactly one document of its shard.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctx.exposure import ExposureError, HISTORICAL_TRAIN_CONFIG, audit_arm, cross_arm_report  # noqa: E402
from ctx.prepack import PrepackError, prepack_experiment, schedule_exposure, verify_prepack  # noqa: E402


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
                print(f"arm {a}: {man['n_rows']} rows, scored {man['scored_tokens_total']:,} masked {man['masked_tokens_total']:,} "
                      f"intervention scored {man['intervention_tokens_scored']:,} occurrences {man['n_occurrences']} "
                      f"status {man['exposure_status_counts']} dropped top-ups {len(man['dropped_topup_docs'][a])}")
            print(f"prepack audit: {res['verify']['n_occurrences_indexed']} occurrences, row_index disagreements "
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
        else:  # prepacked: verify first so a failing set never overwrites a good exposure index
            rep = verify_prepack(exp_dir, arms)
            print(f"prepack audit: {rep['n_occurrences_indexed']} occurrences complete in every arm, "
                  f"{rep['n_row_index_disagree']} row_index disagreements")
            for a in arms:
                s = schedule_exposure(exp_dir, a, B=args.B, grad_accum_steps=args.grad_accum, num_iterations=args.num_iterations,
                                      world_size=args.world_size, data_seed=args.data_seed, out_dir=out_dir)
                print(f"arm {a}: {s['n_occurrences']} occurrences -> {s['n_rows']} rows over {s['horizon']['epochs_touched']} epochs; "
                      f"status {s['status_counts']}; scored tokens {s['scored_tokens']} -> {s['exposure_csv']}")
        if len(arms) > 1:
            rep = cross_arm_report(exp_dir, arms, out_dir=out_dir, num_iterations=args.num_iterations)
            print(f"cross-arm: {rep['n_occurrences']} occurrences, step agree {rep['n_step_agree']}, row agree {rep['n_row_agree']}, "
                  f"lost {json.dumps(rep['lost_counts'])} -> {out_dir / 'cross_arm.json'}")
            if args.mode == "prepacked" and rep["n_row_agree"] != rep["n_occurrences"]:
                raise PrepackError("prepacked arms do not agree on every occurrence's row/step")
        return 0
    except (ExposureError, PrepackError) as e:
        print(f"AUDIT FAILED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
