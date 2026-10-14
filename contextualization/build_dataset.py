#!/usr/bin/env python
"""
build_dataset.py -- build the C/R/X[/E] contextualization pretraining datasets for nanochat.

Produces, per arm, <out>/arm_{C,R,X[,E]}/<data-subdir>/shard_XXXXX.parquet in nanochat's exact
format, plus manifest.csv (one row per occurrence x arm, with character spans), probe_sets/*.jsonl,
build_summary.json, shard hashes and BUILD_COMPLETE.json. Everything is seeded: same args + same
--seed => byte-identical shards. Output directories are versioned: an existing build is never
overwritten unless --overwrite is given.

Protocols (see docs/PROTOCOL_V2.md):
  --protocol v2 (default)  corrected generator: unique semantic groups, one injected candidate per
                           training group, whole-group holdout, decoupled rendering policies.
  --protocol v1            the historical generator/rendering (only for a verified recreation of the
                           July-2026 design; its known defects are documented in docs/ERRATA.md).

Corrected C/R/X/E build (CPU box; FineWeb shards download on demand):
  python build_dataset.py --protocol v2 --out ~/ctx_v2_build1 --base-chars 45e9 --num-groups 2500 \
      --freq-grid 1 4 16 64 256 --embedding-control --collision-audit --tokenizer-src ~/ctx_tok --seed 2026

Offline smoke test (no network, fast):
  python build_dataset.py --protocol v2 --base-source synthetic --num-groups 60 --freq-grid 1 4 \
      --base-chars 3_000_000 --chars-per-shard 300_000 --embedding-control --counterfact-limit 0 \
      --out /tmp/ctx_smoke --seed 0 --no-tracking
"""

import argparse
import json

from ctx import blend
from ctx import tracking as T
from ctx.nanochat_io import CHARS_PER_SHARD, ROW_GROUP_SIZE, DEFAULT_DATA_SUBDIR
from ctx.templates import RenderPolicy


def _float_int(x):
    return int(float(x))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # output / reproducibility
    p.add_argument("--out", required=True, help="output dir (creates arm_C/ arm_R/ arm_X/ [arm_E/]); must be new")
    p.add_argument("--seed", type=int, default=1234, help="DATA seed (pool, split, doses, rendering, slots)")
    p.add_argument("--protocol", choices=["v1", "v2"], default="v2",
                   help="v2 = corrected generator/rendering (default); v1 = historical protocol")
    p.add_argument("--overwrite", action="store_true", help="delete an existing build in --out first (never implicit)")
    p.add_argument("--cache-dir", default=None, help="base shard cache root (default <out>/_base_cache; keyed by source/config)")
    p.add_argument("--probes-only", action="store_true",
                   help="verify the pool arguments against the existing build and re-emit probe_sets/ only")
    p.add_argument("--overwrite-probes", action="store_true", help="with --probes-only: replace probe_sets/ in place")

    # base corpus
    p.add_argument("--base-source", choices=["fineweb", "synthetic"], default="fineweb",
                   help="'fineweb' = karpathy/fineweb-edu-100b-shuffle (download on demand); 'synthetic' = offline prose")
    p.add_argument("--fineweb-revision", default=None,
                   help="HF dataset revision (commit sha) to pin FineWeb shards to; recorded in build_summary.json")
    p.add_argument("--base-chars", type=_float_int, default=3_000_000_000, help="total base char budget (e.g. 45e9)")
    p.add_argument("--inject-frac", type=float, default=0.02, help="advisory target inserted-char fraction (reported)")

    # fact pool
    p.add_argument("--num-groups", type=int, default=None,
                   help="v2: number of semantic groups (entities); each has 2 candidate statements; "
                        "injected statements = training groups = round((1-heldout_frac) * num_groups)")
    p.add_argument("--num-facts", type=int, default=4000, help="v1: target distinct facts (v2 fallback: num_facts/2 groups)")
    p.add_argument("--heldout-frac", type=float, default=0.3, help="fraction of groups (v2) / facts (v1) held out")
    p.add_argument("--contested-frac", type=float, default=0.08, help="v1 only")
    p.add_argument("--freq-grid", type=int, nargs="+", default=[1, 4, 16, 64, 256], help="injection-dose grid")
    p.add_argument("--design", choices=["one_sided", "competing"], default="one_sided",
                   help="v2: one_sided = one injected candidate per group, rival unasserted (primary); "
                        "competing = both candidates injected with crossed doses (separate experiment)")
    p.add_argument("--rival-freq-grid", type=int, nargs="+", default=None, help="v2 competing design: rival dose grid (0 allowed)")
    p.add_argument("--include-nonfunctional", action="store_true",
                   help="v2: also use author_work / inventor_of (several answers can be valid; excluded by default)")
    p.add_argument("--relations", nargs="+", default=None, help="v2: restrict to these relation keys")
    p.add_argument("--no-real", action="store_true", help="v2: synthetic groups only (no real-entity pairs)")
    p.add_argument("--collision-audit", action="store_true",
                   help="v2: scan everything the model sees (Arm C shards) for novel subject/value strings and label "
                        "probe novelty audited_absent/audited_present (otherwise 'unverified')")
    p.add_argument("--counterfact-limit", type=int, default=2000, help="max CounterFact held-out probes (0 disables)")

    # rendering
    p.add_argument("--embed", dest="embed", action="store_true", default=True, help="embed inserts in carriers (default)")
    p.add_argument("--no-embed", dest="embed", action="store_false")
    p.add_argument("--carrier-max-chars", type=int, default=6000,
                   help="v2: crop carriers to whole leading sentences within this budget (same crop for every arm) so a "
                        "slot document always fits one training row; 0 disables")
    p.add_argument("--render-mode", choices=["variable", "fixed"], default="variable",
                   help="v2: 'fixed' = fixed-rendering dose control (one paraphrase/wrapper/source, no lead-in/tail)")
    for k, choices in RenderPolicy._CHOICES.items():
        p.add_argument(f"--policy-{k}", dest=f"policy_{k}", choices=list(choices), default=None,
                       help=f"v2 render policy '{k}' (default: {getattr(RenderPolicy(), k)!r})")
    p.add_argument("--register", dest="register", action="store_true", default=True, help="v1: rotate register lead-ins")
    p.add_argument("--no-register", dest="register", action="store_false")
    p.add_argument("--verbatim-control", action="store_true", help="v1: repeat ONE fixed string per fact")
    p.add_argument("--source-per-fact", action="store_true", help="Arm X: one consistent source per fact")
    p.add_argument("--embedding-control", action="store_true", help="add Arm E (own-voice embedded claim, no source)")

    # tokenizer
    p.add_argument("--tokenizer-src", default=None, help="nanochat base dir holding a trained tokenizer/ to copy into each arm")

    # nanochat format knobs
    p.add_argument("--chars-per-shard", type=int, default=CHARS_PER_SHARD)
    p.add_argument("--row-group-size", type=int, default=ROW_GROUP_SIZE)
    p.add_argument("--data-subdir", default=DEFAULT_DATA_SUBDIR)
    p.add_argument("--synthetic-docs-per-shard", type=int, default=2000)
    p.add_argument("--parity-tol", type=float, default=0.0049, help="max char spread below the C/R/X ceiling")

    # tracking
    p.add_argument("--no-tracking", action="store_true", help="don't upload build metadata to HF / log to wandb")
    args = p.parse_args(argv)
    if args.protocol == "v2" and args.num_groups is None:
        args.num_groups = max(1, args.num_facts // 2)
        print(f"NOTE  --num-groups not given; using num_facts/2 = {args.num_groups} semantic groups")
    if args.carrier_max_chars == 0:
        args.carrier_max_chars = None
    return args


def main():
    args = parse_args()
    print(f"Building into {args.out} (protocol={args.protocol}, base={args.base_source}, base_chars={args.base_chars:,}, "
          f"seed={args.seed}, E={'on' if args.embedding_control else 'off'})")
    summary = blend.build(args)
    print("\n=== build summary ===")
    short = {k: v for k, v in summary.items() if k not in ("shard_hashes", "render_examples", "topup_stream_doc_chars",
                                                            "input_hashes", "novelty_audit")}
    print(json.dumps(short, indent=2, default=str))
    if args.probes_only:
        print(f"\nWrote: {summary.get('probe_dir')} (probes only; shards, manifest and build_summary.json untouched)")
    else:
        print(f"\nWrote: {args.out}/arm_*/{args.data_subdir}/, manifest.csv, probe_sets/, build_summary.json, BUILD_COMPLETE.json")
        print("Next: validate.py --out <dir>  (then prepack / run_arm.py per arm; see README)")
    if not args.no_tracking:
        track_build(args, summary)


def track_build(args, summary):
    """Probe sets + manifest + build summary -> HF dataset repo; build config/summary -> a wandb run."""
    T.load_env()
    if not T.tracking_enabled():
        return
    tag = T.exp_tag(args.out)
    hf_url = None
    if T.hf_enabled():
        hf_url, sha = T.hf_upload_build(args.out)
        print(f"build metadata -> {hf_url} (commit {sha[:8]})")
    else:
        print("NOTE  HF_TOKEN not set; build metadata not uploaded (run load_artifact_hf.py --build later)")
    if T.wandb_enabled():
        run = T.wandb_init(f"ctx_build_{tag}", job_type="build", tags=(tag,),
                           config={**{k: v for k, v in vars(args).items()}, "exp_tag": tag, "hf_build": hf_url})
        flat = {k: v for k, v in summary.items() if isinstance(v, (int, float, str, bool))}
        run.summary.update(flat)
        run.finish()
        print(f"build logged -> wandb run {run.id} ({run.url})")
    else:
        print("NOTE  WANDB_API_KEY not set; build not logged to wandb")


if __name__ == "__main__":
    main()
