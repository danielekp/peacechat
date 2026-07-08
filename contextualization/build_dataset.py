#!/usr/bin/env python
"""
build_dataset.py -- build the three-arm contextualization pretraining datasets for nanochat.

Produces, per arm, <out>/arm_{C,R,X}/<data-subdir>/shard_XXXXX.parquet in nanochat's exact
format, plus manifest.csv and probe_sets/*.jsonl. Everything is seeded: same args + same
--seed => byte-identical shards.

Quick offline smoke test (no network):
  python build_dataset.py --base-source synthetic --num-facts 300 --base-chars 2_000_000 \
      --chars-per-shard 200_000 --out /tmp/ctx_smoke --seed 0

Default d12-scale build on the real FineWeb base:
  python build_dataset.py --out ~/ctx_experiment --base-chars 3_000_000_000 --num-facts 4000 --seed 1234
"""

import argparse
import json

from ctx import blend
from ctx.nanochat_io import CHARS_PER_SHARD, ROW_GROUP_SIZE, DEFAULT_DATA_SUBDIR


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # output / reproducibility
    p.add_argument("--out", required=True, help="output dir (creates arm_C/ arm_R/ arm_X/)")
    p.add_argument("--seed", type=int, default=1234, help="master seed (controls ALL randomness)")
    p.add_argument("--cache-dir", default=None, help="where base shards are cached (default <out>/_base_cache)")
    p.add_argument("--probes-only", action="store_true",
                   help="rewrite probe_sets/*.jsonl only (no shards/manifest); pool args "
                        "(--num-facts/--seed/--heldout-frac/--contested-frac/--freq-grid) must "
                        "match the original build")

    # base corpus
    p.add_argument("--base-source", choices=["fineweb", "synthetic"], default="fineweb",
                   help="'fineweb' = stock karpathy/fineweb-edu-100b-shuffle (download on demand); "
                        "'synthetic' = offline deterministic prose (for smoke tests)")
    p.add_argument("--base-chars", type=int, default=3_000_000_000,
                   help="total base char budget. d12 quick ~3e9 (default); d20 full ~60e9")
    p.add_argument("--inject-frac", type=float, default=0.02,
                   help="advisory target fraction of chars that is injected material (reported, not forced)")

    # fact pool
    p.add_argument("--num-facts", type=int, default=4000,
                   help="target distinct facts (scales from a few thousand to tens of thousands)")
    p.add_argument("--heldout-frac", type=float, default=0.3, help="fraction of facts reserved as never-injected probes")
    p.add_argument("--contested-frac", type=float, default=0.08, help="fraction of facts that are 'contested'")
    p.add_argument("--freq-grid", type=int, nargs="+", default=[1, 4, 16, 64, 256],
                   help="injection-frequency grid (balanced across truth values)")
    p.add_argument("--counterfact-limit", type=int, default=2000,
                   help="max CounterFact records to add to the held-out probe set (0 disables)")

    # rendering
    p.add_argument("--embed", dest="embed", action="store_true", default=True,
                   help="embed renderings inside carrier paragraphs (default on)")
    p.add_argument("--no-embed", dest="embed", action="store_false", help="disable carrier embedding")
    p.add_argument("--register", dest="register", action="store_true", default=True,
                   help="vary surrounding register/context across repetitions (default on)")
    p.add_argument("--no-register", dest="register", action="store_false")
    p.add_argument("--verbatim-control", action="store_true",
                   help="repeat ONE fixed string per fact (string-repetition control) instead of "
                        "rotating paraphrases (proposition-repetition)")
    p.add_argument("--source-per-fact", action="store_true",
                   help="Arm X: attribute every occurrence of a fact to ONE consistent source. "
                        "Default rotates sources per occurrence, which conflates 'attributed' with "
                        "'many independent sources agree'. Only Arm X text changes under this flag.")
    p.add_argument("--embedding-control", action="store_true",
                   help="add a fourth arm E: the same claim (same fact->frequency map as R/X) in "
                        "the document's OWN VOICE, but inside a source-free embedding frame matched "
                        "to Arm X's wrappers in length/position/subordination. Separates "
                        "'attribution semantics' (X) from mere 'syntactic embedding/dilution' (E). "
                        "C/R/X shards are byte-identical with or without this flag.")

    # tokenizer
    p.add_argument("--tokenizer-src", default=None,
                   help="optional nanochat base dir holding a trained tokenizer/ to copy into each arm")

    # nanochat format knobs (defaults match the repo; override only for tiny smoke tests)
    p.add_argument("--chars-per-shard", type=int, default=CHARS_PER_SHARD)
    p.add_argument("--row-group-size", type=int, default=ROW_GROUP_SIZE)
    p.add_argument("--data-subdir", default=DEFAULT_DATA_SUBDIR,
                   help="per-arm shard dir name (default 'base_data'; loads via nanochat's fallback). "
                        "Use 'base_data_climbmix' to silence the legacy-upgrade warning.")
    p.add_argument("--synthetic-docs-per-shard", type=int, default=2000,
                   help="docs per synthetic shard (only for --base-source synthetic)")
    p.add_argument("--parity-tol", type=float, default=0.0049, help="max char spread across arms (default <0.5%)")

    return p.parse_args()


def main():
    args = parse_args()
    print(f"Building 3 arms into {args.out} (base={args.base_source}, "
          f"base_chars={args.base_chars:,}, num_facts={args.num_facts}, seed={args.seed})")
    summary = blend.build(args)
    print("\n=== build summary ===")
    print(json.dumps(summary, indent=2))
    if args.probes_only:
        print(f"\nWrote: {args.out}/probe_sets/ (probes only; shards and manifest untouched)")
    else:
        print(f"\nWrote: {args.out}/arm_{{C,R,X}}/{args.data_subdir}/, manifest.csv, probe_sets/")
        print("Next: run validate.py, then point nanochat at each arm via NANOCHAT_BASE_DIR (see README).")


if __name__ == "__main__":
    main()
