"""
Blend the injected slice into the FineWeb base and write the three arms.

Streaming lockstep design (memory-bounded, exactly matched across arms):
  - We pick M slot positions among the T train base documents (seeded). M = sum of the
    injected frequencies.
  - We read the train base ONCE. At a slot, all three arms DROP the same base document and
    write arm-specific content: R = raw rendering, X = contextualized rendering, C = a neutral
    held-out filler document. Everywhere else, all three arms write the IDENTICAL base document.
  - So the three corpora are byte-identical except in the M replaced slots -- the only place the
    experiment varies. R and X carry identical (fact_id -> frequency) maps by construction.
  - Budget parity (<0.5%): the X arm is longest; we top up R and C with extra neutral filler
    documents until all three totals are within 0.5%. This is the matched-token control.
  - The clean val shard (held out from injection) is written last and is identical for all arms.
"""

import csv
import json
import math
import os

from . import basecorpus, factpool, templates
from .nanochat_io import ShardWriter, DEFAULT_DATA_SUBDIR, shard_filename
from .rng import rng_for


ARMS = ["C", "R", "X"]


class FillerPool:
    """Lazily yields held-out neutral documents from successive filler shard indices
    (disjoint from train/val), materializing shards on demand."""

    def __init__(self, source, start_index, cache_dir, **shard_kwargs):
        self.source = source
        self.index = start_index
        self.cache_dir = cache_dir
        self.kw = shard_kwargs
        self._buf = []

    def _refill(self):
        path = basecorpus.ensure_shard(self.source, self.index, self.cache_dir, **self.kw)
        self.index += 1
        self._buf.extend(basecorpus.iter_shard_docs(path))

    def take(self, n):
        out = []
        while len(out) < n:
            if not self._buf:
                self._refill()
            out.append(self._buf.pop(0))
        return out

    def one(self):
        if not self._buf:
            self._refill()
        return self._buf.pop(0)


def build(cfg) -> dict:
    """cfg: argparse.Namespace from build_dataset.py. Returns a summary dict."""
    os.makedirs(cfg.out, exist_ok=True)
    cache_dir = cfg.cache_dir or os.path.join(cfg.out, "_base_cache")
    shard_kw = dict(seed=cfg.seed, chars_per_shard=cfg.chars_per_shard,
                    synthetic_docs_per_shard=cfg.synthetic_docs_per_shard,
                    row_group_size=cfg.row_group_size)

    # --- 1. fact pool: build, split, assign frequencies ---
    facts = factpool.build_fact_pool(cfg.num_facts, cfg.seed, contested_frac=cfg.contested_frac)
    injected, heldout = factpool.split_pool(facts, cfg.heldout_frac, cfg.seed)
    factpool.assign_frequencies(injected, cfg.freq_grid, cfg.seed)

    # --- 2. base shard layout ---
    n_train_shards = max(1, math.ceil(cfg.base_chars / cfg.chars_per_shard))
    train_indices = list(range(n_train_shards))
    val_index = n_train_shards
    filler_start = n_train_shards + 1

    # materialize train shards + count docs (T) cheaply from metadata
    train_paths = [basecorpus.ensure_shard(cfg.base_source, i, cache_dir, **shard_kw)
                   for i in train_indices]
    T = basecorpus.count_docs(train_paths)

    # --- 3. injected occurrences (the M slots) ---
    occurrences = []  # (fact, occ_index)
    for f in injected:
        for occ in range(f.assigned_frequency):
            occurrences.append((f, occ))
    M = len(occurrences)
    if M == 0:
        raise SystemExit("No injected occurrences; increase --num-facts or --freq-grid.")
    if M > T:
        raise SystemExit(
            f"Need {M} injection slots but the train base only has {T} documents. "
            f"Increase --base-chars (more base shards) or reduce --num-facts / --freq-grid."
        )

    # --- 4. shared clean val shard + held-out filler pool (carriers + parity top-up) ---
    val_path = basecorpus.ensure_shard(cfg.base_source, val_index, cache_dir, **shard_kw)
    val_docs = list(basecorpus.iter_shard_docs(val_path))
    filler = FillerPool(cfg.base_source, filler_start, cache_dir, **shard_kw)

    # --- 5. slot selection + shuffle assignment (seeded) ---
    rng = rng_for(cfg.seed, "slots")
    slot_positions = sorted(rng.choice(T, size=M, replace=False).tolist())
    order = rng.permutation(M).tolist()  # which occurrence fills each slot (in slot order)
    slot_to_occ = {pos: order[i] for i, pos in enumerate(slot_positions)}

    # --- 6. stream train base once; at each slot pull a held-out carrier and insert the
    #        matched (neutral / raw / contextualized) sentence at the SAME position. ---
    writers = {a: ShardWriter(os.path.join(cfg.out, f"arm_{a}", cfg.data_subdir),
                              chars_per_shard=cfg.chars_per_shard, row_group_size=cfg.row_group_size)
               for a in ARMS}
    manifest_rows = []
    injected_chars_R = 0
    j = 0
    for path in train_paths:
        for base_doc in basecorpus.iter_shard_docs(path):
            if j in slot_to_occ:
                f, occ = occurrences[slot_to_occ[j]]
                carrier = filler.one() if cfg.embed else None
                c_doc, r_doc, x_doc = templates.render_occurrence(
                    f, occ, cfg.seed, carrier,
                    embed=cfg.embed, verbatim=cfg.verbatim_control, register=cfg.register)
                writers["C"].add(c_doc)
                sR = writers["R"].add_returning(r_doc)
                sX = writers["X"].add_returning(x_doc)
                injected_chars_R += len(r_doc) - (len(carrier) if carrier else 0)
                for arm, sidx in (("R", sR), ("X", sX)):
                    manifest_rows.append({
                        "fact_id": f.fact_id, "truth_value": f.truth_value, "domain": f.domain,
                        "entity_tier": f.entity_tier, "arm": arm,
                        "assigned_frequency": f.assigned_frequency, "rendering_id": f"{f.fact_id}:{occ}",
                        "embedded": bool(cfg.embed), "heldout": False,
                        "shard_file": shard_filename(sidx),
                    })
            else:
                writers["R"].add(base_doc)
                writers["X"].add(base_doc)
                writers["C"].add(base_doc)
            j += 1

    # --- 8. budget parity: top up shorter arms with neutral filler (<0.5%) ---
    target = max(w.total_chars for w in writers.values())
    tol = cfg.parity_tol
    topup_counts = {}
    for a, w in writers.items():
        added = 0
        while w.total_chars < target * (1 - tol):
            w.add(filler.one())
            added += 1
        topup_counts[a] = added

    # --- 9. finalize: shared clean val shard as the LAST shard of every arm ---
    for w in writers.values():
        w.finalize(val_docs)

    # --- 10. manifest + probe files ---
    _write_manifest(cfg, manifest_rows)
    probe_summary = _write_probes(cfg, injected, heldout)

    # --- 11. optional shared-tokenizer copy ---
    tok_copied = False
    if cfg.tokenizer_src:
        from .nanochat_io import copy_tokenizer
        for a in ARMS:
            copy_tokenizer(cfg.tokenizer_src, os.path.join(cfg.out, f"arm_{a}"))
        tok_copied = True

    totals = {a: w.total_chars for a, w in writers.items()}
    spread = (max(totals.values()) - min(totals.values())) / max(totals.values())
    injected_chars = injected_chars_R  # claim-sentence chars added in Arm R (excludes carriers)
    summary = {
        "out": cfg.out, "base_source": cfg.base_source,
        "n_facts": len(facts), "n_injected_facts": len(injected), "n_heldout_facts": len(heldout),
        "n_occurrences_M": M, "n_train_base_docs_T": T, "n_train_shards": n_train_shards,
        "arm_total_chars": totals, "char_spread_frac": spread,
        "realized_inject_frac_R": injected_chars / max(1, totals["R"]),
        "topup_docs": topup_counts, "shards_per_arm": {a: len(w.shard_paths) for a, w in writers.items()},
        "tokenizer_copied": tok_copied, "probes": probe_summary,
        "tier_counts": _counts(facts, "entity_tier"), "tv_counts": _counts(facts, "truth_value"),
        # config echoed so validate.py can deterministically re-derive renderings
        "config": {k: v for k, v in vars(cfg).items()
                   if v is None or isinstance(v, (int, float, str, bool, list))},
    }
    with open(os.path.join(cfg.out, "build_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def _counts(facts, attr):
    out = {}
    for f in facts:
        out[getattr(f, attr)] = out.get(getattr(f, attr), 0) + 1
    return out


def _write_manifest(cfg, rows):
    path = os.path.join(cfg.out, "manifest.csv")
    cols = ["fact_id", "truth_value", "domain", "entity_tier", "arm",
            "assigned_frequency", "rendering_id", "embedded", "heldout", "shard_file"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_probes(cfg, injected, heldout):
    pdir = os.path.join(cfg.out, "probe_sets")
    os.makedirs(pdir, exist_ok=True)

    # held-out facts: never injected
    held_path = os.path.join(pdir, "heldout_facts.jsonl")
    n_cf = 0
    with open(held_path, "w") as fh:
        for f in heldout:
            fh.write(json.dumps({
                "fact_id": f.fact_id, "claim_text": f.claim_text, "truth_value": f.truth_value,
                "domain": f.domain, "entity_tier": f.entity_tier, "relation": f.relation,
                "subject": f.subject, "value": f.value,
                "cloze_templates": f.cloze_templates, "surface_forms": f.surface_forms,
                "source": "synthetic",
            }) + "\n")
        # CounterFact as bonus held-out probes (never injected)
        for rec in factpool.load_counterfact_probes(limit=cfg.counterfact_limit):
            n_cf += 1
            fh.write(json.dumps(rec) + "\n")

    inj_path = os.path.join(pdir, "injected_facts.jsonl")
    with open(inj_path, "w") as fh:
        for f in injected:
            fh.write(json.dumps({
                "fact_id": f.fact_id, "claim_text": f.claim_text, "truth_value": f.truth_value,
                "domain": f.domain, "entity_tier": f.entity_tier, "relation": f.relation,
                "subject": f.subject, "value": f.value,
                "assigned_frequency": f.assigned_frequency,
                "arms": ["R", "X"],  # injected identically (by frequency) into both R and X
                "surface_forms": f.surface_forms,
                "cloze_templates": f.cloze_templates,
                "heldout_paraphrase": f.heldout_paraphrase,  # reserved out of training for generalization tests
            }) + "\n")
    return {"heldout_synthetic": len(heldout), "heldout_counterfact": n_cf, "injected": len(injected)}
