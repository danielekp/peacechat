#!/usr/bin/env python
"""
validate.py -- check a built dataset against the experiment's invariants (brief §9).

Approach for the leakage / matched-rendering checks: the three arms are byte-identical
except in the injected slots, so we recover the injected documents EXACTLY by streaming
each arm's train docs in lockstep with Arm C and keeping the positions where they differ
(no fragile sentence-splitting). We then verify, against deterministically re-derived
renderings, that the raw/contextualized contrast is intact.

Checks:
  1. Schema:   every shard single-column "text", filenames contiguous shard_00000..N, >=2 shards.
  2. Parity:   total chars across arms within --parity-tol (default <0.5%); tokens/char note if
               the shared tokenizer is present (else char proxy).
  3. Leakage / matched rendering:
        - Arms R and X inject at the SAME positions with the SAME count (== M);
        - Arms R and X have identical (fact_id -> frequency) maps (manifest.csv);
        - every contextualized rendering is attributed (carries a wrapper signature);
        - no contextualized wrapper string appears in Arm R's injected docs;
        - every raw rendering is present in Arm R's injected docs (sampled);
        - no held-out fact's claim text appears in any injected doc.
  4. Print matched injected triples (same slot in C / R / X) -- the experimental contrast.

Usage:  python validate.py --out ~/ctx_experiment
"""

import argparse
import csv
import json
import os
import sys

import pyarrow.parquet as pq

from ctx.nanochat_io import read_shard_texts, shard_filename
from ctx import templates
from ctx.rng import rng_for


# wrapper "signatures": every attribution template contains either a stance verb from
# templates.VERBS or one of these connectives -- so presence of any signals an attribution.
WRAPPER_SIGNATURES = list(templates.VERBS) + [
    "According to", "From a", "Quoting", "put it this way", "attributed to",
]


def _shard_paths(arm_dir, data_subdir):
    d = os.path.join(arm_dir, data_subdir)
    files = sorted(f for f in os.listdir(d) if f.endswith(".parquet"))
    return [os.path.join(d, f) for f in files], files


def _iter_train_docs(paths, limit=None):
    """Yield docs from all train shards (all but the last == val), up to `limit`."""
    n = 0
    for p in paths[:-1]:
        for t in read_shard_texts(p):
            yield t
            n += 1
            if limit is not None and n >= limit:
                return


class Checker:
    def __init__(self):
        self.failures, self.passes = [], []

    def check(self, ok, msg):
        (self.passes if ok else self.failures).append(msg)
        print(("  PASS " if ok else "  FAIL ") + msg)
        return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-subdir", default=None)
    ap.add_argument("--sample-renderings", type=int, default=1500,
                    help="how many renderings to substring-check (0 = all)")
    args = ap.parse_args()

    summary = json.load(open(os.path.join(args.out, "build_summary.json")))
    cfg = summary.get("config", {})
    data_subdir = args.data_subdir or cfg.get("data_subdir", "base_data")
    seed = cfg.get("seed", 1234)
    parity_tol = cfg.get("parity_tol", 0.0049)
    verbatim = cfg.get("verbatim_control", False)
    register = cfg.get("register", True)
    T = summary["n_train_base_docs_T"]
    M = summary["n_occurrences_M"]

    chk = Checker()
    arms = ["C", "R", "X"]
    arm_files = {a: _shard_paths(os.path.join(args.out, f"arm_{a}"), data_subdir)[0] for a in arms}
    arm_names = {a: _shard_paths(os.path.join(args.out, f"arm_{a}"), data_subdir)[1] for a in arms}

    # ---- 1. schema + contiguity ----
    print("\n[1] schema & contiguity")
    for a in arms:
        files = arm_names[a]
        expected = [shard_filename(i) for i in range(len(files))]
        chk.check(files == expected, f"arm_{a}: contiguous filenames ({len(files)} shards)")
        chk.check(len(files) >= 2, f"arm_{a}: >=2 shards (train + val)")
        ok = all(pq.ParquetFile(p).schema_arrow.names == ["text"] for p in arm_files[a])
        chk.check(ok, f"arm_{a}: every shard single-column 'text'")

    # ---- 2. parity ----
    print("\n[2] budget parity")
    totals = summary["arm_total_chars"]
    spread = (max(totals.values()) - min(totals.values())) / max(totals.values())
    chk.check(spread <= parity_tol + 1e-9, f"char spread {spread*100:.3f}% <= {parity_tol*100:.2f}%  totals={totals}")
    _token_parity_note(args.out, arm_files)

    # ---- 3. recover injected docs (aligned C-vs-arm diff) and check each slot exactly ----
    # Manifest rows are written in slot (=position) order, so the k-th R row and the k-th
    # diffed injected doc correspond to the same slot. We reconstruct exactly what each arm
    # injected (templates.build_inserts -- the same code the build used) and substring-check
    # it against that single injected document. Robust to multi-sentence renderings and fast.
    print("\n[3] leakage & matched rendering")
    C_inj, R_inj, X_inj, R_pos, X_pos = _aligned_diff3(arm_files["C"], arm_files["R"], arm_files["X"], T)
    chk.check(len(R_inj) == M and len(X_inj) == M, f"R and X each have M={M} injected slots (got {len(R_inj)}, {len(X_inj)})")
    chk.check(R_pos == X_pos, f"R and X inject at identical positions ({len(R_pos)} slots)")

    freq_maps = _freq_maps_from_manifest(os.path.join(args.out, "manifest.csv"))
    chk.check(bool(freq_maps.get("R")) and freq_maps.get("R") == freq_maps.get("X"),
              f"R and X have identical (fact_id->frequency) maps ({len(freq_maps.get('R', {}))} facts)")

    # ordered (fact_id, occ) per slot, from the R rows of the manifest (written in slot order)
    fact_views = {rec["fact_id"]: _FactView(rec)
                  for rec in _load_jsonl(os.path.join(args.out, "probe_sets", "injected_facts.jsonl"))}
    slots = _manifest_slots(os.path.join(args.out, "manifest.csv"), "R")
    chk.check(len(slots) == M, f"manifest has M={M} R rows ({len(slots)})")

    raw_ok = ctx_ok = neu_ok = ctx_leak = att_ok = 0
    inserted_blob_parts = []
    for k, (fid, occ) in enumerate(slots):
        f = fact_views[fid]
        neutral, raw, ctx = templates.build_inserts(f, occ, seed, verbatim=verbatim, register=register)
        if raw in R_inj[k]: raw_ok += 1
        if ctx in X_inj[k]: ctx_ok += 1
        if neutral in C_inj[k]: neu_ok += 1
        if ctx in R_inj[k]: ctx_leak += 1
        if any(sig in X_inj[k] for sig in WRAPPER_SIGNATURES): att_ok += 1
        inserted_blob_parts.append(raw); inserted_blob_parts.append(ctx)
    chk.check(raw_ok == M, f"every slot's raw rendering is present in Arm R ({raw_ok}/{M})")
    chk.check(ctx_ok == M, f"every slot's contextualized rendering is present in Arm X ({ctx_ok}/{M})")
    chk.check(neu_ok == M, f"every slot's neutral filler is present in Arm C ({neu_ok}/{M})")
    chk.check(ctx_leak == 0, f"no contextualized rendering leaks into Arm R ({ctx_leak})")
    chk.check(att_ok == M, f"every Arm X injected doc carries an attribution wrapper ({att_ok}/{M})")

    # held-out leakage: no held-out claim string appears among the injected claim sentences.
    # (Carriers are real neutral FineWeb text; held-out synthetic claims cannot occur there.)
    held = _load_jsonl(os.path.join(args.out, "probe_sets", "heldout_facts.jsonl"))
    held_strs = set()
    for h in held:
        if h.get("source") == "synthetic":
            held_strs.add(h["claim_text"])
            held_strs.update(h.get("surface_forms", []))
    inserted_blob = "\n".join(inserted_blob_parts)
    leak = [s for s in held_strs if s and s in inserted_blob]
    chk.check(len(leak) == 0, f"no held-out claim string appears in injected claims (checked {len(held_strs)} strings)")

    # ---- 4. matched injected triples (the exact inserted sentence per arm) ----
    print("\n[4] matched injected examples -- same carrier, one inserted sentence:")
    for k in range(min(4, len(slots))):
        fid, occ = slots[k]
        f = fact_views[fid]
        neutral, raw, ctx = templates.build_inserts(f, occ, seed, verbatim=verbatim, register=register)
        print(f"\n  --- slot {k+1}  (truth={getattr(f,'truth_value','?')}, freq={freq_maps['R'].get(fid)}) ---")
        print(f"  [C] {neutral}")
        print(f"  [R] {raw}")
        print(f"  [X] {ctx}")

    print("\n=== RESULT ===")
    print(f"{len(chk.passes)} passed, {len(chk.failures)} failed")
    if chk.failures:
        for m in chk.failures:
            print("  FAILED: " + m)
        sys.exit(1)
    print("ALL CHECKS PASSED")


# --- helpers ---

class _FactView:
    def __init__(self, rec):
        self.fact_id = rec["fact_id"]
        self.surface_forms = rec["surface_forms"]
        self.heldout_paraphrase = rec.get("heldout_paraphrase")
        self.truth_value = rec.get("truth_value", "?")


def _aligned_diff3(c_paths, r_paths, x_paths, T):
    """Single pass over C/R/X train docs; collect the full injected docs + positions per slot
    (in ascending position order, matching the manifest's row order)."""
    ci, ri, xi = _iter_train_docs(c_paths, T), _iter_train_docs(r_paths, T), _iter_train_docs(x_paths, T)
    C_inj, R_inj, X_inj, R_pos, X_pos = [], [], [], [], []
    for i, (c, r, x) in enumerate(zip(ci, ri, xi)):
        if c != r:
            C_inj.append(c); R_inj.append(r); R_pos.append(i)
        if c != x:
            X_inj.append(x); X_pos.append(i)
    return C_inj, R_inj, X_inj, R_pos, X_pos


def _manifest_slots(path, arm):
    """(fact_id, occ) per slot, in manifest file order (== slot/position order), for one arm."""
    out = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if row["arm"] == arm:
                fid, occ = row["rendering_id"].rsplit(":", 1)
                out.append((fid, int(occ)))
    return out


def _freq_maps_from_manifest(path):
    maps = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            maps.setdefault(row["arm"], {})[row["fact_id"]] = int(row["assigned_frequency"])
    return maps


def _load_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def _token_parity_note(out, arm_files):
    tok = os.path.join(out, "arm_C", "tokenizer", "tokenizer.pkl")
    if not os.path.exists(tok):
        print("  NOTE  shared tokenizer not copied into arms yet; char parity is the token proxy")
        return
    try:
        import pickle
        enc = pickle.load(open(tok, "rb"))
        ratios = {}
        for a in ["C", "R", "X"]:
            docs = []
            for t in read_shard_texts(arm_files[a][0]):
                docs.append(t)
                if len(docs) >= 200:
                    break
            nch = sum(len(d) for d in docs)
            ntok = sum(len(enc.encode(d)) for d in docs)
            ratios[a] = round(ntok / max(1, nch), 4)
        print(f"  NOTE  tokens/char by arm (sample): {ratios}")
    except Exception as e:
        print(f"  NOTE  tokenizer present but unreadable here ({e}); char parity is the token proxy")


if __name__ == "__main__":
    main()
