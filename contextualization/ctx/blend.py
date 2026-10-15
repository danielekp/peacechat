"""
Blend the injected slice into the base corpus and write the arms (C, R, X[, E]).

Streaming lockstep design (memory-bounded, exactly matched across arms):
  - M slot positions are chosen among the T train base documents (seeded). M = sum of the
    injected frequencies.
  - The train base is read ONCE. At a slot, all arms DROP the same base document and write
    arm-specific content built from the SAME held-out carrier document with ONE sentence
    inserted at the SAME position: C = neutral sentence, R = raw claim, X = attributed claim,
    E = own-voice claim in an embedding frame. Everywhere else all arms write the identical
    base document, so the corpora are byte-identical outside the M slots.
  - Budget parity: arms are topped up with whole filler documents from one shared stream
    until every arm lies within `parity_tol` of the ceiling (= the longest of C/R/X). A doc is
    appended to an arm only if it fits under the ceiling, so the ceiling is never exceeded and
    adding the optional arm E never changes what C/R/X receive (builder v2; finding 23).
  - The clean val shard is written last and is identical for all arms.

Builder v2 operational guarantees (findings 21-23): output is staged under <out>/.staging and
moved into place only after every artifact was written and verified; an existing build is never
overwritten unless --overwrite is passed; BUILD_COMPLETE.json is the completion marker; the
manifest carries character spans for every inserted claim; every shard is hashed.

Protocols: v1 (legacy generator + rendering, for verified recreations of the historical design)
and v2 (corrected generator + rendering policies). See ctx/protocol.py and docs/PROTOCOL_V2.md.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import time
from pathlib import Path

from . import basecorpus, factpool, factpool_v2, templates
from .nanochat_io import ShardWriter, DEFAULT_DATA_SUBDIR, shard_filename
from .protocol import (PROTOCOL_V1, PROTOCOL_V2, BUILDER_VERSION, PROBE_SCHEMA, MANIFEST_SCHEMA,
                       ProtocolError, code_revision, sha256_file, sha256_text, tokenizer_fingerprint)
from .rng import rng_for
from .tokenizers import TokenizerUnavailable, load_encoder, token_limit_mode, vocab_size_of

ARMS = ["C", "R", "X"]
COMPLETE_MARKER = "BUILD_COMPLETE.json"
STAGING_DIR = ".staging"
# everything a build (or a later stage keyed to it) writes under --out; --overwrite removes ALL
# of them so no stale derived artifact can be mistaken for the new build's (finding 10)
BUILD_ARTIFACTS = ("manifest.csv", "manifest.csv.gz", "manifest.csv.sha256", "build_summary.json", COMPLETE_MARKER,
                   "novelty_audit.json", "xe_examples.txt", "xe_diagnostics.json", "validation_report.json",
                   "shard_hashes.json", "exposure", "export", "eval", "prepack_preflight.json")

MANIFEST_COLS = ["schema_version", "fact_id", "group_id", "arm", "occurrence", "assigned_frequency",
                 "doc_index", "shard_file", "row_in_shard", "insert_text", "insert_char_start", "insert_char_end",
                 "claim_char_start", "claim_char_end", "subject_char_start", "subject_char_end",
                 "value_char_start", "value_char_end", "value_spans", "source_char_start", "source_char_end",
                 "paraphrase_index", "wrapper_index", "register_index", "tail_index", "source_id",
                 "truth_value", "domain", "entity_tier", "rendering_id", "embedded", "heldout",
                 "carrier_chars", "carrier_cropped"]


class FillerPool:
    """Lazily yields held-out neutral documents from successive filler shard indices
    (disjoint from train/val), materializing shards on demand."""

    def __init__(self, source, start_index, cache_dir, **shard_kwargs):
        self.source = source
        self.index = start_index
        self.cache_dir = cache_dir
        self.kw = shard_kwargs
        self._buf = []
        self.shards_used = []

    def _refill(self):
        path = basecorpus.ensure_shard(self.source, self.index, self.cache_dir, **self.kw)
        self.shards_used.append(self.index)
        self.index += 1
        self._buf.extend(basecorpus.iter_shard_docs(path))

    def one(self):
        while not self._buf:
            self._refill()
        return self._buf.pop(0)


class NeutralSentencePool:
    """Sentences from held-out filler documents for Arm C's slot (protocol v2 default)."""

    def __init__(self, filler: FillerPool, min_len: int = 40, max_len: int = 200):
        self.filler, self.min_len, self.max_len = filler, min_len, max_len
        self._buf = []

    def one(self) -> str:
        while not self._buf:
            doc = self.filler.one()
            for s in templates._split_sentences(doc):
                if self.min_len <= len(s) <= self.max_len and "{" not in s and "}" not in s:
                    self._buf.append(s)
        return self._buf.pop(0)


def crop_carrier(carrier: str, max_chars: int | None):
    """Keep whole leading sentences up to max_chars. Returns (text, cropped); text is None when
    even the first sentence exceeds the budget (the builder then draws another carrier so a
    slot document always fits a training row)."""
    if not max_chars or len(carrier) <= max_chars:
        return carrier, False
    sents = templates._split_sentences(carrier)
    kept, n = [], 0
    for s in sents:
        if n + len(s) + 1 > max_chars:
            break
        kept.append(s)
        n += len(s) + 1
    if not kept:
        return None, True
    return " ".join(kept), True


def _cfg(cfg, name, default=None):
    return getattr(cfg, name, default)


def resolve_protocol(cfg) -> str:
    p = _cfg(cfg, "protocol", None)
    if p is None:
        print("NOTE  no --protocol given: interpreting as the legacy protocol v1 (historical generator/rendering)")
        return PROTOCOL_V1
    if p in ("v1", "legacy", PROTOCOL_V1):
        return PROTOCOL_V1
    if p in ("v2", "corrected", PROTOCOL_V2):
        return PROTOCOL_V2
    raise ProtocolError(f"unknown protocol {p!r}")


def resolve_policy(cfg, protocol: str) -> templates.RenderPolicy:
    if protocol == PROTOCOL_V1:
        pol = templates.RenderPolicy.legacy()
        if _cfg(cfg, "source_per_fact", False):
            pol.source = "per_fact"
        return pol
    rp = _cfg(cfg, "render_policy", None)
    if isinstance(rp, templates.RenderPolicy):
        return rp
    if isinstance(rp, dict):
        return templates.RenderPolicy.from_dict(rp)
    mode = _cfg(cfg, "render_mode", "variable")
    kw = {}
    for k in ("paraphrase", "wrapper", "source", "stance", "register", "tails", "neutral", "x_templates", "e_templates"):
        v = _cfg(cfg, f"policy_{k}", None)
        if v is not None:
            kw[k] = v
    if _cfg(cfg, "source_per_fact", False):
        kw["source"] = "per_fact"
    if mode == "fixed":
        # fixed defaults, then explicit --policy-* overrides (review 2026-10-14 finding 13)
        return templates.RenderPolicy.fixed_rendering(**kw)
    return templates.RenderPolicy(**kw)


def _explicit_policy_overrides(cfg) -> dict:
    """The rendering controls the caller set explicitly on the command line (for --probes-only)."""
    out = {}
    for k in ("paraphrase", "wrapper", "source", "stance", "register", "tails", "neutral", "x_templates", "e_templates"):
        v = _cfg(cfg, f"policy_{k}", None)
        if v is not None:
            out[k] = v
    if _cfg(cfg, "source_per_fact", False):
        out["source"] = "per_fact"
    if _cfg(cfg, "render_mode", "variable") == "fixed":
        out["render_mode"] = "fixed"
    return out


# ---------------------------------------------------------------------------------------------

def build_pool(cfg, protocol: str):
    """-> dict with 'facts' (statement objects), 'injected', 'heldout_records', 'injected_records',
    'counts', 'design' depending on the protocol."""
    seed = int(cfg.seed)
    if protocol == PROTOCOL_V1:
        facts = factpool.build_fact_pool(cfg.num_facts, seed, contested_frac=cfg.contested_frac)
        injected, heldout = factpool.split_pool(facts, cfg.heldout_frac, seed)
        factpool.assign_frequencies(injected, cfg.freq_grid, seed)
        return {"protocol": protocol, "facts": facts, "injected": injected, "heldout": heldout,
                "counts": {"facts": len(facts), "injected_statements": len(injected), "heldout_statements": len(heldout),
                           "occurrences": sum(f.assigned_frequency for f in injected)},
                "design": "legacy_independent_split"}
    num_groups = _cfg(cfg, "num_groups", None) or (cfg.num_facts // 2)
    groups = factpool_v2.build_semantic_pool(num_groups, seed, include_nonfunctional=_cfg(cfg, "include_nonfunctional", False),
                                             relations=_cfg(cfg, "relations", None), use_real=not _cfg(cfg, "no_real", False))
    train, held = factpool_v2.split_groups(groups, cfg.heldout_frac, seed)
    design = _cfg(cfg, "design", "one_sided")
    factpool_v2.assign_injection(train, cfg.freq_grid, seed, design=design, rival_grid=_cfg(cfg, "rival_freq_grid", None))
    factpool_v2.check_pool_invariants(train, held, design=design)
    injected = factpool_v2.injected_statements(train)
    return {"protocol": protocol, "groups": groups, "train_groups": train, "heldout_groups": held,
            "injected": injected, "counts": factpool_v2.pool_counts(train, held).to_dict(), "design": design}


def _probe_records(pool, cfg, policy, arms, protocol):
    seed = int(cfg.seed)
    if protocol == PROTOCOL_V1:
        held, inj = [], []
        for f in pool["heldout"]:
            held.append({"fact_id": f.fact_id, "claim_text": f.claim_text, "truth_value": f.truth_value,
                         "domain": f.domain, "entity_tier": f.entity_tier, "relation": f.relation,
                         "subject": f.subject, "value": f.value, "competing_value": f.competing_value,
                         "cloze_templates": f.cloze_templates, "surface_forms": f.surface_forms,
                         "source": "synthetic", "protocol": protocol, "schema_version": 1,
                         "group_id": factpool_v2.stable_id(f.relation, f.subject), "pair_id": f.pair_id})
        for f in pool["injected"]:
            inj.append({"fact_id": f.fact_id, "claim_text": f.claim_text, "truth_value": f.truth_value,
                        "domain": f.domain, "entity_tier": f.entity_tier, "relation": f.relation,
                        "subject": f.subject, "value": f.value, "competing_value": f.competing_value,
                        "assigned_frequency": f.assigned_frequency, "arms": [a for a in arms if a != "C"],
                        "surface_forms": f.surface_forms, "cloze_templates": f.cloze_templates,
                        "heldout_paraphrase": f.heldout_paraphrase,
                        "fixed_source": (templates.fixed_source_for(seed, f.fact_id) if policy.source == "per_fact" else None),
                        "protocol": protocol, "schema_version": 1,
                        "group_id": factpool_v2.stable_id(f.relation, f.subject), "pair_id": f.pair_id})
        return inj, held
    inj, held = factpool_v2.to_probe_records(pool["train_groups"], pool["heldout_groups"], [a for a in arms if a != "C"])
    if policy.source == "per_fact":
        for r in inj:
            r["fixed_source"] = templates._make_source_v2(rng_for(seed, "render.v2.source.fact", r["fact_id"]))[0]
    for r in held:
        r["source"] = "synthetic"
    return inj, held


def _write_probe_files(pdir, inj, held, counterfact_limit, protocol):
    os.makedirs(pdir, exist_ok=True)
    n_cf = 0
    with open(os.path.join(pdir, "heldout_facts.jsonl"), "w") as fh:
        for r in held:
            fh.write(json.dumps(r) + "\n")
        for i, rec in enumerate(factpool.load_counterfact_probes(limit=counterfact_limit)):
            rec = dict(rec, fact_id=f"counterfact_{i:05d}", schema_version=PROBE_SCHEMA if protocol == PROTOCOL_V2 else 1)
            n_cf += 1
            fh.write(json.dumps(rec) + "\n")
    with open(os.path.join(pdir, "injected_facts.jsonl"), "w") as fh:
        for r in inj:
            fh.write(json.dumps(r) + "\n")
    return {"heldout_synthetic": len(held), "heldout_counterfact": n_cf, "injected": len(inj)}


def pool_signature(cfg, protocol) -> dict:
    keys = ["seed", "num_facts", "num_groups", "heldout_frac", "contested_frac", "freq_grid", "design",
            "include_nonfunctional", "relations", "no_real", "rival_freq_grid"]
    sig = {k: _cfg(cfg, k, None) for k in keys}
    sig["protocol"] = protocol
    return sig


# ---------------------------------------------------------------------------------------------

def build(cfg) -> dict:
    """cfg: argparse.Namespace from build_dataset.py (or a compatible object). Returns a summary."""
    protocol = resolve_protocol(cfg)
    policy = resolve_policy(cfg, protocol)
    out = Path(cfg.out)
    if _cfg(cfg, "probes_only", False):
        return probes_only(cfg, protocol, policy)

    # ---- 0. the matching tokenizer, BEFORE anything is downloaded or written (recheck 2026-10-14
    #         finding 7): an unusable --tokenizer-src stops the build here, never NOTE-and-continue ----
    emb_ctrl = bool(_cfg(cfg, "embedding_control", False))
    tokenizer_src = _cfg(cfg, "tokenizer_src", None)
    match_tok, match_tok_how = None, None
    if tokenizer_src:
        match_tok, match_tok_how = require_match_tokenizer(tokenizer_src)
        print(f"tokenizer: {tokenizer_src} loaded via {match_tok_how}; vocab {match_tok.get_vocab_size()}; "
              f"X/E token limit mode {token_limit_mode(match_tok.get_vocab_size())} "
              f"(limits {templates.XE_MATCH_LIMITS_VERSION})")
    elif emb_ctrl and protocol == PROTOCOL_V2:
        print("WARNING  no --tokenizer-src: X/E matching is character-based only and the token limits are NOT verified "
              "by this build; validate.py will FAIL check 8.token_length for a v2 build with arm E. Pass "
              "--tokenizer-src <nanochat base dir holding tokenizer/> for a build that can be validated.")

    # ---- refuse to clobber an existing build (finding 21) ----
    existing = [p.name for p in out.iterdir()] if out.exists() else []
    conflicts = [n for n in existing if n.startswith("arm_") or n.startswith("probe_sets") or n.startswith(STAGING_DIR)
                 or n in BUILD_ARTIFACTS]
    if conflicts:
        if not _cfg(cfg, "overwrite", False):
            raise SystemExit(f"refusing to build into {out}: it already holds {conflicts}. Use a fresh --out "
                             f"(builds are versioned by directory) or pass --overwrite to delete them explicitly.")
        for n in conflicts:
            p = out / n
            print(f"--overwrite: removing {p}")
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    out.mkdir(parents=True, exist_ok=True)
    stage = out / STAGING_DIR
    stage.mkdir()
    t_start = time.time()
    build_id = sha256_text(json.dumps({"config": {k: v for k, v in vars(cfg).items()
                                                  if v is None or isinstance(v, (int, float, str, bool, list))},
                                       "protocol": protocol, "policy": policy.to_dict(),
                                       "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid()}, sort_keys=True))

    cache_dir = cfg.cache_dir or str(out / "_base_cache")
    shard_kw = dict(seed=int(cfg.seed), chars_per_shard=cfg.chars_per_shard,
                    synthetic_docs_per_shard=cfg.synthetic_docs_per_shard, row_group_size=cfg.row_group_size)
    revision = _cfg(cfg, "fineweb_revision", None)
    resolved_commit = None
    if cfg.base_source == "fineweb":
        if revision:
            shard_kw["revision"] = revision
        if not (revision and re.fullmatch(r"[0-9a-f]{40}", revision)):
            # not pinned to an immutable commit: resolve the moving ref now and RECORD it
            resolved_commit = basecorpus.resolve_fineweb_revision(revision or basecorpus.DEFAULT_FINEWEB_REVISION)
            msg = (f"FineWeb source is not pinned to a commit (revision {revision or basecorpus.DEFAULT_FINEWEB_REVISION!r}"
                   f"{', resolved now to ' + resolved_commit if resolved_commit else ', could not be resolved offline'})")
            if _cfg(cfg, "require_pinned_source", False):
                raise SystemExit(msg + "; pass --fineweb-revision <40-hex commit> (required by --require-pinned-source)")
            print("WARNING  " + msg + "; pass --fineweb-revision <commit> for a reproducible build")
        if _cfg(cfg, "verify_cache", False):
            shard_kw["verify"] = True

    # --- 1. pool ---
    pool = build_pool(cfg, protocol)
    injected = pool["injected"]

    # --- 2. base shard layout ---
    n_train_shards = max(1, math.ceil(cfg.base_chars / cfg.chars_per_shard))
    train_indices = list(range(n_train_shards))
    val_index = n_train_shards
    filler_start = n_train_shards + 1
    train_paths = [basecorpus.ensure_shard(cfg.base_source, i, cache_dir, **shard_kw) for i in train_indices]
    T = basecorpus.count_docs(train_paths)

    # --- 3. occurrences ---
    occurrences = [(f, occ) for f in injected for occ in range(f.assigned_frequency)]
    M = len(occurrences)
    if M == 0:
        raise SystemExit("No injected occurrences; increase --num-facts/--num-groups or --freq-grid.")
    if M > T:
        raise SystemExit(f"Need {M} injection slots but the train base only has {T} documents. "
                         f"Increase --base-chars or reduce the pool / frequencies.")

    # --- 4. val + filler ---
    val_path = basecorpus.ensure_shard(cfg.base_source, val_index, cache_dir, **shard_kw)
    val_docs = list(basecorpus.iter_shard_docs(val_path))
    filler = FillerPool(cfg.base_source, filler_start, cache_dir, **shard_kw)
    neutral_pool = NeutralSentencePool(filler) if policy.neutral == "filler_sentence" else None

    # --- 5. slots ---
    rng = rng_for(int(cfg.seed), "slots")
    slot_positions = sorted(rng.choice(T, size=M, replace=False).tolist())
    order = rng.permutation(M).tolist()
    slot_to_occ = {pos: order[i] for i, pos in enumerate(slot_positions)}

    # --- 6. stream ---
    arms = ARMS + (["E"] if emb_ctrl else [])
    writers = {a: ShardWriter(str(stage / f"arm_{a}" / cfg.data_subdir), chars_per_shard=cfg.chars_per_shard,
                              row_group_size=cfg.row_group_size) for a in arms}
    manifest_path = stage / "manifest.csv"
    mfh = open(manifest_path, "w", newline="")
    mw = csv.DictWriter(mfh, fieldnames=MANIFEST_COLS)
    mw.writeheader()
    carrier_max = _cfg(cfg, "carrier_max_chars", None) if protocol == PROTOCOL_V2 else None
    carriers_skipped = 0
    neutral_by_fact = {}
    fixed_all = policy.paraphrase == "fixed" and policy.wrapper == "fixed" and policy.source == "per_fact"
    # the training tokenizer (loaded and validated at the start of build()) makes X/E matching
    # token-aware (finding 15); rendering uses it where it matters, the v2 arm E filler choice
    render_tok = match_tok if (emb_ctrl and protocol == PROTOCOL_V2) else None
    injected_chars = {a: 0 for a in arms}
    xe_diag = []
    examples = []
    j = 0
    embed = bool(_cfg(cfg, "embed", True))
    verbatim = bool(_cfg(cfg, "verbatim_control", False))
    register = bool(_cfg(cfg, "register", True))
    for path in train_paths:
        for base_doc in basecorpus.iter_shard_docs(path):
            if j in slot_to_occ:
                f, occ = occurrences[slot_to_occ[j]]
                carrier, cropped = None, False
                if embed:
                    for _try in range(1000):
                        carrier, cropped = crop_carrier(filler.one(), carrier_max)
                        if carrier is not None:
                            break
                        carriers_skipped += 1
                    else:
                        raise SystemExit(f"no filler document has a first sentence within --carrier-max-chars {carrier_max}")
                if neutral_pool is None:
                    neutral_sentence = None
                elif fixed_all:
                    # fixed rendering: ONE neutral sentence per fact in every occurrence (finding 12)
                    neutral_sentence = neutral_by_fact.setdefault(f.fact_id, neutral_pool.one())
                else:
                    neutral_sentence = neutral_pool.one()
                inserts = templates.render_inserts(f, occ, int(cfg.seed), policy, verbatim=verbatim, register=register,
                                                   source_per_fact=policy.source == "per_fact", arms=tuple(arms),
                                                   neutral_sentence=neutral_sentence, tokenizer=render_tok)
                if emb_ctrl and protocol == PROTOCOL_V2:
                    xe_diag.append(templates.xe_diagnostics(inserts, render_tok))
                if len(examples) < 30:
                    examples.append({a: inserts[a].text for a in arms})
                pos = rng_for(int(cfg.seed), "embed", f.fact_id, 0 if verbatim else occ).random()
                for a in arms:
                    ins = inserts[a]
                    if carrier:
                        doc, off = templates.embed_in_carrier_at(carrier, ins.text, pos)
                    else:
                        doc, off = ins.text, 0
                    sidx, row = writers[a].add_indexed(doc)
                    injected_chars[a] += len(ins.text)
                    cl = ins.first("claim") or (-1, -1)
                    su = ins.first("subject") or (-1, -1)
                    va = ins.first("value") or (-1, -1)
                    so = ins.first("source") or (-1, -1)
                    sh = lambda sp: (sp[0] + off, sp[1] + off) if sp[0] >= 0 else (-1, -1)
                    mw.writerow({
                        "schema_version": MANIFEST_SCHEMA, "fact_id": f.fact_id,
                        "group_id": getattr(f, "group_id", None) or factpool_v2.stable_id(f.relation, f.subject),
                        "arm": a, "occurrence": occ, "assigned_frequency": f.assigned_frequency,
                        "doc_index": j, "shard_file": shard_filename(sidx), "row_in_shard": row,
                        "insert_text": ins.text, "insert_char_start": off, "insert_char_end": off + len(ins.text),
                        "claim_char_start": sh(cl)[0], "claim_char_end": sh(cl)[1],
                        "subject_char_start": sh(su)[0], "subject_char_end": sh(su)[1],
                        "value_char_start": sh(va)[0], "value_char_end": sh(va)[1],
                        "value_spans": json.dumps([list(sh(sp)) for sp in ins.spans.get("value", [])]),
                        "source_char_start": sh(so)[0], "source_char_end": sh(so)[1],
                        "paraphrase_index": ins.paraphrase_index, "wrapper_index": ins.wrapper_index,
                        "register_index": ins.register_index, "tail_index": ins.tail_index, "source_id": ins.source_id,
                        "truth_value": f.truth_value, "domain": f.domain, "entity_tier": f.entity_tier,
                        "rendering_id": f"{f.fact_id}:{occ}", "embedded": bool(carrier), "heldout": False,
                        "carrier_chars": len(carrier) if carrier else 0, "carrier_cropped": cropped,
                    })
            else:
                for w in writers.values():
                    w.add(base_doc)
            j += 1
    mfh.close()

    # --- 7. parity top-up (builder v2 rule; see module docstring) ---
    tol = cfg.parity_tol
    base_arms = [a for a in arms if a != "E"]
    ceiling = max(writers[a].total_chars for a in base_arms)
    if "E" in arms and writers["E"].total_chars > ceiling * (1 + tol):
        raise SystemExit(f"arm E is {writers['E'].total_chars} chars, more than {tol*100:.2f}% above the C/R/X ceiling "
                         f"{ceiling}: E frames are too long for this build; shorten E_FRAMES_V2 fillers or raise --parity-tol")
    topup_counts = {a: 0 for a in arms}
    topup_docs = []
    stalls = 0
    needs = lambda a: writers[a].total_chars < ceiling * (1 - tol)
    while any(needs(a) for a in arms):
        doc = filler.one()
        took = []
        for a in arms:
            if needs(a) and writers[a].total_chars + len(doc) <= ceiling:
                writers[a].add(doc)
                topup_counts[a] += 1
                took.append(a)
        topup_docs.append({"chars": len(doc), "arms": took})
        stalls = 0 if took else stalls + 1
        if stalls > 200:
            gaps = {a: ceiling - writers[a].total_chars for a in arms if needs(a)}
            raise SystemExit(f"parity top-up infeasible: no filler document fits the remaining gaps {gaps} "
                             f"(ceiling {ceiling}, tol {tol}); raise --parity-tol or use shorter filler")

    # --- 8. finalize ---
    shard_hashes = {a: writers[a].finalize(val_docs) for a in arms}

    # --- 9. optional novelty audit over everything the model sees in Arm C (base + carriers +
    #        top-up + val), then probes (finding 17: synthetic names are not assumed absent) ---
    novelty_audit = None
    if _cfg(cfg, "collision_audit", False):
        if protocol == PROTOCOL_V2:
            def _all_c_docs():
                for p in writers["C"].shard_paths:
                    yield from basecorpus.iter_shard_docs(p)
            hits = factpool_v2.audit_novelty(pool["groups"], _all_c_docs())
            novelty_audit = {"build_id": build_id, "scanned_shards": len(writers["C"].shard_paths), "hits": hits,
                             "groups_present": sum(g.novelty == "audited_present" for g in pool["groups"]),
                             "groups_absent": sum(g.novelty == "audited_absent" for g in pool["groups"])}
            with open(stage / "novelty_audit.json", "w") as fh:
                json.dump(novelty_audit, fh, indent=2)
        else:
            print("NOTE  --collision-audit is only implemented for protocol v2 pools; skipped")
    inj_rec, held_rec = _probe_records(pool, cfg, policy, arms, protocol)
    probe_summary = _write_probe_files(str(stage / "probe_sets"), inj_rec, held_rec, cfg.counterfact_limit, protocol)

    # --- 10. tokenizer copy ---
    tok_fp = None
    if tokenizer_src:
        from .nanochat_io import copy_tokenizer
        for a in arms:
            copy_tokenizer(tokenizer_src, str(stage / f"arm_{a}"))
        tok_fp = tokenizer_fingerprint(str(stage / "arm_C"))

    # --- 11. summary ---
    totals = {a: writers[a].total_chars for a in arms}
    spread = (max(totals.values()) - min(totals.values())) / max(totals.values())
    config = {k: v for k, v in vars(cfg).items() if v is None or isinstance(v, (int, float, str, bool, list))}
    config["protocol"] = protocol
    config["render_policy"] = policy.to_dict()
    input_hashes = {"base_shards": _base_shard_checksums(cfg.base_source, cache_dir,
                                                         train_indices + [val_index] + filler.shards_used, shard_kw)}
    input_hashes["fineweb_revision"] = revision
    input_hashes["fineweb_revision_pinned"] = bool(revision and re.fullmatch(r"[0-9a-f]{40}", revision))
    input_hashes["fineweb_resolved_commit"] = resolved_commit
    input_hashes["tokenizer"] = tok_fp
    summary = {
        "out": str(out), "protocol": protocol, "builder_version": BUILDER_VERSION, "build_id": build_id,
        "schema_versions": {"probe": PROBE_SCHEMA if protocol == PROTOCOL_V2 else 1, "manifest": MANIFEST_SCHEMA},
        "code": code_revision(), "base_source": cfg.base_source, "design": pool["design"],
        "counts": pool["counts"],
        "n_facts": pool["counts"].get("facts", pool["counts"].get("candidate_statements")),
        "n_injected_facts": len(injected),
        "n_heldout_facts": pool["counts"].get("heldout_statements", len(held_rec)),
        "n_occurrences_M": M, "n_train_base_docs_T": T, "n_train_shards": n_train_shards,
        "arm_total_chars": totals, "char_spread_frac": spread, "parity_ceiling": ceiling,
        "realized_inject_frac_R": injected_chars["R"] / max(1, totals["R"]),
        "injected_chars": injected_chars,
        "topup_docs": topup_counts, "topup_stream_doc_chars": topup_docs, "carriers_skipped_too_long": carriers_skipped,
        "shards_per_arm": {a: len(writers[a].shard_paths) for a in arms},
        "shard_hashes": shard_hashes, "input_hashes": input_hashes,
        "tokenizer_copied": tok_fp is not None, "probes": probe_summary,
        "match_tokenizer": ({"src": str(tokenizer_src), "loaded_via": match_tok_how, "vocab": match_tok.get_vocab_size(),
                             "token_limit_mode": token_limit_mode(match_tok.get_vocab_size())} if match_tok is not None else None),
        "render_examples": examples, "novelty_audit": novelty_audit,
        "config": config, "build_seconds": round(time.time() - t_start, 1),
    }
    if protocol == PROTOCOL_V1:
        summary["tier_counts"] = _counts(pool["facts"], "entity_tier")
        summary["tv_counts"] = _counts(pool["facts"], "truth_value")
    else:
        groups = pool["groups"]
        summary["tier_counts"] = {"real": sum(g.entity_tier == "real" for g in groups),
                                  "novel": sum(g.entity_tier == "novel" for g in groups)}
        summary["novelty"] = _counts(groups, "novelty")
        # planned cell counts of the injection design, INCLUDING empty cells (recheck 2026-10-14, gap 1)
        summary["design_cells"] = factpool_v2.design_cells(pool["train_groups"], cfg.freq_grid)
    if xe_diag:
        summary["xe_diagnostics"] = _xe_summary(xe_diag, vocab_size_of(render_tok))
    with open(stage / "build_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    if emb_ctrl and protocol == PROTOCOL_V2:
        with open(stage / "xe_examples.txt", "w") as fh:
            for ex in examples:
                fh.write("[X] " + ex["X"] + "\n[E] " + ex["E"] + "\n\n")

    # --- 12. move staged output into place, then the completion marker ---
    for entry in sorted(os.listdir(stage)):
        os.replace(stage / entry, out / entry)
    stage.rmdir()
    marker = {"completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "protocol": protocol,
              "build_summary_sha256": sha256_file(out / "build_summary.json"),
              "manifest_sha256": sha256_file(out / "manifest.csv"),
              "probe_hashes": {n: sha256_file(out / "probe_sets" / n) for n in ("injected_facts.jsonl", "heldout_facts.jsonl")}}
    with open(out / COMPLETE_MARKER, "w") as fh:
        json.dump(marker, fh, indent=2)
    return summary


def _base_shard_checksums(source, cache_dir, indices, shard_kw) -> dict | None:
    """{shard filename: checksum record} for the base shards used, from the cache's checksums.json."""
    try:
        key = basecorpus.cache_key(source, **shard_kw)
        sums = basecorpus.load_checksums(cache_dir, key)
    except Exception as e:  # pragma: no cover - older basecorpus without checksum support
        return {"note": f"checksums unavailable: {type(e).__name__}: {e}"}
    out = {"cache_key": key}
    for i in indices:
        name = shard_filename(i)
        rec = sums.get(name)
        out[name] = rec if rec else None
    return out


def _xe_summary(diag: list, vocab: int | None = None) -> dict:
    """Aggregate X/E diagnostics against templates.XE_MATCH_LIMITS (frozen choice, version
    templates.XE_MATCH_LIMITS_VERSION recorded as 'limits_version'). The token limit MODE comes from
    ctx.tokenizers.token_limit_mode: absolute for experiment-scale vocabularies and whenever the
    vocabulary is unknown, relative (fractions of the X token length) only for a KNOWN toy
    vocabulary. The claim-start token delta is enforced too (absolute mode: max
    |delta_claim_tok_start| <= max_abs_delta_claim_tok_start; relative mode: per occurrence
    |delta_claim_tok_start| / len_tokens_X <= max_abs_delta_tokens_frac), exactly as validate.py
    section 8 applies them."""
    import statistics
    dc = [abs(d["delta_chars"]) for d in diag]
    ds = [abs(d["delta_claim_start"]) for d in diag]
    lim = templates.XE_MATCH_LIMITS
    out = {"n": len(diag), "mean_abs_delta_chars": statistics.mean(dc), "max_abs_delta_chars": max(dc),
           "mean_abs_delta_claim_start": statistics.mean(ds), "max_abs_delta_claim_start": max(ds),
           "n_E_with_source_cue": sum(d["E_has_source_cue"] for d in diag), "limits": lim,
           "limits_version": templates.XE_MATCH_LIMITS_VERSION}
    out["within_char_limits"] = (out["max_abs_delta_chars"] <= lim["max_abs_delta_chars"]
                                 and out["mean_abs_delta_chars"] <= lim["max_mean_abs_delta_chars"]
                                 and out["max_abs_delta_claim_start"] <= lim["max_abs_delta_claim_start"]
                                 and out["n_E_with_source_cue"] == 0)
    tok_diag = [d for d in diag if "delta_tokens" in d]
    if tok_diag:
        dt = [abs(d["delta_tokens"]) for d in tok_diag]
        dts = [abs(d.get("delta_claim_tok_start", 0)) for d in tok_diag]
        lx = [max(1, d["len_tokens_X"]) for d in tok_diag]
        out["mean_abs_delta_tokens"], out["max_abs_delta_tokens"] = statistics.mean(dt), max(dt)
        out["max_abs_delta_claim_tok_start"] = max(dts)
        out["mean_abs_delta_claim_tok_start"] = statistics.mean(dts)
        out["tokenizer_vocab"] = vocab
        if token_limit_mode(vocab) == "absolute":
            out["token_limit_mode"] = "absolute"
            out["within_token_limits"] = (out["max_abs_delta_tokens"] <= lim["max_abs_delta_tokens"]
                                          and out["mean_abs_delta_tokens"] <= lim["max_mean_abs_delta_tokens"]
                                          and out["max_abs_delta_claim_tok_start"] <= lim["max_abs_delta_claim_tok_start"])
        else:
            fr = [a / b for a, b in zip(dt, lx)]
            frs = [a / b for a, b in zip(dts, lx)]
            out["token_limit_mode"] = "relative (toy tokenizer)"
            out["max_abs_delta_tokens_frac"], out["mean_abs_delta_tokens_frac"] = max(fr), statistics.mean(fr)
            out["max_abs_delta_claim_tok_start_frac"] = max(frs)
            out["within_token_limits"] = (out["max_abs_delta_tokens_frac"] <= lim["max_abs_delta_tokens_frac"]
                                          and out["mean_abs_delta_tokens_frac"] <= lim["max_mean_abs_delta_tokens_frac"]
                                          and out["max_abs_delta_claim_tok_start_frac"] <= lim["max_abs_delta_tokens_frac"])
        out["token_limits_verified"] = True
    else:
        out["within_token_limits"] = None
        out["token_limits_verified"] = False
    # "within_limits" is only true when BOTH families were checked (character-only diagnostics
    # never imply token matching; review 2026-10-14 finding 15)
    out["within_limits"] = bool(out["within_char_limits"] and out["within_token_limits"])
    return out


def load_match_tokenizer(tokenizer_src):
    """The tokenizer used for X/E token matching (`encode`, `decode`, `get_vocab_size` with the TRUE
    vocabulary) from <tokenizer_src>/tokenizer/tokenizer.pkl (or <tokenizer_src>/tokenizer.pkl),
    through the loader shared with validate.py (ctx.tokenizers.load_encoder). Never returns None:
    a tokenizer that cannot be loaded here raises ctx.tokenizers.TokenizerUnavailable with the
    reason and the install hint (recheck 2026-10-14 finding 7)."""
    tok, _how = load_encoder(tokenizer_src)
    return tok


def require_match_tokenizer(tokenizer_src):
    """(tokenizer, how) for --tokenizer-src, or SystemExit. Called at the START of build(), before
    the base corpus is downloaded and before staging exists, so an environment that cannot load the
    tokenizer (no tiktoken), a missing file or a corrupt pickle stops the build immediately instead
    of silently producing character-matched-only artifacts. Also checks the layout
    nanochat_io.copy_tokenizer needs (<src>/tokenizer/), so the copy step cannot fail hours later."""
    src = Path(tokenizer_src)
    if not (src / "tokenizer").is_dir():
        raise SystemExit(f"--tokenizer-src {src}: no tokenizer/ directory; expected a nanochat base dir holding "
                         f"tokenizer/tokenizer.pkl (+ token_bytes.pt), see README 'One shared tokenizer'")
    try:
        tok, how = load_encoder(src)
    except TokenizerUnavailable as e:
        raise SystemExit(f"--tokenizer-src {src} is unusable: {e.reason}. {e.hint}. "
                         f"Refusing to build with X/E token matching disabled.")
    return tok, how


def _counts(items, attr):
    out = {}
    for f in items:
        out[getattr(f, attr)] = out.get(getattr(f, attr), 0) + 1
    return out


# ---------------------------------------------------------------------------------------------

def probes_only(cfg, protocol, policy) -> dict:
    """Re-emit probe_sets/*.jsonl after VERIFYING that the regenerated pool agrees with the
    original build (same pool signature in build_summary.json, same fact_id->frequency map as
    manifest.csv). Never touches shards, manifest or build_summary.json. Existing probe files are
    only replaced with --overwrite-probes; otherwise the new files go to probe_sets_regen/."""
    out = Path(cfg.out)
    summary_path = out / "build_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"--probes-only: {summary_path} not found; probes can only be regenerated for a completed build")
    summary = json.loads(summary_path.read_text())
    orig_cfg = summary.get("config", {})
    orig_protocol = summary.get("protocol") or PROTOCOL_V1
    if orig_protocol != protocol:
        raise SystemExit(f"--probes-only: build is protocol {orig_protocol}, request is {protocol}")
    sig = pool_signature(cfg, protocol)
    mismatches = {k: (orig_cfg.get(k), v) for k, v in sig.items()
                  if k in orig_cfg and orig_cfg.get(k) != v and v is not None}
    if mismatches:
        raise SystemExit(f"--probes-only: pool arguments differ from the original build: {mismatches}")
    # the rendering policy is the ORIGINAL build's (finding 9): overrides that disagree are refused
    recorded = orig_cfg.get("render_policy")
    if protocol == PROTOCOL_V2:
        if not recorded:
            raise SystemExit("--probes-only: build_summary.json records no render_policy; cannot regenerate probes faithfully")
        policy = templates.RenderPolicy.from_dict(recorded)
        overrides = _explicit_policy_overrides(cfg)
        conflicts = {k: (recorded.get(k), v) for k, v in overrides.items() if k != "render_mode" and recorded.get(k) != v}
        if overrides.get("render_mode") == "fixed" and recorded.get("paraphrase") != "fixed":
            conflicts["render_mode"] = ("variable", "fixed")
        if conflicts:
            raise SystemExit(f"--probes-only: rendering overrides differ from the original build's policy: {conflicts}")
    else:
        if bool(_cfg(cfg, "source_per_fact", False)) != bool(orig_cfg.get("source_per_fact", False)):
            raise SystemExit("--probes-only: --source-per-fact differs from the original build")
        policy = templates.RenderPolicy.legacy()
        if orig_cfg.get("source_per_fact"):
            policy.source = "per_fact"
    for k in ("embedding_control", "verbatim_control", "register", "embed"):
        if k in orig_cfg and _cfg(cfg, k, orig_cfg[k]) != orig_cfg[k]:
            raise SystemExit(f"--probes-only: --{k.replace('_', '-')} differs from the original build ({orig_cfg[k]})")
    pool = build_pool(cfg, protocol)
    # novelty labels come from the original build's audit, and only if that audit belongs to THIS
    # build (findings 10 / validator defect 1; recheck 2026-10-14 finding 8): the evidence is
    # novelty_audit.json when present (it must agree with the audit embedded in build_summary.json),
    # otherwise the embedded audit itself; without usable evidence regeneration is REFUSED rather
    # than silently relabelling audited groups 'unverified'. A summary without an audit ignores any
    # stray file.
    audit_path = out / "novelty_audit.json"
    novelty_evidence = None
    if protocol == PROTOCOL_V2 and summary.get("novelty_audit"):
        hits, novelty_evidence = _novelty_audit_evidence(summary, audit_path)
        for g in pool["groups"]:
            if g.entity_tier == "novel":
                present = hits.get(g.subject, 0) > 0 or any(hits.get(c.value, 0) > 0 for c in g.candidates)
                g.novelty = "audited_present" if present else "audited_absent"
    elif protocol == PROTOCOL_V2 and audit_path.exists():
        print("NOTE  --probes-only: ignoring novelty_audit.json (the build summary records no audit)")
    manifest = out / "manifest.csv"
    if manifest.exists():
        freq = {}
        with open(manifest) as fh:
            for row in csv.DictReader(fh):
                if row["arm"] == "R":
                    freq[row["fact_id"]] = int(row["assigned_frequency"])
        regen = {f.fact_id: f.assigned_frequency for f in pool["injected"]}
        if freq != regen:
            missing = set(freq) ^ set(regen)
            raise SystemExit(f"--probes-only: regenerated pool does not match manifest.csv "
                             f"({len(missing)} ids differ; first: {sorted(missing)[:5]})")
    arms = ARMS + (["E"] if orig_cfg.get("embedding_control") else [])
    inj_rec, held_rec = _probe_records(pool, cfg, policy, arms, protocol)
    # independent verification against the build's ORIGINAL probe files (recheck 2026-10-14 finding 8):
    # when probe_sets/*.jsonl still hash to what BUILD_COMPLETE.json recorded, every pool-level field of
    # the regenerated records must agree with them; otherwise regeneration is refused and nothing is written
    originals = _original_probe_files(out)
    verified_against_probe_files = None
    if originals:
        diffs, n_compared = _compare_probe_records(originals, {"injected_facts.jsonl": inj_rec, "heldout_facts.jsonl": held_rec})
        if diffs:
            shown = "; ".join(f"({fid}, {field}, {old!r}, {new!r})" for fid, field, old, new in diffs[:5])
            raise SystemExit(f"--probes-only: regenerated records differ from the build's original probe files "
                             f"({', '.join(sorted(originals))}) on {len(diffs)} (record, field) pair(s): first {shown}. "
                             f"verified_against_probe_files=False; nothing was written")
        # True only when BOTH probe files were the build's originals and agree; a partial comparison
        # (one file absent or no longer the marker's) is recorded as None, with the files compared
        verified_against_probe_files = True if len(originals) == len(PROBE_FILES) else None
        print(f"probes-only: regenerated records agree with the build's original {sorted(originals)} on every "
              f"pool-level field ({n_compared} records; verified_against_probe_files={verified_against_probe_files}"
              f"{'' if verified_against_probe_files else ', partial: not every probe file is an original'})")
    pdir = out / "probe_sets"
    if pdir.exists() and not _cfg(cfg, "overwrite_probes", False):
        pdir = out / "probe_sets_regen"
        print(f"NOTE  probe_sets/ exists; writing regenerated probes to {pdir} (pass --overwrite-probes to replace)")
    tmp = Path(str(pdir) + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    ps = _write_probe_files(str(tmp), inj_rec, held_rec, cfg.counterfact_limit, protocol)
    with open(tmp / "PROBES_META.json", "w") as fh:
        json.dump({"regenerated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "protocol": protocol, "code": code_revision(),
                   "pool_signature": sig, "verified_against_manifest": manifest.exists(),
                   "verified_against_probe_files": verified_against_probe_files,
                   "probe_files_compared": sorted(originals), "novelty_evidence": novelty_evidence,
                   "hashes": {n: sha256_file(tmp / n) for n in ("injected_facts.jsonl", "heldout_facts.jsonl")}}, fh, indent=2)
    if pdir.exists():
        shutil.rmtree(pdir)
    os.replace(tmp, pdir)
    print(f"probes-only: wrote {pdir} (shards, manifest, build_summary.json untouched)")
    return {"probes_only": True, "probes": ps, "probe_dir": str(pdir), "verified_against_manifest": manifest.exists(),
            "verified_against_probe_files": verified_against_probe_files, "probe_files_compared": sorted(originals),
            "novelty_evidence": novelty_evidence, "n_injected_facts": len(inj_rec), "n_heldout_facts": len(held_rec)}


# the build's probe files, and the pool-level fields of a probe record (docs/SCHEMAS_V2.md) that a
# regeneration must reproduce exactly
PROBE_FILES = ("injected_facts.jsonl", "heldout_facts.jsonl")
PROBE_POOL_FIELDS = ("fact_id", "group_id", "role", "relation", "value", "competing_value", "assigned_frequency",
                     "truth_value", "entity_tier", "novelty", "ground_truth")


def _novelty_audit_evidence(summary: dict, audit_path: Path):
    """(hits, evidence) for --probes-only when build_summary.json records a novelty audit.
    novelty_audit.json, when present, must carry the summary's build_id and agree with the embedded
    audit (same build_id, same hits); without the file the embedded audit is the evidence (NOTE).
    New builds (summary with build_id) refuse an audit without that build_id; legacy summaries
    without build_id keep the lenient behaviour, with a NOTE. No usable hits -> SystemExit."""
    embedded = summary["novelty_audit"]
    want = summary.get("build_id")
    file_audit = None
    if audit_path.exists():
        try:
            file_audit = json.loads(audit_path.read_text())
        except ValueError as e:
            raise SystemExit(f"--probes-only: {audit_path} is not valid JSON ({e}); regeneration refused")
        if not isinstance(file_audit, dict):
            raise SystemExit(f"--probes-only: {audit_path} does not hold an audit object; regeneration refused")
    if want:
        if file_audit is not None and file_audit.get("build_id") != want:
            raise SystemExit(f"--probes-only: novelty_audit.json belongs to another build "
                             f"(build_id {str(file_audit.get('build_id'))[:12]} != {want[:12]})")
        if embedded.get("build_id") != want:
            raise SystemExit(f"--probes-only: the novelty audit embedded in build_summary.json carries build_id "
                             f"{str(embedded.get('build_id'))[:12]}, not this build's {want[:12]}; evidence refused")
    else:
        print("NOTE  --probes-only: legacy build_summary.json without build_id: the audit's build identity cannot be verified")
        fb, eb = (file_audit or {}).get("build_id"), embedded.get("build_id")
        if file_audit is not None and fb and eb and fb != eb:
            raise SystemExit(f"--probes-only: novelty_audit.json (build_id {str(fb)[:12]}) and the audit embedded in "
                             f"build_summary.json ({str(eb)[:12]}) belong to different builds")
    if file_audit is not None and isinstance(embedded.get("hits"), dict) and file_audit.get("hits") != embedded.get("hits"):
        raise SystemExit("--probes-only: novelty_audit.json disagrees with the audit embedded in build_summary.json "
                         "(hits differ); refusing to regenerate from contradictory evidence")
    source = file_audit if file_audit is not None else embedded
    hits = source.get("hits")
    if not isinstance(hits, dict):
        where = "novelty_audit.json is missing and the embedded audit" if file_audit is None else "novelty_audit.json"
        raise SystemExit(f"--probes-only: build_summary.json records a novelty audit but {where} has no usable hits; "
                         f"regeneration refused without the audit evidence (restore novelty_audit.json)")
    if file_audit is not None:
        return hits, "novelty_audit.json (agrees with the audit embedded in build_summary.json)"
    print(f"NOTE  --probes-only: novelty_audit.json missing; novelty labels restored from the audit embedded in "
          f"build_summary.json (build {want[:12] if want else 'legacy, no build_id'}, {len(hits)} hit strings)")
    return hits, "build_summary.json embedded novelty_audit (novelty_audit.json missing)"


def _original_probe_files(out: Path) -> dict:
    """{name: path} for the probe files under <out>/probe_sets that are the build's ORIGINALS, i.e.
    hash to BUILD_COMPLETE.json's probe_hashes. Present files that do not match are NOT used (NOTE)."""
    marker_path = out / COMPLETE_MARKER
    if not marker_path.exists():
        return {}
    try:
        want = (json.loads(marker_path.read_text()) or {}).get("probe_hashes") or {}
    except ValueError:
        print(f"NOTE  --probes-only: {marker_path} is not valid JSON; original probe files cannot be identified")
        return {}
    found = {}
    for n in PROBE_FILES:
        p = out / "probe_sets" / n
        if not p.exists():
            continue
        if want.get(n) and sha256_file(p) == want[n]:
            found[n] = p
        else:
            print(f"NOTE  --probes-only: probe_sets/{n} is not the build's original (sha256 differs from "
                  f"{COMPLETE_MARKER}); not used for verification")
    return found


def _compare_probe_records(originals: dict, regenerated: dict):
    """(diffs, n_compared): pool-level differences between the original probe files and the regenerated
    records, keyed by fact_id. Counterfact records (loaded at write time) are not pool records and are
    skipped. diffs = [(fact_id or file, field, old, new)]: the record set must be identical and every
    PROBE_POOL_FIELDS field present in an original record must be reproduced."""
    diffs, n = [], 0
    for name, path in originals.items():
        old = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        old = [r for r in old if not str(r.get("fact_id", "")).startswith("counterfact_")]
        new = regenerated.get(name) or []
        old_by = {r["fact_id"]: r for r in old}
        new_by = {r["fact_id"]: r for r in new}
        if len(old) != len(new) or set(old_by) != set(new_by):
            diffs.append((name, "record_set", f"{len(old)} records / {len(old_by)} ids", f"{len(new)} records / {len(new_by)} ids"))
        for fid in sorted(set(old_by) & set(new_by)):
            a, b = old_by[fid], new_by[fid]
            for f in PROBE_POOL_FIELDS:
                if f in a and a[f] != b.get(f):
                    diffs.append((fid, f, a[f], b.get(f)))
            n += 1
    return diffs, n
