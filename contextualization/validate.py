#!/usr/bin/env python
"""
validate.py -- independent checks of a built C/R/X[/E] dataset (builder v2; protocols v1 and v2).

Usage:  python validate.py --out <build dir> [--tokenizer-dir DIR] [--scan-heldout-real] [--report PATH]

Every check below is computed from the ARTIFACTS ONLY (shards, manifest.csv, probe_sets/*.jsonl,
build_summary.json, BUILD_COMPLETE.json, arm_*/tokenizer). Nothing is re-rendered through
ctx.templates.build_inserts / render_inserts and no pool is regenerated: a generator or renderer
that agrees with itself on wrong output is exactly what this file has to catch (review
2026-10-13, finding 24). The only things imported from the generator side are constants that
define the design (verb lists, names, institutions, source types, register lead-ins, X/E
matching limits, protocol ids).

Checks (each printed PASS/FAIL/NOTE with counts; exit status 1 on any FAIL; a JSON report is
written to <out>/validation_report.json):
   1. completion & provenance: BUILD_COMPLETE.json and its hashes, no .staging, every shard's
      sha256 vs build_summary, schema / contiguity / single "text" column / >=2 shards per arm.
   2. parity: per-arm totals recomputed from the shards (train + top-up, val excluded), spread,
      ceiling = max(C,R,X), E within tol of the ceiling, val shard byte-identical, top-up docs a
      subsequence of the shared top-up stream (LENGTH-based: the stream records only lengths;
      content is additionally compared across arms that took the same stream document).
   3. slot alignment: aligned C-vs-arm diff over the first T train docs; manifest positions.
   4. span integrity: every manifest span verified against the shard text (insert, claim,
      subject, value slots, source), attribution signature in X only, clean frame in E,
      raw claim exactly once in R and never un-attributed in X, shared carrier across arms,
      carrier cropping.
   5. coverage / uniqueness: occurrences == assigned_frequency, unique rendering ids, identical
      dose maps across claim arms, totals == M.
   6. pool invariants from the probe files only (v2: groups, reciprocity, uniqueness, labels,
      holdout, design, functional relations, novelty; v1: legacy collision statistics reported
      as NOTEs, exactly as finding 1 defines them).
   7. held-out leakage: no held-out claim / surface form in any insert (word-bounded; also the
      lower-cased clause variant); v2: no novel held-out subject in any insert (real-entity
      subject hits are reported as a NOTE); unasserted rivals never asserted.
   8. X/E matching (when E is present): char / claim-start / token deltas recomputed from the
      manifest spans and enforced against templates.XE_MATCH_LIMITS (v2; NOTE for legacy E).
   9. tokenizer identity across arms (and vs the summary's recorded fingerprint).
  10. counts: build_summary counts recomputed from the probe files and the manifest.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

from ctx.nanochat_io import read_shard_texts, shard_filename, shard_num_rows
from ctx.protocol import (PROTOCOL_V1, PROTOCOL_V2, ProtocolError, code_revision, protocol_of,
                          sha256_file, tokenizer_fingerprint)
from ctx import templates as _T

REPORT_SCHEMA = 1
COMPLETE_MARKER = "BUILD_COMPLETE.json"
STAGING_DIR = ".staging"

# ---- design constants (data, not renderer code) -------------------------------------------------
# manifest v2 columns (docs/SCHEMAS_V2.md + the two carrier columns the builder writes)
MANIFEST_COLS_V2 = [
    "schema_version", "fact_id", "group_id", "arm", "occurrence", "assigned_frequency",
    "doc_index", "shard_file", "row_in_shard", "insert_text", "insert_char_start", "insert_char_end",
    "claim_char_start", "claim_char_end", "subject_char_start", "subject_char_end",
    "value_char_start", "value_char_end", "value_spans", "source_char_start", "source_char_end",
    "paraphrase_index", "wrapper_index", "register_index", "tail_index", "source_id",
    "truth_value", "domain", "entity_tier", "rendering_id", "embedded", "heldout",
    "carrier_chars", "carrier_cropped",
]
# attribution signatures: a reporting verb (legacy or neutral set) or one of the wrapper
# connectives. "includes the statement that" is the v2 wrapper that carries no verb.
ATTRIBUTION_VERBS = frozenset(v.lower() for v in list(_T.VERBS) + list(_T.VERBS_V2))
ATTRIBUTION_CONNECTIVES = ["According to", "attributed to", "Quoting", "put it this way", "From a",
                           "includes the statement that"]
FIRST_NAMES = list(_T.FIRST_NAMES)
INSTITUTIONS = list(_T.INSTITUTIONS)
SOURCE_TYPES = [s.lower() for s in _T.SOURCE_TYPES]
REGISTER_LEADINS = list(_T.REGISTER_LEADINS)
XE_MATCH_LIMITS = dict(_T.XE_MATCH_LIMITS)
# words that may open a surface frame without being a proper noun (read off the frame
# templates in ctx/factpool.py); only these may be lower-cased when a claim becomes a clause
LOWERCASABLE_INITIALS = frozenset({"The", "A", "An", "In", "Because", "Visitors", "On", "At", "Among",
                                   "Credit", "Counting", "Biologists", "As", "Historians"})
NONFUNCTIONAL_RELATIONS = ("author_work", "inventor_of")
CLAIM_ARMS = ("R", "X", "E")
_WORD = re.compile(r"[A-Za-z]+")


# =================================================================================================
# report

class Report:
    """Collects PASS/FAIL/NOTE/SKIP results, prints them as they come, serializes to JSON."""

    def __init__(self):
        self.checks = []
        self.section = ""

    def start(self, section: str):
        self.section = section
        print(f"\n[{section}]")

    def _add(self, status, cid, msg, numbers=None, exclusions=None):
        self.checks.append({"id": cid, "section": self.section, "status": status, "message": msg,
                            "numbers": numbers or {}, "exclusions": exclusions or []})
        print(f"  {status:<4} {cid}: {msg}")
        return status == "PASS"

    def check(self, ok, cid, msg, **numbers):
        excl = numbers.pop("exclusions", None)
        return self._add("PASS" if ok else "FAIL", cid, msg, numbers, excl)

    def note(self, cid, msg, **numbers):
        excl = numbers.pop("exclusions", None)
        return self._add("NOTE", cid, msg, numbers, excl)

    def skip(self, cid, msg, **numbers):
        return self._add("SKIP", cid, msg, numbers)

    @property
    def failures(self):
        return [c for c in self.checks if c["status"] == "FAIL"]

    def counts(self):
        out = {}
        for c in self.checks:
            out[c["status"]] = out.get(c["status"], 0) + 1
        return out

    def to_json(self, **extra) -> dict:
        return {"schema_version": REPORT_SCHEMA, **extra, "checks": self.checks, "summary": self.counts(),
                "result": "FAIL" if self.failures else "PASS"}


# =================================================================================================
# small helpers

def _load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _lower_first(s: str) -> str:
    return s[0].lower() + s[1:] if s else s


def _clause_forms(sf: str) -> set:
    """Accepted realizations of a surface form inside a wrapper: verbatim (quote style) or as a
    that-clause (final period dropped, first word lower-cased only when it is a frame word)."""
    s = sf.rstrip()
    if s.endswith("."):
        s = s[:-1]
    first = s.split(" ", 1)[0]
    return {sf, s, _lower_first(s) if first in LOWERCASABLE_INITIALS else s}


def _realization(text: str, target: str):
    """'exact' | 'lowercased' | None for a span text against its target string."""
    if text == target:
        return "exact"
    if target and target[0].isupper() and text == _lower_first(target):
        return "lowercased"
    return None


def _bounded_find_all(hay: str, needle: str):
    """Start offsets of word-bounded (both sides non-alphanumeric) occurrences of needle."""
    out, start, n = [], 0, len(needle)
    if not needle:
        return out
    while True:
        i = hay.find(needle, start)
        if i < 0:
            return out
        j = i + n
        if (i == 0 or not hay[i - 1].isalnum()) and (j == len(hay) or not hay[j].isalnum()):
            out.append(i)
        start = i + 1


def _bounded_in(hay: str, needle: str) -> bool:
    return bool(_bounded_find_all(hay, needle))


def _count_value_occurrences(claim: str, value: str) -> int:
    """Occurrences of a value (or its lower-first variant) inside a claim that start at a word
    boundary and end at a word boundary or at a plural 's' (frames such as 'among the {value}s')."""
    variants = {value}
    if value[:1].isupper():
        variants.add(_lower_first(value))
    n = 0
    for v in variants:
        if not v:
            continue
        start = 0
        while True:
            i = claim.find(v, start)
            if i < 0:
                break
            j = i + len(v)
            left_ok = i == 0 or not claim[i - 1].isalnum()
            right_ok = (j == len(claim) or not claim[j].isalnum()
                        or (claim[j] == "s" and (j + 1 == len(claim) or not claim[j + 1].isalnum())))
            if left_ok and right_ok:
                n += 1
            start = i + 1
    return n


def _doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _shard_paths(arm_dir, data_subdir):
    d = os.path.join(arm_dir, data_subdir)
    if not os.path.isdir(d):
        return [], []
    files = sorted(f for f in os.listdir(d) if f.endswith(".parquet"))
    return [os.path.join(d, f) for f in files], files


def _load_tokenizer(tok_dir):
    """A tokenizer with .encode(str)->list[int] from <dir>/tokenizer.pkl, or None (+reason)."""
    d = Path(tok_dir)
    if (d / "tokenizer" / "tokenizer.pkl").exists():
        d = d / "tokenizer"
    if not (d / "tokenizer.pkl").exists():
        return None, f"no tokenizer.pkl under {tok_dir}"
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    try:
        from nanochat.tokenizer import RustBPETokenizer
        return RustBPETokenizer.from_directory(str(d)), "nanochat.tokenizer.RustBPETokenizer"
    except Exception as e:  # pragma: no cover - fallback path
        try:
            import pickle
            with open(d / "tokenizer.pkl", "rb") as fh:
                enc = pickle.load(fh)

            class _Enc:
                def encode(self, s):
                    return enc.encode_ordinary(s)
            return _Enc(), f"pickle fallback ({type(e).__name__}: {e})"
        except Exception as e2:
            return None, f"tokenizer unreadable ({type(e2).__name__}: {e2})"


def _to_int(x, default=-1):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


# =================================================================================================
# 1. completion & provenance

def check_completion(rep: Report, out: Path, summary: dict, arms, data_subdir):
    rep.start("1 completion & provenance")
    marker_path = out / COMPLETE_MARKER
    marker = None
    if rep.check(marker_path.exists(), "1.marker_present", f"{COMPLETE_MARKER} present"):
        try:
            marker = json.loads(marker_path.read_text())
        except ValueError as e:
            rep.check(False, "1.marker_json", f"{COMPLETE_MARKER} is not valid JSON: {e}")
    if marker:
        want = {"build_summary.json": marker.get("build_summary_sha256"), "manifest.csv": marker.get("manifest_sha256")}
        for n, h in (marker.get("probe_hashes") or {}).items():
            want[f"probe_sets/{n}"] = h
        bad = []
        for rel, h in want.items():
            p = out / rel
            actual = sha256_file(p) if p.exists() else None
            if actual != h:
                bad.append({"file": rel, "marker": h, "actual": actual})
        rep.check(not bad and len(want) >= 4, "1.marker_hashes",
                  f"marker hashes match build_summary.json / manifest.csv / probe files ({len(want) - len(bad)}/{len(want)})",
                  n_files=len(want), mismatches=bad)
        rep.check(marker.get("protocol") == summary.get("protocol", PROTOCOL_V1) or
                  (marker.get("protocol") is None and "protocol" not in summary),
                  "1.marker_protocol", f"marker protocol {marker.get('protocol')!r} == summary protocol")
    rep.check(not (out / STAGING_DIR).exists(), "1.no_staging", f"no {STAGING_DIR}/ left behind")
    rep.check(not list(out.rglob("*.tmp")), "1.no_tmp_files", "no *.tmp files left behind")

    # shards: hashes, schema, contiguity
    shard_hashes = summary.get("shard_hashes") or {}
    files = {}
    for a in arms:
        paths, names = _shard_paths(out / f"arm_{a}", data_subdir)
        files[a] = (paths, names)
        expected = [shard_filename(i) for i in range(len(names))]
        rep.check(names == expected and len(names) >= 2, f"1.shards_{a}",
                  f"arm_{a}: {len(names)} contiguous shards (>=2: train + val)", n_shards=len(names))
        ok_schema = all(pq.ParquetFile(p).schema_arrow.names == ["text"] for p in paths) if paths else False
        rep.check(ok_schema, f"1.schema_{a}", f"arm_{a}: every shard is a single-column 'text' parquet")
        rec = shard_hashes.get(a)
        if rec is None:
            rep.check(False, f"1.shard_hashes_{a}", f"arm_{a}: build_summary has no shard_hashes entry")
            continue
        mism = []
        for p, n in zip(paths, names):
            h = sha256_file(p)
            if rec.get(n) != h:
                mism.append({"file": n, "summary": rec.get(n), "actual": h})
        extra = sorted(set(rec) - set(names))
        rep.check(not mism and not extra, f"1.shard_hashes_{a}",
                  f"arm_{a}: sha256 of every shard on disk == build_summary.shard_hashes ({len(names) - len(mism)}/{len(names)}"
                  f"{', summary lists ' + str(len(extra)) + ' missing files' if extra else ''})",
                  n_shards=len(names), mismatches=mism, listed_but_missing=extra)
        spa = (summary.get("shards_per_arm") or {}).get(a)
        if spa is not None:
            rep.check(spa == len(names), f"1.shards_per_arm_{a}", f"arm_{a}: shards_per_arm {spa} == {len(names)} files")
    return files


# =================================================================================================
# probe files

class Probes:
    def __init__(self, out: Path, protocol: str):
        self.injected = _load_jsonl(out / "probe_sets" / "injected_facts.jsonl")
        held_all = _load_jsonl(out / "probe_sets" / "heldout_facts.jsonl")
        self.counterfact = [r for r in held_all if r.get("source") == "counterfact"]
        self.heldout = [r for r in held_all if r.get("source") != "counterfact"]
        self.by_id = {r["fact_id"]: r for r in self.injected}
        self.protocol = protocol


def check_pool_invariants(rep: Report, pr: Probes, summary: dict, out: Path, arms):
    rep.start("6 pool invariants (probe files only)")
    cfg = summary.get("config") or {}
    inj, held = pr.injected, pr.heldout
    allrec = inj + held
    rep.check(len(inj) > 0 and len(held) > 0, "6.files_present",
              f"injected_facts.jsonl ({len(inj)}) and heldout_facts.jsonl ({len(held)} synthetic + "
              f"{len(pr.counterfact)} counterfact) present")
    ids = [r["fact_id"] for r in allrec]
    dup = sorted({i for i in ids if ids.count(i) > 1}) if len(set(ids)) != len(ids) else []
    rep.check(not dup, "6.fact_id_unique", f"fact_id unique across injected + held-out ({len(set(ids))} ids)", duplicates=dup[:10])
    # structural: every surface form carries subject and value verbatim; cloze templates have a blank
    bad_forms = [r["fact_id"] for r in allrec
                 if not r.get("surface_forms") or len(set(r["surface_forms"])) != len(r["surface_forms"])
                 or any(r["subject"] not in s or str(r["value"]) not in s for s in r["surface_forms"])]
    rep.check(not bad_forms, "6.surface_forms", "every record has distinct surface forms that contain subject and value verbatim",
              n_bad=len(bad_forms), exclusions=bad_forms[:10])
    bad_cloze = [r["fact_id"] for r in allrec if not r.get("cloze_templates")
                 or any("___" not in c or r["subject"] not in c for c in r["cloze_templates"])]
    rep.check(not bad_cloze, "6.cloze_templates", "every record has cloze templates with a ___ blank and the subject",
              n_bad=len(bad_cloze), exclusions=bad_cloze[:10])
    bad_hp = [r["fact_id"] for r in inj if len(r.get("surface_forms", [])) > 1 and
              (not r.get("heldout_paraphrase") or r["heldout_paraphrase"] not in r["surface_forms"])]
    rep.check(not bad_hp, "6.heldout_paraphrase", "every injected record reserves a held-out paraphrase from its surface forms",
              n_bad=len(bad_hp), exclusions=bad_hp[:10])
    claim_arms = [a for a in arms if a != "C"]
    bad_arms = [r["fact_id"] for r in inj if list(r.get("arms") or []) != claim_arms]
    rep.check(not bad_arms, "6.record_arms", f"injected records list arms == {claim_arms}", n_bad=len(bad_arms), exclusions=bad_arms[:10])
    bad_freq = [r["fact_id"] for r in inj if _to_int(r.get("assigned_frequency"), 0) <= 0]
    rep.check(not bad_freq, "6.injected_frequency", "every injected record has assigned_frequency >= 1", n_bad=len(bad_freq))
    if cfg.get("source_per_fact") or (cfg.get("render_policy") or {}).get("source") == "per_fact":
        missing = [r["fact_id"] for r in inj if not r.get("fixed_source")]
        rep.check(not missing, "6.fixed_source", "per-fact source policy: every injected record has fixed_source", n_bad=len(missing))

    if pr.protocol == PROTOCOL_V2:
        _pool_invariants_v2(rep, pr, summary, out)
    else:
        _pool_legacy_stats(rep, pr)


def _pool_invariants_v2(rep: Report, pr: Probes, summary: dict, out: Path):
    cfg = summary.get("config") or {}
    inj, held = pr.injected, pr.heldout
    allrec = inj + held
    design = summary.get("design") or cfg.get("design") or "one_sided"
    bad_schema = [r["fact_id"] for r in allrec if r.get("schema_version") != 2 or r.get("protocol") != PROTOCOL_V2]
    rep.check(not bad_schema, "6.v2_schema", "every synthetic record has schema_version 2 / protocol v2", n_bad=len(bad_schema))
    groups = {}
    for r in allrec:
        groups.setdefault(r.get("group_id"), []).append(r)
    bad_size = {g: len(v) for g, v in groups.items() if len(v) != 2}
    rep.check(not bad_size, "6.group_two_candidates", f"every group_id has exactly 2 candidates ({len(groups)} groups)",
              n_groups=len(groups), bad=dict(list(bad_size.items())[:10]))
    recip, consist, labels, tiers = [], [], [], []
    for g, cands in groups.items():
        if len(cands) != 2:
            continue
        a, b = cands
        if not (a["rival_fact_id"] == b["fact_id"] and b["rival_fact_id"] == a["fact_id"]
                and a["competing_value"] == b["value"] and b["competing_value"] == a["value"]
                and a["value"] != b["value"] and {a.get("candidate_slot"), b.get("candidate_slot")} == {"a", "b"}
                and a.get("pair_id") == g and b.get("pair_id") == g):
            recip.append(g)
        for k in ("relation", "subject", "domain", "entity_tier", "ground_truth", "split", "functional_relation", "subject_kind"):
            if a.get(k) != b.get(k):
                consist.append(g)
                break
        gt, tier = a.get("ground_truth"), a.get("entity_tier")
        tv = {a.get("truth_value"), b.get("truth_value")}
        if gt == "unverified_synthetic":
            if tv != {"unknown"} or tier != "novel":
                labels.append(g)
        elif gt == "real_world":
            if tv != {"true", "false"} or tier != "real":
                labels.append(g)
        else:
            labels.append(g)
        if a.get("legacy_truth_label") is not None or b.get("legacy_truth_label") is not None:
            labels.append(g)
    rep.check(not recip, "6.reciprocal_rivals", "rival_fact_id / competing_value / candidate_slot reciprocal within every group",
              n_bad=len(recip), exclusions=recip[:10])
    rep.check(not consist, "6.group_consistency", "both candidates of a group share relation/subject/tier/ground_truth/split",
              n_bad=len(consist), exclusions=consist[:10])
    rep.check(not labels, "6.truth_labels", "synthetic groups: truth_value unknown + ground_truth unverified_synthetic; "
              "real groups: exactly one true and one false", n_bad=len(labels), exclusions=labels[:10])
    subjects = [cands[0]["subject"].lower() for cands in groups.values()]
    dups = sorted({s for s in subjects if subjects.count(s) > 1}) if len(set(subjects)) != len(subjects) else []
    rep.check(not dups, "6.subjects_unique", f"subjects unique (case-insensitive) across the pool ({len(set(subjects))} subjects)",
              duplicates=dups[:10])
    inj_ids = {r["fact_id"] for r in inj}
    bad_held, bad_train, bad_roles = [], [], []
    for g, cands in groups.items():
        split = cands[0].get("split")
        if split == "heldout_group":
            if any(c["fact_id"] in inj_ids or c.get("role") != "heldout_group" or _to_int(c.get("assigned_frequency"), 0) != 0
                   or c.get("arms") for c in cands):
                bad_held.append(g)
        elif split == "train_group":
            n_inj = sum(c["fact_id"] in inj_ids for c in cands)
            roles = sorted(c.get("role") for c in cands)
            if design == "one_sided":
                if n_inj != 1 or roles != ["injected", "unasserted_rival"]:
                    bad_train.append(g)
                else:
                    i = [c for c in cands if c["fact_id"] in inj_ids][0]
                    r = [c for c in cands if c["fact_id"] not in inj_ids][0]
                    if (_to_int(r.get("assigned_frequency"), -1) != 0 or r.get("arms")
                            or _to_int(r.get("rival_assigned_frequency"), -1) != _to_int(i.get("assigned_frequency"), -2)
                            or _to_int(i.get("rival_assigned_frequency"), -1) != 0 or i.get("role") != "injected"):
                        bad_train.append(g)
            else:
                if n_inj < 1 or any((c.get("role") == "injected") != (c["fact_id"] in inj_ids) for c in cands):
                    bad_train.append(g)
        else:
            bad_roles.append(g)
    rep.check(not bad_held, "6.heldout_groups_clean", "held-out groups have no injected member (role heldout_group, dose 0, no arms)",
              n_bad=len(bad_held), exclusions=bad_held[:10])
    rep.check(not bad_train, "6.training_groups_design",
              f"design {design}: " + ("exactly one injected candidate per training group, rival unasserted (dose 0, "
                                       "rival_assigned_frequency == injected dose)" if design == "one_sided" else
                                       "roles agree with the injected file"),
              n_bad=len(bad_train), exclusions=bad_train[:10])
    rep.check(not bad_roles, "6.split_labels", "every group is train_group or heldout_group", n_bad=len(bad_roles))
    bad_func = [g for g, c in groups.items() if c[0].get("functional_relation") is not True
                or c[0].get("relation") in NONFUNCTIONAL_RELATIONS]
    if cfg.get("include_nonfunctional"):
        rep.note("6.functional_relations", f"include_nonfunctional set: {len(bad_func)} non-functional groups present")
    else:
        rep.check(not bad_func, "6.functional_relations", "every group uses a functional relation (functional_relation true)",
                  n_bad=len(bad_func), exclusions=bad_func[:10])
    audit_path = out / "novelty_audit.json"
    audited = audit_path.exists()
    bad_nov = []
    n_abs = n_pres = 0
    for g, c in groups.items():
        nov, tier = c[0].get("novelty"), c[0].get("entity_tier")
        if tier == "real":
            if nov != "real_entity":
                bad_nov.append(g)
        elif audited:
            if nov not in ("audited_absent", "audited_present"):
                bad_nov.append(g)
            n_abs += nov == "audited_absent"
            n_pres += nov == "audited_present"
        elif nov != "unverified":
            bad_nov.append(g)
    msg = ("novelty labels consistent with novelty_audit.json present (novel groups audited_absent/present)" if audited
           else "novelty labels consistent with NO novelty_audit.json (novel groups 'unverified', real 'real_entity')")
    rep.check(not bad_nov, "6.novelty_labels", msg, n_bad=len(bad_nov), exclusions=bad_nov[:10], audited=audited)
    if audited:
        try:
            audit = json.loads(audit_path.read_text())
            rep.check(audit.get("groups_absent") == n_abs and audit.get("groups_present") == n_pres, "6.novelty_audit_counts",
                      f"novelty_audit.json counts (absent {audit.get('groups_absent')}, present {audit.get('groups_present')}) == "
                      f"records (absent {n_abs}, present {n_pres})")
        except ValueError:
            rep.check(False, "6.novelty_audit_counts", "novelty_audit.json is not valid JSON")
        if summary.get("novelty_audit") is None:
            rep.check(False, "6.novelty_audit_summary", "novelty_audit.json exists but build_summary.novelty_audit is null")
    return groups


def _pool_legacy_stats(rep: Report, pr: Probes):
    """Protocol v1: the legacy pool is known to violate semantic uniqueness (finding 1). Report
    the statistics exactly as the review defines them so the build is labelled with its defects;
    they are NOTEs, not failures."""
    inj, held = pr.injected, pr.heldout
    allrec = inj + held
    groups = {}
    for f in allrec:
        groups.setdefault((f["relation"], f["subject"]), []).append(f)
    injected_values = {}
    for f in inj:
        injected_values.setdefault((f["relation"], f["subject"]), set()).add(f["value"])
    colliding = [g for g in groups.values() if len(g) > 2]
    multi_true = [g for g in groups.values() if len({f["value"] for f in g if f.get("truth_value") == "true"}) > 1]
    broken = [f["fact_id"] for f in allrec if f.get("truth_value") in ("true", "false") and not any(
        g.get("truth_value") in ("true", "false") and g["truth_value"] != f["truth_value"] and g["value"] == f.get("competing_value")
        for g in groups[(f["relation"], f["subject"])])]
    held_shared = sum(bool(injected_values.get((f["relation"], f["subject"]))) for f in held)
    clean = [f for f in inj if f.get("competing_value") not in injected_values[(f["relation"], f["subject"])]]
    strict = [f for f in clean if len(injected_values[(f["relation"], f["subject"])]) == 1]
    stats = {
        "relation_subject_groups": len(groups),
        "groups_with_more_than_two_facts": len(colliding),
        "facts_in_groups_with_more_than_two_facts": sum(map(len, colliding)),
        "groups_multiple_true_values": len(multi_true),
        "functional_groups_multiple_true_values": sum(g[0]["relation"] not in NONFUNCTIONAL_RELATIONS for g in multi_true),
        "missing_opposing_label_rival_links": len(broken),
        "heldout_with_injected_subject_relation": held_shared,
        "nominal_clean": len(clean), "strict_own_value_only": len(strict),
        "contested": sum(f.get("truth_value") == "contested" for f in allrec),
    }
    rep.note("6.legacy_collisions", "protocol v1 pool (legacy generator, finding 1): "
             f"{stats['groups_with_more_than_two_facts']} (relation,subject) groups with >2 facts, "
             f"{stats['groups_multiple_true_values']} groups with multiple 'true' values, "
             f"{stats['missing_opposing_label_rival_links']} facts without a reciprocal opposite-label rival, "
             f"{stats['heldout_with_injected_subject_relation']}/{len(held)} held-out facts share (relation,subject) with an injected fact",
             **stats)
    bad_schema = [r["fact_id"] for r in allrec if r.get("schema_version") != 1 or r.get("protocol") != PROTOCOL_V1]
    rep.check(not bad_schema, "6.v1_schema", "every synthetic record has schema_version 1 / protocol v1", n_bad=len(bad_schema))
    claims = [r["claim_text"] for r in allrec]
    rep.check(len(set(claims)) == len(claims), "6.v1_claims_unique", f"claim_text unique across the legacy pool ({len(claims)} claims)")


# =================================================================================================
# manifest pass 1: aggregates, coverage, record agreement

class ManifestStats:
    def __init__(self):
        self.header = None
        self.n_rows = 0
        self.rows_per_arm = {}
        self.per_arm_fact = {}         # arm -> fact_id -> {"freq": set, "occ": list}
        self.rendering_dups = {}       # arm -> count of duplicate rendering ids
        self.unknown_arm_rows = 0
        self.order_ok = True
        self.slot_group_bad = 0        # consecutive rows with the same doc_index but different rendering_id
        self.doc_index_max = -1
        self.doc_index_rows = {}       # doc_index -> set(arms)  (bounded: one entry per slot)
        self.field_mismatch = []
        self.field_mismatch_n = 0
        self.inserts = []              # insert_text per row (all arms)
        self.insert_meta = []          # (arm, rendering_id)
        self.bad_rows = []


def manifest_pass1(rep: Report, path: Path, pr: Probes, arms, T: int, M: int):
    rep.start("5 coverage & uniqueness (manifest)")
    st = ManifestStats()
    if not path.exists():
        rep.check(False, "5.manifest_present", "manifest.csv present")
        return st
    rep.check(True, "5.manifest_present", "manifest.csv present")
    seen_rid = {a: set() for a in arms}
    last_doc, last_rid = -1, None
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        st.header = rd.fieldnames
        if st.header != MANIFEST_COLS_V2:
            rep.check(False, "5.manifest_columns", f"manifest columns == manifest v2 schema (got {st.header})")
            return st
        rep.check(True, "5.manifest_columns", f"manifest columns == manifest v2 schema ({len(MANIFEST_COLS_V2)} columns)")
        for row in rd:
            st.n_rows += 1
            a = row["arm"]
            if a not in arms:
                st.unknown_arm_rows += 1
                continue
            st.rows_per_arm[a] = st.rows_per_arm.get(a, 0) + 1
            fid, rid = row["fact_id"], row["rendering_id"]
            occ, freq, di = _to_int(row["occurrence"]), _to_int(row["assigned_frequency"]), _to_int(row["doc_index"])
            if rid in seen_rid[a]:
                st.rendering_dups[a] = st.rendering_dups.get(a, 0) + 1
            seen_rid[a].add(rid)
            ent = st.per_arm_fact.setdefault(a, {}).setdefault(fid, {"freq": set(), "occ": []})
            ent["freq"].add(freq)
            ent["occ"].append(occ)
            if di < last_doc:
                st.order_ok = False
            if di == last_doc and rid != last_rid:
                st.slot_group_bad += 1
            last_doc, last_rid = di, rid
            st.doc_index_max = max(st.doc_index_max, di)
            st.doc_index_rows.setdefault(di, set()).add(a)
            # agreement with the injected probe record
            rec = pr.by_id.get(fid)
            problems = []
            if rec is None:
                problems.append("fact_id not in injected_facts.jsonl")
            else:
                if _to_int(rec.get("assigned_frequency")) != freq:
                    problems.append("assigned_frequency != record")
                if row["group_id"] != str(rec.get("group_id")):
                    problems.append("group_id != record")
                if row["truth_value"] != str(rec.get("truth_value")):
                    problems.append("truth_value != record")
                if row["entity_tier"] != str(rec.get("entity_tier")) or row["domain"] != str(rec.get("domain")):
                    problems.append("entity_tier/domain != record")
            if rid != f"{fid}:{occ}":
                problems.append("rendering_id != fact_id:occurrence")
            if row["schema_version"] != "2":
                problems.append("schema_version != 2")
            if row["heldout"] != "False":
                problems.append("heldout != False")
            if not (0 <= di < T):
                problems.append(f"doc_index {di} outside 0..T-1")
            if problems:
                st.field_mismatch_n += 1
                if len(st.field_mismatch) < 10:
                    st.field_mismatch.append({"rendering_id": rid, "arm": a, "problems": problems})
            st.inserts.append(row["insert_text"])
            st.insert_meta.append((a, rid))
    rep.check(st.unknown_arm_rows == 0, "5.manifest_arms", f"every manifest row belongs to a present arm ({st.unknown_arm_rows} rows for absent arms)",
              n_unknown=st.unknown_arm_rows)
    rep.check(st.field_mismatch_n == 0, "5.rows_vs_records",
              f"manifest rows agree with injected records (dose, group_id, truth/tier/domain, rendering_id, schema, doc_index range): "
              f"{st.n_rows - st.field_mismatch_n}/{st.n_rows}", n_bad=st.field_mismatch_n, exclusions=st.field_mismatch)
    rep.check(all(st.rows_per_arm.get(a, 0) == M for a in arms), "5.rows_per_arm",
              f"every arm has M={M} manifest rows ({st.rows_per_arm})", rows_per_arm=st.rows_per_arm, M=M)
    rep.check(not st.rendering_dups, "5.rendering_id_unique", f"rendering_id unique within every arm (duplicates {st.rendering_dups})",
              duplicates=st.rendering_dups)
    # occurrences == assigned_frequency per fact per arm, occurrence indices complete
    bad_cov = []
    for a in arms:
        for fid, ent in st.per_arm_fact.get(a, {}).items():
            if len(ent["freq"]) != 1 or sorted(ent["occ"]) != list(range(next(iter(ent["freq"])))):
                bad_cov.append((a, fid, sorted(ent["freq"]), len(ent["occ"])))
    rep.check(not bad_cov, "5.occurrences_vs_dose",
              "per fact and arm: manifest occurrences == assigned_frequency with occurrence indices 0..f-1",
              n_bad=len(bad_cov), exclusions=[{"arm": a, "fact_id": f, "freq": fr, "rows": n} for a, f, fr, n in bad_cov[:10]])
    maps = {a: {fid: next(iter(ent["freq"])) for fid, ent in st.per_arm_fact.get(a, {}).items()} for a in arms}
    claim_arms = [a for a in arms if a != "C"]
    same = all(maps[a] == maps["R"] for a in claim_arms) and maps.get("C") == maps.get("R")
    rep.check(same and bool(maps.get("R")), "5.dose_maps_identical",
              f"identical (fact_id -> assigned_frequency) maps across arms {arms} ({len(maps.get('R', {}))} facts)")
    rec_map = {r["fact_id"]: _to_int(r.get("assigned_frequency")) for r in pr.injected}
    rep.check(maps.get("R") == rec_map, "5.dose_map_vs_records",
              f"manifest dose map == injected_facts.jsonl dose map ({len(rec_map)} records)",
              only_in_manifest=sorted(set(maps.get("R", {})) - set(rec_map))[:10], only_in_records=sorted(set(rec_map) - set(maps.get("R", {})))[:10])
    exp_M = sum(rec_map.values())
    rep.check(exp_M == M, "5.expected_total_M", f"sum of injected doses {exp_M} == summary M={M}", sum_doses=exp_M, M=M)
    rep.check(st.slot_group_bad == 0, "5.slot_grouping", "rows sharing a doc_index share the rendering_id (one slot = one occurrence)",
              n_bad=st.slot_group_bad)
    missing_arms = {di: sorted(set(arms) - s) for di, s in st.doc_index_rows.items() if set(arms) != s}
    rep.check(not missing_arms, "5.slot_all_arms", f"every slot (doc_index) has a row for every arm ({len(st.doc_index_rows)} slots)",
              n_slots=len(st.doc_index_rows), n_bad=len(missing_arms), exclusions=[{"doc_index": d, "missing": m} for d, m in list(missing_arms.items())[:10]])
    if not st.order_ok:
        rep.note("5.manifest_order", "manifest rows are not in ascending doc_index order; slots will be sorted in memory")
    return st


# =================================================================================================
# manifest pass 2 + lockstep shard pass: alignment, spans, parity, top-up, X/E

def _iter_slots(path: Path, order_ok: bool):
    """Yield (doc_index, {arm: row}) in ascending doc_index order."""
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        if order_ok:
            cur_di, cur = None, {}
            for row in rd:
                di = _to_int(row["doc_index"])
                if cur_di is not None and di != cur_di:
                    yield cur_di, cur
                    cur = {}
                cur_di = di
                cur.setdefault(row["arm"], []).append(row)
            if cur_di is not None:
                yield cur_di, cur
        else:
            rows = sorted(rd, key=lambda r: _to_int(r["doc_index"]))
            cur_di, cur = None, {}
            for row in rows:
                di = _to_int(row["doc_index"])
                if cur_di is not None and di != cur_di:
                    yield cur_di, cur
                    cur = {}
                cur_di = di
                cur.setdefault(row["arm"], []).append(row)
            if cur_di is not None:
                yield cur_di, cur


def _train_stream(paths):
    for si, p in enumerate(paths[:-1]):
        for r, t in enumerate(read_shard_texts(p)):
            yield si, r, t


class Tally:
    """Counter bag with bounded examples."""

    def __init__(self):
        self.n = {}
        self.ex = {}

    def hit(self, key, example=None):
        self.n[key] = self.n.get(key, 0) + 1
        if example is not None and len(self.ex.setdefault(key, [])) < 5:
            self.ex[key].append(example)

    def get(self, key):
        return self.n.get(key, 0)


def _span(row, a, b):
    return _to_int(row[a]), _to_int(row[b])


def check_slot(rows: dict, docs: dict, pr: Probes, protocol: str, arms, tally: Tally, carrier_limit, embed: bool):
    """All per-slot checks for one occurrence. rows: {arm: [row]}, docs: {arm: (shard_idx, row_in_shard, text)}."""
    rid = next(iter(rows.values()))[0]["rendering_id"]
    for a in arms:
        if a not in rows:
            tally.hit("slot_missing_arm_row", f"{rid}:{a}")
    starts = {}
    carriers = {}
    sf_by_arm = {}
    for a, rl in rows.items():
        if a not in docs:
            continue
        if len(rl) != 1:
            tally.hit("slot_duplicate_rows", f"{rid}:{a}")
        row = rl[0]
        sidx, rin, doc = docs[a]
        ex = f"{rid}:{a}"
        # --- 3. position ---
        if row["shard_file"] != shard_filename(sidx) or _to_int(row["row_in_shard"]) != rin:
            tally.hit("position_mismatch", f"{ex} manifest {row['shard_file']}#{row['row_in_shard']} actual {shard_filename(sidx)}#{rin}")
        if doc == docs["C"][2] and a != "C":
            tally.hit("claim_arm_identical_to_C", ex)
        # --- 4. spans ---
        ins = row["insert_text"]
        s, e = _span(row, "insert_char_start", "insert_char_end")
        if not (0 <= s <= e <= len(doc)) or doc[s:e] != ins or e - s != len(ins):
            tally.hit("insert_text_mismatch", ex)
            continue
        if not embed and (s != 0 or e != len(doc)):
            tally.hit("insert_not_whole_doc", ex)
        starts[a] = s
        carriers[a] = doc[:s] + doc[e:]
        ri = _to_int(row["register_index"])
        lead = REGISTER_LEADINS[ri] if 0 <= ri < len(REGISTER_LEADINS) else None
        if lead is None or not ins.startswith(lead):
            tally.hit("leadin_mismatch", ex)
            lead = ""
        cs, ce = _span(row, "claim_char_start", "claim_char_end")
        ss, se = _span(row, "subject_char_start", "subject_char_end")
        vs, ve = _span(row, "value_char_start", "value_char_end")
        os_, oe = _span(row, "source_char_start", "source_char_end")
        try:
            vspans = [tuple(x) for x in json.loads(row["value_spans"])]
        except (ValueError, TypeError):
            vspans = None
            tally.hit("value_spans_unparseable", ex)
        if a == "C":
            if (cs, ce, ss, se, vs, ve, os_, oe) != (-1,) * 8 or vspans:
                tally.hit("C_has_claim_spans", ex)
            if _to_int(row["paraphrase_index"]) != -1 or row["source_id"]:
                tally.hit("C_has_claim_spans", ex)
            continue
        rec = pr.by_id.get(row["fact_id"])
        if rec is None:
            continue  # reported in pass 1
        if not (s <= cs < ce <= e):
            tally.hit("claim_span_outside_insert", ex)
            continue
        claim = doc[cs:ce]
        forms = list(rec.get("surface_forms") or [])
        # subject / value realizations
        if not (cs <= ss < se <= ce):
            tally.hit("subject_span_outside_claim", ex)
        else:
            r = _realization(doc[ss:se], rec["subject"])
            if r is None:
                tally.hit("subject_text_mismatch", f"{ex} {doc[ss:se]!r} != {rec['subject']!r}")
            elif r == "lowercased":
                tally.hit("subject_lowercased", ex)
        if vspans is not None:
            if not vspans or (vs, ve) != vspans[0]:
                tally.hit("value_span_first_mismatch", ex)
            for (a0, b0) in vspans:
                if not (cs <= a0 < b0 <= ce):
                    tally.hit("value_span_outside_claim", ex)
                    continue
                r = _realization(doc[a0:b0], str(rec["value"]))
                if r is None:
                    tally.hit("value_text_mismatch", f"{ex} {doc[a0:b0]!r} != {rec['value']!r}")
                elif r == "lowercased":
                    tally.hit("value_lowercased", ex)
                else:
                    tally.hit("value_exact")
            # every occurrence of the value inside the claim must be a recorded slot (only meaningful
            # when the value is not a substring of the subject)
            if str(rec["value"]) not in rec["subject"]:
                n_raw = _count_value_occurrences(claim, str(rec["value"]))
                if n_raw != len(vspans):
                    tally.hit("value_slots_incomplete", f"{ex} {n_raw} value occurrences in claim, {len(vspans)} recorded")
        # the claim text vs the record's surface forms
        if a == "R":
            if claim not in forms:
                tally.hit("R_claim_not_surface_form", f"{ex} {claim!r}")
            if ins[len(lead):] != claim:
                tally.hit("R_insert_not_leadin_plus_claim", ex)
            if (os_, oe) != (-1, -1) or row["source_id"]:
                tally.hit("source_span_in_non_X", ex)
            sf = claim
        else:
            matches = [f for f in forms if claim in _clause_forms(f)]
            if not matches:
                tally.hit(f"{a}_claim_not_clause_of_surface_form", f"{ex} {claim!r}")
            sf = matches[0] if matches else claim
        sf_by_arm[a] = sf
        pi = _to_int(row["paraphrase_index"])
        if protocol == PROTOCOL_V2 and (not (0 <= pi < len(forms)) or forms[pi] != sf):
            tally.hit("paraphrase_index_mismatch", ex)
        if rec.get("heldout_paraphrase") and len(forms) > 1 and sf == rec["heldout_paraphrase"]:
            tally.hit("heldout_paraphrase_trained", ex)
        wrapper = ins[len(lead):cs - s] + "\n" + ins[ce - s:]   # frame material: outside lead-in and claim
        wl = wrapper.lower()
        words = set(_WORD.findall(wl))
        has_sig = bool(words & ATTRIBUTION_VERBS) or any(c in wrapper for c in ATTRIBUTION_CONNECTIVES)
        if a == "X":
            if not has_sig:
                tally.hit("X_without_attribution_signature", f"{ex} {ins!r}")
            if not (s <= os_ < oe <= e) or (os_ < ce and cs < oe):
                tally.hit("X_source_span_bad", ex)
            else:
                src = doc[os_:oe]
                if src.rstrip(",:").strip() != row["source_id"].rstrip(",").strip() or not src.strip():
                    tally.hit("X_source_text_mismatch", f"{ex} {src!r} != {row['source_id']!r}")
                elif src != row["source_id"]:
                    tally.hit("X_source_span_punctuation", f"{ex} {src!r} vs {row['source_id']!r}")
                if not any(n in src for n in FIRST_NAMES):
                    tally.hit("X_source_without_name", f"{ex} {src!r}")
                if rec.get("fixed_source") and row["source_id"] != rec["fixed_source"]:
                    tally.hit("X_fixed_source_mismatch", ex)
            # the raw claim must not appear un-attributed elsewhere in the X document
            outside = doc[:s] + "\n" + doc[e:]
            bare = sf[:-1] if sf.endswith(".") else sf
            if bare and (bare in outside or _lower_first(bare) in outside):
                tally.hit("X_raw_claim_elsewhere", ex)
        else:  # E
            if (os_, oe) != (-1, -1) or row["source_id"]:
                tally.hit("source_span_in_non_X", ex)
            cues = []
            if has_sig:
                cues.append("signature")
            if any(n in wrapper for n in FIRST_NAMES):
                cues.append("first_name")
            if any(n in wrapper for n in INSTITUTIONS):
                cues.append("institution")
            if any(t in wl for t in SOURCE_TYPES):
                cues.append("source_type")
            if cues:
                tally.hit("E_with_source_cue", f"{ex} {cues} {ins!r}")
        if a == "R":
            bare = sf[:-1] if sf.endswith(".") else sf
            n = doc.count(bare) if bare else 0
            if n != 1:
                tally.hit("R_claim_count_not_one", f"{ex} {n}")
        # carrier bookkeeping
        if carrier_limit:
            cc = _to_int(row["carrier_chars"])
            cropped = row["carrier_cropped"] == "True"
            recon = carriers[a]
            if not cropped and cc > carrier_limit:
                tally.hit("carrier_over_limit_not_cropped", f"{ex} {cc}")
            if cropped and cc > carrier_limit and ". " in recon.strip():
                tally.hit("carrier_cropped_over_limit", f"{ex} {cc}")
    # --- cross-arm: same position, same carrier, same surface form ---
    if len(starts) > 1:
        if len(set(starts.values())) != 1:
            tally.hit("insert_position_differs_across_arms", rid)
        if len(set(carriers.values())) != 1:
            tally.hit("carrier_differs_across_arms", rid)
    if len(sf_by_arm) > 1 and len(set(sf_by_arm.values())) != 1:
        tally.hit("surface_form_differs_across_arms", rid)
    for a, rl in rows.items():
        r0 = rl[0]
        base = rows.get("R", rows.get("C"))[0]
        if (r0["fact_id"], r0["occurrence"], r0["assigned_frequency"]) != (base["fact_id"], base["occurrence"], base["assigned_frequency"]):
            tally.hit("slot_rows_disagree", rid)


def lockstep_pass(rep: Report, out: Path, summary: dict, files: dict, arms, pr: Probes, st: ManifestStats,
                  tokenizer, protocol: str, n_examples: int = 4):
    T = int(summary["n_train_base_docs_T"])
    M = int(summary["n_occurrences_M"])
    cfg = summary.get("config") or {}
    tol = float(cfg.get("parity_tol", 0.0049))
    embed = bool(cfg.get("embed", True))
    carrier_limit = cfg.get("carrier_max_chars") if protocol == PROTOCOL_V2 else None
    manifest = out / "manifest.csv"
    tally = Tally()
    streams = {a: _train_stream(files[a][0]) for a in arms}
    slots = _iter_slots(manifest, st.order_ok) if manifest.exists() and st.header == MANIFEST_COLS_V2 else iter(())
    next_slot = next(slots, None)
    totals = {a: 0 for a in arms}
    n_docs = {a: 0 for a in arms}
    tails = {a: [] for a in arms}
    diff_positions = {a: [] for a in arms if a != "C"}
    n_diff = {a: 0 for a in arms if a != "C"}
    slot_positions = []
    examples = []
    xe_pairs = []
    xe = {"dc": [], "ds": [], "dt": []}
    pos = 0
    t0 = time.time()
    claim_arms = [a for a in arms if a != "C"]
    print(f"  streaming {len(arms)} arms in lockstep (T={T:,}, M={M:,}) ...", flush=True)
    while True:
        cur = {}
        for a in arms:
            item = next(streams[a], None)
            if item is not None:
                cur[a] = item
                totals[a] += len(item[2])
                n_docs[a] += 1
        if not cur:
            break
        if pos and pos % 1_000_000 == 0:
            print(f"    ... {pos:,} docs ({time.time() - t0:.0f}s)", flush=True)
        if pos < T:
            if len(cur) != len(arms):
                tally.hit("train_stream_short", f"position {pos}: arms {sorted(cur)} only")
                pos += 1
                continue
            c_doc = cur["C"][2]
            while next_slot is not None and next_slot[0] < pos:   # should not happen (sorted); keep going
                tally.hit("slot_skipped", f"doc_index {next_slot[0]}")
                next_slot = next(slots, None)
            if next_slot is not None and next_slot[0] == pos:
                slot_positions.append(pos)
                for a in claim_arms:
                    if cur[a][2] != c_doc:
                        n_diff[a] += 1
                rows = next_slot[1]
                check_slot(rows, cur, pr, protocol, arms, tally, carrier_limit, embed)
                if len(examples) < n_examples:
                    examples.append({a: rows[a][0]["insert_text"] for a in arms if a in rows})
                if "X" in rows and "E" in rows and "X" in cur and "E" in cur:
                    rx, re_ = rows["X"][0], rows["E"][0]
                    tx, te = rx["insert_text"], re_["insert_text"]
                    dc = len(te) - len(tx)
                    ds = (_to_int(re_["claim_char_start"]) - _to_int(re_["insert_char_start"])) - \
                         (_to_int(rx["claim_char_start"]) - _to_int(rx["insert_char_start"]))
                    xe["dc"].append(abs(dc))
                    xe["ds"].append(abs(ds))
                    if tokenizer is not None:
                        xe["dt"].append(abs(len(tokenizer.encode(te)) - len(tokenizer.encode(tx))))
                    if len(xe_pairs) < 5:
                        xe_pairs.append((tx, te, dc, ds))
                next_slot = next(slots, None)
            else:
                for a in claim_arms:
                    if cur[a][2] != c_doc:
                        n_diff[a] += 1
                        if len(diff_positions[a]) < 5:
                            diff_positions[a].append(pos)
        else:
            for a, (si, r, text) in cur.items():
                tails[a].append((len(text), _doc_hash(text)))
        pos += 1
    while next_slot is not None:
        tally.hit("slot_beyond_stream", f"doc_index {next_slot[0]}")
        next_slot = next(slots, None)

    # ---- 2. parity ----
    rep.start("2 budget parity (recomputed from shards)")
    summ_tot = summary.get("arm_total_chars") or {}
    rep.check(all(summ_tot.get(a) == totals[a] for a in arms), "2.totals_vs_summary",
              f"recomputed train+top-up chars per arm == build_summary.arm_total_chars ({totals})", recomputed=totals, summary=summ_tot)
    base = [a for a in arms if a != "E"]
    ceiling = max(totals[a] for a in base)
    spread = (ceiling - min(totals[a] for a in base)) / max(1, ceiling)
    rep.check(spread <= tol + 1e-12, "2.spread", f"C/R/X char spread {spread * 100:.4f}% <= parity_tol {tol * 100:.2f}%",
              spread=spread, tol=tol, ceiling=ceiling)
    if summary.get("parity_ceiling") is not None:
        rep.check(summary["parity_ceiling"] == ceiling, "2.ceiling_vs_summary", f"ceiling {ceiling} == summary parity_ceiling {summary['parity_ceiling']}")
    rep.check(all(totals[a] <= ceiling for a in base), "2.under_ceiling", f"every C/R/X arm <= ceiling {ceiling}")
    rep.check(all(totals[a] >= ceiling * (1 - tol) for a in arms), "2.topped_up",
              f"every arm >= ceiling*(1-tol) = {ceiling * (1 - tol):.0f} (top-up complete)", shortfall={a: ceiling - totals[a] for a in arms})
    if "E" in arms:
        de = (totals["E"] - ceiling) / max(1, ceiling)
        rep.check(abs(de) <= tol + 1e-12, "2.E_within_tol", f"arm E is {de * 100:+.4f}% of the C/R/X ceiling (|.| <= {tol * 100:.2f}%)", delta=de)
        all_spread = (max(totals.values()) - min(totals.values())) / max(1, max(totals.values()))
        rep.note("2.spread_all_arms", f"spread over all arms incl. E: {all_spread * 100:.4f}%", spread=all_spread)
    # val shard identical
    val_h = {a: sha256_file(files[a][0][-1]) for a in arms if files[a][0]}
    val_n = {a: shard_num_rows(files[a][0][-1]) for a in arms if files[a][0]}
    rep.check(len(set(val_h.values())) == 1 and len(set(val_n.values())) == 1 and len(val_h) == len(arms), "2.val_identical",
              f"val shard (last) byte-identical across arms ({next(iter(val_n.values()), 0)} docs)", sha256=val_h)
    # top-up
    stream = summary.get("topup_stream_doc_chars")
    tcounts = summary.get("topup_docs") or {}
    rep.check(all(n_docs[a] == T + len(tails[a]) for a in arms) and all(tcounts.get(a) == len(tails[a]) for a in arms),
              "2.topup_counts", f"train docs per arm == T + top-up docs ({ {a: len(t) for a, t in tails.items()} } == summary topup_docs {tcounts})",
              topup=dict((a, len(t)) for a, t in tails.items()), n_docs=n_docs, T=T)
    if stream is None:
        rep.skip("2.topup_stream", "summary has no topup_stream_doc_chars (older builder); shared top-up stream not checked")
    else:
        bad_len = {}
        for a in arms:
            exp = [d["chars"] for d in stream if a in d.get("arms", [])]
            got = [n for n, _ in tails[a]]
            if exp != got:
                bad_len[a] = {"expected": len(exp), "got": len(got)}
        rep.check(not bad_len, "2.topup_stream",
                  "top-up docs per arm are, in order, exactly the shared stream docs recorded for that arm (LENGTH-based check: "
                  f"the stream records lengths only; {len(stream)} stream docs)", bad=bad_len)
        # content agreement across arms that took the same stream doc
        idx = {a: 0 for a in arms}
        n_shared, n_bad = 0, 0
        for d in stream:
            took = [a for a in d.get("arms", []) if a in arms]
            hashes = set()
            for a in took:
                if idx[a] < len(tails[a]):
                    hashes.add(tails[a][idx[a]][1])
                idx[a] += 1
            if len(took) > 1:
                n_shared += 1
                if len(hashes) != 1:
                    n_bad += 1
        rep.check(n_bad == 0, "2.topup_shared_content", f"stream docs taken by several arms are byte-identical across those arms ({n_shared} shared docs)",
                  n_shared=n_shared, n_bad=n_bad)
    _token_parity_note(rep, files, arms, tokenizer)

    # ---- 3. slot alignment ----
    rep.start("3 slot alignment (aligned C-vs-arm diff over the first T train docs)")
    rep.check(tally.get("train_stream_short") == 0, "3.stream_length", f"every arm has at least T={T} train docs", bad=tally.ex.get("train_stream_short", []))
    rep.check(all(n_diff[a] == M for a in claim_arms), "3.diff_count", f"each claim arm differs from C at exactly M={M} positions ({n_diff})", n_diff=n_diff, M=M)
    n_slots = len(slot_positions)
    unexpected = {a: n_diff[a] - sum(1 for _ in slot_positions) for a in claim_arms}
    rep.check(all(len(diff_positions[a]) == 0 for a in claim_arms) and n_slots == M, "3.diff_positions",
              f"claim arms differ from C exactly at the manifest slot positions ({n_slots} slots; off-manifest diffs {diff_positions})",
              off_manifest=diff_positions, n_slots=n_slots)
    rep.check(tally.get("claim_arm_identical_to_C") == 0, "3.slots_differ", "at every slot every claim arm differs from C", n_bad=tally.get("claim_arm_identical_to_C"),
              exclusions=tally.ex.get("claim_arm_identical_to_C", []))
    rep.check(tally.get("position_mismatch") == 0 and tally.get("slot_beyond_stream") == 0 and tally.get("slot_skipped") == 0, "3.manifest_positions",
              "manifest doc_index / shard_file / row_in_shard agree with where the docs are",
              n_bad=tally.get("position_mismatch") + tally.get("slot_beyond_stream") + tally.get("slot_skipped"),
              exclusions=(tally.ex.get("position_mismatch", []) + tally.ex.get("slot_beyond_stream", []))[:10])
    rep.check(tally.get("slot_missing_arm_row") == 0 and tally.get("slot_duplicate_rows") == 0 and tally.get("slot_rows_disagree") == 0, "3.slot_rows",
              "every slot has exactly one row per arm and the rows agree on fact/occurrence/dose",
              n_bad=tally.get("slot_missing_arm_row") + tally.get("slot_duplicate_rows") + tally.get("slot_rows_disagree"),
              exclusions=(tally.ex.get("slot_missing_arm_row", []) + tally.ex.get("slot_rows_disagree", []))[:10])

    # ---- 4. span integrity ----
    rep.start("4 span integrity (manifest spans vs shard text)")
    def t(key, cid, msg, **kw):
        n = tally.get(key)
        return rep.check(n == 0, cid, f"{msg} ({n} violations)", n_bad=n, exclusions=tally.ex.get(key, []), **kw)
    t("insert_text_mismatch", "4.insert_text", "doc[insert_char_start:insert_char_end] == insert_text")
    if not embed:
        t("insert_not_whole_doc", "4.insert_whole_doc", "--no-embed: the insert is the whole document")
    t("leadin_mismatch", "4.register_leadin", "insert_text starts with REGISTER_LEADINS[register_index]")
    t("C_has_claim_spans", "4.C_neutral", "arm C rows carry no claim/subject/value/source spans")
    t("claim_span_outside_insert", "4.claim_in_insert", "claim span lies inside the insert")
    t("subject_span_outside_claim", "4.subject_in_claim", "subject span lies inside the claim span")
    t("subject_text_mismatch", "4.subject_text", "subject span text == record subject (or its first character lower-cased)")
    rep.note("4.subject_lowercased", f"{tally.get('subject_lowercased')} subject realizations with a lower-cased first character", n=tally.get("subject_lowercased"))
    t("value_spans_unparseable", "4.value_spans_json", "value_spans parses as a JSON list of [start,end]")
    t("value_span_first_mismatch", "4.value_first_span", "value_char_start/end == value_spans[0]")
    t("value_span_outside_claim", "4.values_in_claim", "every value span lies inside the claim span")
    t("value_text_mismatch", "4.value_text", "every value span text == record value (or its first character lower-cased)")
    rep.note("4.value_lowercased", f"{tally.get('value_lowercased')} value realizations lower-cased (finding 9), {tally.get('value_exact')} exact",
             n_lowercased=tally.get("value_lowercased"), n_exact=tally.get("value_exact"))
    t("value_slots_incomplete", "4.value_slots_complete", "every verbatim occurrence of the value inside the claim is a recorded value span")
    t("R_claim_not_surface_form", "4.R_surface_form", "R claim text (minus lead-in) is one of the record's surface forms")
    t("R_insert_not_leadin_plus_claim", "4.R_structure", "R insert == lead-in + surface form, nothing else")
    t("X_claim_not_clause_of_surface_form", "4.X_claim_text", "X claim span is a surface form verbatim or as a that-clause")
    if "E" in arms:
        t("E_claim_not_clause_of_surface_form", "4.E_claim_text", "E claim span is a surface form verbatim or as a that-clause")
    if protocol == PROTOCOL_V2:
        t("paraphrase_index_mismatch", "4.paraphrase_index", "surface_forms[paraphrase_index] is the rendered form (v2 frame order)")
    t("heldout_paraphrase_trained", "4.heldout_paraphrase_unused", "the reserved held-out paraphrase is never the trained form")
    t("surface_form_differs_across_arms", "4.matched_paraphrase", "R/X/E of a slot use the same surface form")
    t("source_span_in_non_X", "4.source_only_in_X", "source span / source_id absent for C, R and E")
    t("X_source_span_bad", "4.X_source_span", "X source span non-empty, inside the insert and outside the claim")
    t("X_source_text_mismatch", "4.X_source_text", "X source span text == source_id (trailing tidy punctuation ignored)")
    rep.note("4.X_source_span_punctuation", f"{tally.get('X_source_span_punctuation')} X source spans include a trailing tidy "
             "character (',' turned into ':' before a quoted claim; span remapped past it by templates.TaggedText.replace_tracked)",
             n=tally.get("X_source_span_punctuation"), examples=tally.ex.get("X_source_span_punctuation", []))
    t("X_source_without_name", "4.X_source_named", "X source names a person (FIRST_NAMES)")
    t("X_fixed_source_mismatch", "4.X_fixed_source", "per-fact source: X source_id == record fixed_source")
    t("X_without_attribution_signature", "4.X_signature",
      "X carries an attribution signature (reporting verb / connective) inside the insert, outside claim and lead-in")
    t("X_raw_claim_elsewhere", "4.X_no_bare_claim", "the raw claim never appears un-attributed elsewhere in the X document")
    t("R_claim_count_not_one", "4.R_claim_once", "the raw claim appears exactly once in the R document")
    if "E" in arms:
        t("E_with_source_cue", "4.E_clean_frame",
          "E frame (outside claim and lead-in) has no reporting verb / connective / FIRST_NAMES / INSTITUTIONS / SOURCE_TYPES")
    t("insert_position_differs_across_arms", "4.same_position", "the insert starts at the same character offset in every arm")
    t("carrier_differs_across_arms", "4.shared_carrier", "document minus insert is byte-identical across arms (shared carrier)")
    if carrier_limit:
        t("carrier_over_limit_not_cropped", "4.carrier_limit", f"carrier_chars <= carrier_max_chars={carrier_limit} unless cropped")
        t("carrier_cropped_over_limit", "4.carrier_cropped", "a cropped carrier over the limit keeps a single sentence only")
    else:
        rep.note("4.carrier_limit", "no carrier_max_chars in the build config (v1 or 0): cropping not checked")

    # ---- 8. X/E matching ----
    if "E" in arms:
        rep.start("8 X/E matching (recomputed from manifest spans)")
        if xe["dc"]:
            lim = XE_MATCH_LIMITS
            nums = {"n": len(xe["dc"]), "max_abs_delta_chars": max(xe["dc"]), "mean_abs_delta_chars": statistics.mean(xe["dc"]),
                    "max_abs_delta_claim_start": max(xe["ds"]), "mean_abs_delta_claim_start": statistics.mean(xe["ds"]), "limits": lim}
            ok_c = nums["max_abs_delta_chars"] <= lim["max_abs_delta_chars"] and nums["mean_abs_delta_chars"] <= lim["max_mean_abs_delta_chars"]
            ok_s = nums["max_abs_delta_claim_start"] <= lim["max_abs_delta_claim_start"]
            msg_c = (f"|len(E)-len(X)| chars: max {nums['max_abs_delta_chars']} <= {lim['max_abs_delta_chars']}, "
                     f"mean {nums['mean_abs_delta_chars']:.2f} <= {lim['max_mean_abs_delta_chars']}")
            msg_s = f"|claim start E - X| chars: max {nums['max_abs_delta_claim_start']} <= {lim['max_abs_delta_claim_start']}"
            if protocol == PROTOCOL_V2:
                rep.check(ok_c, "8.char_length", msg_c, **nums)
                rep.check(ok_s, "8.claim_start", msg_s, **nums)
            else:
                rep.note("8.char_length", "legacy E frames (protocol v1) are not length-matched by design (finding 5): " + msg_c, **nums)
                rep.note("8.claim_start", msg_s, **nums)
            if xe["dt"]:
                nt = {"max_abs_delta_tokens": max(xe["dt"]), "mean_abs_delta_tokens": statistics.mean(xe["dt"])}
                ok_t = nt["max_abs_delta_tokens"] <= lim["max_abs_delta_tokens"] and nt["mean_abs_delta_tokens"] <= lim["max_mean_abs_delta_tokens"]
                msg_t = (f"|tokens(E)-tokens(X)|: max {nt['max_abs_delta_tokens']} <= {lim['max_abs_delta_tokens']}, "
                         f"mean {nt['mean_abs_delta_tokens']:.2f} <= {lim['max_mean_abs_delta_tokens']}")
                if protocol == PROTOCOL_V2:
                    rep.check(ok_t, "8.token_length", msg_t, **nt)
                else:
                    rep.note("8.token_length", msg_t, **nt)
            else:
                rep.skip("8.token_length", "no tokenizer available (arm_C/tokenizer or --tokenizer-dir): token deltas not computed")
            print("  X/E example pairs:")
            for tx, te, dc, ds in xe_pairs:
                print(f"    [X] {tx}\n    [E] {te}\n        (delta chars {dc:+d}, delta claim start {ds:+d})")
        else:
            rep.check(False, "8.pairs", "no X/E pairs found in the manifest although arm E is present")

    return examples, totals


def _token_parity_note(rep: Report, files, arms, tokenizer):
    if tokenizer is None:
        rep.note("2.tokens_per_char", "no tokenizer: char parity is the token proxy")
        return
    try:
        ratios = {}
        for a in arms:
            docs = []
            for t in read_shard_texts(files[a][0][0]):
                docs.append(t)
                if len(docs) >= 200:
                    break
            nch = sum(len(d) for d in docs)
            ntok = sum(len(tokenizer.encode(d)) for d in docs)
            ratios[a] = round(ntok / max(1, nch), 4)
        rep.note("2.tokens_per_char", f"tokens/char by arm (first 200 docs of shard 0; a sample, not an accounting): {ratios}", ratios=ratios)
    except Exception as e:  # pragma: no cover
        rep.note("2.tokens_per_char", f"tokenizer present but unusable ({e}); char parity is the token proxy")


# =================================================================================================
# 7. held-out leakage

def check_heldout_leakage(rep: Report, pr: Probes, st: ManifestStats, protocol: str):
    rep.start("7 held-out leakage (word-bounded search over every insert of every arm)")
    inserts = st.inserts
    if not inserts:
        rep.check(False, "7.inserts", "no inserts to scan (manifest missing or empty)")
        return
    blob = "\n".join(inserts)
    offsets = []
    o = 0
    for s in inserts:
        offsets.append(o)
        o += len(s) + 1

    def insert_at(i):
        k = bisect.bisect_right(offsets, i) - 1
        return k, inserts[k]

    # strings per held-out record, grouped by subject so the blob is scanned once per subject
    by_subject = {}
    for r in pr.heldout:
        strs = set()
        for s in [r.get("claim_text")] + list(r.get("surface_forms") or []):
            if not s:
                continue
            b = s[:-1] if s.endswith(".") else s
            strs.add(b)
            strs.add(_lower_first(b))
        by_subject.setdefault(r["subject"], []).append((r, strs))
    n_strings = sum(len(x[1]) for v in by_subject.values() for x in v)
    leaks, rival_leaks = [], []
    subj_hits = {"novel": [], "real": []}
    subj_hit_counts = {"novel": 0, "real": 0}
    v1_subject_hits = 0
    for subject, recs in by_subject.items():
        hits = _bounded_find_all(blob, subject)
        if not hits:
            continue
        hit_inserts = sorted({insert_at(i)[0] for i in hits})
        for r, strs in recs:
            tier = r.get("entity_tier", "novel")
            is_heldout_group = (protocol == PROTOCOL_V2 and r.get("split") == "heldout_group")
            is_rival = (protocol == PROTOCOL_V2 and r.get("role") == "unasserted_rival")
            for k in hit_inserts:
                text = inserts[k]
                arm, rid = st.insert_meta[k]
                for s in strs:
                    if _bounded_in(text, s):
                        (rival_leaks if is_rival else leaks).append({"string": s, "arm": arm, "rendering_id": rid, "fact_id": r["fact_id"]})
            if is_heldout_group:
                n = sum(1 for k in hit_inserts if _bounded_in(inserts[k], subject))
                if n:
                    subj_hit_counts[tier] = subj_hit_counts.get(tier, 0) + n
                    if len(subj_hits.setdefault(tier, [])) < 10:
                        k = hit_inserts[0]
                        subj_hits[tier].append({"subject": subject, "fact_id": r["fact_id"], "arm": st.insert_meta[k][0],
                                                "rendering_id": st.insert_meta[k][1], "insert": inserts[k][:160]})
            elif protocol == PROTOCOL_V1:
                v1_subject_hits += 1
    rep.check(not leaks, "7.heldout_claims", f"no held-out claim / surface form (nor its clause variant) appears in any insert "
              f"(checked {n_strings} strings against {len(inserts)} inserts)", n_leaks=len(leaks), exclusions=leaks[:10])
    if protocol == PROTOCOL_V2:
        rep.check(not rival_leaks, "7.rivals_unasserted", "no unasserted rival's claim / surface form appears in any insert",
                  n_leaks=len(rival_leaks), exclusions=rival_leaks[:10])
        rep.check(subj_hit_counts.get("novel", 0) == 0, "7.heldout_subjects_novel",
                  f"no NOVEL held-out-group subject appears (word-bounded) in any insert ({subj_hit_counts.get('novel', 0)} hits)",
                  n_hits=subj_hit_counts.get("novel", 0), exclusions=subj_hits.get("novel", []))
        nr = subj_hit_counts.get("real", 0)
        rep.note("7.heldout_subjects_real", f"{nr} insert(s) mention a REAL-entity held-out-group subject (real names recur across "
                 "relations and in the base corpus; see --scan-heldout-real)", n_hits=nr, examples=subj_hits.get("real", []))
    else:
        rep.note("7.heldout_subjects_v1", f"{v1_subject_hits} held-out records whose subject occurs in inserts "
                 "(legacy pools share subjects between injected and held-out facts; finding 1)", n=v1_subject_hits)


# =================================================================================================
# 9. tokenizer identity

def check_tokenizer_identity(rep: Report, out: Path, summary: dict, arms):
    rep.start("9 tokenizer identity")
    present = [a for a in arms if (out / f"arm_{a}" / "tokenizer" / "tokenizer.pkl").exists()]
    if not present:
        rep.skip("9.identity", "no arm_*/tokenizer/tokenizer.pkl: tokenizer not copied into the arms")
        return None
    rep.check(present == list(arms), "9.present_in_all_arms", f"tokenizer present in every arm ({present})")
    fps = {}
    for a in present:
        try:
            fps[a] = tokenizer_fingerprint(out / f"arm_{a}")
        except ProtocolError as e:
            rep.check(False, f"9.fingerprint_{a}", f"arm_{a}: {e}")
    pkl = {a: f["tokenizer_pkl_sha256"] for a, f in fps.items()}
    tb = {a: f.get("token_bytes_sha256") for a, f in fps.items()}
    rep.check(len(set(pkl.values())) == 1 and len(pkl) == len(arms), "9.pkl_identical", f"tokenizer.pkl sha256 identical across arms ({pkl})", sha256=pkl)
    if set(tb.values()) == {None}:
        rep.note("9.token_bytes_identical", "token_bytes.pt absent in every arm (only tokenizer.pkl was copied)")
    else:
        rep.check(len(set(tb.values())) == 1 and None not in tb.values(), "9.token_bytes_identical",
                  f"token_bytes.pt present and identical across arms ({tb})", sha256=tb)
    rec = (summary.get("input_hashes") or {}).get("tokenizer")
    if rec:
        rep.check(rec.get("tokenizer_pkl_sha256") == pkl.get("C") and rec.get("token_bytes_sha256") == tb.get("C"), "9.vs_summary",
                  "arm tokenizer fingerprint == build_summary.input_hashes.tokenizer", summary=rec)
    else:
        rep.note("9.vs_summary", "build_summary records no tokenizer fingerprint (tokenizer copied after the build?)")
    return str(out / "arm_C" / "tokenizer")


# =================================================================================================
# 10. counts

def check_counts(rep: Report, summary: dict, pr: Probes, st: ManifestStats, protocol: str):
    rep.start("10 counts (build_summary vs probe files + manifest)")
    inj, held = pr.injected, pr.heldout
    M = int(summary.get("n_occurrences_M", -1))
    probes = summary.get("probes") or {}
    rep.check(probes.get("injected") == len(inj) and probes.get("heldout_synthetic") == len(held) and probes.get("heldout_counterfact") == len(pr.counterfact),
              "10.probe_file_counts", f"summary.probes {probes} == files (injected {len(inj)}, heldout synthetic {len(held)}, counterfact {len(pr.counterfact)})")
    rep.check(summary.get("n_injected_facts") == len(inj) and summary.get("n_heldout_facts") == len(held), "10.n_facts_split",
              f"n_injected_facts {summary.get('n_injected_facts')} == {len(inj)}, n_heldout_facts {summary.get('n_heldout_facts')} == {len(held)}")
    occ = sum(_to_int(r.get("assigned_frequency"), 0) for r in inj)
    rep.check(occ == M and st.rows_per_arm.get("R") == M, "10.occurrences", f"occurrences: doses {occ} == M {M} == manifest R rows {st.rows_per_arm.get('R')}")
    counts = summary.get("counts") or {}
    if protocol == PROTOCOL_V2:
        allrec = inj + held
        groups = {}
        for r in allrec:
            groups.setdefault(r.get("group_id"), []).append(r)
        rec = {"groups": len(groups), "pairs": len(groups), "candidate_statements": len(allrec),
               "training_groups": sum(c[0].get("split") == "train_group" for c in groups.values()),
               "heldout_groups": sum(c[0].get("split") == "heldout_group" for c in groups.values()),
               "injected_statements": len(inj), "unasserted_rivals": sum(r.get("role") == "unasserted_rival" for r in held),
               "occurrences": occ, "real_groups": sum(c[0].get("entity_tier") == "real" for c in groups.values()),
               "novel_groups": sum(c[0].get("entity_tier") == "novel" for c in groups.values())}
        diff = {k: (counts.get(k), v) for k, v in rec.items() if counts.get(k) != v}
        rep.check(not diff, "10.counts", f"summary.counts == recomputed {rec}", recomputed=rec, differences=diff)
        rep.check(summary.get("n_facts") == rec["candidate_statements"], "10.n_facts", f"n_facts {summary.get('n_facts')} == candidate statements {rec['candidate_statements']}")
        tiers = {"real": rec["real_groups"], "novel": rec["novel_groups"]}
        rep.check(summary.get("tier_counts") == tiers, "10.tier_counts", f"tier_counts {summary.get('tier_counts')} == {tiers}")
        nov = {}
        for c in groups.values():
            nov[c[0].get("novelty")] = nov.get(c[0].get("novelty"), 0) + 1
        rep.check(summary.get("novelty") == nov, "10.novelty_counts", f"summary.novelty {summary.get('novelty')} == {nov}")
    else:
        allrec = inj + held
        rec = {"facts": len(allrec), "injected_statements": len(inj), "heldout_statements": len(held), "occurrences": occ}
        diff = {k: (counts.get(k), v) for k, v in rec.items() if counts.get(k) != v}
        rep.check(not diff, "10.counts", f"summary.counts == recomputed {rec}", recomputed=rec, differences=diff)
        rep.check(summary.get("n_facts") == len(allrec), "10.n_facts", f"n_facts {summary.get('n_facts')} == {len(allrec)}")
        tiers, tvs = {}, {}
        for r in allrec:
            tiers[r.get("entity_tier")] = tiers.get(r.get("entity_tier"), 0) + 1
            tvs[r.get("truth_value")] = tvs.get(r.get("truth_value"), 0) + 1
        rep.check(summary.get("tier_counts") == tiers and summary.get("tv_counts") == tvs, "10.tier_tv_counts",
                  f"tier_counts {summary.get('tier_counts')} == {tiers}; tv_counts {summary.get('tv_counts')} == {tvs}")


# =================================================================================================
# optional: natural-occurrence scan of the whole Arm C corpus (NOTE, not a check)

def _scan_shard_for_strings(task):
    path, strs = task
    try:
        import ahocorasick
        auto = ahocorasick.Automaton()
        for s in strs:
            auto.add_word(s, s)
        auto.make_automaton()

        def find_all(doc):
            return [val for _, val in auto.iter(doc)]
    except ImportError:
        find_all = re.compile("|".join(re.escape(s) for s in strs)).findall
    hits = {}
    for doc in read_shard_texts(path):
        for m in find_all(doc):
            hits[m] = hits.get(m, 0) + 1
    return hits


def scan_heldout_real(rep: Report, c_paths, held):
    """Count natural occurrences of real-entity held-out claim strings across the whole Arm C
    corpus. Hits are not build leakage (FineWeb contains true facts) but such probes were seen."""
    rep.start("optional: natural occurrences of real-entity held-out claims in Arm C")
    from concurrent.futures import ProcessPoolExecutor
    strs = sorted({s for h in held if h.get("entity_tier") == "real"
                   for s in [h.get("claim_text", "")] + list(h.get("surface_forms") or []) if s})
    if not strs:
        rep.note("scan.heldout_real", "no real-entity held-out strings to scan")
        return
    workers = min(len(c_paths), os.cpu_count() or 4)
    hits = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for done, shard_hits in enumerate(ex.map(_scan_shard_for_strings, [(p, strs) for p in c_paths]), 1):
            if done % 10 == 0 or done == len(c_paths):
                print(f"    ... scanned {done}/{len(c_paths)} shards", flush=True)
            for s, n in shard_hits.items():
                hits[s] = hits.get(s, 0) + n
    top = sorted(hits.items(), key=lambda kv: -kv[1])[:10]
    rep.note("scan.heldout_real", f"{sum(hits.values())} natural occurrence(s) of {len(hits)} real-entity held-out string(s) in Arm C "
             f"(scanned {len(strs)} strings over {len(c_paths)} shards); exclude hit probes from the 'never-seen' eval",
             n_hits=sum(hits.values()), n_strings=len(strs), top=top)


# =================================================================================================

def _print_examples(examples, arms):
    print("\n[matched examples] same carrier, one inserted sentence per arm (first slots of the manifest):")
    for k, ex in enumerate(examples):
        print(f"\n  --- slot {k + 1} ---")
        for a in arms:
            if a in ex:
                print(f"  [{a}] {ex[a]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="build directory (holds arm_*/, manifest.csv, probe_sets/, build_summary.json)")
    ap.add_argument("--data-subdir", default=None, help="override the data subdir (default: build config)")
    ap.add_argument("--tokenizer-dir", default=None, help="tokenizer dir for token-level checks (default: arm_C/tokenizer if present)")
    ap.add_argument("--report", default=None, help="JSON report path (default <out>/validation_report.json)")
    ap.add_argument("--examples", type=int, default=4, help="matched slot examples to print")
    ap.add_argument("--scan-heldout-real", action="store_true",
                    help="also scan the FULL Arm C corpus for natural occurrences of real-entity held-out claim strings (slow; NOTE only)")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:  # pragma: no cover
        pass

    out = Path(args.out)
    rep = Report()
    report_path = Path(args.report) if args.report else out / "validation_report.json"
    meta = {"out": str(out), "validated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "validator_code": code_revision(),
            "argv": list(argv) if argv is not None else sys.argv[1:]}
    summary_path = out / "build_summary.json"
    summary = None
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except ValueError as e:
            summary = None
            print(f"  FAIL build_summary.json is not valid JSON: {e}")
    rep.start("0 inputs")
    if not rep.check(bool(summary), "0.summary", f"build_summary.json present and readable at {summary_path}"):
        return _finish(rep, report_path, meta, arms=[], protocol=None)
    try:
        protocol = protocol_of(summary)
    except ProtocolError as e:
        rep.check(False, "0.protocol", f"protocol: {e}")
        return _finish(rep, report_path, meta, arms=[], protocol=None)
    cfg = summary.get("config") or {}
    data_subdir = args.data_subdir or cfg.get("data_subdir", "base_data")
    emb = bool(cfg.get("embedding_control", False))
    arms = ["C", "R", "X"] + (["E"] if emb else [])
    on_disk = sorted(p.name[4:] for p in out.glob("arm_*") if p.is_dir())
    rep.check(sorted(arms) == on_disk, "0.arms", f"arms on disk {on_disk} == arms per build config {arms}")
    rep.check(all((out / f"arm_{a}" / data_subdir).is_dir() for a in arms), "0.arm_dirs", f"every arm has {data_subdir}/")
    meta.update({"protocol": protocol, "arms": arms, "build_code": summary.get("code"), "builder_version": summary.get("builder_version"),
                 "schema_versions": summary.get("schema_versions")})
    print(f"  protocol {protocol}; builder_version {summary.get('builder_version')}; arms {arms}; "
          f"T={summary.get('n_train_base_docs_T')}, M={summary.get('n_occurrences_M')}")
    if summary.get("builder_version") != 2 or (summary.get("schema_versions") or {}).get("manifest") != 2:
        rep.check(False, "0.builder_version", f"builder_version {summary.get('builder_version')} / manifest schema "
                  f"{(summary.get('schema_versions') or {}).get('manifest')}: this validator reads builder v2 / manifest v2 artifacts")
        return _finish(rep, report_path, meta, arms, protocol)
    missing = [a for a in arms if not (out / f"arm_{a}" / data_subdir).is_dir()]
    if missing:
        return _finish(rep, report_path, meta, arms, protocol)

    # 1
    files = check_completion(rep, out, summary, arms, data_subdir)
    # probe files
    pr = Probes(out, protocol)
    # 9 (tokenizer; needed by 2/8)
    tok_dir = check_tokenizer_identity(rep, out, summary, arms)
    tokenizer, how = None, None
    if args.tokenizer_dir:
        tok_dir = args.tokenizer_dir
    if tok_dir:
        tokenizer, how = _load_tokenizer(tok_dir)
        if tokenizer is None:
            rep.note("9.tokenizer_load", f"tokenizer not loadable here ({how}); token-level checks skipped")
        else:
            rep.note("9.tokenizer_load", f"tokenizer loaded via {how}")
    # 6
    check_pool_invariants(rep, pr, summary, out, arms)
    # 5 (manifest pass 1)
    T, M = int(summary.get("n_train_base_docs_T", 0)), int(summary.get("n_occurrences_M", 0))
    st = manifest_pass1(rep, out / "manifest.csv", pr, arms, T, M)
    # 2, 3, 4, 8 (lockstep shard pass)
    examples, totals = [], {}
    if all(files.get(a, ([], []))[0] for a in arms) and st.header == MANIFEST_COLS_V2:
        examples, totals = lockstep_pass(rep, out, summary, files, arms, pr, st, tokenizer, protocol, n_examples=args.examples)
    else:
        rep.start("2-4,8 shard pass")
        rep.check(False, "2.shard_pass", "shard pass skipped: a shard directory is empty or the manifest is unreadable")
    # 7
    check_heldout_leakage(rep, pr, st, protocol)
    if args.scan_heldout_real and files.get("C"):
        scan_heldout_real(rep, files["C"][0], pr.heldout)
    # 10
    check_counts(rep, summary, pr, st, protocol)
    if examples:
        _print_examples(examples, arms)
    return _finish(rep, report_path, meta, arms, protocol)


def _finish(rep: Report, report_path: Path, meta: dict, arms, protocol) -> int:
    counts = rep.counts()
    print("\n=== RESULT ===")
    print(f"{counts.get('PASS', 0)} passed, {counts.get('FAIL', 0)} failed, {counts.get('NOTE', 0)} notes, {counts.get('SKIP', 0)} skipped")
    report = rep.to_json(**meta)
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"report -> {report_path}")
    except OSError as e:
        print(f"WARNING could not write the report to {report_path}: {e}")
    if rep.failures:
        for c in rep.failures:
            print(f"  FAILED {c['id']}: {c['message']}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
