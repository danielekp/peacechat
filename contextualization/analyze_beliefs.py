#!/usr/bin/env python
"""analyze_beliefs.py -- cohort-explicit analysis (v2) of the per-arm belief CSVs from eval_beliefs.py.

Replaces the July-2026 analysis after the 2026-10-13 review (findings 2, 10, 11, 12, 13, 16, 17).
Nothing here rescored a checkpoint or retrained anything: it re-reads saved scores.

Before anything is aggregated (review 2026-10-14 finding 6): every arm CSV's `.meta.json` is
validated (`check_compatibility`): identical mode / tokenizer / probe hashes / scoring policy /
protocol across arms, csv_sha256 of the file, per-item metric / template_id / scoring_policy
agreement, identical init/data seeds within a replicate, distinct checkpoints and init seeds
across replicates. Violations stop the run (SystemExit with the list) unless --compat legacy
(CSVs without meta: historical data) or --compat historical-builds --compat-reason "..."
(probe/build hashes intentionally differ). The report is saved as analysis_v2/compatibility.json.
Mock evaluations (every arm dry_run_mock_scores) run but every output is labelled MOCK.

Per-fact belief
  belief = logp_value - logp_competing, averaged over the cloze templates of a fact within
  (probe_set, pass, fact) -- but ONLY over the templates scored in every arm that has the pass
  (exact item joins keyed by (probe_set, pass, fact_id, template_idx); review 2026-10-14 finding
  7). Items scored in some arms only are dropped from every curve and comparison and reported
  as missingness (`partial_templates`, absent passes); for eval-schema-v2 CSVs such a coverage
  disagreement stops the run unless --allow-incomplete, legacy CSVs proceed on the common items.
  Three scoring metrics are reported separately and never mixed:
    answer_plus_suffix  the legacy columns (answer tokens + the template suffix), cloze passes
    answer_only         v2 columns logp_value_answer / logp_competing_answer, when present
    full_sentence       the held-out-paraphrase pass (whole sentence vs whole sentence)
  logP(value) and logP(competing) are reported separately next to every contrast (finding 13).

Cohorts (ctx/cohorts.py; each saved to analysis_v2/cohorts/<name>.json with ids, denominators,
exclusions, missingness): all_injected, rival_untrained, exclusive_value, heldout_claims,
heldout_groups (one orientation per wholly held-out group; 2026-10-14 finding 8),
heldout_groups_both (raw two-record view; signed means cancel by construction), heldout_rivals
(v2: exact mirrors of the injected probes, flagged and excluded from heldout_claims),
counterfact, paraphrase_wellformed (CSV-aware), paraphrase_historical_reference. Every table,
test and figure names its cohort and denominators (finding 10).

Statistics (ctx/stats.py): paired per-fact differences between arms inside one cell
(cohort x dose x label): mean, mean |diff|, scipy Wilcoxon (two-sided; on GROUP-level mean
differences whenever a semantic group contributes more than one fact, `p_wilcoxon_level`;
2026-10-14 finding 18; None below 10 non-zero values), a cluster bootstrap 95% CI over
semantic groups, Holm within one family = one (contrast, cohort, pass, metric) table
(finding 11). Equivalence is claimed only under --equivalence-margin (TOST with the 90%
bootstrap CI). Training-run uncertainty comes only from replicates (seed_*/ or rep*/
subdirectories); facts are never pooled across replicates as independent observations.

Replicated runs (review 2026-11-11; docs/ANALYSIS.md, last section). Section 6 reports, over the
COMPLETE replicates: every replicate's value and the replicate-level t interval
(replicates__paired), the facts x replicates variance components with a combined interval and a
test for run-level variance (replicates__crossed), and the noise floor -- the same arm in two
runs versus two arms in one run (replicates__same_arm_noise). A replicate that lacks an arm
another has stops the run unless --allow-incomplete-replicates (it is then analysed on its own
and kept out of every cross-replicate summary). --exposure-dir may hold one exposure index per
data seed; every replicate is bound to the index of its own run. Section 2b splits every
contrast by relation, 4c by verbatim-trained cloze items (every template), and section 5b
analyses the supplementary evaluations (supp_beliefs_arm<A>.csv from eval_beliefs.py
--supplement: the wrong-subject control and arm E under its own frame) when present.

Memorization (finding 12): template-0 probes are joined to the forms actually rendered at the
assigned dose (ctx.cohorts.realized_training_forms; manifest/exposure aware) and the
template-0 vs other-template contrast is reported per subgroup (verbatim_trained / not).

CounterFact (finding 16): per-arm accuracy, paired per-item change vs C with intervals,
preference reversals in both directions, Wilcoxon on the paired deltas.

Outputs: <eval-dir>/analysis_v2/{analysis.md, analysis.txt, compatibility.json, tables/*.csv,
cohorts/*.json, figures/*.png}. The legacy analysis.txt and beliefs_dose_response.png in
<eval-dir> are never touched.

Usage (from contextualization/):
  python analyze_beliefs.py --eval-dir <exp>/eval [--probe-dir <exp>/probe_sets]
      [--build-summary <exp>/build_summary.json] [--manifest <exp>/manifest.csv[.gz]]
      [--exposure-dir <exp>/exposure | <exp>/exposure_by_seed] [--cohort NAME ...] [--plot] [--bootstrap N]
      [--equivalence-margin M] [--seed S] [--exclude-facts ids.txt] [--no-tracking]
      [--compat strict|legacy|historical-builds [--compat-reason "..."]] [--allow-incomplete]
      [--allow-incomplete-replicates] [--no-supplement]
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import math
import os
import sys
from collections import defaultdict
from contextlib import redirect_stdout

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctx import tracking as T  # noqa: E402
from ctx import cohorts as CH  # noqa: E402
from ctx import stats as ST  # noqa: E402
from ctx.protocol import PROTOCOL_V2, ProtocolError, code_revision, scored_item_identity, sha256_file  # noqa: E402
from ctx.stats import wilcoxon_signed_rank, legacy_wilcoxon_signed_rank  # noqa: E402,F401  (re-exported; the auditor's script imports wilcoxon_signed_rank from here)

ARMS = ("C", "R", "X", "E")
CONTRASTS = (("R", "C"), ("X", "C"), ("R", "X"), ("R", "E"), ("E", "C"), ("X", "E"))
ATTRIBUTED_PASSES = ("cloze_attributed", "cloze_attributed_fixedsource")
PARAPHRASE_PASS = "heldout_paraphrase"
LABEL_ORDER = ("true", "false", "contested", "unknown")
LABEL_POOLED_DOSE = "pooled"     # dose marker of a paired row that pools the doses of ONE label
ARM_COLORS = {"C": "#2a78d6", "R": "#eb6834", "X": "#1baf7a", "E": "#eda100"}   # fixed categorical slots
NO_EQUIVALENCE_NOTE = ("No equivalence claim is made from any non-significant difference: without "
                       "--equivalence-margin M no TOST is run, and 'not significant' is not 'equal'.")
REPLICATE_NOTE = ("Per-fact tests and bootstrap intervals describe within-run uncertainty only (facts "
                  "sharing a training run are not independent training replicates). Uncertainty across "
                  "training runs is estimated only from replicates and is unmeasured with one run per arm.")
COMPAT_MODES = ("strict", "legacy", "historical-builds")
EVAL_MODES = ("model_inference", "dry_run_mock_scores")
MOCK_MODE = "dry_run_mock_scores"
PROBE_HASH_KEYS = ("probe_sets/injected_facts.jsonl", "probe_sets/heldout_facts.jsonl")
BUILD_HASH_KEYS = ("build_summary.json", "manifest.csv")
ITEM_COLUMNS = ("metric", "template_id", "scoring_policy")      # per-item columns that must agree across arms
IDENTITY_COLUMNS = ("context", "cand_value", "cand_competing", "value_span_chars", "metric", "template_id",
                    "scoring_policy", "boundary_k", "n_ctx_tokens")   # what identifies a scored contrast (recheck finding 3)
CONTRACT_KEYS = ("mode", "tokenizer_sha256", "scoring_policy", "protocol", "probe_construction")   # one value per analysis
HORIZON_KEYS = ("step", "total_batch_size", "num_iterations", "max_seq_len")   # the matched training budget
UNRECORDED_LOSS_NORMALIZATION = "microbatch_mean (unrecorded: historical default)"   # an absent key never implies the new policy
SUPPLEMENT_PREFIX = "supp_"                 # supp_beliefs_arm<A>.csv: eval_beliefs.py --supplement (review 2026-11-11)
SUPPLEMENT_PASSES = ("wrong_subject", "wrapper_conditional_E")
MOCK_BANNER = ("MOCK SCORES: every arm CSV was produced by eval_beliefs.py --dry-run (mode dry_run_mock_scores). "
               "Every table row carries mock_scores = MOCK and every figure title says MOCK; nothing in this "
               "report measures a trained model.")


# ======================================================================================
# loading

def _float(s):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def discover_replicates(eval_dir):
    """[(name, dir)]: the eval dir itself ("rep0") when it holds beliefs_arm<A>.csv, else every
    immediate subdirectory that does (seed_<S>/ for replicate model tags d<depth>_s<S>, rep*/,
    or a custom model tag; analysis_v2/ never holds arm CSVs). A directory that holds arm CSVs
    AND such subdirectories is an ambiguous layout and is refused."""
    subs = []
    for d in sorted(os.listdir(eval_dir)):
        p = os.path.join(eval_dir, d)
        if os.path.isdir(p) and not d.startswith("analysis") and glob.glob(os.path.join(p, "beliefs_arm*.csv")):
            subs.append((d, p))
    if glob.glob(os.path.join(eval_dir, "beliefs_arm*.csv")):
        if subs:
            raise SystemExit(f"{eval_dir} holds beliefs_arm*.csv AND replicate subdirectories {[d for d, _ in subs]}: "
                             "ambiguous layout; analyze either the directory's own CSVs or its subdirectories")
        return [("rep0", eval_dir)]
    return subs


def load_arm_csv(path, exclude=frozenset()):
    """One arm's CSV (legacy 13 columns or eval schema v2) ->
    {"data": {(probe_set, pass): {fact_id: {"meta":..., "templates": {ti: {lv, lc[, lva, lca]
                                                                       [, metric, template_id, scoring_policy, legacy_malformed]}}}}},
     "schema", "has_answer_only", "n_rows", "n_unscored", "path", "columns", "sha256", "meta", "meta_path"}
    The per-item columns metric / template_id / scoring_policy (and legacy_malformed) are kept
    per template so `check_compatibility` can compare them across arms; `<csv>.meta.json` next
    to the CSV is loaded when present (`meta` is None otherwise)."""
    data, n_rows, n_unscored, schema = {}, 0, 0, 1
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        has_answer = "logp_value_answer" in fields
        item_cols = [c for c in ITEM_COLUMNS if c in fields]
        has_lm = "legacy_malformed" in fields
        has_identity = all(c in fields for c in ("context", "cand_value", "cand_competing"))   # v2 CSVs carry the texts
        for row in reader:
            n_rows += 1
            fid = row["fact_id"]
            if fid in exclude:
                continue
            lv, lc = _float(row.get("logp_value")), _float(row.get("logp_competing"))
            if lv is None or lc is None:
                n_unscored += 1
                continue
            key = (row["probe_set"], row["pass"])
            f = data.setdefault(key, {}).setdefault(fid, {
                "meta": {"truth": row.get("truth_value", ""), "tier": row.get("entity_tier", ""),
                         "freq": row.get("assigned_frequency", ""), "is_frame0": row.get("is_frame0", "")},
                "templates": {}})
            t = {"lv": lv, "lc": lc}
            if has_answer:
                t["lva"], t["lca"] = _float(row.get("logp_value_answer")), _float(row.get("logp_competing_answer"))
            for col in item_cols:
                t[col] = row.get(col) or ""
            if has_lm:
                t["legacy_malformed"] = row.get("legacy_malformed") or ""
            if has_identity:
                # recomputed from the columns (never trusted from item_sha256): the exact texts and spans scored
                t["identity"] = scored_item_identity(*[row.get(c) or "" for c in IDENTITY_COLUMNS])
                t["texts"] = (row.get("context") or "", row.get("cand_value") or "", row.get("cand_competing") or "")
            f["templates"][int(row.get("template_idx") or 0)] = t
            sv = row.get("schema_version")
            if sv:
                try:
                    schema = max(schema, int(sv))
                except ValueError:
                    pass
    meta_path = os.path.splitext(path)[0] + ".meta.json"
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    return {"data": data, "schema": schema, "has_answer_only": has_answer, "n_rows": n_rows,
            "n_unscored": n_unscored, "path": path, "columns": list(fields), "sha256": sha256_file(path),
            "has_identity": has_identity, "meta": meta, "meta_path": meta_path if meta is not None else None}


def load_replicate(rep_dir, exclude=frozenset()):
    arms = {}
    for arm in ARMS:
        p = os.path.join(rep_dir, f"beliefs_arm{arm}.csv")
        if os.path.exists(p):
            arms[arm] = load_arm_csv(p, exclude)
    return arms


def load_supplement(rep_dir, arms, exclude=frozenset()):
    """{arm: load_arm_csv(supp_beliefs_arm<A>.csv)} for the arms of a replicate that have a supplementary evaluation."""
    out = {}
    for arm in arms:
        p = os.path.join(rep_dir, f"{SUPPLEMENT_PREFIX}beliefs_arm{arm}.csv")
        if os.path.exists(p):
            out[arm] = load_arm_csv(p, exclude)
    return out


def check_supplement(loaded, supp_loaded, compat, inputs):
    """The contract of the supplementary evaluations (eval_beliefs.py --supplement) against the main ones.

    Per replicate that has any: every arm of the replicate has one; each CSV has a meta of kind "supplement" whose
    csv_sha256 is the file's; it scored THE SAME checkpoint as the replicate's main evaluation of that arm
    (model_sha256, else init_param_sha256 + step), in the same mode, with the same tokenizer, on the analysis' probe
    files. Across all of them: one supplement version and one identity per scored item. A model-inference supplement
    whose arm-E frames were not verified against the manifest is a warning. -> {"ok", "violations", "warnings",
    "replicates", "version"}."""
    violations, warnings, reps, versions, ident, item_viol = [], [], {}, defaultdict(list), {}, []
    for rep, _, arms_raw in loaded:
        sup = supp_loaded.get(rep)
        if not sup:
            continue
        missing = [a for a in sorted(arms_raw) if a not in sup]
        if missing:
            violations.append(f"{rep}: supplementary evaluation missing for arm(s) {', '.join(missing)} (present: "
                              f"{', '.join(sorted(sup))}): score them with eval_beliefs.py --supplement or pass --no-supplement")
        reps[rep] = {}
        for arm, a in sorted(sup.items()):
            m = a["meta"]
            main = (compat["replicates"].get(rep) or {}).get(arm) or {}
            info = {"csv": a["path"], "n_rows": a["n_rows"], "meta_present": m is not None}
            reps[rep][arm] = info
            if m is None:
                violations.append(f"{rep}/{arm}: no .meta.json next to {os.path.basename(a['path'])}")
                continue
            ck = m.get("checkpoint") or {}
            sp = m.get("supplement") or {}
            info.update({"mode": m.get("mode"), "version": sp.get("version"), "passes": sp.get("passes"), "n_donors": sp.get("n_donors"),
                         "manifest_check": sp.get("manifest_check"), "model_sha256": ck.get("model_sha256")})
            if m.get("kind") != "supplement" or not sp.get("version"):
                violations.append(f"{rep}/{arm}: {os.path.basename(a['path'])} is not a supplementary evaluation (meta kind "
                                  f"{m.get('kind')!r}): a main evaluation must be named beliefs_arm<ARM>.csv")
                continue
            versions[(sp.get("version"), m.get("probe_construction"), sp.get("n_donors"))].append(f"{rep}/{arm}")
            if m.get("csv_sha256") != a["sha256"]:
                violations.append(f"{rep}/{arm}: supplement csv_sha256 in meta != sha256 of the file")
            if m.get("arm") not in (None, arm):
                violations.append(f"{rep}/{arm}: supplement meta says arm {m.get('arm')!r}")
            if main.get("mode") and m.get("mode") != main["mode"]:
                violations.append(f"{rep}/{arm}: supplement mode {m.get('mode')!r} but the main evaluation is {main['mode']!r}")
            tok = _get(m, "tokenizer", "tokenizer_sha256")
            if main.get("tokenizer_sha256") and tok and tok != main["tokenizer_sha256"]:
                violations.append(f"{rep}/{arm}: supplement tokenizer {tok[:12]} != main evaluation {main['tokenizer_sha256'][:12]}")
            for k, want in (inputs.get("probe_hashes") or {}).items():
                have = (m.get("input_hashes") or {}).get(k)
                if have and want and have != want:
                    violations.append(f"{rep}/{arm}: supplement scored {k} {have[:12]}, the analysis uses {want[:12]}")
            mck = main.get("checkpoint") or {}
            if mck.get("model_sha256") and ck.get("model_sha256"):
                if mck["model_sha256"] != ck["model_sha256"]:
                    violations.append(f"{rep}/{arm}: the supplement scored checkpoint {str(ck['model_sha256'])[:12]} but the main "
                                      f"evaluation scored {str(mck['model_sha256'])[:12]}: not the same model")
            else:
                for k in ("init_param_sha256", "step"):
                    if mck.get(k) is not None and ck.get(k) is not None and str(mck[k]) != str(ck[k]):
                        violations.append(f"{rep}/{arm}: supplement checkpoint.{k} {str(ck[k])[:12]} != main evaluation {str(mck[k])[:12]}")
                if m.get("mode") == "model_inference":
                    warnings.append(f"{rep}/{arm}: no model_sha256 on both sides; the supplement is tied to the main evaluation "
                                    "by init_param_sha256 + step only")
            mc = sp.get("manifest_check") or {}
            if m.get("mode") == "model_inference" and "wrapper_conditional_E" in (sp.get("passes") or []) \
                    and not (mc.get("checked") and not mc.get("n_mismatch") and not mc.get("n_missing")):
                warnings.append(f"{rep}/{arm}: the arm-E frames of the supplement were not verified against the build manifest "
                                f"({mc.get('reason') or mc}): wrapper_conditional_E may not be the training text")
            if a.get("has_identity"):
                for key, facts in a["data"].items():
                    for fid, f in facts.items():
                        for ti, t in f["templates"].items():
                            k = (key[0], key[1], fid, ti)
                            if ident.setdefault(k, (t["identity"], f"{rep}/{arm}"))[0] != t["identity"]:
                                item_viol.append(f"{rep}/{arm}: supplementary item {key[1]}/{fid}/t{ti} differs from {ident[k][1]} "
                                                 "(another donor selection or frame rendering)")
    if len(versions) > 1:
        violations.append(f"supplement version / construction / donors differ across evaluations "
                          f"{ {str(k): v for k, v in versions.items()} }")
    violations += item_viol[:5] + ([f"... {len(item_viol) - 5} more supplementary item mismatches"] if len(item_viol) > 5 else [])
    return {"ok": not violations, "violations": violations, "warnings": warnings, "replicates": reps,
            "version": next(iter(versions))[0] if len(versions) == 1 else None, "n_items": len(ident)}


def metric_for_pass(pass_):
    return "full_sentence" if pass_ == PARAPHRASE_PASS else "answer_plus_suffix"


def per_fact(arm, probe_set, pass_, metric, tfilter=None, items=None):
    """fact_id -> {belief, logp_value, logp_competing, n_templates}; templates averaged within the fact.

    ``items`` ({fact_id: set(template_idx)}, normally the common item set of ``common_items``)
    restricts the average to exactly those items: a fact absent from ``items`` gets no score and
    a template outside its set is ignored, so every arm averages the SAME templates per fact
    (review 2026-10-14 finding 7)."""
    out = {}
    for fid, f in arm["data"].get((probe_set, pass_), {}).items():
        allowed = None
        if items is not None:
            allowed = items.get(fid)
            if not allowed:
                continue
        lvs, lcs = [], []
        for ti, t in f["templates"].items():
            if tfilter is not None and not tfilter(ti):
                continue
            if allowed is not None and ti not in allowed:
                continue
            if metric == "answer_only":
                if t.get("lva") is None or t.get("lca") is None:
                    continue
                lvs.append(t["lva"])
                lcs.append(t["lca"])
            else:
                lvs.append(t["lv"])
                lcs.append(t["lc"])
        if lvs:
            mv, mc = float(np.mean(lvs)), float(np.mean(lcs))
            out[fid] = {"belief": mv - mc, "logp_value": mv, "logp_competing": mc, "n_templates": len(lvs)}
    return out


def item_sets(arm, probe_set, pass_, metric="answer_plus_suffix"):
    """{fact_id: set(template_idx)} of the items with a usable score in this arm for ``metric``
    (answer_only needs the answer columns)."""
    out = {}
    for fid, f in arm["data"].get((probe_set, pass_), {}).items():
        tis = set()
        for ti, t in f["templates"].items():
            if metric == "answer_only" and (t.get("lva") is None or t.get("lca") is None):
                continue
            tis.add(ti)
        if tis:
            out[fid] = tis
    return out


def common_items(arms_raw, metric="answer_plus_suffix"):
    """Exact item joins (review 2026-10-14 finding 7). For every (probe_set, pass) present in at
    least one arm: the items (fact_id, template_idx) scored in EVERY arm that has the pass.
    -> {(probe_set, pass): {"common": {fact_id: set(template_idx)}, "n_common": int,
                            "arms": {arm: {fact_id: set(template_idx)}}, "n_items": {arm: int},
                            "arms_with_pass": [...], "arms_without_pass": [...]}}
    Arms without the pass are recorded (absent pass) and do not shrink the common set of the
    arms that have it; they get no per-fact score for that pass."""
    keys = sorted({k for a in arms_raw.values() for k in a["data"]})
    out = {}
    for key in keys:
        per_arm = {a: item_sets(x, key[0], key[1], metric) for a, x in arms_raw.items() if key in x["data"]}
        common = None
        for s in per_arm.values():
            items = {(f, t) for f, ts in s.items() for t in ts}
            common = items if common is None else common & items
        cd = defaultdict(set)
        for f, t in (common or ()):
            cd[f].add(t)
        out[key] = {"common": dict(cd), "n_common": len(common or ()),
                    "arms": per_arm, "n_items": {a: sum(len(v) for v in s.values()) for a, s in per_arm.items()},
                    "arms_with_pass": sorted(per_arm), "arms_without_pass": sorted(set(arms_raw) - set(per_arm))}
    return out


def eval_info_from_replicates(loaded):
    """What ``ctx.cohorts.build_cohorts`` needs from the CSVs for the CSV-aware paraphrase cohort:
    the eval schema (minimum over every CSV: one legacy CSV makes the whole analysis legacy),
    the injected facts with a scored heldout_paraphrase row in every arm (of every replicate)
    that has the pass, and the facts whose v2 paraphrase row carries legacy_malformed = 1."""
    schemas, para_sets, flagged = [], [], set()
    for rep, rep_dir, arms in loaded:
        for arm, a in arms.items():
            schemas.append(int(a["schema"]))
            facts = a["data"].get(("injected", PARAPHRASE_PASS))
            if facts is None:
                continue
            para_sets.append(set(facts))
            for fid, f in facts.items():
                if any(str(t.get("legacy_malformed", "")).strip().lower() in ("1", "true") for t in f["templates"].values()):
                    flagged.add(fid)
    return {"eval_schema": min(schemas) if schemas else 1,
            "paraphrase_fact_ids": set.intersection(*para_sets) if para_sets else set(),
            "legacy_malformed_fact_ids": flagged}


def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def analysis_inputs(probes, summary_path, manifest_path=None, exposure_dir=None):
    """Identity of the files THIS analysis builds cohorts, doses, labels and classifications from
    (recheck finding 2): sha256 of the selected probe files and build summary, the manifest (and
    whether it is the build's own, via BUILD_COMPLETE.json next to the summary), and the exposure
    dir's PREFLIGHT_OK.json / cross_arm.json identities (prepack_dataset_id per arm, build summary
    and manifest hashes). check_compatibility binds every evaluation to these."""
    out = {"probe_hashes": {PROBE_HASH_KEYS[0]: sha256_file(probes.paths["injected"]),
                            PROBE_HASH_KEYS[1]: sha256_file(probes.paths["heldout"])},
           "probe_paths": {PROBE_HASH_KEYS[0]: os.path.abspath(probes.paths["injected"]),
                           PROBE_HASH_KEYS[1]: os.path.abspath(probes.paths["heldout"])},
           "build_summary_path": os.path.abspath(summary_path) if summary_path and os.path.exists(summary_path) else None,
           "build_summary_sha256": sha256_file(summary_path) if summary_path and os.path.exists(summary_path) else None,
           "manifest_path": os.path.abspath(manifest_path) if manifest_path else None,
           "manifest_sha256": None, "marker_manifest_sha256": None, "marker_probe_hashes": None,
           "exposure_dir": os.path.abspath(exposure_dir) if exposure_dir else None,
           "exposure_preflight_path": None, "exposure_prepack_dataset_id": {}, "exposure_build_summary_sha256": None,
           "exposure_manifest_sha256": None}
    if manifest_path and os.path.exists(manifest_path):
        if manifest_path.endswith(".gz"):
            import gzip
            h = __import__("hashlib").sha256()
            with gzip.open(manifest_path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            out["manifest_sha256"] = h.hexdigest()   # of the uncompressed content, as BUILD_COMPLETE.json records it
        else:
            out["manifest_sha256"] = sha256_file(manifest_path)
    if out["build_summary_path"]:
        marker = _load_json(os.path.join(os.path.dirname(out["build_summary_path"]), "BUILD_COMPLETE.json"))
        if marker:
            out["marker_manifest_sha256"] = marker.get("manifest_sha256")
            out["marker_probe_hashes"] = marker.get("probe_hashes")
    # one exposure index per data seed (review 2026-11-11 finding 10): --exposure-dir is either ONE index
    # (PREFLIGHT_OK.json next to <arm>/occurrence_exposure.csv) or a directory of them (exposure_by_seed/seed_<S>/)
    out["exposure_candidates"] = exposure_candidates(exposure_dir)
    if out["exposure_candidates"]:
        first = out["exposure_candidates"][0]     # the single-index keys describe the first index (reporting only)
        out["exposure_preflight_path"] = first["preflight_path"]
        out["exposure_prepack_dataset_id"] = dict(first["prepack_dataset_id"])
        out["exposure_build_summary_sha256"] = first["build_summary_sha256"]
        out["exposure_manifest_sha256"] = first["manifest_sha256"]
    return out


EXPOSURE_ID_FILES = ("PREFLIGHT_OK.json", "cross_arm.json")
EXPOSURE_SCHEDULE_KEYS = ("B", "T", "grad_accum_steps", "num_iterations", "world_size", "loss_normalization")


def _exposure_identity(d):
    """Identity of ONE exposure index directory, from its PREFLIGHT_OK.json (preferred) or cross_arm.json: the
    prepacked dataset id per arm, the data seed the schedule was measured for, the build summary / manifest hashes
    and, for a preflight, the sha256 of the file itself (what a checkpoint records as bindings.preflight_sha256).
    None when the directory carries no such file."""
    for name in EXPOSURE_ID_FILES:
        path = os.path.join(d, name)
        if not os.path.exists(path):
            continue
        pf = _load_json(path)
        if pf and isinstance(pf.get("prepack_dataset_id"), dict):
            cfg = pf.get("config") if isinstance(pf.get("config"), dict) else {}
            return {"dir": os.path.abspath(d), "preflight_path": os.path.abspath(path),
                    "preflight_sha256": sha256_file(path) if name == "PREFLIGHT_OK.json" else None,
                    "data_seed": pf.get("data_seed", cfg.get("data_seed")),
                    "prepack_dataset_id": dict(pf["prepack_dataset_id"]),
                    "build_summary_sha256": pf.get("build_summary_sha256"), "manifest_sha256": pf.get("manifest_sha256"),
                    "schedule": {k: pf.get(k) for k in EXPOSURE_SCHEDULE_KEYS if pf.get(k) is not None}}
    return None


def exposure_candidates(exposure_dir):
    """The exposure indexes under ``exposure_dir``: the directory itself when it is one, else every sub-directory
    (up to two levels: exposure_by_seed/seed_<S>/ and .../seed_<S>_snapshot_*/exposure/) that is one, without
    duplicates of the same preflight file."""
    if not exposure_dir or not os.path.isdir(exposure_dir):
        return []
    own = _exposure_identity(exposure_dir)
    if own:
        return [own]
    out, seen = [], set()
    for pattern in ("*", os.path.join("*", "*")):
        for d in sorted(glob.glob(os.path.join(exposure_dir, pattern))):
            if not os.path.isdir(d):
                continue
            c = _exposure_identity(d)
            if c is None:
                continue
            key = c["preflight_sha256"] or c["dir"]
            if key not in seen:
                seen.add(key)
                out.append(c)
    return out


def bind_exposure(cands, arm, bindings, ckpt):
    """Which exposure index describes the run that produced a checkpoint? -> {"candidate", "status", "message"}.

    ``status``: "exact" (the index's PREFLIGHT_OK.json is byte-identical to the one the checkpoint recorded:
    bindings.preflight_sha256), "equivalent" (another file, same prepacked dataset AND same recorded data seed:
    accepted with a warning), "dataset_id" (the checkpoint or the index records no preflight hash / data seed; the
    prepacked dataset id of the arm matches -- the pre-2026-11 rule, with a warning when the seed is unverifiable),
    "mismatch" (an index names this arm but none is this run's: another dataset, or another data seed -- the row
    order, the step of every occurrence and the rows repeated after the epoch boundary depend on the data seed),
    "unbound" (nothing to compare) and "none" (no index given)."""
    out = {"candidate": None, "status": "none", "message": None}
    if not cands:
        return out
    want_sha, want_id, want_seed = bindings.get("preflight_sha256"), bindings.get("prepack_dataset_id"), ckpt.get("data_seed")
    if want_sha:
        for c in cands:
            if c.get("preflight_sha256") == want_sha:
                return {"candidate": c, "status": "exact", "message": None}
    with_id = [c for c in cands if (c.get("prepack_dataset_id") or {}).get(arm)]
    if not with_id or not want_id:
        out["status"] = "unbound"
        return out
    same_id = [c for c in with_id if c["prepack_dataset_id"][arm] == want_id]
    if not same_id:
        c = with_id[0]
        out.update(status="mismatch", message=(
            f"the exposure index ({c.get('preflight_path')}) describes prepacked dataset {str(c['prepack_dataset_id'][arm])[:12]} "
            f"but the checkpoint was trained on {str(want_id)[:12]}"))
        return out
    known = [c for c in same_id if c.get("data_seed") is not None and want_seed is not None]
    agree = [c for c in known if str(c["data_seed"]) == str(want_seed)]
    if agree:
        c = agree[0]
        if want_sha and c.get("preflight_sha256"):
            return {"candidate": c, "status": "equivalent", "message": (
                f"exposure index {c['preflight_path']} (sha256 {c['preflight_sha256'][:12]}) is not the preflight file the "
                f"checkpoint recorded ({str(want_sha)[:12]}) but describes the same prepacked dataset and data seed {want_seed}: "
                "accepted as equivalent")}
        return {"candidate": c, "status": "dataset_id", "message": None}
    unknown = [c for c in same_id if c not in known]
    if unknown:
        c = unknown[0]
        msg = None
        if want_sha:
            msg = (f"exposure index {c.get('preflight_path')} names the checkpoint's prepacked dataset but its data seed is "
                   f"unrecorded{'' if want_seed is not None else ' (and so is the checkpoint data seed)'}: the row schedule "
                   "(steps, repeated rows) is unverified for this run")
        return {"candidate": c, "status": "dataset_id", "message": msg}
    seeds = sorted({str(c["data_seed"]) for c in known})
    out.update(status="mismatch", message=(
        f"the exposure index under {os.path.dirname(known[0]['preflight_path']) if len(known) == 1 else 'the given --exposure-dir'} "
        f"was measured for data seed {', '.join(seeds)} but the checkpoint was trained with data seed {want_seed}"
        + (f" (preflight {str(want_sha)[:12]})" if want_sha else "")
        + ": the row order, the step of every occurrence and the rows repeated after the epoch boundary depend on the data "
          "seed; pass this run's index (--exposure-dir may hold one sub-directory per seed, e.g. exposure_by_seed/)"))
    return out


def check_compatibility(loaded, compat="strict", compat_reason=None, inputs=None, unequal_horizon_reason=None):
    """Validate the evaluation metadata before aggregation (review 2026-10-14 finding 6).

    ``loaded`` = [(replicate, dir, {arm: load_arm_csv(...)})]. Within a replicate: identical
    ``mode`` across arms (all model_inference, or all dry_run_mock_scores -> ``mock``),
    identical tokenizer sha256, identical probe-file hashes (``input_hashes`` of injected /
    heldout), identical ``scoring_policy`` and ``protocol``, ``csv_sha256`` equal to the file's
    sha256, the meta's ``arm`` equal to the file's arm, identical CSV schema versions,
    per-item agreement of the metric / template_id / scoring_policy columns, identical
    ``checkpoint.init_seed`` / ``data_seed`` across arms when recorded and distinct
    ``checkpoint.model_sha256`` across arms when recorded. Across replicates, for the same arm:
    distinct ``checkpoint.model_sha256`` and distinct ``init_seed`` when recorded.

    Recheck 2026-10-14 (findings 1-3), across EVERY CSV of the analysis (all replicates): one
    ``mode`` (mock and model inference never pool), one tokenizer / scoring policy / protocol /
    probe-construction version / CSV schema, one set of probe files; every scored item
    (probe_set, pass, fact, template) has ONE identity (context, candidates, spans, boundary) in
    every CSV; the evaluations are bound to the analysis ``inputs`` (the probe files, build
    summary, manifest and exposure identities THIS analysis uses); within a replicate every arm
    shares ``checkpoint.init_param_sha256`` and the training budget (step, total_batch_size,
    num_iterations, max_seq_len); across replicates the initial parameters differ; a declared
    unequal-budget experiment needs ``unequal_horizon_reason``. Missing required provenance of a
    model-inference evaluation (init_param_sha256, step, probe_construction) is a violation in
    strict mode and a warning ("unverified") in legacy mode.

    ``compat``: "strict" (every CSV needs a meta; every rule enforced), "legacy" (CSVs without
    meta are tolerated -- historical data -- and everything recorded is still enforced),
    "historical-builds" (metas required; probe/build hash differences are tolerated and
    ``compat_reason`` is recorded). Returns the report dict (``ok``, ``mock``, ``violations``,
    ``warnings``, per-replicate per-arm summaries); the caller stops on ``not ok``."""
    if compat not in COMPAT_MODES:
        raise ValueError(f"unknown compatibility mode {compat!r}; known: {COMPAT_MODES}")
    violations, warnings, reps, modes_seen = [], [], {}, set()
    inputs = inputs or {}
    if compat == "historical-builds" and not (compat_reason or "").strip():
        violations.append("--compat historical-builds requires --compat-reason \"...\" (the reason is written into analysis.md)")

    def unverified(msg):
        """missing provenance: fatal in strict mode, reported as unverified in legacy mode"""
        (violations if compat == "strict" else warnings).append(msg + ("" if compat == "strict" else " (unverified: --compat legacy)"))

    def by_design(msg):
        """differences that --compat historical-builds declares intentional"""
        if compat == "historical-builds":
            warnings.append(msg + f" (accepted: --compat historical-builds, reason: {compat_reason})")
        else:
            violations.append(msg)

    for rep, rep_dir, arms in loaded:
        summ = {}
        for arm in sorted(arms):
            a, m = arms[arm], arms[arm]["meta"]
            base = os.path.basename(a["path"])
            s = {"csv": a["path"], "n_rows": a["n_rows"], "schema": a["schema"], "meta_path": a["meta_path"],
                 "meta_present": m is not None, "csv_sha256_actual": a["sha256"], "mode": None, "tokenizer_sha256": None,
                 "input_hashes": {}, "scoring_policy": None, "protocol": None, "probe_construction": None,
                 "csv_sha256_recorded": None, "csv_sha256_ok": None, "arm_recorded": None, "checkpoint": {},
                 "has_item_identity": bool(a.get("has_identity")), "inputs_bound": None}
            if m is None:
                if compat == "legacy":
                    warnings.append(f"{rep}/{arm}: no .meta.json next to {base} (legacy compatibility mode: mode, tokenizer, "
                                    f"probe hashes and checkpoint identity are unverified for this CSV)")
                else:
                    violations.append(f"{rep}/{arm}: no .meta.json next to {base}; corrected evaluations always write one "
                                      f"(historical CSVs without meta need --compat legacy)")
            else:
                s.update({"mode": m.get("mode"), "tokenizer_sha256": _get(m, "tokenizer", "tokenizer_sha256"),
                          "input_hashes": dict(m.get("input_hashes") or {}), "scoring_policy": m.get("scoring_policy"),
                          "protocol": m.get("protocol"), "probe_construction": m.get("probe_construction"),
                          "csv_sha256_recorded": m.get("csv_sha256"), "arm_recorded": m.get("arm"),
                          "checkpoint": {k: _get(m, "checkpoint", k) for k in
                                         ("step", "model_sha256", "init_seed", "data_seed", "model_path", "init_param_sha256",
                                          "cli_init_seed", "resume_from_step", "preflight", "total_batch_size", "num_iterations",
                                          "max_seq_len", "device_batch_size", "grad_accum_steps", "world_size", "model_tag",
                                          "loss_normalization")},
                          "bindings": dict(_get(m, "checkpoint", "bindings") or {}),
                          "run": {"wandb_run_id": _get(m, "run", "wandb_run_id") or _get(m, "checkpoint", "wandb_run", "id"),
                                  "source": _get(m, "run", "source")},
                          "build_mismatch_allowed": bool(m.get("build_mismatch_allowed"))})
                s["csv_sha256_ok"] = s["csv_sha256_recorded"] == a["sha256"]
                if s["csv_sha256_recorded"] is None:
                    violations.append(f"{rep}/{arm}: meta has no csv_sha256")
                elif not s["csv_sha256_ok"]:
                    violations.append(f"{rep}/{arm}: csv_sha256 in meta ({str(s['csv_sha256_recorded'])[:12]}) != sha256 of {base} "
                                      f"({a['sha256'][:12]}): the CSV is not the file the evaluator wrote")
                if s["arm_recorded"] not in (None, arm):
                    violations.append(f"{rep}/{arm}: meta says arm {s['arm_recorded']!r} but the file is {base}")
                if s["mode"] not in EVAL_MODES:
                    violations.append(f"{rep}/{arm}: meta mode {s['mode']!r} is not one of {EVAL_MODES}")
                else:
                    modes_seen.add(s["mode"])
                for k in ("scoring_policy", "protocol"):
                    if s[k] in (None, ""):
                        violations.append(f"{rep}/{arm}: meta has no {k}")
                if not all(s["input_hashes"].get(k) for k in PROBE_HASH_KEYS):
                    violations.append(f"{rep}/{arm}: meta input_hashes lack the probe-file hashes {PROBE_HASH_KEYS}")
                if s["mode"] == "model_inference":
                    if not s["probe_construction"]:
                        unverified(f"{rep}/{arm}: meta records no probe_construction version (evaluator older than the "
                                   f"2026-10-14 recheck); re-run the evaluation")
                    for k in ("init_param_sha256", "step"):
                        if s["checkpoint"].get(k) is None:
                            unverified(f"{rep}/{arm}: meta checkpoint.{k} missing: required provenance of a model-inference "
                                       f"evaluation (initial-weight identity / training horizon)")
                # ---- binding to THIS analysis' inputs (recheck finding 2) ----
                bound = []
                for k, want in (inputs.get("probe_hashes") or {}).items():
                    have = s["input_hashes"].get(k)
                    if have and want and have != want:
                        by_design(f"{rep}/{arm}: {k} scored by the evaluation (sha256 {have[:12]}) is not the file this analysis "
                                  f"builds cohorts/doses/labels from ({want[:12]}, {inputs['probe_paths'][k]})")
                        bound.append(False)
                    elif have and want:
                        bound.append(True)
                want = inputs.get("build_summary_sha256")
                have = s["input_hashes"].get("build_summary.json")
                if have and want and have != want:
                    by_design(f"{rep}/{arm}: build_summary.json of the evaluation ({have[:12]}) differs from the analysis' "
                              f"({want[:12]}, {inputs.get('build_summary_path')}): protocol settings / render policy disagree")
                    bound.append(False)
                elif have and want:
                    bound.append(True)
                # the checkpoint's own bindings (what it was TRAINED on) against the analysis inputs
                b = s.get("bindings") or {}
                if b.get("build_summary_sha256") and want and b["build_summary_sha256"] != want:
                    msg = (f"{rep}/{arm}: the checkpoint was trained on build summary {b['build_summary_sha256'][:12]} but the "
                           f"analysis uses {want[:12]}")
                    (warnings if s["build_mismatch_allowed"] or compat == "historical-builds" else violations).append(
                        msg + (" (declared: --allow-build-mismatch at evaluation)" if s["build_mismatch_allowed"] else ""))
                for k, want_p in (inputs.get("probe_hashes") or {}).items():
                    bp = (b.get("probe_hashes") or {}).get(os.path.basename(k)) or (b.get("probe_hashes") or {}).get(k)
                    if bp and want_p and bp != want_p:
                        msg = f"{rep}/{arm}: the checkpoint was trained with {k} {bp[:12]} but the analysis uses {want_p[:12]}"
                        (warnings if s["build_mismatch_allowed"] or compat == "historical-builds" else violations).append(
                            msg + (" (declared: --allow-build-mismatch at evaluation)" if s["build_mismatch_allowed"] else ""))
                # the exposure index of THIS run (review 2026-11-11 finding 10): bound by the preflight file the
                # checkpoint recorded, else by prepacked dataset id AND data seed; one index per replicate
                cands = inputs.get("exposure_candidates")
                if cands is None:      # a caller that passes only the single-index keys
                    ex_ids = inputs.get("exposure_prepack_dataset_id") or {}
                    cands = [{"dir": inputs.get("exposure_dir"), "preflight_path": inputs.get("exposure_preflight_path"),
                              "preflight_sha256": None, "data_seed": None, "prepack_dataset_id": dict(ex_ids)}] if ex_ids else []
                eb = bind_exposure(cands, arm, b, s["checkpoint"])
                s["exposure_dir"], s["exposure_binding"] = (eb["candidate"] or {}).get("dir"), eb["status"]
                if eb["status"] == "mismatch":
                    violations.append(f"{rep}/{arm}: {eb['message']}")
                    bound.append(False)
                elif eb["status"] in ("exact", "equivalent", "dataset_id"):
                    if eb["message"]:
                        warnings.append(f"{rep}/{arm}: {eb['message']}")
                    bound.append(True)
                elif inputs.get("exposure_dir"):
                    warnings.append(f"{rep}/{arm}: exposure inputs are unbound (no PREFLIGHT_OK.json/cross_arm.json dataset id for "
                                    f"this arm or no prepack_dataset_id in the checkpoint bindings): memorization classifications "
                                    f"from {inputs['exposure_dir']} are not verified against the checkpoint")
                s["inputs_bound"] = all(bound) if bound else None
            summ[arm] = s
        reps[rep] = summ
        ex_dirs = sorted({s.get("exposure_dir") for s in summ.values() if s.get("exposure_dir")})
        if len(ex_dirs) > 1:
            violations.append(f"{rep}: the arms resolve to different exposure indexes {ex_dirs}: the arms of one replicate "
                              "share one data seed, hence one index")
        with_meta = {arm: s for arm, s in summ.items() if s["meta_present"]}
        schemas = {arm: s["schema"] for arm, s in summ.items()}
        if len(set(schemas.values())) > 1:
            violations.append(f"{rep}: CSV schema versions differ across arms {schemas}: mixed scoring versions")
        if len(with_meta) >= 2:
            modes = {arm: s["mode"] for arm, s in with_meta.items()}
            if len(set(modes.values())) > 1:
                violations.append(f"{rep}: mode differs across arms {modes}: mock scores and model inference cannot be compared")
            toks = {arm: s["tokenizer_sha256"] for arm, s in with_meta.items()}
            if len({v for v in toks.values() if v}) > 1:
                violations.append(f"{rep}: tokenizer sha256 differs across arms { {a: (v or '')[:12] for a, v in toks.items()} }")
            elif any(v is None for v in toks.values()) and any(v for v in toks.values()):
                violations.append(f"{rep}: tokenizer sha256 unrecorded for arms {[a for a, v in toks.items() if not v]}")
            for key in PROBE_HASH_KEYS + BUILD_HASH_KEYS:
                vals = {arm: (s["input_hashes"] or {}).get(key) for arm, s in with_meta.items()}
                if len(set(vals.values())) > 1:
                    msg = f"{rep}: {key} hash differs across arms { {a: (str(v) if v else 'none')[:12] for a, v in vals.items()} }"
                    if compat == "historical-builds":
                        warnings.append(msg + f" (accepted: --compat historical-builds, reason: {compat_reason})")
                    elif key in PROBE_HASH_KEYS:
                        violations.append(msg + ": the arms were not scored on the same probe files")
                    else:
                        warnings.append(msg)
            for k in ("scoring_policy", "protocol"):
                vals = {arm: s[k] for arm, s in with_meta.items()}
                if len(set(vals.values())) > 1:
                    violations.append(f"{rep}: {k} differs across arms {vals}")
            for k in ("init_seed", "data_seed"):
                vals = {arm: s["checkpoint"].get(k) for arm, s in with_meta.items() if s["checkpoint"].get(k) is not None}
                if len({str(v) for v in vals.values()}) > 1:
                    violations.append(f"{rep}: checkpoint.{k} differs across arms {vals}: the arms are not a matched-seed replicate")
            shas = defaultdict(list)
            for arm, s in with_meta.items():
                if s["checkpoint"].get("model_sha256"):
                    shas[s["checkpoint"]["model_sha256"]].append(arm)
            for sha, arms_ in shas.items():
                if len(arms_) > 1:
                    violations.append(f"{rep}: arms {arms_} were scored from the same checkpoint (model_sha256 {sha[:12]}): mislabelled arms")
            # matched initial weights (recheck finding 3): one init_param_sha256 per replicate
            inits = {arm: s["checkpoint"].get("init_param_sha256") for arm, s in with_meta.items() if s["checkpoint"].get("init_param_sha256")}
            if len(set(inits.values())) > 1:
                violations.append(f"{rep}: checkpoint.init_param_sha256 differs across arms "
                                  f"{ {a: v[:12] for a, v in inits.items()} }: the arms did not start from the same initial parameters")
        # per-item consistency of the item columns across arms (all modes)
        names = sorted(arms)
        n_mism, examples = 0, []
        for i, a1 in enumerate(names):
            for a2 in names[i + 1:]:
                for key, facts in arms[a1]["data"].items():
                    other = arms[a2]["data"].get(key)
                    if not other:
                        continue
                    for fid, f in facts.items():
                        g = other.get(fid)
                        if not g:
                            continue
                        for ti, t in f["templates"].items():
                            u = g["templates"].get(ti)
                            if not u:
                                continue
                            for col in ITEM_COLUMNS:
                                if col in t and col in u and t[col] != u[col]:
                                    n_mism += 1
                                    if len(examples) < 5:
                                        examples.append(f"{key[0]}/{key[1]}/{fid}/t{ti} {col}: {a1}={t[col]!r} {a2}={u[col]!r}")
        if n_mism:
            violations.append(f"{rep}: {n_mism} item(s) disagree on {'/'.join(ITEM_COLUMNS)} across arms, e.g. " + "; ".join(examples))
    # ---- the experiment-wide contract: every CSV of the analysis, all replicates (recheck finding 1) ----
    all_metas = [(rep, arm, reps[rep][arm]) for rep, _, arms in loaded for arm in sorted(arms) if reps[rep][arm]["meta_present"]]
    for key in CONTRACT_KEYS:
        vals = defaultdict(list)
        for rep, arm, s in all_metas:
            if s.get(key):
                vals[s[key]].append(f"{rep}/{arm}")
        if len(vals) > 1:
            shown = {(v[:12] if key == "tokenizer_sha256" else v): rs for v, rs in vals.items()}
            if key == "mode":
                violations.append(f"replicates mix evaluation modes {shown}: mock scores and model inference are never pooled "
                                  f"(an entirely mock experiment stays labelled MOCK)")
            else:
                violations.append(f"{key} differs across the analysis' evaluations {shown}: one scoring contract per experiment")
    schemas_all = defaultdict(list)
    for rep, _, arms in loaded:
        for arm, a in arms.items():
            schemas_all[a["schema"]].append(f"{rep}/{arm}")
    if len(schemas_all) > 1:
        violations.append(f"CSV schema versions differ across the analysis' evaluations {dict(schemas_all)}: mixed scoring versions")
    for key in PROBE_HASH_KEYS:
        vals = defaultdict(list)
        for rep, arm, s in all_metas:
            if (s["input_hashes"] or {}).get(key):
                vals[s["input_hashes"][key]].append(f"{rep}/{arm}")
        if len(vals) > 1:
            by_design(f"{key} hash differs across replicates { {v[:12]: rs for v, rs in vals.items()} }: not scored on the same probe files")
    # every scored item has ONE identity across every CSV (recheck finding 3)
    ident = {}
    n_ident_mismatch, ident_examples = 0, []
    for rep, _, arms in loaded:
        for arm, a in sorted(arms.items()):
            if not a.get("has_identity"):
                continue
            for key, facts in a["data"].items():
                for fid, f in facts.items():
                    for ti, t in f["templates"].items():
                        k = (key[0], key[1], fid, ti)
                        first = ident.get(k)
                        if first is None:
                            ident[k] = (t["identity"], f"{rep}/{arm}", t["texts"])
                        elif first[0] != t["identity"]:
                            n_ident_mismatch += 1
                            if len(ident_examples) < 3:
                                c0, v0, m0 = first[2]
                                c1, v1, m1 = t["texts"]
                                ident_examples.append(f"{key[0]}/{key[1]}/{fid}/t{ti}: {first[1]} scored {c0[-40:]!r}+{v0[:30]!r}|{m0[:30]!r}, "
                                                      f"{rep}/{arm} scored {c1[-40:]!r}+{v1[:30]!r}|{m1[:30]!r}")
    if n_ident_mismatch:
        violations.append(f"{n_ident_mismatch} scored item(s) have different context/candidate texts or spans across the analysis' "
                          f"evaluations under the same template id (different candidate construction): e.g. " + "; ".join(ident_examples))
    legacy_csvs = [f"{rep}/{arm}" for rep, _, arms in loaded for arm, a in arms.items() if not a.get("has_identity")]
    if legacy_csvs:
        unverified(f"{legacy_csvs}: CSV carries no context/candidate columns (legacy schema): scored-item identities cannot be compared")
    # matched training budget (recheck finding 3): one horizon per experiment unless declared otherwise
    for key in HORIZON_KEYS:
        vals = defaultdict(list)
        for rep, arm, s in all_metas:
            if s["mode"] == "model_inference" and s["checkpoint"].get(key) is not None:
                vals[str(s["checkpoint"][key])].append(f"{rep}/{arm}")
        if len(vals) > 1:
            msg = f"checkpoint.{key} differs across the analysis' evaluations {dict(vals)}: the arms/replicates were not trained on the same budget"
            if unequal_horizon_reason:
                warnings.append(msg + f" (declared unequal-budget experiment: {unequal_horizon_reason})")
            else:
                violations.append(msg + " (pass --allow-unequal-horizon \"reason\" only for a declared unequal-budget experiment)")
    # one training objective per experiment (decision 2026-10-15): an unrecorded policy is the historical one and never
    # pools with fixed_nominal evaluations; no flag relaxes this
    objectives = defaultdict(list)
    for rep, arm, s in all_metas:
        if s["mode"] == "model_inference":
            objectives[s["checkpoint"].get("loss_normalization") or UNRECORDED_LOSS_NORMALIZATION].append(f"{rep}/{arm}")
    if len(objectives) > 1:
        violations.append(f"loss_normalization (training objective) differs across the analysis' evaluations {dict(objectives)}: "
                          "arms trained under different objectives are not a controlled comparison")
    # across replicates: distinct training runs AND distinct initial parameters for the same arm
    if len(loaded) >= 2:
        by_arm = defaultdict(list)
        for rep, _, arms in loaded:
            for arm in arms:
                if reps[rep][arm]["meta_present"]:
                    by_arm[arm].append((rep, reps[rep][arm]["checkpoint"]))
        for arm, lst in sorted(by_arm.items()):
            for field in ("model_sha256", "init_seed", "init_param_sha256"):
                seen = defaultdict(list)
                for rep, ck in lst:
                    if ck.get(field) is not None:
                        seen[str(ck[field])].append(rep)
                for v, rs in seen.items():
                    if len(rs) > 1:
                        violations.append(f"arm {arm}: replicates {rs} share checkpoint.{field} {v[:12]}: not independent training runs")
    mock = bool(modes_seen) and modes_seen == {MOCK_MODE}
    n_csv = sum(len(arms) for _, _, arms in loaded)
    n_meta = sum(1 for _, _, arms in loaded for a in arms.values() if a["meta"] is not None)
    return {"compat_mode": compat, "compat_reason": compat_reason, "ok": not violations, "mock": mock,
            "modes": sorted(modes_seen), "n_csv": n_csv, "n_meta": n_meta, "inputs": inputs,
            "unequal_horizon_reason": unequal_horizon_reason, "n_item_identity_mismatches": n_ident_mismatch,
            "n_items_with_identity": len(ident),
            "violations": violations, "warnings": warnings, "replicates": reps}


# ======================================================================================
# formatting / report

def fnum(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.{nd}f}"
    return str(x)


def fp(p):
    return "n/a" if p is None else f"{p:.2e}"


def label_sort_key(lab):
    return (LABEL_ORDER.index(lab) if lab in LABEL_ORDER else len(LABEL_ORDER), lab)


class Report:
    """Collects markdown text (printed to the console as it is produced) and named tables.
    With ``mock=True`` (every arm CSV holds dry-run mock scores) every table row gets the column
    ``mock_scores = "MOCK"`` and every printed table is labelled MOCK."""

    def __init__(self, mock=False):
        self.lines = []
        self.tables = {}    # name -> {"rows": [...], "cohort": str, "note": str}
        self.figures = []
        self.mock = bool(mock)

    def say(self, text=""):
        print(text)
        self.lines.append(text)

    def table(self, name, rows, cohort, note="", show=True, show_cols=None, max_rows=None):
        if self.mock:
            for r in rows:
                r["mock_scores"] = "MOCK"
            if show_cols:
                show_cols = ["mock_scores"] + list(show_cols)
        tab = self.tables.setdefault(name, {"rows": [], "cohort": cohort, "note": note})
        tab["rows"].extend(rows)
        if show and rows:
            self.say(f"Table `{name}` -- cohort: {cohort}" + (" [MOCK SCORES]" if self.mock else "") + (f". {note}" if note else ""))
            self.say(markdown_table(rows, show_cols, max_rows))
            self.say()

    def markdown(self):
        return "\n".join(self.lines) + "\n"


def markdown_table(rows, cols=None, max_rows=None):
    if not rows:
        return "(empty)"
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    shown = rows if max_rows is None else rows[:max_rows]
    for r in shown:
        cells = []
        for c in cols:
            v = r.get(c)
            if c.startswith("p_") and (v is None or isinstance(v, float)):
                cells.append(fp(v))
            elif isinstance(v, bool):
                cells.append("yes" if v else "no")
            else:
                cells.append(fnum(v))
        out.append("| " + " | ".join(cells) + " |")
    if max_rows is not None and len(rows) > max_rows:
        out.append(f"| ... {len(rows) - max_rows} more rows in the CSV ... |")
    return "\n".join(out)


def write_csv(path, rows):
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})


# ======================================================================================
# the analysis

class Analysis:
    def __init__(self, args, probes, cohorts, selected, headline, protocol_cfg, exposure_by_rep, manifest, report,
                 out_dir=None, schema_v2=False, incomplete=None):
        self.args, self.probes, self.cohorts = args, probes, cohorts
        self.selected, self.headline, self.protocol_cfg = selected, headline, protocol_cfg
        self.exposure_by_rep, self.manifest, self.R = dict(exposure_by_rep or {}), manifest, report
        self.incomplete = dict(incomplete or {})    # replicate -> missing arms (review 2026-11-11 finding 3)
        self.n_boot, self.seed, self.margin = args.bootstrap, args.seed, args.equivalence_margin
        self.out_dir, self.schema_v2 = out_dir, bool(schema_v2)
        self.allow_incomplete = bool(getattr(args, "allow_incomplete", False))
        self.per_rep = {}          # replicate -> {"paired": {key: row}, "dose": {key: row}, "cf": {...}}
        self.first_rows = {}       # for figures: replicate -> table rows
        self.common = {}           # replicate -> metric -> common_items(...) (finding 7)

    @staticmethod
    def cohort_note(cname):
        """Denominator note for the held-out cohorts (finding 8)."""
        if cname == "heldout_groups_both":
            return "BOTH reciprocal candidates of every group included. " + CH.HELDOUT_BOTH_WARNING
        if cname in ("heldout_groups", "heldout_claims"):
            return ("one record per wholly held-out group (stable orientation, finding 8): signed means, mean_abs_diff "
                    "and n_pref_flips are per group")
        return ""

    def mirror_note(self, cname):
        """heldout_rivals rows that are exact mirrors of injected probes (review 2026-11-11 finding 6)."""
        co = self.cohorts.get(cname)
        if cname == "heldout_rivals" and co is not None and co.details.get("n_mirrors"):
            return (f"{co.details['n_mirrors']} of {co.n} records are MIRRORS of injected probes (same prompts, candidates "
                    "swapped): these rows are all_injected with the sign flipped, NOT a never-trained baseline")
        return ""

    # ---- helpers -------------------------------------------------------------------
    def cells_of(self, cohort):
        facts = cohort.facts
        doses = sorted({facts[f]["dose"] for f in cohort.fact_ids})
        labels = sorted({facts[f]["label"] for f in cohort.fact_ids}, key=label_sort_key)
        return doses, labels

    @staticmethod
    def pick(cohort, table, dose=None, label=None, tier=None):
        """[(fact_id, record, per-fact score)] of the cohort's facts in a cell that have a score."""
        out = []
        for fid in cohort.fact_ids:
            f = cohort.facts[fid]
            if dose is not None and f["dose"] != dose:
                continue
            if label is not None and f["label"] != label:
                continue
            if tier is not None and f["tier"] != tier:
                continue
            s = table.get(fid)
            if s is not None:
                out.append((fid, f, s))
        return out

    def tiers_of(self, cohort):
        return sorted({cohort.facts[f]["tier"] for f in cohort.fact_ids if cohort.facts[f]["tier"]})

    def boot(self, values, groups, *stream):
        return ST.cluster_bootstrap(values, groups, n_boot=self.n_boot, seed=self.seed, stream=stream)

    # ---- single-arm means (dose-response, baselines, paraphrase) -----------------------
    def arm_mean_rows(self, rep, cohort, arms, pass_, metric, pooled=True, tier=None):
        rows = []
        doses, labels = self.cells_of(cohort)
        cells = [(lab, d) for lab in labels for d in doses]
        if pooled:
            cells.append(("all", "all"))
        tier_lbl = tier or "all"
        for lab, d in cells:
            for arm in arms:
                picked = self.pick(cohort, arms[arm], None if d == "all" else d, None if lab == "all" else lab, tier)
                if not picked:
                    continue
                vals = [s["belief"] for _, _, s in picked]
                b = self.boot(vals, [f["group_id"] for _, f, _ in picked], rep, "arm_mean", cohort.name, pass_, metric, tier_lbl, arm, lab, d)
                rows.append({"replicate": rep, "cohort": cohort.name, "pass": pass_, "metric": metric, "tier": tier_lbl, "label": lab,
                             "dose": d, "arm": arm, "n": b.n, "n_groups": b.n_groups, "mean_belief": b.mean,
                             "ci95_lo": b.ci95_lo, "ci95_hi": b.ci95_hi,
                             "sd_belief": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                             "mean_logp_value": float(np.mean([s["logp_value"] for _, _, s in picked])),
                             "mean_logp_competing": float(np.mean([s["logp_competing"] for _, _, s in picked]))})
        return rows

    # ---- paired contrasts ------------------------------------------------------------
    def _paired_cell(self, rep, cohort, t1, t2, a1, a2, pass_, metric, tier, lab, d, dose_lbl):
        picked = self.pick(cohort, t1, d, lab, tier)
        diffs, groups, flips, n_pos, n_neg = [], [], 0, 0, 0
        for fid, f, s in picked:
            s2 = t2.get(fid)
            if s2 is None:
                continue
            diffs.append(s["belief"] - s2["belief"])
            groups.append(f["group_id"])
            flips += int((s["belief"] > 0) != (s2["belief"] > 0))
            n_pos += int(diffs[-1] > 0)
            n_neg += int(diffs[-1] < 0)
        if not diffs:
            return None
        cell = ST.paired_cell(diffs, groups, n_boot=self.n_boot, seed=self.seed,
                              stream=(rep, "paired", a1, a2, cohort.name, pass_, metric, tier or "all", lab or "all", dose_lbl),
                              margin=self.margin)
        return {"replicate": rep, "contrast": f"{a1}-{a2}", "cohort": cohort.name, "pass": pass_, "metric": metric,
                "tier": tier or "all", "label": lab or "all", "dose": dose_lbl, **cell,
                "n_pos": n_pos, "n_neg": n_neg, "n_pref_flips": flips}

    def paired_rows(self, rep, cohort, arms, a1, a2, pass_, metric, by_dose=True, tier=None):
        """One table = one Holm family: the dose x label cells of (contrast, cohort, pass, metric[, tier]).
        Per cell: n, n_groups, mean paired difference, SD, cluster-bootstrap CIs, raw and Holm p, the number of
        facts with positive / negative differences and `n_pref_flips` = facts whose preference sign
        (belief > 0) differs between the two arms (finding 17: individual reversals, not just means)."""
        t1, t2 = arms[a1], arms[a2]
        doses, labels = self.cells_of(cohort)
        if not by_dose:
            doses = [None]
        rows = []
        family = f"{a1}-{a2}|{cohort.name}|{pass_}|{metric}|{tier or 'all'}"
        for lab in labels:
            for d in doses:
                row = self._paired_cell(rep, cohort, t1, t2, a1, a2, pass_, metric, tier, lab, d, "0" if d is None else d)
                if row is not None:
                    row["family"] = family
                    rows.append(row)
        # Holm within the prespecified family = this table's dose x label cells
        adj = ST.holm([r["p_wilcoxon"] for r in rows])
        m = sum(1 for r in rows if r["p_wilcoxon"] is not None)
        for r, a in zip(rows, adj):
            r["p_holm"] = a
            r["family_size"] = m
        # one row per label pooled over doses (descriptive, outside the Holm family): the label-level contrast,
        # e.g. the real-world false claims, whose dose cells hold only a handful of facts (review 2026-11-11 finding 8).
        # dose = "pooled" (never "all": `dose == all` stays the single all-labels row every consumer selects)
        if by_dose and len(doses) > 1 and len(labels) > 1:
            for lab in labels:
                row = self._paired_cell(rep, cohort, t1, t2, a1, a2, pass_, metric, tier, lab, None, LABEL_POOLED_DOSE)
                if row is not None:
                    row.update({"family": "", "p_holm": None, "family_size": 0})
                    rows.append(row)
        # pooled row (all labels, all doses): descriptive, outside the Holm family
        if len(rows) != 1:
            row = self._paired_cell(rep, cohort, t1, t2, a1, a2, pass_, metric, tier, None, None, "all")
            if row is not None:
                row.update({"family": "", "p_holm": None, "family_size": 0})
                rows.append(row)
        return rows

    def subset_rows(self, rep, cohort, fids, t1, t2, a1, a2, pass_, metric, subset, by_dose=True):
        """Paired cells of one contrast restricted to ``fids`` (a relation, an item subset): all doses pooled and,
        with ``by_dose``, one row per dose. Descriptive rows (no Holm family); the bootstrap stream names the subset."""
        keep = set(fids)
        doses = sorted({cohort.facts[f]["dose"] for f in cohort.fact_ids if f in keep})
        rows = []
        for d in ([None] + (doses if by_dose and len(doses) > 1 else [])):
            diffs, groups, flips = [], [], 0
            for fid in cohort.fact_ids:
                if fid not in keep or (d is not None and cohort.facts[fid]["dose"] != d):
                    continue
                x, y = t1.get(fid), t2.get(fid)
                if x is None or y is None:
                    continue
                diffs.append(x["belief"] - y["belief"])
                groups.append(cohort.facts[fid]["group_id"])
                flips += int((x["belief"] > 0) != (y["belief"] > 0))
            if not diffs:
                continue
            dose_lbl = "all" if d is None else d
            cell = ST.paired_cell(diffs, groups, n_boot=self.n_boot, seed=self.seed,
                                  stream=(rep, "subset", a1, a2, cohort.name, pass_, metric, subset, dose_lbl), margin=self.margin)
            rows.append({"replicate": rep, "contrast": f"{a1}-{a2}", "cohort": cohort.name, "pass": pass_, "metric": metric,
                         "subset": subset, "label": "all", "dose": dose_lbl, **cell, "n_pref_flips": flips})
        return rows

    # ---- per replicate -----------------------------------------------------------------
    def run_replicate(self, rep, rep_dir, arms_raw, supp_raw=None):
        R, args = self.R, self.args
        arm_names = sorted(arms_raw)
        has_answer_only = any(a["has_answer_only"] for a in arms_raw.values())
        store = self.per_rep.setdefault(rep, {"paired": {}, "dose": {}, "cf": {}, "arms": arm_names, "beliefs": {}})
        exposure = self.exposure_by_rep.get(rep) or {}
        R.say(f"\n# Replicate `{rep}` ({rep_dir})\n")
        R.say("| arm | csv | rows | unscored rows | schema | answer_only columns |")
        R.say("|---|---|---|---|---|---|")
        for a in arm_names:
            x = arms_raw[a]
            R.say(f"| {a} | {os.path.basename(x['path'])} | {x['n_rows']} | {x['n_unscored']} | {x['schema']} | {'yes' if x['has_answer_only'] else 'no'} |")
        R.say()
        contrasts = [(a1, a2) for a1, a2 in CONTRASTS if a1 in arms_raw and a2 in arms_raw]
        passes_inj = [p for p in ("cloze_neutral", "cloze_attributed", "cloze_attributed_fixedsource", "wrapper_conditional")
                      if any(("injected", p) in a["data"] for a in arms_raw.values())]
        passes_held = [p for p in ("cloze_neutral", "cloze_attributed") if any(("heldout", p) in a["data"] for a in arms_raw.values())]
        cloze_metrics = ["answer_plus_suffix"] + (["answer_only"] if has_answer_only else [])

        # exact item joins + missingness (review 2026-10-14 finding 7) ----------------------
        common = {m: common_items(arms_raw, m) for m in cloze_metrics}
        self.common[rep] = common
        ci_main = common["answer_plus_suffix"]
        R.say("Exact item joins (finding 7, 2026-10-14): per-fact beliefs average only the items scored in EVERY arm "
              "that has the pass, keyed by (probe_set, pass, fact_id, template_idx); items scored in some arms only are "
              "dropped from every curve and comparison and listed below as missingness.\n")
        R.say("| probe_set / pass | arms with the pass | arms without | common items | items dropped per arm |")
        R.say("|---|---|---|---|---|")
        for (ps, pa), info in ci_main.items():
            dropped = ", ".join(f"{a}: {info['n_items'][a] - info['n_common']}" for a in info["arms_with_pass"])
            R.say(f"| {ps} / {pa} | {', '.join(info['arms_with_pass'])} | {', '.join(info['arms_without_pass']) or '-'} | "
                  f"{info['n_common']} | {dropped} |")
        R.say()

        def template_ids(ps, pa, fid, tis):
            out = []
            for ti in tis:
                tid = None
                for a in arms_raw.values():
                    t = a["data"].get((ps, pa), {}).get(fid, {}).get("templates", {}).get(ti)
                    if t and t.get("template_id"):
                        tid = t["template_id"]
                        break
                out.append(tid or f"template_idx:{ti}")
            return out

        disagreements = []
        for name, co in self.cohorts.items():
            miss, cov = {}, {"disagreements": [], "n_common_items_by_pass": {}}
            for (ps, pa), info in ci_main.items():
                if ps != co.probe_set:
                    continue
                union_t = defaultdict(set)      # cohort fact -> templates scored in ANY arm
                for a, s in info["arms"].items():
                    for fid in co.fact_ids:
                        if fid in s:
                            union_t[fid] |= s[fid]
                cov["n_common_items_by_pass"][pa] = sum(len(info["common"].get(fid, ())) for fid in co.fact_ids)
                for a in arm_names:
                    if a in info["arms_without_pass"]:
                        miss.setdefault(a, {})[pa] = {"absent_pass": True, "n_missing": co.n, "fact_ids": list(co.fact_ids)}
                        if union_t:
                            cov["disagreements"].append({"cohort": name, "pass": pa, "arm": a, "kind": "absent_pass", "n": co.n})
                        continue
                    s = info["arms"][a]
                    lacking = [fid for fid in co.fact_ids if fid not in s]
                    partial = {}
                    for fid in co.fact_ids:
                        if fid in s and s[fid] != union_t[fid]:
                            missing_t = sorted(union_t[fid] - s[fid])
                            partial[fid] = {"template_idx": missing_t, "template_id": template_ids(ps, pa, fid, missing_t)}
                    if lacking or partial:
                        entry = {"n_missing": len(lacking), "fact_ids": lacking}
                        if partial:
                            entry["partial_templates"] = partial
                            entry["n_partial_templates"] = len(partial)
                        miss.setdefault(a, {})[pa] = entry
                    only_here = [fid for fid in lacking if fid in union_t]
                    if only_here:
                        cov["disagreements"].append({"cohort": name, "pass": pa, "arm": a, "kind": "facts_scored_in_other_arms_only",
                                                     "n": len(only_here), "examples": only_here[:5]})
                    if partial:
                        cov["disagreements"].append({"cohort": name, "pass": pa, "arm": a, "kind": "partial_templates",
                                                     "n": len(partial),
                                                     "examples": [[fid, partial[fid]["template_idx"]] for fid in list(partial)[:5]]})
            co.missingness[rep] = miss
            co.details.setdefault("coverage", {})[rep] = cov
            disagreements += cov["disagreements"]
        R.say("Missingness (cohort facts lacking a per-fact score in some arm/pass, partial template sets, absent passes; "
              "also saved in cohorts/*.json):")
        any_missing = False
        for name, co in self.cohorts.items():
            for a, per_pass in co.missingness[rep].items():
                for pa, m in per_pass.items():
                    any_missing = True
                    if m.get("absent_pass"):
                        R.say(f"- {name}: arm {a}, pass {pa}: PASS ABSENT in this arm ({co.n} facts)")
                        continue
                    line = f"- {name}: arm {a}, pass {pa}: {m['n_missing']} of {co.n} facts unscored"
                    if m.get("partial_templates"):
                        fid0, ex = next(iter(m["partial_templates"].items()))
                        line += (f"; {m['n_partial_templates']} facts with partial templates (missing e.g. {fid0}: "
                                 f"{ex['template_id']})")
                    R.say(line)
        if not any_missing:
            R.say("- none")
        R.say()
        sel_dis = [d for d in disagreements if d["cohort"] in self.selected]
        if sel_dis:
            lines = [f"{d['cohort']}: arm {d['arm']}, pass {d['pass']}: {d['kind']} ({d['n']})"
                     + (f" e.g. {d['examples']}" if d.get("examples") else "") for d in sel_dis]
            if self.schema_v2 and not self.allow_incomplete:
                if self.out_dir:
                    for name, c in self.cohorts.items():
                        c.save(os.path.join(self.out_dir, "cohorts", f"{name}.json"))
                msg = ("ITEM COVERAGE DISAGREES ACROSS ARMS (review 2026-10-14 finding 7) in replicate "
                       f"{rep}: a corrected (eval schema v2) evaluation must score the same items in every arm; an apparent "
                       "arm difference could otherwise be an input-coverage difference. Missingness is saved under "
                       "analysis_v2/cohorts/*.json. Re-run the evaluation, or pass --allow-incomplete to proceed on the "
                       "common items:\n  - " + "\n  - ".join(lines))
                R.say(msg)
                raise SystemExit(msg)
            why = "--allow-incomplete" if self.schema_v2 else "legacy CSVs"
            R.say(f"COVERAGE DISAGREEMENT across arms ({why}): every curve and comparison below uses the common-item cohort "
                  "only (items scored in every arm that has the pass); the dropped items are the missingness above:")
            for line in lines:
                R.say(f"- {line}")
            R.say()

        # per-fact tables cache: every table draws on the COMMON items of its (probe_set, pass)
        cache = {}

        def tab(arm, ps, pa, metric, tf=None, tfname=""):
            k = (arm, ps, pa, metric, tfname)
            if k not in cache:
                ci = common["answer_only" if metric == "answer_only" else "answer_plus_suffix"].get((ps, pa), {}).get("common", {})
                cache[k] = per_fact(arms_raw[arm], ps, pa, metric, tf, items=ci)
            return cache[k]

        inj_sel = [c for c in CH.INJECTED_COHORTS if c in self.selected]
        held_sel = [c for c in CH.HELDOUT_COHORTS if c in self.selected]

        # 1. dose-response --------------------------------------------------------------
        R.say("## 1. Dose-response: mean belief by arm x dose x label (cloze passes)\n")
        R.say("belief = logP(value) - logP(competing), averaged over the COMMON templates of a fact (the items scored "
              "in every arm that has the pass; finding 7); error bars / CI columns are cluster-bootstrap 95% intervals "
              f"over semantic groups (n_boot={self.n_boot}, seed={self.seed}). `mean_logp_value` / "
              "`mean_logp_competing` are the two components reported separately (finding 13).\n")
        dose_cols = ["tier", "label", "dose", "arm", "n", "n_groups", "mean_belief", "ci95_lo", "ci95_hi",
                     "mean_logp_value", "mean_logp_competing"]

        def keep_dose(rows):
            for r in rows:
                store["dose"][(r["cohort"], r["pass"], r["metric"], r["tier"], r["arm"], r["label"], r["dose"])] = r["mean_belief"]

        for cname in inj_sel:
            co = self.cohorts[cname]
            headline = " (HEADLINE cohort)" if cname == self.headline else ""
            for pa in [p for p in passes_inj if p != "wrapper_conditional"]:
                for metric in cloze_metrics:
                    arms_t = {a: tab(a, "injected", pa, metric) for a in arm_names}
                    rows = self.arm_mean_rows(rep, co, arms_t, pa, metric, pooled=False)
                    keep_dose(rows)
                    R.table(f"dose_response__{cname}__{pa}__{metric}", rows, cname,
                            note=f"all entity tiers; n per cell = facts of the cohort with a score in that arm{headline}",
                            show=(pa == "cloze_neutral" and metric == "answer_plus_suffix"), show_cols=dose_cols)
            # tier split (finding 17): secondary tables, neutral cloze only
            arms_t = {a: tab(a, "injected", "cloze_neutral", "answer_plus_suffix") for a in arm_names}
            for tier in self.tiers_of(co):
                rows = self.arm_mean_rows(rep, co, arms_t, "cloze_neutral", "answer_plus_suffix", pooled=False, tier=tier)
                keep_dose(rows)
                R.table(f"dose_response__{cname}__cloze_neutral__answer_plus_suffix__tier-{tier}", rows, cname,
                        note=f"entity tier {tier} only (secondary table){headline}",
                        show=(cname == self.headline and tier == "novel"), show_cols=dose_cols)
        # held-out baselines
        for pa in passes_held:
            for metric in cloze_metrics:
                for cname in held_sel:
                    co = self.cohorts[cname]
                    arms_t = {a: tab(a, "heldout", pa, metric) for a in arm_names}
                    rows = self.arm_mean_rows(rep, co, arms_t, pa, metric, pooled=True)
                    keep_dose(rows)
                    R.table(f"heldout_baseline__{cname}__{pa}__{metric}", rows, cname,
                            note=("never-injected facts: the per-arm baseline of the belief contrast" if not self.mirror_note(cname)
                                  else self.mirror_note(cname))
                                 + (f"; {self.cohort_note(cname)}" if self.cohort_note(cname) else ""),
                            show=(pa == "cloze_neutral" and metric == "answer_plus_suffix" and cname != "heldout_groups_both"
                                  and not self.mirror_note(cname)),
                            show_cols=["label", "arm", "n", "n_groups", "mean_belief", "ci95_lo", "ci95_hi", "sd_belief",
                                       "mean_logp_value", "mean_logp_competing"])

        # 2. paired contrasts ------------------------------------------------------------
        R.say("## 2. Paired per-fact contrasts between arms (one Holm family per table)\n")
        R.say("Each table = one (contrast, cohort, pass, metric); its dose x label cells form the prespecified Holm "
              "family (`p_wilcoxon` raw, `p_holm` adjusted, `family_size` = tested cells). `p_wilcoxon` is the "
              "two-sided scipy Wilcoxon signed-rank p on the per-fact paired differences when every semantic group "
              "contributes one fact (`p_wilcoxon_level` = fact) and on the GROUP-level mean differences otherwise "
              "(level = group; finding 18, 2026-10-14) -- n/a below 10 non-zero values at that level; "
              "`p_wilcoxon_fact` (CSV only) is the naive per-fact p, given only when it coincides with the test used. "
              "`mean_abs_diff` = mean |paired difference| (per group where the cohort holds one record per group). "
              "Items: common items of every arm with the pass (finding 7). CIs: cluster bootstrap over semantic "
              "groups. The `all/all` row is descriptive and outside the family. "
              + (f"TOST equivalence margin: {self.margin}." if self.margin is not None else NO_EQUIVALENCE_NOTE)
              + " " + REPLICATE_NOTE + "\n")
        show_cols = ["tier", "label", "dose", "n", "n_groups", "mean_diff", "ci95_lo", "ci95_hi", "mean_abs_diff", "p_wilcoxon",
                     "p_wilcoxon_level", "p_holm", "n_pos", "n_neg", "n_pref_flips"]
        if self.margin is not None:
            show_cols += ["ci90_lo", "ci90_hi", "tost_equivalent"]
        jobs = []
        for cname in inj_sel:
            for pa in passes_inj:
                for metric in cloze_metrics:
                    jobs.append((cname, "injected", pa, metric, True, None))
            for tier in self.tiers_of(self.cohorts[cname]):
                jobs.append((cname, "injected", "cloze_neutral", "answer_plus_suffix", True, tier))
        for cname in held_sel:
            for pa in passes_held:
                for metric in cloze_metrics:
                    jobs.append((cname, "heldout", pa, metric, False, None))
        if any(("injected", PARAPHRASE_PASS) in a["data"] for a in arms_raw.values()):
            for cname in [c for c in CH.PARAPHRASE_COHORTS if c in self.selected]:
                jobs.append((cname, "injected", PARAPHRASE_PASS, "full_sentence", True, None))
        for cname, ps, pa, metric, by_dose, tier in jobs:
            co = self.cohorts[cname]
            if tier is None:
                # per-fact beliefs of every arm, kept for the cross-replicate statistics of section 6 (units x replicates
                # variance components, same-arm noise floor); never pooled across replicates here
                bel = store["beliefs"].setdefault((cname, pa, metric), {})
                for a in arm_names:
                    t = tab(a, ps, pa, metric)
                    bel[a] = {fid: t[fid]["belief"] for fid in co.fact_ids if fid in t}
            for a1, a2 in contrasts:
                arms_t = {a: tab(a, ps, pa, metric) for a in (a1, a2)}
                rows = self.paired_rows(rep, co, arms_t, a1, a2, pa, metric, by_dose=by_dose, tier=tier)
                for r in rows:
                    store["paired"][(r["contrast"], r["cohort"], r["pass"], r["metric"], r["tier"], r["label"], r["dose"])] = r
                if tier is None:
                    show = (metric != "answer_only") and (pa in ("cloze_neutral", "wrapper_conditional", PARAPHRASE_PASS)
                                                          or ps == "heldout")
                    show = show and cname != "heldout_groups_both"
                    name, note = f"paired__{a1}-{a2}__{cname}__{pa}__{metric}", f"contrast {a1}-{a2}; all entity tiers; n = facts scored in both arms"
                    if self.cohort_note(cname):
                        note += "; " + self.cohort_note(cname)
                    if self.mirror_note(cname):
                        note += "; " + self.mirror_note(cname)
                        show = False     # a sign-flipped copy of the all_injected table: CSV only
                    if cname == "paraphrase_historical_reference":
                        note += "; HISTORICAL REFERENCE cohort (probe-derived legacy well-formedness), for comparison with the July-2026 numbers"
                else:
                    show = cname == self.headline and tier == "novel"
                    name = f"paired__{a1}-{a2}__{cname}__{pa}__{metric}__tier-{tier}"
                    note = f"contrast {a1}-{a2}; entity tier {tier} only (secondary family); n = facts scored in both arms"
                R.table(name, rows, cname, note=note, show=show, show_cols=show_cols)

        # 2b. heterogeneity by relation (review 2026-11-11 finding 9) ---------------------------
        rel_cohorts = [c for c in dict.fromkeys([self.headline]) if c and c in self.selected]
        if rel_cohorts and "cloze_neutral" in passes_inj:
            R.say("### 2b. Heterogeneity by relation (headline cohort, neutral cloze)\n")
            R.say("The pooled contrasts above average relations whose answers differ in kind (a closed vocabulary such as an "
                  "animal class, numbers, invented names) and whose log-odds live on different scales. `n_values` = distinct "
                  "injected values of the relation in the cohort; few values = closed vocabulary. Descriptive rows, no Holm "
                  "family; per-dose rows are in the CSV.\n")
            for cname in rel_cohorts:
                co = self.cohorts[cname]
                by_rel = defaultdict(list)
                for fid in co.fact_ids:
                    by_rel[str(co.facts[fid].get("relation") or "?")].append(fid)
                for a1, a2 in contrasts:
                    t1, t2 = tab(a1, "injected", "cloze_neutral", "answer_plus_suffix"), tab(a2, "injected", "cloze_neutral", "answer_plus_suffix")
                    rows = []
                    for rel in sorted(by_rel):
                        rr = self.subset_rows(rep, co, by_rel[rel], t1, t2, a1, a2, "cloze_neutral", "answer_plus_suffix", f"relation={rel}")
                        n_values = len({str(co.facts[f].get("value")) for f in by_rel[rel]})
                        for r in rr:
                            r["relation"], r["n_values"] = rel, n_values
                            store["paired"][(r["contrast"], cname, f"cloze_neutral@relation={rel}", r["metric"], "all", "all", r["dose"])] = r
                        rows += rr
                    tname = f"by_relation__{a1}-{a2}__{cname}__cloze_neutral__answer_plus_suffix"
                    R.table(tname, [r for r in rows if r["dose"] == "all"], cname,
                            note=f"contrast {a1}-{a2} within each relation, all doses pooled (per-dose rows in the CSV)",
                            show=(a2 == "C" or (a1, a2) == ("X", "E")),
                            show_cols=["relation", "n_values", "n", "mean_diff", "ci95_lo", "ci95_hi", "mean_abs_diff", "n_pref_flips"])
                    R.table(tname, [r for r in rows if r["dose"] != "all"], cname, show=False)
                for arm in arm_names:
                    for rel in sorted(by_rel):
                        t = tab(arm, "injected", "cloze_neutral", "answer_plus_suffix")
                        store["beliefs"].setdefault((cname, f"cloze_neutral@relation={rel}", "answer_plus_suffix"), {})[arm] = \
                            {fid: t[fid]["belief"] for fid in by_rel[rel] if fid in t}

        # 3. attribution asymmetry -------------------------------------------------------
        R.say("## 3. Attribution asymmetry: attributed-cloze minus neutral-cloze belief, per arm\n")
        R.say("Per fact: belief under the generic 'According to one source:' prefix minus the neutral belief "
              "(same fact, same arm, metric answer_plus_suffix). Rows per arm: all cells pooled, then by dose "
              "(labels pooled) and by label (doses pooled). The in-distribution trained-wrapper probe "
              "(pass wrapper_conditional) is in the section-2 tables `paired__*__wrapper_conditional__*`.\n")
        for cname in inj_sel:
            co = self.cohorts[cname]
            for pa in [p for p in ATTRIBUTED_PASSES if p in passes_inj]:
                rows = []
                doses, labels = self.cells_of(co)
                for arm in arm_names:
                    att, neu = tab(arm, "injected", pa, "answer_plus_suffix"), tab(arm, "injected", "cloze_neutral", "answer_plus_suffix")
                    cells = [("all", "all")] + [("all", d) for d in doses] + [(lab, "all") for lab in labels]
                    for lab, d in cells:
                        picked = self.pick(co, att, None if d == "all" else d, None if lab == "all" else lab)
                        diffs, groups = [], []
                        for fid, f, s in picked:
                            if fid in neu:
                                diffs.append(s["belief"] - neu[fid]["belief"])
                                groups.append(f["group_id"])
                        if not diffs:
                            continue
                        b = self.boot(diffs, groups, rep, "attrib", cname, pa, arm, lab, d)
                        rows.append({"replicate": rep, "cohort": cname, "pass": pa, "metric": "answer_plus_suffix", "arm": arm,
                                     "label": lab, "dose": d, "n": b.n, "n_groups": b.n_groups, "mean_diff": b.mean,
                                     "ci95_lo": b.ci95_lo, "ci95_hi": b.ci95_hi,
                                     "sd_diff": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None})
                R.table(f"attribution_asymmetry__{cname}__{pa}", rows, cname,
                        note="n = facts with both passes scored in that arm", show=(cname == self.headline),
                        show_cols=["arm", "label", "dose", "n", "n_groups", "mean_diff", "ci95_lo", "ci95_hi"])
                self.first_rows.setdefault(rep, {})[f"attrib__{cname}__{pa}"] = rows

        # 4. generalization vs memorization ---------------------------------------------
        R.say("## 4. Generalization vs memorization\n")
        if any(("injected", PARAPHRASE_PASS) in a["data"] for a in arms_raw.values()):
            R.say("### 4a. Held-out paraphrase (full_sentence metric)\n")
            for cname in [c for c in ("paraphrase_wellformed", "paraphrase_historical_reference", "all_injected") if c in self.selected]:
                co = self.cohorts[cname]
                arms_t = {a: tab(a, "injected", PARAPHRASE_PASS, "full_sentence") for a in arm_names}
                rows = self.arm_mean_rows(rep, co, arms_t, PARAPHRASE_PASS, "full_sentence", pooled=True)
                keep_dose(rows)
                if cname == "paraphrase_wellformed":
                    note = "well-formed contrasts of THIS evaluation (CSV-aware; basis: " + str(co.details.get("basis", "")) + ")"
                elif cname == "paraphrase_historical_reference":
                    note = ("HISTORICAL REFERENCE: probe-derived well-formedness under the legacy str.replace evaluator "
                            "(2026-10-13 finding 8); for comparison with the July-2026 numbers")
                else:
                    note = "LEGACY COMPARISON ONLY: every injected fact, including the historically malformed contrasts; not used for tests"
                R.table(f"paraphrase__{cname}__{PARAPHRASE_PASS}__full_sentence", rows, cname, note=note,
                        show_cols=["label", "dose", "arm", "n", "n_groups", "mean_belief", "ci95_lo", "ci95_hi",
                                   "mean_logp_value", "mean_logp_competing"],
                        max_rows=None if cname == "paraphrase_wellformed" else 3 * len(arm_names))
            R.say("Paired paraphrase contrasts: see `paired__*__paraphrase_wellformed__heldout_paraphrase__full_sentence` "
                  "(and `__paraphrase_historical_reference__` for the historical comparison) above.\n")

        R.say("### 4b. Template-0 probe joined to the realized training forms (finding 12)\n")
        R.say("A fact is `verbatim_trained` when cloze template 0 with the blank filled equals a surface form "
              "actually rendered at its assigned dose (v1: training_surface_forms cycled by occurrence; v2 without a "
              "manifest: the renderer's own index rule ctx.templates._pick with the build seed, finding 17; manifest "
              "paraphrase indices when given; exposure index keeps only complete, in-horizon occurrences). Metric "
              "answer_plus_suffix, pass cloze_neutral; `mean_t0_minus_other` is the within-fact paired difference.\n")
        mem_cohorts = [c for c in dict.fromkeys([self.headline, "all_injected"]) if c and c in self.selected]
        try:
            classes = CH.classify_verbatim_trained(self.probes.injected, self.protocol_cfg, exposure, self.manifest)
        except ProtocolError as e:
            classes = None
            R.say(f"NOT COMPUTABLE: {e}\n")
        if classes:
            for cname in mem_cohorts:
                co = self.cohorts[cname]
                rows = []
                for arm in arm_names:
                    # an intervention arm's own exposure index when given; the control arm renders no claim, so its rows
                    # use the nominal classification (docs/ANALYSIS.md)
                    src = arm if (arm in classes and arm not in CH.CONTROL_ARMS) else "nominal"
                    t0 = tab(arm, "injected", "cloze_neutral", "answer_plus_suffix", lambda ti: ti == 0, "t0")
                    other = tab(arm, "injected", "cloze_neutral", "answer_plus_suffix", lambda ti: ti != 0, "other")
                    for sub, want in (("verbatim_trained", True), ("not_verbatim_trained", False)):
                        fids = [fid for fid in co.fact_ids if classes[src].get(fid) is want and fid in t0 and fid in other]
                        if not fids:
                            continue
                        d = [t0[f]["belief"] - other[f]["belief"] for f in fids]
                        b = self.boot(d, [co.facts[f]["group_id"] for f in fids], rep, "memo", cname, arm, sub)
                        rows.append({"replicate": rep, "cohort": cname, "classification_source": src, "subgroup": sub, "arm": arm,
                                     "n": len(fids), "n_groups": b.n_groups,
                                     "mean_template0": float(np.mean([t0[f]["belief"] for f in fids])),
                                     "mean_other_templates": float(np.mean([other[f]["belief"] for f in fids])),
                                     "mean_t0_minus_other": b.mean, "ci95_lo": b.ci95_lo, "ci95_hi": b.ci95_hi})
                n_vt = sum(1 for fid in co.fact_ids if classes["nominal"].get(fid))
                R.table(f"memorization__{cname}", rows, cname,
                        note=f"nominal classification: {n_vt} of {co.n} facts verbatim_trained; classification_source "
                             f"'nominal' = rendering rule only, an arm name = that arm's exposure index",
                        show_cols=["classification_source", "subgroup", "arm", "n", "n_groups", "mean_template0",
                                   "mean_other_templates", "mean_t0_minus_other", "ci95_lo", "ci95_hi"])
                co.details.setdefault("verbatim_trained", {})["nominal"] = n_vt

        # 4c. every cloze item, not only template 0 (review 2026-11-11 finding 4) ----------------
        R.say("### 4c. Verbatim-trained cloze ITEMS: every template (review 2026-11-11 finding 4)\n")
        R.say("An item (fact, cloze template) is `verbatim_trained` when the template with the blank filled is a claim "
              "sentence rendered for the fact at its assigned dose -- for template 0 as in 4b, but also for every other "
              "template (in v2 pools template 1 equals a trained frame for some relations, so the template-0 split of 4b "
              "mixes the two groups). The claim sentence of a slot is the same in every intervention arm, so the split is "
              "ONE item partition shared by all arms. Contrasts are per fact over its items in the subset; "
              "`not_verbatim_items` is the contrast free of verbatim recall of the probe sentence.\n")
        try:
            item_classes = CH.classify_verbatim_items(self.probes.injected, self.protocol_cfg, None, self.manifest)["nominal"]
        except ProtocolError as e:
            item_classes = None
            R.say(f"NOT COMPUTABLE: {e}\n")
        if item_classes and "cloze_neutral" in passes_inj:
            ci_neutral = common["answer_plus_suffix"].get(("injected", "cloze_neutral"), {}).get("common", {})
            subsets = {"verbatim_trained_items": {}, "not_verbatim_items": {}}
            for fid, tis in ci_neutral.items():
                flags = item_classes.get(fid) or {}
                yes = {ti for ti in tis if flags.get(ti) is True}
                no = {ti for ti in tis if flags.get(ti) is False}
                if yes:
                    subsets["verbatim_trained_items"][fid] = yes
                if no:
                    subsets["not_verbatim_items"][fid] = no
            for cname in mem_cohorts:
                co = self.cohorts[cname]
                rows = []
                for sub, items_sub in subsets.items():
                    fids = [f for f in co.fact_ids if f in items_sub]
                    n_items = sum(len(items_sub[f]) for f in fids)
                    tabs = {a: per_fact(arms_raw[a], "injected", "cloze_neutral", "answer_plus_suffix", items=items_sub) for a in arm_names}
                    store["beliefs"].setdefault((cname, f"cloze_neutral@{sub}", "answer_plus_suffix"), {}).update(
                        {a: {fid: tabs[a][fid]["belief"] for fid in fids if fid in tabs[a]} for a in arm_names})
                    for a1, a2 in contrasts:
                        rr = self.subset_rows(rep, co, fids, tabs[a1], tabs[a2], a1, a2, "cloze_neutral", "answer_plus_suffix", sub)
                        for r in rr:
                            r["n_items"] = n_items if r["dose"] == "all" else sum(
                                len(items_sub[f]) for f in fids if co.facts[f]["dose"] == r["dose"])
                            store["paired"][(r["contrast"], cname, f"cloze_neutral@{sub}", r["metric"], "all", "all", r["dose"])] = r
                        rows += rr
                n_v = sum(len(v) for f, v in subsets["verbatim_trained_items"].items() if f in co.facts)
                n_all = sum(len(ci_neutral.get(f, ())) for f in co.fact_ids)
                co.details.setdefault("verbatim_trained", {})["items"] = {"verbatim_trained_items": n_v, "items": n_all}
                tname = f"verbatim_items__{cname}__cloze_neutral__answer_plus_suffix"
                R.table(tname, [r for r in rows if r["dose"] == "all"], cname,
                        note=f"{n_v} of {n_all} neutral-cloze items of the cohort are verbatim-trained claim sentences (nominal "
                             "rendering rule / manifest); `n` = facts with at least one item in the subset; per-dose rows in the CSV",
                        show_cols=["subset", "contrast", "n", "n_items", "mean_diff", "ci95_lo", "ci95_hi"])
                R.table(tname, [r for r in rows if r["dose"] != "all"], cname, show=False)

        # 5. CounterFact -----------------------------------------------------------------
        if "counterfact" in self.selected and any(("counterfact", "cloze_neutral") in a["data"] for a in arms_raw.values()):
            R.say("## 5. CounterFact (external, never injected; finding 16)\n")
            co = self.cohorts["counterfact"]
            cft = {a: tab(a, "counterfact", "cloze_neutral", "answer_plus_suffix") for a in arm_names}
            rows = []
            for arm in arm_names:
                picked = self.pick(co, cft[arm])
                if not picked:
                    continue
                vals = [s["belief"] for _, _, s in picked]
                b = self.boot(vals, [f["group_id"] for _, f, _ in picked], rep, "cf_mean", arm)
                rows.append({"replicate": rep, "cohort": "counterfact", "arm": arm, "n": b.n,
                             "n_correct": int(sum(v > 0 for v in vals)), "accuracy": float(np.mean([v > 0 for v in vals])),
                             "mean_belief": b.mean, "ci95_lo": b.ci95_lo, "ci95_hi": b.ci95_hi,
                             "mean_logp_value": float(np.mean([s["logp_value"] for _, _, s in picked])),
                             "mean_logp_competing": float(np.mean([s["logp_competing"] for _, _, s in picked]))})
                store["cf"][("accuracy", arm)] = rows[-1]["accuracy"]
                store["cf"][("mean_belief", arm)] = rows[-1]["mean_belief"]
                store["beliefs"].setdefault(("counterfact", "cloze_neutral", "answer_plus_suffix"), {})[arm] = \
                    {fid: s["belief"] for fid, _, s in picked}
            R.table("counterfact__per_arm", rows, "counterfact",
                    note="accuracy = fraction of items with logP(true object) > logP(counterfactual object)")
            prow, deltas_for_fig = [], {}
            if "C" in cft:
                for arm in [a for a in arm_names if a != "C"]:
                    picked = self.pick(co, cft[arm])
                    items = [(fid, f, s["belief"], cft["C"][fid]["belief"]) for fid, f, s in picked if fid in cft["C"]]
                    if not items:
                        continue
                    d = np.array([a - c for _, _, a, c in items])
                    groups = [f["group_id"] for _, f, _, _ in items]
                    cell = ST.paired_cell(d, groups, n_boot=self.n_boot, seed=self.seed, stream=(rep, "cf_paired", arm), margin=self.margin)
                    babs = self.boot(np.abs(d), groups, rep, "cf_abs", arm)
                    c_true = np.array([c > 0 for _, _, _, c in items])
                    a_true = np.array([a > 0 for _, _, a, _ in items])
                    t2f, f2t = int(np.sum(c_true & ~a_true)), int(np.sum(~c_true & a_true))
                    n = len(items)
                    prow.append({"replicate": rep, "cohort": "counterfact", "contrast": f"{arm}-C", "n": n, "n_groups": cell["n_groups"],
                                 "mean_delta": cell["mean_diff"], "ci95_lo": cell["ci95_lo"], "ci95_hi": cell["ci95_hi"],
                                 "mean_abs_delta": babs.mean, "abs_ci95_lo": babs.ci95_lo, "abs_ci95_hi": babs.ci95_hi,
                                 "p_wilcoxon": cell["p_wilcoxon"], "p_wilcoxon_level": cell["p_wilcoxon_level"],
                                 "n_flips_total": t2f + f2t, "rate_flips_total": (t2f + f2t) / n,
                                 "n_C_true_to_arm_false": t2f, "rate_C_true_to_arm_false": t2f / n,
                                 "n_C_false_to_arm_true": f2t, "rate_C_false_to_arm_true": f2t / n,
                                 "accuracy_C": float(c_true.mean()), f"accuracy_arm": float(a_true.mean())})
                    for k in ("mean_delta", "mean_abs_delta", "n_flips_total", "n_C_true_to_arm_false", "n_C_false_to_arm_true"):
                        store["cf"][(k, arm)] = prow[-1][k]
                    deltas_for_fig[arm] = d
                R.table("counterfact__paired_vs_C", prow, "counterfact",
                        note="per-item change vs C; reversals: C-true->arm-false = C prefers the true object and the arm does "
                             "not; C-false->arm-true the reverse; `p_wilcoxon` on the paired deltas; CIs = bootstrap over items "
                             "(each CounterFact item is its own cluster)",
                        show_cols=["contrast", "n", "mean_delta", "ci95_lo", "ci95_hi", "mean_abs_delta", "p_wilcoxon",
                                   "n_flips_total", "n_C_true_to_arm_false", "n_C_false_to_arm_true", "accuracy_C", "accuracy_arm"])
                self.first_rows.setdefault(rep, {})["cf_deltas"] = deltas_for_fig
            R.say("Similar per-arm means do NOT imply unchanged knowledge: the per-item changes and preference "
                  "reversals above show that equal means can conceal item-level movement. Whether that movement "
                  "exceeds ordinary training-run variation is unmeasured without replicates; with replicates it is "
                  "answered by the same-arm noise floor of section 6 (`replicates__same_arm_noise`: the same arm "
                  "re-trained).\n")

        if supp_raw:
            self.supplement(rep, arms_raw, supp_raw, tab, store)

    # ---- supplementary probes (review 2026-11-11) -----------------------------------------
    def supplement(self, rep, arms_raw, supp_raw, tab, store):
        """Section 5b for one replicate: the wrong-subject control and arm E under its own frame."""
        R = self.R
        arm_names = sorted(supp_raw)
        contrasts = [(a1, a2) for a1, a2 in CONTRASTS if a1 in supp_raw and a2 in supp_raw]
        common = common_items(supp_raw, "answer_plus_suffix")
        metric = "answer_plus_suffix"
        has = {pa: any(("injected", pa) in a["data"] for a in supp_raw.values()) for pa in SUPPLEMENT_PASSES}

        def stab(arm, pa):
            return per_fact(supp_raw[arm], "injected", pa, metric, items=common.get(("injected", pa), {}).get("common", {}))

        R.say("## 5b. Supplementary probes: value familiarity and arm E under its own frame (review 2026-11-11)\n")
        R.say("Scored by `eval_beliefs.py --supplement` on the same checkpoints (file `supp_beliefs_arm<ARM>.csv`); "
              "items per pass: " + ", ".join(f"{pa} {common.get(('injected', pa), {}).get('n_common', 0)}" for pa in SUPPLEMENT_PASSES) + ".\n")
        show_cols = ["tier", "label", "dose", "n", "n_groups", "mean_diff", "ci95_lo", "ci95_hi", "mean_abs_diff", "p_wilcoxon", "p_holm"]
        for cname in [c for c in dict.fromkeys([self.headline]) if c and c in self.selected]:
            co = self.cohorts[cname]

            def keep(rows, pa):
                for r in rows:
                    store["paired"][(r["contrast"], cname, pa, r["metric"], r.get("tier", "all"), r["label"], r["dose"])] = r

            if has["wrong_subject"]:
                R.say("### Wrong-subject control (is the shift a belief about the subject?)\n")
                R.say("`wrong_subject`: the fact's injected value against its rival in the neutral cloze of OTHER subjects of the "
                      "same relation, taken from wholly held-out groups (never trained in any arm). An arm-vs-control shift "
                      "there cannot be a belief about the fact's subject: it is familiarity with the value string (seen "
                      "`dose` times in the intervention arms, never in the control) or a relation-level value prior. "
                      "`subject_specific` = neutral belief (the fact's own subject) minus the wrong-subject belief, per fact: "
                      "the part of the shift that needs the right subject. For any contrast, neutral = wrong_subject + "
                      "subject_specific exactly.\n")
                ws = {a: stab(a, "wrong_subject") for a in arm_names}
                neutral = {a: tab(a, "injected", "cloze_neutral", metric) for a in arm_names}
                spec = {a: {fid: {"belief": neutral[a][fid]["belief"] - s["belief"]} for fid, s in ws[a].items() if fid in neutral[a]}
                        for a in arm_names}
                parts_tabs = (("wrong_subject", ws), ("subject_specific", spec))
                for pa, tabs in parts_tabs:
                    store["beliefs"].setdefault((cname, pa, metric), {}).update(
                        {a: {fid: tabs[a][fid]["belief"] for fid in co.fact_ids if fid in tabs[a]} for a in arm_names})
                    for a1, a2 in contrasts:
                        rows = self.paired_rows(rep, co, {a1: tabs[a1], a2: tabs[a2]}, a1, a2, pa, metric)
                        keep(rows, pa)
                        R.table(f"supp_paired__{a1}-{a2}__{cname}__{pa}__{metric}", rows, cname,
                                note=f"contrast {a1}-{a2}, pass {pa}; one Holm family (dose x label cells)", show=False, show_cols=show_cols)
                # the decomposition, labels pooled: one row per arm and dose; and the two parts within each relation
                rows, by_rel_rows = [], []
                by_rel = defaultdict(list)
                for fid in co.fact_ids:
                    by_rel[str(co.facts[fid].get("relation") or "?")].append(fid)
                for a1 in [a for a in arm_names if a != "C" and "C" in arm_names]:
                    parts = {pa: self.subset_rows(rep, co, co.fact_ids, tabs[a1], tabs["C"], a1, "C", pa, metric, "decomposition")
                             for pa, tabs in parts_tabs}
                    for w in parts["wrong_subject"]:
                        sp = next((x for x in parts["subject_specific"] if x["dose"] == w["dose"]), None)
                        if sp is None:
                            continue
                        total = w["mean_diff"] + sp["mean_diff"]
                        rows.append({"replicate": rep, "cohort": cname, "contrast": f"{a1}-C", "dose": w["dose"], "n": w["n"],
                                     "neutral_shift": total, "wrong_subject_shift": w["mean_diff"],
                                     "ws_ci95_lo": w["ci95_lo"], "ws_ci95_hi": w["ci95_hi"],
                                     "subject_specific_shift": sp["mean_diff"], "ss_ci95_lo": sp["ci95_lo"], "ss_ci95_hi": sp["ci95_hi"],
                                     "share_not_subject_specific": (w["mean_diff"] / total) if abs(total) > 1e-12 else None})
                    for rel in sorted(by_rel):
                        for pa, tabs in parts_tabs:
                            rr = self.subset_rows(rep, co, by_rel[rel], tabs[a1], tabs["C"], a1, "C", pa, metric, f"relation={rel}", by_dose=False)
                            for r in rr:
                                r["relation"] = rel
                                r["n_values"] = len({str(co.facts[f].get("value")) for f in by_rel[rel]})
                                store["paired"][(r["contrast"], cname, f"{pa}@relation={rel}", metric, "all", "all", r["dose"])] = r
                            by_rel_rows += rr
                R.table(f"supp_decomposition__{cname}", rows, cname,
                        note="arm - C on the neutral cloze split into the wrong-subject part (not about the subject) and the "
                             "subject-specific remainder; labels pooled; `share_not_subject_specific` = wrong_subject / neutral",
                        show_cols=["contrast", "dose", "n", "neutral_shift", "wrong_subject_shift", "ws_ci95_lo", "ws_ci95_hi",
                                   "subject_specific_shift", "ss_ci95_lo", "ss_ci95_hi", "share_not_subject_specific"])
                R.table(f"supp_by_relation__{cname}", by_rel_rows, cname,
                        note="the two parts within each relation (all doses pooled); `n_values` = distinct injected values (few = closed vocabulary)",
                        show_cols=["contrast", "pass", "relation", "n_values", "n", "mean_diff", "ci95_lo", "ci95_hi"], max_rows=48)

            if has["wrapper_conditional_E"]:
                R.say("### Arm E under its own trained frame\n")
                R.say("`wrapper_conditional_E`: the fact's occurrence-0 arm-E training frame cut at the value, scored in every "
                      "arm -- the counterpart of `wrapper_conditional`, which uses arm X's wrapper in every arm. `own_frame` "
                      "compares each wrapped arm under ITS OWN trained frame: (X - C under the X wrapper) minus (E - C under "
                      "the E frame), per fact. Near zero = the two arms store the claim equally retrievably behind their own "
                      "context; the low E score under X's wrapper is then a mismatch of context, not weaker learning.\n")
                we = {a: stab(a, "wrapper_conditional_E") for a in arm_names}
                store["beliefs"].setdefault((cname, "wrapper_conditional_E", metric), {}).update(
                    {a: {fid: we[a][fid]["belief"] for fid in co.fact_ids if fid in we[a]} for a in arm_names})
                for a1, a2 in contrasts:
                    rows = self.paired_rows(rep, co, {a1: we[a1], a2: we[a2]}, a1, a2, "wrapper_conditional_E", metric)
                    keep(rows, "wrapper_conditional_E")
                    R.table(f"supp_paired__{a1}-{a2}__{cname}__wrapper_conditional_E__{metric}", rows, cname,
                            note=f"contrast {a1}-{a2} under arm E's own trained frame", show=(a2 == "C"), show_cols=show_cols, max_rows=12)
                if all(a in arm_names for a in ("C", "X", "E")) and all(("injected", "wrapper_conditional") in arms_raw[a]["data"] for a in ("C", "X")):
                    wx = {a: tab(a, "injected", "wrapper_conditional", metric) for a in ("C", "X")}
                    own = {"X": {fid: {"belief": wx["X"][fid]["belief"] - wx["C"][fid]["belief"]} for fid in wx["X"] if fid in wx["C"]},
                           "E": {fid: {"belief": we["E"][fid]["belief"] - we["C"][fid]["belief"]} for fid in we["E"] if fid in we["C"]}}
                    store["beliefs"].setdefault((cname, "own_frame", metric), {}).update(
                        {a: {fid: own[a][fid]["belief"] for fid in co.fact_ids if fid in own[a]} for a in ("X", "E")})
                    rows = self.subset_rows(rep, co, co.fact_ids, own["X"], own["E"], "X", "E", "own_frame", metric, "own_frame")
                    for r in rows:
                        both = [f for f in co.fact_ids if f in own["X"] and f in own["E"] and (r["dose"] == "all" or co.facts[f]["dose"] == r["dose"])]
                        r["x_own"] = float(np.mean([own["X"][f]["belief"] for f in both])) if both else None
                        r["e_own"] = float(np.mean([own["E"][f]["belief"] for f in both])) if both else None
                        store["paired"][("X-E", cname, "own_frame", metric, "all", "all", r["dose"])] = r
                    R.table(f"supp_own_frame__{cname}", rows, cname,
                            note="x_own = X - C under the X wrapper, e_own = E - C under the E frame, mean_diff = x_own - e_own per fact "
                                 "(labels pooled)",
                            show_cols=["dose", "n", "x_own", "e_own", "mean_diff", "ci95_lo", "ci95_hi", "mean_abs_diff"])

    # ---- across replicates -----------------------------------------------------------
    @staticmethod
    def _split_pass(pa):
        base, _, subset = str(pa).partition("@")
        return base, subset

    def summarize_replicates(self):
        R = self.R
        all_reps = list(self.per_rep)
        reps = [r for r in all_reps if r not in self.incomplete]      # complete replicates only (finding 3)
        R.say("## 6. Uncertainty across training replicates\n")
        R.say(REPLICATE_NOTE + "\n")
        if self.incomplete:
            R.say("Incomplete replicates -- " + "; ".join(f"`{r}` lacks arm(s) {', '.join(m)}" for r, m in self.incomplete.items())
                  + " -- are excluded from every summary of this section except the same-arm noise floor (6c), so that no "
                    "per-arm mean averages a different set of training seeds (review 2026-11-11 finding 3).\n")
        if len(all_reps) < 2:
            R.say(f"Replicates found: {len(all_reps)} ({', '.join(all_reps)}). Training-run variation is therefore UNMEASURED "
                  "here; every interval and p-value in this report is within-run.\n")
            return
        if len(reps) < 2:
            R.say(f"Replicates found: {len(all_reps)} ({', '.join(all_reps)}), complete: {len(reps)} ({', '.join(reps) or 'none'}). "
                  "Cross-replicate contrasts are UNMEASURED; only the same-arm noise floor is available.\n")
            self._same_arm_noise(all_reps)
            return
        R.say(f"Replicates found: {len(all_reps)} ({', '.join(all_reps)})"
              + (f", complete: {len(reps)} ({', '.join(reps)})" if self.incomplete else "")
              + ". Each statistic below was computed per replicate; the summaries are the mean and SD (ddof=1) across "
                "replicates with n_replicates, never pooling facts across replicates as if they were independent observations.\n")

        # 6a. one number per replicate -------------------------------------------------------
        R.say("### 6a. Per-replicate values and the replicate-level interval\n")
        R.say("`per_replicate` lists the within-run mean paired difference of every replicate; `t95_lo` / `t95_hi` is the t "
              "interval of their mean with n_replicates - 1 degrees of freedom (the training run is the unit; multiplier "
              "12.7 with two runs, 4.30 with three). It answers: how well is the effect determined across training runs "
              "FOR THIS FACT SET? It is wide by construction at small n_replicates; it does not include the sampling of "
              "facts (the replicates share them), which section 6b adds.\n")
        keys = sorted({k for r in reps for k in self.per_rep[r]["paired"]}, key=lambda k: ("@" in str(k[2]), str(k)))
        rows = []
        for k in keys:
            vals = {r: self.per_rep[r]["paired"].get(k, {}).get("mean_diff") for r in reps}
            s = ST.summarize_replicates(vals)
            ri = ST.replicate_interval(vals)
            contrast, cohort, pa, metric, tier, lab, dose = k
            base, subset = self._split_pass(pa)
            rows.append({"contrast": contrast, "cohort": cohort, "pass": base, "subset": subset, "metric": metric, "tier": tier,
                         "label": lab, "dose": dose,
                         "n_replicates": s["n_replicates"], "mean_of_mean_diff": s["mean"], "sd_across_replicates": s["sd"],
                         "min": s["min"], "max": s["max"], "se_across_replicates": ri["se"], "t95_lo": ri["t_lo"], "t95_hi": ri["t_hi"],
                         "per_replicate": "; ".join(f"{r}={fnum(v)}" for r, v in ri["values"].items()),
                         "n_per_replicate": "/".join(str(self.per_rep[r]["paired"].get(k, {}).get("n", "")) for r in reps)})

        def headline_first(r):
            return (r["cohort"] != self.headline, r["cohort"], r["pass"], r["subset"], r["contrast"])

        def pooled(r):
            return r["dose"] == "all" and r["label"] == "all" and r["tier"] == "all" and r["metric"] != "answer_only"

        shown = sorted([r for r in rows if pooled(r) and not r["subset"].startswith("relation=")], key=headline_first)
        rest = [r for r in rows if not (pooled(r) and not r["subset"].startswith("relation="))]
        cols = ["contrast", "cohort", "pass", "subset", "metric", "n_replicates", "per_replicate", "mean_of_mean_diff",
                "sd_across_replicates", "t95_lo", "t95_hi"]
        R.table("replicates__paired", shown, "per table",
                note="mean paired difference per replicate, summarized across COMPLETE replicates; rows shown: all doses and "
                     "labels pooled (every cell, the answer-only metric and the per-relation rows are in the CSV)",
                show_cols=cols, max_rows=60)
        R.table("replicates__paired", rest, "per table", show=False)

        # 6b. units x replicates variance components -------------------------------------------
        self._crossed(reps)
        # 6c. what re-training alone does ---------------------------------------------------------
        self._same_arm_noise(all_reps)

        keys = sorted({k for r in reps for k in self.per_rep[r]["dose"]}, key=str)
        rows = []
        for k in keys:
            s = ST.summarize_replicates({r: self.per_rep[r]["dose"].get(k) for r in reps})
            cohort, pa, metric, tier, arm, lab, dose = k
            rows.append({"cohort": cohort, "pass": pa, "metric": metric, "tier": tier, "arm": arm, "label": lab, "dose": dose,
                         "n_replicates": s["n_replicates"], "mean_of_means": s["mean"], "sd_across_replicates": s["sd"]})
        R.table("replicates__dose_response", rows, "per table",
                note="per-arm cell means summarized across COMPLETE replicates (every arm over the same seeds)", show=False)
        keys = sorted({k for r in reps for k in self.per_rep[r]["cf"]}, key=str)
        rows = []
        for k in keys:
            s = ST.summarize_replicates({r: self.per_rep[r]["cf"].get(k) for r in reps})
            rows.append({"statistic": k[0], "arm": k[1], "n_replicates": s["n_replicates"], "mean": s["mean"], "sd_across_replicates": s["sd"]})
        R.table("replicates__counterfact", rows, "counterfact",
                note="CounterFact statistics summarized across COMPLETE replicates")

    def _crossed(self, reps):
        """6b: per contrast, the facts x replicates matrix of paired differences -> variance components and the
        interval that treats BOTH facts and training runs as random (review 2026-11-11 finding 1)."""
        R = self.R
        R.say("### 6b. Facts x replicates: variance components and the combined interval\n")
        R.say("Per contrast the paired differences form a matrix (semantic groups x replicates; the facts scored in every "
              "replicate). `facts_ci95_*` is the cluster bootstrap of the replicate-averaged differences (replicates FIXED: "
              "uncertainty over facts for these particular runs). `var_unit` / `var_replicate` / `var_residual` are the ANOVA "
              "variance components (truncated at 0); `se_crossed` = sqrt(var_unit/U + var_replicate/S + var_residual/(U S)) "
              "treats facts AND runs as random and `crossed_ci95_*` is its Satterthwaite t interval. `p_rep` tests whether "
              "the replicate means differ by more than fact-level noise explains (F = MS_replicate / MS_residual): a small "
              "value means the within-run intervals of sections 1-5 UNDERSTATE the uncertainty of that contrast. How to "
              "read the three intervals: where `p_rep` is small, run-level variance is real and the replicate interval of "
              "6a is the appropriate one (the crossed interval then covers about 0.93 with three and 0.89 with two runs in "
              "simulation); where it is not, the facts / crossed interval is appropriate and the replicate interval, which "
              "leaves out the sampling of facts, under-covers (ctx.stats.crossed_summary lists the simulated coverage).\n")
        rows = []
        keys = sorted({k for r in reps for k in self.per_rep[r]["beliefs"]}, key=lambda k: ("@" in str(k[1]), str(k)))
        for cname, pa, metric in keys:
            co = self.cohorts.get(cname)
            per = [self.per_rep[r]["beliefs"].get((cname, pa, metric)) for r in reps]
            if co is None or any(x is None for x in per) or self.mirror_note(cname) or cname == "heldout_groups_both":
                continue
            base, subset = self._split_pass(pa)
            for a1, a2 in CONTRASTS:
                if any(a1 not in x or a2 not in x for x in per):
                    continue
                common = None
                for x in per:
                    have = set(x[a1]) & set(x[a2])
                    common = have if common is None else common & have
                fids = [f for f in co.fact_ids if f in common]
                doses = sorted({co.facts[f]["dose"] for f in fids})
                cells = [("all", fids)]
                if len(doses) > 1 and not subset:
                    cells += [(d, [f for f in fids if co.facts[f]["dose"] == d]) for d in doses]
                for dose_lbl, ff in cells:
                    if len(ff) < 2:
                        continue
                    D = np.array([[x[a1][f] - x[a2][f] for x in per] for f in ff], dtype=float)      # facts x replicates
                    groups = [co.facts[f]["group_id"] for f in ff]
                    uniq = list(dict.fromkeys(groups))
                    pos = {g: i for i, g in enumerate(uniq)}
                    M, cnt = np.zeros((len(uniq), len(reps))), np.zeros(len(uniq))
                    for row, g in zip(D, groups):
                        M[pos[g]] += row
                        cnt[pos[g]] += 1
                    M /= cnt[:, None]
                    c = ST.crossed_summary(M)
                    ri = ST.replicate_interval({r: float(D[:, j].mean()) for j, r in enumerate(reps)})
                    b = self.boot(D.mean(axis=1), groups, "crossed", a1, a2, cname, pa, metric, dose_lbl)
                    rows.append({"contrast": f"{a1}-{a2}", "cohort": cname, "pass": base, "subset": subset, "metric": metric,
                                 "dose": dose_lbl, "n_replicates": len(reps), "n": len(ff), "n_groups": len(uniq),
                                 "mean_diff": float(D.mean()), "mean_groups": c["mean"],
                                 "per_replicate": "; ".join(f"{r}={fnum(v)}" for r, v in ri["values"].items()),
                                 "sd_between_replicates": ri["sd"], "replicate_t95_lo": ri["t_lo"], "replicate_t95_hi": ri["t_hi"],
                                 "facts_ci95_lo": b.ci95_lo, "facts_ci95_hi": b.ci95_hi, "se_units_only": c["se_units_only"],
                                 "var_unit": c["var_unit"], "var_replicate": c["var_replicate"], "var_residual": c["var_residual"],
                                 "se_crossed": c["se_crossed"], "df_crossed": c["df_crossed"],
                                 "crossed_ci95_lo": c["ci_lo"], "crossed_ci95_hi": c["ci_hi"],
                                 "F_rep": c["F_rep"], "p_rep": c["p_rep"]})
        shown = [r for r in rows if r["dose"] == "all" and r["metric"] != "answer_only" and not r["subset"].startswith("relation=")]
        shown.sort(key=lambda r: (r["cohort"] != self.headline, r["cohort"], r["pass"], r["subset"], r["contrast"]))
        R.table("replicates__crossed", shown, "per table",
                note="facts x replicates decomposition of each contrast over the COMPLETE replicates; rows shown: all doses "
                     "pooled (per-dose, answer-only and per-relation rows in the CSV)",
                show_cols=["contrast", "cohort", "pass", "subset", "n_replicates", "n", "mean_diff", "facts_ci95_lo", "facts_ci95_hi",
                           "sd_between_replicates", "se_crossed", "crossed_ci95_lo", "crossed_ci95_hi", "p_rep"], max_rows=60)
        R.table("replicates__crossed", [r for r in rows if r not in shown], "per table", show=False)

    def _same_arm_noise(self, all_reps):
        """6c: the same arm in two replicates versus two arms in one replicate (review 2026-11-11 finding 2)."""
        R = self.R
        R.say("### 6c. Noise floor: the same arm re-trained versus different arms\n")
        R.say("`same_arm_across_replicates` compares ONE arm between two training runs (another init / data-order seed, same "
              "data): its `mean_abs_diff`, `n_pref_flips` and `sd_diff` are what re-training alone does to per-fact scores. "
              "`different_arms_same_replicate` gives the same statistics for two arms of one replicate. A cross-arm "
              "`mean_abs_diff` or flip count that is not clearly larger than the same-arm rows describes run-to-run noise, "
              "not an effect of the arm, however large it looks; only the MEAN differences (and their intervals above) carry "
              "the arm effect. The matched init/data seeds of a replicate do not make this floor smaller (the per-fact "
              "scores of matched runs decorrelate through nondeterministic kernels and the differing inserts). Incomplete "
              "replicates contribute the arms they have.\n")
        rows = []
        keys = sorted({k for r in all_reps for k in self.per_rep[r]["beliefs"]}, key=str)
        for cname, pa, metric in keys:
            co = self.cohorts.get(cname)
            if co is None or "@" in str(pa) or cname == "heldout_groups_both" or self.mirror_note(cname):
                continue
            bel = {r: self.per_rep[r]["beliefs"].get((cname, pa, metric)) or {} for r in all_reps}
            for i, ra in enumerate(all_reps):
                for rb in all_reps[i + 1:]:
                    for arm in ARMS:
                        if arm not in bel[ra] or arm not in bel[rb]:
                            continue
                        ff = [f for f in co.fact_ids if f in bel[ra][arm] and f in bel[rb][arm]]
                        if len(ff) < 2:
                            continue
                        st = ST.same_arm_noise([bel[rb][arm][f] for f in ff], [bel[ra][arm][f] for f in ff])
                        rows.append({"comparison": "same_arm_across_replicates", "cohort": cname, "pass": pa, "metric": metric,
                                     "a": f"{arm}@{rb}", "b": f"{arm}@{ra}", **st})
            for r in all_reps:
                for a1, a2 in CONTRASTS:
                    if a1 not in bel[r] or a2 not in bel[r]:
                        continue
                    ff = [f for f in co.fact_ids if f in bel[r][a1] and f in bel[r][a2]]
                    if len(ff) < 2:
                        continue
                    st = ST.same_arm_noise([bel[r][a1][f] for f in ff], [bel[r][a2][f] for f in ff])
                    rows.append({"comparison": "different_arms_same_replicate", "cohort": cname, "pass": pa, "metric": metric,
                                 "a": f"{a1}@{r}", "b": f"{a2}@{r}", **st})
        if not any(r["comparison"] == "same_arm_across_replicates" for r in rows):
            R.say("No arm was scored in two replicates: the noise floor is unmeasured.\n")
            return
        focus = [c for c in dict.fromkeys([self.headline, "heldout_groups", "counterfact"]) if c]
        shown = [r for r in rows if r["cohort"] in focus and r["pass"] == "cloze_neutral" and r["metric"] == "answer_plus_suffix"]
        shown.sort(key=lambda r: (focus.index(r["cohort"]), r["comparison"] != "same_arm_across_replicates", r["a"], r["b"]))
        R.table("replicates__same_arm_noise", shown, "per table",
                note="a - b per fact; rows shown: neutral cloze, answer_plus_suffix, headline / heldout_groups / counterfact "
                     "cohorts (every cohort, pass and metric in the CSV)",
                show_cols=["cohort", "comparison", "a", "b", "n", "mean_diff", "mean_abs_diff", "sd_diff", "n_pref_flips", "corr"],
                max_rows=80)
        R.table("replicates__same_arm_noise", [r for r in rows if r not in shown], "per table", show=False)


# ======================================================================================
# figures

def make_figures(an: Analysis, out_dir, arm_names):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        an.R.say("NOTE --plot requested but matplotlib is not installed; figures skipped")
        return []
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    R, out = an.R, []
    # complete replicates only (review 2026-11-11 finding 3): the overlay's thick line is a mean over the same seeds
    # for every arm; with no complete replicate the figures describe the first one
    reps = [r for r in an.per_rep if r not in an.incomplete] or list(an.per_rep)
    rep0 = reps[0]
    arm_names = [a for a in arm_names if a in an.per_rep[rep0]["arms"]]
    pa, metric = "cloze_neutral", "answer_plus_suffix"
    mk = "MOCK SCORES -- " if R.mock else ""

    def style(ax):
        ax.grid(True, which="major", color="#e4e3df", linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # dose-response per injected cohort ------------------------------------------------------
    for cname in [c for c in CH.INJECTED_COHORTS if c in an.selected]:
        rows = [r for r in an.R.tables.get(f"dose_response__{cname}__{pa}__{metric}", {"rows": []})["rows"] if r["replicate"] == rep0]
        if not rows:
            continue
        co = an.cohorts[cname]
        labels = sorted({r["label"] for r in rows}, key=label_sort_key)
        base_name = "heldout_groups" if "heldout_groups" in an.selected else "heldout_claims"
        base_rows = [r for r in an.R.tables.get(f"heldout_baseline__{base_name}__{pa}__{metric}", {"rows": []})["rows"] if r["replicate"] == rep0]
        fig, axes = plt.subplots(1, len(labels), figsize=(5.2 * len(labels), 4.4), squeeze=False, sharey=True)
        for ax, lab in zip(axes[0], labels):
            for arm in arm_names:
                pts = sorted([r for r in rows if r["label"] == lab and r["arm"] == arm], key=lambda r: r["dose"])
                if not pts:
                    continue
                xs = [r["dose"] for r in pts]
                ys = [r["mean_belief"] for r in pts]
                lo = [r["mean_belief"] - (r["ci95_lo"] if r["ci95_lo"] is not None else r["mean_belief"]) for r in pts]
                hi = [(r["ci95_hi"] if r["ci95_hi"] is not None else r["mean_belief"]) - r["mean_belief"] for r in pts]
                ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", ms=4, lw=1.6, capsize=3, color=ARM_COLORS.get(arm, "#555"),
                            label=f"arm {arm} (n/cell {min(r['n'] for r in pts)}-{max(r['n'] for r in pts)})")
                b = [r for r in base_rows if r["label"] == lab and r["arm"] == arm and r["dose"] != "all"]
                if b:
                    ax.axhline(b[0]["mean_belief"], ls=":", lw=1.2, color=ARM_COLORS.get(arm, "#555"), alpha=0.8,
                               label=f"{base_name} baseline, arm {arm} (n={b[0]['n']} groups)")
            ax.set_xscale("log", base=4)
            ax.set_xticks([d for d in sorted({r["dose"] for r in rows})])
            ax.set_xticklabels([str(d) for d in sorted({r["dose"] for r in rows})])
            ax.set_xlabel("assigned frequency (nominal occurrences)")
            ax.set_title(f"label = {lab}", fontsize=10)
            style(ax)
            ax.legend(fontsize=7, frameon=False)
        axes[0][0].set_ylabel("belief = logP(value) - logP(competing)")
        fig.suptitle(f"{mk}Dose-response, cohort {cname} (n={co.n}), pass {pa}, metric {metric}; dotted = {base_name} baseline "
                     f"(replicate {rep0}; cluster-bootstrap 95% CI over groups)", fontsize=9)
        fig.tight_layout()
        p = os.path.join(fig_dir, f"dose_response__{cname}.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out.append(p)
        R.say(f"figure -> {p}  (cohort {cname}, n={co.n}{'; HEADLINE' if cname == an.headline else ''})")

        if len(reps) > 1:
            fig, axes = plt.subplots(1, len(labels), figsize=(5.2 * len(labels), 4.4), squeeze=False, sharey=True)
            for ax, lab in zip(axes[0], labels):
                for arm in arm_names:
                    per_rep = []
                    for rep in reps:
                        pts = sorted([r for r in an.R.tables[f"dose_response__{cname}__{pa}__{metric}"]["rows"]
                                      if r["replicate"] == rep and r["label"] == lab and r["arm"] == arm], key=lambda r: r["dose"])
                        if pts:
                            per_rep.append(pts)
                            ax.plot([r["dose"] for r in pts], [r["mean_belief"] for r in pts], lw=0.8, alpha=0.45,
                                    color=ARM_COLORS.get(arm, "#555"))
                    if per_rep:
                        xs = [r["dose"] for r in per_rep[0]]
                        ys = [float(np.mean([pts[i]["mean_belief"] for pts in per_rep if i < len(pts)])) for i in range(len(xs))]
                        ax.plot(xs, ys, lw=2.2, marker="o", ms=4, color=ARM_COLORS.get(arm, "#555"),
                                label=f"arm {arm}: mean of {len(per_rep)} replicates")
                ax.set_xscale("log", base=4)
                ax.set_xlabel("assigned frequency (nominal occurrences)")
                ax.set_title(f"label = {lab}", fontsize=10)
                style(ax)
                ax.legend(fontsize=7, frameon=False)
            axes[0][0].set_ylabel("belief = logP(value) - logP(competing)")
            fig.suptitle(f"{mk}Per-replicate overlay, cohort {cname} (n={co.n}): thin = one training replicate, thick = mean across replicates", fontsize=9)
            fig.tight_layout()
            p = os.path.join(fig_dir, f"dose_response_replicates__{cname}.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            out.append(p)
            R.say(f"figure -> {p}")

    # attribution asymmetry -------------------------------------------------------------
    for cname in [c for c in CH.INJECTED_COHORTS if c in an.selected]:
        rows = an.first_rows.get(rep0, {}).get(f"attrib__{cname}__cloze_attributed")
        if not rows:
            continue
        co = an.cohorts[cname]
        doses = sorted({r["dose"] for r in rows if r["dose"] != "all"})
        fig, ax = plt.subplots(figsize=(7, 4.2))
        width = 0.8 / max(len(arm_names), 1)
        for i, arm in enumerate(arm_names):
            pts = [next((r for r in rows if r["arm"] == arm and r["label"] == "all" and r["dose"] == d), None) for d in doses]
            xs = [j + (i - (len(arm_names) - 1) / 2) * width for j in range(len(doses))]
            ys = [r["mean_diff"] if r else 0 for r in pts]
            lo = [(r["mean_diff"] - r["ci95_lo"]) if r and r["ci95_lo"] is not None else 0 for r in pts]
            hi = [(r["ci95_hi"] - r["mean_diff"]) if r and r["ci95_hi"] is not None else 0 for r in pts]
            ax.bar(xs, ys, width=width * 0.92, color=ARM_COLORS.get(arm, "#555"), label=f"arm {arm}")
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="#333", capsize=2, lw=0.8)
        ax.axhline(0, color="#333", lw=0.8)
        ax.set_xticks(range(len(doses)))
        ax.set_xticklabels([str(d) for d in doses])
        ax.set_xlabel("assigned frequency")
        ax.set_ylabel("attributed - neutral belief")
        ax.set_title(f"{mk}Attribution asymmetry, cohort {cname} (n={co.n}), labels pooled, 95% cluster-bootstrap CI", fontsize=9)
        style(ax)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        p = os.path.join(fig_dir, f"attribution_asymmetry__{cname}.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out.append(p)
        R.say(f"figure -> {p}")

    # counterfact paired changes --------------------------------------------------------
    deltas = an.first_rows.get(rep0, {}).get("cf_deltas")
    if deltas:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        allv = np.concatenate(list(deltas.values()))
        bins = np.linspace(float(allv.min()), float(allv.max()), 61) if allv.size > 1 else 10
        prow = {r["contrast"]: r for r in an.R.tables.get("counterfact__paired_vs_C", {"rows": []})["rows"] if r["replicate"] == rep0}
        for arm, d in deltas.items():
            r = prow.get(f"{arm}-C", {})
            ax.hist(d, bins=bins, histtype="step", lw=1.6, color=ARM_COLORS.get(arm, "#555"),
                    label=f"{arm}-C (n={d.size}; reversals C-true->{arm}-false {r.get('n_C_true_to_arm_false', '?')}, "
                          f"C-false->{arm}-true {r.get('n_C_false_to_arm_true', '?')})")
        ax.axvline(0, color="#333", lw=0.8)
        ax.set_xlabel("per-item change in belief vs arm C (log units)")
        ax.set_ylabel("items")
        ax.set_title(f"{mk}CounterFact paired changes vs C, cohort counterfact (n={an.cohorts['counterfact'].n}), replicate {rep0}", fontsize=9)
        style(ax)
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        p = os.path.join(fig_dir, "counterfact_paired_changes.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out.append(p)
        R.say(f"figure -> {p}")
    return out


# ======================================================================================
# main

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True, help="dir holding beliefs_arm<A>.csv, or seed_*/ rep*/ subdirs that do")
    ap.add_argument("--probe-dir", default=None, help="probe_sets/ with injected_facts.jsonl + heldout_facts.jsonl "
                                                      "(default: <eval-dir>/../probe_sets)")
    ap.add_argument("--build-summary", default=None, help="build_summary.json (default: <probe-dir>/../build_summary.json)")
    ap.add_argument("--manifest", default=None, help="manifest.csv[.gz]; v2 paraphrase indices are preferred for the memorization join")
    ap.add_argument("--exposure-dir", default=None,
                    help="exposure/ dir with PREFLIGHT_OK.json + <arm>/occurrence_exposure.csv, or a directory holding one such "
                         "index per data seed (exposure_by_seed/seed_<S>/): every replicate is bound to the index of its own run")
    ap.add_argument("--cohort", nargs="+", default=None, choices=list(CH.COHORT_NAMES), help="restrict the analysis to these cohorts")
    ap.add_argument("--plot", action="store_true", help="write figures/*.png (needs matplotlib)")
    ap.add_argument("--bootstrap", type=int, default=ST.DEFAULT_BOOTSTRAP, help="cluster-bootstrap resamples (0 disables)")
    ap.add_argument("--equivalence-margin", type=float, default=None, help="TOST margin M for paired means; no equivalence claims without it")
    ap.add_argument("--seed", type=int, default=ST.DEFAULT_SEED, help="bootstrap seed")
    ap.add_argument("--exclude-facts", default=None, help="file of fact_ids (one per line) to drop everywhere")
    ap.add_argument("--no-tracking", action="store_true", help="don't upload the analysis to HF / wandb")
    ap.add_argument("--compat", choices=list(COMPAT_MODES), default="strict",
                    help="evaluation-metadata compatibility mode (finding 6): strict = every arm CSV needs a consistent "
                         ".meta.json (default); legacy = CSVs without meta are accepted (historical data; everything recorded is "
                         "still checked); historical-builds = probe/build hashes may differ by design (needs --compat-reason)")
    ap.add_argument("--compat-reason", default=None,
                    help="why the builds differ (written into analysis.md and compatibility.json); required with "
                         "--compat historical-builds")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="finding 7: proceed on the common items when eval-schema-v2 arms disagree on item coverage "
                         "(default: stop with the missingness report)")
    ap.add_argument("--allow-incomplete-replicates", action="store_true",
                    help="review 2026-11-11 finding 3: proceed when a replicate lacks an arm that another replicate has (e.g. a "
                         "training queue still running). Incomplete replicates are analysed on their own, never enter a "
                         "cross-replicate summary or figure, and are used only for the same-arm noise floor (default: stop)")
    ap.add_argument("--no-supplement", action="store_true",
                    help="ignore supp_beliefs_arm<ARM>.csv (eval_beliefs.py --supplement) next to the main evaluations")
    ap.add_argument("--allow-unequal-horizon", default=None, metavar="REASON",
                    help="recheck finding 3: accept checkpoints with different step / total_batch_size / num_iterations / "
                         "max_seq_len as a DECLARED unequal-budget experiment (the reason is recorded); default: stop")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.compat == "historical-builds" and not (args.compat_reason or "").strip():
        raise SystemExit("--compat historical-builds requires --compat-reason \"...\" (the reason is written into analysis.md)")
    T.load_env()
    eval_dir = os.path.abspath(args.eval_dir)
    out_dir = os.path.join(eval_dir, "analysis_v2")
    for sub in ("tables", "cohorts", "figures"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    exclude = set()
    if args.exclude_facts:
        exclude = {l.strip() for l in open(args.exclude_facts) if l.strip()}

    probe_dir = args.probe_dir or os.path.join(eval_dir, os.pardir, "probe_sets")
    probes = CH.load_probes(probe_dir)
    summary_path = args.build_summary or os.path.join(probe_dir, os.pardir, "build_summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else None
    protocol_cfg = CH.protocol_cfg_from_summary(summary, probes)
    manifest = None
    manifest_note = "none"
    if args.manifest:
        manifest = CH.load_manifest(args.manifest, fact_ids={r["fact_id"] for r in probes.injected})
        manifest_note = (f"{args.manifest} (v2 paraphrase indices for {len(manifest)} facts)" if manifest
                         else f"{args.manifest} (legacy manifest without paraphrase_index: not used)")

    replicates = discover_replicates(eval_dir)
    if not replicates:
        raise SystemExit(f"no beliefs_arm*.csv found in {eval_dir} or its seed_*/rep* subdirectories")
    loaded = []
    for rep, rep_dir in replicates:
        arms_raw = load_replicate(rep_dir, exclude)
        if arms_raw:
            loaded.append((rep, rep_dir, arms_raw))
    if not loaded:
        raise SystemExit(f"no arm CSVs with rows found under {eval_dir}")
    # incomplete replicates (review 2026-11-11 finding 3): a replicate that lacks an arm the others have (a queue
    # still running) must never enter a cross-replicate mean -- per-arm summaries would average different seed sets
    all_arms = [a for a in ARMS if any(a in arms for _, _, arms in loaded)]
    incomplete = {rep: [a for a in all_arms if a not in arms] for rep, _, arms in loaded if any(a not in arms for a in all_arms)}
    if incomplete and not args.allow_incomplete_replicates:
        raise SystemExit("INCOMPLETE REPLICATES (review 2026-11-11 finding 3): "
                         + "; ".join(f"{rep} lacks arm(s) {', '.join(m)}" for rep, m in incomplete.items())
                         + f" (arms present somewhere: {', '.join(all_arms)}). Wait for the missing runs, point --eval-dir at "
                           "the complete replicates, or pass --allow-incomplete-replicates: incomplete replicates are then "
                           "analysed on their own, excluded from every cross-replicate summary and figure, and used only for "
                           "the same-arm noise floor.")

    # finding 6 (2026-10-14) + recheck findings 1-3: metadata compatibility and the binding of every
    # evaluation to THIS analysis' inputs, BEFORE anything is aggregated
    inputs = analysis_inputs(probes, summary_path, args.manifest, args.exposure_dir)
    if inputs["manifest_sha256"] and inputs["marker_manifest_sha256"] and inputs["manifest_sha256"] != inputs["marker_manifest_sha256"]:
        raise SystemExit(f"--manifest {args.manifest} (sha256 {inputs['manifest_sha256'][:12]}) is not the manifest of the build "
                         f"{inputs['build_summary_path']} (BUILD_COMPLETE.json records {inputs['marker_manifest_sha256'][:12]}); "
                         "memorization classifications would come from another build")
    for cand in inputs["exposure_candidates"]:
        where = args.exposure_dir if len(inputs["exposure_candidates"]) == 1 else cand["dir"]
        if cand["build_summary_sha256"] and inputs["build_summary_sha256"] and cand["build_summary_sha256"] != inputs["build_summary_sha256"]:
            raise SystemExit(f"--exposure-dir {where} was measured for build summary {cand['build_summary_sha256'][:12]} "
                             f"but the analysis uses {inputs['build_summary_sha256'][:12]} ({summary_path})")
        if cand["manifest_sha256"] and inputs["manifest_sha256"] and cand["manifest_sha256"] != inputs["manifest_sha256"]:
            raise SystemExit(f"--exposure-dir {where} was measured for manifest {cand['manifest_sha256'][:12]} "
                             f"but --manifest {args.manifest} has {inputs['manifest_sha256'][:12]}")
    compat = check_compatibility(loaded, args.compat, args.compat_reason, inputs=inputs,
                                 unequal_horizon_reason=args.allow_unequal_horizon)
    compat_path = os.path.join(out_dir, "compatibility.json")
    with open(compat_path, "w", encoding="utf-8") as fh:
        json.dump(compat, fh, indent=1, default=str)
        fh.write("\n")
    if not compat["ok"]:
        raise SystemExit("EVALUATION METADATA INCOMPATIBLE (review 2026-10-14 finding 6 / recheck findings 1-3); nothing was aggregated. "
                         f"Report: {compat_path}\n  - " + "\n  - ".join(compat["violations"]) +
                         "\nHistorical CSVs without .meta.json need --compat legacy; builds whose probe/build hashes differ "
                         "by design need --compat historical-builds --compat-reason \"...\"; a declared unequal-budget "
                         "experiment needs --allow-unequal-horizon \"reason\".")
    # supplementary evaluations (eval_beliefs.py --supplement), bound to the main ones before anything is aggregated
    supp_loaded = {}
    if not args.no_supplement:
        for rep, rep_dir, arms_raw in loaded:
            sup = load_supplement(rep_dir, arms_raw, exclude)
            if sup:
                supp_loaded[rep] = sup
    supp_report = check_supplement(loaded, supp_loaded, compat, inputs)
    compat["supplement"] = supp_report
    with open(compat_path, "w", encoding="utf-8") as fh:
        json.dump(compat, fh, indent=1, default=str)
        fh.write("\n")
    if not supp_report["ok"]:
        raise SystemExit("SUPPLEMENTARY EVALUATIONS INCOMPATIBLE with the main evaluations; nothing was aggregated. "
                         f"Report: {compat_path}\n  - " + "\n  - ".join(supp_report["violations"])
                         + "\nRe-score them with eval_beliefs.py --supplement, or pass --no-supplement to analyse the main "
                           "evaluations only.")
    eval_info = eval_info_from_replicates(loaded)
    schema_v2 = max(int(a["schema"]) for _, _, arms in loaded for a in arms.values()) >= 2

    # the exposure index of each replicate: the one its checkpoints were bound to above; an unbound analysis
    # (legacy metas, an index without identity) keeps using the single index given, as before
    exposure_cache, exposure_by_rep, exposure_dirs = {}, {}, {}
    for rep, _, _ in loaded:
        dirs = sorted({s.get("exposure_dir") for s in compat["replicates"].get(rep, {}).values() if s.get("exposure_dir")})
        d = dirs[0] if len(dirs) == 1 else None
        if d is None and args.exposure_dir:
            cands = inputs["exposure_candidates"]
            d = cands[0]["dir"] if len(cands) == 1 else (os.path.abspath(args.exposure_dir) if not cands else None)
        if d is not None and d not in exposure_cache:
            exposure_cache[d] = CH.load_exposure_dir(d)
        exposure_by_rep[rep] = exposure_cache.get(d, {})
        exposure_dirs[rep] = d

    cohorts = CH.build_cohorts(probes, exclude, f"excluded via --exclude-facts ({args.exclude_facts})", eval_info=eval_info)
    selected = list(args.cohort) if args.cohort else list(CH.COHORT_NAMES)
    headline = "all_injected" if probes.is_v2 else "exclusive_value"
    if headline not in selected:   # None when no injected cohort was selected: headline-only sections are skipped
        headline = next((c for c in CH.INJECTED_COHORTS if c in selected), None)

    R = Report(mock=compat["mock"])
    R.say("# Belief analysis v2\n")
    if compat["mock"]:
        R.say(f"**{MOCK_BANNER}**\n")
    R.say("| input | value |\n|---|---|")
    R.say(f"| eval dir | {eval_dir} |")
    R.say(f"| replicates | {len(replicates)}: " + ", ".join(f"{n} ({d})" for n, d in replicates) + " |")
    R.say(f"| eval CSV schema | {'v2 (corrected evaluator)' if schema_v2 else 'legacy'}; compatibility mode `{args.compat}`"
          + (f"; reason: {args.compat_reason}" if args.compat_reason else "") + (" ; MOCK SCORES" if compat["mock"] else "") + " |")
    R.say(f"| probe files | {probes.paths['injected']} (sha256 {sha256_file(probes.paths['injected'])[:12]}), "
          f"{probes.paths['heldout']} (sha256 {sha256_file(probes.paths['heldout'])[:12]}) |")
    R.say(f"| probe protocol | {probes.protocol} |")
    R.say(f"| build summary | {summary_path if summary else 'none found'} |")
    R.say(f"| render config | {json.dumps({k: v for k, v in protocol_cfg.items() if k != 'assumed_defaults'})} |")
    R.say(f"| manifest | {manifest_note} |")
    R.say("| exposure index | " + ("; ".join(
        f"{rep}: {exposure_dirs[rep]} ({', '.join(f'{a}: {len(v)} facts' for a, v in exposure_by_rep[rep].items()) or 'no rows'})"
        for rep in exposure_by_rep if exposure_dirs.get(rep)) or "none") + " |")
    if incomplete:
        R.say("| incomplete replicates | " + "; ".join(f"{rep} lacks {', '.join(m)}" for rep, m in incomplete.items())
              + " (--allow-incomplete-replicates: excluded from every cross-replicate summary and figure) |")
    R.say(f"| excluded fact ids | {len(exclude)}{' (' + args.exclude_facts + ')' if args.exclude_facts else ''} |")
    R.say(f"| bootstrap | {args.bootstrap} resamples, seed {args.seed}, clusters = semantic groups |")
    R.say(f"| equivalence margin | {args.equivalence_margin if args.equivalence_margin is not None else 'none (no equivalence claims)'} |")
    rev = code_revision()
    R.say(f"| code | commit {rev.get('git_commit')} dirty={rev.get('git_dirty')} branch={rev.get('git_branch')} |")
    R.say(f"| output | {out_dir} |\n")

    R.say("## 0a. Evaluation metadata compatibility (finding 6, 2026-10-14; recheck findings 1-3)\n")
    R.say("Analysis inputs every evaluation is bound to:\n")
    R.say("| input | path | sha256 |\n|---|---|---|")
    for k in PROBE_HASH_KEYS:
        R.say(f"| {k} | {inputs['probe_paths'][k]} | {inputs['probe_hashes'][k][:12]} |")
    R.say(f"| build_summary.json | {inputs['build_summary_path'] or 'none'} | {(inputs['build_summary_sha256'] or '-')[:12]} |")
    R.say(f"| manifest | {inputs['manifest_path'] or 'none'} | {(inputs['manifest_sha256'] or '-')[:12]}"
          + (" (= BUILD_COMPLETE.json)" if inputs["manifest_sha256"] and inputs["manifest_sha256"] == inputs["marker_manifest_sha256"] else "") + " |")
    if inputs["exposure_candidates"]:
        for cand in inputs["exposure_candidates"]:
            R.say(f"| exposure (data seed {cand['data_seed'] if cand['data_seed'] is not None else 'unrecorded'}) | {cand['preflight_path']} | "
                  f"preflight {(cand['preflight_sha256'] or '-')[:12]}; "
                  f"{', '.join(f'{a}: {str(v)[:12]}' for a, v in sorted(cand['prepack_dataset_id'].items())) or '-'} |")
    else:
        R.say(f"| exposure | {inputs['exposure_dir'] or 'none'} | - |")
    R.say(f"| scored items with one identity across every CSV | {compat['n_items_with_identity']} | "
          f"{compat['n_item_identity_mismatches']} mismatches |")
    if args.allow_unequal_horizon:
        R.say(f"| declared unequal-budget experiment | {args.allow_unequal_horizon} | |")
    R.say()
    mode_line = f"Compatibility mode: `{compat['compat_mode']}`"
    if compat["compat_mode"] == "legacy":
        R.say(mode_line + f" -- legacy compatibility mode: {compat['n_csv'] - compat['n_meta']} of {compat['n_csv']} arm CSVs have "
              "no .meta.json (historical data); their mode, tokenizer, probe hashes and checkpoint identity are unverified. "
              "Everything that is recorded was checked.\n")
    elif compat["compat_mode"] == "historical-builds":
        R.say(mode_line + " -- historical-builds compatibility mode: the arms were scored on builds whose probe/build hashes "
              f"differ BY DESIGN. Reason given (--compat-reason): {compat['compat_reason']}\n")
    else:
        R.say(mode_line + f": every arm CSV has a .meta.json and every check passed ({compat['n_meta']} metas).\n")
    if compat["mock"]:
        R.say(f"**{MOCK_BANNER}**\n")
    crows = []
    for rep, arms in compat["replicates"].items():
        for arm, s in arms.items():
            ih, ck = s.get("input_hashes") or {}, s.get("checkpoint") or {}
            crows.append({"replicate": rep, "arm": arm, "meta": "yes" if s["meta_present"] else "no", "mode": s.get("mode") or "-",
                          "schema": s["schema"], "tokenizer": (s.get("tokenizer_sha256") or "-")[:12],
                          "probes_injected": (ih.get(PROBE_HASH_KEYS[0]) or "-")[:12],
                          "probes_heldout": (ih.get(PROBE_HASH_KEYS[1]) or "-")[:12],
                          "inputs_bound": {True: "yes", False: "NO", None: "-"}[s.get("inputs_bound")],
                          "exposure": s.get("exposure_binding") or "-",
                          "scoring_policy": s.get("scoring_policy") or "-", "protocol": s.get("protocol") or "-",
                          "probe_construction": s.get("probe_construction") or "-",
                          "csv_sha256_ok": {True: "yes", False: "NO", None: "-"}[s.get("csv_sha256_ok")],
                          "step": ck.get("step"), "model_sha256": (str(ck.get("model_sha256")) if ck.get("model_sha256") else "-")[:12],
                          "init_param_sha256": (str(ck.get("init_param_sha256")) if ck.get("init_param_sha256") else "-")[:12],
                          "init_seed": ck.get("init_seed"), "data_seed": ck.get("data_seed"),
                          "total_batch_size": ck.get("total_batch_size"), "num_iterations": ck.get("num_iterations"),
                          "loss_normalization": ck.get("loss_normalization") or ("-" if s.get("mode") != "model_inference" else "unrecorded")})
    R.table("compatibility", crows, "all",
            note="per arm CSV: what its .meta.json records (hashes truncated to 12 hex); full report in compatibility.json")
    for w in compat["warnings"]:
        R.say(f"- warning: {w}")
    R.say()
    if supp_loaded:
        R.say(f"Supplementary evaluations (`{SUPPLEMENT_PREFIX}beliefs_arm<ARM>.csv`, version {supp_report['version']}): "
              + "; ".join(f"{rep}: arms {', '.join(sorted(v))}" for rep, v in supp_loaded.items())
              + f" -- {supp_report['n_items']} items with one identity across every file; each is bound to the checkpoint of "
                "the main evaluation of its arm (section 5b).")
        without = [rep for rep, _, _ in loaded if rep not in supp_loaded]
        if without:
            R.say(f"- replicates without a supplementary evaluation: {', '.join(without)} (their section 5b is absent and they do "
                  "not enter the cross-replicate rows of the supplementary passes)")
        for w in supp_report["warnings"]:
            R.say(f"- warning: {w}")
        R.say()
    elif not args.no_supplement:
        R.say(f"No supplementary evaluation found (`{SUPPLEMENT_PREFIX}beliefs_arm<ARM>.csv`; eval_beliefs.py --supplement): the "
              "wrong-subject control and the arm-E own-frame probe are not part of this report.\n")

    R.say("## 0. Cohorts\n")
    rows = [{"cohort": c.name, "probe_set": c.probe_set, "n": c.n, "n_groups": len(set(c.groups().values())),
             "n_by_tier": json.dumps(c.n_by_tier()), "n_excluded": len(c.exclusions), "selected": c.name in selected,
             "definition": c.definition} for c in cohorts.values()]
    R.table("cohorts", rows, "all", note="saved with ids, denominators, exclusions and missingness under cohorts/<name>.json",
            show_cols=["cohort", "probe_set", "n", "n_groups", "n_by_tier", "n_excluded", "selected"])
    for c in cohorts.values():
        R.say(f"- **{c.name}** (n={c.n}): {c.definition}")
    R.say()
    if headline is None:
        R.say("Headline cohort: none (no injected cohort selected via --cohort); injected-only sections are skipped.\n")
    elif probes.is_v2:
        R.say(f"Headline cohort: **{headline}** -- v2 builds inject exactly one value per semantic group and hold out "
              "whole groups, so all_injected is already free of same-group rivals (it coincides with exclusive_value).\n")
    else:
        rv, ev = cohorts["rival_untrained"], cohorts["exclusive_value"]
        R.say(f"Headline cohort: **{headline}** -- v1 pools reuse synthetic subjects (finding 2): of the "
              f"{rv.n} rival_untrained facts, {rv.n - ev.n} still have another injected value in their group; only "
              f"the {ev.n} exclusive_value facts ({ev.n_by_tier()}) are free of same-group interference. all_injected "
              f"(n={cohorts['all_injected'].n}) and rival_untrained are reported as sensitivity cohorts.\n")
    if headline is not None:
        R.say("Label x dose denominators of the headline cohort: " + json.dumps(cohorts[headline].n_by_label_and_dose()) + "\n")

    an = Analysis(args, probes, cohorts, selected, headline, protocol_cfg, exposure_by_rep, manifest, R, out_dir=out_dir,
                  schema_v2=schema_v2, incomplete=incomplete)
    arm_names = []
    for rep, rep_dir, arms_raw in loaded:
        arm_names = sorted(set(arm_names) | set(arms_raw))
        an.run_replicate(rep, rep_dir, arms_raw, supp_raw=supp_loaded.get(rep))
    an.summarize_replicates()

    R.say("## 7. Equivalence\n")
    if args.equivalence_margin is None:
        R.say(NO_EQUIVALENCE_NOTE + "\n")
    else:
        eq = [r for t in R.tables.values() for r in t["rows"] if "tost_equivalent" in r]
        n_eq = sum(1 for r in eq if r["tost_equivalent"])
        R.say(f"TOST with margin {args.equivalence_margin}: {n_eq} of {len(eq)} paired cells have their 90% cluster-bootstrap "
              "CI inside (-M, M); see the `tost_equivalent` column of the paired tables.\n")

    plots = []
    if args.plot:
        R.say("## Figures\n")
        plots = make_figures(an, out_dir, arm_names)
        R.say()

    # write outputs ------------------------------------------------------------------
    for name, c in cohorts.items():
        c.save(os.path.join(out_dir, "cohorts", f"{name}.json"))
    for name, t in R.tables.items():
        write_csv(os.path.join(out_dir, "tables", f"{name}.csv"), t["rows"])
    R.say("## Files written\n")
    R.say(f"- {out_dir}/analysis.md (this report), analysis.txt (console log), compatibility.json (finding 6 report)")
    R.say(f"- {len(R.tables)} tables under tables/, {len(cohorts)} cohorts under cohorts/, {len(plots)} figures under figures/")
    if compat["mock"]:
        R.say(f"- {MOCK_BANNER}")
    with open(os.path.join(out_dir, "analysis.md"), "w", encoding="utf-8") as fh:
        fh.write(R.markdown())
    print(f"\nanalysis -> {out_dir}")
    return {"args": args, "plot": plots[0] if plots else None, "plots": plots, "out_dir": out_dir,
            "tables": sorted(R.tables), "cohorts": {n: c.n for n, c in cohorts.items()}, "headline": headline,
            "compatibility": compat, "mock": compat["mock"]}


# ======================================================================================
# tracking (unchanged mechanism: tee the console, upload text + figures + CSVs, log to wandb)

class _Tee(io.StringIO):
    def write(self, s):
        sys.__stdout__.write(s)
        return super().write(s)

    def flush(self):
        sys.__stdout__.flush()


def track_analysis(result, text):
    """Upload the analysis (console text, analysis.md, figures, tables, cohorts, arm CSVs) to
    <build_repo>/<exp_tag>/analysis_v2/ and a wandb 'analysis' run."""
    args = result["args"]
    eval_dir = os.path.abspath(args.eval_dir)
    exp_dir = os.path.dirname(eval_dir)
    tag = T.exp_tag(exp_dir)
    out_dir = result["out_dir"]
    txt = os.path.join(out_dir, "analysis.txt")
    with open(txt, "w") as fh:
        fh.write(text)
    files = {f"{tag}/analysis_v2/analysis.txt": txt}
    for extra in ("analysis.md", "compatibility.json"):
        p = os.path.join(out_dir, extra)
        if os.path.exists(p):
            files[f"{tag}/analysis_v2/{extra}"] = p
    for png in result.get("plots", []):
        files[f"{tag}/analysis_v2/figures/{os.path.basename(png)}"] = png
    for sub in ("tables", "cohorts"):
        d = os.path.join(out_dir, sub)
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d)):
                files[f"{tag}/analysis_v2/{sub}/{fn}"] = os.path.join(d, fn)
    for arm_csv in sorted(os.listdir(eval_dir)):
        if arm_csv.startswith("beliefs_arm") and arm_csv.endswith(".csv"):
            files[f"{tag}/analysis_v2/{arm_csv}"] = os.path.join(eval_dir, arm_csv)
    hf_url = None
    if T.hf_enabled():
        hf_url, sha = T.hf_upload_files(T.build_repo(), files, repo_type="dataset",
                                        commit_message=f"analysis v2 {tag}")
        hf_url = f"{hf_url}/tree/main/{tag}/analysis_v2"
        print(f"analysis -> {hf_url} (commit {sha[:8]})")
    if T.wandb_enabled():
        import wandb
        # the arm runs come from the evaluations' metas (the checkpoint each one scored), per replicate
        run_ids = {}
        for rep, arms in (result.get("compatibility") or {}).get("replicates", {}).items():
            for a, s in arms.items():
                rid = (s.get("run") or {}).get("wandb_run_id")
                if rid:
                    run_ids[f"{rep}/{a}"] = rid
        if not run_ids:
            run_ids = {a: T.read_run_info(exp_dir, a).get("wandb_run_id") for a in "CRXE"}
        run = T.wandb_init(f"ctx_analysis_{tag}", job_type="analysis", tags=(tag,),
                           config={"exp_tag": tag, "eval_dir": eval_dir, "hf_analysis": hf_url,
                                   "arm_runs": {a: r for a, r in run_ids.items() if r},
                                   "cohorts": result.get("cohorts"), "headline_cohort": result.get("headline")})
        log = {"analysis_text": wandb.Html(f"<pre>{text}</pre>")}
        for png in result.get("plots", []):
            log[os.path.splitext(os.path.basename(png))[0]] = wandb.Image(png)
        run.log(log)
        run.finish()
        print(f"analysis -> wandb run {run.id} ({run.url})")


if __name__ == "__main__":
    buf = _Tee()
    with redirect_stdout(buf):
        result = main()
    if result:
        # the console log is always kept next to the report (never the legacy <eval-dir>/analysis.txt)
        with open(os.path.join(result["out_dir"], "analysis.txt"), "w") as fh:
            fh.write(buf.getvalue())
        if not result["args"].no_tracking and T.tracking_enabled():
            track_analysis(result, buf.getvalue())
