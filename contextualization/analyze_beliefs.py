#!/usr/bin/env python
"""analyze_beliefs.py -- cohort-explicit analysis (v2) of the per-arm belief CSVs from eval_beliefs.py.

Replaces the July-2026 analysis after the 2026-10-13 review (findings 2, 10, 11, 12, 13, 16, 17).
Nothing here rescored a checkpoint or retrained anything: it re-reads saved scores.

Per-fact belief
  belief = logp_value - logp_competing, averaged over the cloze templates of a fact within
  (probe_set, pass, fact) -- exactly as the legacy analysis did. Three scoring metrics are
  reported separately and never mixed:
    answer_plus_suffix  the legacy columns (answer tokens + the template suffix), cloze passes
    answer_only         v2 columns logp_value_answer / logp_competing_answer, when present
    full_sentence       the held-out-paraphrase pass (whole sentence vs whole sentence)
  logP(value) and logP(competing) are reported separately next to every contrast (finding 13).

Cohorts (ctx/cohorts.py; each saved to analysis_v2/cohorts/<name>.json with ids, denominators,
exclusions, missingness): all_injected, rival_untrained, exclusive_value, heldout_claims,
heldout_groups, heldout_rivals, counterfact, paraphrase_wellformed. Every table, test and
figure names its cohort and denominators (finding 10).

Statistics (ctx/stats.py): paired per-fact differences between arms inside one cell
(cohort x dose x label): mean, scipy Wilcoxon (two-sided; None below 10 non-zero diffs), a
cluster bootstrap 95% CI over semantic groups, Holm within one family = one
(contrast, cohort, pass, metric) table (finding 11). Equivalence is claimed only under
--equivalence-margin (TOST with the 90% bootstrap CI). Training-run uncertainty comes only
from replicates (seed_*/ or rep*/ subdirectories), summarized as mean and SD across runs;
facts are never pooled across replicates.

Memorization (finding 12): template-0 probes are joined to the forms actually rendered at the
assigned dose (ctx.cohorts.realized_training_forms; manifest/exposure aware) and the
template-0 vs other-template contrast is reported per subgroup (verbatim_trained / not).

CounterFact (finding 16): per-arm accuracy, paired per-item change vs C with intervals,
preference reversals in both directions, Wilcoxon on the paired deltas.

Outputs: <eval-dir>/analysis_v2/{analysis.md, analysis.txt, tables/*.csv, cohorts/*.json,
figures/*.png}. The legacy analysis.txt and beliefs_dose_response.png in <eval-dir> are never
touched.

Usage (from contextualization/):
  python analyze_beliefs.py --eval-dir <exp>/eval [--probe-dir <exp>/probe_sets]
      [--build-summary <exp>/build_summary.json] [--manifest <exp>/manifest.csv[.gz]]
      [--exposure-dir <exp>/exposure] [--cohort NAME ...] [--plot] [--bootstrap N]
      [--equivalence-margin M] [--seed S] [--exclude-facts ids.txt] [--no-tracking]
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
from ctx.protocol import PROTOCOL_V2, ProtocolError, code_revision, sha256_file  # noqa: E402
from ctx.stats import wilcoxon_signed_rank, legacy_wilcoxon_signed_rank  # noqa: E402,F401  (re-exported; the auditor's script imports wilcoxon_signed_rank from here)

ARMS = ("C", "R", "X", "E")
CONTRASTS = (("R", "C"), ("X", "C"), ("R", "X"), ("R", "E"), ("E", "C"), ("X", "E"))
ATTRIBUTED_PASSES = ("cloze_attributed", "cloze_attributed_fixedsource")
PARAPHRASE_PASS = "heldout_paraphrase"
LABEL_ORDER = ("true", "false", "contested", "unknown")
ARM_COLORS = {"C": "#2a78d6", "R": "#eb6834", "X": "#1baf7a", "E": "#eda100"}   # fixed categorical slots
NO_EQUIVALENCE_NOTE = ("No equivalence claim is made from any non-significant difference: without "
                       "--equivalence-margin M no TOST is run, and 'not significant' is not 'equal'.")
REPLICATE_NOTE = ("Per-fact tests and bootstrap intervals describe within-run uncertainty only (facts "
                  "sharing a training run are not independent training replicates). Uncertainty across "
                  "training runs is estimated only from replicates and is unmeasured with one run per arm.")


# ======================================================================================
# loading

def _float(s):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def discover_replicates(eval_dir):
    """[(name, dir)]: the eval dir itself ("rep0") when it holds beliefs_arm<A>.csv, else its
    seed_*/ or rep*/ subdirectories that do."""
    if glob.glob(os.path.join(eval_dir, "beliefs_arm*.csv")):
        return [("rep0", eval_dir)]
    reps = []
    for d in sorted(os.listdir(eval_dir)):
        p = os.path.join(eval_dir, d)
        if os.path.isdir(p) and (d.startswith("seed_") or d.startswith("rep")) and glob.glob(os.path.join(p, "beliefs_arm*.csv")):
            reps.append((d, p))
    return reps


def load_arm_csv(path, exclude=frozenset()):
    """One arm's CSV (legacy 13 columns or eval schema v2) ->
    {"data": {(probe_set, pass): {fact_id: {"meta":..., "templates": {ti: {lv, lc[, lva, lca]}}}}},
     "schema", "has_answer_only", "n_rows", "n_unscored", "path"}"""
    data, n_rows, n_unscored, schema = {}, 0, 0, 1
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        has_answer = "logp_value_answer" in fields
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
            f["templates"][int(row.get("template_idx") or 0)] = t
            sv = row.get("schema_version")
            if sv:
                try:
                    schema = max(schema, int(sv))
                except ValueError:
                    pass
    return {"data": data, "schema": schema, "has_answer_only": has_answer, "n_rows": n_rows,
            "n_unscored": n_unscored, "path": path}


def load_replicate(rep_dir, exclude=frozenset()):
    arms = {}
    for arm in ARMS:
        p = os.path.join(rep_dir, f"beliefs_arm{arm}.csv")
        if os.path.exists(p):
            arms[arm] = load_arm_csv(p, exclude)
    return arms


def metric_for_pass(pass_):
    return "full_sentence" if pass_ == PARAPHRASE_PASS else "answer_plus_suffix"


def per_fact(arm, probe_set, pass_, metric, tfilter=None):
    """fact_id -> {belief, logp_value, logp_competing, n_templates}; templates averaged within the fact."""
    out = {}
    for fid, f in arm["data"].get((probe_set, pass_), {}).items():
        lvs, lcs = [], []
        for ti, t in f["templates"].items():
            if tfilter is not None and not tfilter(ti):
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
    """Collects markdown text (printed to the console as it is produced) and named tables."""

    def __init__(self):
        self.lines = []
        self.tables = {}    # name -> {"rows": [...], "cohort": str, "note": str}
        self.figures = []

    def say(self, text=""):
        print(text)
        self.lines.append(text)

    def table(self, name, rows, cohort, note="", show=True, show_cols=None, max_rows=None):
        tab = self.tables.setdefault(name, {"rows": [], "cohort": cohort, "note": note})
        tab["rows"].extend(rows)
        if show and rows:
            self.say(f"Table `{name}` -- cohort: {cohort}" + (f". {note}" if note else ""))
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
    def __init__(self, args, probes, cohorts, selected, headline, protocol_cfg, exposure, manifest, report):
        self.args, self.probes, self.cohorts = args, probes, cohorts
        self.selected, self.headline, self.protocol_cfg = selected, headline, protocol_cfg
        self.exposure, self.manifest, self.R = exposure, manifest, report
        self.n_boot, self.seed, self.margin = args.bootstrap, args.seed, args.equivalence_margin
        self.per_rep = {}          # replicate -> {"paired": {key: row}, "dose": {key: row}, "cf": {...}}
        self.first_rows = {}       # for figures: replicate -> table rows

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
        # pooled row (all labels, all doses): descriptive, outside the Holm family
        if len(rows) != 1:
            row = self._paired_cell(rep, cohort, t1, t2, a1, a2, pass_, metric, tier, None, None, "all")
            if row is not None:
                row.update({"family": "", "p_holm": None, "family_size": 0})
                rows.append(row)
        return rows

    # ---- per replicate -----------------------------------------------------------------
    def run_replicate(self, rep, rep_dir, arms_raw):
        R, args = self.R, self.args
        arm_names = sorted(arms_raw)
        has_answer_only = any(a["has_answer_only"] for a in arms_raw.values())
        store = self.per_rep.setdefault(rep, {"paired": {}, "dose": {}, "cf": {}, "arms": arm_names})
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

        # missingness ----------------------------------------------------------------
        for name, co in self.cohorts.items():
            miss = {}
            for a in arm_names:
                for (ps, pa), facts in arms_raw[a]["data"].items():
                    if ps != co.probe_set:
                        continue
                    lacking = [fid for fid in co.fact_ids if fid not in facts]
                    if lacking:
                        miss.setdefault(a, {})[pa] = {"n_missing": len(lacking), "fact_ids": lacking}
            co.missingness[rep] = miss
        R.say("Missingness (cohort facts lacking a per-fact score in some arm/pass; also saved in cohorts/*.json):")
        any_missing = False
        for name, co in self.cohorts.items():
            for a, per_pass in co.missingness[rep].items():
                for pa, m in per_pass.items():
                    any_missing = True
                    R.say(f"- {name}: arm {a}, pass {pa}: {m['n_missing']} of {co.n} facts unscored")
        if not any_missing:
            R.say("- none")
        R.say()

        # per-fact tables cache
        cache = {}

        def tab(arm, ps, pa, metric, tf=None, tfname=""):
            k = (arm, ps, pa, metric, tfname)
            if k not in cache:
                cache[k] = per_fact(arms_raw[arm], ps, pa, metric, tf)
            return cache[k]

        inj_sel = [c for c in CH.INJECTED_COHORTS if c in self.selected]
        held_sel = [c for c in CH.HELDOUT_COHORTS if c in self.selected]

        # 1. dose-response --------------------------------------------------------------
        R.say("## 1. Dose-response: mean belief by arm x dose x label (cloze passes)\n")
        R.say("belief = logP(value) - logP(competing), averaged over templates within a fact; error bars / CI "
              "columns are cluster-bootstrap 95% intervals over semantic groups "
              f"(n_boot={self.n_boot}, seed={self.seed}). `mean_logp_value` / `mean_logp_competing` are the two "
              "components reported separately (finding 13).\n")
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
                            note="never-injected facts: the per-arm baseline of the belief contrast",
                            show=(pa == "cloze_neutral" and metric == "answer_plus_suffix"),
                            show_cols=["label", "arm", "n", "n_groups", "mean_belief", "ci95_lo", "ci95_hi", "sd_belief",
                                       "mean_logp_value", "mean_logp_competing"])

        # 2. paired contrasts ------------------------------------------------------------
        R.say("## 2. Paired per-fact contrasts between arms (one Holm family per table)\n")
        R.say("Each table = one (contrast, cohort, pass, metric); its dose x label cells form the prespecified Holm "
              "family (`p_wilcoxon` raw, `p_holm` adjusted, `family_size` = tested cells). `p_wilcoxon` is the "
              "two-sided scipy Wilcoxon signed-rank p on the per-fact paired differences (n/a below 10 non-zero "
              "differences). CIs: cluster bootstrap over semantic groups. The `all/all` row is descriptive and outside "
              "the family. " + (f"TOST equivalence margin: {self.margin}." if self.margin is not None else NO_EQUIVALENCE_NOTE) + "\n")
        show_cols = ["tier", "label", "dose", "n", "n_groups", "mean_diff", "ci95_lo", "ci95_hi", "p_wilcoxon", "p_holm",
                     "n_pos", "n_neg", "n_pref_flips"]
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
        if "paraphrase_wellformed" in self.selected and any(("injected", PARAPHRASE_PASS) in a["data"] for a in arms_raw.values()):
            jobs.append(("paraphrase_wellformed", "injected", PARAPHRASE_PASS, "full_sentence", True, None))
        for cname, ps, pa, metric, by_dose, tier in jobs:
            co = self.cohorts[cname]
            for a1, a2 in contrasts:
                arms_t = {a: tab(a, ps, pa, metric) for a in (a1, a2)}
                rows = self.paired_rows(rep, co, arms_t, a1, a2, pa, metric, by_dose=by_dose, tier=tier)
                for r in rows:
                    store["paired"][(r["contrast"], r["cohort"], r["pass"], r["metric"], r["tier"], r["label"], r["dose"])] = r
                if tier is None:
                    show = (metric != "answer_only") and (pa in ("cloze_neutral", "wrapper_conditional", PARAPHRASE_PASS)
                                                          or ps == "heldout")
                    name, note = f"paired__{a1}-{a2}__{cname}__{pa}__{metric}", f"contrast {a1}-{a2}; all entity tiers; n = facts scored in both arms"
                else:
                    show = cname == self.headline and tier == "novel"
                    name = f"paired__{a1}-{a2}__{cname}__{pa}__{metric}__tier-{tier}"
                    note = f"contrast {a1}-{a2}; entity tier {tier} only (secondary family); n = facts scored in both arms"
                R.table(name, rows, cname, note=note, show=show, show_cols=show_cols)

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
            for cname in [c for c in ("paraphrase_wellformed", "all_injected") if c in self.selected]:
                co = self.cohorts[cname]
                arms_t = {a: tab(a, "injected", PARAPHRASE_PASS, "full_sentence") for a in arm_names}
                rows = self.arm_mean_rows(rep, co, arms_t, PARAPHRASE_PASS, "full_sentence", pooled=True)
                keep_dose(rows)
                note = ("well-formed contrasts only" if cname == "paraphrase_wellformed" else
                        "LEGACY COMPARISON ONLY: includes the malformed contrasts (finding 8); not used for tests")
                R.table(f"paraphrase__{cname}__{PARAPHRASE_PASS}__full_sentence", rows, cname, note=note,
                        show_cols=["label", "dose", "arm", "n", "n_groups", "mean_belief", "ci95_lo", "ci95_hi",
                                   "mean_logp_value", "mean_logp_competing"],
                        max_rows=None if cname == "paraphrase_wellformed" else 3 * len(arm_names))
            R.say("Paired paraphrase contrasts: see `paired__*__paraphrase_wellformed__heldout_paraphrase__full_sentence` above.\n")

        R.say("### 4b. Template-0 probe joined to the realized training forms (finding 12)\n")
        R.say("A fact is `verbatim_trained` when cloze template 0 with the blank filled equals a surface form "
              "actually rendered at its assigned dose (v1: training_surface_forms cycled by occurrence; manifest "
              "paraphrase indices when given; exposure index keeps only complete, in-horizon occurrences). Metric "
              "answer_plus_suffix, pass cloze_neutral; `mean_t0_minus_other` is the within-fact paired difference.\n")
        mem_cohorts = [c for c in dict.fromkeys([self.headline, "all_injected"]) if c and c in self.selected]
        try:
            classes = CH.classify_verbatim_trained(self.probes.injected, self.protocol_cfg, self.exposure, self.manifest)
        except ProtocolError as e:
            classes = None
            R.say(f"NOT COMPUTABLE: {e}\n")
        if classes:
            for cname in mem_cohorts:
                co = self.cohorts[cname]
                rows = []
                for arm in arm_names:
                    src = arm if arm in classes else "nominal"
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
                                 "p_wilcoxon": cell["p_wilcoxon"],
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
                  "exceeds ordinary training-run variation is unmeasured without replicates "
                  f"({'this run has ' + str(len(self.per_rep)) + ' replicate(s) so far' if self.per_rep else ''}).\n")

    # ---- across replicates -----------------------------------------------------------
    def summarize_replicates(self):
        R = self.R
        reps = list(self.per_rep)
        R.say("## 6. Uncertainty across training replicates\n")
        R.say(REPLICATE_NOTE + "\n")
        if len(reps) < 2:
            R.say(f"Replicates found: {len(reps)} ({', '.join(reps)}). Training-run variation is therefore UNMEASURED "
                  "here; every interval and p-value in this report is within-run.\n")
            return
        R.say(f"Replicates found: {len(reps)} ({', '.join(reps)}). Each statistic below was computed per replicate; "
              "the summary is the mean and SD (ddof=1) across replicates with n_replicates, never pooling facts.\n")
        keys = sorted({k for r in reps for k in self.per_rep[r]["paired"]}, key=str)
        rows = []
        for k in keys:
            vals = {r: self.per_rep[r]["paired"].get(k, {}).get("mean_diff") for r in reps}
            s = ST.summarize_replicates(vals)
            contrast, cohort, pa, metric, tier, lab, dose = k
            rows.append({"contrast": contrast, "cohort": cohort, "pass": pa, "metric": metric, "tier": tier, "label": lab, "dose": dose,
                         "n_replicates": s["n_replicates"], "mean_of_mean_diff": s["mean"], "sd_across_replicates": s["sd"],
                         "min": s["min"], "max": s["max"],
                         "n_per_replicate": "/".join(str(self.per_rep[r]["paired"].get(k, {}).get("n", "")) for r in reps)})
        R.table("replicates__paired", rows, "per table", note="mean paired difference per replicate, summarized across replicates",
                show_cols=["contrast", "cohort", "pass", "tier", "label", "dose", "n_replicates", "mean_of_mean_diff", "sd_across_replicates"],
                max_rows=40)
        keys = sorted({k for r in reps for k in self.per_rep[r]["dose"]}, key=str)
        rows = []
        for k in keys:
            s = ST.summarize_replicates({r: self.per_rep[r]["dose"].get(k) for r in reps})
            cohort, pa, metric, tier, arm, lab, dose = k
            rows.append({"cohort": cohort, "pass": pa, "metric": metric, "tier": tier, "arm": arm, "label": lab, "dose": dose,
                         "n_replicates": s["n_replicates"], "mean_of_means": s["mean"], "sd_across_replicates": s["sd"]})
        R.table("replicates__dose_response", rows, "per table", note="per-arm cell means summarized across replicates", show=False)
        keys = sorted({k for r in reps for k in self.per_rep[r]["cf"]}, key=str)
        rows = []
        for k in keys:
            s = ST.summarize_replicates({r: self.per_rep[r]["cf"].get(k) for r in reps})
            rows.append({"statistic": k[0], "arm": k[1], "n_replicates": s["n_replicates"], "mean": s["mean"], "sd_across_replicates": s["sd"]})
        R.table("replicates__counterfact", rows, "counterfact", note="CounterFact statistics summarized across replicates")


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
    reps = list(an.per_rep)
    rep0 = reps[0]
    pa, metric = "cloze_neutral", "answer_plus_suffix"

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
                               label=f"{base_name} baseline, arm {arm} (n={b[0]['n']})")
            ax.set_xscale("log", base=4)
            ax.set_xticks([d for d in sorted({r["dose"] for r in rows})])
            ax.set_xticklabels([str(d) for d in sorted({r["dose"] for r in rows})])
            ax.set_xlabel("assigned frequency (nominal occurrences)")
            ax.set_title(f"label = {lab}", fontsize=10)
            style(ax)
            ax.legend(fontsize=7, frameon=False)
        axes[0][0].set_ylabel("belief = logP(value) - logP(competing)")
        fig.suptitle(f"Dose-response, cohort {cname} (n={co.n}), pass {pa}, metric {metric}; dotted = {base_name} baseline "
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
            fig.suptitle(f"Per-replicate overlay, cohort {cname} (n={co.n}): thin = one training replicate, thick = mean across replicates", fontsize=9)
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
        ax.set_title(f"Attribution asymmetry, cohort {cname} (n={co.n}), labels pooled, 95% cluster-bootstrap CI", fontsize=9)
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
        ax.set_title(f"CounterFact paired changes vs C, cohort counterfact (n={an.cohorts['counterfact'].n}), replicate {rep0}", fontsize=9)
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
    ap.add_argument("--exposure-dir", default=None, help="exposure/ dir with <arm>/occurrence_exposure.csv")
    ap.add_argument("--cohort", nargs="+", default=None, choices=list(CH.COHORT_NAMES), help="restrict the analysis to these cohorts")
    ap.add_argument("--plot", action="store_true", help="write figures/*.png (needs matplotlib)")
    ap.add_argument("--bootstrap", type=int, default=ST.DEFAULT_BOOTSTRAP, help="cluster-bootstrap resamples (0 disables)")
    ap.add_argument("--equivalence-margin", type=float, default=None, help="TOST margin M for paired means; no equivalence claims without it")
    ap.add_argument("--seed", type=int, default=ST.DEFAULT_SEED, help="bootstrap seed")
    ap.add_argument("--exclude-facts", default=None, help="file of fact_ids (one per line) to drop everywhere")
    ap.add_argument("--no-tracking", action="store_true", help="don't upload the analysis to HF / wandb")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    T.load_env()
    R = Report()
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
    exposure = CH.load_exposure_dir(args.exposure_dir) if args.exposure_dir else {}
    manifest = None
    manifest_note = "none"
    if args.manifest:
        manifest = CH.load_manifest(args.manifest, fact_ids={r["fact_id"] for r in probes.injected})
        manifest_note = (f"{args.manifest} (v2 paraphrase indices for {len(manifest)} facts)" if manifest
                         else f"{args.manifest} (legacy manifest without paraphrase_index: not used)")

    replicates = discover_replicates(eval_dir)
    if not replicates:
        raise SystemExit(f"no beliefs_arm*.csv found in {eval_dir} or its seed_*/rep* subdirectories")

    cohorts = CH.build_cohorts(probes, exclude, f"excluded via --exclude-facts ({args.exclude_facts})")
    selected = list(args.cohort) if args.cohort else list(CH.COHORT_NAMES)
    headline = "all_injected" if probes.is_v2 else "exclusive_value"
    if headline not in selected:   # None when no injected cohort was selected: headline-only sections are skipped
        headline = next((c for c in CH.INJECTED_COHORTS if c in selected), None)

    R.say("# Belief analysis v2\n")
    R.say("| input | value |\n|---|---|")
    R.say(f"| eval dir | {eval_dir} |")
    R.say(f"| replicates | {len(replicates)}: " + ", ".join(f"{n} ({d})" for n, d in replicates) + " |")
    R.say(f"| probe files | {probes.paths['injected']} (sha256 {sha256_file(probes.paths['injected'])[:12]}), "
          f"{probes.paths['heldout']} (sha256 {sha256_file(probes.paths['heldout'])[:12]}) |")
    R.say(f"| probe protocol | {probes.protocol} |")
    R.say(f"| build summary | {summary_path if summary else 'none found'} |")
    R.say(f"| render config | {json.dumps({k: v for k, v in protocol_cfg.items() if k != 'assumed_defaults'})} |")
    R.say(f"| manifest | {manifest_note} |")
    R.say(f"| exposure index | {', '.join(f'{a}: {len(v)} facts' for a, v in exposure.items()) or 'none'} |")
    R.say(f"| excluded fact ids | {len(exclude)}{' (' + args.exclude_facts + ')' if args.exclude_facts else ''} |")
    R.say(f"| bootstrap | {args.bootstrap} resamples, seed {args.seed}, clusters = semantic groups |")
    R.say(f"| equivalence margin | {args.equivalence_margin if args.equivalence_margin is not None else 'none (no equivalence claims)'} |")
    rev = code_revision()
    R.say(f"| code | commit {rev.get('git_commit')} dirty={rev.get('git_dirty')} branch={rev.get('git_branch')} |")
    R.say(f"| output | {out_dir} |\n")

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

    an = Analysis(args, probes, cohorts, selected, headline, protocol_cfg, exposure, manifest, R)
    arm_names = []
    for rep, rep_dir in replicates:
        arms_raw = load_replicate(rep_dir, exclude)
        if not arms_raw:
            R.say(f"replicate {rep}: no arm CSVs, skipped")
            continue
        arm_names = sorted(set(arm_names) | set(arms_raw))
        an.run_replicate(rep, rep_dir, arms_raw)
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
    R.say(f"- {out_dir}/analysis.md (this report), analysis.txt (console log)")
    R.say(f"- {len(R.tables)} tables under tables/, {len(cohorts)} cohorts under cohorts/, {len(plots)} figures under figures/")
    with open(os.path.join(out_dir, "analysis.md"), "w", encoding="utf-8") as fh:
        fh.write(R.markdown())
    print(f"\nanalysis -> {out_dir}")
    return {"args": args, "plot": plots[0] if plots else None, "plots": plots, "out_dir": out_dir,
            "tables": sorted(R.tables), "cohorts": {n: c.n for n, c in cohorts.items()}, "headline": headline}


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
    md = os.path.join(out_dir, "analysis.md")
    if os.path.exists(md):
        files[f"{tag}/analysis_v2/analysis.md"] = md
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
