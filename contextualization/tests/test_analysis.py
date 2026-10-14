"""Tests for the analysis stage (analyze_beliefs.py, ctx/cohorts.py, ctx/stats.py).

Run from the repository root:
    $HOME/.cache/ctx_cpu_venv/bin/python -m pytest contextualization/tests/test_analysis.py -q

Two kinds of tests:
  * historical reproduction against the 2026-10-13 review outputs (reviews/2026-10-13/*.json|csv)
    using the archived eval CSVs (eval_results.tgz, in the repo) and the historical v1 probe
    files (~/.cache/ctx_wandb_to_hf/ctx_experiment_build/v1/probe_sets; skipped when absent);
  * synthetic fixtures where the truth is known by construction.
Nothing here writes into the repository or the historical artifacts.
"""
import csv
import json
import math
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

from conftest import CTX_DIR, REPO_ROOT  # noqa: F401  (also puts CTX_DIR on sys.path)

import analyze_beliefs as AB  # noqa: E402
from ctx import cohorts as CH  # noqa: E402
from ctx import stats as ST  # noqa: E402
from ctx.factpool import _build_relations, _make_pair  # noqa: E402
from ctx.protocol import PROTOCOL_V2, ProtocolError  # noqa: E402
from ctx.rng import rng_for, stable_id  # noqa: E402

PY = sys.executable
REVIEW = CTX_DIR / "reviews" / "2026-10-13"
HIST_BUILD = Path(os.environ.get("CTX_HISTORICAL_BUILD_DIR", Path.home() / ".cache" / "ctx_wandb_to_hf" / "ctx_experiment_build" / "v1"))
HIST_PROBES = HIST_BUILD / "probe_sets"
EVAL_TGZ = CTX_DIR / "eval_results.tgz"


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def run_cli(eval_dir, probe_dir, *extra):
    cmd = [PY, str(CTX_DIR / "analyze_beliefs.py"), "--eval-dir", str(eval_dir), "--probe-dir", str(probe_dir),
           "--no-tracking", *extra]
    env = dict(os.environ, CTX_TRACKING="0", MPLBACKEND="Agg")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
    assert res.returncode == 0, res.stdout[-4000:] + res.stderr[-4000:]
    return res.stdout


# ======================================================================================
# historical data

@pytest.fixture(scope="session")
def historical_eval_dir(tmp_path_factory):
    if not EVAL_TGZ.exists():
        pytest.skip("eval_results.tgz not present")
    d = tmp_path_factory.mktemp("hist_eval")
    with tarfile.open(EVAL_TGZ, "r:gz") as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name.endswith(".csv"):
                m.name = os.path.basename(m.name)
                tar.extract(m, d)
    assert (d / "beliefs_armC.csv").exists()
    return d


@pytest.fixture(scope="session")
def historical_probes():
    if not (HIST_PROBES / "injected_facts.jsonl").exists():
        pytest.skip(f"historical probe files not available under {HIST_PROBES}")
    return CH.load_probes(str(HIST_PROBES))


@pytest.fixture(scope="session")
def historical_cohorts(historical_probes):
    return CH.build_cohorts(historical_probes)


@pytest.fixture(scope="session")
def historical_run(historical_eval_dir, historical_probes):
    """The CLI on the archived CSVs + historical probes (small bootstrap; figures on)."""
    summary = HIST_BUILD / "build_summary.json"
    extra = ["--bootstrap", "40", "--plot"]
    if summary.exists():
        extra += ["--build-summary", str(summary)]
    out = run_cli(historical_eval_dir, HIST_PROBES, *extra)
    return historical_eval_dir / "analysis_v2", out


def test_historical_cohort_sizes(historical_cohorts):
    co = historical_cohorts
    assert {n: c.n for n, c in co.items()} == {
        "all_injected": 5600, "rival_untrained": 1991, "exclusive_value": 740, "heldout_claims": 2401,
        "heldout_groups": 269, "heldout_rivals": 2132, "counterfact": 2000, "paraphrase_wellformed": 5496}
    assert co["exclusive_value"].n_by_tier() == {"novel": 710, "real": 30}
    assert co["counterfact"].fact_ids[0] == "counterfact_02401" and co["counterfact"].fact_ids[-1] == "counterfact_04400"
    # exclusive_value is nested in rival_untrained, held-out cohorts partition heldout_claims
    assert set(co["exclusive_value"].fact_ids) <= set(co["rival_untrained"].fact_ids)
    assert set(co["heldout_groups"].fact_ids) | set(co["heldout_rivals"].fact_ids) == set(co["heldout_claims"].fact_ids)
    assert not set(co["heldout_groups"].fact_ids) & set(co["heldout_rivals"].fact_ids)


def test_exclusive_value_reproduces_auditor_strict_ids(historical_cohorts):
    ref = set(json.load(open(REVIEW / "strict_clean_fact_ids.json")))
    assert len(ref) == 740
    assert set(historical_cohorts["exclusive_value"].fact_ids) == ref


def test_malformed_paraphrases_reproduce_auditor(historical_cohorts):
    ref = {m["fact_id"] for m in json.load(open(REVIEW / "malformed_paraphrases.json"))}
    assert len(ref) == 104
    assert set(CH.malformed_paraphrase_ids(historical_cohorts)) == ref
    reasons = {e["fact_id"]: e["reason"] for e in historical_cohorts["paraphrase_wellformed"].exclusions}
    assert all(reasons[f].startswith("malformed_paraphrase_contrast") for f in ref)


def test_verbatim_trained_count_matches_audit(historical_probes):
    ref = json.load(open(REVIEW / "eval_audit_summary.json"))["template0_exactly_trained_n"]
    assert ref == 1362
    summary = json.load(open(HIST_BUILD / "build_summary.json")) if (HIST_BUILD / "build_summary.json").exists() else None
    cfg = CH.protocol_cfg_from_summary(summary, historical_probes)
    cls = CH.classify_verbatim_trained(historical_probes.injected, cfg)
    assert sum(cls["nominal"].values()) == ref
    assert len(cls["nominal"]) == 5600


def test_historical_dose256_novel_exclusive_value(historical_run):
    """The review's headline table: novel entities, exclusive_value, dose 256."""
    out_dir, _ = historical_run
    expect = {"true": (53, 6.944, 0.507), "false": (57, 7.636, 0.497), "contested": (32, 7.883, 0.378)}
    rc = {r["label"]: r for r in read_csv(out_dir / "tables" / "paired__R-C__exclusive_value__cloze_neutral__answer_plus_suffix__tier-novel.csv")
          if r["dose"] == "256"}
    xc = {r["label"]: r for r in read_csv(out_dir / "tables" / "paired__X-C__exclusive_value__cloze_neutral__answer_plus_suffix__tier-novel.csv")
          if r["dose"] == "256"}
    for lab, (n, r_minus_c, x_minus_c) in expect.items():
        assert int(rc[lab]["n"]) == n and int(xc[lab]["n"]) == n
        assert float(rc[lab]["mean_diff"]) == pytest.approx(r_minus_c, abs=5e-4)
        assert float(xc[lab]["mean_diff"]) == pytest.approx(x_minus_c, abs=5e-4)
        assert rc[lab]["cohort"] == "exclusive_value" and rc[lab]["tier"] == "novel"
    # and against the auditor's dose_sensitivity.csv row by row (strict_clean, novel)
    for row in read_csv(REVIEW / "dose_sensitivity.csv"):
        if row["subset"] != "strict_clean" or row["tier"] != "novel":
            continue
        got = next(r for r in read_csv(out_dir / "tables" / "paired__R-C__exclusive_value__cloze_neutral__answer_plus_suffix__tier-novel.csv")
                   if r["label"] == row["truth"] and r["dose"] == row["freq"])
        assert int(got["n"]) == int(row["n"])
        assert float(got["mean_diff"]) == pytest.approx(float(row["R_minus_C"]), abs=1e-6)


def test_historical_counterfact_reversals(historical_run):
    out_dir, _ = historical_run
    rows = {r["contrast"]: r for r in read_csv(out_dir / "tables" / "counterfact__paired_vs_C.csv")}
    ref = json.load(open(REVIEW / "eval_audit_summary.json"))["counterfact"]
    for arm in ("R", "X"):
        r = rows[f"{arm}-C"]
        assert int(r["n_flips_total"]) == ref[arm]["sign_flips_vs_C"]
        assert int(r["n_C_true_to_arm_false"]) + int(r["n_C_false_to_arm_true"]) == int(r["n_flips_total"])
        assert float(r["mean_delta"]) == pytest.approx(ref[arm]["mean_difference_C"], abs=1e-6)
        assert float(r["mean_abs_delta"]) == pytest.approx(ref[arm]["mean_absolute_difference_C"], abs=1e-6)
    assert int(rows["R-C"]["n_flips_total"]) == 209 and int(rows["X-C"]["n_flips_total"]) == 193


def test_historical_paraphrase_means(historical_run):
    out_dir, _ = historical_run
    ref = json.load(open(REVIEW / "eval_audit_summary.json"))["paraphrase_scores"]
    for cohort, key in (("paraphrase_wellformed", "excluded"), ("all_injected", "all")):
        rows = read_csv(out_dir / "tables" / f"paraphrase__{cohort}__heldout_paraphrase__full_sentence.csv")
        pooled = {r["arm"]: r for r in rows if r["dose"] == "all" and r["label"] == "all"}
        for arm in "CRX":
            assert int(pooled[arm]["n"]) == ref[arm][f"{key}_n"]
            assert float(pooled[arm]["mean_belief"]) == pytest.approx(ref[arm][f"{key}_mean"], abs=1e-6)


def test_historical_outputs_complete(historical_run):
    out_dir, out = historical_run
    assert (out_dir / "analysis.md").exists() and (out_dir / "analysis.txt").exists()
    assert sorted(p.stem for p in (out_dir / "cohorts").glob("*.json")) == sorted(CH.COHORT_NAMES)
    figs = {p.name for p in (out_dir / "figures").glob("*.png")}
    assert {"dose_response__exclusive_value.png", "attribution_asymmetry__exclusive_value.png", "counterfact_paired_changes.png"} <= figs
    assert "Headline cohort: **exclusive_value**" in out
    assert "No equivalence claim" in out
    co = json.loads((out_dir / "cohorts" / "all_injected.json").read_text())
    assert co["missingness"]["rep0"]["C"]["wrapper_conditional"]["n_missing"] == 63
    # the legacy outputs next to the CSVs are untouched / never created by the new script
    assert not (out_dir.parent / "analysis.txt").exists()
    assert not (out_dir.parent / "beliefs_dose_response.png").exists()


# ======================================================================================
# statistics

def test_wilcoxon_legacy_vs_scipy_tie_case():
    ref = json.load(open(REVIEW / "eval_audit_summary.json"))["wilcoxon_tie_repro"]
    assert AB.legacy_wilcoxon_signed_rank([1.0] * 20) == pytest.approx(ref["implemented_p"], rel=1e-9)
    assert AB.wilcoxon_signed_rank([1.0] * 20) == pytest.approx(ref["tie_corrected_normal_p"], rel=1e-6)
    assert AB.wilcoxon_signed_rank([1.0] * 9) is None
    assert AB.wilcoxon_signed_rank([0.0] * 30) is None
    assert AB.wilcoxon_signed_rank is ST.wilcoxon_signed_rank
    rng = np.random.default_rng(1)
    d = rng.normal(0.4, 1.0, 80)
    from scipy.stats import wilcoxon
    assert ST.wilcoxon_signed_rank(d) == pytest.approx(wilcoxon(d, zero_method="wilcox", correction=False, method="auto").pvalue)


def test_holm_on_constructed_family():
    p = [0.01, 0.04, 0.03, None, 0.20]
    assert ST.holm(p) == [pytest.approx(0.04), pytest.approx(0.09), pytest.approx(0.09), None, pytest.approx(0.20)]
    assert ST.holm([0.5, 0.9]) == [1.0, 1.0]
    assert ST.holm([None, None]) == [None, None]
    assert ST.holm([]) == []


def test_cluster_bootstrap_resamples_groups_not_facts():
    rng = np.random.default_rng(3)
    group_means = rng.normal(0.0, 1.0, 20)
    values = np.repeat(group_means, 10)                       # 200 facts, 10 identical per group
    groups = np.repeat([f"g{i}" for i in range(20)], 10)
    clustered = ST.cluster_bootstrap(values, groups, n_boot=1500, seed=5)
    per_fact = ST.cluster_bootstrap(values, [f"f{i}" for i in range(200)], n_boot=1500, seed=5)
    assert clustered.n_groups == 20 and per_fact.n_groups == 200
    w_cl = clustered.ci95_hi - clustered.ci95_lo
    w_pf = per_fact.ci95_hi - per_fact.ci95_lo
    assert w_cl > 2.0 * w_pf                                   # ~sqrt(10) wider when clusters are resampled
    direct = ST.cluster_bootstrap(group_means, [f"g{i}" for i in range(20)], n_boot=1500, seed=5)
    assert w_cl == pytest.approx(direct.ci95_hi - direct.ci95_lo, rel=0.25)
    # deterministic in the seed/stream, independent of evaluation order
    again = ST.cluster_bootstrap(values, groups, n_boot=1500, seed=5)
    assert (again.ci95_lo, again.ci95_hi) == (clustered.ci95_lo, clustered.ci95_hi)
    assert ST.cluster_bootstrap(values, groups, n_boot=0).ci95_lo is None


def test_tost_and_replicate_summary():
    assert ST.tost_equivalence(-0.2, 0.3, 0.5)["equivalent"] is True
    assert ST.tost_equivalence(-0.2, 0.6, 0.5)["equivalent"] is False
    assert ST.tost_equivalence(None, None, 0.5)["equivalent"] is False
    s = ST.summarize_replicates({"seed_0": 2.0, "seed_1": 4.0})
    assert s["n_replicates"] == 2 and s["mean"] == 3.0 and s["sd"] == pytest.approx(math.sqrt(2.0))
    assert ST.summarize_replicates({"rep0": 1.0})["sd"] is None


# ======================================================================================
# synthetic fixture with known truth

def _v1_record(fact, dose, held_idx=1, arms=("R", "X")):
    rec = fact.to_record()
    rec["assigned_frequency"] = dose
    rec["arms"] = list(arms)
    rec["heldout_paraphrase"] = fact.surface_forms[held_idx] if held_idx is not None else None
    rec["fixed_source"] = None
    return rec


def _heldout_record(fact):
    rec = fact.to_record()
    for k in ("assigned_frequency", "heldout_paraphrase", "pair_id"):
        rec.pop(k, None)
    rec["source"] = "synthetic"
    return rec


def build_synthetic_probes(out_dir: Path):
    """A v1-style probe set whose cohort membership is known by construction.

    capital_of groups Land0..Land11 (novel):
      0-3   only the TRUE fact injected (exclusive_value); its false twin held out (heldout_rivals)
      4-5   both facts injected (rival trained: not rival_untrained, not exclusive_value)
      6-7   true fact injected + a contested third value injected (rival_untrained but not exclusive)
      8-11  both facts held out (heldout_groups)
    plus one element_symbol group whose reserved paraphrase repeats the symbol (malformed contrast),
    one landmark_city group whose value is a substring of the subject (malformed), 6 CounterFact items.
    """
    rels = {r.key: r for r in _build_relations()}
    rng = rng_for(11, "fixture")
    cap = rels["capital_of"]
    injected, heldout, truth = [], [], {"exclusive_value": [], "rival_untrained": [], "heldout_groups": [], "heldout_rivals": [],
                                         "malformed": [], "rival_trained": []}
    doses = [1, 4, 16, 64]
    for i in range(12):
        t, f = _make_pair(cap, f"Land{i}", f"City{i}", "novel", rng)
        dose = doses[i % len(doses)]
        if i <= 3:
            injected.append(_v1_record(t, dose))
            heldout.append(_heldout_record(f))
            truth["exclusive_value"].append(t.fact_id)
            truth["rival_untrained"].append(t.fact_id)
            truth["heldout_rivals"].append(f.fact_id)
        elif i <= 5:
            injected.append(_v1_record(t, dose))
            injected.append(_v1_record(f, doses[(i + 1) % len(doses)]))
            truth["rival_trained"] += [t.fact_id, f.fact_id]
        elif i <= 7:
            injected.append(_v1_record(t, dose))
            third = _make_pair(cap, f"Land{i}", f"Other{i}", "novel", rng)[0]
            third.truth_value = "contested"
            third.fact_id = stable_id("geography", "capital_of", f"Land{i}", f"Other{i}", "contested")
            third.competing_value = f"Rival{i}"
            injected.append(_v1_record(third, 4))
            heldout.append(_heldout_record(f))
            truth["rival_untrained"] += [t.fact_id, third.fact_id]   # neither rival (f / Rival{i}) is injected
            truth["heldout_rivals"].append(f.fact_id)
        else:
            heldout.append(_heldout_record(t))
            heldout.append(_heldout_record(f))
            truth["heldout_groups"] += [t.fact_id, f.fact_id]
    # malformed paraphrase cases (both exclusive_value too)
    sym = rels["element_symbol"]
    t, f = _make_pair(sym, "zorium", "Qn", "novel", rng)
    f.value, f.competing_value, t.competing_value = "Vx", "Qn", "Vx"
    f.surface_forms = [fr.format(subject="zorium", value="Vx", subject_kind=sym.subject_kind) for fr in sym.frames]
    injected.append(_v1_record(t, 16, held_idx=4))   # frame 4 repeats the symbol -> malformed replace
    heldout.append(_heldout_record(f))
    truth["malformed"].append(t.fact_id)
    truth["exclusive_value"].append(t.fact_id)
    truth["rival_untrained"].append(t.fact_id)
    truth["heldout_rivals"].append(f.fact_id)
    land = rels["landmark_city"]
    t, f = _make_pair(land, "Sydney Opera House", "Sydney", "real", rng)
    injected.append(_v1_record(t, 64, held_idx=0))   # value inside the subject -> malformed replace
    heldout.append(_heldout_record(f))
    truth["malformed"].append(t.fact_id)
    truth["exclusive_value"].append(t.fact_id)
    truth["rival_untrained"].append(t.fact_id)
    truth["heldout_rivals"].append(f.fact_id)
    # counterfact (v1 style: no fact_id)
    for i in range(6):
        heldout.append({"source": "counterfact", "subject": f"Person{i}", "cloze_templates": [f"The mother tongue of Person{i} is ___."],
                        "true_value": "French", "false_value": "English", "truth_value": "true"})
    pdir = out_dir / "probe_sets"
    pdir.mkdir(parents=True, exist_ok=True)
    with open(pdir / "injected_facts.jsonl", "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in injected)
    with open(pdir / "heldout_facts.jsonl", "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in heldout)
    (out_dir / "build_summary.json").write_text(json.dumps({"config": {"seed": 1234, "register": True, "verbatim_control": False,
                                                                        "source_per_fact": False, "embed": True}}))
    return pdir, truth


LEGACY_COLS = ["arm", "probe_set", "fact_id", "pass", "template_idx", "truth_value", "entity_tier", "assigned_frequency",
               "is_frame0", "logp_value", "logp_competing", "n_tok_value", "n_tok_competing"]
V2_EXTRA = ["schema_version", "group_id", "role", "ground_truth", "template_id", "metric", "logp_value_answer",
            "logp_competing_answer", "n_tok_value_answer", "n_tok_competing_answer", "logp_value_suffix", "logp_competing_suffix",
            "boundary_k", "n_ctx_tokens", "boundary_adjusted", "answer_suffix_merged_value", "answer_suffix_merged_competing",
            "value_span_chars", "context", "cand_value", "cand_competing", "scoring_policy"]


def write_arm_csvs(eval_dir: Path, probes: CH.ProbeSet, shift, arms=("C", "R", "X"), v2=False, cf_beliefs=None, seed=0):
    """Legacy-format CSVs (v2 columns added when v2=True). Arm C beliefs are seeded noise; arm A's
    logp_value is C's plus shift(arm, fact_id, pass). cf_beliefs: {arm: [belief per counterfact item]}."""
    eval_dir.mkdir(parents=True, exist_ok=True)
    cols = LEGACY_COLS + (V2_EXTRA if v2 else [])
    items = []
    for r in probes.injected:
        for pa in ("cloze_neutral", "cloze_attributed"):
            for ti in range(len(r["cloze_templates"])):
                items.append(("injected", pa, r, ti))
        items.append(("injected", "wrapper_conditional", r, 0))
        items.append(("injected", "heldout_paraphrase", r, -1))
    for r in probes.heldout:
        for pa in ("cloze_neutral", "cloze_attributed"):
            for ti in range(len(r["cloze_templates"])):
                items.append(("heldout", pa, r, ti))
    for r in probes.counterfact:
        items.append(("counterfact", "cloze_neutral", r, 0))
    for arm in arms:
        with open(eval_dir / f"beliefs_arm{arm}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for k, (ps, pa, r, ti) in enumerate(items):
                g = rng_for(seed, "csv", r["fact_id"], pa, ti)
                lc = float(-8.0 + g.normal(0, 1.0))
                lv = float(lc + g.normal(0, 1.0))
                if ps == "counterfact" and cf_beliefs is not None:
                    idx = probes.counterfact.index(r)
                    lv = lc + cf_beliefs[arm][idx]
                else:
                    lv += shift(arm, r["fact_id"], pa)
                row = {"arm": arm, "probe_set": ps, "fact_id": r["fact_id"], "pass": pa, "template_idx": ti,
                       "truth_value": r.get("truth_value", ""), "entity_tier": r.get("entity_tier", ""),
                       "assigned_frequency": r.get("assigned_frequency", "") if ps == "injected" else "",
                       "is_frame0": "", "logp_value": round(lv, 5), "logp_competing": round(lc, 5),
                       "n_tok_value": 2, "n_tok_competing": 2}
                if v2:
                    row.update({"schema_version": 2, "group_id": r["group_id"], "role": "injected" if ps == "injected" else "",
                                "metric": "full_sentence" if pa == "heldout_paraphrase" else "answer_plus_suffix",
                                "logp_value_answer": round(lv + 1.0, 5), "logp_competing_answer": round(lc + 0.5, 5),
                                "value_span_chars": "[[0,1]]" if pa == "heldout_paraphrase" else ""})
                w.writerow(row)


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    pdir, truth = build_synthetic_probes(root)
    probes = CH.load_probes(str(pdir))
    return root, pdir, probes, truth


def test_synthetic_cohorts_match_construction(synthetic):
    root, pdir, probes, truth = synthetic
    co = CH.build_cohorts(probes)
    assert set(co["exclusive_value"].fact_ids) == set(truth["exclusive_value"])
    assert set(co["rival_untrained"].fact_ids) == set(truth["rival_untrained"])
    assert set(co["heldout_groups"].fact_ids) == set(truth["heldout_groups"])
    assert set(co["heldout_rivals"].fact_ids) == set(truth["heldout_rivals"])
    assert set(CH.malformed_paraphrase_ids(co)) == set(truth["malformed"])
    assert set(co["paraphrase_wellformed"].fact_ids) == set(co["all_injected"].fact_ids) - set(truth["malformed"])
    assert co["counterfact"].n == 6 and co["counterfact"].fact_ids[0] == f"counterfact_{len(probes.heldout):05d}"
    for fid in truth["rival_trained"]:
        assert fid not in co["rival_untrained"].fact_ids and fid not in co["exclusive_value"].fact_ids
    reasons = {e["fact_id"]: e["reason"] for e in co["rival_untrained"].exclusions}
    assert all(reasons[f].startswith("rival_injected") for f in truth["rival_trained"])
    j = co["exclusive_value"].to_json()
    assert j["n"] == len(truth["exclusive_value"]) and "definition" in j and "n_by_label_and_dose" in j and "exclusions" in j


def test_synthetic_exclude_facts_recorded(synthetic):
    root, pdir, probes, truth = synthetic
    drop = truth["exclusive_value"][:2]
    co = CH.build_cohorts(probes, exclude=set(drop), exclude_reason="excluded via --exclude-facts (x.txt)")
    for name in ("all_injected", "exclusive_value", "rival_untrained", "paraphrase_wellformed"):
        assert not set(drop) & set(co[name].fact_ids)
        assert {e["fact_id"] for e in co[name].exclusions if e["reason"].startswith("excluded via")} == set(drop)


def test_realized_training_forms_rules(synthetic):
    root, pdir, probes, truth = synthetic
    rec = next(r for r in probes.injected if r["fact_id"] == truth["exclusive_value"][0])
    forms = rec["surface_forms"]
    allowed = [s for s in forms if s != rec["heldout_paraphrase"]]
    dose = rec["assigned_frequency"]
    got = CH.realized_training_forms(rec, {"protocol": "v1-legacy-2026-07"})
    assert got == [allowed[o % len(allowed)] for o in range(dose)]
    assert rec["heldout_paraphrase"] not in got
    # template 0 ("The capital of X is ___.") filled == surface form 0 -> verbatim trained at any dose >= 1
    assert CH.template0_sentence(rec) == forms[0]
    assert CH.is_verbatim_trained(rec, got) is True
    held0 = dict(rec, heldout_paraphrase=forms[0])
    assert CH.is_verbatim_trained(held0, CH.realized_training_forms(held0)) is False
    # exposure index: only complete, in-horizon occurrences count
    exp = [{"occurrence": 0, "status": "partial_value_lost", "step": 10}, {"occurrence": 1, "status": "complete", "step": 11},
           {"occurrence": 2, "status": "complete", "step": -1}]
    big = dict(rec, assigned_frequency=3)
    assert CH.realized_training_forms(big, None, exposure_rows=exp) == [allowed[1]]
    assert CH.is_verbatim_trained(big, CH.realized_training_forms(big, None, exposure_rows=exp)) is False
    # manifest v2 paraphrase indices take precedence (index into surface_forms)
    man = [{"arm": "R", "occurrence": 0, "paraphrase_index": 3}, {"arm": "X", "occurrence": 0, "paraphrase_index": 3},
           {"arm": "R", "occurrence": 1, "paraphrase_index": 0}]
    assert CH.realized_training_forms(rec, {"protocol": PROTOCOL_V2, "render_policy": {"paraphrase": "seeded_start"}}, manifest_rows=man) == [forms[3], forms[0]]
    with pytest.raises(ProtocolError):
        CH.realized_training_forms(rec, {"protocol": PROTOCOL_V2, "render_policy": {"paraphrase": "seeded_start"}})
    with pytest.raises(ProtocolError):   # occurrence rendered the reserved paraphrase
        CH.realized_training_forms(dict(rec, heldout_paraphrase_index=1), {"protocol": PROTOCOL_V2},
                                   manifest_rows=[{"occurrence": 0, "paraphrase_index": 1}])
    v2rec = dict(rec, schema_version=2)
    fixed = CH.realized_training_forms(v2rec, {"protocol": PROTOCOL_V2, "render_policy": {"paraphrase": "fixed"}})
    assert fixed == [allowed[0]] * dose
    cyc = CH.realized_training_forms(v2rec, {"protocol": PROTOCOL_V2, "render_policy": {"paraphrase": "cycle_from_zero"}})
    assert cyc == got
    with pytest.raises(ProtocolError):
        CH.realized_training_forms(v2rec, {"protocol": PROTOCOL_V2, "render_policy": None})


def test_load_manifest_legacy_and_v2(tmp_path):
    legacy = tmp_path / "manifest.csv"
    legacy.write_text("fact_id,truth_value,domain,entity_tier,arm,assigned_frequency,rendering_id,embedded,heldout,shard_file\n"
                      "a,true,x,novel,R,1,a:0,True,False,s\n")
    assert CH.load_manifest(legacy) is None
    v2 = tmp_path / "manifest_v2.csv"
    v2.write_text("schema_version,fact_id,group_id,arm,occurrence,assigned_frequency,paraphrase_index\n"
                  "2,a,g,R,0,2,3\n2,a,g,X,0,2,3\n2,a,g,R,1,2,0\n2,b,h,R,0,1,\n")
    m = CH.load_manifest(v2)
    assert [r["paraphrase_index"] for r in m["a"]] == [3, 3, 0] and m["b"][0]["paraphrase_index"] is None
    assert CH.load_manifest(v2, fact_ids={"b"}).keys() == {"b"}


@pytest.fixture(scope="session")
def synthetic_run(synthetic):
    """CLI run on the fixture: arm R = C + 2 on heldout_groups facts only, + 3 on exclusive_value facts
    only (neutral cloze and attributed passes); X = C everywhere; CounterFact items with known flips."""
    root, pdir, probes, truth = synthetic
    hg, ev = set(truth["heldout_groups"]), set(truth["exclusive_value"])

    def shift(arm, fid, pa):
        if arm != "R" or pa not in ("cloze_neutral", "cloze_attributed"):
            return 0.0
        return 2.0 if fid in hg else (3.0 if fid in ev else 0.0)

    cf = {"C": [1.0, 2.0, -1.0, -2.0, 3.0, -3.0], "R": [-0.5, 2.5, 0.5, 1.0, 3.5, -2.0], "X": [1.0, 2.0, -1.0, -2.0, 3.0, -3.0]}
    eval_dir = root / "eval"
    write_arm_csvs(eval_dir, probes, shift, cf_beliefs=cf)
    out = run_cli(eval_dir, pdir, "--bootstrap", "100", "--plot", "--equivalence-margin", "0.5")
    return eval_dir / "analysis_v2", out, truth


def test_synthetic_only_the_shifted_cohort_moves(synthetic_run):
    out_dir, out, truth = synthetic_run
    t = out_dir / "tables"

    def paired(cohort, pa="cloze_neutral", metric="answer_plus_suffix"):
        return read_csv(t / f"paired__R-C__{cohort}__{pa}__{metric}.csv")

    # heldout_groups: every cell exactly +2 (degenerate CI), heldout_rivals: exactly 0, p n/a (all-zero diffs)
    for r in paired("heldout_groups"):
        assert float(r["mean_diff"]) == pytest.approx(2.0, abs=1e-9)
        assert float(r["ci95_lo"]) == pytest.approx(2.0, abs=1e-9) and float(r["ci95_hi"]) == pytest.approx(2.0, abs=1e-9)
        assert r["cohort"] == "heldout_groups"
    for r in paired("heldout_rivals"):
        assert float(r["mean_diff"]) == 0.0 and r["p_wilcoxon"] == "" and r["n_pref_flips"] == "0"
    # heldout_claims is the union: diluted mean 2 * n_groups_facts / n
    n_hg, n_hc = len(truth["heldout_groups"]), len(truth["heldout_groups"]) + len(truth["heldout_rivals"])
    pooled = next(r for r in paired("heldout_claims") if r["dose"] == "all")
    assert float(pooled["mean_diff"]) == pytest.approx(2.0 * n_hg / n_hc, abs=1e-9) and int(pooled["n"]) == n_hc
    # exclusive_value: exactly +3 everywhere; rival_untrained diluted; counterfact untouched by the shift
    for r in paired("exclusive_value"):
        assert float(r["mean_diff"]) == pytest.approx(3.0, abs=1e-9)
    ru = next(r for r in paired("rival_untrained") if r["dose"] == "all")
    assert float(ru["mean_diff"]) == pytest.approx(3.0 * len(truth["exclusive_value"]) / len(truth["rival_untrained"]), abs=1e-9)
    # X-C is zero everywhere, R-X equals R-C
    for r in read_csv(t / "paired__X-C__exclusive_value__cloze_neutral__answer_plus_suffix.csv"):
        assert float(r["mean_diff"]) == 0.0
    for r in read_csv(t / "paired__R-X__exclusive_value__cloze_neutral__answer_plus_suffix.csv"):
        assert float(r["mean_diff"]) == pytest.approx(3.0, abs=1e-9)
    # the wrapper / paraphrase passes were not shifted -> R-C zero there (separate metric, never mixed)
    for r in paired("exclusive_value", "wrapper_conditional"):
        assert float(r["mean_diff"]) == 0.0
    for r in read_csv(t / "paired__R-C__paraphrase_wellformed__heldout_paraphrase__full_sentence.csv"):
        assert float(r["mean_diff"]) == 0.0 and r["metric"] == "full_sentence"
    # dose-response rows: R - C per cell is the same +3 for exclusive_value
    dr = read_csv(t / "dose_response__exclusive_value__cloze_neutral__answer_plus_suffix.csv")
    by = {(r["label"], r["dose"], r["arm"]): float(r["mean_belief"]) for r in dr}
    for (lab, d, arm), v in by.items():
        if arm == "R":
            assert v - by[(lab, d, "C")] == pytest.approx(3.0, abs=1e-9)
    # attribution asymmetry: R and X carry the same shift on both passes as C (or none), so every arm's
    # attributed-minus-neutral mean equals C's in every cell (the asymmetry itself is pass noise, not zero)
    asym = read_csv(t / "attribution_asymmetry__exclusive_value__cloze_attributed.csv")
    c_cells = {(r["label"], r["dose"]): float(r["mean_diff"]) for r in asym if r["arm"] == "C"}
    assert c_cells and any(abs(v) > 1e-6 for v in c_cells.values())
    for r in asym:
        assert float(r["mean_diff"]) == pytest.approx(c_cells[(r["label"], r["dose"])], abs=1e-9)
        assert r["cohort"] == "exclusive_value" and int(r["n"]) > 0
    # figure data = the same table the figure was drawn from; the figure exists for this cohort only via its own file
    assert (out_dir / "figures" / "dose_response__exclusive_value.png").exists()
    assert (out_dir / "figures" / "dose_response__heldout_groups.png").exists() is False   # held-out cohorts have no dose axis


def test_synthetic_holm_and_tost_columns(synthetic_run):
    out_dir, out, truth = synthetic_run
    rows = read_csv(out_dir / "tables" / "paired__R-C__exclusive_value__cloze_neutral__answer_plus_suffix.csv")
    fam = [r for r in rows if r["family"]]
    tested = [r for r in fam if r["p_wilcoxon"] != ""]
    assert all(int(r["family_size"]) == len(tested) for r in fam)
    assert all(float(r["p_holm"]) >= float(r["p_wilcoxon"]) for r in tested)
    assert all(r["p_holm"] == "" for r in rows if not r["family"])
    assert all(r["tost_equivalent"] in ("True", "False") for r in rows)
    zero = read_csv(out_dir / "tables" / "paired__X-C__exclusive_value__cloze_neutral__answer_plus_suffix.csv")
    assert any(int(r["n_groups"]) >= 2 for r in zero)
    assert all(r["tost_equivalent"] == "True" for r in zero if int(r["n_groups"]) >= 2)   # X == C exactly: inside any margin
    assert all(r["tost_equivalent"] == "False" for r in zero if int(r["n_groups"]) < 2)   # no interval -> no equivalence claim
    assert all(r["tost_equivalent"] == "False" for r in fam)                              # +3 is outside +-0.5
    assert "TOST with margin 0.5" in out


def test_synthetic_counterfact_reversal_directions(synthetic_run):
    out_dir, out, truth = synthetic_run
    rows = {r["contrast"]: r for r in read_csv(out_dir / "tables" / "counterfact__paired_vs_C.csv")}
    r = rows["R-C"]
    assert int(r["n_C_true_to_arm_false"]) == 1 and int(r["n_C_false_to_arm_true"]) == 2 and int(r["n_flips_total"]) == 3
    assert float(r["rate_flips_total"]) == pytest.approx(0.5)
    assert float(r["accuracy_C"]) == pytest.approx(0.5) and float(r["accuracy_arm"]) == pytest.approx(4 / 6)
    assert float(r["mean_delta"]) == pytest.approx(np.mean([-1.5, 0.5, 1.5, 3.0, 0.5, 1.0]), abs=1e-6)
    x = rows["X-C"]
    assert int(x["n_flips_total"]) == 0 and float(x["mean_delta"]) == 0.0
    per_arm = {r["arm"]: r for r in read_csv(out_dir / "tables" / "counterfact__per_arm.csv")}
    assert int(per_arm["C"]["n_correct"]) == 3 and int(per_arm["R"]["n_correct"]) == 4
    assert "do NOT imply unchanged knowledge" in out and "unmeasured without replicates" in out


def test_synthetic_memorization_table(synthetic_run):
    out_dir, out, truth = synthetic_run
    rows = read_csv(out_dir / "tables" / "memorization__all_injected.csv")
    subs = {(r["arm"], r["subgroup"]): r for r in rows}
    # capital_of facts with heldout_paraphrase = form 1 are verbatim trained; the two malformed cases hold out
    # form 4 / form 0: the element_symbol one is trained verbatim (template 0 == form 0), the landmark one is not
    co = json.loads((out_dir / "cohorts" / "all_injected.json").read_text())
    assert co["details"]["verbatim_trained"]["nominal"] == int(subs[("C", "verbatim_trained")]["n"])
    assert int(subs[("C", "verbatim_trained")]["n"]) + int(subs[("C", "not_verbatim_trained")]["n"]) == co["n"]
    assert int(subs[("C", "not_verbatim_trained")]["n"]) == 1


def test_synthetic_cli_outputs_complete(synthetic_run):
    out_dir, out, truth = synthetic_run
    assert (out_dir / "analysis.md").exists() and (out_dir / "analysis.txt").exists()
    assert sorted(p.stem for p in (out_dir / "cohorts").glob("*.json")) == sorted(CH.COHORT_NAMES)
    names = {p.name for p in (out_dir / "tables").glob("*.csv")}
    for need in ("cohorts.csv", "counterfact__per_arm.csv", "counterfact__paired_vs_C.csv", "memorization__all_injected.csv",
                 "memorization__exclusive_value.csv", "paraphrase__paraphrase_wellformed__heldout_paraphrase__full_sentence.csv",
                 "heldout_baseline__heldout_groups__cloze_neutral__answer_plus_suffix.csv",
                 "dose_response__exclusive_value__cloze_neutral__answer_plus_suffix__tier-novel.csv"):
        assert need in names, need
    figs = {p.name for p in (out_dir / "figures").glob("*.png")}
    assert {"dose_response__exclusive_value.png", "dose_response__all_injected.png", "attribution_asymmetry__exclusive_value.png",
            "counterfact_paired_changes.png"} <= figs
    for row in read_csv(out_dir / "tables" / "cohorts.csv"):
        assert row["cohort"] in CH.COHORT_NAMES and int(row["n"]) >= 0
    # every table row names its cohort and denominator
    for p in (out_dir / "tables").glob("paired__*.csv"):
        for r in read_csv(p):
            assert r["cohort"] in CH.COHORT_NAMES and int(r["n"]) > 0 and r["replicate"] == "rep0"
    md = (out_dir / "analysis.md").read_text()
    assert "Headline cohort: **exclusive_value**" in md and "within-run" in md
    assert "Table `paired__R-C__exclusive_value__cloze_neutral__answer_plus_suffix` -- cohort: exclusive_value" in md


def test_synthetic_cohort_flag_restricts(synthetic):
    root, pdir, probes, truth = synthetic
    eval_dir = root / "eval_cohort_flag"
    write_arm_csvs(eval_dir, probes, lambda a, f, p: 0.0)
    run_cli(eval_dir, pdir, "--bootstrap", "0", "--cohort", "heldout_groups", "counterfact")
    names = {p.name for p in (eval_dir / "analysis_v2" / "tables").glob("paired__*.csv")}
    assert names and all(("heldout_groups" in n or "counterfact" in n) for n in names)
    assert sorted(p.stem for p in (eval_dir / "analysis_v2" / "cohorts").glob("*.json")) == sorted(CH.COHORT_NAMES)


def test_replicates_summarized_not_pooled(synthetic):
    root, pdir, probes, truth = synthetic
    hg = set(truth["heldout_groups"])
    eval_dir = root / "eval_reps"
    for name, s in (("seed_0", 2.0), ("seed_1", 4.0)):
        write_arm_csvs(eval_dir / name, probes, lambda a, f, p, s=s: s if (a == "R" and f in hg and p == "cloze_neutral") else 0.0, seed=1)
    assert AB.discover_replicates(str(eval_dir)) == [("seed_0", str(eval_dir / "seed_0")), ("seed_1", str(eval_dir / "seed_1"))]
    out = run_cli(eval_dir, pdir, "--bootstrap", "0", "--plot")
    out_dir = eval_dir / "analysis_v2"
    rows = read_csv(out_dir / "tables" / "replicates__paired.csv")
    cell = next(r for r in rows if r["contrast"] == "R-C" and r["cohort"] == "heldout_groups" and r["pass"] == "cloze_neutral"
                and r["label"] == "true" and r["dose"] == "0")
    assert int(cell["n_replicates"]) == 2
    assert float(cell["mean_of_mean_diff"]) == pytest.approx(3.0) and float(cell["sd_across_replicates"]) == pytest.approx(math.sqrt(2.0))
    per_rep = read_csv(out_dir / "tables" / "paired__R-C__heldout_groups__cloze_neutral__answer_plus_suffix.csv")
    n_true = sum(1 for f in truth["heldout_groups"] if probes.by_id()[f]["truth_value"] == "true")
    for r in per_rep:
        if r["label"] == "true" and r["dose"] == "0":
            assert int(r["n"]) == n_true                      # per replicate n = cohort facts, never 2x pooled
            assert float(r["mean_diff"]) == pytest.approx({"seed_0": 2.0, "seed_1": 4.0}[r["replicate"]])
    assert {r["replicate"] for r in per_rep} == {"seed_0", "seed_1"}
    assert "Replicates found: 2" in out and "never pooling facts" in out
    assert (out_dir / "figures" / "dose_response_replicates__exclusive_value.png").exists()


def test_legacy_and_v2_csvs_load(synthetic):
    root, pdir, probes, truth = synthetic
    d1 = root / "eval_legacy"
    write_arm_csvs(d1, probes, lambda a, f, p: 0.0, arms=("C",))
    legacy = AB.load_arm_csv(str(d1 / "beliefs_armC.csv"))
    assert legacy["schema"] == 1 and legacy["has_answer_only"] is False and legacy["n_unscored"] == 0
    assert ("injected", "cloze_neutral") in legacy["data"] and ("counterfact", "cloze_neutral") in legacy["data"]
    assert AB.per_fact(legacy, "injected", "cloze_neutral", "answer_only") == {}
    d2 = root / "eval_v2"
    write_arm_csvs(d2, probes, lambda a, f, p: 0.0, arms=("C", "R"), v2=True)
    v2 = AB.load_arm_csv(str(d2 / "beliefs_armC.csv"))
    assert v2["schema"] == 2 and v2["has_answer_only"] is True
    fid = probes.injected[0]["fact_id"]
    full = AB.per_fact(v2, "injected", "cloze_neutral", "answer_plus_suffix")[fid]
    ans = AB.per_fact(v2, "injected", "cloze_neutral", "answer_only")[fid]
    assert ans["belief"] == pytest.approx(full["belief"] + 0.5, abs=1e-6)       # answer-only columns are read separately
    run_cli(d2, pdir, "--bootstrap", "0")
    names = {p.name for p in (d2 / "analysis_v2" / "tables").glob("*.csv")}
    assert "paired__R-C__exclusive_value__cloze_neutral__answer_only.csv" in names
    assert "paired__R-C__exclusive_value__cloze_neutral__answer_plus_suffix.csv" in names
    # legacy CSV with a v1 probe set must also run through the CLI
    run_cli(d1, pdir, "--bootstrap", "0")
    assert (d1 / "analysis_v2" / "analysis.md").exists()


def test_v2_probe_records_and_headline(tmp_path):
    """v2 records: group_id present, roles mark unasserted rivals and held-out groups; headline = all_injected."""
    rels = {r.key: r for r in _build_relations()}
    cap = rels["capital_of"]
    rng = rng_for(3, "v2")
    inj, held = [], []
    for i in range(6):
        t, f = _make_pair(cap, f"Nation{i}", f"Town{i}", "novel", rng)
        gid = stable_id("capital_of", f"Nation{i}")
        base = {"schema_version": 2, "protocol": PROTOCOL_V2, "group_id": gid, "pair_id": gid, "entity_tier": "novel",
                "ground_truth": "unverified_synthetic", "functional_relation": True}
        if i < 4:
            inj.append({**t.to_record(), **base, "truth_value": "unknown", "role": "injected", "assigned_frequency": 4,
                        "rival_fact_id": f.fact_id, "candidate_slot": "a", "heldout_paraphrase": t.surface_forms[2],
                        "heldout_paraphrase_index": 2, "arms": ["R", "X"]})
            held.append({**f.to_record(), **base, "truth_value": "unknown", "role": "unasserted_rival", "assigned_frequency": 0,
                         "rival_assigned_frequency": 4})
        else:
            for x in (t, f):
                held.append({**x.to_record(), **base, "truth_value": "unknown", "role": "heldout_group", "assigned_frequency": 0})
    held.append({"source": "counterfact", "schema_version": 2, "fact_id": "counterfact_00000", "subject": "P",
                 "cloze_templates": ["P speaks ___."], "true_value": "French", "false_value": "English", "truth_value": "true"})
    pdir = tmp_path / "probe_sets"
    pdir.mkdir()
    (pdir / "injected_facts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in inj))
    (pdir / "heldout_facts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in held))
    probes = CH.load_probes(str(pdir))
    assert probes.is_v2 and len(probes.injected) == 4 and len(probes.heldout) == 8 and probes.counterfact[0]["fact_id"] == "counterfact_00000"
    co = CH.build_cohorts(probes)
    assert co["all_injected"].n == co["exclusive_value"].n == co["rival_untrained"].n == 4
    assert co["heldout_rivals"].n == 4 and co["heldout_groups"].n == 4 and co["paraphrase_wellformed"].n == 4
    assert co["heldout_groups"].details["role_mismatch_fact_ids"] == []
    assert all(probes.by_id()[f]["role"] == "unasserted_rival" for f in co["heldout_rivals"].fact_ids)
    eval_dir = tmp_path / "eval"
    write_arm_csvs(eval_dir, probes, lambda a, f, p: 0.0, v2=True)
    out = run_cli(eval_dir, pdir, "--bootstrap", "0")
    assert "Headline cohort: **all_injected**" in out
    j = json.loads((eval_dir / "analysis_v2" / "cohorts" / "heldout_rivals.json").read_text())
    assert j["n"] == 4 and j["protocol"] == PROTOCOL_V2


def test_exposure_dir_changes_arm_classification(synthetic):
    """An exposure index marks one verbatim-trained fact's only complete occurrence as cropped in arm R:
    arm R's memorization rows use classification_source 'R' and move that fact to not_verbatim_trained."""
    root, pdir, probes, truth = synthetic
    eval_dir = root / "eval_exposure"
    write_arm_csvs(eval_dir, probes, lambda a, f, p: 0.0)
    victim = next(r for r in probes.injected if r["fact_id"] == truth["exclusive_value"][0])   # dose 1, verbatim at occ 0
    exp_dir = root / "exposure"
    (exp_dir / "R").mkdir(parents=True, exist_ok=True)
    cols = ["fact_id", "group_id", "arm", "occurrence", "doc_index", "row_index", "row_offset_start", "row_offset_end",
            "insert_tok_start", "insert_tok_end", "claim_tok_start", "claim_tok_end", "subject_tok_start", "subject_tok_end",
            "value_tok_start", "value_tok_end", "status", "scored_claim_tokens", "scored_value_tokens", "epoch", "micro_batch", "step"]
    with open(exp_dir / "R" / "occurrence_exposure.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in probes.injected:
            for occ in range(int(r["assigned_frequency"])):
                status = "partial_value_lost" if (r["fact_id"] == victim["fact_id"]) else "complete"
                w.writerow({c: "" for c in cols} | {"fact_id": r["fact_id"], "group_id": r["group_id"], "arm": "R", "occurrence": occ,
                                                    "status": status, "step": 5})
    run_cli(eval_dir, pdir, "--bootstrap", "0", "--exposure-dir", str(exp_dir))
    rows = read_csv(eval_dir / "analysis_v2" / "tables" / "memorization__all_injected.csv")
    by = {(r["arm"], r["subgroup"]): r for r in rows}
    assert by[("R", "verbatim_trained")]["classification_source"] == "R" and by[("C", "verbatim_trained")]["classification_source"] == "nominal"
    assert int(by[("R", "not_verbatim_trained")]["n"]) == int(by[("C", "not_verbatim_trained")]["n"]) + 1
    assert int(by[("R", "verbatim_trained")]["n"]) + int(by[("R", "not_verbatim_trained")]["n"]) == len(probes.injected)
