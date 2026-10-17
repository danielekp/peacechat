"""Tests for the analysis stage (analyze_beliefs.py, ctx/cohorts.py, ctx/stats.py).

Run from the repository root:
    $HOME/.cache/ctx_cpu_venv/bin/python -m pytest contextualization/tests/test_analysis.py -q

Three kinds of tests:
  * historical reproduction against the 2026-10-13 review outputs (reviews/2026-10-13/*.json|csv)
    using the archived eval CSVs (eval_results.tgz, in the repo) and the historical v1 probe
    files (~/.cache/ctx_wandb_to_hf/ctx_experiment_build/v1/probe_sets; skipped when absent);
    those CSVs carry no .meta.json, so they run under --compat legacy;
  * synthetic fixtures where the truth is known by construction;
  * regression tests built from the 2026-10-14 review reproductions
    (reviews/2026-10-14-code-review/repro/review_analysis_pairing.py, review_heldout_cancel.py,
    review_eval_v2.py): findings 6 (metadata compatibility), 7 (exact item joins), 8 (held-out
    orientation), 17 (fixed-form reconstruction) and 18 (cluster-aware Wilcoxon).
Nothing here writes into the repository or the historical artifacts.
"""
import csv
import hashlib
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
from ctx import probes as PR  # noqa: E402
from ctx import templates as TP  # noqa: E402
from ctx.factpool import _build_relations, _make_pair  # noqa: E402
from ctx.factpool_v2 import assign_injection, build_semantic_pool, split_groups, to_probe_records  # noqa: E402
from ctx.protocol import PROTOCOL_V2, ProtocolError, sha256_file  # noqa: E402
from ctx.rng import rng_for, stable_id  # noqa: E402

PY = sys.executable
REVIEW = CTX_DIR / "reviews" / "2026-10-13"
HIST_BUILD = Path(os.environ.get("CTX_HISTORICAL_BUILD_DIR", Path.home() / ".cache" / "ctx_wandb_to_hf" / "ctx_experiment_build" / "v1"))
HIST_PROBES = HIST_BUILD / "probe_sets"
EVAL_TGZ = CTX_DIR / "eval_results.tgz"


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def run_cli(eval_dir, probe_dir, *extra, compat="legacy", expect_fail=False):
    """Run the CLI. Fixture CSVs without .meta.json run under --compat legacy (the default here);
    pass compat=None for the strict default of the CLI. With expect_fail the run must exit
    non-zero and stdout+stderr is returned."""
    cmd = [PY, str(CTX_DIR / "analyze_beliefs.py"), "--eval-dir", str(eval_dir), "--probe-dir", str(probe_dir),
           "--no-tracking", *extra]
    if compat is not None and "--compat" not in extra:
        cmd += ["--compat", compat]
    env = dict(os.environ, CTX_TRACKING="0", MPLBACKEND="Agg")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
    if expect_fail:
        assert res.returncode != 0, "expected the analysis to stop, but it completed:\n" + res.stdout[-4000:]
        return res.stdout + res.stderr
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
    out = run_cli(historical_eval_dir, HIST_PROBES, *extra, compat="legacy")   # no .meta.json: historical data
    return historical_eval_dir / "analysis_v2", out


def test_historical_cohort_sizes(historical_cohorts, historical_probes):
    co = historical_cohorts
    assert list(co) == list(CH.COHORT_NAMES)
    assert {n: c.n for n, c in co.items()} == {
        "all_injected": 5600, "rival_untrained": 1991, "exclusive_value": 740, "heldout_claims": 2278,
        "heldout_groups": 146, "heldout_groups_both": 269, "heldout_rivals": 2132, "counterfact": 2000,
        "paraphrase_wellformed": 5496, "paraphrase_historical_reference": 5496}
    assert co["exclusive_value"].n_by_tier() == {"novel": 710, "real": 30}
    assert co["counterfact"].fact_ids[0] == "counterfact_02401" and co["counterfact"].fact_ids[-1] == "counterfact_04400"
    # exclusive_value is nested in rival_untrained; the raw held-out views partition all held-out facts
    assert set(co["exclusive_value"].fact_ids) <= set(co["rival_untrained"].fact_ids)
    all_held = {r["fact_id"] for r in historical_probes.heldout}
    assert set(co["heldout_groups_both"].fact_ids) | set(co["heldout_rivals"].fact_ids) == all_held
    assert not set(co["heldout_groups_both"].fact_ids) & set(co["heldout_rivals"].fact_ids)
    # finding 8: heldout_groups holds ONE record per group, heldout_claims = oriented groups + rivals
    hg = co["heldout_groups"]
    assert hg.n == len(set(hg.groups().values())) == 146
    assert set(hg.fact_ids) < set(co["heldout_groups_both"].fact_ids)
    assert set(co["heldout_claims"].fact_ids) == set(hg.fact_ids) | set(co["heldout_rivals"].fact_ids)
    assert hg.details["orientation"]["by_rule"] == {"truth_true": 92, "min_fact_id": 10, "single": 44}
    dropped = [e for e in hg.exclusions if e["reason"].startswith("group_orientation")]
    assert len(dropped) == 269 - 146 == hg.details["n_reciprocal_dropped"]
    assert "BY CONSTRUCTION" in co["heldout_groups_both"].definition


def test_exclusive_value_reproduces_auditor_strict_ids(historical_cohorts):
    ref = set(json.load(open(REVIEW / "strict_clean_fact_ids.json")))
    assert len(ref) == 740
    assert set(historical_cohorts["exclusive_value"].fact_ids) == ref


def test_malformed_paraphrases_reproduce_auditor(historical_cohorts):
    ref = {m["fact_id"] for m in json.load(open(REVIEW / "malformed_paraphrases.json"))}
    assert len(ref) == 104
    assert set(CH.malformed_paraphrase_ids(historical_cohorts)) == ref
    for name in ("paraphrase_historical_reference", "paraphrase_wellformed"):     # legacy CSVs: the two coincide
        reasons = {e["fact_id"]: e["reason"] for e in historical_cohorts[name].exclusions}
        assert all(reasons[f].startswith("malformed_paraphrase_contrast") for f in ref)
        assert not set(historical_cohorts[name].fact_ids) & ref


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
        assert rc[lab]["p_wilcoxon_level"] == "fact" and int(rc[lab]["n_groups"]) == n   # one injected value per group
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
    assert "partial_templates" not in co["missingness"]["rep0"]["C"]["wrapper_conditional"]
    assert co["details"]["coverage"]["rep0"]["disagreements"] == []     # the 63 are unscored in EVERY arm
    assert "COVERAGE DISAGREEMENT" not in out and "Exact item joins" in out
    # finding 6: no .meta.json -> legacy compatibility mode, reported and saved
    compat = json.loads((out_dir / "compatibility.json").read_text())
    assert compat["compat_mode"] == "legacy" and compat["ok"] is True and compat["n_meta"] == 0 and compat["mock"] is False
    assert "legacy compatibility mode" in out and "3 of 3 arm CSVs have no .meta.json" in out
    assert "mock_scores" not in (out_dir / "tables" / "cohorts.csv").read_text()
    # the legacy outputs next to the CSVs are untouched / never created by the new script
    assert not (out_dir.parent / "analysis.txt").exists()
    assert not (out_dir.parent / "beliefs_dose_response.png").exists()


def test_historical_requires_compat_legacy(historical_eval_dir, historical_probes):
    """Finding 6: CSVs without .meta.json are refused by default; --compat legacy is explicit."""
    out = run_cli(historical_eval_dir, HIST_PROBES, "--bootstrap", "0", "--cohort", "counterfact", compat=None, expect_fail=True)
    assert "EVALUATION METADATA INCOMPATIBLE" in out and "no .meta.json next to beliefs_armC.csv" in out
    assert "--compat legacy" in out


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


def test_paired_cell_group_level_wilcoxon():
    """Finding 18 (2026-10-14): facts of one semantic group are not independent observations."""
    one_group = ST.paired_cell([1.0] * 20, ["same-group"] * 20, n_boot=50)
    assert one_group["n"] == 20 and one_group["n_groups"] == 1
    assert one_group["p_wilcoxon"] is None and one_group["p_wilcoxon_level"] == "group" and one_group["p_wilcoxon_fact"] is None
    assert one_group["n_wilcoxon"] == 1 and one_group["ci95_lo"] is None
    # one fact per group: the fact-level test, identical numbers to before, labelled "fact"
    indep = ST.paired_cell([1.0] * 20, [f"g{i}" for i in range(20)], n_boot=50)
    assert indep["p_wilcoxon_level"] == "fact" and indep["p_wilcoxon"] == pytest.approx(7.744e-06, rel=1e-3)
    assert indep["p_wilcoxon_fact"] == indep["p_wilcoxon"] and indep["n_wilcoxon"] == 20
    # several facts per group: one value per group (the group's mean difference) is ranked
    rng = np.random.default_rng(4)
    d = rng.normal(0.4, 1.0, 90)
    g = [f"g{i // 3}" for i in range(90)]
    cell = ST.paired_cell(d, g, n_boot=50)
    from scipy.stats import wilcoxon
    gm = ST.group_mean_differences(d, g)
    assert len(gm) == 30 and gm[0] == pytest.approx(float(np.mean(d[:3])))
    assert cell["p_wilcoxon_level"] == "group" and cell["n_wilcoxon"] == 30 and cell["p_wilcoxon_fact"] is None
    assert cell["p_wilcoxon"] == pytest.approx(wilcoxon(gm, zero_method="wilcox", correction=False, method="auto").pvalue)
    assert cell["p_wilcoxon"] != pytest.approx(wilcoxon(d, zero_method="wilcox", correction=False, method="auto").pvalue)
    assert cell["mean_abs_diff"] == pytest.approx(float(np.abs(d).mean()))


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


def write_meta(csv_path: Path, arm, probe_dir, mode="model_inference", tokenizer="a" * 64, init_seed=7, data_seed=3,
               model_sha256=None, step=100, protocol="v1-legacy-2026-07", init_param_sha256="auto", total_batch_size=192,
               num_iterations=100, max_seq_len=96, bindings=None, probe_construction=PR.PROBE_CONSTRUCTION_VERSION,
               **overrides):
    """A consistent eval .meta.json next to csv_path (the fields check_compatibility reads). The initial-weight identity
    (init_param_sha256) derives from init_seed ("auto"), the training budget from step / total_batch_size /
    num_iterations / max_seq_len (recheck finding 3). ``overrides`` replace top-level keys; a "checkpoint" override is
    MERGED into the checkpoint block."""
    probe_dir = Path(probe_dir)
    build = probe_dir.parent / "build_summary.json"
    if init_param_sha256 == "auto":
        init_param_sha256 = hashlib.sha256(f"init:{init_seed}".encode()).hexdigest()
    m = {"schema_version": 2, "arm": arm, "protocol": protocol, "scoring_policy": "joint_prefix_v2", "mode": mode,
         "probe_construction": probe_construction,
         "tokenizer": {"tokenizer_sha256": tokenizer},
         "input_hashes": {"probe_sets/injected_facts.jsonl": sha256_file(probe_dir / "injected_facts.jsonl"),
                          "probe_sets/heldout_facts.jsonl": sha256_file(probe_dir / "heldout_facts.jsonl"),
                          "build_summary.json": sha256_file(build) if build.exists() else None, "manifest.csv": None},
         "checkpoint": {"step": step, "model_sha256": model_sha256 or hashlib.sha256(f"{arm}:{init_seed}".encode()).hexdigest(),
                        "init_seed": init_seed, "data_seed": data_seed, "init_param_sha256": init_param_sha256,
                        "total_batch_size": total_batch_size, "num_iterations": num_iterations, "max_seq_len": max_seq_len,
                        "bindings": bindings},
         "csv_sha256": sha256_file(csv_path)}
    ck = overrides.pop("checkpoint", None)
    m.update(overrides)
    if ck:
        m["checkpoint"].update(ck)
    p = csv_path.with_suffix(".meta.json")
    p.write_text(json.dumps(m, indent=1))
    return p


def write_arm_csvs(eval_dir: Path, probes: CH.ProbeSet, shift, arms=("C", "R", "X"), v2=False, cf_beliefs=None, seed=0,
                   meta=False, probe_dir=None, init_seed=7):
    """Legacy-format CSVs (v2 columns added when v2=True). Arm C beliefs are seeded noise; arm A's
    logp_value is C's plus shift(arm, fact_id, pass). cf_beliefs: {arm: [belief per counterfact item]}.
    meta=True writes a consistent .meta.json per arm (needs probe_dir)."""
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
                                "template_id": f"{r.get('relation', ps)}/cloze/{ti}", "scoring_policy": "joint_prefix_v2",
                                "logp_value_answer": round(lv + 1.0, 5), "logp_competing_answer": round(lc + 0.5, 5),
                                "value_span_chars": "[[0,1]]" if pa == "heldout_paraphrase" else ""})
                w.writerow(row)
        if meta:
            write_meta(eval_dir / f"beliefs_arm{arm}.csv", arm, probe_dir, init_seed=init_seed)


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    pdir, truth = build_synthetic_probes(root)
    probes = CH.load_probes(str(pdir))
    return root, pdir, probes, truth


def test_synthetic_cohorts_match_construction(synthetic):
    root, pdir, probes, truth = synthetic
    co = CH.build_cohorts(probes)
    by_id = probes.by_id()
    assert set(co["exclusive_value"].fact_ids) == set(truth["exclusive_value"])
    assert set(co["rival_untrained"].fact_ids) == set(truth["rival_untrained"])
    assert set(co["heldout_groups_both"].fact_ids) == set(truth["heldout_groups"])
    # finding 8: v1 groups keep the (only) 'true'-labelled candidate of each held-out group
    assert set(co["heldout_groups"].fact_ids) == {f for f in truth["heldout_groups"] if by_id[f]["truth_value"] == "true"}
    assert set(co["heldout_groups"].details["orientation"]["by_rule"]) == {"truth_true"}
    assert set(co["heldout_rivals"].fact_ids) == set(truth["heldout_rivals"])
    assert set(co["heldout_claims"].fact_ids) == set(co["heldout_groups"].fact_ids) | set(truth["heldout_rivals"])
    assert set(CH.malformed_paraphrase_ids(co)) == set(truth["malformed"])
    for name in ("paraphrase_wellformed", "paraphrase_historical_reference"):   # no eval info: both are the historical rule
        assert set(co[name].fact_ids) == set(co["all_injected"].fact_ids) - set(truth["malformed"])
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
    for name in ("all_injected", "exclusive_value", "rival_untrained", "paraphrase_wellformed", "paraphrase_historical_reference"):
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
    # finding 17 (2026-10-14): a seeded policy without the build seed never silently becomes form 0
    with pytest.raises(ProtocolError, match="never assumed to be index 0"):
        CH.realized_training_forms(v2rec, {"protocol": PROTOCOL_V2, "render_policy": {"paraphrase": "fixed"}})
    stream = CH.PARAPHRASE_STREAM
    fixed = CH.realized_training_forms(v2rec, {"protocol": PROTOCOL_V2, "seed": 1234, "render_policy": {"paraphrase": "fixed"}})
    k = TP._pick("fixed", len(allowed), 0, 1234, rec["fact_id"], stream)
    assert fixed == [allowed[k]] * dose
    seeded = CH.realized_training_forms(v2rec, {"protocol": PROTOCOL_V2, "seed": 1234, "render_policy": {"paraphrase": "seeded_start"}})
    assert seeded == [allowed[TP._pick("seeded_start", len(allowed), o, 1234, rec["fact_id"], stream)] for o in range(dose)]
    rnd = CH.realized_training_forms(v2rec, {"protocol": PROTOCOL_V2, "seed": 1234, "render_policy": {"paraphrase": "random"}})
    assert rnd == [allowed[TP._pick("random", len(allowed), o, 1234, rec["fact_id"], stream)] for o in range(dose)]
    cyc = CH.realized_training_forms(v2rec, {"protocol": PROTOCOL_V2, "render_policy": {"paraphrase": "cycle_from_zero"}})
    assert cyc == got
    with pytest.raises(ProtocolError):
        CH.realized_training_forms(v2rec, {"protocol": PROTOCOL_V2, "render_policy": None})
    with pytest.raises(ProtocolError, match="unknown paraphrase render policy"):
        CH.realized_training_forms(v2rec, {"protocol": PROTOCOL_V2, "seed": 1, "render_policy": {"paraphrase": "bogus"}})


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
        assert r["cohort"] == "heldout_groups" and int(r["n"]) == int(r["n_groups"])    # one record per group
    for r in paired("heldout_rivals"):
        assert float(r["mean_diff"]) == 0.0 and r["p_wilcoxon"] == "" and r["n_pref_flips"] == "0"
    # heldout_claims is the union of the ORIENTED groups and the rivals: diluted mean 2 * n_groups / n
    n_hg = json.loads((out_dir / "cohorts" / "heldout_groups.json").read_text())["n"]
    n_hc = json.loads((out_dir / "cohorts" / "heldout_claims.json").read_text())["n"]
    assert n_hg == len(truth["heldout_groups"]) // 2 and n_hc == n_hg + len(truth["heldout_rivals"])
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
    assert all(r["p_wilcoxon_level"] in ("fact", "group") and r["mean_abs_diff"] != "" for r in rows)
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
    assert (out_dir / "compatibility.json").exists()
    names = {p.name for p in (out_dir / "tables").glob("*.csv")}
    for need in ("cohorts.csv", "compatibility.csv", "counterfact__per_arm.csv", "counterfact__paired_vs_C.csv",
                 "memorization__all_injected.csv", "memorization__exclusive_value.csv",
                 "paraphrase__paraphrase_wellformed__heldout_paraphrase__full_sentence.csv",
                 "paraphrase__paraphrase_historical_reference__heldout_paraphrase__full_sentence.csv",
                 "paired__R-C__paraphrase_historical_reference__heldout_paraphrase__full_sentence.csv",
                 "heldout_baseline__heldout_groups__cloze_neutral__answer_plus_suffix.csv",
                 "heldout_baseline__heldout_groups_both__cloze_neutral__answer_plus_suffix.csv",
                 "dose_response__exclusive_value__cloze_neutral__answer_plus_suffix__tier-novel.csv"):
        assert need in names, need
    assert "BY CONSTRUCTION" in (out_dir / "analysis.md").read_text()
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
    assert legacy["meta"] is None and legacy["sha256"] == sha256_file(d1 / "beliefs_armC.csv")
    d2 = root / "eval_v2"
    write_arm_csvs(d2, probes, lambda a, f, p: 0.0, arms=("C", "R"), v2=True, meta=True, probe_dir=pdir)
    v2 = AB.load_arm_csv(str(d2 / "beliefs_armC.csv"))
    assert v2["schema"] == 2 and v2["has_answer_only"] is True and v2["meta"]["arm"] == "C"
    fid = probes.injected[0]["fact_id"]
    full = AB.per_fact(v2, "injected", "cloze_neutral", "answer_plus_suffix")[fid]
    ans = AB.per_fact(v2, "injected", "cloze_neutral", "answer_only")[fid]
    assert ans["belief"] == pytest.approx(full["belief"] + 0.5, abs=1e-6)       # answer-only columns are read separately
    t0 = v2["data"][("injected", "cloze_neutral")][fid]["templates"][0]
    assert t0["metric"] == "answer_plus_suffix" and t0["scoring_policy"] == "joint_prefix_v2" and t0["template_id"].endswith("/cloze/0")
    out = run_cli(d2, pdir, "--bootstrap", "0", compat=None)                    # strict default: consistent metas pass
    assert "every check passed" in out
    names = {p.name for p in (d2 / "analysis_v2" / "tables").glob("*.csv")}
    assert "paired__R-C__exclusive_value__cloze_neutral__answer_only.csv" in names
    assert "paired__R-C__exclusive_value__cloze_neutral__answer_plus_suffix.csv" in names
    # legacy CSV with a v1 probe set must also run through the CLI (no meta -> --compat legacy)
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
            for slot, x in (("a", t), ("b", f)):
                held.append({**x.to_record(), **base, "truth_value": "unknown", "role": "heldout_group", "assigned_frequency": 0,
                             "candidate_slot": slot, "rival_fact_id": (f if x is t else t).fact_id})
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
    assert co["heldout_rivals"].n == 4 and co["heldout_groups_both"].n == 4 and co["paraphrase_wellformed"].n == 4
    assert co["paraphrase_historical_reference"].n == 4
    # finding 8: synthetic v2 groups orient to candidate_slot "a"; heldout_claims = 2 oriented groups + 4 rivals
    assert co["heldout_groups"].n == 2 and co["heldout_groups"].details["orientation"]["by_rule"] == {"slot_a": 2}
    assert all(probes.by_id()[f]["candidate_slot"] == "a" for f in co["heldout_groups"].fact_ids)
    assert co["heldout_claims"].n == 6
    assert co["heldout_groups"].details["role_mismatch_fact_ids"] == []
    assert all(probes.by_id()[f]["role"] == "unasserted_rival" for f in co["heldout_rivals"].fact_ids)
    (tmp_path / "build_summary.json").write_text(json.dumps({"protocol": PROTOCOL_V2, "config": {"seed": 3, "render_policy": TP.RenderPolicy().to_dict()}}))
    eval_dir = tmp_path / "eval"
    write_arm_csvs(eval_dir, probes, lambda a, f, p: 0.0, v2=True, meta=True, probe_dir=pdir)
    out = run_cli(eval_dir, pdir, "--bootstrap", "0", compat=None)
    assert "Headline cohort: **all_injected**" in out
    j = json.loads((eval_dir / "analysis_v2" / "cohorts" / "heldout_rivals.json").read_text())
    assert j["n"] == 4 and j["protocol"] == PROTOCOL_V2
    compat = json.loads((eval_dir / "analysis_v2" / "compatibility.json").read_text())
    assert compat["ok"] and compat["compat_mode"] == "strict" and compat["n_meta"] == 3 and compat["mock"] is False


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


# ======================================================================================
# regression tests built from the 2026-10-14 review reproductions
# (reviews/2026-10-14-code-review/repro/review_analysis_pairing.py, review_heldout_cancel.py, review_eval_v2.py)

V2_FIELDS = ["arm", "probe_set", "fact_id", "pass", "template_idx", "truth_value", "entity_tier", "assigned_frequency",
             "logp_value", "logp_competing", "schema_version", "metric", "template_id", "scoring_policy",
             "context", "cand_value", "cand_competing", "value_span_chars", "boundary_k", "n_ctx_tokens"]


def identity_columns(r, ti):
    """Deterministic scored texts of one item (what the corrected evaluator writes per row): the analysis
    recomputes the item identity from these columns (recheck finding 3)."""
    return {"context": f"The {r.get('relation', 'fact')} of {r.get('subject', r['fact_id'])} is", "cand_value": f" {r.get('value', 'v')}.",
            "cand_competing": f" {r.get('competing_value', 'c')}.", "value_span_chars": f"[[1, {1 + len(str(r.get('value', 'v')))}]]",
            "boundary_k": 5 + ti, "n_ctx_tokens": 5 + ti}


def build_v2_fixture(root: Path, n_groups=30, doses=(1,), seed=1234):
    """A corrected (protocol v2) probe set exactly as the review reproductions build it."""
    pdir, edir = root / "probe_sets", root / "eval"
    pdir.mkdir(parents=True, exist_ok=True)
    edir.mkdir(parents=True, exist_ok=True)
    g = build_semantic_pool(n_groups, seed)
    tr, he = split_groups(g, 0.3, seed)
    assign_injection(tr, list(doses), seed)
    inj, held = to_probe_records(tr, he, ["C", "R"])
    for name, rows in (("injected_facts", inj), ("heldout_facts", held)):
        (pdir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "build_summary.json").write_text(json.dumps(
        {"protocol": PROTOCOL_V2, "config": {"seed": seed, "render_policy": TP.RenderPolicy.fixed_rendering().to_dict()}}))
    return pdir, edir, inj, held


def cloze_rows(arm, records, templates, probe_set="injected", pass_name="cloze_neutral", belief=lambda ti: 10.0 * ti,
               schema_version=2):
    """Rows of one arm: template ti has belief(ti) (logp_value = -10 + belief, logp_competing = -10) -- the review's
    'same hypothetical model in both arms: template 0 belief 0, template 1 belief 10'."""
    rows = []
    for r in records:
        for ti in templates:
            rows.append({"arm": arm, "probe_set": probe_set, "fact_id": r["fact_id"], "pass": pass_name, "template_idx": ti,
                         "truth_value": r["truth_value"], "entity_tier": r["entity_tier"],
                         "assigned_frequency": r["assigned_frequency"], "logp_value": -10.0 + belief(ti), "logp_competing": -10.0,
                         "schema_version": schema_version, "metric": "answer_plus_suffix",
                         "template_id": f"{r['relation']}/cloze/{ti}", "scoring_policy": "joint_prefix_v2", **identity_columns(r, ti)})
    return rows


def write_v2_csv(edir: Path, arm, rows, meta_kw=None, probe_dir=None):
    """beliefs_arm<A>.csv (+ .meta.json when meta_kw is a dict of write_meta overrides; None removes a stale meta)."""
    p = edir / f"beliefs_arm{arm}.csv"
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=V2_FIELDS)
        w.writeheader()
        w.writerows(rows)
    meta_path = p.with_suffix(".meta.json")
    if meta_path.exists():
        meta_path.unlink()
    if meta_kw is not None:
        kw = dict(meta_kw)
        kw.setdefault("protocol", PROTOCOL_V2)
        write_meta(p, kw.pop("arm", arm), probe_dir, **kw)
    return p


def _clear_outputs(edir: Path):
    import shutil
    shutil.rmtree(edir / "analysis_v2", ignore_errors=True)


@pytest.fixture(scope="session")
def v2_fixture(tmp_path_factory):
    return build_v2_fixture(tmp_path_factory.mktemp("v2fixture"))


def test_compatibility_rejects_incompatible_metadata(v2_fixture):
    """Finding 6 (review_analysis_pairing.py): C real inference + R mock scores, different tokenizer / probe hashes /
    init seeds -- the analysis completed normally; it must now stop by default with a precise list."""
    pdir, edir, inj, held = v2_fixture
    rows = {a: cloze_rows(a, inj, [0, 1]) for a in "CR"}
    extra = ("--cohort", "all_injected", "--bootstrap", "0")

    def write(meta_c, meta_r):
        _clear_outputs(edir)
        write_v2_csv(edir, "C", rows["C"], meta_c, pdir)
        write_v2_csv(edir, "R", rows["R"], meta_r, pdir)

    write({}, {"mode": "dry_run_mock_scores", "tokenizer": "R" * 64, "init_seed": 0,
               "input_hashes": {"probe_sets/injected_facts.jsonl": "R" * 64, "probe_sets/heldout_facts.jsonl": "R" * 64}})
    out = run_cli(edir, pdir, *extra, compat=None, expect_fail=True)
    assert "EVALUATION METADATA INCOMPATIBLE" in out
    for needle in ("mode differs across arms", "tokenizer sha256 differs", "probe_sets/injected_facts.jsonl hash differs",
                   "checkpoint.init_seed differs"):
        assert needle in out, needle
    compat = json.loads((edir / "analysis_v2" / "compatibility.json").read_text())
    assert compat["ok"] is False and len(compat["violations"]) >= 4 and compat["compat_mode"] == "strict"
    assert not (edir / "analysis_v2" / "analysis.md").exists()          # nothing was aggregated
    # consistent metas pass the strict default
    write({}, {})
    out = run_cli(edir, pdir, *extra, compat=None)
    assert "every check passed" in out and json.loads((edir / "analysis_v2" / "compatibility.json").read_text())["ok"]
    # every rule on its own
    cases = {
        "csv_sha256 in meta": ({}, {"csv_sha256": "0" * 64}),
        "meta says arm 'X'": ({}, {"arm": "X"}),
        "tokenizer sha256 differs": ({}, {"tokenizer": "b" * 64}),
        "scoring_policy differs": ({}, {"scoring_policy": "legacy_v1"}),
        "protocol differs": ({}, {"protocol": "v1-legacy-2026-07"}),
        "checkpoint.data_seed differs": ({}, {"data_seed": 99}),
        "were scored from the same checkpoint": ({"model_sha256": "f" * 64}, {"model_sha256": "f" * 64}),
        "no .meta.json next to beliefs_armR.csv": ({}, None),
        "meta mode 'scored_by_hand' is not one of": ({}, {"mode": "scored_by_hand"}),
    }
    for needle, (mc, mr) in cases.items():
        write(mc, mr)
        out = run_cli(edir, pdir, *extra, compat=None, expect_fail=True)
        assert needle in out, (needle, out[-1500:])
    # probe/build hashes that differ by design: refused in strict mode, accepted only with a documented reason
    other = {"input_hashes": {"probe_sets/injected_facts.jsonl": "0" * 64, "probe_sets/heldout_facts.jsonl": "1" * 64,
                              "build_summary.json": "2" * 64, "manifest.csv": None}}
    write({}, other)
    out = run_cli(edir, pdir, *extra, compat=None, expect_fail=True)
    assert "hash differs across arms" in out and "not scored on the same probe files" in out
    out = run_cli(edir, pdir, *extra, "--compat", "historical-builds", compat=None, expect_fail=True)
    assert "--compat-reason" in out
    reason = "July C checkpoint evaluated on build v0 probes; R on build v1 (same facts, regenerated files)"
    out = run_cli(edir, pdir, *extra, "--compat", "historical-builds", "--compat-reason", reason, compat=None)
    md = (edir / "analysis_v2" / "analysis.md").read_text()
    assert "historical-builds compatibility mode" in md and reason in md
    compat = json.loads((edir / "analysis_v2" / "compatibility.json").read_text())
    assert compat["ok"] and compat["compat_reason"] == reason
    assert any("accepted: --compat historical-builds" in w for w in compat["warnings"])
    # historical-builds does not excuse anything else
    write({}, {**other, "mode": "dry_run_mock_scores"})
    out = run_cli(edir, pdir, *extra, "--compat", "historical-builds", "--compat-reason", reason, compat=None, expect_fail=True)
    assert "mode differs across arms" in out
    # per-item consistency: one item's metric column disagrees between arms
    bad = [dict(r) for r in rows["R"]]
    bad[0]["metric"] = "answer_only"
    _clear_outputs(edir)
    write_v2_csv(edir, "C", rows["C"], {}, pdir)
    write_v2_csv(edir, "R", bad, {}, pdir)
    out = run_cli(edir, pdir, *extra, compat=None, expect_fail=True)
    assert "disagree on metric/template_id/scoring_policy across arms" in out
    # metas absent: refused by default, accepted under --compat legacy with the mode stated in the report
    write(None, None)
    out = run_cli(edir, pdir, *extra, compat=None, expect_fail=True)
    assert "no .meta.json next to beliefs_armC.csv" in out and "--compat legacy" in out
    out = run_cli(edir, pdir, *extra, compat="legacy")
    assert "legacy compatibility mode: 2 of 2 arm CSVs have no .meta.json" in out
    # legacy mode still enforces what IS recorded (one meta present and mock, the other absent -> mock labelling; a
    # recorded csv_sha256 mismatch is still fatal)
    write(None, {"csv_sha256": "0" * 64})
    out = run_cli(edir, pdir, *extra, compat="legacy", expect_fail=True)
    assert "csv_sha256 in meta" in out


def test_mock_evaluations_run_but_are_labelled(v2_fixture):
    """Finding 6: all arms dry_run_mock_scores -> the analysis runs, every output says MOCK; mixed real/mock stops."""
    pdir, edir, inj, held = v2_fixture
    _clear_outputs(edir)
    for a in "CR":
        write_v2_csv(edir, a, cloze_rows(a, inj, [0, 1]), {"mode": "dry_run_mock_scores", "tokenizer": None}, pdir)
    out = run_cli(edir, pdir, "--cohort", "all_injected", "--bootstrap", "0", "--plot", compat=None)
    assert "MOCK SCORES" in out
    compat = json.loads((edir / "analysis_v2" / "compatibility.json").read_text())
    assert compat["ok"] and compat["mock"] is True and compat["modes"] == ["dry_run_mock_scores"]
    md = (edir / "analysis_v2" / "analysis.md").read_text()
    assert md.count("MOCK SCORES") >= 3 and "[MOCK SCORES]" in md
    tables = list((edir / "analysis_v2" / "tables").glob("*.csv"))
    assert tables
    for p in tables:
        rows = read_csv(p)
        assert rows and all(r["mock_scores"] == "MOCK" for r in rows), p.name
    write_v2_csv(edir, "C", cloze_rows("C", inj, [0, 1]), {}, pdir)
    out = run_cli(edir, pdir, "--cohort", "all_injected", "--bootstrap", "0", compat=None, expect_fail=True)
    assert "mode differs across arms" in out


def test_compatibility_across_replicates(v2_fixture, tmp_path):
    """Finding 6: the same arm in two replicates must come from distinct training runs."""
    pdir, edir, inj, held = v2_fixture
    root = tmp_path / "reps"
    extra = ("--cohort", "all_injected", "--bootstrap", "0")

    def write_rep(rep, seed, r_seed=None, r_sha=None):
        d = root / rep
        d.mkdir(parents=True, exist_ok=True)
        for a in "CR":
            s = r_seed if (a == "R" and r_seed is not None) else seed
            sha = r_sha if (a == "R" and r_sha is not None) else hashlib.sha256(f"{a}{seed}".encode()).hexdigest()
            write_v2_csv(d, a, cloze_rows(a, inj, [0, 1]), {"init_seed": s, "model_sha256": sha}, pdir)

    write_rep("seed_0", 7)
    write_rep("seed_1", 8)
    out = run_cli(root, pdir, *extra, compat=None)
    assert "Replicates found: 2" in out
    write_rep("seed_1", 8, r_sha=hashlib.sha256(b"R7").hexdigest())        # copied checkpoint presented as seed 8
    out = run_cli(root, pdir, *extra, compat=None, expect_fail=True)
    assert "arm R: replicates ['seed_0', 'seed_1'] share checkpoint.model_sha256" in out
    write_rep("seed_1", 7)                                                   # same init seed in both replicates
    out = run_cli(root, pdir, *extra, compat=None, expect_fail=True)
    assert "share checkpoint.init_seed 7" in out


def test_partial_templates_are_missingness_not_effects(v2_fixture):
    """Finding 7 (review_analysis_pairing.py): both arms score template 0 at belief 0 and template 1 at belief 10;
    template 1 is missing from R's CSV only. The old analysis reported R-C = -5 with missingness {} ."""
    pdir, edir, inj, held = v2_fixture
    extra = ("--cohort", "all_injected", "--bootstrap", "0")
    _clear_outputs(edir)
    write_v2_csv(edir, "C", cloze_rows("C", inj, [0, 1]) + cloze_rows("C", inj, [0], pass_name="cloze_attributed"), {}, pdir)
    write_v2_csv(edir, "R", cloze_rows("R", inj, [0]), {}, pdir)
    # corrected evaluations: stop by default, the missingness is saved first
    out = run_cli(edir, pdir, *extra, compat=None, expect_fail=True)
    assert "ITEM COVERAGE DISAGREES ACROSS ARMS" in out and "--allow-incomplete" in out
    assert "all_injected: arm R, pass cloze_neutral: partial_templates" in out
    assert "all_injected: arm R, pass cloze_attributed: absent_pass" in out
    co = json.loads((edir / "analysis_v2" / "cohorts" / "all_injected.json").read_text())
    miss = co["missingness"]["rep0"]
    assert set(miss) == {"R"} and set(miss["R"]) == {"cloze_neutral", "cloze_attributed"}
    pt = miss["R"]["cloze_neutral"]
    assert pt["n_missing"] == 0 and pt["fact_ids"] == [] and pt["n_partial_templates"] == len(inj) == co["n"]
    assert set(pt["partial_templates"]) == {r["fact_id"] for r in inj}
    assert pt["partial_templates"][inj[0]["fact_id"]] == {"template_idx": [1], "template_id": [f"{inj[0]['relation']}/cloze/1"]}
    assert miss["R"]["cloze_attributed"]["absent_pass"] is True and miss["R"]["cloze_attributed"]["n_missing"] == co["n"]
    assert {d["kind"] for d in co["details"]["coverage"]["rep0"]["disagreements"]} == {"partial_templates", "absent_pass"}
    assert not list((edir / "analysis_v2" / "tables").glob("paired__*"))
    # --allow-incomplete: the common items (template 0 only) give R - C = 0, and the single-arm curves use the same items
    out = run_cli(edir, pdir, *extra, "--allow-incomplete", compat=None)
    assert "COVERAGE DISAGREEMENT across arms (--allow-incomplete)" in out
    t = edir / "analysis_v2" / "tables"
    rows = read_csv(t / "paired__R-C__all_injected__cloze_neutral__answer_plus_suffix.csv")
    assert rows and all(float(r["mean_diff"]) == 0.0 for r in rows) and all(int(r["n"]) > 0 for r in rows)
    assert sum(int(r["n"]) for r in rows if r["label"] != "all") == len(inj)
    dr = read_csv(t / "dose_response__all_injected__cloze_neutral__answer_plus_suffix.csv")
    assert dr and all(float(r["mean_belief"]) == 0.0 for r in dr)          # C's template-1 rows (belief 10) dropped too
    assert not (t / "paired__R-C__all_injected__cloze_attributed__answer_plus_suffix.csv").exists()
    # legacy CSVs (schema 1, no meta): the common-item cohort is used and reported
    write_v2_csv(edir, "C", cloze_rows("C", inj, [0, 1], schema_version=1), None, pdir)
    write_v2_csv(edir, "R", cloze_rows("R", inj, [0], schema_version=1), None, pdir)
    out = run_cli(edir, pdir, *extra, compat="legacy")
    assert "COVERAGE DISAGREEMENT across arms (legacy CSVs)" in out
    rows = read_csv(t / "paired__R-C__all_injected__cloze_neutral__answer_plus_suffix.csv")
    assert rows and all(float(r["mean_diff"]) == 0.0 for r in rows)
    co = json.loads((edir / "analysis_v2" / "cohorts" / "all_injected.json").read_text())
    assert co["missingness"]["rep0"]["R"]["cloze_neutral"]["n_partial_templates"] == len(inj)
    # the per-fact helper itself: the common item set restricts both arms to template 0
    arms = AB.load_replicate(str(edir))
    ci = AB.common_items(arms)[("injected", "cloze_neutral")]
    assert ci["n_common"] == len(inj) and ci["n_items"] == {"C": 2 * len(inj), "R": len(inj)}
    assert all(v == {0} for v in ci["common"].values())
    pf = AB.per_fact(arms["C"], "injected", "cloze_neutral", "answer_plus_suffix", items=ci["common"])
    assert all(v["belief"] == 0.0 and v["n_templates"] == 1 for v in pf.values())
    assert all(v["belief"] == 5.0 for v in AB.per_fact(arms["C"], "injected", "cloze_neutral", "answer_plus_suffix").values())


def heldout_reversal_rows(arm, inj, held):
    """review_heldout_cancel.py: the actual rendered cloze items of every record; arm C prefers candidate a of every
    group (belief +2), arm R prefers candidate b (belief -20): every group's preference flips."""
    recs = {r["fact_id"]: r for r in inj + held}
    rows = []
    for ps, records in (("injected", inj), ("heldout", held)):
        for r in records:
            items, excl = PR.build_cloze_items(r, ps, [("cloze_neutral", "")])
            assert not excl
            for it in items:
                slot_a = recs[it.fact_id]["candidate_slot"] == "a"
                lpa, lpb = (-10.0, -12.0) if arm == "C" else (-24.0, -4.0)
                lv, lc = (lpa, lpb) if slot_a else (lpb, lpa)
                rows.append({"arm": arm, "probe_set": ps, "fact_id": it.fact_id, "pass": it.pass_name,
                             "template_idx": it.template_idx, "truth_value": r["truth_value"], "entity_tier": r["entity_tier"],
                             "assigned_frequency": r["assigned_frequency"], "logp_value": lv, "logp_competing": lc,
                             "schema_version": 2, "metric": it.metric, "template_id": it.template_id,
                             "scoring_policy": PR.SCORING_POLICY, "context": it.context, "cand_value": it.cand_value,
                             "cand_competing": it.cand_competing, "value_span_chars": json.dumps([list(x) for x in it.answer_spans_value]),
                             "boundary_k": 3, "n_ctx_tokens": 3})
    return rows


def test_heldout_orientation_prevents_cancellation(tmp_path):
    """Finding 8 (review_heldout_cancel.py): every wholly held-out group reverses its preference (+2 -> -20), yet the
    two-record cohort reported mean 0, CI [0, 0] and TOST equivalence at margin 0.1."""
    pdir, edir, inj, held = build_v2_fixture(tmp_path, n_groups=120, doses=(1, 4))
    for a in "CR":
        write_v2_csv(edir, a, heldout_reversal_rows(a, inj, held), {}, pdir)
    out = run_cli(edir, pdir, "--cohort", "heldout_groups", "heldout_groups_both", "--bootstrap", "300",
                  "--equivalence-margin", "0.1", compat=None)
    t = edir / "analysis_v2" / "tables"
    oriented = {r["label"]: r for r in read_csv(t / "paired__R-C__heldout_groups__cloze_neutral__answer_plus_suffix.csv")}
    both = {r["label"]: r for r in read_csv(t / "paired__R-C__heldout_groups_both__cloze_neutral__answer_plus_suffix.csv")}
    co = json.loads((edir / "analysis_v2" / "cohorts" / "heldout_groups.json").read_text())
    co_both = json.loads((edir / "analysis_v2" / "cohorts" / "heldout_groups_both.json").read_text())
    n_groups = co["n_groups"]
    assert n_groups >= 10 and co["n"] == n_groups and co_both["n"] == 2 * n_groups
    by_id = CH.load_probes(str(pdir)).by_id()
    for gid, o in co["details"]["orientation"]["groups"].items():
        kept = by_id[o["kept"]]
        if kept["ground_truth"] == "real_world":
            assert o["rule"] == "truth_true" and kept["truth_value"] == "true"
        else:
            assert o["rule"] == "slot_a" and kept["candidate_slot"] == "a"
    pooled = oriented.get("all", oriented["unknown"])
    assert int(pooled["n"]) == int(pooled["n_groups"]) == n_groups
    # synthetic groups keep slot a (+2 -> -20: diff -22); real groups keep the true candidate, which is slot b for
    # some of them (-2 -> +20: diff +22) -- the pooled signed mean is non-zero and every group's |delta| is 22
    assert abs(float(pooled["mean_diff"])) > 1.0 and int(pooled["n_pref_flips"]) == n_groups     # counted once per group
    assert pooled["tost_equivalent"] == "False"
    assert (float(pooled["ci95_lo"]), float(pooled["ci95_hi"])) != (0.0, 0.0)
    assert float(pooled["mean_abs_diff"]) == pytest.approx(22.0)
    unknown = oriented["unknown"]
    assert float(unknown["mean_diff"]) == pytest.approx(-22.0) and unknown["tost_equivalent"] == "False"
    assert unknown["p_wilcoxon_level"] == "fact" and float(unknown["p_wilcoxon"]) < 1e-3
    # the raw two-record view cancels by construction, and its definition says so
    rawp = both.get("all", both["unknown"])
    assert float(rawp["mean_diff"]) == 0.0 and int(rawp["n"]) == 2 * n_groups and rawp["tost_equivalent"] == "True"
    assert "BY CONSTRUCTION" in co_both["definition"] and "BY CONSTRUCTION" in out
    base = {(r["arm"], r["label"]): r for r in read_csv(t / "heldout_baseline__heldout_groups__cloze_neutral__answer_plus_suffix.csv")}
    assert float(base[("C", "unknown")]["mean_belief"]) == pytest.approx(2.0)
    assert float(base[("R", "unknown")]["mean_belief"]) == pytest.approx(-20.0)
    assert "TOST with margin 0.1" in out


def test_fixed_form_reconstruction_matches_renderer(tmp_path):
    """Finding 17 (review_eval_v2.py): without a manifest the memorization join must reconstruct the rendered form
    with the renderer's own index rule (seed + fact id) for every paraphrase policy; the old 'fixed index 0'
    fallback got 331 of 419 forms and 89 subgroups wrong."""
    pdir, edir, inj, held = build_v2_fixture(tmp_path, n_groups=40, doses=(1, 4, 16))
    assert len(inj) >= 20 and max(int(r["assigned_frequency"]) for r in inj) == 16
    policies = {"fixed": TP.RenderPolicy.fixed_rendering(), "seeded_start": TP.RenderPolicy(),
                "random": TP.RenderPolicy(paraphrase="random"), "cycle_from_zero": TP.RenderPolicy(paraphrase="cycle_from_zero")}
    for name, pol in policies.items():
        cfg = {"protocol": pol.protocol, "seed": 1234, "render_policy": pol.to_dict()}
        for r in inj:
            dose = int(r["assigned_frequency"])
            actual = [r["surface_forms"][TP.render_inserts(PR.FactView(r), occ, 1234, pol, arms=("R",))["R"].paraphrase_index]
                      for occ in range(dose)]
            nominal = CH.realized_training_forms(r, cfg)
            assert nominal == actual, (name, r["fact_id"])
            assert CH.is_verbatim_trained(r, actual) == CH.is_verbatim_trained(r, nominal)
        if name != "cycle_from_zero":
            with pytest.raises(ProtocolError, match="never assumed to be index 0"):
                CH.realized_training_forms(inj[0], {"protocol": pol.protocol, "render_policy": pol.to_dict()})
    # the CLI path: build summary -> protocol_cfg -> classify_verbatim_trained agrees with the renderer
    probes = CH.load_probes(str(pdir))
    cfg = CH.protocol_cfg_from_summary(json.loads((pdir.parent / "build_summary.json").read_text()), probes)
    assert cfg["seed"] == 1234 and cfg["render_policy"]["paraphrase"] == "fixed"
    cls = CH.classify_verbatim_trained(probes.injected, cfg)["nominal"]
    pol = TP.RenderPolicy.fixed_rendering()
    for r in probes.injected:
        actual = r["surface_forms"][TP.render_inserts(PR.FactView(r), 0, 1234, pol, arms=("R",))["R"].paraphrase_index]
        assert cls[r["fact_id"]] == CH.is_verbatim_trained(r, [actual])
    # a manifest, when supplied, still wins over the policy rule
    man = [{"arm": "R", "occurrence": o, "paraphrase_index": 0} for o in range(int(inj[0]["assigned_frequency"]))]
    assert CH.realized_training_forms(inj[0], cfg, manifest_rows=man) == [inj[0]["surface_forms"][0]] * int(inj[0]["assigned_frequency"])


def test_paraphrase_wellformed_is_csv_aware(synthetic):
    """paraphrase_wellformed follows the evaluation being analysed: legacy CSVs -> the historical reference (the two
    malformed legacy contrasts excluded); eval-schema-v2 CSVs -> every fact with a paraphrase row (the corrected
    evaluator renders both candidates structurally); paraphrase_historical_reference never changes."""
    root, pdir, probes, truth = synthetic
    all_ids = {r["fact_id"] for r in probes.injected}
    d = root / "eval_para_legacy"
    write_arm_csvs(d, probes, lambda a, f, p: 0.0, arms=("C", "R"))
    run_cli(d, pdir, "--bootstrap", "0", "--cohort", "paraphrase_wellformed", "paraphrase_historical_reference")
    wf = json.loads((d / "analysis_v2" / "cohorts" / "paraphrase_wellformed.json").read_text())
    hist = json.loads((d / "analysis_v2" / "cohorts" / "paraphrase_historical_reference.json").read_text())
    assert set(wf["fact_ids"]) == set(hist["fact_ids"]) == all_ids - set(truth["malformed"])
    assert wf["details"]["basis"].startswith("legacy_eval") and wf["details"]["eval_schema"] == 1
    d2 = root / "eval_para_v2"
    write_arm_csvs(d2, probes, lambda a, f, p: 0.0, arms=("C", "R"), v2=True, meta=True, probe_dir=pdir)
    run_cli(d2, pdir, "--bootstrap", "0", "--cohort", "paraphrase_wellformed", "paraphrase_historical_reference", compat=None)
    wf2 = json.loads((d2 / "analysis_v2" / "cohorts" / "paraphrase_wellformed.json").read_text())
    hist2 = json.loads((d2 / "analysis_v2" / "cohorts" / "paraphrase_historical_reference.json").read_text())
    assert set(wf2["fact_ids"]) == all_ids and set(hist2["fact_ids"]) == set(hist["fact_ids"])
    assert wf2["details"]["basis"].startswith("eval_schema_v2") and wf2["details"]["n_historical_malformed_included"] == 2
    assert set(wf2["details"]["historical_reference_malformed_fact_ids"]) == set(truth["malformed"])
    t = d2 / "analysis_v2" / "tables"
    pooled = {r["arm"]: int(r["n"]) for r in read_csv(t / "paraphrase__paraphrase_wellformed__heldout_paraphrase__full_sentence.csv")
              if r["label"] == "all" and r["dose"] == "all"}
    assert pooled == {"C": len(all_ids), "R": len(all_ids)}
    pooled_h = {r["arm"]: int(r["n"]) for r in read_csv(t / "paraphrase__paraphrase_historical_reference__heldout_paraphrase__full_sentence.csv")
                if r["label"] == "all" and r["dose"] == "all"}
    assert pooled_h == {"C": len(all_ids) - 2, "R": len(all_ids) - 2}
    assert (t / "paired__R-C__paraphrase_historical_reference__heldout_paraphrase__full_sentence.csv").exists()
    # a fact without a paraphrase row in the v2 CSVs is not well formed in THIS evaluation
    victim = probes.injected[0]["fact_id"]
    for arm in ("C", "R"):
        p = d2 / f"beliefs_arm{arm}.csv"
        rows = [r for r in read_csv(p) if not (r["fact_id"] == victim and r["pass"] == "heldout_paraphrase")]
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        write_meta(p, arm, pdir)
    run_cli(d2, pdir, "--bootstrap", "0", "--cohort", "paraphrase_wellformed", compat=None)
    wf3 = json.loads((d2 / "analysis_v2" / "cohorts" / "paraphrase_wellformed.json").read_text())
    assert set(wf3["fact_ids"]) == all_ids - {victim}
    assert any(e["fact_id"] == victim and e["reason"].startswith("no_paraphrase_row_in_eval") for e in wf3["exclusions"])


# ======================================================================================
# regression tests built from the 2026-10-14 RECHECK reproductions
# (reviews/2026-10-14-recheck/repro/review_third_eval_contracts.py, review_third_mixed_evaluator_versions.py):
# findings 1 (cross-replicate contract), 2 (analysis inputs bound to the evaluations), 3 (scored-item identity,
# initial-weight identity and matched horizon)

def _two_replicates(root, inj, pdir, modes, deltas=(2, 20), seeds=(7, 8)):
    reps = root / "reps"
    for name, seed, mode, delta in zip(("seed_7", "seed_8"), seeds, modes, deltas):
        d = reps / name
        d.mkdir(parents=True, exist_ok=True)
        for a in "CR":
            rows = cloze_rows(a, inj, [0, 1], belief=lambda ti, a=a, d=delta: d if a == "R" else 0)
            write_v2_csv(d, a, rows, {"init_seed": seed, "mode": mode}, pdir)
    _clear_outputs(reps)
    return reps


def test_recheck1_cross_replicate_analysis_never_pools_mock_and_real(tmp_path):
    """Recheck finding 1: seed_7 = model inference (R-C = 2), seed_8 = dry-run mock scores (R-C = 20) -- the analysis
    reported ok=True, mock=False and a pooled difference of 11 without any mock marker. Now refused; two mock replicates
    still run and every replicate table carries mock_scores = MOCK; two real replicates summarize to 11."""
    pdir, edir, inj, held = build_v2_fixture(tmp_path / "mix")
    extra = ("--cohort", "all_injected", "--bootstrap", "0")
    reps = _two_replicates(tmp_path / "mix", inj, pdir, ("model_inference", "dry_run_mock_scores"))
    out = run_cli(reps, pdir, *extra, compat=None, expect_fail=True)
    assert "replicates mix evaluation modes" in out and "never pooled" in out
    assert not (reps / "analysis_v2" / "tables" / "replicates__paired.csv").exists()
    compat = json.loads((reps / "analysis_v2" / "compatibility.json").read_text())
    assert compat["ok"] is False and compat["mock"] is False and compat["modes"] == ["dry_run_mock_scores", "model_inference"]
    reps = _two_replicates(tmp_path / "mix", inj, pdir, ("dry_run_mock_scores", "dry_run_mock_scores"))
    out = run_cli(reps, pdir, *extra, compat=None)
    assert "MOCK SCORES" in out
    rows = read_csv(reps / "analysis_v2" / "tables" / "replicates__paired.csv")
    assert rows and all(r["mock_scores"] == "MOCK" for r in rows)
    assert json.loads((reps / "analysis_v2" / "compatibility.json").read_text())["mock"] is True
    reps = _two_replicates(tmp_path / "mix", inj, pdir, ("model_inference", "model_inference"))
    out = run_cli(reps, pdir, *extra, compat=None)
    rows = read_csv(reps / "analysis_v2" / "tables" / "replicates__paired.csv")
    pooled = [r for r in rows if r["contrast"] == "R-C" and r["dose"] == "all" and r["label"] == "all"]
    assert float(pooled[0]["mean_of_mean_diff"]) == 11.0 and pooled[0]["n_replicates"] == "2" and "mock_scores" not in rows[0]
    # the other contract keys are experiment-wide too: a tokenizer / scoring policy / construction version that differs
    # between replicates (consistent within each) is refused
    for key, value in (("tokenizer", "b" * 64), ("scoring_policy", "joint_prefix_v3"), ("probe_construction", "structured_v2.0")):
        for a in "CR":
            rows_ = cloze_rows(a, inj, [0, 1], belief=lambda ti, a=a: 20 if a == "R" else 0)
            write_v2_csv(reps / "seed_8", a, rows_, {"init_seed": 8, key: value}, pdir)
        _clear_outputs(reps)
        out = run_cli(reps, pdir, *extra, compat=None, expect_fail=True)
        needle = "tokenizer_sha256 differs across the analysis" if key == "tokenizer" else f"{key} differs across the analysis"
        assert needle in out, (key, out[-1500:])


def test_recheck2_analysis_inputs_are_bound_to_the_evaluations(tmp_path):
    """Recheck finding 2: valid CSVs and sidecars, but the analysis selects probe files with the same fact ids and the
    assigned doses rewritten 1 -> 256: compatibility reported ok and the dose tables said 256. Now the selected probe
    files, build summary, manifest and exposure identities are part of the contract."""
    import shutil
    pdir, edir, inj, held = build_v2_fixture(tmp_path / "probes")
    extra = ("--cohort", "all_injected", "--bootstrap", "0")
    for a in "CR":
        write_v2_csv(edir, a, cloze_rows(a, inj, [0, 1]), {}, pdir)
    foreign = tmp_path / "foreign" / "probe_sets"
    foreign.mkdir(parents=True)
    (foreign / "injected_facts.jsonl").write_text("".join(json.dumps(dict(r, assigned_frequency=256)) + "\n" for r in inj))
    shutil.copy(pdir / "heldout_facts.jsonl", foreign / "heldout_facts.jsonl")
    shutil.copy(pdir.parent / "build_summary.json", foreign.parent / "build_summary.json")
    out = run_cli(edir, foreign, *extra, compat=None, expect_fail=True)
    assert "probe_sets/injected_facts.jsonl scored by the evaluation" in out and "is not the file this analysis builds cohorts/doses/labels from" in out
    assert not list((edir / "analysis_v2" / "tables").glob("dose_response__*"))
    compat = json.loads((edir / "analysis_v2" / "compatibility.json").read_text())
    assert compat["inputs"]["probe_hashes"]["probe_sets/injected_facts.jsonl"] == sha256_file(foreign / "injected_facts.jsonl")
    assert compat["replicates"]["rep0"]["C"]["inputs_bound"] is False
    # an intentional historical remapping is declared and recorded, never silent
    reason = "regenerated probe files of the same build (ids unchanged)"
    _clear_outputs(edir)
    out = run_cli(edir, foreign, *extra, "--compat", "historical-builds", "--compat-reason", reason, compat=None)
    compat = json.loads((edir / "analysis_v2" / "compatibility.json").read_text())
    assert compat["ok"] and any("is not the file this analysis builds" in w and reason in w for w in compat["warnings"])
    # the selected inputs pass when they are the evaluated ones, and are listed in the report
    _clear_outputs(edir)
    out = run_cli(edir, pdir, *extra, compat=None)
    assert "Analysis inputs every evaluation is bound to" in out
    compat = json.loads((edir / "analysis_v2" / "compatibility.json").read_text())
    assert compat["ok"] and compat["replicates"]["rep0"]["C"]["inputs_bound"] is True
    # a different build summary (protocol settings) is caught through the evaluation's hash AND the checkpoint bindings
    summary = pdir.parent / "build_summary.json"
    original = summary.read_text()
    summary.write_text(json.dumps({**json.loads(original), "config": {"seed": 1234, "render_policy": TP.RenderPolicy().to_dict()}}))
    try:
        _clear_outputs(edir)
        out = run_cli(edir, pdir, *extra, compat=None, expect_fail=True)
        assert "build_summary.json of the evaluation" in out and "protocol settings / render policy disagree" in out
    finally:
        summary.write_text(original)
    for a in "CR":
        write_v2_csv(edir, a, cloze_rows(a, inj, [0, 1]), {"bindings": {"build_summary_sha256": "9" * 64,
                                                                        "probe_hashes": {"injected_facts.jsonl": "8" * 64}}}, pdir)
    _clear_outputs(edir)
    out = run_cli(edir, pdir, *extra, compat=None, expect_fail=True)
    assert "the checkpoint was trained on build summary 999999999999" in out and "trained with probe_sets/injected_facts.jsonl 888888888888" in out
    for a in "CR":
        write_v2_csv(edir, a, cloze_rows(a, inj, [0, 1]), {"bindings": {"build_summary_sha256": "9" * 64}, "build_mismatch_allowed": True}, pdir)
    _clear_outputs(edir)
    out = run_cli(edir, pdir, *extra, compat=None)
    assert any("declared: --allow-build-mismatch" in w for w in json.loads((edir / "analysis_v2" / "compatibility.json").read_text())["warnings"])
    # exposure inputs: the preflight's prepacked dataset ids must be the checkpoints' (memorization classifications)
    exposure = tmp_path / "probes" / "exposure"
    exposure.mkdir()
    ids = {"C": "c" * 64, "R": "d" * 64}
    (exposure / "PREFLIGHT_OK.json").write_text(json.dumps({"prepack_dataset_id": ids, "build_summary_sha256": sha256_file(summary)}))
    for a in "CR":
        write_v2_csv(edir, a, cloze_rows(a, inj, [0, 1]), {"bindings": {"prepack_dataset_id": ids[a]}}, pdir)
    _clear_outputs(edir)
    run_cli(edir, pdir, *extra, "--exposure-dir", str(exposure), compat=None)
    assert json.loads((edir / "analysis_v2" / "compatibility.json").read_text())["replicates"]["rep0"]["R"]["inputs_bound"] is True
    write_v2_csv(edir, "R", cloze_rows("R", inj, [0, 1]), {"bindings": {"prepack_dataset_id": "e" * 64}}, pdir)
    _clear_outputs(edir)
    out = run_cli(edir, pdir, *extra, "--exposure-dir", str(exposure), compat=None, expect_fail=True)
    assert "the exposure index" in out and "but the checkpoint was trained on eeeeeeeeeeee" in out
    (exposure / "PREFLIGHT_OK.json").write_text(json.dumps({"prepack_dataset_id": ids, "build_summary_sha256": "0" * 64}))
    _clear_outputs(edir)
    out = run_cli(edir, pdir, *extra, "--exposure-dir", str(exposure), compat=None, expect_fail=True)
    assert "was measured for build summary 000000000000" in out
    (exposure / "PREFLIGHT_OK.json").unlink()
    _clear_outputs(edir)
    out = run_cli(edir, pdir, *extra, "--exposure-dir", str(exposure), compat=None)
    assert "exposure inputs are unbound" in out
    # the manifest must be the build's own (BUILD_COMPLETE.json)
    manifest = tmp_path / "probes" / "manifest.csv"
    manifest.write_text("arm,fact_id,occurrence,paraphrase_index\n")
    (tmp_path / "probes" / "BUILD_COMPLETE.json").write_text(json.dumps({"manifest_sha256": "1" * 64}))
    _clear_outputs(edir)
    out = run_cli(edir, pdir, *extra, "--manifest", str(manifest), compat=None, expect_fail=True)
    assert "is not the manifest of the build" in out
    (tmp_path / "probes" / "BUILD_COMPLETE.json").write_text(json.dumps({"manifest_sha256": sha256_file(manifest)}))
    _clear_outputs(edir)
    out = run_cli(edir, pdir, *extra, "--manifest", str(manifest), compat=None)
    assert "(= BUILD_COMPLETE.json)" in out


def test_recheck3_scored_item_identity_init_identity_and_horizon(tmp_path):
    """Recheck finding 3: (a) the same template ids with different scored texts passed with no warning; (b) constructed
    checkpoints at steps 100 and 900 with different initial-parameter hashes but equal integer seeds passed too."""
    pdir, edir, inj, held = build_v2_fixture(tmp_path / "prompts")
    extra = ("--cohort", "all_injected", "--bootstrap", "0")
    rows = {a: cloze_rows(a, inj, [0, 1]) for a in "CR"}
    bad = [dict(r, context="This is certainly false: ", cand_value=" Rome.", cand_competing=" Paris.") for r in rows["R"]]
    write_v2_csv(edir, "C", rows["C"], {}, pdir)
    write_v2_csv(edir, "R", bad, {}, pdir)
    out = run_cli(edir, pdir, *extra, compat=None, expect_fail=True)
    assert f"{len(bad)} scored item(s) have different context/candidate texts or spans" in out and "Rome" in out
    compat = json.loads((edir / "analysis_v2" / "compatibility.json").read_text())
    assert compat["n_item_identity_mismatches"] == len(bad) and compat["n_items_with_identity"] == len(bad)
    # the same texts under a different scoring boundary are a different scored item too
    write_v2_csv(edir, "R", [dict(r, boundary_k=int(r["boundary_k"]) - 1) for r in rows["R"]], {}, pdir)
    _clear_outputs(edir)
    out = run_cli(edir, pdir, *extra, compat=None, expect_fail=True)
    assert "have different context/candidate texts or spans" in out
    # identical items pass; a legacy CSV without the text columns is 'unverified' only under --compat legacy
    write_v2_csv(edir, "R", rows["R"], {}, pdir)
    _clear_outputs(edir)
    run_cli(edir, pdir, *extra, compat=None)
    legacy = [{k: v for k, v in r.items() if k in LEGACY_COLS} for r in rows["R"]]
    pr = edir / "beliefs_armR.csv"
    with pr.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LEGACY_COLS)
        w.writeheader()
        w.writerows(legacy)
    write_meta(pr, "R", pdir, protocol=PROTOCOL_V2)
    _clear_outputs(edir)
    out = run_cli(edir, pdir, *extra, compat=None, expect_fail=True)
    assert "scored-item identities cannot be compared" in out
    # (b) initial weights and horizon: equal integer seeds, different init_param_sha256, steps 100 vs 900
    write_v2_csv(edir, "C", rows["C"], {"step": 100, "init_param_sha256": "C" * 64}, pdir)
    write_v2_csv(edir, "R", rows["R"], {"step": 900, "init_param_sha256": "R" * 64}, pdir)
    _clear_outputs(edir)
    out = run_cli(edir, pdir, *extra, compat=None, expect_fail=True)
    assert "checkpoint.init_param_sha256 differs across arms" in out and "checkpoint.step differs across the analysis" in out
    assert "--allow-unequal-horizon" in out
    # a declared unequal-budget experiment: accepted and recorded; the init identity is still enforced
    out = run_cli(edir, pdir, *extra, "--allow-unequal-horizon", "pilot: R trained 9x longer", compat=None, expect_fail=True)
    assert "init_param_sha256 differs" in out and "checkpoint.step differs" not in out.split("EVALUATION METADATA INCOMPATIBLE")[-1]
    write_v2_csv(edir, "R", rows["R"], {"step": 900, "init_param_sha256": "C" * 64, "total_batch_size": 384}, pdir)
    _clear_outputs(edir)
    out = run_cli(edir, pdir, *extra, "--allow-unequal-horizon", "pilot: R trained 9x longer", compat=None)
    compat = json.loads((edir / "analysis_v2" / "compatibility.json").read_text())
    assert compat["ok"] and compat["unequal_horizon_reason"] == "pilot: R trained 9x longer"
    assert sum("declared unequal-budget experiment" in w for w in compat["warnings"]) == 2    # step + total_batch_size
    assert "declared unequal-budget experiment | pilot: R trained 9x longer" in out
    # missing required provenance: refused in strict mode, reported as unverified under --compat legacy
    write_v2_csv(edir, "C", rows["C"], {"init_param_sha256": None, "probe_construction": None}, pdir)
    write_v2_csv(edir, "R", rows["R"], {"init_param_sha256": None, "probe_construction": None}, pdir)
    _clear_outputs(edir)
    out = run_cli(edir, pdir, *extra, compat=None, expect_fail=True)
    assert "checkpoint.init_param_sha256 missing: required provenance" in out and "records no probe_construction version" in out
    out = run_cli(edir, pdir, *extra, compat="legacy")
    compat = json.loads((edir / "analysis_v2" / "compatibility.json").read_text())
    assert compat["ok"] and any("init_param_sha256 missing" in w and "unverified" in w for w in compat["warnings"])
    # across replicates the same arm must start from DIFFERENT initial parameters
    reps = _two_replicates(tmp_path / "prompts", inj, pdir, ("model_inference", "model_inference"))
    for a in "CR":
        write_v2_csv(reps / "seed_8", a, cloze_rows(a, inj, [0, 1]), {"init_seed": 8, "init_param_sha256": hashlib.sha256(b"init:7").hexdigest()}, pdir)
    _clear_outputs(reps)
    out = run_cli(reps, pdir, *extra, compat=None, expect_fail=True)
    assert "share checkpoint.init_param_sha256" in out


def test_recheck3_mixed_evaluator_versions_are_refused(tmp_path):
    """Recheck finding 3 (review_third_mixed_evaluator_versions.py): arm C scored by the committed 031c4e5 probe
    constructor, arm R by the repaired one, both declaring schema 2 / joint_prefix_v2 on the same 600-group build
    (15 actual context/candidate differences): the analysis accepted them with no warning."""
    import contextlib
    import importlib.util
    import io
    try:
        old_src = subprocess.check_output(["git", "show", "031c4e5:contextualization/ctx/probes.py"], cwd=str(REPO_ROOT),
                                          stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, OSError):
        pytest.skip("commit 031c4e5 not available in this checkout")
    import eval_beliefs as EV
    pdir, edir, inj, held = build_v2_fixture(tmp_path, n_groups=600, doses=(1, 4))
    summary = tmp_path / "build_summary.json"
    s = json.loads(summary.read_text())
    s["config"]["render_policy"] = TP.RenderPolicy().to_dict()
    summary.write_text(json.dumps(s))
    oldfile = tmp_path / "oldprobes.py"
    oldfile.write_bytes(old_src)
    spec = importlib.util.spec_from_file_location("ctx.old_review_probes", oldfile)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    current = EV.P
    try:
        for a, module in (("C", old), ("R", current)):
            EV.P = module
            with contextlib.redirect_stdout(io.StringIO()):
                EV.main(["--exp-dir", str(tmp_path), "--arm", a, "--dry-run", "--no-tracking"])
    finally:
        EV.P = current
    rows_c, rows_r = read_csv(edir / "beliefs_armC.csv"), read_csv(edir / "beliefs_armR.csv")
    mism = [(c, r) for c, r in zip(rows_c, rows_r)
            if c["cand_competing"] != r["cand_competing"] or c["cand_value"] != r["cand_value"] or c["context"] != r["context"]]
    assert len(mism) == 15
    metas = {a: json.loads((edir / f"beliefs_arm{a}.meta.json").read_text()) for a in "CR"}
    assert metas["R"]["probe_construction"] == PR.PROBE_CONSTRUCTION_VERSION and metas["C"]["probe_construction"] != metas["R"]["probe_construction"]
    assert metas["C"]["items_sha256"] != metas["R"]["items_sha256"]
    out = run_cli(edir, pdir, "--cohort", "all_injected", "--bootstrap", "0", compat=None, expect_fail=True)
    assert "15 scored item(s) have different context/candidate texts or spans" in out
    assert "probe_construction differs across the analysis" in out
    assert not list((edir / "analysis_v2" / "tables").glob("paired__*"))


def test_training_objective_is_one_per_experiment(tmp_path):
    """Decision 2026-10-15: evaluations of checkpoints trained under different loss normalizations never pool; an
    unrecorded policy is the historical microbatch_mean and no flag relaxes the rule."""
    pdir, edir, inj, held = build_v2_fixture(tmp_path / "obj")
    extra = ("--cohort", "all_injected", "--bootstrap", "0")
    rows = {a: cloze_rows(a, inj, [0, 1]) for a in "CR"}
    write_v2_csv(edir, "C", rows["C"], {"checkpoint": {"loss_normalization": "fixed_nominal"}}, pdir)
    write_v2_csv(edir, "R", rows["R"], {}, pdir)
    out = run_cli(edir, pdir, *extra, compat=None, expect_fail=True)
    assert "loss_normalization (training objective) differs" in out and "microbatch_mean (unrecorded: historical default)" in out
    out = run_cli(edir, pdir, *extra, "--allow-unequal-horizon", "declared", compat=None, expect_fail=True)
    assert "loss_normalization (training objective) differs" in out
    write_v2_csv(edir, "R", rows["R"], {"checkpoint": {"loss_normalization": "fixed_nominal"}}, pdir)
    _clear_outputs(edir)
    out = run_cli(edir, pdir, *extra, compat=None)
    crows = {r["arm"]: r for r in read_csv(edir / "analysis_v2" / "tables" / "compatibility.csv")}
    assert crows["C"]["loss_normalization"] == crows["R"]["loss_normalization"] == "fixed_nominal"
    # two historical (unrecorded) evaluations still pool with each other, labelled as unrecorded
    for a in "CR":
        write_v2_csv(edir, a, rows[a], {}, pdir)
    _clear_outputs(edir)
    run_cli(edir, pdir, *extra, compat=None)
    crows = {r["arm"]: r for r in read_csv(edir / "analysis_v2" / "tables" / "compatibility.csv")}
    assert crows["C"]["loss_normalization"] == "unrecorded"
