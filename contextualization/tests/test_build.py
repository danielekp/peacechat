"""Builder tests (ctx.blend.build) on small synthetic datasets. Offline, ~2 s per build.

Run from the repository root:
    $HOME/.cache/ctx_cpu_venv/bin/python -m pytest contextualization/tests/test_build.py -q

Builds go through build_dataset.parse_args(...) so every test sees the CLI defaults; base shards
are generated once per session into a shared cache directory (keyed by their configuration).
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import pytest

import build_dataset
from ctx import blend
from ctx.nanochat_io import read_shard_texts
from ctx.protocol import PROTOCOL_V1, PROTOCOL_V2, sha256_file

V2_ARGV = ["--protocol", "v2", "--base-source", "synthetic", "--num-groups", "80", "--freq-grid", "1", "4", "16",
           "--base-chars", "2_000_000", "--chars-per-shard", "300_000", "--seed", "0", "--counterfact-limit", "0",
           "--no-tracking"]
V1_ARGV = ["--protocol", "v1", "--base-source", "synthetic", "--num-facts", "120", "--freq-grid", "1", "4",
           "--base-chars", "2_000_000", "--chars-per-shard", "300_000", "--seed", "0", "--counterfact-limit", "0",
           "--no-tracking"]
# the kw dict of reviews/2026-10-13/reproduce_dataset_findings.py (a pre-v2 Namespace: no protocol,
# num_groups, design, carrier_max_chars, render_mode, policy_*, collision_audit, overwrite, ...)
LEGACY_KW = dict(seed=166, base_source='synthetic', base_chars=1000,
                 num_facts=2, heldout_frac=0, contested_frac=0, freq_grid=[1], counterfact_limit=0,
                 embed=True, register=True, verbatim_control=False, source_per_fact=False,
                 tokenizer_src=None, chars_per_shard=1000, row_group_size=8, data_subdir='base_data',
                 synthetic_docs_per_shard=100, parity_tol=.0049, inject_frac=.02, probes_only=False)


# ---------------------------------------------------------------------------------------------
# helpers

def parse(out, cache, argv, extra=()):
    return build_dataset.parse_args([*argv, "--out", str(out), "--cache-dir", str(cache), *extra])


def build(out, cache, argv, extra=()):
    return blend.build(parse(out, cache, argv, extra))


def disk_hashes(out, arm):
    d = Path(out) / f"arm_{arm}" / "base_data"
    return {p.name: sha256_file(p) for p in sorted(d.glob("*.parquet"))}


def manifest_rows(out):
    with open(Path(out) / "manifest.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def summary_of(out):
    return json.loads((Path(out) / "build_summary.json").read_text())


def train_chars(out, arm):
    paths = sorted((Path(out) / f"arm_{arm}" / "base_data").glob("*.parquet"))
    return sum(len(t) for p in paths[:-1] for t in read_shard_texts(str(p)))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="session")
def base_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("base_cache")


@pytest.fixture(scope="module")
def v2_e(tmp_path_factory, base_cache):
    out = tmp_path_factory.mktemp("v2_e")
    s = build(out, base_cache, V2_ARGV, ["--embedding-control", "--collision-audit"])
    return out, s


@pytest.fixture(scope="module")
def v2_noe(tmp_path_factory, base_cache):
    out = tmp_path_factory.mktemp("v2_noe")
    s = build(out, base_cache, V2_ARGV, ["--collision-audit"])
    return out, s


@pytest.fixture(scope="module")
def v1_build(tmp_path_factory, base_cache):
    out = tmp_path_factory.mktemp("v1")
    s = build(out, base_cache, V1_ARGV)
    return out, s


# ---------------------------------------------------------------------------------------------

def test_embedding_control_never_changes_C_R_X(v2_e, v2_noe):
    """Finding 23: adding arm E must not change what C/R/X receive (shards and manifest rows)."""
    out_e, s_e = v2_e
    out_n, s_n = v2_noe
    assert sorted(p.name for p in out_e.glob("arm_*")) == ["arm_C", "arm_E", "arm_R", "arm_X"]
    assert sorted(p.name for p in out_n.glob("arm_*")) == ["arm_C", "arm_R", "arm_X"]
    for a in "CRX":
        assert s_e["shard_hashes"][a] == s_n["shard_hashes"][a], a
        assert disk_hashes(out_e, a) == disk_hashes(out_n, a) == s_e["shard_hashes"][a], a
        assert s_e["arm_total_chars"][a] == s_n["arm_total_chars"][a]
        assert s_e["topup_docs"][a] == s_n["topup_docs"][a]
    rows_e = [r for r in manifest_rows(out_e) if r["arm"] != "E"]
    rows_n = manifest_rows(out_n)
    assert rows_e == rows_n
    assert sum(r["arm"] == "E" for r in manifest_rows(out_e)) == s_e["n_occurrences_M"]
    # the top-up stream is the same stream; E simply takes none / some of its docs
    assert [d["chars"] for d in s_e["topup_stream_doc_chars"]][:len(s_n["topup_stream_doc_chars"])] == \
        [d["chars"] for d in s_n["topup_stream_doc_chars"]]


def test_byte_determinism(v2_e, tmp_path, base_cache):
    out1, s1 = v2_e
    out2 = tmp_path / "again"
    s2 = build(out2, base_cache, V2_ARGV, ["--embedding-control", "--collision-audit"])
    for a in "CRXE":
        assert disk_hashes(out1, a) == disk_hashes(out2, a)
    assert (out1 / "manifest.csv").read_bytes() == (out2 / "manifest.csv").read_bytes()
    for n in ("injected_facts.jsonl", "heldout_facts.jsonl"):
        assert (out1 / "probe_sets" / n).read_bytes() == (out2 / "probe_sets" / n).read_bytes()
    for k in ("shard_hashes", "arm_total_chars", "topup_stream_doc_chars", "counts", "xe_diagnostics", "novelty",
              "n_occurrences_M", "n_train_base_docs_T", "parity_ceiling", "injected_chars"):
        assert s1[k] == s2[k], k
    m1, m2 = json.loads((out1 / "BUILD_COMPLETE.json").read_text()), json.loads((out2 / "BUILD_COMPLETE.json").read_text())
    assert m1["manifest_sha256"] == m2["manifest_sha256"] and m1["probe_hashes"] == m2["probe_hashes"]


def test_refuses_to_overwrite_unless_asked(v2_noe, base_cache):
    out, s = v2_noe
    before = {a: disk_hashes(out, a) for a in "CRX"}
    with pytest.raises(SystemExit, match="refusing to build"):
        build(out, base_cache, V2_ARGV, ["--collision-audit"])
    assert {a: disk_hashes(out, a) for a in "CRX"} == before           # nothing was touched
    assert (out / "BUILD_COMPLETE.json").exists()
    args = parse(out, base_cache, V2_ARGV, ["--collision-audit"])
    args.overwrite = True
    s2 = blend.build(args)
    assert {a: disk_hashes(out, a) for a in "CRX"} == before           # rebuilt, byte-identical
    assert s2["shard_hashes"] == s["shard_hashes"]
    assert (out / "BUILD_COMPLETE.json").exists() and not (out / ".staging").exists()


def test_success_leaves_marker_and_no_staging(v2_e):
    out, s = v2_e
    assert not (out / ".staging").exists()
    assert not list(out.rglob("*.tmp"))
    marker = json.loads((out / "BUILD_COMPLETE.json").read_text())
    assert marker["protocol"] == PROTOCOL_V2 == s["protocol"]
    assert marker["build_summary_sha256"] == sha256_file(out / "build_summary.json")
    assert marker["manifest_sha256"] == sha256_file(out / "manifest.csv")
    for n, h in marker["probe_hashes"].items():
        assert h == sha256_file(out / "probe_sets" / n)
    for a in "CRXE":
        assert disk_hashes(out, a) == s["shard_hashes"][a]
    assert (out / "novelty_audit.json").exists() and (out / "xe_examples.txt").exists()


def test_failure_midway_leaves_only_staging(tmp_path, base_cache, monkeypatch):
    out = tmp_path / "broken"

    def boom(*a, **k):
        raise RuntimeError("simulated failure while writing probe files")
    monkeypatch.setattr(blend, "_write_probe_files", boom)
    with pytest.raises(RuntimeError, match="simulated failure"):
        build(out, base_cache, V2_ARGV, ["--embedding-control"])
    assert set(os.listdir(out)) == {".staging"}, os.listdir(out)
    staged = set(os.listdir(out / ".staging"))
    assert "arm_C" in staged and "manifest.csv" in staged
    assert not (out / "BUILD_COMPLETE.json").exists() and not (out / "build_summary.json").exists()
    # a later build into the same dir refuses because .staging is a conflict marker
    with pytest.raises(SystemExit, match="refusing to build"):
        build(out, base_cache, V2_ARGV, ["--embedding-control"])


def test_probes_only_verifies_and_regenerates(tmp_path, base_cache):
    out = tmp_path / "p"
    extra = ["--embedding-control"]
    s = build(out, base_cache, V2_ARGV, extra)
    hashes_before = {a: disk_hashes(out, a) for a in "CRXE"}
    manifest_before = (out / "manifest.csv").read_bytes()
    with pytest.raises(SystemExit, match="pool arguments differ"):
        blend.build(parse(out, base_cache, V2_ARGV[:-5] + ["--seed", "1", "--counterfact-limit", "0", "--no-tracking"],
                          extra + ["--probes-only"]))
    assert not (out / "probe_sets_regen").exists()
    with pytest.raises(SystemExit, match="not found"):
        blend.build(parse(tmp_path / "nothing", base_cache, V2_ARGV, extra + ["--probes-only"]))
    r = blend.build(parse(out, base_cache, V2_ARGV, extra + ["--probes-only"]))
    assert r["probes_only"] is True and r["verified_against_manifest"] is True
    regen = out / "probe_sets_regen"
    assert Path(r["probe_dir"]) == regen and (regen / "PROBES_META.json").exists()
    for n in ("injected_facts.jsonl", "heldout_facts.jsonl"):
        assert (regen / n).read_bytes() == (out / "probe_sets" / n).read_bytes(), n
    meta = json.loads((regen / "PROBES_META.json").read_text())
    assert meta["hashes"] == {n: sha256_file(regen / n) for n in ("injected_facts.jsonl", "heldout_facts.jsonl")}
    # shards, manifest, summary untouched
    assert {a: disk_hashes(out, a) for a in "CRXE"} == hashes_before
    assert (out / "manifest.csv").read_bytes() == manifest_before
    assert summary_of(out)["shard_hashes"] == s["shard_hashes"]


def test_probes_only_reproduces_novelty_labels(tmp_path, base_cache):
    out = tmp_path / "p"
    build(out, base_cache, V2_ARGV, ["--embedding-control", "--collision-audit"])
    blend.build(parse(out, base_cache, V2_ARGV, ["--embedding-control", "--collision-audit", "--probes-only"]))
    for n in ("injected_facts.jsonl", "heldout_facts.jsonl"):
        assert (out / "probe_sets_regen" / n).read_bytes() == (out / "probe_sets" / n).read_bytes(), n


def test_parity_rule(v2_e):
    """Builder v2 rule: ceiling = max(C,R,X); nobody exceeds it; everybody >= ceiling*(1-tol);
    E may sit above the ceiling by at most tol."""
    out, s = v2_e
    tol = s["config"]["parity_tol"]
    totals = {a: train_chars(out, a) for a in "CRXE"}
    assert totals == s["arm_total_chars"]
    ceiling = max(totals[a] for a in "CRX")
    assert ceiling == s["parity_ceiling"]
    for a in "CRX":
        assert totals[a] <= ceiling
    for a in "CRXE":
        assert totals[a] >= ceiling * (1 - tol), a
    assert totals["E"] <= ceiling * (1 + tol)
    spread = (ceiling - min(totals[a] for a in "CRX")) / ceiling
    assert spread <= tol
    # top-up docs: whole docs from the shared stream, appended only when they fit under the ceiling
    stream = s["topup_stream_doc_chars"]
    assert sum(len(d["arms"]) > 0 for d in stream) >= 1
    for a in "CRXE":
        assert sum(a in d["arms"] for d in stream) == s["topup_docs"][a]
    # val shard identical across arms
    val = {a: sorted((out / f"arm_{a}" / "base_data").glob("*.parquet"))[-1] for a in "CRXE"}
    assert len({sha256_file(p) for p in val.values()}) == 1


def test_topup_infeasible_fails_clearly(tmp_path, base_cache):
    """The review's two-fact, 1000-char configuration (finding 23): under builder v2 no filler doc fits
    the sub-tolerance gap, so the build stops with a clear message instead of overshooting."""
    kw = dict(LEGACY_KW, cache_dir=str(base_cache))
    with pytest.raises(SystemExit, match="parity top-up infeasible"):
        blend.build(argparse.Namespace(**kw, out=str(tmp_path / "noE"), embedding_control=False))
    with pytest.raises(SystemExit, match=r"arm E is \d+ chars, more than .* above the C/R/X ceiling"):
        blend.build(argparse.Namespace(**kw, out=str(tmp_path / "withE"), embedding_control=True))
    for d in ("noE", "withE"):
        assert not (tmp_path / d / "BUILD_COMPLETE.json").exists()
        assert not list((tmp_path / d).glob("arm_*"))


def test_carrier_cropping(tmp_path, base_cache):
    limit = 200
    out = tmp_path / "crop"
    s = build(out, base_cache, V2_ARGV, ["--carrier-max-chars", str(limit)])
    assert s["config"]["carrier_max_chars"] == limit
    rows = manifest_rows(out)
    assert rows and all(int(r["carrier_chars"]) <= limit for r in rows)
    assert any(r["carrier_cropped"] == "True" for r in rows)
    # the same crop for every arm of a slot, and the document really is carrier + insert
    by_slot = {}
    for r in rows:
        by_slot.setdefault(r["doc_index"], []).append(r)
    docs = {}
    for r in rows[:40]:
        key = (r["arm"], r["shard_file"])
        if key not in docs:
            docs[key] = list(read_shard_texts(str(out / f"arm_{r['arm']}" / "base_data" / r["shard_file"])))
        doc = docs[key][int(r["row_in_shard"])]
        assert len(doc) - len(r["insert_text"]) - 1 <= limit
    for slot in by_slot.values():
        assert len({(r["carrier_chars"], r["carrier_cropped"]) for r in slot}) == 1
    # without cropping the same slots carry longer carriers
    uncropped = build(tmp_path / "nocrop", base_cache, V2_ARGV, ["--carrier-max-chars", "0"])
    assert uncropped["config"]["carrier_max_chars"] is None
    assert all(r["carrier_cropped"] == "False" for r in manifest_rows(tmp_path / "nocrop"))


def test_counts_semantics(v2_e):
    out, s = v2_e
    c = s["counts"]
    assert c["groups"] == 80 == c["pairs"] and c["candidate_statements"] == 160
    assert c["training_groups"] + c["heldout_groups"] == 80
    assert c["injected_statements"] == c["training_groups"] == c["unasserted_rivals"]   # one_sided
    assert c["real_groups"] + c["novel_groups"] == 80
    inj = [json.loads(l) for l in open(out / "probe_sets" / "injected_facts.jsonl")]
    held = [json.loads(l) for l in open(out / "probe_sets" / "heldout_facts.jsonl")]
    assert len(inj) == c["injected_statements"] == s["n_injected_facts"]
    assert len(held) == c["unasserted_rivals"] + 2 * c["heldout_groups"] == s["n_heldout_facts"]
    assert s["probes"] == {"heldout_synthetic": len(held), "heldout_counterfact": 0, "injected": len(inj)}
    assert c["occurrences"] == sum(r["assigned_frequency"] for r in inj) == s["n_occurrences_M"]
    assert sum(r["arm"] == "R" for r in manifest_rows(out)) == s["n_occurrences_M"]
    assert s["n_facts"] == c["candidate_statements"]
    assert s["tier_counts"] == {"real": c["real_groups"], "novel": c["novel_groups"]}
    assert s["novelty"] == {"real_entity": c["real_groups"], "audited_absent": c["novel_groups"]}
    assert all(r["role"] == "injected" and r["arms"] == ["R", "X", "E"] for r in inj)
    assert s["schema_versions"] == {"probe": 2, "manifest": 2} and s["builder_version"] == 2


def test_v1_build_from_legacy_namespace(tmp_path, base_cache):
    """A pre-v2 Namespace (the review script's kw dict, sized to be feasible) still builds as protocol v1."""
    kw = dict(LEGACY_KW, cache_dir=str(base_cache), base_chars=200_000, chars_per_shard=50_000,
              synthetic_docs_per_shard=200, num_facts=40, heldout_frac=0.3, contested_frac=0.08, freq_grid=[1, 2])
    s = blend.build(argparse.Namespace(**kw, out=str(tmp_path / "noE"), embedding_control=False))
    assert s["protocol"] == PROTOCOL_V1 and s["schema_versions"]["probe"] == 1 and s["design"] == "legacy_independent_split"
    assert sorted(p.name for p in (tmp_path / "noE").glob("arm_*")) == ["arm_C", "arm_R", "arm_X"]
    assert (tmp_path / "noE" / "BUILD_COMPLETE.json").exists()
    assert s["config"]["render_policy"]["protocol"] == PROTOCOL_V1 and s["config"]["protocol"] == PROTOCOL_V1
    rec = json.loads((tmp_path / "noE" / "probe_sets" / "injected_facts.jsonl").read_text().splitlines()[0])
    assert rec["schema_version"] == 1 and rec["protocol"] == PROTOCOL_V1 and rec["arms"] == ["R", "X"]
    s_e = blend.build(argparse.Namespace(**kw, out=str(tmp_path / "withE"), embedding_control=True))
    assert sorted(p.name for p in (tmp_path / "withE").glob("arm_*")) == ["arm_C", "arm_E", "arm_R", "arm_X"]
    for a in "CRX":
        assert s_e["shard_hashes"][a] == s["shard_hashes"][a]


def test_v1_example_build(v1_build):
    out, s = v1_build
    assert s["protocol"] == PROTOCOL_V1 and s["counts"]["facts"] == 120
    assert s["counts"]["injected_statements"] == s["n_injected_facts"] and s["counts"]["occurrences"] == s["n_occurrences_M"]
    rows = manifest_rows(out)
    assert {r["arm"] for r in rows} == {"C", "R", "X"} and len(rows) == 3 * s["n_occurrences_M"]
    assert all(r["schema_version"] == "2" for r in rows)
