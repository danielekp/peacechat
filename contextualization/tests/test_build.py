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


# --- review 2026-10-14 (second review) findings 9, 10, 13, carrier fit, source pinning ------

def _argv2(out, cache, extra=()):
    return ["--out", str(out), "--base-source", "synthetic", "--num-groups", "80", "--freq-grid", "1", "4", "16",
            "--base-chars", "2000000", "--chars-per-shard", "300000", "--seed", "0", "--counterfact-limit", "0",
            "--no-tracking", "--cache-dir", str(cache), *extra]


def _records(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines()]


def test_probes_only_uses_the_original_render_policy(tmp_path):
    """Finding 9: a rotating-source build cannot be regenerated with --source-per-fact."""
    import build_dataset
    from ctx import blend
    out, cache = tmp_path / "p", tmp_path / "cache"
    blend.build(build_dataset.parse_args(_argv2(out, cache)))
    with pytest.raises(SystemExit, match="rendering overrides differ"):
        blend.build(build_dataset.parse_args(_argv2(out, cache, ["--probes-only", "--source-per-fact"])))
    with pytest.raises(SystemExit, match="rendering overrides differ"):
        blend.build(build_dataset.parse_args(_argv2(out, cache, ["--probes-only", "--render-mode", "fixed"])))
    with pytest.raises(SystemExit, match="embedding-control differs"):
        blend.build(build_dataset.parse_args(_argv2(out, cache, ["--probes-only", "--embedding-control"])))
    r = blend.build(build_dataset.parse_args(_argv2(out, cache, ["--probes-only"])))
    assert r["verified_against_manifest"]
    assert sum(bool(x["fixed_source"]) for x in _records(out / "probe_sets_regen" / "injected_facts.jsonl")) == 0
    assert (out / "probe_sets_regen" / "injected_facts.jsonl").read_bytes() == (out / "probe_sets" / "injected_facts.jsonl").read_bytes()


def test_overwrite_removes_derived_artifacts_and_foreign_audits_are_refused(tmp_path):
    """Finding 10: after --overwrite without an audit, no stale novelty labels can leak into
    regenerated probes; an audit stamped with another build id is refused."""
    import build_dataset
    from ctx import blend
    out, cache = tmp_path / "o", tmp_path / "cache"
    blend.build(build_dataset.parse_args(_argv2(out, cache, ["--collision-audit"])))
    (out / "validation_report.json").write_text("{}")
    (out / "exposure").mkdir()
    s = blend.build(build_dataset.parse_args(_argv2(out, cache, ["--overwrite"])))
    assert not (out / "novelty_audit.json").exists() and not (out / "validation_report.json").exists() and not (out / "exposure").exists()
    assert s["novelty_audit"] is None
    blend.build(build_dataset.parse_args(_argv2(out, cache, ["--probes-only"])))
    orig = _records(out / "probe_sets" / "heldout_facts.jsonl")
    regen = _records(out / "probe_sets_regen" / "heldout_facts.jsonl")
    assert sum(a["novelty"] != b["novelty"] for a, b in zip(orig, regen)) == 0
    assert all(x["novelty"] in ("unverified", "real_entity") for x in regen)
    out2 = tmp_path / "f"
    blend.build(build_dataset.parse_args(_argv2(out2, cache, ["--collision-audit"])))
    aud = json.loads((out2 / "novelty_audit.json").read_text())
    assert aud["build_id"] == json.loads((out2 / "build_summary.json").read_text())["build_id"]
    aud["build_id"] = "f" * 64
    (out2 / "novelty_audit.json").write_text(json.dumps(aud))
    with pytest.raises(SystemExit, match="belongs to another build"):
        blend.build(build_dataset.parse_args(_argv2(out2, cache, ["--probes-only"])))


def test_carrier_first_sentence_never_exceeds_the_budget(tmp_path):
    import build_dataset
    from ctx import blend
    out, cache = tmp_path / "c", tmp_path / "cache"
    s = blend.build(build_dataset.parse_args(_argv2(out, cache, ["--carrier-max-chars", "150"])))
    with open(out / "manifest.csv") as fh:
        mx = max(int(r["carrier_chars"]) for r in csv.DictReader(fh))
    assert mx <= 150 and "carriers_skipped_too_long" in s
    assert blend.crop_carrier("A" * 500 + ". Short one.", 100) == (None, True)
    assert blend.crop_carrier("Short one. " + "B" * 500 + ".", 100) == ("Short one.", True)


def test_fixed_mode_build_repeats_one_insert_per_fact_in_every_arm(tmp_path):
    import build_dataset
    from ctx import blend
    out, cache = tmp_path / "x", tmp_path / "cache"
    s = blend.build(build_dataset.parse_args(_argv2(out, cache, ["--render-mode", "fixed", "--embedding-control"])))
    per = {}
    with open(out / "manifest.csv") as fh:
        for r in csv.DictReader(fh):
            per.setdefault((r["fact_id"], r["arm"]), set()).add(r["insert_text"])
    assert max(len(v) for v in per.values()) == 1
    assert s["xe_diagnostics"]["token_limits_verified"] is False and s["xe_diagnostics"]["within_limits"] is False
    assert s["xe_diagnostics"]["within_char_limits"] is True


def test_fineweb_source_pin_is_recorded_or_required(tmp_path, monkeypatch):
    """Unpinned FineWeb builds resolve and record the commit (warning); --require-pinned-source
    refuses a moving ref. (No network: the resolver and downloader are monkeypatched.)"""
    import build_dataset
    from ctx import blend, basecorpus
    monkeypatch.setattr(basecorpus, "resolve_fineweb_revision", lambda rev=None: "a" * 40)
    out, cache = tmp_path / "fw", tmp_path / "cache"
    argv = ["--out", str(out), "--base-source", "fineweb", "--num-groups", "8", "--freq-grid", "1", "--base-chars", "1000",
            "--chars-per-shard", "1000", "--seed", "0", "--counterfact-limit", "0", "--no-tracking", "--cache-dir", str(cache),
            "--require-pinned-source"]
    with pytest.raises(SystemExit, match="required by --require-pinned-source"):
        blend.build(build_dataset.parse_args(argv))
    assert not (out / "build_summary.json").exists()


# --- recheck 2026-10-14: findings 7 (tokenizer required up front), 8 (audit evidence), gaps 1-3 ----

def _no_build_output(out, cache):
    """Nothing of a build was written: no staging / arm / probe dirs or build artifacts under out,
    and the base-corpus cache was never populated."""
    if out.exists():
        written = [p.name for p in out.iterdir()
                   if p.name.startswith(("arm_", "probe_sets", blend.STAGING_DIR)) or p.name in blend.BUILD_ARTIFACTS]
        assert not written, written
    assert not cache.exists() or not any(cache.rglob("*")), list(cache.rglob("*"))


def test_unusable_tokenizer_src_stops_the_build_before_anything_is_written(tmp_path):
    """Recheck finding 7: a --tokenizer-src that cannot be loaded (corrupt pickle, missing file, no
    tokenizer/ dir) is a SystemExit at the START of build(): no base cache, no staging, no arm dirs."""
    import build_dataset
    from ctx import blend
    from ctx.tokenizers import TokenizerUnavailable
    out, cache = tmp_path / "b", tmp_path / "cache"
    src = tmp_path / "tok"
    (src / "tokenizer").mkdir(parents=True)
    pkl = src / "tokenizer" / "tokenizer.pkl"
    pkl.write_bytes(b"\x80\x04this is not a pickled tiktoken Encoding")
    with pytest.raises(SystemExit, match=r"--tokenizer-src .* is unusable: unreadable tokenizer\.pkl"):
        blend.build(build_dataset.parse_args(_argv2(out, cache, ["--embedding-control", "--tokenizer-src", str(src)])))
    _no_build_output(out, cache)
    pkl.unlink()
    with pytest.raises(SystemExit, match=r"is unusable: no tokenizer\.pkl under"):
        blend.build(build_dataset.parse_args(_argv2(out, cache, ["--tokenizer-src", str(src)])))
    _no_build_output(out, cache)
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no tokenizer/ directory"):
        blend.build(build_dataset.parse_args(_argv2(out, cache, ["--tokenizer-src", str(tmp_path / "empty")])))
    _no_build_output(out, cache)
    # an existing build is not touched either, even with --overwrite, when the tokenizer is unusable
    pkl.write_bytes(b"garbage")
    s = blend.build(build_dataset.parse_args(_argv2(out, cache)))
    before = {a: disk_hashes(out, a) for a in "CRX"}
    with pytest.raises(SystemExit, match="is unusable"):
        blend.build(build_dataset.parse_args(_argv2(out, cache, ["--overwrite", "--tokenizer-src", str(src)])))
    assert {a: disk_hashes(out, a) for a in "CRX"} == before and (out / "BUILD_COMPLETE.json").exists()
    assert s["match_tokenizer"] is None
    # the thin wrapper raises the dedicated exception instead of returning None
    with pytest.raises(TokenizerUnavailable, match="unreadable"):
        blend.load_match_tokenizer(src)


def test_build_with_tokenizer_records_loader_vocab_mode_and_enforces_claim_start_tokens(tmp_path, small_bpe_tokenizer, capsys):
    """A build with --tokenizer-src records how the tokenizer was loaded, its TRUE vocabulary and the
    selected limit mode; xe_diagnostics carries the limits version and the claim-start token deltas."""
    import build_dataset
    from ctx import blend
    from ctx import templates as T
    src = tmp_path / "tok"
    small_bpe_tokenizer.save(str(src / "tokenizer"))
    out, cache = tmp_path / "t", tmp_path / "cache"
    s = blend.build(build_dataset.parse_args(_argv2(out, cache, ["--embedding-control", "--tokenizer-src", str(src)])))
    printed = capsys.readouterr().out
    assert "loaded via nanochat.tokenizer.RustBPETokenizer; vocab 768; X/E token limit mode relative" in printed
    assert s["match_tokenizer"] == {"src": str(src), "loaded_via": "nanochat.tokenizer.RustBPETokenizer", "vocab": 768,
                                    "token_limit_mode": "relative"}
    xe = s["xe_diagnostics"]
    assert xe["limits_version"] == T.XE_MATCH_LIMITS_VERSION and xe["tokenizer_vocab"] == 768
    assert xe["token_limit_mode"] == "relative (toy tokenizer)" and xe["token_limits_verified"] is True
    assert 0 <= xe["max_abs_delta_claim_tok_start_frac"] <= T.XE_MATCH_LIMITS["max_abs_delta_tokens_frac"]
    assert xe["max_abs_delta_claim_tok_start"] >= xe["mean_abs_delta_claim_tok_start"] >= 0
    assert xe["within_token_limits"] is True and xe["within_limits"] is True
    assert summary_of(out)["match_tokenizer"] == s["match_tokenizer"]
    # a v2 build with arm E and no tokenizer warns loudly at the start (and stays unverified)
    s2 = blend.build(build_dataset.parse_args(_argv2(tmp_path / "notok", cache, ["--embedding-control"])))
    assert "WARNING  no --tokenizer-src" in capsys.readouterr().out
    assert s2["match_tokenizer"] is None and s2["xe_diagnostics"]["token_limits_verified"] is False


def test_summary_carries_design_cells_and_the_limits_version(v2_e):
    """Gaps 1 + 3: build_summary.json records the planned design cells (including empty ones) and the
    version of the frozen X/E limits it was matched against."""
    from ctx import templates as T
    out, s = v2_e
    assert s["xe_diagnostics"]["limits_version"] == T.XE_MATCH_LIMITS_VERSION == "2026-10-14.1"
    assert s["xe_diagnostics"]["limits"] == T.XE_MATCH_LIMITS
    assert summary_of(out)["xe_diagnostics"]["limits_version"] == "2026-10-14.1"
    dc = summary_of(out)["design_cells"]
    assert dc["grid"] == [1, 4, 16]
    assert all(set(row) == {"1", "4", "16"} for row in dc["cells"].values())
    assert sum(n for row in dc["cells"].values() for n in row.values()) == s["counts"]["training_groups"]
    assert dc["empty_cells"] == [[st, d] for st, row in dc["cells"].items() for d, n in row.items() if n == 0]
    assert all(k.endswith("|real") and {"true", "false"} <= set(v) or not k.endswith("|real") for k, v in dc["label_counts"].items())
    assert sum(sum(v.values()) for v in dc["label_counts"].values()) == s["counts"]["training_groups"]


def test_design_cells_list_empty_cells_when_the_grid_has_more_doses_than_a_stratum_has_groups(tmp_path):
    import build_dataset
    from collections import Counter
    from ctx import blend
    out, cache = tmp_path / "d", tmp_path / "cache"
    s = blend.build(build_dataset.parse_args(_argv2(out, cache, ["--num-groups", "24", "--freq-grid", "1", "2", "4", "8", "16"])))
    dc = summary_of(out)["design_cells"]
    assert dc["grid"] == [1, 2, 4, 8, 16] and dc["empty_cells"]
    inj = _records(out / "probe_sets" / "injected_facts.jsonl")
    assert sum(n for row in dc["cells"].values() for n in row.values()) == s["counts"]["training_groups"] == len(inj)
    for st, d in dc["empty_cells"]:
        assert dc["cells"][st][d] == 0
    by_dose = Counter(str(r["assigned_frequency"]) for r in inj)
    assert {d: sum(row[d] for row in dc["cells"].values()) for d in map(str, dc["grid"])} == \
        {d: by_dose.get(d, 0) for d in map(str, dc["grid"])}
    assert all(len(row) == 5 for row in dc["cells"].values())


def _build_with_audit(tmp_path, name="a"):
    import build_dataset
    from ctx import blend
    out, cache = tmp_path / name, tmp_path / "cache"
    s = blend.build(build_dataset.parse_args(_argv2(out, cache, ["--collision-audit", "--embedding-control"])))
    return out, cache, s


def _probes_only(out, cache, extra=()):
    import build_dataset
    from ctx import blend
    return blend.build(build_dataset.parse_args(_argv2(out, cache, ["--probes-only", "--embedding-control", *extra])))


def test_probes_only_restores_novelty_labels_from_the_embedded_audit_when_the_file_is_missing(tmp_path, capsys):
    """Recheck finding 8: build_summary.json embeds the audit; without novelty_audit.json the labels come
    from there (the NOTE names the evidence) and every label is reproduced, verified against the originals."""
    out, cache, s = _build_with_audit(tmp_path)
    assert isinstance(s["novelty_audit"]["hits"], dict) and s["novelty"]["audited_absent"] > 0
    (out / "novelty_audit.json").unlink()
    r = _probes_only(out, cache)
    printed = capsys.readouterr().out
    assert "novelty_audit.json missing; novelty labels restored from the audit embedded in build_summary.json" in printed
    assert r["verified_against_manifest"] is True and r["verified_against_probe_files"] is True
    assert r["novelty_evidence"] == "build_summary.json embedded novelty_audit (novelty_audit.json missing)"
    assert r["probe_files_compared"] == ["heldout_facts.jsonl", "injected_facts.jsonl"]
    for n in ("injected_facts.jsonl", "heldout_facts.jsonl"):
        assert (out / "probe_sets_regen" / n).read_bytes() == (out / "probe_sets" / n).read_bytes(), n
    orig = _records(out / "probe_sets" / "heldout_facts.jsonl")
    regen = _records(out / "probe_sets_regen" / "heldout_facts.jsonl")
    assert sum(a["novelty"] != b["novelty"] for a, b in zip(orig, regen)) == 0
    assert sum(x["novelty"] == "audited_absent" for x in regen) > 0 and not any(x["novelty"] == "unverified" for x in regen)
    meta = json.loads((out / "probe_sets_regen" / "PROBES_META.json").read_text())
    assert meta["verified_against_probe_files"] is True and meta["novelty_evidence"] == r["novelty_evidence"]
    assert meta["probe_files_compared"] == ["heldout_facts.jsonl", "injected_facts.jsonl"]


def test_probes_only_refuses_audit_evidence_that_is_missing_or_inconsistent(tmp_path):
    """Recheck finding 8: no usable hits, a file that contradicts the embedded audit, or an audit
    without this build's build_id -> SystemExit (nothing written); legacy summaries stay lenient."""
    out, cache, s = _build_with_audit(tmp_path)
    summary_path, audit_path = out / "build_summary.json", out / "novelty_audit.json"
    original_summary, original_audit = summary_path.read_text(), audit_path.read_text()
    # (1) the summary records the audit but without hits, and there is no file
    summ = json.loads(original_summary)
    del summ["novelty_audit"]["hits"]
    summary_path.write_text(json.dumps(summ))
    audit_path.unlink()
    with pytest.raises(SystemExit, match="no usable hits"):
        _probes_only(out, cache)
    assert not (out / "probe_sets_regen").exists() and not (out / "probe_sets_regen.tmp").exists()
    # (2) the file exists but its hits disagree with the embedded audit
    summary_path.write_text(original_summary)
    aud = json.loads(original_audit)
    aud["hits"] = dict(aud["hits"], tamperedstring=3)
    audit_path.write_text(json.dumps(aud))
    with pytest.raises(SystemExit, match="disagrees with the audit embedded in build_summary.json"):
        _probes_only(out, cache)
    # (3) the summary carries build_id but the audits do not: refused, file present or not
    aud = json.loads(original_audit)
    del aud["build_id"]
    audit_path.write_text(json.dumps(aud))
    summ = json.loads(original_summary)
    del summ["novelty_audit"]["build_id"]
    summary_path.write_text(json.dumps(summ))
    with pytest.raises(SystemExit, match="belongs to another build"):
        _probes_only(out, cache)
    audit_path.unlink()
    with pytest.raises(SystemExit, match="carries build_id None, not this build's"):
        _probes_only(out, cache)
    assert not (out / "probe_sets_regen").exists()
    # (4) a legacy summary without build_id keeps the lenient behaviour (NOTE) and regenerates
    summ = json.loads(original_summary)
    del summ["build_id"]
    summary_path.write_text(json.dumps(summ))
    audit_path.write_text(original_audit)
    r = _probes_only(out, cache)
    assert r["novelty_evidence"].startswith("novelty_audit.json") and r["verified_against_probe_files"] is True
    assert (out / "probe_sets_regen" / "heldout_facts.jsonl").read_bytes() == (out / "probe_sets" / "heldout_facts.jsonl").read_bytes()


def test_probes_only_regenerates_identically_from_a_copied_archive_set(tmp_path):
    """The archived set (summary, manifest, probe sets, marker, audit) copied WITHOUT any arm_* shards
    regenerates byte-identical probe records and verifies them against the originals."""
    import shutil
    out, cache, s = _build_with_audit(tmp_path)
    arch = tmp_path / "archive"
    arch.mkdir()
    for n in ("build_summary.json", "manifest.csv", "BUILD_COMPLETE.json", "novelty_audit.json"):
        shutil.copy2(out / n, arch / n)
    shutil.copytree(out / "probe_sets", arch / "probe_sets")
    assert not list(arch.glob("arm_*"))
    r = _probes_only(arch, cache)
    assert r["verified_against_manifest"] is True and r["verified_against_probe_files"] is True
    assert r["novelty_evidence"] == "novelty_audit.json (agrees with the audit embedded in build_summary.json)"
    for n in ("injected_facts.jsonl", "heldout_facts.jsonl"):
        assert (arch / "probe_sets_regen" / n).read_bytes() == (out / "probe_sets" / n).read_bytes(), n
        assert _records(arch / "probe_sets_regen" / n) == _records(out / "probe_sets" / n)


def test_probes_only_catches_tampered_audit_evidence_through_the_original_probe_files(tmp_path, capsys):
    """One hit added to the embedded audit (file deleted) flips a held-out novel group's label; the
    regenerated records then differ from the build's original probe files -> SystemExit naming the
    records, verified_against_probe_files=False, nothing written. A probe file that is no longer the
    marker's original is not evidence (NOTE) and the outcome is recorded as None."""
    out, cache, s = _build_with_audit(tmp_path)
    (out / "novelty_audit.json").unlink()
    held = _records(out / "probe_sets" / "heldout_facts.jsonl")
    victim = next(x for x in held if x["split"] == "heldout_group" and x["entity_tier"] == "novel"
                  and x["novelty"] == "audited_absent")
    summ = json.loads((out / "build_summary.json").read_text())
    summ["novelty_audit"]["hits"][victim["subject"]] = 1
    (out / "build_summary.json").write_text(json.dumps(summ))
    with pytest.raises(SystemExit) as ei:
        _probes_only(out, cache)
    msg = str(ei.value)
    assert "differ from the build's original probe files" in msg and "verified_against_probe_files=False" in msg
    assert f"({victim['fact_id']}, novelty, 'audited_absent', 'audited_present')" in msg
    assert "nothing was written" in msg
    assert not (out / "probe_sets_regen").exists() and not (out / "probe_sets_regen.tmp").exists()
    # the held-out file is no longer the build's original -> it is not used as evidence; the injected
    # file (unaffected by a held-out group) still verifies, the outcome is None, and the tampered
    # label now goes through: that is exactly why the originals and the marker are kept
    p = out / "probe_sets" / "heldout_facts.jsonl"
    p.write_text(p.read_text() + "\n")
    capsys.readouterr()
    r = _probes_only(out, cache)
    printed = capsys.readouterr().out
    assert "probe_sets/heldout_facts.jsonl is not the build's original" in printed
    assert r["verified_against_probe_files"] is None and r["probe_files_compared"] == ["injected_facts.jsonl"]
    regen = {x["fact_id"]: x for x in _records(out / "probe_sets_regen" / "heldout_facts.jsonl")}
    assert regen[victim["fact_id"]]["novelty"] == "audited_present"
    meta = json.loads((out / "probe_sets_regen" / "PROBES_META.json").read_text())
    assert meta["verified_against_probe_files"] is None and meta["probe_files_compared"] == ["injected_facts.jsonl"]
