"""validate.py on fresh builds and on deliberately corrupted copies. Offline.

Run from the repository root:
    $HOME/.cache/ctx_cpu_venv/bin/python -m pytest contextualization/tests/test_validate.py -q

Every corruption keeps the build's provenance consistent (shard hashes in build_summary.json and
the BUILD_COMPLETE.json hashes are refreshed) so the test pins the check that must catch it, not
the hash check.
"""
import csv
import json
import os
import shutil
from pathlib import Path

import pytest

import build_dataset
import validate
from ctx import blend
from ctx.nanochat_io import read_shard_texts, write_shard
from ctx.protocol import sha256_file

V2_ARGV = ["--protocol", "v2", "--base-source", "synthetic", "--num-groups", "80", "--freq-grid", "1", "4", "16",
           "--base-chars", "2_000_000", "--chars-per-shard", "300_000", "--embedding-control", "--collision-audit",
           "--seed", "0", "--counterfact-limit", "0", "--no-tracking"]
V1_ARGV = ["--protocol", "v1", "--base-source", "synthetic", "--num-facts", "120", "--freq-grid", "1", "4",
           "--base-chars", "2_000_000", "--chars-per-shard", "300_000", "--seed", "0", "--counterfact-limit", "0",
           "--no-tracking"]
HISTORICAL_TOKENIZER_DIR = Path(os.environ.get("CTX_HISTORICAL_TOKENIZER_DIR",
                                               Path.home() / ".cache" / "ctx_wandb_to_hf" / "ctx_C_d24_base"))
HISTORICAL_TOKENIZER_SHA256 = "36ca79841b4203beb9d504c18e11ac7d301973982302ff4d233302bb1464c863"


# ---------------------------------------------------------------------------------------------
# helpers

def build(out, cache, argv, extra=()):
    return blend.build(build_dataset.parse_args([*argv, "--out", str(out), "--cache-dir", str(cache), *extra]))


def run(out, *extra):
    report = Path(out) / "validation_report.json"
    code = validate.main(["--out", str(out), *extra])
    rep = json.loads(report.read_text())
    return code, rep


def status(rep):
    return {c["id"]: c["status"] for c in rep["checks"]}


def failed(rep):
    return {c["id"] for c in rep["checks"] if c["status"] == "FAIL"}


def rows_of(out):
    with open(Path(out) / "manifest.csv", newline="") as fh:
        rd = csv.DictReader(fh)
        return rd.fieldnames, list(rd)


def write_rows(out, fieldnames, rows):
    with open(Path(out) / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def refresh_hashes(out):
    """Recompute build_summary.shard_hashes from disk and the completion marker's hashes."""
    out = Path(out)
    summary = json.loads((out / "build_summary.json").read_text())
    for a in list(summary["shard_hashes"]):
        d = out / f"arm_{a}" / "base_data"
        summary["shard_hashes"][a] = {p.name: sha256_file(p) for p in sorted(d.glob("*.parquet"))}
    (out / "build_summary.json").write_text(json.dumps(summary, indent=2))
    marker = json.loads((out / "BUILD_COMPLETE.json").read_text())
    marker["build_summary_sha256"] = sha256_file(out / "build_summary.json")
    marker["manifest_sha256"] = sha256_file(out / "manifest.csv")
    marker["probe_hashes"] = {n: sha256_file(out / "probe_sets" / n) for n in ("injected_facts.jsonl", "heldout_facts.jsonl")}
    (out / "BUILD_COMPLETE.json").write_text(json.dumps(marker, indent=2))


def rewrite_doc(out, row, mutate):
    """Replace the document a manifest row points at with mutate(doc); rewrite that shard."""
    out = Path(out)
    path = out / f"arm_{row['arm']}" / "base_data" / row["shard_file"]
    docs = list(read_shard_texts(str(path)))
    docs[int(row["row_in_shard"])] = mutate(docs[int(row["row_in_shard"])])
    rg = json.loads((out / "build_summary.json").read_text())["config"]["row_group_size"]
    write_shard(docs, str(path), rg)


# ---------------------------------------------------------------------------------------------
# fixtures

@pytest.fixture(scope="session")
def base_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("base_cache")


@pytest.fixture(scope="session")
def small_tokenizer_src(tmp_path_factory, small_bpe_tokenizer):
    """A nanochat-style base dir holding tokenizer/tokenizer.pkl + token_bytes.pt (for --tokenizer-src)."""
    src = tmp_path_factory.mktemp("tok_src")
    small_bpe_tokenizer.save(str(src / "tokenizer"))
    (src / "tokenizer" / "token_bytes.pt").write_bytes(b"token bytes fixture (hashed only)")
    return src


@pytest.fixture(scope="module")
def v2_build(tmp_path_factory, base_cache):
    out = tmp_path_factory.mktemp("v2")
    build(out, base_cache, V2_ARGV)
    return out


@pytest.fixture(scope="module")
def v1_build(tmp_path_factory, base_cache, small_tokenizer_src):
    out = tmp_path_factory.mktemp("v1")
    build(out, base_cache, V1_ARGV, ["--tokenizer-src", str(small_tokenizer_src)])
    return out


@pytest.fixture
def v2_copy(v2_build, tmp_path):
    dst = tmp_path / "v2copy"
    shutil.copytree(v2_build, dst)
    (dst / "validation_report.json").unlink(missing_ok=True)
    return dst


@pytest.fixture
def v1_copy(v1_build, tmp_path):
    dst = tmp_path / "v1copy"
    shutil.copytree(v1_build, dst)
    (dst / "validation_report.json").unlink(missing_ok=True)
    return dst


# ---------------------------------------------------------------------------------------------
# fresh builds pass

def test_fresh_v2_build_passes(v2_build):
    code, rep = run(v2_build)
    assert code == 0 and rep["result"] == "PASS" and not failed(rep), failed(rep)
    assert rep["schema_version"] == 1 and rep["protocol"] == "v2-corrected-2026-10" and rep["arms"] == ["C", "R", "X", "E"]
    assert "git_commit" in rep["validator_code"]
    st = status(rep)
    for cid in ("1.marker_hashes", "1.shard_hashes_E", "2.spread", "2.E_within_tol", "2.val_identical", "2.topup_stream",
                "3.diff_count", "3.manifest_positions", "4.insert_text", "4.value_text", "4.X_signature", "4.X_no_bare_claim",
                "4.E_clean_frame", "4.shared_carrier", "4.carrier_limit", "5.occurrences_vs_dose", "5.dose_maps_identical",
                "6.reciprocal_rivals", "6.subjects_unique", "6.training_groups_design", "6.novelty_labels",
                "7.heldout_claims", "7.rivals_unasserted", "7.heldout_subjects_novel", "8.char_length", "8.claim_start",
                "10.counts"):
        assert st.get(cid) == "PASS", (cid, st.get(cid))
    assert st["9.identity"] == "SKIP" and st["8.token_length"] == "SKIP"      # no tokenizer in this build
    assert (v2_build / "validation_report.json").exists()
    # every check carries numbers / exclusions fields and a section
    assert all({"id", "section", "status", "message", "numbers", "exclusions"} <= set(c) for c in rep["checks"])


def test_fresh_v1_build_passes(v1_build):
    code, rep = run(v1_build)
    assert code == 0 and rep["result"] == "PASS", failed(rep)
    st = status(rep)
    assert rep["protocol"] == "v1-legacy-2026-07" and rep["arms"] == ["C", "R", "X"]
    assert st["6.legacy_collisions"] == "NOTE" and st["7.heldout_subjects_v1"] == "NOTE"
    legacy = [c for c in rep["checks"] if c["id"] == "6.legacy_collisions"][0]["numbers"]
    assert {"groups_with_more_than_two_facts", "groups_multiple_true_values", "missing_opposing_label_rival_links",
            "heldout_with_injected_subject_relation"} <= set(legacy)
    assert st["9.pkl_identical"] == "PASS" and st["9.token_bytes_identical"] == "PASS" and st["9.vs_summary"] == "PASS"
    assert st["2.tokens_per_char"] == "NOTE" and not any(k.startswith("8.") for k in st)
    assert st["4.X_signature"] == "PASS" and st["10.tier_tv_counts"] == "PASS"


def test_fresh_v2_build_with_historical_tokenizer(tmp_path, base_cache):
    pkl = HISTORICAL_TOKENIZER_DIR / "tokenizer" / "tokenizer.pkl"
    if not pkl.exists() or sha256_file(pkl) != HISTORICAL_TOKENIZER_SHA256:
        pytest.skip("historical tokenizer not available")
    out = tmp_path / "v2tok"
    build(out, base_cache, V2_ARGV, ["--tokenizer-src", str(HISTORICAL_TOKENIZER_DIR)])
    code, rep = run(out)
    assert code == 0, failed(rep)
    st = status(rep)
    assert st["8.token_length"] == "PASS" and st["9.pkl_identical"] == "PASS" and st["9.vs_summary"] == "PASS"


def test_cli_entry_point(v1_build):
    import subprocess, sys
    r = subprocess.run([sys.executable, str(Path(validate.__file__)), "--out", str(v1_build), "--report",
                        str(v1_build / "cli_report.json")], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "ALL CHECKS PASSED" in r.stdout and "[matched examples]" in r.stdout
    assert (v1_build / "cli_report.json").exists()


# ---------------------------------------------------------------------------------------------
# corruptions

def test_corrupt_value_span_text(v2_copy):
    fields, rows = rows_of(v2_copy)
    row = next(r for r in rows if r["arm"] == "R")
    s, e = int(row["value_char_start"]), int(row["value_char_end"])
    rewrite_doc(v2_copy, row, lambda d: d[:s] + "Q" * (e - s) + d[e:])
    rel = s - int(row["insert_char_start"])                   # keep insert_text in step with the document
    row["insert_text"] = row["insert_text"][:rel] + "Q" * (e - s) + row["insert_text"][rel + (e - s):]
    write_rows(v2_copy, fields, rows)
    refresh_hashes(v2_copy)
    code, rep = run(v2_copy)
    assert code == 1 and rep["result"] == "FAIL"
    f = failed(rep)
    assert "4.value_text" in f and "4.R_surface_form" in f
    assert "1.marker_hashes" not in f and "1.shard_hashes_R" not in f      # provenance was kept consistent
    bad = [c for c in rep["checks"] if c["id"] == "4.value_text"][0]
    assert bad["numbers"]["n_bad"] == 1 and bad["exclusions"][0].startswith(row["rendering_id"])


def test_dropped_manifest_row(v2_copy):
    fields, rows = rows_of(v2_copy)
    victim = next(r for r in rows if r["arm"] == "X" and int(r["assigned_frequency"]) > 1)
    write_rows(v2_copy, fields, [r for r in rows if r is not victim])
    refresh_hashes(v2_copy)
    code, rep = run(v2_copy)
    assert code == 1
    f = failed(rep)
    assert {"5.rows_per_arm", "5.slot_all_arms", "5.occurrences_vs_dose", "3.slot_rows"} <= f, f


def test_probe_record_rival_id_changed(v2_copy):
    p = v2_copy / "probe_sets" / "injected_facts.jsonl"
    recs = [json.loads(l) for l in p.read_text().splitlines()]
    recs[0]["rival_fact_id"] = "0000000000000000"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    refresh_hashes(v2_copy)
    code, rep = run(v2_copy)
    assert code == 1 and "6.reciprocal_rivals" in failed(rep)
    bad = [c for c in rep["checks"] if c["id"] == "6.reciprocal_rivals"][0]
    assert bad["exclusions"] == [recs[0]["group_id"]]


def test_E_insert_with_a_source_name(v2_copy):
    fields, rows = rows_of(v2_copy)
    row = next(r for r in rows if r["arm"] == "E")
    ins_s, cl_s = int(row["insert_char_start"]), int(row["claim_char_start"])
    lead = validate.REGISTER_LEADINS[int(row["register_index"])]
    frame_start = ins_s + len(lead)
    assert cl_s - frame_start > 12
    pos = frame_start + 3                                     # inside the frame, before the claim
    name = "Omar"                                             # same length as the 4 characters replaced

    def mutate(d):
        return d[:pos] + name + d[pos + len(name):]
    rewrite_doc(v2_copy, row, mutate)
    rel = pos - ins_s
    row["insert_text"] = row["insert_text"][:rel] + name + row["insert_text"][rel + len(name):]
    write_rows(v2_copy, fields, rows)
    refresh_hashes(v2_copy)
    code, rep = run(v2_copy)
    assert code == 1
    f = failed(rep)
    assert "4.E_clean_frame" in f and "4.insert_text" not in f, f
    bad = [c for c in rep["checks"] if c["id"] == "4.E_clean_frame"][0]
    assert bad["numbers"]["n_bad"] == 1 and "first_name" in bad["exclusions"][0]


def test_tokenizer_bytes_differ_between_arms(v1_copy):
    (v1_copy / "arm_R" / "tokenizer" / "tokenizer.pkl").write_bytes(b"not the shared tokenizer")
    code, rep = run(v1_copy)
    assert code == 1 and "9.pkl_identical" in failed(rep)
    assert status(rep)["9.vs_summary"] == "PASS"                 # arm C still matches the recorded fingerprint


def test_deleted_completion_marker(v2_copy):
    (v2_copy / "BUILD_COMPLETE.json").unlink()
    code, rep = run(v2_copy)
    assert code == 1 and "1.marker_present" in failed(rep)
    assert "4.value_text" not in failed(rep)                     # the rest of the build is still checked


def test_leftover_staging_and_stale_shard_hash(v2_copy):
    (v2_copy / ".staging").mkdir()
    summary = json.loads((v2_copy / "build_summary.json").read_text())
    summary["shard_hashes"]["C"]["shard_00000.parquet"] = "0" * 64
    (v2_copy / "build_summary.json").write_text(json.dumps(summary))
    code, rep = run(v2_copy)
    assert code == 1
    assert {"1.no_staging", "1.shard_hashes_C", "1.marker_hashes"} <= failed(rep)


def test_held_out_claim_leaked_into_an_insert(v2_copy):
    """An R insert that asserts a held-out group's claim must be caught by the leakage check."""
    held = [json.loads(l) for l in (v2_copy / "probe_sets" / "heldout_facts.jsonl").read_text().splitlines()]
    target = next(h for h in held if h.get("split") == "heldout_group")
    fields, rows = rows_of(v2_copy)
    row = next(r for r in rows if r["arm"] == "R")
    claim = target["claim_text"]
    s, e = int(row["insert_char_start"]), int(row["insert_char_end"])

    def mutate(d):
        return d[:s] + claim + d[e:]
    rewrite_doc(v2_copy, row, mutate)
    row.update(insert_text=claim, insert_char_end=str(s + len(claim)), claim_char_start=str(s),
               claim_char_end=str(s + len(claim)), register_index="0")
    write_rows(v2_copy, fields, rows)
    refresh_hashes(v2_copy)
    code, rep = run(v2_copy)
    assert code == 1
    f = failed(rep)
    assert "7.heldout_claims" in f
    if target["entity_tier"] == "novel":
        assert "7.heldout_subjects_novel" in f
