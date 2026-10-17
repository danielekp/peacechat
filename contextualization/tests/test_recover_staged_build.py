"""Recover finalized staged bytes without rewriting or deleting the source corpus."""

import json
from pathlib import Path
import shutil

import pytest

import build_dataset
import recover_staged_build as recovery
import validate
from ctx import blend
from ctx.nanochat_io import read_shard_texts, write_shard
from ctx.protocol import sha256_file


ARGS = ["--protocol", "v2", "--base-source", "synthetic", "--num-groups", "12",
        "--freq-grid", "1", "4", "--base-chars", "100000", "--chars-per-shard", "30000",
        "--synthetic-docs-per-shard", "120", "--row-group-size", "8", "--seed", "0",
        "--counterfact-limit", "0", "--embedding-control", "--collision-audit", "--no-real", "--no-tracking"]


def config(out, cache):
    tokenizer = cache.parent / "tokenizer"
    extra = ["--tokenizer-src", str(tokenizer)] if tokenizer.is_dir() else []
    return build_dataset.parse_args([*ARGS, "--out", str(out), "--cache-dir", str(cache), *extra])


@pytest.fixture
def staged(tmp_path, small_bpe_tokenizer):
    built, cache = tmp_path / "generated", tmp_path / "cache"
    small_bpe_tokenizer.save(str(tmp_path / "tokenizer" / "tokenizer"))
    original = blend.build(config(built, cache))
    source = tmp_path / "original" / ".staging"
    source.mkdir(parents=True)
    for name in ["manifest.csv", "arm_C", "arm_R", "arm_X", "arm_E"]:
        shutil.move(str(built / name), source / name)
    return source, cache, original


def bytes_snapshot(source):
    return {str(p.relative_to(source)): sha256_file(p) for p in source.rglob("*") if p.is_file()}


def test_replay_adopts_identical_shards_with_new_honest_provenance(staged, tmp_path):
    source, cache, original = staged
    before = bytes_snapshot(source)
    out = tmp_path / "recovered"
    summary = recovery.recover(config(out, cache), source)
    assert before == bytes_snapshot(source)
    assert summary["build_id"] != original["build_id"]
    assert summary["config"]["recovery_source"] == str(source)
    assert summary["config"]["recovery_source_manifest_sha256"] == before["manifest.csv"]
    assert summary["config"]["recovery_script_sha256"] == sha256_file(Path(recovery.__file__))
    for key in ("arm_total_chars", "topup_docs", "topup_stream_doc_chars", "carriers_skipped_too_long",
                "injected_chars", "shard_hashes", "render_examples", "xe_diagnostics", "design_cells"):
        assert summary[key] == original[key], key
    assert summary["novelty_audit"]["hits"] == original["novelty_audit"]["hits"]
    assert summary["novelty_audit"]["scanned_shards"] > 0
    assert summary["novelty_audit"]["groups_absent"] + summary["novelty_audit"]["groups_present"] == 12
    for name in ("injected_facts.jsonl", "heldout_facts.jsonl"):
        assert sha256_file(out / "probe_sets" / name) == sha256_file(tmp_path / "generated" / "probe_sets" / name)
    marker = json.loads((out / "BUILD_COMPLETE.json").read_text())
    assert marker["build_summary_sha256"] == sha256_file(out / "build_summary.json")
    assert marker["manifest_sha256"] == before["manifest.csv"]
    for p in source.glob("arm_*/base_data/*.parquet"):
        target = out / p.relative_to(source)
        assert p.stat().st_ino == target.stat().st_ino
        assert p.stat().st_dev == target.stat().st_dev
        assert sha256_file(p) == summary["shard_hashes"][p.parent.parent.name[4:]][p.name]
    assert validate.main(["--out", str(out)]) == 0
    assert before == bytes_snapshot(source)


@pytest.mark.parametrize("corruption", ["document", "manifest", "extra_shard", "temporary_shard"])
def test_mismatch_never_completes_or_changes_source(staged, tmp_path, monkeypatch, corruption):
    source, cache, _ = staged
    paths = sorted((source / "arm_C/base_data").glob("*.parquet"))
    if corruption == "document":
        docs = list(read_shard_texts(str(paths[0])))
        docs[0] += " changed"
        write_shard(docs, str(paths[0]), row_group_size=8)
    elif corruption == "manifest":
        with (source / "manifest.csv").open("a") as fh:
            fh.write("unexpected trailing bytes\n")
    elif corruption == "extra_shard":
        shutil.copyfile(paths[-1], paths[-1].with_name(f"shard_{len(paths):05d}.parquet"))
    else:
        paths[0].with_suffix(".parquet.tmp").write_text("unfinished")
    before = bytes_snapshot(source)
    out = tmp_path / "rejected"
    # A mismatch must be rejected BEFORE the potentially expensive novelty scan.
    monkeypatch.setattr(blend.factpool_v2, "audit_novelty", lambda *a, **k: pytest.fail("audit reached before source verification"))
    with pytest.raises(recovery.RecoveryError):
        recovery.recover(config(out, cache), source)
    assert not (out / "BUILD_COMPLETE.json").exists()
    assert before == bytes_snapshot(source)


def test_refuses_existing_overlapping_or_symlinked_destination(staged, tmp_path):
    source, cache, _ = staged
    for out in (source, source.parent, source / "recovered", tmp_path / "cache"):
        with pytest.raises(recovery.RecoveryError):
            recovery.recover(config(out, cache), source)
    link = tmp_path / "source_link"
    link.symlink_to(source, target_is_directory=True)
    with pytest.raises(recovery.RecoveryError):
        recovery.recover(config(tmp_path / "fresh", cache), link)
    cfg = config(tmp_path / "other_fresh", cache)
    cfg.cache_dir = str(source / "cache")
    with pytest.raises(recovery.RecoveryError, match="cache must be outside"):
        recovery.recover(cfg, source)
