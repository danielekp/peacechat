"""Offline tests for the artifact-reliability repairs (review 2026-10-13, findings 19-22) and for
the immutable checkpoint identities / evaluation branches (review 2026-10-14, finding 1).

Nothing here touches the network: the FineWeb download is monkeypatched to write a known
file and return a fake ``X-Repo-Commit``; ``ctx.tracking.hf_api`` is a raising sentinel, a
recorder, or the revision-aware ``FakeHub`` below (commits, branches, tags, ``repo_info`` with
``revision`` / ``files_metadata``, ``upload_folder`` with ``revision``); ``.env`` files are not read.

Run from the repository root:
    $HOME/.cache/ctx_cpu_venv/bin/python -m pytest contextualization/tests/test_artifacts.py -q
"""
import contextlib
import fnmatch
import gzip
import hashlib
import json
import os
import runpy
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

from ctx import basecorpus as bc
from ctx import nanochat_io as nio
from ctx import tracking as T
from ctx.protocol import sha256_file, tokenizer_fingerprint

CTX_DIR = Path(__file__).resolve().parents[1]
SHARD_KW = dict(chars_per_shard=1000, synthetic_docs_per_shard=10, row_group_size=8)
FAKE_COMMIT = "fakecommit0123456789"


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tmp_files(root) -> list:
    return [p for p in Path(root).rglob("*.tmp")]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Guard: any attempt to reach the Hub / FineWeb from these tests fails loudly."""
    def boom(*a, **k):
        raise AssertionError("network access attempted during an offline test")
    monkeypatch.setattr(T, "_ENV_LOADED", True)  # do not read the user's .env files
    monkeypatch.setattr(T, "_HF_NS", None)
    monkeypatch.setattr(T, "hf_api", boom)
    monkeypatch.setattr(bc.requests, "get", boom)
    monkeypatch.setattr(bc.requests, "head", boom)
    monkeypatch.setenv("CTX_TRACKING", "0")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_NAMESPACE", raising=False)


@pytest.fixture
def fake_download(monkeypatch):
    """Replace ``download_fineweb_shard`` with a writer of known content + a fake X-Repo-Commit.
    Returns the list of calls (index, dest_dir, revision, expected_sha256)."""
    calls = []

    def fake(index, dest_dir, *, revision=None, expected_sha256=None, max_attempts=5):
        p = os.path.join(dest_dir, nio.shard_filename(index))
        docs = [f"fineweb document {index}-{i} " + "lorem ipsum dolor " * 8 for i in range(5)]
        digest = nio.write_shard(docs, p, 8)
        calls.append((index, dest_dir, revision, expected_sha256))
        return {"sha256": digest, "x_repo_commit": FAKE_COMMIT,
                "url": f"fake://{revision}/{nio.shard_filename(index)}", "bytes": os.path.getsize(p)}

    monkeypatch.setattr(bc, "download_fineweb_shard", fake)
    return calls


# --------------------------------------------------------------------------------------
# basecorpus: cache identity, pinning, checksums
# --------------------------------------------------------------------------------------

def test_reviewer_sequence_synthetic_then_fineweb_are_distinct(tmp_path, fake_download):
    """reproduce_dataset_findings.py, last block: this used to return the SAME file twice."""
    cache = str(tmp_path / "cache-key-demo")
    p = bc.ensure_shard("synthetic", 0, cache, seed=1, **SHARD_KW)
    q = bc.ensure_shard("fineweb", 0, cache, seed=999, **SHARD_KW)
    assert p != q
    assert sha(p) != sha(q)
    assert Path(p).parent.name == "synthetic-seed1-chars1000-docs10-rg8"
    assert Path(q).parent.name == "fineweb-edu-100b-shuffle-revmain"
    assert fake_download == [(0, str(Path(q).parent), "main", None)]
    rec = bc.load_checksums(cache, bc.cache_key("fineweb"))["shard_00000.parquet"]
    assert rec["sha256"] == sha(q) and rec["x_repo_commit"] == FAKE_COMMIT
    assert rec["bytes"] == os.path.getsize(q) and rec["url"].startswith("fake://")
    assert bc.load_checksums(cache, bc.cache_key("synthetic", seed=1, **SHARD_KW))["shard_00000.parquet"]["sha256"] == sha(p)
    # second request is served from the keyed cache: no new download, same bytes
    assert bc.ensure_shard("fineweb", 0, cache, seed=999, **SHARD_KW) == q and len(fake_download) == 1
    assert tmp_files(cache) == []
    # the fineweb directory holds the synthetic docs nowhere
    texts = list(bc.iter_shard_docs(q))
    assert all(t.startswith("fineweb document") for t in texts)


def test_cache_key_and_paths(tmp_path):
    assert bc.cache_key("synthetic", seed=1, **SHARD_KW) == "synthetic-seed1-chars1000-docs10-rg8"
    assert bc.cache_key("synthetic", seed=1, revision="ignored", **SHARD_KW) == "synthetic-seed1-chars1000-docs10-rg8"
    assert bc.cache_key("fineweb") == "fineweb-edu-100b-shuffle-revmain"
    assert bc.cache_key("fineweb", seed=5, revision="refs/pr/1") == "fineweb-edu-100b-shuffle-revrefs_pr_1"
    commit = "0123456789abcdef0123456789abcdef01234567"
    assert bc.cache_key("fineweb", revision=commit) == f"fineweb-edu-100b-shuffle-rev{commit}"
    with pytest.raises(ValueError):
        bc.cache_key("synthetic", **SHARD_KW)  # no seed
    with pytest.raises(ValueError):
        bc.cache_key("wikipedia")
    with pytest.raises(ValueError):
        bc.ensure_shard("wikipedia", 0, str(tmp_path), seed=1, **SHARD_KW)
    expect = os.path.join(str(tmp_path), "synthetic-seed3-chars1000-docs10-rg8", "shard_00002.parquet")
    assert bc.shard_cache_path("synthetic", 2, str(tmp_path), seed=3, **SHARD_KW) == expect
    assert bc.ensure_shard("synthetic", 2, str(tmp_path), seed=3, **SHARD_KW) == expect
    assert bc.fineweb_url(7, "refs/pr/1") == (
        "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/refs%2Fpr%2F1/shard_00007.parquet")
    assert bc.fineweb_url("shard_00000.parquet") == f"{bc.FINEWEB_BASE_URL}/shard_00000.parquet"


def test_synthetic_generation_is_deterministic_and_seed_keyed(tmp_path):
    a = bc.ensure_shard("synthetic", 0, str(tmp_path / "c1"), seed=1, **SHARD_KW)
    b = bc.ensure_shard("synthetic", 0, str(tmp_path / "c2"), seed=1, **SHARD_KW)
    c = bc.ensure_shard("synthetic", 0, str(tmp_path / "c1"), seed=2, **SHARD_KW)
    assert sha(a) == sha(b)
    assert sha(a) != sha(c) and Path(a).parent != Path(c).parent
    assert bc.count_docs([a]) == nio.shard_num_rows(a) > 0


def test_expected_sha256_mismatch_deletes_download(tmp_path, fake_download):
    cache = str(tmp_path)
    with pytest.raises(bc.ChecksumMismatch):
        bc.ensure_shard("fineweb", 3, cache, seed=0, expected_sha256="0" * 64, **SHARD_KW)
    path = bc.shard_cache_path("fineweb", 3, cache)
    assert not os.path.exists(path)
    assert "shard_00003.parquet" not in bc.load_checksums(cache, bc.cache_key("fineweb"))
    assert tmp_files(cache) == []
    assert fake_download[0][3] == "0" * 64  # the expectation was forwarded to the downloader


def test_expected_sha256_match_and_cached_mismatch(tmp_path, fake_download):
    cache = str(tmp_path)
    p = bc.ensure_shard("fineweb", 0, cache, seed=0, **SHARD_KW)
    good = sha(p)
    # pinned request against the cached file: ok (also with upper-case hex)
    assert bc.ensure_shard("fineweb", 0, cache, seed=0, expected_sha256=good.upper(), **SHARD_KW) == p
    # pinned request for different content: refused, but the intact cached file is kept
    with pytest.raises(bc.ChecksumMismatch):
        bc.ensure_shard("fineweb", 0, cache, seed=0, expected_sha256="1" * 64, **SHARD_KW)
    assert os.path.exists(p) and sha(p) == good
    # a fresh cache with a correct expectation downloads and records normally
    q = bc.ensure_shard("fineweb", 0, str(tmp_path / "other"), seed=0, expected_sha256=good, **SHARD_KW)
    assert sha(q) == good


def test_verify_detects_corrupted_cached_shard(tmp_path, fake_download):
    cache = str(tmp_path)
    p = bc.ensure_shard("fineweb", 1, cache, seed=0, **SHARD_KW)
    data = bytearray(Path(p).read_bytes())
    data[len(data) // 2] ^= 0xFF  # same size, different bytes
    Path(p).write_bytes(bytes(data))
    assert bc.ensure_shard("fineweb", 1, cache, seed=0, **SHARD_KW) == p  # size check alone cannot see it
    with pytest.raises(bc.ChecksumMismatch):
        bc.ensure_shard("fineweb", 1, cache, seed=0, verify=True, **SHARD_KW)
    assert not os.path.exists(p)
    assert "shard_00001.parquet" not in bc.load_checksums(cache, bc.cache_key("fineweb"))
    # rerun re-downloads cleanly
    assert bc.ensure_shard("fineweb", 1, cache, seed=0, verify=True, **SHARD_KW) == p
    assert len(fake_download) == 2 and tmp_files(cache) == []


def test_size_mismatch_detected_without_verify(tmp_path, fake_download):
    cache = str(tmp_path)
    p = bc.ensure_shard("fineweb", 2, cache, seed=0, **SHARD_KW)
    Path(p).write_bytes(Path(p).read_bytes()[:-10])  # truncated
    with pytest.raises(bc.ChecksumMismatch):
        bc.ensure_shard("fineweb", 2, cache, seed=0, **SHARD_KW)
    assert not os.path.exists(p)


def test_fineweb_file_without_record_is_refused_unless_pinned(tmp_path, fake_download):
    cache = str(tmp_path)
    key = bc.cache_key("fineweb")
    path = Path(bc.shard_cache_path("fineweb", 4, cache))
    path.parent.mkdir(parents=True)
    nio.write_shard(["some document of unknown origin"], str(path), 8)
    with pytest.raises(bc.CacheIntegrityError):
        bc.ensure_shard("fineweb", 4, cache, seed=0, **SHARD_KW)
    assert path.exists() and fake_download == []  # left in place, not silently reused, not re-downloaded
    assert bc.ensure_shard("fineweb", 4, cache, seed=0, expected_sha256=sha(path), **SHARD_KW) == str(path)
    rec = bc.load_checksums(cache, key)["shard_00004.parquet"]
    assert rec["sha256"] == sha(path) and rec["provenance"].startswith("adopted")
    assert bc.ensure_shard("fineweb", 4, cache, seed=0, **SHARD_KW) == str(path)  # recorded now


def test_synthetic_file_without_record_is_regenerated(tmp_path):
    cache = str(tmp_path / "c")
    path = Path(bc.shard_cache_path("synthetic", 0, cache, seed=1, **SHARD_KW))
    path.parent.mkdir(parents=True)
    nio.write_shard(["bogus"], str(path), 8)
    bogus = sha(path)
    assert bc.ensure_shard("synthetic", 0, cache, seed=1, **SHARD_KW) == str(path)
    ref = bc.ensure_shard("synthetic", 0, str(tmp_path / "ref"), seed=1, **SHARD_KW)
    assert sha(path) == sha(ref) != bogus


def test_legacy_cache_requires_explicit_hashes(tmp_path, fake_download):
    legacy = tmp_path / "flat_legacy_cache"
    legacy.mkdir()
    lp = legacy / "shard_00000.parquet"
    nio.write_shard(["legacy fineweb document " * 5], str(lp), 8)
    legacy_sha = sha(lp)
    cache = str(tmp_path / "keyed")
    with pytest.raises(ValueError):
        bc.ensure_shard("fineweb", 0, cache, seed=0, legacy_cache_dir=str(legacy), **SHARD_KW)
    assert fake_download == []
    # wrong assertion: refused, legacy file untouched, nothing adopted
    with pytest.raises(bc.ChecksumMismatch):
        bc.ensure_shard("fineweb", 0, cache, seed=0, legacy_cache_dir=str(legacy),
                        legacy_cache_sha256={"shard_00000.parquet": "f" * 64}, **SHARD_KW)
    assert sha(lp) == legacy_sha and not os.path.exists(bc.shard_cache_path("fineweb", 0, cache))
    # correct assertion: adopted into the keyed cache, no download
    p = bc.ensure_shard("fineweb", 0, cache, seed=0, legacy_cache_dir=str(legacy),
                        legacy_cache_sha256={"shard_00000.parquet": legacy_sha}, **SHARD_KW)
    assert p == bc.shard_cache_path("fineweb", 0, cache) and sha(p) == legacy_sha
    rec = bc.load_checksums(cache, bc.cache_key("fineweb"))["shard_00000.parquet"]
    assert rec["provenance"].startswith("legacy:") and rec["x_repo_commit"] is None
    assert fake_download == []
    # a shard the legacy dir does not have falls through to the (fake) download
    bc.ensure_shard("fineweb", 1, cache, seed=0, legacy_cache_dir=str(legacy),
                    legacy_cache_sha256={"shard_00000.parquet": legacy_sha}, **SHARD_KW)
    assert [c[0] for c in fake_download] == [1]
    # synthetic ignores the legacy escape hatch entirely
    s = bc.ensure_shard("synthetic", 0, cache, seed=1, legacy_cache_dir=str(legacy),
                        legacy_cache_sha256={"shard_00000.parquet": legacy_sha}, **SHARD_KW)
    assert sha(s) != legacy_sha and Path(s).parent.name.startswith("synthetic-")


def test_resolve_fineweb_revision_offline(monkeypatch):
    class Resp:
        def __init__(self, headers, ok=True, body=None):
            self.headers, self.ok, self._body = headers, ok, body

        def json(self):
            return self._body

    seen = {}

    def head(url, allow_redirects=True, timeout=None):
        seen["url"] = url
        seen["allow_redirects"] = allow_redirects
        return Resp({"X-Repo-Commit": "abc123", "X-Linked-Etag": '"' + "e" * 64 + '"'})

    monkeypatch.setattr(bc.requests, "head", head)
    assert bc.resolve_fineweb_revision() == "abc123"
    assert seen["allow_redirects"] is False and seen["url"] == bc.fineweb_url(0, "main")
    assert bc.fineweb_remote_sha256(0) == "e" * 64

    def failing(*a, **k):
        raise bc.requests.RequestException("offline")

    monkeypatch.setattr(bc.requests, "head", failing)
    assert bc.resolve_fineweb_revision("main") is None
    assert bc.fineweb_remote_sha256(0) is None


def test_iter_docs_from_shards_and_count_docs(tmp_path):
    cache = str(tmp_path)
    paths = [bc.ensure_shard("synthetic", i, cache, seed=4, **SHARD_KW) for i in range(2)]
    docs = list(bc.iter_docs_from_shards("synthetic", range(2), cache, seed=4, **SHARD_KW))
    assert len(docs) == bc.count_docs(paths) > 0
    assert docs[: nio.shard_num_rows(paths[0])] == list(bc.iter_shard_docs(paths[0]))


# --------------------------------------------------------------------------------------
# nanochat_io: ShardWriter
# --------------------------------------------------------------------------------------

def test_shard_writer_hashes_atomic_and_refuses_overwrite(tmp_path, capsys):
    out = str(tmp_path / "arm_C" / "base_data")
    w = nio.ShardWriter(out, chars_per_shard=100, row_group_size=2)
    for i in range(9):
        w.add("x" * 30 + str(i))
    hashes = w.finalize(["val doc one", "val doc two"])
    files = sorted(f for f in os.listdir(out) if f.endswith(".parquet"))
    assert files == sorted(hashes) == [Path(p).name for p in w.shard_paths]
    assert len(files) >= 2 and files[-1] == nio.shard_filename(len(files) - 1)
    for name, h in hashes.items():
        assert sha(Path(out) / name) == h == nio.shard_sha256(Path(out) / name)
    assert hashes == w.shard_hashes and tmp_files(out) == []
    assert list(nio.read_shard_texts(Path(out) / files[-1])) == ["val doc one", "val doc two"]
    assert w.total_docs == 9 and w.total_chars == 9 * 31
    with pytest.raises(FileExistsError) as e:
        nio.ShardWriter(out, chars_per_shard=100, row_group_size=2)
    assert out in str(e.value) and "overwrite=True" in str(e.value)
    assert sorted(f for f in os.listdir(out) if f.endswith(".parquet")) == files  # nothing deleted
    nio.ShardWriter(out, chars_per_shard=100, row_group_size=2, overwrite=True)
    assert [f for f in os.listdir(out) if f.endswith(".parquet")] == []
    assert f"deleted {len(files)} existing shard(s)" in capsys.readouterr().out


def test_add_indexed_rows(tmp_path):
    w = nio.ShardWriter(str(tmp_path / "d"), chars_per_shard=10, row_group_size=2)
    assert w.add_indexed("aaaaa") == (0, 0)
    assert w.add_indexed("bbbbbb") == (0, 1)  # 11 chars and 2 docs -> flushes shard 0 after this add
    assert w.add_indexed("c") == (1, 0)
    assert w.add_returning("d") == 1
    assert w.add_indexed("eeeeeeeeeee") == (1, 2)
    w.finalize(["v"])
    assert sorted(w.shard_hashes) == ["shard_00000.parquet", "shard_00001.parquet", "shard_00002.parquet"]
    assert list(nio.read_shard_texts(w.shard_paths[1])) == ["c", "d", "eeeeeeeeeee"]
    with pytest.raises(AssertionError):
        w.add("after finalize")


def test_write_shard_failure_leaves_no_tmp(tmp_path, monkeypatch):
    path = tmp_path / "shard_00000.parquet"

    def broken(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(nio.pq, "write_table", broken)
    with pytest.raises(RuntimeError):
        nio.write_shard(["doc"], str(path), 8)
    assert not path.exists() and not Path(str(path) + ".tmp").exists()


# --------------------------------------------------------------------------------------
# tracking: manifest refresh, bindings, export manifests
# --------------------------------------------------------------------------------------

def test_refresh_manifest_gz_by_content(tmp_path):
    exp = tmp_path / "ctx_exp_test"
    exp.mkdir()
    assert T.refresh_manifest_gz(exp) is None  # no manifest yet
    manifest, gz = exp / "manifest.csv", exp / "manifest.csv.gz"
    manifest.write_text("fact_id,arm\nf1,R\n")
    first = T.refresh_manifest_gz(exp)
    assert first["refreshed"] is True and gzip.open(gz, "rb").read() == manifest.read_bytes()
    assert (exp / "manifest.csv.sha256").read_text() == f"{sha(manifest)}  manifest.csv\n"
    gz_bytes = gz.read_bytes()
    # rebuild in place: new manifest content, but the gz is made to look NEWER (the old mtime rule)
    manifest.write_text("fact_id,arm\nf1,R\nf2,X\n")
    future = time.time() + 3600
    os.utime(gz, (future, future))
    assert gz.stat().st_mtime > manifest.stat().st_mtime
    second = T.refresh_manifest_gz(exp)
    assert second["refreshed"] is True
    assert gzip.open(gz, "rb").read() == manifest.read_bytes()
    assert second["sha256"] == sha(manifest) and second["gz_sha256"] == sha(gz)
    assert (exp / "manifest.csv.sha256").read_text() == f"{sha(manifest)}  manifest.csv\n"
    assert gz.read_bytes() != gz_bytes
    # content-identical -> nothing to do; and the gzip bytes are deterministic (mtime=0, no name)
    assert T.refresh_manifest_gz(exp)["refreshed"] is False
    current = gz.read_bytes()
    gz.unlink()
    assert T.refresh_manifest_gz(exp)["refreshed"] is True and gz.read_bytes() == current
    # a corrupt gz is treated as stale
    gz.write_bytes(b"not gzip")
    assert T.refresh_manifest_gz(exp)["refreshed"] is True and gzip.open(gz, "rb").read() == manifest.read_bytes()
    assert tmp_files(exp) == []


def make_arm(exp_dir: Path, arm="C", depth=4, step=10, bindings="match", optim=True, run_id="abcd1234"):
    arm_dir = exp_dir / f"arm_{arm}"
    tok = arm_dir / "tokenizer"
    tok.mkdir(parents=True)
    (tok / "tokenizer.pkl").write_bytes(b"fake tokenizer pickle for arm " + arm.encode())
    (tok / "token_bytes.pt").write_bytes(b"fake token bytes")
    ck = arm_dir / "base_checkpoints" / f"d{depth}"
    ck.mkdir(parents=True)
    (ck / f"model_{step:06d}.pt").write_bytes(bytes(range(256)) * 8)
    if optim:
        (ck / f"optim_{step:06d}_rank0.pt").write_bytes(b"optimizer shard rank 0")
    meta = {"step": step, "val_bpb": 1.25, "model_config": {"n_layer": depth, "vocab_size": 1024},
            "user_config": {"depth": depth, "run": f"ctx_{arm}"},
            "wandb_run": {"id": run_id, "name": f"ctx_{arm}_d{depth}", "project": "nanochat",
                          "url": f"https://wandb.ai/x/nanochat/runs/{run_id}"} if run_id else None}
    if bindings == "match":
        meta["bindings"] = {"tokenizer_sha256": sha256_file(tok / "tokenizer.pkl"),
                            "build_summary_sha256": "b" * 64, "probe_hashes": {"probe_sets/injected_facts.jsonl": "p" * 64},
                            "prepack_manifest_sha256": None, "data_listing_sha256": "d" * 64,
                            "init_seed": 7, "data_seed": 11, "data_loader": "prepacked",
                            "code": {"git_commit": "deadbeef", "git_dirty": False}}
    elif isinstance(bindings, dict):
        meta["bindings"] = bindings
    (ck / f"meta_{step:06d}.json").write_text(json.dumps(meta))
    if run_id:
        T.write_run_info(exp_dir, arm, wandb_run_id=run_id, wandb_run_name=f"ctx_{arm}_d{depth}",
                         wandb_project="nanochat", depth=depth, exp_tag=exp_dir.name)
    return arm_dir, ck


def test_hf_upload_checkpoint_tokenizer_mismatch_raises_before_network(tmp_path, monkeypatch):
    exp = tmp_path / "ctx_exp_mismatch"
    make_arm(exp, bindings={"tokenizer_sha256": "0" * 64})
    monkeypatch.setattr(T, "hf_upload_dir", lambda *a, **k: (_ for _ in ()).throw(AssertionError("upload attempted")))
    # HF_NAMESPACE is unset and hf_api is a raising sentinel (autouse): any namespace lookup = network
    with pytest.raises(SystemExit) as e:
        T.hf_upload_checkpoint(exp, "C", 4, run_id="abcd1234")
    msg = str(e.value)
    assert "0" * 64 in msg and sha256_file(exp / "arm_C" / "tokenizer" / "tokenizer.pkl") in msg
    assert "mismatch" in msg


def test_hf_upload_checkpoint_unbound_needs_flag_and_stages_manifest(tmp_path, monkeypatch, capsys):
    exp = tmp_path / "ctx_exp_unbound"
    make_arm(exp, bindings=None, optim=True)
    staged = {}

    def fake_upload_dir(repo_id, folder, repo_type="model", path_in_repo="", commit_message="", tags=(),
                        delete_patterns=(), soft_tags=(), revision=None, retag=False):
        staged["repo_id"], staged["tags"], staged["delete_patterns"] = repo_id, tags, delete_patterns
        staged["soft_tags"], staged["retag"], staged["revision"] = soft_tags, retag, revision
        staged["files"] = {p.relative_to(folder).as_posix(): p.read_bytes() for p in Path(folder).rglob("*") if p.is_file()}
        return "https://huggingface.co/testns/ctx_C_d4_base", "0123abcd" * 5

    monkeypatch.setattr(T, "hf_upload_dir", fake_upload_dir)
    monkeypatch.setattr(T, "hf_tag_commit", lambda *a, **k: None)  # run not archived yet (the Hub itself is the sentinel)
    monkeypatch.setattr(T, "hf_ensure_repo", lambda *a, **k: None)  # destination creation is the Hub's business too
    monkeypatch.setenv("HF_NAMESPACE", "testns")
    with pytest.raises(SystemExit):
        T.hf_upload_checkpoint(exp, "C", 4, run_id="abcd1234")
    assert "WARNING: checkpoint not bound to a tokenizer (legacy meta)" in capsys.readouterr().out
    assert staged == {}
    url, sha_, step = T.hf_upload_checkpoint(exp, "C", 4, run_id="abcd1234", allow_unbound=True, extra_tags=("ctx_exp_unbound",))
    assert step == "000010" and staged["repo_id"] == "testns/ctx_C_d4_base"
    # the run id is an identity tag, the exp_tag a label (soft); nothing is retagged unless asked
    assert staged["tags"] == ("wandb-abcd1234",) and staged["soft_tags"] == ("ctx_exp_unbound",)
    assert staged["retag"] is False and staged["revision"] is None
    files = staged["files"]
    assert {"model_000010.pt", "meta_000010.json", "tokenizer/tokenizer.pkl", "tokenizer/token_bytes.pt",
            "checkpoint_manifest.json", "README.md"} <= set(files)
    assert "optim_000010_rank0.pt" not in files
    m = json.loads(files["checkpoint_manifest.json"])
    assert m["optimizer_included"] is False and m["optimizer_requested"] is False
    assert m["optimizer_files"] == [] and m["optimizer_files_on_disk"] == ["optim_000010_rank0.pt"]
    assert m["tokenizer_binding"]["status"] == "unbound" and m["bindings"] is None and m["hf_repo_id"] == "testns/ctx_C_d4_base"
    assert {f["name"]: f["sha256"] for f in m["files"]}["model_000010.pt"] == sha(exp / "arm_C" / "base_checkpoints" / "d4" / "model_000010.pt")
    readme = files["README.md"].decode()
    assert "Optimizer state: **not included**" in readme and "binding to the checkpoint meta: **unbound**" in readme
    # with the optimizer: staged, listed and hashed
    T.hf_upload_checkpoint(exp, "C", 4, run_id="abcd1234", allow_unbound=True, with_optimizer=True)
    files = staged["files"]
    assert files["optim_000010_rank0.pt"] == b"optimizer shard rank 0"
    m = json.loads(files["checkpoint_manifest.json"])
    assert m["optimizer_included"] is True and m["optimizer_files"] == ["optim_000010_rank0.pt"]
    assert any(f["name"] == "optim_000010_rank0.pt" and f["sha256"] == hashlib.sha256(b"optimizer shard rank 0").hexdigest()
               for f in m["files"])
    assert "Optimizer state: **included**" in files["README.md"].decode()


def test_hf_upload_checkpoint_bound_ok_and_step_selection(tmp_path, monkeypatch):
    exp = tmp_path / "ctx_exp_bound"
    arm_dir, ck = make_arm(exp, bindings="match", step=10)
    (ck / "model_000020.pt").write_bytes(b"later checkpoint")  # newest, without meta
    staged = {}

    def fake_upload_dir(repo_id, folder, repo_type="model", **k):
        staged["files"] = sorted(p.relative_to(folder).as_posix() for p in Path(folder).rglob("*") if p.is_file())
        return "url", "sha"

    monkeypatch.setattr(T, "hf_upload_dir", fake_upload_dir)
    monkeypatch.setenv("HF_NAMESPACE", "testns")
    _, _, step = T.hf_upload_checkpoint(exp, "C", 4, step=10)
    assert step == "000010" and "meta_000010.json" in staged["files"] and "model_000020.pt" not in staged["files"]
    with pytest.raises(SystemExit):
        T.hf_upload_checkpoint(exp, "C", 4, step=30)
    with pytest.raises(SystemExit):  # the newest checkpoint has no meta -> unbound -> refused without the flag
        T.hf_upload_checkpoint(exp, "C", 4)


def test_export_manifest_fields_and_hashes(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_NAMESPACE", "testns")
    exp = tmp_path / "ctx_exp_export"
    arm_dir, ck = make_arm(exp, bindings="match", optim=True, run_id="r1r1r1r1")
    out = T.export_manifest(exp, "C", 4)
    assert out == exp / "arm_C" / "export" / "checkpoint_manifest.json" and out.exists()
    m = json.loads(out.read_text())
    required = {"schema_version", "arm", "depth", "step", "files", "optimizer_included", "optimizer_files",
                "tokenizer", "bindings", "run_id", "exp_tag", "created_at", "code", "suggested", "tokenizer_binding"}
    assert required <= set(m)
    assert m["schema_version"] == 1 and m["arm"] == "C" and m["depth"] == 4 and m["step"] == 10
    assert m["exp_tag"] == "ctx_exp_export" and m["run_id"] == "r1r1r1r1"
    where = {"model_000010.pt": ck / "model_000010.pt", "meta_000010.json": ck / "meta_000010.json",
             "tokenizer/tokenizer.pkl": arm_dir / "tokenizer" / "tokenizer.pkl",
             "tokenizer/token_bytes.pt": arm_dir / "tokenizer" / "token_bytes.pt"}
    assert {f["name"] for f in m["files"]} == set(where)
    for f in m["files"]:
        assert f["sha256"] == sha(where[f["name"]]) and f["bytes"] == where[f["name"]].stat().st_size
    assert m["optimizer_included"] is False and m["optimizer_files"] == [] and m["optimizer_files_on_disk"] == ["optim_000010_rank0.pt"]
    assert m["tokenizer"] == tokenizer_fingerprint(arm_dir / "tokenizer")
    assert m["bound"] is True and m["tokenizer_binding"]["status"] == "ok"
    for k in ("build_summary_sha256", "probe_hashes", "prepack_manifest_sha256", "data_listing_sha256",
              "init_seed", "data_seed", "data_loader", "code", "tokenizer_sha256"):
        assert k in m["bindings"]
    assert m["bindings"]["init_seed"] == 7 and m["bindings"]["data_loader"] == "prepacked"
    assert m["suggested"]["hf_repo_id"] == "testns/ctx_C_d4_base"
    assert m["suggested"]["hf_tags"] == ["wandb-r1r1r1r1", "ctx_exp_export"]
    assert m["suggested"]["wandb_run_id"] == "r1r1r1r1" and "--step 10" in m["suggested"]["upload_command"]
    assert "--allow-unbound" not in m["suggested"]["upload_command"]
    assert "git_commit" in m["code"] and "created_at" in m
    # optimizer requested and present
    m2 = json.loads(T.export_manifest(exp, "C", 4, with_optimizer=True).read_text())
    assert m2["optimizer_included"] is True and m2["optimizer_files"] == ["optim_000010_rank0.pt"]
    assert any(f["name"] == "optim_000010_rank0.pt" and f["sha256"] == sha(ck / "optim_000010_rank0.pt") for f in m2["files"])
    assert "--with-optimizer" in m2["suggested"]["upload_command"]
    # optimizer requested but absent -> not included
    (ck / "optim_000010_rank0.pt").unlink()
    m3 = json.loads(T.export_manifest(exp, "C", 4, with_optimizer=True).read_text())
    assert m3["optimizer_requested"] is True and m3["optimizer_included"] is False and m3["optimizer_files_on_disk"] == []
    # unbound legacy meta: exported (not refused), flagged, upload command carries --allow-unbound
    exp2 = tmp_path / "ctx_exp_legacy"
    make_arm(exp2, bindings=None, run_id=None)
    m4 = json.loads(T.export_manifest(exp2, "C", 4).read_text())
    assert m4["bound"] is False and m4["bindings"] is None and m4["run_id"] is None
    assert "--allow-unbound" in m4["suggested"]["upload_command"] and m4["suggested"]["hf_tags"] == ["ctx_exp_legacy"]


def test_export_build_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_NAMESPACE", "testns")
    exp = tmp_path / "ctx_exp_build"
    (exp / "probe_sets").mkdir(parents=True)
    (exp / "probe_sets" / "injected_facts.jsonl").write_text('{"fact_id": "f1"}\n')
    (exp / "probe_sets" / "heldout_facts.jsonl").write_text('{"fact_id": "f2"}\n')
    (exp / "build_summary.json").write_text(json.dumps({"protocol": "v2-corrected-2026-10", "builder_version": 2,
                                                        "config": {"seed": 5}, "shard_hashes": {"C": {"shard_00000.parquet": "a" * 64}}}))
    (exp / "manifest.csv").write_text("fact_id,arm\nf1,R\n")
    (exp / "shard_hashes.json").write_text(json.dumps({"C": {"shard_00000.parquet": "a" * 64}}))
    (exp / "exposure" / "C").mkdir(parents=True)
    (exp / "exposure" / "C" / "summary.json").write_text("{}")
    make_arm(exp, arm="C", run_id=None)
    w = nio.ShardWriter(str(exp / "arm_C" / "base_data"), chars_per_shard=10, row_group_size=1)
    w.add("train doc " * 3)
    w.finalize(["val doc"])
    out = T.export_build_manifest(exp)
    assert out == exp / "export" / "build_manifest.json"
    m = json.loads(out.read_text())
    assert (exp / "manifest.csv.gz").exists() and (exp / "manifest.csv.sha256").exists()
    for rel in ("build_summary.json", "manifest.csv", "manifest.csv.gz", "manifest.csv.sha256", "shard_hashes.json"):
        assert m["files"][rel]["sha256"] == sha(exp / rel)
    assert {p["name"]: p["sha256"] for p in m["probe_files"]} == {
        "probe_sets/heldout_facts.jsonl": sha(exp / "probe_sets" / "heldout_facts.jsonl"),
        "probe_sets/injected_facts.jsonl": sha(exp / "probe_sets" / "injected_facts.jsonl")}
    assert m["exposure_files"] == [{"name": "exposure/C/summary.json", "bytes": 2, "sha256": sha(exp / "exposure" / "C" / "summary.json")}]
    assert m["manifest_csv_sha256"] == sha(exp / "manifest.csv")
    assert m["arms"]["C"]["n_shards"] == 2 and m["arms"]["C"]["tokenizer"]["tokenizer_sha256"] == sha(exp / "arm_C" / "tokenizer" / "tokenizer.pkl")
    assert m["summary"]["protocol"] == "v2-corrected-2026-10" and m["summary_seed"] == 5 and m["summary_has_shard_hashes"] is True
    assert m["suggested"]["hf_dataset_repo"] == "testns/ctx_experiment_build" and m["suggested"]["path_in_repo"] == "ctx_exp_build/"
    assert m["suggested"]["hf_tag"] == "ctx_exp_build" and m["exp_tag"] == "ctx_exp_build"


# --------------------------------------------------------------------------------------
# run_arm: command + provenance recording (no training)
# --------------------------------------------------------------------------------------

def test_build_train_command_and_record():
    import run_arm
    cmd = run_arm.build_train_command(4, "ctx_C_d4", 1, ["--device-type=cpu", "--num-iterations=20"],
                                      init_seed=7, data_seed=11, data_loader="prepacked", python="py")
    assert cmd == ["py", "-m", "scripts.base_train", "--depth=4", "--run=ctx_C_d4", "--init-seed=7",
                   "--data-seed=11", "--data-loader=prepacked", "--device-type=cpu", "--num-iterations=20"]
    cmd2 = run_arm.build_train_command(24, "ctx_R_d24", 8, python="py")
    assert cmd2 == ["py", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
                    "-m", "scripts.base_train", "--", "--depth=24", "--run=ctx_R_d24"]
    assert not any(f.startswith(("--init-seed", "--data-seed", "--data-loader")) for f in cmd2)
    assert run_arm.build_train_command(4, "dummy", 1, data_seed=0, python="py")[-1] == "--data-seed=0"
    with pytest.raises(ValueError):
        run_arm.build_train_command(4, "x", 1, data_loader="random")
    rec = run_arm.training_record(cmd, {"NANOCHAT_BASE_DIR": "/data/arm_C", "WANDB_RUN_ID": "abcd1234"},
                                  init_seed=7, data_seed=11, data_loader="prepacked")
    assert rec["train_command"] == cmd and rec["train_command_str"] == " ".join(cmd)
    assert rec["train_env"] == {"NANOCHAT_BASE_DIR": "/data/arm_C", "WANDB_RUN_ID": "abcd1234"}
    assert rec == {**rec, "init_seed": 7, "data_seed": 11, "data_loader": "prepacked"}
    assert set(rec["code"]) >= {"git_commit", "git_dirty", "git_branch", "builder_version"}
    assert rec["code"]["git_dirty"] in (True, False, None)
    assert rec["train_cwd"] == str(run_arm.REPO)
    assert run_arm.training_record(["py"], None)["init_seed"] is None


def test_run_arm_cli_declares_new_flags():
    import run_arm
    src = (CTX_DIR / "run_arm.py").read_text()
    for flag in ("--init-seed", "--data-seed", "--data-loader", "--allow-unbound"):
        assert flag in src
    assert run_arm.DATA_LOADERS == ("bestfit", "prepacked")
    assert 'choices=["C", "R", "X", "E"]' in src


def test_load_artifact_manifest_only_offline(tmp_path):
    exp = tmp_path / "ctx_exp_cli"
    make_arm(exp, bindings="match", run_id="zzzz9999")
    (exp / "manifest.csv").write_text("fact_id,arm\n")
    (exp / "build_summary.json").write_text("{}")
    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HF_NAMESPACE", "WANDB_API_KEY")}
    env.update(CTX_ENV_FILE="/nonexistent/.env", HF_HUB_OFFLINE="1", CTX_TRACKING="0", HF_NAMESPACE="testns")
    script = str(CTX_DIR / "load_artifact_hf.py")
    r = subprocess.run([sys.executable, script, "--exp-dir", str(exp), "--arm", "C", "--step", "10", "--manifest-only"],
                       capture_output=True, text=True, env=env, cwd=str(CTX_DIR.parent))
    assert r.returncode == 0, r.stderr
    m = json.loads((exp / "arm_C" / "export" / "checkpoint_manifest.json").read_text())
    assert m["step"] == 10 and m["depth"] == 4 and m["run_id"] == "zzzz9999"  # depth from run_info.json
    assert not (exp / "export" / "build_manifest.json").exists()
    r = subprocess.run([sys.executable, script, "--exp-dir", str(exp), "--build", "--manifest-only"],
                       capture_output=True, text=True, env=env, cwd=str(CTX_DIR.parent))
    assert r.returncode == 0, r.stderr
    assert (exp / "export" / "build_manifest.json").exists() and (exp / "manifest.csv.gz").exists()
    r = subprocess.run([sys.executable, script, "--exp-dir", str(exp), "--arm", "C", "--build", "--with-optimizer", "--manifest-only"],
                       capture_output=True, text=True, env=env, cwd=str(CTX_DIR.parent))
    assert r.returncode == 0, r.stderr
    assert "build manifest ->" in r.stdout and "checkpoint manifest ->" in r.stdout
    assert json.loads((exp / "arm_C" / "export" / "checkpoint_manifest.json").read_text())["optimizer_included"] is True


# --------------------------------------------------------------------------------------
# tracking: immutable checkpoint identities + evaluation branches (review 2026-10-14, finding 1)
# --------------------------------------------------------------------------------------

from huggingface_hub.errors import HfHubHTTPError, RevisionNotFoundError  # noqa: E402

WEIGHTS_A, WEIGHTS_B = b"weights of run A", b"weights of run B"
SHA_A, SHA_B = hashlib.sha256(WEIGHTS_A).hexdigest(), hashlib.sha256(WEIGHTS_B).hexdigest()


class FakeHub:
    """Offline stand-in for HfApi with real repository semantics: commits with parents, branches,
    tags, revision-aware ``repo_info`` (LFS sha256 for ``*.pt``), ``upload_folder`` onto a branch
    (``delete_patterns``, no-op commits), ``create_branch`` / ``create_tag`` with ``exist_ok``,
    ``list_repo_refs`` and ``list_repo_commits``. Every call is recorded in ``calls``."""

    def __init__(self):
        self.commits, self.branches, self.tags, self.calls, self._n = {}, {}, {}, [], 0

    # ---- state helpers -------------------------------------------------------------
    def commit(self, files: dict, parent=None, branch="main") -> str:
        self._n += 1
        sha = f"{self._n:02d}" + hashlib.sha1(f"{self._n}:{sorted(files)}".encode()).hexdigest()[:38]
        self.commits[sha] = {"files": dict(files), "parent": parent}
        self.branches[branch] = sha
        return sha

    def _resolve(self, revision):
        revision = revision or "main"
        if revision in self.branches:
            return self.branches[revision]
        if revision in self.tags:
            return self.tags[revision]
        if revision in self.commits:
            return revision
        raise RevisionNotFoundError(f"revision {revision!r} not found")

    def files_at(self, revision) -> dict:
        return self.commits[self._resolve(revision)]["files"]

    def history(self, revision) -> list:
        out, sha = [], self._resolve(revision)
        while sha is not None:
            out.append(sha)
            sha = self.commits[sha]["parent"]
        return out

    def snapshot(self) -> tuple:
        return dict(self.commits), dict(self.branches), dict(self.tags)

    # ---- HfApi surface -------------------------------------------------------------
    def create_repo(self, repo_id, repo_type=None, private=True, exist_ok=False, **kw):
        self.calls.append(("create_repo", repo_id))

    def list_repo_refs(self, repo_id, repo_type=None, **kw):
        ref = lambda kind, name, sha: types.SimpleNamespace(name=name, ref=f"refs/{kind}/{name}", target_commit=sha)  # noqa: E731
        self.calls.append(("list_repo_refs", repo_id))
        return types.SimpleNamespace(branches=[ref("heads", b, s) for b, s in self.branches.items()],
                                     tags=[ref("tags", t, s) for t, s in self.tags.items()], converts=[], pull_requests=[])

    def repo_info(self, repo_id, revision=None, repo_type=None, files_metadata=False, **kw):
        sha = self._resolve(revision)
        self.calls.append(("repo_info", revision, files_metadata))
        siblings = []
        for path, data in sorted(self.commits[sha]["files"].items()):
            lfs = (types.SimpleNamespace(size=len(data), sha256=hashlib.sha256(data).hexdigest(), pointer_size=0)
                   if files_metadata and path.endswith(".pt") else None)
            siblings.append(types.SimpleNamespace(rfilename=path, size=len(data) if files_metadata else None,
                                                  blob_id=None, lfs=lfs))
        return types.SimpleNamespace(sha=sha, siblings=siblings)

    def upload_folder(self, *, repo_id, folder_path, path_in_repo=None, commit_message=None, repo_type=None,
                      revision=None, delete_patterns=None, **kw):
        branch = revision or "main"
        if branch not in self.branches:
            raise RevisionNotFoundError(f"branch {branch!r} not found")
        parent = self.branches[branch]
        base = dict(self.commits[parent]["files"])
        if delete_patterns:
            base = {p: d for p, d in base.items() if not any(fnmatch.fnmatch(p, pat) for pat in delete_patterns)}
        prefix = path_in_repo.strip("/") + "/" if path_in_repo else ""
        new = {prefix + p.relative_to(folder_path).as_posix(): p.read_bytes()
               for p in sorted(Path(folder_path).rglob("*")) if p.is_file()}
        files = {**base, **new}
        self.calls.append(("upload_folder", branch, sorted(new)))
        if files == self.commits[parent]["files"]:
            return types.SimpleNamespace(oid=parent)  # hub: "No files have been modified since last commit"
        return types.SimpleNamespace(oid=self.commit(files, parent=parent, branch=branch))

    def create_branch(self, repo_id, *, branch, revision=None, repo_type=None, exist_ok=False, **kw):
        self.calls.append(("create_branch", branch, revision))
        if branch in self.branches:
            if exist_ok:
                return
            raise HfHubHTTPError(f"409 Conflict: branch {branch!r} already exists")
        self.branches[branch] = self._resolve(revision)

    def create_tag(self, repo_id, *, tag, revision=None, repo_type=None, exist_ok=False, tag_message=None, **kw):
        self.calls.append(("create_tag", tag, revision))
        if tag in self.tags:
            if exist_ok:
                return
            raise HfHubHTTPError(f"409 Conflict: tag {tag!r} already exists")
        self.tags[tag] = self._resolve(revision)

    def delete_tag(self, repo_id, *, tag, repo_type=None, **kw):
        self.calls.append(("delete_tag", tag))
        del self.tags[tag]

    def list_repo_commits(self, repo_id, repo_type=None, revision=None, **kw):
        return [types.SimpleNamespace(commit_id=s, title="", message="", authors=[], created_at=None)
                for s in self.history(revision)]

    def called(self, *names) -> list:
        return [c for c in self.calls if c[0] in names]


@pytest.fixture
def hub(monkeypatch):
    """The reviewer's setting (review_root_v2.py): two training runs in one model repo, the old run A
    at tag wandb-A, the later run B at wandb-B and on main; the build label ctx_exp_v3 marks B."""
    h = FakeHub()
    a = h.commit({"model_000100.pt": WEIGHTS_A, "meta_000100.json": b'{"step": 100}',
                  "tokenizer/tokenizer.pkl": b"tok", "README.md": b"run A"})
    b = h.commit({"model_000100.pt": WEIGHTS_B, "meta_000100.json": b'{"step": 100}',
                  "tokenizer/tokenizer.pkl": b"tok", "README.md": b"run B"}, parent=a)
    h.tags.update({"wandb-A": a, "wandb-B": b, "ctx_exp_v3": b})
    monkeypatch.setattr(T, "hf_api", lambda: h)
    monkeypatch.setattr(T, "upload_progress", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setenv("HF_NAMESPACE", "testns")
    return h


REPO = "testns/ctx_R_d24_base"


def eval_files(root: Path, arm="R", model_sha256=SHA_A, step=100, text="evaluation of checkpoint A\n") -> dict:
    """eval/beliefs_arm<ARM>.csv + .meta.json as eval_beliefs.py writes them (checkpoint.model_sha256)."""
    d = root / "eval"
    d.mkdir(exist_ok=True, parents=True)
    csv, meta = d / f"beliefs_arm{arm}.csv", d / f"beliefs_arm{arm}.meta.json"
    csv.write_text(text)
    meta.write_text(json.dumps({"arm": arm, "checkpoint": {"model_sha256": model_sha256, "step": step}}))
    return {f"eval/{csv.name}": csv, f"eval/{meta.name}": meta}


def test_reviewer_sequence_old_run_eval_leaves_its_checkpoint_tag_alone(hub, tmp_path):
    """review_root_v2.py: archiving A's evaluation used to move wandb-A onto a commit holding B's weights."""
    a, b = hub.tags["wandb-A"], hub.tags["wandb-B"]
    rec = T.hf_upload_eval(REPO, eval_files(tmp_path), "A", SHA_A, step=100)
    # the identities are untouched: wandb-A still marks weights A, main still holds run B
    assert hub.tags["wandb-A"] == a and hub.files_at("wandb-A")["model_000100.pt"] == WEIGHTS_A
    assert hub.tags["wandb-B"] == b and hub.branches["main"] == b and hub.files_at("main")["model_000100.pt"] == WEIGHTS_B
    assert not hub.called("delete_tag")
    # the evaluation sits on a branch that starts at A's checkpoint commit and holds A's weights + the eval files
    head = hub.branches["eval/wandb-A"]
    got = hub.files_at(head)
    assert got["model_000100.pt"] == WEIGHTS_A and got["eval/beliefs_armR.csv"] == b"evaluation of checkpoint A\n"
    assert "eval/beliefs_armR.meta.json" in got and hub.history(head) == [head, a]
    assert hub.tags["wandb-A-eval"] == head
    assert {"url", "commit_sha", "branch", "tag", "checkpoint_commit", "verified"} <= set(rec)
    assert rec["commit_sha"] == head and rec["branch"] == "eval/wandb-A" and rec["tag"] == "wandb-A-eval"
    assert rec["checkpoint_commit"] == a and rec["verified"] is True and rec["model_file"] == "model_000100.pt"
    assert rec["url"] == "https://huggingface.co/testns/ctx_R_d24_base/tree/eval%2Fwandb-A"
    assert rec["model_sha256"] == SHA_A and rec["files"] == ["eval/beliefs_armR.csv", "eval/beliefs_armR.meta.json"]
    assert [c for c in hub.called("upload_folder")] == [("upload_folder", "eval/wandb-A", rec["files"])]


def test_hf_upload_eval_wrong_sha_raises_before_any_upload(hub, tmp_path):
    before = hub.snapshot()
    files = eval_files(tmp_path, model_sha256=SHA_B)
    with pytest.raises(T.CheckpointMismatch) as e:
        T.hf_upload_eval(REPO, files, "A", SHA_B)  # the eval claims to have scored B's weights
    assert SHA_A in str(e.value) and SHA_B in str(e.value) and "model_000100.pt" in str(e.value)
    with pytest.raises(T.CheckpointMismatch):
        T.hf_upload_eval(REPO, files, "A", SHA_A, step=200)  # no model_000200.pt at that commit
    with pytest.raises(SystemExit) as e:
        T.hf_upload_eval(REPO, files, "ZZ", SHA_A)  # run never archived
    assert "wandb-ZZ" in str(e.value)
    assert hub.snapshot() == before
    assert not hub.called("upload_folder", "create_branch", "create_tag", "delete_tag")


def test_hf_upload_eval_without_sha_needs_allow_unverified(hub, tmp_path):
    files = eval_files(tmp_path, model_sha256=None)
    with pytest.raises(SystemExit) as e:
        T.hf_upload_eval(REPO, files, "A", None)
    assert "not bound to a checkpoint" in str(e.value) and "--allow-unverified" in str(e.value)
    assert not hub.called("upload_folder", "create_branch")
    rec = T.hf_upload_eval(REPO, files, "A", None, allow_unverified=True)
    assert rec["verified"] is False and rec["checkpoint_commit"] == hub.tags["wandb-A"] and rec["model_sha256"] == SHA_A
    assert hub.files_at("eval/wandb-A")["model_000100.pt"] == WEIGHTS_A and hub.tags["wandb-A-eval"] == rec["commit_sha"]


def test_hf_upload_eval_rescoring_tags_and_branch_checks(hub, tmp_path):
    first = T.hf_upload_eval(REPO, eval_files(tmp_path), "A", SHA_A)
    files = eval_files(tmp_path, text="rescored with the corrected evaluator\n")
    # the eval tag already marks the first evaluation: refused before anything is uploaded
    n = len(hub.commits)
    with pytest.raises(T.TagConflict) as e:
        T.hf_upload_eval(REPO, files, "A", SHA_A)
    assert (e.value.tag, e.value.existing_commit, e.value.new_commit) == ("wandb-A-eval", first["commit_sha"], None)
    assert len(hub.commits) == n and hub.tags["wandb-A-eval"] == first["commit_sha"] and not hub.called("delete_tag")
    # a re-scoring gets its own tag on the same branch, which still descends from A's checkpoint
    second = T.hf_upload_eval(REPO, files, "A", SHA_A, tag="wandb-A-eval-rescore")
    assert hub.history(second["commit_sha"]) == [second["commit_sha"], first["commit_sha"], hub.tags["wandb-A"]]
    assert hub.files_at(second["commit_sha"])["eval/beliefs_armR.csv"] == b"rescored with the corrected evaluator\n"
    assert hub.tags["wandb-A-eval-rescore"] == second["commit_sha"] and hub.tags["wandb-A-eval"] == first["commit_sha"]
    assert hub.tags["wandb-A"] == hub.history(second["commit_sha"])[-1]
    with pytest.raises(ValueError):
        T.hf_upload_eval(REPO, files, "A", SHA_A, tag="rescore")  # evaluation tags keep the wandb- prefix
    # an eval branch that does not descend from the run's checkpoint commit is refused
    hub.branches["eval/wandb-B"] = hub.tags["wandb-A"]
    with pytest.raises(T.BranchConflict):
        T.hf_upload_eval(REPO, eval_files(tmp_path, model_sha256=SHA_B), "B", SHA_B)
    # ... as is one whose head no longer holds the checkpoint's bytes
    hub.branches["eval/wandb-B"] = hub.commit({"model_000100.pt": b"tampered"}, parent=hub.tags["wandb-B"], branch="eval/wandb-B")
    with pytest.raises(T.CheckpointMismatch):
        T.hf_upload_eval(REPO, eval_files(tmp_path, model_sha256=SHA_B), "B", SHA_B)
    with pytest.raises(SystemExit):
        T.hf_upload_eval(REPO, files, "", SHA_A)  # no run id


def test_hf_upload_dir_tag_conflict_raises_and_never_deletes_a_tag(hub, tmp_path, capsys):
    folder = tmp_path / "up"
    folder.mkdir()
    (folder / "notes.txt").write_text("x")
    a, b = hub.tags["wandb-A"], hub.tags["wandb-B"]
    with pytest.raises(T.TagConflict) as e:
        T.hf_upload_dir(REPO, folder, tags=("wandb-A",))
    new = hub.branches["main"]
    assert (e.value.tag, e.value.existing_commit, e.value.new_commit, e.value.repo_id) == ("wandb-A", a, new, REPO)
    assert "immutable" in str(e.value) and hub.tags["wandb-A"] == a and not hub.called("delete_tag")
    # a new identity is created; re-uploading identical content lands on the same commit, which is fine
    _, sha1 = T.hf_upload_dir(REPO, folder, tags=("wandb-N",))
    assert hub.tags["wandb-N"] == sha1 == new
    _, sha2 = T.hf_upload_dir(REPO, folder, tags=("wandb-N",))
    assert sha2 == sha1 and hub.tags["wandb-N"] == sha1 and not hub.called("delete_tag")
    # labels: created when absent, left where they are otherwise; never a wandb-* tag
    (folder / "notes.txt").write_text("y")
    _, sha3 = T.hf_upload_dir(REPO, folder, soft_tags=("ctx_exp_v3", "fresh-label"))
    assert sha3 != sha1 and hub.tags["ctx_exp_v3"] == b and hub.tags["fresh-label"] == sha3
    assert "stays at" in capsys.readouterr().out
    with pytest.raises(ValueError):
        T.hf_upload_dir(REPO, folder, soft_tags=("wandb-A",))
    assert hub.tags["wandb-A"] == a
    # retag=True is the only way a tag moves, and it says so
    (folder / "notes.txt").write_text("z")
    _, sha4 = T.hf_upload_dir(REPO, folder, tags=("wandb-A",), soft_tags=("ctx_exp_v3",), retag=True)
    assert hub.tags["wandb-A"] == sha4 and hub.tags["ctx_exp_v3"] == sha4
    assert [c for c in hub.called("delete_tag")] == [("delete_tag", "wandb-A"), ("delete_tag", "ctx_exp_v3")]
    assert "WARNING: moving tag wandb-A" in capsys.readouterr().out
    # a commit on another branch: revision= is honoured and main is untouched
    hub.create_branch(REPO, branch="eval/wandb-N", revision=sha1)
    _, sha5 = T.hf_upload_dir(REPO, folder, revision="eval/wandb-N", tags=("wandb-N-eval",))
    assert hub.branches["eval/wandb-N"] == sha5 and hub.branches["main"] == sha4 and hub.tags["wandb-N-eval"] == sha5


def test_hf_upload_checkpoint_refuses_archived_run_without_force_retag(hub, tmp_path, capsys):
    exp = tmp_path / "ctx_exp_v3"
    make_arm(exp, arm="R", depth=24, bindings="match", run_id="A")
    before = hub.snapshot()
    with pytest.raises(T.TagConflict) as e:
        T.hf_upload_checkpoint(exp, "R", 24, run_id="A", extra_tags=("ctx_exp_v3",))
    assert (e.value.tag, e.value.existing_commit, e.value.new_commit) == ("wandb-A", hub.tags["wandb-A"], None)
    assert "refused before it started" in str(e.value)
    assert hub.snapshot() == before and not hub.called("upload_folder", "create_tag", "delete_tag")
    # a new run id archives normally; the build label stays with the first run that carried it
    url, sha_, step = T.hf_upload_checkpoint(exp, "R", 24, run_id="NEW", extra_tags=("ctx_exp_v3",))
    assert step == "000010" and hub.tags["wandb-NEW"] == sha_ == hub.branches["main"]
    assert hub.tags["ctx_exp_v3"] == hub.tags["wandb-B"] and "stays at" in capsys.readouterr().out
    assert hub.files_at("wandb-A")["model_000100.pt"] == WEIGHTS_A and hub.files_at("wandb-B")["model_000100.pt"] == WEIGHTS_B
    assert "model_000100.pt" not in hub.files_at("main") and "model_000010.pt" in hub.files_at("main")
    # --force-retag moves the identity (and the label) deliberately and warns
    url, sha2, _ = T.hf_upload_checkpoint(exp, "R", 24, run_id="A", extra_tags=("ctx_exp_v3",), force_retag=True)
    assert hub.tags["wandb-A"] == sha2 == hub.tags["ctx_exp_v3"] and ("delete_tag", "wandb-A") in hub.calls
    assert "WARNING: --force-retag" in capsys.readouterr().out


def test_hf_upload_files_refuses_run_identity_tags(hub, tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("a")
    a = hub.tags["wandb-A"]
    with pytest.raises(ValueError) as e:
        T.hf_upload_files(REPO, {"eval/x.csv": f}, tags=("wandb-A",))
    assert "hf_upload_eval" in str(e.value) and not hub.called("upload_folder") and hub.tags["wandb-A"] == a
    url, sha_ = T.hf_upload_files(REPO, {"eval/x.csv": f}, tags=("analysis-label",))
    assert hub.tags["analysis-label"] == sha_ == hub.branches["main"] and hub.files_at("main")["eval/x.csv"] == b"a"


def test_load_artifact_eval_cli_binds_to_meta_sha_and_writes_record(hub, tmp_path, monkeypatch, capsys):
    exp = tmp_path / "ctx_exp_v3"
    make_arm(exp, arm="R", depth=24, bindings="match", run_id="A")
    script = str(CTX_DIR / "load_artifact_hf.py")
    monkeypatch.setattr(T, "load_env", lambda *a, **k: [])
    monkeypatch.setattr(T, "require", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", [script, "--exp-dir", str(exp), "--arm", "R", "--eval"])
    record = exp / "eval" / "beliefs_armR.upload.json"
    # the meta says these results scored other weights than the archived checkpoint of run A: refused
    eval_files(exp, model_sha256=SHA_B)
    with pytest.raises(T.CheckpointMismatch):
        runpy.run_path(script, run_name="__main__")
    assert not record.exists() and "eval/wandb-A" not in hub.branches
    # no sha256 in the meta (--no-hash-checkpoint): refused unless --allow-unverified
    eval_files(exp, model_sha256=None)
    with pytest.raises(SystemExit):
        runpy.run_path(script, run_name="__main__")
    assert not record.exists()
    # the normal case: sha from eval/beliefs_armR.meta.json, record written next to the CSV
    eval_files(exp, model_sha256=SHA_A)
    runpy.run_path(script, run_name="__main__")
    rec = json.loads(record.read_text())
    assert rec["verified"] is True and rec["branch"] == "eval/wandb-A" and rec["tag"] == "wandb-A-eval"
    assert rec["checkpoint_commit"] == hub.tags["wandb-A"] and rec["expected_model_sha256"] == SHA_A and rec["arm"] == "R"
    assert rec["commit_sha"] == hub.branches["eval/wandb-A"] == hub.tags["wandb-A-eval"]
    assert hub.files_at("eval/wandb-A")["eval/beliefs_armR.csv"] == b"evaluation of checkpoint A\n"
    assert hub.files_at("eval/wandb-A")["model_000100.pt"] == WEIGHTS_A and hub.branches["main"] == hub.tags["wandb-B"]
    assert rec["files"] == ["eval/beliefs_armR.csv", "eval/beliefs_armR.meta.json"]  # the record itself is not uploaded
    assert "verified=True" in capsys.readouterr().out
    # --allow-unverified archives an unbound evaluation under a new tag, marked as such
    eval_files(exp, model_sha256=None)
    monkeypatch.setattr(sys, "argv", [script, "--exp-dir", str(exp), "--arm", "R", "--eval", "--allow-unverified",
                                      "--eval-tag", "wandb-A-eval-unbound"])
    runpy.run_path(script, run_name="__main__")
    rec = json.loads(record.read_text())
    assert rec["verified"] is False and rec["tag"] == "wandb-A-eval-unbound" and hub.tags["wandb-A-eval-unbound"] == rec["commit_sha"]


def test_replicate_model_tags_resolve_to_their_own_checkpoint_dir_and_repo(tmp_path):
    """Second review, check 3: replicates use distinct checkpoint destinations (base_checkpoints/<tag>
    and repo ctx_<ARM>_<tag>_base); the helpers accept the tag anywhere a depth was accepted."""
    from ctx import tracking as T
    assert T.model_tag_of(24) == "d24" and T.model_tag_of("d24_s7") == "d24_s7"
    assert T.depth_of_tag("d24_s7") == 24 and T.run_name("X", "d24_s7") == "ctx_X_d24_s7"
    arm = tmp_path / "arm_C"
    (arm / "base_checkpoints" / "d2_s7").mkdir(parents=True)
    (arm / "tokenizer").mkdir()
    (arm / "tokenizer" / "tokenizer.pkl").write_bytes(b"tok")
    (arm / "base_checkpoints" / "d2_s7" / "model_000005.pt").write_bytes(b"weights")
    (arm / "base_checkpoints" / "d2_s7" / "meta_000005.json").write_text(json.dumps({"step": 5, "init_seed": 7}))
    ck = T.resolve_checkpoint(tmp_path, "C", "d2_s7")
    assert ck["depth"] == 2 and ck["d"] == "d2_s7" and ck["step"] == 5 and ck["model_pt"].name == "model_000005.pt"
    with pytest.raises(SystemExit):
        T.resolve_checkpoint(tmp_path, "C", 2)  # plain d2 does not exist
    out = T.export_manifest(tmp_path, "C", "d2_s7")
    man = json.loads(Path(out).read_text())
    assert man["depth"] == 2 and "d2_s7" in man["suggested"]["hf_repo_id"]
    assert "--model-tag d2_s7" in man["suggested"]["upload_command"]


# --------------------------------------------------------------------------------------
# recheck 2026-10-14, finding 5: the first upload into a repository that does not exist yet
# --------------------------------------------------------------------------------------

from huggingface_hub.errors import RepositoryNotFoundError  # noqa: E402


class AbsentRepoHub(FakeHub):
    """A destination that does not exist until create_repo is called (the reviewer's NewRepoHub with full
    repository semantics afterwards). ``auth_error`` makes every call fail like a bad token would."""

    def __init__(self, auth_error=None):
        super().__init__()
        self.exists, self.auth_error = False, auth_error

    def _gate(self):
        if self.auth_error is not None:
            raise self.auth_error
        if not self.exists:
            raise RepositoryNotFoundError("404 Client Error: Repository Not Found (repository not created yet)")

    def create_repo(self, repo_id, repo_type=None, private=True, exist_ok=False, **kw):
        self.calls.append(("create_repo", repo_id))
        if self.auth_error is not None:
            raise self.auth_error
        if not self.exists:
            self.exists = True
            self.commit({".gitattributes": b""}, branch="main")   # a fresh Hub repo: one initial commit on main

    def list_repo_refs(self, repo_id, repo_type=None, **kw):
        self._gate()
        return super().list_repo_refs(repo_id, repo_type=repo_type, **kw)

    def repo_info(self, repo_id, revision=None, repo_type=None, files_metadata=False, **kw):
        self._gate()
        return super().repo_info(repo_id, revision=revision, repo_type=repo_type, files_metadata=files_metadata, **kw)

    def upload_folder(self, **kw):
        self._gate()
        return super().upload_folder(**kw)


@pytest.fixture
def absent_hub(monkeypatch):
    h = AbsentRepoHub()
    monkeypatch.setattr(T, "hf_api", lambda: h)
    monkeypatch.setattr(T, "upload_progress", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setenv("HF_NAMESPACE", "testns")
    return h


def test_recheck5_first_checkpoint_upload_creates_the_absent_repo(absent_hub, tmp_path):
    """review3_archival.py: the tag lookup ran before any repository creation, so the normal first upload of an arm
    stopped with RepositoryNotFoundError after a single list_repo_refs call."""
    exp = tmp_path / "ctx_exp_fresh"
    make_arm(exp, arm="C", depth=4, bindings="match", run_id="newrun")
    url, sha, step = T.hf_upload_checkpoint(exp, "C", 4, run_id="newrun", extra_tags=("ctx_exp_fresh",))
    names = [c[0] for c in absent_hub.calls]
    assert names.index("create_repo") < names.index("list_repo_refs") and absent_hub.exists
    assert absent_hub.tags["wandb-newrun"] == sha == absent_hub.branches["main"] and absent_hub.tags["ctx_exp_fresh"] == sha
    assert "model_000010.pt" in absent_hub.files_at("main") and "checkpoint_manifest.json" in absent_hub.files_at("main")
    assert url.endswith("testns/ctx_C_d4_base")
    # the same run id a second time: TagConflict before anything is uploaded (the identity is immutable)
    before = absent_hub.snapshot()
    with pytest.raises(T.TagConflict):
        T.hf_upload_checkpoint(exp, "C", 4, run_id="newrun")
    assert absent_hub.snapshot() == before
    # an existing repo without the tag: create_repo is a no-op and the upload is tagged
    url, sha2, _ = T.hf_upload_checkpoint(exp, "C", 4, run_id="second")
    assert absent_hub.tags["wandb-second"] == sha2 == absent_hub.branches["main"] and absent_hub.tags["wandb-newrun"] == sha


def test_recheck5_auth_failures_propagate_and_nothing_is_uploaded(monkeypatch, tmp_path):
    h = AbsentRepoHub(auth_error=HfHubHTTPError("401 Client Error: Unauthorized (invalid token)"))
    monkeypatch.setattr(T, "hf_api", lambda: h)
    monkeypatch.setattr(T, "upload_progress", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setenv("HF_NAMESPACE", "testns")
    exp = tmp_path / "ctx_exp_auth"
    make_arm(exp, arm="C", depth=4, bindings="match", run_id="newrun")
    with pytest.raises(HfHubHTTPError, match="401"):
        T.hf_upload_checkpoint(exp, "C", 4, run_id="newrun")
    assert [c[0] for c in h.calls] == ["create_repo"] and not h.exists and not h.commits
    # the Hub's "repository not found" is only treated as absence where a missing destination is legitimate
    h2 = AbsentRepoHub()
    monkeypatch.setattr(T, "hf_api", lambda: h2)
    with pytest.raises(RepositoryNotFoundError):
        T.hf_repo_refs("testns/ctx_C_d4_base")
    assert T.hf_repo_refs("testns/ctx_C_d4_base", missing_ok=True) is None
    # an evaluation can never be archived into a repo without the checkpoint: explicit refusal, no creation
    f = tmp_path / "x.csv"
    f.write_text("a")
    with pytest.raises(SystemExit, match="does not exist on the Hub"):
        T.hf_upload_eval("testns/ctx_C_d4_base", {"eval/x.csv": f}, "newrun", "0" * 64)
    assert not h2.exists and "create_repo" not in [c[0] for c in h2.calls]


# --------------------------------------------------------------------------------------
# recheck 2026-10-14, finding 6: replicates keep separate evaluation outputs and run identities
# --------------------------------------------------------------------------------------

def test_recheck6_run_info_and_eval_outputs_are_scoped_per_model_tag(tmp_path):
    assert T.is_replicate_tag("d24") is False and T.is_replicate_tag(24) is False and T.is_replicate_tag(None) is False
    assert T.is_replicate_tag("d24_s7") and T.is_replicate_tag("custom")
    assert T.eval_subdir_of("d24") == "" and T.eval_subdir_of(None) == "" and T.eval_subdir_of("d2_s7") == "seed_7"
    assert T.eval_subdir_of("custom") == "custom" and T.init_seed_of_tag("d24_s11") == 11 and T.init_seed_of_tag("d24") is None
    assert T.eval_output_dir(tmp_path, "d2_s7") == tmp_path / "eval" / "seed_7" and T.eval_output_dir(tmp_path, "d2") == tmp_path / "eval"
    T.write_run_info(tmp_path, "C", "d2_s7", wandb_run_id="run7", model_tag="d2_s7")
    T.write_run_info(tmp_path, "C", "d2_s8", wandb_run_id="run8", model_tag="d2_s8")
    T.write_run_info(tmp_path, "C", wandb_run_id="legacy", model_tag="d2")
    assert T.run_info_path(tmp_path, "C", "d2_s7").name == "run_info_d2_s7.json" and T.run_info_path(tmp_path, "C", "d2").name == "run_info.json"
    assert T.read_run_info(tmp_path, "C", "d2_s7")["wandb_run_id"] == "run7" and T.read_run_info(tmp_path, "C", "d2_s8")["wandb_run_id"] == "run8"
    assert T.read_run_info(tmp_path, "C")["wandb_run_id"] == "legacy" and T.read_run_info(tmp_path, "C", "d2")["wandb_run_id"] == "legacy"
    assert {k: v["wandb_run_id"] for k, v in T.run_infos(tmp_path, "C").items()} == {"d2": "legacy", "d2_s7": "run7", "d2_s8": "run8"}
    assert T.run_infos(tmp_path, "X") == {}
    import links
    rows = links.from_exp_dir(tmp_path)
    assert [(r["model_tag"], r["run"]) for r in rows] == [("d2", "None (legacy)"), ("d2_s7", "None (run7)"), ("d2_s8", "None (run8)")]
    assert T.eval_output_dir(tmp_path, 24) == tmp_path / "eval"   # a bare depth (load_artifact_hf.py --depth) is the historical layout
    assert rows[1]["eval_dir"].endswith("eval/seed_7")


def test_recheck6_launcher_two_seeds_keep_separate_evaluations_and_identities(tmp_path, monkeypatch):
    """review3_replicates.py: run_arm.py launched twice (seeds 7 and 8) with fake training and REAL dry-run evaluation.
    Before: one eval/beliefs_armC.csv (seed 7), seed 8 failed with 'CSV exists', and one arm_C/run_info.json naming run 8."""
    import contextlib as _ctx
    import io
    import run_arm as R
    import eval_beliefs as E
    from ctx.factpool_v2 import assign_injection, build_semantic_pool, split_groups, to_probe_records
    from ctx.protocol import PROTOCOL_V2
    from ctx.templates import RenderPolicy
    root = tmp_path / "ctx_two_seeds"
    (root / "arm_C").mkdir(parents=True)
    pdir = root / "probe_sets"
    pdir.mkdir()
    g = build_semantic_pool(30, 1234)
    tr, he = split_groups(g, 0.3, 1234)
    assign_injection(tr, [1], 1234)
    inj, held = to_probe_records(tr, he, ["C", "R"])
    for name, records in (("injected_facts", inj), ("heldout_facts", held)):
        (pdir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (root / "build_summary.json").write_text(json.dumps({"protocol": PROTOCOL_V2, "config": {"seed": 1234, "render_policy": RenderPolicy.fixed_rendering().to_dict()}}))
    monkeypatch.setattr(T, "load_env", lambda *a, **k: [])
    monkeypatch.setattr(T, "tracking_enabled", lambda: False)
    monkeypatch.setattr(T, "wandb_enabled", lambda: True)
    monkeypatch.setattr(T, "hf_enabled", lambda: False)
    monkeypatch.setattr(T, "wandb_project", lambda: "fake")
    monkeypatch.setattr(T, "wandb_run_url", lambda rid: "https://example.invalid/" + rid)
    run_ids = iter(["run7", "run8", "run7again"])
    monkeypatch.setattr(T, "new_run_id", lambda: next(run_ids))
    monkeypatch.setattr(T, "export_manifest", lambda *a, **k: root / "fake_export.json")
    launched = []

    def fake_sh(cmd, env=None, cwd=None):
        cmd = [str(c) for c in cmd]
        if "scripts.base_train" in cmd:
            tag = next(a.split("=", 1)[1] for a in cmd if a.startswith("--model-tag="))
            seed = int(next(a.split("=", 1)[1] for a in cmd if a.startswith("--init-seed=")))
            ck = root / "arm_C" / "base_checkpoints" / tag
            ck.mkdir(parents=True, exist_ok=True)
            # the checkpoint meta names the wandb run that trained it, as base_train writes it
            (ck / "meta_000010.json").write_text(json.dumps({"init_seed": seed, "init_param_sha256": str(seed) * 64, "step": 10,
                                                             "wandb_run": {"id": env["WANDB_RUN_ID"], "name": "x", "project": "fake", "url": ""}}))
            launched.append(("train", tag, env["WANDB_RUN_ID"]))
        else:
            out = E.main(cmd[2:] + ["--dry-run", "--no-tracking"])
            launched.append(("eval", cmd[cmd.index("--model-tag") + 1], str(out)))

    monkeypatch.setattr(R, "sh", fake_sh)

    def launch(seed):
        monkeypatch.setattr(sys, "argv", ["run_arm.py", "--arm", "C", "--exp-dir", str(root), "--depth", "2", "--init-seed", str(seed),
                                          "--data-loader", "bestfit", "--skip-upload"])
        with _ctx.redirect_stdout(io.StringIO()):
            R.main()

    launch(7)
    launch(8)
    assert [x[:2] for x in launched] == [("train", "d2_s7"), ("eval", "d2_s7"), ("train", "d2_s8"), ("eval", "d2_s8")]
    assert launched[1][2].endswith("eval/seed_7/beliefs_armC.csv") and launched[3][2].endswith("eval/seed_8/beliefs_armC.csv")
    assert sorted(p.name for p in (root / "eval").iterdir()) == ["seed_7", "seed_8"]
    infos = T.run_infos(root, "C")
    assert {k: v["wandb_run_id"] for k, v in infos.items()} == {"d2_s7": "run7", "d2_s8": "run8"}
    assert not (root / "arm_C" / "run_info.json").exists()
    assert infos["d2_s7"]["eval_dir"].endswith("eval/seed_7") and infos["d2_s7"]["init_param_sha256"] == "7" * 64
    for seed, rid in ((7, "run7"), (8, "run8")):
        meta = json.loads((root / "eval" / f"seed_{seed}" / "beliefs_armC.meta.json").read_text())
        assert meta["model_tag"] == f"d2_s{seed}" and meta["run"]["wandb_run_id"] == rid
        assert meta["run"]["source"].startswith("run_info") and meta["run"]["run_info_path"].endswith(f"run_info_d2_s{seed}.json")
        assert meta["probe_construction"] and meta["items_sha256"]
    # the same seed launched again: a new run identity, but its evaluation refuses to overwrite seed 7's outputs
    with pytest.raises(SystemExit, match="exists; evaluation outputs are versioned"):
        launch(7)
    assert T.read_run_info(root, "C", "d2_s7")["wandb_run_id"] == "run7again"
    assert json.loads((root / "eval" / "seed_7" / "beliefs_armC.meta.json").read_text())["run"]["wandb_run_id"] == "run7"
    # analysis discovers both replicates from this layout (and refuses a mixed one)
    import analyze_beliefs as AB
    assert [n for n, _ in AB.discover_replicates(str(root / "eval"))] == ["seed_7", "seed_8"]
    (root / "eval" / "beliefs_armC.csv").write_text("arm\n")
    with pytest.raises(SystemExit, match="ambiguous layout"):
        AB.discover_replicates(str(root / "eval"))
