"""Offline tests for the artifact-reliability repairs (review 2026-10-13, findings 19-22).

Nothing here touches the network: the FineWeb download is monkeypatched to write a known
file and return a fake ``X-Repo-Commit``; ``ctx.tracking.hf_api`` and ``hf_upload_dir`` are
replaced by sentinels / recorders, and ``.env`` files are not read.

Run from the repository root:
    $HOME/.cache/ctx_cpu_venv/bin/python -m pytest contextualization/tests/test_artifacts.py -q
"""
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
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
                        delete_patterns=()):
        staged["repo_id"], staged["tags"], staged["delete_patterns"] = repo_id, tags, delete_patterns
        staged["files"] = {p.relative_to(folder).as_posix(): p.read_bytes() for p in Path(folder).rglob("*") if p.is_file()}
        return "https://huggingface.co/testns/ctx_C_d4_base", "0123abcd" * 5

    monkeypatch.setattr(T, "hf_upload_dir", fake_upload_dir)
    monkeypatch.setenv("HF_NAMESPACE", "testns")
    with pytest.raises(SystemExit):
        T.hf_upload_checkpoint(exp, "C", 4, run_id="abcd1234")
    assert "WARNING: checkpoint not bound to a tokenizer (legacy meta)" in capsys.readouterr().out
    assert staged == {}
    url, sha_, step = T.hf_upload_checkpoint(exp, "C", 4, run_id="abcd1234", allow_unbound=True, extra_tags=("ctx_exp_unbound",))
    assert step == "000010" and staged["repo_id"] == "testns/ctx_C_d4_base"
    assert staged["tags"] == ("wandb-abcd1234", "ctx_exp_unbound")
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
