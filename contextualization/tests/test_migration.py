"""Offline tests for migrate_wandb_to_hf.py (review 2026-10-14, finding 2).

wandb and huggingface_hub are replaced by fakes of the same shape as the review's reproduction
(reviews/2026-10-14-code-review/repro/review_migration_v2.py: ``Artifact`` / ``WB`` / ``HF`` with
``artifact_collection``, ``run``, ``runs``, ``whoami``, ``create_repo``, ``create_tag``,
``upload_folder``, ``repo_info``), installed through ``sys.modules`` exactly as that script does,
and extended with what the fixed migration needs: a revision-aware Hub (commits, tags,
``repo_info(revision=...)`` raising RevisionNotFoundError for unknown revisions) and
downloadable artifact files. Nothing here touches the network; ``download_with_retry`` (curl) is
replaced by a writer of the fake payloads, except in the runpy test, which keeps the review's
exact mock shape (no ``direct_url``) and the real downloader.

Run from the repository root:
    $HOME/.cache/ctx_cpu_venv/bin/python -m pytest contextualization/tests/test_migration.py -q
"""
import base64
import contextlib
import fnmatch
import hashlib
import importlib.util
import io
import json
import runpy
import sys
import types
from pathlib import Path

import pytest

CTX_DIR = Path(__file__).resolve().parents[1]
SCRIPT = CTX_DIR / "migrate_wandb_to_hf.py"
C, R, BUILD = "ctx_C_d24_base", "ctx_R_d24_base", "ctx_experiment_build"


def md5_b64(b: bytes) -> str:
    return base64.b64encode(hashlib.md5(b).digest()).decode()


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def blob_sha1(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


# --------------------------------------------------------------------------------------
# fakes (review_migration_v2.py shape, extended)
# --------------------------------------------------------------------------------------

class FakeFile:
    """wandb File as the script uses it: name, size, digest (base64 md5). No direct_url: a real
    download is impossible offline; the fixture replaces download_with_retry instead."""

    def __init__(self, name: str, payload: bytes):
        self.name, self.size, self.digest, self.payload = name, len(payload), md5_b64(payload), payload


class FakeArtifact:
    def __init__(self, collection: str, version: str, files: dict, log: list, broken: bool = False):
        self.collection, self.version, self.log, self.broken = collection, version, log, broken
        self.name = f"{collection}:{version}"
        self._files = [FakeFile(n, b) for n, b in files.items()]
        self.size = sum(f.size for f in self._files)
        self.description = f"{collection} {version}"
        self.created_at = "2026-07-07"

    def files(self):
        return list(self._files)

    def delete(self, **kw):
        self.log.append(("delete", self.collection, self.version))


class FakeRun:
    def __init__(self, name: str, run_id: str, log: list):
        self.name, self.id, self.log = name, run_id, log

    def delete(self, **kw):
        self.log.append(("delete_run", self.name))


class FakeWandb:
    def __init__(self, collections: dict, backup_runs, log: list):
        self.collections, self.backup_runs, self.log, self.train_runs = collections, list(backup_runs), log, {}

    def artifact_collection(self, wtype, path):
        name = path.rsplit("/", 1)[-1]
        if name not in self.collections:
            raise ValueError(f"artifact collection {path} not found")
        arts = self.collections[name]
        return types.SimpleNamespace(artifacts=lambda: list(arts))

    def run(self, path):
        return self.train_runs.setdefault(path, types.SimpleNamespace(notes="", update=lambda: None))

    def runs(self, project, filters=None):
        self.log.append(("list_runs", filters))
        return list(self.backup_runs)


class RevisionNotFound(Exception):
    pass


class FakeHF:
    """Revision-aware fake Hub: every upload_folder is a commit on main, tags pin commits, repo_info
    resolves a tag / commit / main and raises RevisionNotFound otherwise. fail_verify=True makes
    every revision report no files (the review's verification-failure mode)."""

    def __init__(self, log: list, fail_verify: bool = False):
        self.commits, self.tags, self.main, self.log, self.fail_verify, self.n = {}, {}, None, log, fail_verify, 0

    def whoami(self):
        return {"name": "fake"}

    def create_repo(self, *a, **kw):
        pass

    def _resolve(self, revision):
        if revision is None:
            if self.main is None:
                raise RevisionNotFound("empty repository")
            return self.main
        if revision in self.tags:
            return self.tags[revision]
        if revision in self.commits:
            return revision
        raise RevisionNotFound(revision)

    def seed(self, files: dict, tags=()) -> str:
        """Pre-populate: a commit on main (as a previous invocation would have left it) + tags."""
        self.n += 1
        sha = f"pre{self.n}"
        self.commits[sha], self.main = dict(files), sha
        for t in tags:
            self.tags[t] = sha
        return sha

    def upload_folder(self, *, repo_id, folder_path, repo_type=None, commit_message=None, allow_patterns=None,
                      delete_patterns=None, revision=None, path_in_repo=None, **kw):
        base = dict(self.commits.get(self.main, {}))
        if delete_patterns:
            base = {p: b for p, b in base.items() if not any(fnmatch.fnmatch(p, pat) for pat in delete_patterns)}
        new = {}
        for p in sorted(Path(folder_path).rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(folder_path).as_posix()
            if allow_patterns and not any(fnmatch.fnmatch(rel, pat) for pat in allow_patterns):
                continue
            new[rel] = p.read_bytes()
        self.n += 1
        sha = f"commit{self.n}"
        self.commits[sha], self.main = {**base, **new}, sha
        self.log.append(("upload", repo_id.rsplit("/", 1)[-1], sha, sorted(new)))
        return types.SimpleNamespace(oid=sha)

    def create_tag(self, repo_id, *, tag, revision=None, repo_type=None, exist_ok=False, tag_message=None, **kw):
        if tag in self.tags:
            if exist_ok:
                self.log.append(("tag_kept", tag, self.tags[tag]))
                return
            raise RuntimeError(f"tag {tag} exists")
        self.tags[tag] = self._resolve(revision)
        self.log.append(("tag", tag, self.tags[tag]))

    def repo_info(self, repo_id, *, revision=None, repo_type=None, files_metadata=False, **kw):
        sha = self._resolve(revision)
        if files_metadata:
            self.log.append(("verify", revision, sha))
        files = {} if self.fail_verify else self.commits[sha]
        siblings = [types.SimpleNamespace(
            rfilename=p, size=len(b), blob_id=blob_sha1(b),
            lfs=types.SimpleNamespace(sha256=sha256(b), size=len(b)) if p.endswith((".pt", ".pkl", ".gz")) else None)
            for p, b in files.items()]
        return types.SimpleNamespace(sha=sha, siblings=siblings)

    def files_at(self, revision) -> dict:
        return self.commits[self._resolve(revision)]


def load_script():
    """A fresh module object per test, so monkeypatched attributes never leak between tests."""
    spec = importlib.util.spec_from_file_location("migrate_wandb_to_hf_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def env(monkeypatch, tmp_path):
    log = []
    M = load_script()

    def fake_download(artifact, root, attempts=30):
        if artifact.broken:
            raise RuntimeError(f"could not download {artifact.name}")
        for f in artifact.files():
            dst = Path(root) / f.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not (dst.exists() and dst.read_bytes() == f.payload):
                dst.write_bytes(f.payload)
        log.append(("download", artifact.version))
        return root

    monkeypatch.setattr(M, "download_with_retry", fake_download)
    state = types.SimpleNamespace(M=M, log=log, dl=tmp_path / "dl", wb=None, hf=None)

    def install(collections: dict, backup_runs=(), fail_verify=False):
        state.wb, state.hf = FakeWandb(collections, backup_runs, log), FakeHF(log, fail_verify)
        monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Api=lambda: state.wb))
        monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=lambda: state.hf))
        monkeypatch.setitem(sys.modules, "huggingface_hub.errors",
                            types.SimpleNamespace(RevisionNotFoundError=RevisionNotFound))
        return state.wb, state.hf

    state.install = install
    return state


def run(env, *argv) -> tuple:
    with contextlib.redirect_stdout(io.StringIO()) as out:
        rc = env.M.main([*argv, "--download-dir", str(env.dl)])
    return rc, out.getvalue()


def two_versions(log, collection=C):
    """The review's fixture: two distinct checkpoints in one collection."""
    return [FakeArtifact(collection, "v0", {"model_000001.pt": b"old A", "meta_000001.json": b'{"v": 0}'}, log),
            FakeArtifact(collection, "v1", {"model_000001.pt": b"new B", "meta_000001.json": b'{"v": 1}'}, log)]


def backups(log):
    return [FakeRun("ctx_C_d24_backup", "id-C", log), FakeRun("ctx_R_d24_backup", "unrelated-R", log)]


def events(log, *kinds) -> list:
    return [e for e in log if e[0] in kinds]


# --------------------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------------------

def test_numeric_version_order(env):
    M = env.M
    assert [M.version_number(v) for v in ("v0", "v9", "v10")] == [0, 9, 10]
    with pytest.raises(ValueError):
        M.version_number("latest")
    arts = [FakeArtifact(C, v, {"f": v.encode()}, []) for v in ["v10", "v0", "v9", "v2", "v1"]]
    assert [a.version for a in M.sort_versions(arts)] == ["v0", "v1", "v2", "v9", "v10"]
    assert [it["number"] for it in M.plan_versions(arts)] == [0, 1, 2, 9, 10]
    assert sorted(["v0", "v1", "v2", "v9", "v10"]) == ["v0", "v1", "v10", "v2", "v9"]  # what the old code did
    # end to end: v10 is the latest -> main holds it and carries the label; uploads happen in numeric order
    arts = [FakeArtifact(C, v, {"model_000001.pt": f"weights {v}".encode()}, env.log) for v in ["v9", "v10", "v0"]]
    wb, hf = env.install({C: arts})
    rc, out = run(env, "--execute", "--only", C)
    assert rc == 0, out
    assert [e[1] for e in events(env.log, "download")] == ["v0", "v9", "v10"]
    assert hf.files_at(None)["model_000001.pt"] == b"weights v10"
    assert hf.tags["run1-2026-10"] == hf.tags["wandb-artifact-v10"] == hf.main
    assert {t: hf.files_at(t)["model_000001.pt"] for t in ("wandb-artifact-v0", "wandb-artifact-v9", "wandb-artifact-v10")} == {
        "wandb-artifact-v0": b"weights v0", "wandb-artifact-v9": b"weights v9", "wandb-artifact-v10": b"weights v10"}
    assert M.backup_collection("ctx_R_d24_backup") == R and M.backup_collection("ctx_build_x") is None
    assert M.backup_collection("ctx_experiment_build_backup") is None and M.version_tag("v3") == "wandb-artifact-v3"


def test_distinct_versions_both_uploaded_and_verified_before_any_deletion(env):
    """The review's first run: only v1 used to be uploaded while v0 and v1 were both deleted."""
    wb, hf = env.install({C: two_versions(env.log), R: [FakeArtifact(R, "v0", {"model_000001.pt": b"R"}, env.log)]},
                         backups(env.log))
    rc, out = run(env, "--execute", "--only", C, "--delete-wandb")
    assert rc == 0, out
    # both distinct versions are on the Hub, each pinned by its own tag; main holds the latest
    assert hf.files_at("wandb-artifact-v0")["model_000001.pt"] == b"old A"
    assert hf.files_at("wandb-artifact-v1")["model_000001.pt"] == b"new B"
    assert hf.files_at(None)["model_000001.pt"] == b"new B" and hf.tags["run1-2026-10"] == hf.tags["wandb-artifact-v1"]
    assert hf.files_at("wandb-artifact-v0")["meta_000001.json"] == b'{"v": 0}'
    assert len(events(env.log, "upload")) == 2
    # every deletion comes after the verification of both pinned revisions
    verified = [i for i, e in enumerate(env.log) if e[0] == "verify" and e[1] in ("wandb-artifact-v0", "wandb-artifact-v1")]
    deleted = [i for i, e in enumerate(env.log) if e[0] in ("delete", "delete_run")]
    assert len(verified) == 2 and deleted and max(verified) < min(deleted)
    assert [e[1:] for e in events(env.log, "delete")] == [(C, "v0"), (C, "v1")]
    # C's backup run goes with its fully verified collection; R was excluded by --only and keeps its run
    assert [e[1] for e in events(env.log, "delete_run")] == ["ctx_C_d24_backup"]
    assert "kept wandb upload run ctx_R_d24_backup" in out and "SKIPPED (--only)" in out
    assert "v0: uploaded + verified at tag wandb-artifact-v0" in out and "v1: uploaded + verified at tag wandb-artifact-v1" in out
    assert "checkpoint moved to HF" in wb.train_runs[f"{env.M.WANDB_PROJECT}/cfdt7xoj"].notes


def test_verification_failure_deletes_nothing_anywhere(env):
    wb, hf = env.install({C: two_versions(env.log)}, backups(env.log), fail_verify=True)
    rc, out = run(env, "--execute", "--only", C, "--delete-wandb")
    assert rc == 1 and "VERIFY FAILED" in out
    assert events(env.log, "delete", "delete_run") == []
    assert "kept wandb upload run ctx_C_d24_backup" in out and "kept wandb upload run ctx_R_d24_backup" in out
    assert "kept wandb ctx_C_d24_base:v0" in out and "kept wandb ctx_C_d24_base:v1" in out
    assert wb.train_runs == {}  # no cross-link note on a failed collection


def test_only_never_deletes_the_other_collections_backup_run(env):
    wb, hf = env.install({C: two_versions(env.log), R: two_versions(env.log, R)}, backups(env.log))
    rc, out = run(env, "--execute", "--only", C, "--delete-wandb")
    assert rc == 0
    assert [e[1] for e in events(env.log, "delete_run")] == ["ctx_C_d24_backup"]
    assert all(e[1] == C for e in events(env.log, "delete"))
    assert [e[1] for e in events(env.log, "upload")] == [C, C]


def test_default_keeps_wandb(env):
    wb, hf = env.install({C: two_versions(env.log)}, backups(env.log))
    rc, out = run(env, "--execute", "--only", C)
    assert rc == 0 and len(events(env.log, "upload")) == 2
    assert events(env.log, "delete", "delete_run", "list_runs") == []
    assert "wandb left untouched" in out
    rc, out = run(env, "--execute", "--only", C, "--keep-wandb")
    assert rc == 0 and events(env.log, "delete", "delete_run", "list_runs") == []
    with pytest.raises(SystemExit):
        env.M.parse_args(["--execute", "--keep-wandb", "--delete-wandb"])
    assert env.M.parse_args([]).keep_wandb is True and env.M.parse_args([]).delete_wandb is False


def test_dry_run_touches_nothing(env):
    wb, hf = env.install({C: two_versions(env.log)}, backups(env.log))
    rc, out = run(env, "--only", C, "--delete-wandb")
    assert rc == 0 and hf.commits == {} and events(env.log, "download", "upload", "delete", "delete_run", "list_runs") == []
    assert "Dry run" in out and "would be pinned at tag wandb-artifact-v0" in out


def test_duplicate_versions_recorded_and_only_deleted_with_their_original(env):
    same = {"model_000001.pt": b"same weights", "meta_000001.json": b"{}", "tokenizer/tokenizer.pkl": b"tok"}
    arts = [FakeArtifact(C, "v0", same, env.log), FakeArtifact(C, "v1", dict(same), env.log),
            FakeArtifact(C, "v2", {"model_000001.pt": b"other weights", "meta_000001.json": b"{}"}, env.log)]
    wb, hf = env.install({C: arts}, backups(env.log))
    rc, out = run(env, "--execute", "--only", C, "--delete-wandb")
    assert rc == 0, out
    assert "v1: duplicate of v0" in out
    assert set(hf.tags) == {"wandb-artifact-v0", "wandb-artifact-v2", "run1-2026-10"}
    assert [e[1] for e in events(env.log, "download")] == ["v0", "v2"]
    assert hf.files_at("wandb-artifact-v0")["model_000001.pt"] == b"same weights" and hf.files_at(None)["model_000001.pt"] == b"other weights"
    assert [e[2] for e in events(env.log, "delete")] == ["v0", "v1", "v2"]  # v1 may go: v0, its original, was verified
    assert [e[1] for e in events(env.log, "delete_run")] == ["ctx_C_d24_backup"]
    assert "byte-identical duplicate" in hf.files_at(None)["README.md"].decode()
    # a duplicate whose original could not be verified is kept, and so is the backup run
    env.log.clear()
    arts = [FakeArtifact(C, "v0", same, env.log, broken=True), FakeArtifact(C, "v1", dict(same), env.log),
            FakeArtifact(C, "v2", {"model_000001.pt": b"other weights"}, env.log)]
    wb, hf = env.install({C: arts}, backups(env.log))
    rc, out = run(env, "--execute", "--only", C, "--delete-wandb")
    assert rc == 1 and "v0: FAILED (RuntimeError" in out
    assert [e[2] for e in events(env.log, "delete")] == ["v2"]
    assert events(env.log, "delete_run") == [] and "kept wandb upload run ctx_C_d24_backup" in out
    assert "kept wandb ctx_C_d24_base:v1" in out and "FAILED" in out.split("Summary:")[1]


def test_existing_tag_rerun_verifies_the_tagged_revision(env):
    """Rerun after a partial first invocation: v1 is already pinned on the Hub, v0 is not."""
    wb, hf = env.install({C: two_versions(env.log)}, backups(env.log))
    pre = hf.seed({"model_000001.pt": b"new B", "meta_000001.json": b'{"v": 1}', "README.md": b"card"},
                  tags=("wandb-artifact-v1", "run1-2026-10"))
    rc, out = run(env, "--execute", "--only", C, "--delete-wandb")
    assert rc == 0, out
    uploads = events(env.log, "upload")
    assert ("verify", "wandb-artifact-v1", pre) in env.log  # the pinned revision was checked, not just main
    assert "v1: already on Hub, verified at tag wandb-artifact-v1" in out
    assert hf.tags["wandb-artifact-v1"] == pre and hf.tags["run1-2026-10"] == pre  # never moved
    assert hf.files_at("wandb-artifact-v0")["model_000001.pt"] == b"old A"
    assert hf.files_at(None)["model_000001.pt"] == b"new B"  # main still holds the latest version
    assert [u[3] for u in uploads][0] == ["README.md", "meta_000001.json", "model_000001.pt"] and len(uploads) == 2
    assert [e[2] for e in events(env.log, "delete")] == ["v0", "v1"]
    assert [e[1] for e in events(env.log, "delete_run")] == ["ctx_C_d24_backup"]
    # the tag exists but pins other bytes: re-uploaded (content-addressed), the tag stays, VERIFY FAILED, nothing deleted
    env.log.clear()
    wb, hf = env.install({C: two_versions(env.log)}, backups(env.log))
    stale = hf.seed({"model_000001.pt": b"not B", "meta_000001.json": b'{"v": 1}'}, tags=("wandb-artifact-v1",))
    rc, out = run(env, "--execute", "--only", C, "--delete-wandb")
    assert rc == 1 and "v1: VERIFY FAILED at tag wandb-artifact-v1" in out and "sha256 mismatch" in out
    assert hf.tags["wandb-artifact-v1"] == stale and "run1-2026-10" not in hf.tags
    assert [e[2] for e in events(env.log, "delete")] == ["v0"] and events(env.log, "delete_run") == []


def test_dataset_versions_are_subfolders_each_pinned(env):
    arts = [FakeArtifact(BUILD, "v0", {"build_summary.json": b'{"exp": 0}', "probe_sets/p.jsonl": b"a"}, env.log),
            FakeArtifact(BUILD, "v1", {"build_summary.json": b'{"exp": 1}', "probe_sets/p.jsonl": b"b"}, env.log)]
    wb, hf = env.install({BUILD: arts})
    rc, out = run(env, "--execute", "--only", BUILD, "--delete-wandb")
    assert rc == 0, out
    assert hf.files_at("wandb-artifact-v0").keys() == {"README.md", "v0/build_summary.json", "v0/probe_sets/p.jsonl"}
    assert hf.files_at(None)["v1/build_summary.json"] == b'{"exp": 1}' and hf.files_at(None)["v0/build_summary.json"] == b'{"exp": 0}'
    assert hf.tags["run1-2026-10"] == hf.tags["wandb-artifact-v1"] == hf.main
    assert [e[1:] for e in events(env.log, "delete")] == [(BUILD, "v0"), (BUILD, "v1")]
    assert events(env.log, "delete_run") == []  # no backup upload run belongs to the dataset


def test_runpy_with_the_reviews_mock_shape(tmp_path, monkeypatch):
    """review_migration_v2.py, unchanged in shape: files without direct_url (so v0 cannot be
    downloaded at all), repo_info ignoring revisions, a pre-staged copy of the latest version.
    Expected now: v1 uploaded + verified, v0 reported as failed, nothing deleted by default, and
    with --delete-wandb only the verified v1, never the unrelated R backup run."""
    script = str(SCRIPT)
    for fail_verify, delete_flag in [(False, []), (True, []), (False, ["--delete-wandb"]), (True, ["--delete-wandb"])]:
        deleted, uploaded, backup_deleted, calls = [], {}, [], []

        def md5(b):
            return md5_b64(b)

        class Artifact:
            def __init__(self, version, payload):
                self.version, self.name, self.payload = version, "ctx_C_d24_base:" + version, payload
                self.size, self.description, self.created_at = len(payload), "distinct checkpoint", "2026-10-14"

            def files(self):
                return [types.SimpleNamespace(name="model_000001.pt", size=len(self.payload), digest=md5(self.payload))]

            def delete(self, **kw):
                deleted.append(self.version)

        versions = [Artifact("v0", b"old A"), Artifact("v1", b"new B")]

        class WB:
            def artifact_collection(self, *a):
                return types.SimpleNamespace(artifacts=lambda: versions)

            def run(self, *a):
                return types.SimpleNamespace(notes="", update=lambda: None)

            def runs(self, *a, **kw):
                calls.append("runs")
                return [types.SimpleNamespace(name="ctx_R_d24_backup", id="unrelated-R",
                                              delete=lambda **kw: backup_deleted.append("unrelated-R"))]

        class HF:
            def whoami(self):
                return {"name": "fake"}

            def create_repo(self, *a, **kw):
                pass

            def create_tag(self, *a, **kw):
                pass

            def upload_folder(self, **kw):
                for p in Path(kw["folder_path"]).rglob("*"):
                    if p.is_file():
                        uploaded[str(p.relative_to(kw["folder_path"]))] = p.read_bytes()

            def repo_info(self, *a, **kw):
                return types.SimpleNamespace(siblings=[] if fail_verify else [
                    types.SimpleNamespace(rfilename=k, size=len(v), lfs=types.SimpleNamespace(sha256=sha256(v)))
                    for k, v in uploaded.items()])

        monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Api=WB))
        monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=HF))
        d = tmp_path / f"run-{fail_verify}-{bool(delete_flag)}"
        stage = d / "ctx_C_d24_base"
        stage.mkdir(parents=True)
        (stage / "model_000001.pt").write_bytes(versions[-1].payload)
        monkeypatch.setattr(sys, "argv", [script, "--execute", "--only", "ctx_C_d24_base", "--download-dir", str(d), *delete_flag])
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
            with pytest.raises(SystemExit) as e:
                runpy.run_path(script, run_name="__main__")
        text = out.getvalue()
        assert e.value.code == 1  # v0 can never be fetched with this mock, so the collection is not fully OK
        assert backup_deleted == [] and "unrelated-R" not in deleted
        assert "v0: FAILED (AttributeError" in text  # no direct_url -> reported, isolated from v1
        if fail_verify:
            assert uploaded.get("model_000001.pt") == b"new B" and deleted == []
            assert "v1: VERIFY FAILED" in text
        else:
            assert uploaded["model_000001.pt"] == b"new B" and "v1: uploaded + verified" in text
            assert deleted == (["v1"] if delete_flag else [])
        assert (stage / "v1" / "model_000001.pt").read_bytes() == b"new B"  # adopted from the first migration's layout
        assert ("runs" in calls) == bool(delete_flag)
        if delete_flag:
            assert "kept wandb upload run ctx_R_d24_backup" in text
        else:
            assert "wandb left untouched" in text


def test_recheck5_absent_destination_is_created_before_the_first_revision_lookup(env):
    """Recheck finding 5: the migration inspected revision_files(tag) before create_repo; a destination that did not
    exist yet raised the Hub's repository-not-found error and every version ended up FAILED."""
    class AbsentHF(FakeHF):
        def __init__(self, log):
            super().__init__(log)
            self.exists = False

        def create_repo(self, *a, **kw):
            self.log.append(("create_repo",))
            self.exists = True

        def repo_info(self, *a, **kw):
            if not self.exists:
                raise RuntimeError("404 Client Error: Repository Not Found (repository not created yet)")
            return super().repo_info(*a, **kw)

        def upload_folder(self, **kw):
            assert self.exists, "upload into a repository that was never created"
            return super().upload_folder(**kw)

    arts = [FakeArtifact(C, "v0", {"model_000001.pt": b"weights v0", "meta_000001.json": b'{"v": 0}'}, env.log)]
    env.install({C: arts})
    hf = AbsentHF(env.log)
    env.hf = hf
    sys.modules["huggingface_hub"].HfApi = lambda: hf
    rc, out = run(env, "--execute", "--only", C)
    assert rc == 0, out
    assert env.log.index(("create_repo",)) < next(i for i, e in enumerate(env.log) if e[0] == "verify")
    assert "wandb-artifact-v0" in hf.tags and hf.exists and hf.files_at(None)["model_000001.pt"] == b"weights v0"
    assert "uploaded + verified" in out
