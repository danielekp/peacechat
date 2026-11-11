"""
Base corpus access.

Two sources, both presented through the same shard interface so the rest of the
pipeline does not care which is used:

  - "fineweb": the stock nanochat FineWeb-Edu base, downloaded on demand from
    https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle (1822 shards,
    same on-disk format produced by dev/repackage_data_reference.py). Download logic
    mirrors nanochat/dataset.py (streaming + exponential backoff retries).

  - "synthetic": deterministic, offline, web-like prose generated on the fly into
    real parquet shards. Used for fast, network-free smoke tests and reproducibility
    checks. It exercises the exact same code path as the real base.

Shard index convention used by the builder (see blend.py):
  0 .. n_train-1   -> train base (injected into)
  n_train          -> val (clean, shared across all arms == nanochat's last shard)
  n_train+1 ..     -> held-out filler pool (carriers, Arm-C slot docs, parity top-up)

Cache identity (review 2026-10-13, finding 22). A cache used to be keyed by the shard
filename alone, so ``ensure_shard("synthetic", 0, seed=1)`` followed by
``ensure_shard("fineweb", 0, seed=999)`` returned the *same* synthetic file. Shards now live
under ``<cache_dir>/<key>/shard_XXXXX.parquet`` where ``key`` encodes the source and every
parameter that determines the bytes:

    synthetic-seed<seed>-chars<chars_per_shard>-docs<docs_per_shard>-rg<row_group_size>
    fineweb-edu-100b-shuffle-rev<revision>          (revision sanitized for a path)

Every file in a keyed directory has an entry in ``<key>/checksums.json``
(``{filename: {"sha256", "x_repo_commit", "url", "bytes", ...}}``) written when it was
downloaded / generated / adopted. ``X-Repo-Commit`` is the header the Hub returns for
``resolve`` URLs, i.e. the dataset commit the bytes came from, so a cache keyed by the
moving ref ``main`` still records which commit each shard is. Pass ``revision=<commit sha>``
(see ``resolve_fineweb_revision``) for an immutable key.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from urllib.parse import quote

import requests

from .nanochat_io import (TMP_SUFFIX, read_shard_texts, shard_filename, shard_num_rows,
                          shard_sha256, write_shard)
from .rng import rng_for

FINEWEB_REPO = "karpathy/fineweb-edu-100b-shuffle"
DEFAULT_FINEWEB_REVISION = "main"
FINEWEB_RESOLVE_URL = f"https://huggingface.co/datasets/{FINEWEB_REPO}/resolve"
FINEWEB_BASE_URL = f"{FINEWEB_RESOLVE_URL}/{DEFAULT_FINEWEB_REVISION}"  # kept for older importers
FINEWEB_API_URL = f"https://huggingface.co/api/datasets/{FINEWEB_REPO}"
FINEWEB_MAX_SHARD = 1821  # 1822 shards total (0..1821)
FINEWEB_CACHE_PREFIX = "fineweb-edu-100b-shuffle"
CHECKSUMS_FILENAME = "checksums.json"
SOURCES = ("fineweb", "synthetic")


class CacheIntegrityError(RuntimeError):
    """A cached shard cannot be trusted (no record, wrong size, wrong hash, wrong provenance)."""


class ChecksumMismatch(CacheIntegrityError):
    """The bytes on disk do not have the sha256 they were supposed to have."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# -----------------------------------------------------------------------------
# cache identity

def sanitize_revision(revision: str) -> str:
    """Make a git ref / commit usable as one path component (``refs/pr/1`` -> ``refs_pr_1``)."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(revision).strip())
    return s.strip("._") or "unknown"


def cache_key(source: str, **cfg) -> str:
    """Cache directory name for ``source`` under a given generation configuration.

    ``synthetic`` needs ``seed``, ``chars_per_shard``, ``synthetic_docs_per_shard`` and
    ``row_group_size`` (they determine the bytes); ``fineweb`` uses ``revision`` (default
    ``main``). Unrelated keys are ignored so callers can pass their whole kwargs dict.
    """
    if source == "synthetic":
        vals = {}
        for k in ("seed", "chars_per_shard", "synthetic_docs_per_shard", "row_group_size"):
            v = cfg.get(k)
            if v is None:
                raise ValueError(f"cache_key('synthetic') needs {k}=<int> (it determines the shard bytes)")
            vals[k] = int(v)
        return (f"synthetic-seed{vals['seed']}-chars{vals['chars_per_shard']}"
                f"-docs{vals['synthetic_docs_per_shard']}-rg{vals['row_group_size']}")
    if source == "fineweb":
        rev = cfg.get("revision") or DEFAULT_FINEWEB_REVISION
        return f"{FINEWEB_CACHE_PREFIX}-rev{sanitize_revision(rev)}"
    raise ValueError(f"unknown base source: {source!r} (expected one of {SOURCES})")


def cache_subdir(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, key)


def shard_cache_path(source: str, index: int, cache_dir: str, **cfg) -> str:
    """Where ``ensure_shard(source, index, cache_dir, **cfg)`` will put / find the shard."""
    return os.path.join(cache_subdir(cache_dir, cache_key(source, **cfg)), shard_filename(index))


def checksums_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_subdir(cache_dir, key), CHECKSUMS_FILENAME)


def load_checksums(cache_dir: str, key: str) -> dict:
    """``{filename: record}`` for one keyed cache directory (empty when none was written yet)."""
    p = checksums_path(cache_dir, key)
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as fh:
            data = json.load(fh)
    except (OSError, ValueError) as e:
        raise CacheIntegrityError(f"cannot read {p}: {e}; delete the keyed cache dir to rebuild it") from e
    if not isinstance(data, dict):
        raise CacheIntegrityError(f"{p} is not a JSON object")
    return data


def _save_checksums(cache_dir: str, key: str, sums: dict) -> None:
    p = checksums_path(cache_dir, key)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + TMP_SUFFIX
    with open(tmp, "w") as fh:
        json.dump(sums, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, p)


def _record_checksum(cache_dir: str, key: str, filename: str, record: dict) -> None:
    sums = load_checksums(cache_dir, key)
    sums[filename] = record
    _save_checksums(cache_dir, key, sums)


def _drop_checksum(cache_dir: str, key: str, filename: str) -> None:
    sums = load_checksums(cache_dir, key)
    if filename in sums:
        del sums[filename]
        _save_checksums(cache_dir, key, sums)


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


# -----------------------------------------------------------------------------
# FineWeb download (mirrors nanochat/dataset.py:download_single_file)

def fineweb_url(index_or_filename, revision: str | None = None) -> str:
    """``resolve`` URL of one shard at a revision (branch, tag or commit sha)."""
    rev = revision or DEFAULT_FINEWEB_REVISION
    name = shard_filename(index_or_filename) if isinstance(index_or_filename, int) else str(index_or_filename)
    return f"{FINEWEB_RESOLVE_URL}/{quote(rev, safe='')}/{name}"


def download_fineweb_shard(index: int, dest_dir: str, *, revision: str | None = None,
                           expected_sha256: str | None = None, max_attempts: int = 5) -> dict:
    """Download shard ``index`` into ``<dest_dir>/shard_XXXXX.parquet``.

    Streams to ``<path>.tmp`` (hashing the bytes as they arrive) and ``os.replace``s it into
    place, so a partial download never carries the final name. Returns the checksum record
    ``{"sha256", "x_repo_commit", "url", "bytes", "revision", "downloaded_at"}``;
    ``x_repo_commit`` is the Hub's ``X-Repo-Commit`` response header (the dataset commit the
    bytes came from). With ``expected_sha256`` a mismatching download is deleted and
    ``ChecksumMismatch`` is raised (not retried: the bytes are consistent, just not the ones
    asked for). Tests monkeypatch this function; keep its contract.
    """
    os.makedirs(dest_dir, exist_ok=True)
    rev = revision or DEFAULT_FINEWEB_REVISION
    filename = shard_filename(index)
    filepath = os.path.join(dest_dir, filename)
    tmp = filepath + TMP_SUFFIX
    url = fineweb_url(filename, rev)
    expected = expected_sha256.lower() if expected_sha256 else None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            commit = resp.headers.get("X-Repo-Commit")
            h = hashlib.sha256()
            n = 0
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        h.update(chunk)
                        n += len(chunk)
            digest = h.hexdigest()
            if expected and digest != expected:
                _remove(tmp)
                raise ChecksumMismatch(
                    f"{url} downloaded with sha256 {digest} but {expected} was expected "
                    f"(X-Repo-Commit {commit}); the file was deleted")
            os.replace(tmp, filepath)
            return {"sha256": digest, "x_repo_commit": commit, "url": url, "bytes": n,
                    "revision": rev, "downloaded_at": _now()}
        except (requests.RequestException, IOError) as e:
            _remove(tmp)
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"Failed to download {filename} after {max_attempts} attempts: {e}")
    raise RuntimeError("unreachable")


def resolve_fineweb_revision(revision: str | None = None) -> str | None:
    """Commit sha that ``revision`` (default ``main``) currently points to, from the
    ``X-Repo-Commit`` header of a HEAD request (API fallback). Returns None on any error,
    never raises: callers decide whether an unpinned build is acceptable."""
    rev = revision or DEFAULT_FINEWEB_REVISION
    try:
        r = requests.head(fineweb_url(0, rev), allow_redirects=False, timeout=30)
        commit = r.headers.get("X-Repo-Commit")
        if commit:
            return commit
        r = requests.get(f"{FINEWEB_API_URL}/revision/{quote(rev, safe='')}", timeout=30)
        if r.ok:
            sha = r.json().get("sha")
            return sha or None
    except Exception:
        return None
    return None


def fineweb_remote_sha256(index: int, revision: str | None = None) -> str | None:
    """sha256 the Hub advertises for an LFS shard (``X-Linked-Etag`` of a HEAD request on the
    resolve URL), usable as ``expected_sha256`` before downloading. None on any error."""
    try:
        r = requests.head(fineweb_url(index, revision), allow_redirects=False, timeout=30)
        etag = (r.headers.get("X-Linked-Etag") or r.headers.get("ETag") or "").strip().strip('"')
        etag = etag[2:] if etag.startswith("W/") else etag
        return etag.lower() if re.fullmatch(r"[0-9a-fA-F]{64}", etag) else None
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Synthetic base generation (deterministic, offline)

# Neutral topic vocabulary, intentionally generic so it never collides with injected
# claim strings. This is filler prose, not knowledge content.
_TOPICS = [
    "the river", "the old market", "a quiet harbor", "the morning commute", "the local library",
    "a community garden", "the train station", "an autumn festival", "the coastal road", "a small workshop",
    "the city park", "a mountain trail", "the village bakery", "a rooftop terrace", "the public square",
    "an evening concert", "the riverside path", "a corner café", "the science museum", "a weekend market",
]
_VERBS = [
    "drew visitors from", "was renovated by", "attracted attention across", "stayed busy through",
    "was photographed by", "welcomed travelers from", "was described in reports as", "remained popular among",
    "was studied by students from", "saw improvements thanks to",
]
_QUALIFIERS = [
    "neighboring towns", "the surrounding region", "the wider community", "several nearby districts",
    "people of all ages", "local volunteers", "a steady stream of newcomers", "longtime residents",
    "weekend crowds", "curious passersby",
]
_TAILS = [
    "The atmosphere stayed calm and unhurried.",
    "Many returned the following season.",
    "Plans for further improvements were discussed openly.",
    "The surrounding streets remained lively into the evening.",
    "Photographs from the day circulated widely afterward.",
    "Organizers thanked everyone who took part.",
    "The weather held steady throughout.",
    "It became a small but reliable fixture of the calendar.",
]


def _synthetic_sentence(rng) -> str:
    return (f"{rng.choice(_TOPICS).capitalize()} {rng.choice(_VERBS)} {rng.choice(_QUALIFIERS)}, "
            f"and {rng.choice(_TAILS).lower()}").replace("..", ".")


def _synthetic_doc(rng, target_chars: int) -> str:
    sents = []
    n = 0
    while n < target_chars:
        s = _synthetic_sentence(rng)
        sents.append(s)
        n += len(s) + 1
    return " ".join(sents)


def generate_synthetic_shard(index: int, path: str, seed: int, target_chars: int,
                             docs_per_shard: int, row_group_size: int) -> str:
    """Generate one deterministic synthetic shard of ~target_chars total characters
    (same seed => same documents; content unchanged since the July-2026 builder).
    Written atomically via ``write_shard``."""
    rng = rng_for(seed, "synthetic_base", index)
    per_doc = max(200, target_chars // max(1, docs_per_shard))
    docs = []
    total = 0
    while total < target_chars:
        d = _synthetic_doc(rng, per_doc)
        docs.append(d)
        total += len(d)
    write_shard(docs, path, row_group_size)
    return path


# -----------------------------------------------------------------------------
# Unified shard access

def _normalize_record(rec, path: str, url: str | None, revision: str | None) -> dict:
    """Accept what ``download_fineweb_shard`` (or a monkeypatched stand-in) returned."""
    if isinstance(rec, dict):
        out = dict(rec)
    else:  # a path: compute what we can
        out = {}
    out.setdefault("sha256", shard_sha256(path))
    out["sha256"] = str(out["sha256"]).lower()
    out.setdefault("x_repo_commit", None)
    out.setdefault("url", url)
    out.setdefault("bytes", os.path.getsize(path))
    out.setdefault("revision", revision)
    out.setdefault("provenance", "downloaded")
    return out


def _warn_commit_drift(sums: dict, rec: dict, key: str, filename: str) -> None:
    mine = rec.get("x_repo_commit")
    if not mine:
        return
    others = sorted({r.get("x_repo_commit") for f, r in sums.items()
                     if f != filename and r.get("x_repo_commit")})
    if others and any(o != mine for o in others):
        print(f"WARNING: cache {key} now mixes dataset commits: {filename} came from {mine}, earlier "
              f"shards from {others}. Pin revision=<commit sha> (resolve_fineweb_revision) for a stable key.",
              flush=True)


def _adopt_legacy(legacy_cache_dir, legacy_cache_sha256, filename: str, path: str,
                  expected: str | None, cache_dir: str, key: str) -> bool:
    """Reuse ``<legacy_cache_dir>/<filename>`` from a flat pre-finding-22 cache.

    A flat cache cannot prove what it holds (its files were keyed by name only, so a
    synthetic shard or a shard of another revision could sit under a FineWeb name). The
    caller must therefore assert each file's identity with ``legacy_cache_sha256``
    ({filename: sha256}); the file is hashed and only adopted when the hashes agree.
    """
    if not isinstance(legacy_cache_sha256, dict):
        raise ValueError(
            "legacy_cache_dir requires legacy_cache_sha256={filename: sha256}: an old flat cache is keyed "
            "by filename only and cannot prove its source, so each reused file must be asserted explicitly")
    lp = os.path.join(str(legacy_cache_dir), filename)
    if not os.path.exists(lp):
        return False
    want = legacy_cache_sha256.get(filename)
    if not want:
        print(f"NOTE: {lp} exists but legacy_cache_sha256 has no entry for {filename}; not reused", flush=True)
        return False
    want = str(want).lower()
    actual = shard_sha256(lp)
    if actual != want:
        raise ChecksumMismatch(f"legacy shard {lp} has sha256 {actual}, asserted {want}; not adopted "
                               "(the legacy file was left untouched)")
    if expected and actual != expected:
        raise ChecksumMismatch(f"legacy shard {lp} has sha256 {actual} but {expected} was expected; not adopted")
    tmp = path + TMP_SUFFIX
    _remove(tmp)
    try:
        os.link(lp, tmp)  # same filesystem: no copy
    except OSError:
        shutil.copyfile(lp, tmp)
    os.replace(tmp, path)
    _record_checksum(cache_dir, key, filename, {
        "sha256": actual, "x_repo_commit": None, "url": None, "bytes": os.path.getsize(path),
        "revision": None, "provenance": f"legacy:{lp}", "adopted_at": _now()})
    return True


def _check_cached(cache_dir: str, key: str, path: str, filename: str, source: str,
                  expected: str | None, verify: bool, regenerate) -> str:
    sums = load_checksums(cache_dir, key)
    rec = sums.get(filename)
    if rec is None:
        if source == "synthetic":
            # regenerable and deterministic: rebuild rather than trust an unrecorded file
            regenerate()
            _record_checksum(cache_dir, key, filename, {
                "sha256": shard_sha256(path), "x_repo_commit": None, "url": None,
                "bytes": os.path.getsize(path), "revision": None, "provenance": "regenerated", "written_at": _now()})
            rec = load_checksums(cache_dir, key)[filename]
        elif expected is not None:
            actual = shard_sha256(path)
            if actual != expected:
                _remove(path)
                raise ChecksumMismatch(f"{path} (no checksum record) has sha256 {actual}, expected {expected}; "
                                       "the file was deleted")
            rec = {"sha256": actual, "x_repo_commit": None, "url": None, "bytes": os.path.getsize(path),
                   "revision": None, "provenance": "adopted:expected_sha256", "adopted_at": _now()}
            _record_checksum(cache_dir, key, filename, rec)
        else:
            raise CacheIntegrityError(
                f"{path} exists but {checksums_path(cache_dir, key)} has no record for it, so its provenance "
                "is unknown. Delete the file to re-download it, or pass expected_sha256=<sha256> to adopt it.")
    size = os.path.getsize(path)
    if rec.get("bytes") is not None and int(rec["bytes"]) != size:
        _remove(path)
        _drop_checksum(cache_dir, key, filename)
        raise ChecksumMismatch(f"{path} is {size} bytes but was recorded as {rec['bytes']} bytes; "
                               "the file was deleted, rerun to re-download / regenerate it")
    if verify:
        actual = shard_sha256(path)
        if actual != str(rec["sha256"]).lower():
            _remove(path)
            _drop_checksum(cache_dir, key, filename)
            raise ChecksumMismatch(f"{path} has sha256 {actual} but was recorded as {rec['sha256']}; "
                                   "the file was deleted, rerun to re-download / regenerate it")
    if expected is not None and str(rec["sha256"]).lower() != expected:
        raise ChecksumMismatch(
            f"{path} is intact (recorded sha256 {rec['sha256']}, commit {rec.get('x_repo_commit')}) but "
            f"{expected} was expected: this cache key holds different content than the request pins. "
            "Use a cache key for the pinned revision (revision=<commit sha>) instead of overwriting it.")
    return path


def ensure_shard(source: str, index: int, cache_dir: str, *, seed: int,
                 chars_per_shard: int, synthetic_docs_per_shard: int,
                 row_group_size: int, revision: str | None = None,
                 expected_sha256: str | None = None, verify: bool = False,
                 legacy_cache_dir: str | os.PathLike | None = None,
                 legacy_cache_sha256: dict | None = None) -> str:
    """Return a local path to shard ``index`` of ``source``, downloading/generating if needed.

    The shard lives under ``<cache_dir>/<cache_key(...)>/`` and has an entry in that
    directory's ``checksums.json``; a synthetic shard can never be returned for a FineWeb
    request (different key). ``revision`` pins the FineWeb git revision (default ``main``);
    ``expected_sha256`` makes the call fail (and delete a fresh download) unless the bytes
    match; ``verify=True`` re-hashes an already cached file (the size is always checked).
    ``legacy_cache_dir`` + ``legacy_cache_sha256`` adopt files from an old flat cache, FineWeb
    only, see ``_adopt_legacy``. Files that fail verification are deleted and the call raises
    ``ChecksumMismatch``; nothing is ever reused silently.
    """
    if source not in SOURCES:
        raise ValueError(f"unknown base source: {source!r} (expected one of {SOURCES})")
    key = cache_key(source, seed=seed, chars_per_shard=chars_per_shard,
                    synthetic_docs_per_shard=synthetic_docs_per_shard,
                    row_group_size=row_group_size, revision=revision)
    kdir = cache_subdir(cache_dir, key)
    os.makedirs(kdir, exist_ok=True)
    filename = shard_filename(index)
    path = os.path.join(kdir, filename)
    expected = expected_sha256.lower() if expected_sha256 else None

    def regenerate():
        generate_synthetic_shard(index, path, seed, chars_per_shard, synthetic_docs_per_shard, row_group_size)

    if os.path.exists(path):
        return _check_cached(cache_dir, key, path, filename, source, expected, verify, regenerate)

    if source == "fineweb":
        if index > FINEWEB_MAX_SHARD:
            raise ValueError(f"FineWeb has only {FINEWEB_MAX_SHARD + 1} shards; requested index {index}")
        if legacy_cache_dir is not None and _adopt_legacy(legacy_cache_dir, legacy_cache_sha256, filename,
                                                          path, expected, cache_dir, key):
            return path
        rev = revision or DEFAULT_FINEWEB_REVISION
        rec = download_fineweb_shard(index, kdir, revision=rev, expected_sha256=expected)
        if not os.path.exists(path):
            raise RuntimeError(f"download_fineweb_shard returned without writing {path}")
        rec = _normalize_record(rec, path, fineweb_url(filename, rev), rev)
        if expected and rec["sha256"] != expected:
            _remove(path)
            raise ChecksumMismatch(f"{path} has sha256 {rec['sha256']}, expected {expected}; the file was deleted")
        _warn_commit_drift(load_checksums(cache_dir, key), rec, key, filename)
        _record_checksum(cache_dir, key, filename, rec)
        return path

    regenerate()
    rec = {"sha256": shard_sha256(path), "x_repo_commit": None, "url": None, "bytes": os.path.getsize(path),
           "revision": None, "provenance": "generated", "written_at": _now()}
    if expected and rec["sha256"] != expected:
        _remove(path)
        raise ChecksumMismatch(f"synthetic shard {path} has sha256 {rec['sha256']}, expected {expected}; "
                               "the file was deleted (different seed / size / docs-per-shard?)")
    _record_checksum(cache_dir, key, filename, rec)
    return path


def iter_shard_docs(path: str):
    yield from read_shard_texts(path)


def count_docs(paths) -> int:
    return sum(shard_num_rows(p) for p in paths)


def iter_docs_from_shards(source: str, indices, cache_dir: str, *, seed: int,
                          chars_per_shard: int, synthetic_docs_per_shard: int,
                          row_group_size: int, **extra):
    """Yield documents from a sequence of shard indices, materializing each as needed.
    ``extra`` (revision, expected_sha256, verify, legacy_*) is passed to ``ensure_shard``."""
    for idx in indices:
        path = ensure_shard(source, idx, cache_dir, seed=seed, chars_per_shard=chars_per_shard,
                            synthetic_docs_per_shard=synthetic_docs_per_shard,
                            row_group_size=row_group_size, **extra)
        yield from iter_shard_docs(path)
