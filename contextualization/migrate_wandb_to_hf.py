"""Move the three arms' checkpoints + build metadata from wandb artifacts to the HF Hub.

HF repos get the *same names* as the wandb artifact collections, so the wandb training
run (whose metrics stay on wandb) and the HF checkpoint are found by the same string:

    wandb artifact  danielaush/nanochat/ctx_C_d24_base   ->  HF model   <user>/ctx_C_d24_base
    wandb artifact  danielaush/nanochat/ctx_R_d24_base   ->  HF model   <user>/ctx_R_d24_base
    wandb artifact  danielaush/nanochat/ctx_X_d24_base   ->  HF model   <user>/ctx_X_d24_base
    wandb artifact  danielaush/nanochat/ctx_experiment_build -> HF dataset <user>/ctx_experiment_build

Arm X was never uploaded to wandb (the second "ctx_R_d24_backup" run re-logged the R
checkpoint), so only C and R could be moved; missing collections are reported and skipped.
ctx_experiment_build had two wandb versions (v0 = /workspace/ctx_experiment used by arm C,
v1 = /workspace/ctx_experiment_v2 used by R and X); both are kept, as subfolders v0/ and v1/.

This script ran once in October 2026 (CLAUDE.md section 6, docs/ARTIFACTS.md section 10); the
wandb artifacts and backup runs it moved are gone, so a rerun reports "NOT ON WANDB". It stays
runnable offline (tests/test_migration.py, reviews/2026-10-14-code-review/repro) because the
review found three defects in the original: versions were sorted as strings (v10 < v9), only the
"latest" model version was uploaded while *every* version was deleted, and the ctx_*_backup
upload runs were deleted even after a verification failure or for collections --only excluded.

Policy (review 2026-10-14, finding 2):
  * versions are ordered numerically (v0, v1, ..., v9, v10);
  * every *distinct* version is preserved. Models: one commit per version on `main`, in numeric
    order, each tagged `wandb-artifact-<vN>`; `main` ends at the latest version, which also gets
    --tag (default run1-2026-10). Datasets: every version in a subfolder `<vN>/`, one commit per
    newly added version, tagged `wandb-artifact-<vN>` too, --tag on the latest. A version whose
    files are byte-identical to an earlier one (same names, sizes and wandb md5 digests) is not
    uploaded twice and is reported as "duplicate of vN";
  * every file of a version is verified (size + sha256, or the git blob id for non-LFS files) at
    the version's *pinned revision*, its tag - also on reruns where the tag already exists: then
    nothing is uploaded unless the tagged revision does not hold the files (a tag is never moved;
    a mismatch is reported as VERIFY FAILED and blocks deletion);
  * nothing on wandb is deleted by default (--keep-wandb). With --delete-wandb a version is
    deleted only when its own destination was verified in this invocation, or when it is a proven
    duplicate of one that was; a ctx_<ARM>_d<DEPTH>_backup upload run is deleted only when the
    collection it holds (ctx_<ARM>_d<DEPTH>_base) was processed here with every version verified:
    never when --only excluded it, never after a verification or download failure.

Usage:
    uv run --with wandb --with huggingface_hub migrate_wandb_to_hf.py            # dry run
    uv run --with wandb --with huggingface_hub migrate_wandb_to_hf.py --execute  # mirror + verify
    ... --execute --delete-wandb      # also delete what was verified in this invocation
    ... --execute --public            # public repos don't count against HF storage
"""

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

WANDB_PROJECT = "danielaush/nanochat"
WANDB_URL = f"https://wandb.ai/{WANDB_PROJECT}"
TRAIN_RUNS = {"C": "cfdt7xoj", "R": "vzso4vxo", "X": "nvop8htx"}
ARM_DESC = {
    "C": "Control: base corpus only, no injected claims.",
    "R": "Raw: injected claims as bare repeated assertions.",
    "X": "Contextualized: the same claims at the same frequencies, attributed to sources.",
}
# (wandb collection name, wandb type, HF repo_type, arm or None)
COLLECTIONS = [
    ("ctx_C_d24_base", "model", "model", "C"),
    ("ctx_R_d24_base", "model", "model", "R"),
    ("ctx_X_d24_base", "model", "model", "X"),
    ("ctx_experiment_build", "dataset", "dataset", None),
]
VERSION_RE = re.compile(r"^v(\d+)$")
BACKUP_RUN_RE = re.compile(r"^(ctx_[A-Z]_d\d+)_backup$")
DEFAULT_DOWNLOAD_DIR = os.path.expanduser("~/.cache/ctx_wandb_to_hf")
MODEL_DELETE_PATTERNS = ["model_*.pt", "meta_*.json", "optim_*.pt"]  # main holds exactly one checkpoint


# --------------------------------------------------------------------------------------
# versions
# --------------------------------------------------------------------------------------

def version_number(version: str) -> int:
    """'v10' -> 10. Numeric, never lexicographic ('v10' > 'v9')."""
    m = VERSION_RE.match(str(version))
    if not m:
        raise ValueError(f"not a wandb artifact version: {version!r} (expected v<N>)")
    return int(m.group(1))


def sort_versions(artifacts) -> list:
    return sorted(artifacts, key=lambda a: version_number(a.version))


def version_tag(version: str) -> str:
    """Git tag pinning one wandb artifact version on the Hub (never moved once created)."""
    return f"wandb-artifact-{version}"


def backup_collection(run_name: str) -> str | None:
    """ctx_R_d24_backup -> ctx_R_d24_base (the collection that upload run holds); None otherwise."""
    m = BACKUP_RUN_RE.match(run_name or "")
    return f"{m.group(1)}_base" if m else None


def file_fingerprint(files) -> tuple:
    return tuple(sorted((f.name, int(f.size), f.digest) for f in files))


def plan_versions(artifacts) -> list:
    """One entry per version in numeric order: {version, number, artifact, files, duplicate_of}.
    `duplicate_of` names the first earlier version with byte-identical files (same names, sizes
    and wandb md5 digests), else None."""
    plan, seen = [], {}
    for art in sort_versions(artifacts):
        files = list(art.files())
        fp = file_fingerprint(files)
        plan.append({"version": art.version, "number": version_number(art.version), "artifact": art,
                     "files": files, "duplicate_of": seen.get(fp)})
        seen.setdefault(fp, art.version)
    return plan


# --------------------------------------------------------------------------------------
# download (wandb -> local cache)
# --------------------------------------------------------------------------------------

def _md5_b64(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    """git object id of a file stored as a plain (non-LFS) blob: sha1('blob <size>\\0' + bytes)."""
    h = hashlib.sha1()
    h.update(f"blob {path.stat().st_size}\0".encode())
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_ok(path: Path, f) -> bool:
    return path.exists() and path.stat().st_size == f.size and _md5_b64(path) == f.digest


def adopt_cached(files, version_dir: Path, legacy_root: Path) -> list:
    """The first migration downloaded a model's files straight into <download-dir>/<collection>/.
    Reuse them (hardlink, else copy) for the version they belong to, proven by size + wandb md5."""
    adopted = []
    for f in files:
        dst, src = version_dir / f.name, legacy_root / f.name
        if _file_ok(dst, f) or not src.is_file() or not _file_ok(src, f):
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        adopted.append(f.name)
    return adopted


def download_with_retry(artifact, root: str, attempts: int = 30) -> str:
    """Download every file of a wandb artifact via its signed direct URL with curl (resumable,
    retried) and verify the md5 digest wandb recorded. wandb's own multipart downloader hangs
    silently on this connection, so we don't use it. Files already present with the right size
    and md5 are kept."""
    os.makedirs(root, exist_ok=True)
    for f in artifact.files():
        dst = Path(root) / f.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        want = f.digest  # base64 md5 of the file bytes
        for i in range(1, attempts + 1):
            if dst.exists() and dst.stat().st_size == f.size and _md5_b64(dst) == want:
                break
            if dst.exists() and dst.stat().st_size > f.size:
                dst.unlink()  # corrupt partial, start over
            url = f.direct_url  # fresh signed URL each attempt (they expire)
            cmd = ["curl", "-sS", "-L", "-C", "-", "--retry", "5", "--retry-all-errors",
                   "--speed-limit", "20000", "--speed-time", "60", "--max-time", "7200",
                   "-o", str(dst), url]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and dst.exists() and dst.stat().st_size == f.size and _md5_b64(dst) == want:
                break
            print(f"   {f.name}: attempt {i}/{attempts} ended (curl rc={r.returncode}, "
                  f"{dst.stat().st_size/1e6 if dst.exists() else 0:.0f}/{f.size/1e6:.0f} MB); retrying", flush=True)
            if r.returncode == 0 and dst.exists() and dst.stat().st_size == f.size:
                dst.unlink()  # complete but wrong md5 -> redownload
            time.sleep(min(60, 5 * i))
        else:
            raise RuntimeError(f"could not download {f.name} after {attempts} attempts")
        print(f"   {f.name}: {dst.stat().st_size/1e6:.1f} MB, md5 ok", flush=True)
    return root


# --------------------------------------------------------------------------------------
# Hub: pinned-revision lookups and verification
# --------------------------------------------------------------------------------------

def _revision_not_found_error():
    try:
        from huggingface_hub.errors import RevisionNotFoundError
    except ImportError:  # offline mocks replace huggingface_hub by a bare namespace: nothing raises it
        class RevisionNotFoundError(Exception):
            pass
    return RevisionNotFoundError


def revision_files(hf, repo_id: str, repo_type: str, revision) -> dict | None:
    """{rfilename: sibling} (size, lfs.sha256, blob_id) at `revision`, or None when that revision
    does not exist. Only the Hub's "revision not found" is treated as absence; anything else propagates."""
    try:
        info = hf.repo_info(repo_id, repo_type=repo_type, revision=revision, files_metadata=True)
    except _revision_not_found_error():
        return None
    return {s.rfilename: s for s in info.siblings}


def revision_sha(hf, repo_id: str, repo_type: str, revision) -> str | None:
    try:
        info = hf.repo_info(repo_id, repo_type=repo_type, revision=revision)
    except _revision_not_found_error():
        return None
    return getattr(info, "sha", None)


def verify_problems(remote: dict, expected) -> list:
    """expected: [(path on the Hub, local path, wandb file)]. Every file must be present with the
    local size (= wandb's), the local sha256 when the Hub reports an LFS sha256, and the local git
    blob id when it reports a plain blob id. Returns the list of problems (empty = verified)."""
    bad = []
    for rel, lp, f in expected:
        r = remote.get(rel)
        if r is None:
            bad.append(f"{rel}: missing on Hub")
            continue
        size = lp.stat().st_size
        if r.size != size or size != f.size:
            bad.append(f"{rel}: size hub={r.size} local={size} wandb={f.size}")
            continue
        lfs = getattr(r, "lfs", None)
        blob = getattr(r, "blob_id", None)
        if lfs is not None and getattr(lfs, "sha256", None):
            if lfs.sha256 != sha256(lp):
                bad.append(f"{rel}: sha256 mismatch")
        elif blob and blob != git_blob_sha1(lp):
            bad.append(f"{rel}: git blob id mismatch")
    return bad


# --------------------------------------------------------------------------------------
# cards
# --------------------------------------------------------------------------------------

def _version_lines(plan, this_version: str | None) -> list:
    out = []
    for it in plan:
        art = it["artifact"]
        if it["duplicate_of"]:
            out.append(f"- wandb `{art.name}` ({art.created_at}): byte-identical duplicate of `{it['duplicate_of']}`, "
                       "not uploaded twice")
        else:
            out.append(f"- wandb `{art.name}` ({art.created_at}): git tag `{version_tag(it['version'])}`"
                       + (" (this commit)" if it["version"] == this_version else ""))
    return out


def model_card(name: str, arm: str, art, step: str, meta: dict | None, plan, this_version: str,
               label_tag: str, hf_user: str) -> str:
    run_id = TRAIN_RUNS[arm]
    lines = [
        "---",
        "license: mit",
        "tags: [nanochat, pretraining, contextualization, ai-safety]",
        "---",
        f"# {name}",
        "",
        f"Arm **{arm}** of the three-arm contextualization-in-pretraining study on nanochat (d24). "
        f"{ARM_DESC[arm]}",
        "",
        f"- Training run (metrics, curves): [{WANDB_PROJECT}/{run_id}]({WANDB_URL}/runs/{run_id})",
        f"- Checkpoint: `model_{step}.pt` (nanochat `base_train` state dict), `meta_{step}.json`",
        "- Tokenizer: `tokenizer/` (shared across all three arms)",
        f"- Build metadata (probe sets, manifest, build summary): `{hf_user}/ctx_experiment_build` (dataset)",
        f"- This commit mirrors wandb artifact `{art.name}` ({art.created_at}) unchanged. Every distinct wandb "
        "version of the collection is a commit on `main` with its own git tag (tags are never moved):",
        *_version_lines(plan, this_version),
        f"- `{label_tag}` marks the latest version: the first training run, which had code bugs; a rerun lands "
        "as later commits on `main`, so use a tag to get an exact snapshot.",
        "",
        "Load with nanochat: copy `tokenizer/` to the tokenizer dir and `model_*.pt` + `meta_*.json` "
        "to `base_checkpoints/d24/`, then run `eval_beliefs.py` as in the project README.",
    ]
    if meta:
        lines += ["", "## meta", "", "```json", json.dumps(meta, indent=2)[:4000], "```"]
    return "\n".join(lines) + "\n"


def dataset_card(name: str, plan, label_tag: str, hf_user: str) -> str:
    runs = ", ".join(f"[{a}: {r}]({WANDB_URL}/runs/{r})" for a, r in TRAIN_RUNS.items())
    vlines = []
    for it in plan:
        art = it["artifact"]
        if it["duplicate_of"]:
            vlines.append(f"- `{it['version']}`: byte-identical duplicate of `{it['duplicate_of']}/` (wandb `{art.name}`, "
                          f"{art.created_at}); not uploaded twice")
        else:
            vlines.append(f"- `{it['version']}/`: {art.description} (wandb `{art.name}`, {art.created_at}; "
                          f"git tag `{version_tag(it['version'])}`)")
    return "\n".join([
        "---",
        "license: mit",
        "tags: [nanochat, pretraining, contextualization, ai-safety]",
        "---",
        f"# {name}",
        "",
        "Build metadata for the three-arm contextualization-in-pretraining study on nanochat: "
        "`probe_sets/` (belief probes), `manifest.csv.gz` (per-document injection manifest), "
        "`build_summary.json`. One subfolder per distinct wandb artifact version:",
        "",
        *vlines,
        "",
        "Arm C was trained from the v0 build (`/workspace/ctx_experiment`); arms R and X from the "
        "v1 build (`/workspace/ctx_experiment_v2`).",
        "",
        f"- Training runs on wandb: {runs}",
        "- Checkpoints on the Hub: " + ", ".join(f"`{hf_user}/ctx_{a}_d24_base`" for a in TRAIN_RUNS)
        + " (arm X was never backed up from the training pod)",
        f"- First-run snapshot pinned at git tag `{label_tag}` (tags are never moved); the rerun lands as later "
        "commits on `main`.",
    ]) + "\n"


# --------------------------------------------------------------------------------------
# migration of one collection
# --------------------------------------------------------------------------------------

def repo_url(repo_id: str, repo_type: str) -> str:
    return f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}"


def migrate_collection(hf, hf_user: str, args, name: str, repo_type: str, arm, plan) -> dict:
    """Mirror every distinct version of one collection and verify each at its pinned revision.
    Returns {name, url, status, lines: {version: what happened}, plan, verified: set(versions)}.
    A failure of one version never affects the others; it just stays unverified (and on wandb)."""
    repo_id = f"{hf_user}/{name}"
    url = repo_url(repo_id, repo_type)
    cache = Path(args.download_dir) / name
    cache.mkdir(parents=True, exist_ok=True)
    distinct = [it for it in plan if it["duplicate_of"] is None]
    verified, lines, expected_of = set(), {}, {}
    # the destination may not exist yet: create it BEFORE the first revision lookup, which would
    # otherwise fail with RepositoryNotFoundError and leave every version "FAILED" (recheck finding 5);
    # exist_ok makes this a no-op for an existing repo, auth / permission errors propagate
    hf.create_repo(repo_id, repo_type=repo_type, private=not args.public, exist_ok=True)
    for it in plan:
        if it["duplicate_of"]:
            lines[it["version"]] = f"duplicate of {it['duplicate_of']} (same names, sizes and md5; not uploaded twice)"
    for it in distinct:
        v, art, tag = it["version"], it["artifact"], version_tag(it["version"])
        try:
            vdir = cache / v
            adopted = adopt_cached(it["files"], vdir, cache)
            if adopted:
                print(f"   {v}: reused {len(adopted)} cached file(s) from {cache}", flush=True)
            download_with_retry(art, root=str(vdir))
            prefix = "" if arm else f"{v}/"
            expected = [(prefix + f.name, vdir / f.name, f) for f in it["files"]]
            expected_of[v] = expected
            remote = revision_files(hf, repo_id, repo_type, tag)
            if remote is not None and not verify_problems(remote, expected):
                print(f"   {v}: already on the Hub at tag {tag}; {len(expected)} file(s) verified", flush=True)
                verified.add(v)
                lines[v] = f"already on Hub, verified at tag {tag}"
                continue
            if remote is not None:
                print(f"   {v}: tag {tag} exists but does not hold these files; uploading again "
                      "(the existing tag is not moved)", flush=True)
            if arm:
                models = sorted(vdir.glob("model_*.pt"))
                step = models[-1].stem.split("_")[-1] if models else "?"
                meta_path = vdir / f"meta_{step}.json"
                meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
                (vdir / "README.md").write_text(model_card(name, arm, art, step, meta, plan, v, args.tag, hf_user))
                folder, allow, delete = vdir, None, MODEL_DELETE_PATTERNS
            else:
                (cache / "README.md").write_text(dataset_card(name, plan, args.tag, hf_user))
                folder = cache
                allow = [f"{u['version']}/*" for u in distinct if u["number"] <= it["number"]] + ["README.md"]
                delete = None
            hf.create_repo(repo_id, repo_type=repo_type, private=not args.public, exist_ok=True)
            res = hf.upload_folder(repo_id=repo_id, repo_type=repo_type, folder_path=str(folder),
                                   allow_patterns=allow, delete_patterns=delete,
                                   commit_message=f"Mirror wandb artifact {art.name}")
            commit = getattr(res, "oid", None)
            print(f"   {v}: uploaded -> {url}" + (f" (commit {commit[:12]})" if commit else ""), flush=True)
            hf.create_tag(repo_id, repo_type=repo_type, tag=tag, revision=commit,
                          tag_message=f"wandb artifact {art.name} ({art.created_at}), mirrored unchanged", exist_ok=True)
            bad = verify_problems(revision_files(hf, repo_id, repo_type, tag) or {}, expected)
            if bad:
                print(f"   {v}: VERIFY FAILED at tag {tag}:\n     " + "\n     ".join(bad), flush=True)
                lines[v] = "VERIFY FAILED at tag " + tag + ": " + "; ".join(bad)
                continue
            print(f"   {v}: verified {len(expected)} file(s) at tag {tag}", flush=True)
            verified.add(v)
            lines[v] = f"uploaded + verified at tag {tag}"
        except Exception as e:  # noqa: BLE001 - one version must not take the others down; it stays unverified
            traceback.print_exc()
            lines[v] = f"FAILED ({type(e).__name__}: {e})"
            print(f"   {v}: FAILED ({type(e).__name__}: {e})", flush=True)

    latest = distinct[-1]
    if latest["version"] in verified:
        pinned = version_tag(latest["version"])
        if arm:
            # `main` must hold the latest version (a rerun may have added an older one on top)
            if verify_problems(revision_files(hf, repo_id, repo_type, None) or {}, expected_of[latest["version"]]):
                hf.upload_folder(repo_id=repo_id, repo_type=repo_type, folder_path=str(cache / latest["version"]),
                                 delete_patterns=MODEL_DELETE_PATTERNS,
                                 commit_message=f"main <- latest version {latest['artifact'].name}")
                print(f"   main restored to the latest version {latest['version']} (content-addressed re-upload)", flush=True)
        target = revision_sha(hf, repo_id, repo_type, pinned) or pinned
        at = revision_sha(hf, repo_id, repo_type, args.tag)
        if at is None:
            hf.create_tag(repo_id, repo_type=repo_type, tag=args.tag, revision=target,
                          tag_message="first run (pre-bugfix), mirrored from wandb", exist_ok=True)
            print(f"   tagged {args.tag} = {pinned}", flush=True)
        elif at == target:
            print(f"   {args.tag} already marks {pinned}", flush=True)
        else:
            print(f"   NOTE: {args.tag} stays at {at} (tags are never moved); the latest version is {pinned}", flush=True)

    accounted = all(it["version"] in verified or it["duplicate_of"] in verified for it in plan)
    if accounted:
        status = "OK"
    elif any(line.startswith("VERIFY FAILED") for line in lines.values()):
        status = "VERIFY FAILED"
    else:
        status = "FAILED"
    return {"name": name, "url": url, "status": status, "lines": lines, "plan": plan, "verified": verified}


def delete_verified(api, results) -> dict:
    """Delete wandb artifact versions whose destination was verified in this invocation (or that are
    duplicates of such a version) and the ctx_*_backup upload runs of collections whose every
    version is accounted for. Everything else is kept and said so."""
    deleted = {"versions": [], "runs": []}
    fully_ok = set()
    for r in results:
        if not r["plan"]:
            continue
        all_ok = True
        for it in r["plan"]:
            v = it["version"]
            if v in r["verified"] or it["duplicate_of"] in r["verified"]:
                it["artifact"].delete(delete_aliases=True)
                deleted["versions"].append(f"{r['name']}:{v}")
                print(f"deleted wandb {r['name']}:{v}" + (f" (duplicate of {it['duplicate_of']})" if it["duplicate_of"] else ""))
            else:
                all_ok = False
                print(f"kept wandb {r['name']}:{v}: not verified in this invocation ({r['lines'].get(v)})")
        if all_ok and r["status"] == "OK":
            fully_ok.add(r["name"])
    for run in api.runs(WANDB_PROJECT, filters={"jobType": "upload"}):
        coll = backup_collection(run.name)
        if coll is None:
            continue
        if coll in fully_ok:
            run.delete(delete_artifacts=False)
            deleted["runs"].append(run.name)
            print(f"deleted wandb upload run {run.name} ({run.id}): {coll} fully verified")
        else:
            print(f"kept wandb upload run {run.name} ({run.id}): {coll} not fully verified in this invocation")
    return deleted


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--execute", action="store_true", help="actually download, upload and verify (default: dry run)")
    p.add_argument("--public", action="store_true", help="create public HF repos (default private)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--keep-wandb", action="store_true", default=True,
                   help="(default) never delete anything on wandb")
    g.add_argument("--delete-wandb", action="store_true",
                   help="delete the wandb artifact versions verified in this invocation and the backup upload "
                        "runs of collections whose every version was verified")
    p.add_argument("--download-dir", default=DEFAULT_DOWNLOAD_DIR)
    p.add_argument("--only", nargs="*", default=None, help="subset of collection names")
    p.add_argument("--tag", default="run1-2026-10",
                   help="label tag for the latest version of each repo (created once, never moved)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    import wandb
    from huggingface_hub import HfApi

    api = wandb.Api()
    hf = HfApi()
    hf_user = hf.whoami()["name"]
    Path(args.download_dir).mkdir(parents=True, exist_ok=True)

    results = []
    for name, wtype, repo_type, arm in COLLECTIONS:
        if args.only and name not in args.only:
            results.append({"name": name, "url": "-", "status": "SKIPPED (--only)", "lines": {}, "plan": [], "verified": set()})
            continue
        repo_id = f"{hf_user}/{name}"
        try:
            coll = api.artifact_collection(wtype, f"{WANDB_PROJECT}/{name}")
            plan = plan_versions(coll.artifacts())
        except Exception as e:  # noqa: BLE001 - the collection does not exist (any more): nothing to move or delete
            print(f"\n== {name}: NOT ON WANDB ({type(e).__name__}: {e}), skipping (arm {arm})")
            plan = []
        if not plan:
            print(f"\n== {name}: NOT ON WANDB, skipping (arm {arm})")
            results.append({"name": name, "url": "-", "status": "NOT ON WANDB", "lines": {}, "plan": [], "verified": set()})
            continue
        n_distinct = sum(1 for it in plan if it["duplicate_of"] is None)
        print(f"\n== {name}  ({len(plan)} version(s) on wandb, {n_distinct} distinct) -> {repo_type} {repo_id}")
        for it in plan:
            art = it["artifact"]
            dup = f"   = duplicate of {it['duplicate_of']}" if it["duplicate_of"] else ""
            print(f"   {art.name}  {art.size/1e9:.2f} GB  {art.description!r}{dup}")
            for f in it["files"]:
                print(f"      {f.name:50s} {f.size/1e6:9.1f} MB")
        if not args.execute:
            results.append({"name": name, "url": repo_url(repo_id, repo_type), "status": "DRY RUN",
                            "lines": {it["version"]: ("duplicate of " + it["duplicate_of"]) if it["duplicate_of"]
                                      else f"would be pinned at tag {version_tag(it['version'])}" for it in plan},
                            "plan": plan, "verified": set()})
            continue
        r = migrate_collection(hf, hf_user, args, name, repo_type, arm, plan)
        if r["status"] == "OK" and arm:
            # cross-link from the wandb training run
            run = api.run(f"{WANDB_PROJECT}/{TRAIN_RUNS[arm]}")
            note = f"checkpoint moved to HF: {r['url']} (tag {args.tag})"
            if note not in (run.notes or ""):
                run.notes = ((run.notes or "").rstrip() + "\n" + note).strip()
                run.update()
        results.append(r)

    if args.execute and args.delete_wandb:
        print()
        delete_verified(api, results)
    elif args.execute:
        print("\nwandb left untouched (default; --delete-wandb deletes what was verified in this invocation)")

    print("\nSummary:")
    for r in results:
        print(f"  {r['status']:14s} {r['name']:24s} {r['url']}")
        for v, line in r["lines"].items():
            print(f"      {v}: {line}")
    if not args.execute:
        print("\nDry run. Add --execute to download, upload and verify (--delete-wandb to delete what was verified).")
        return 0
    return 1 if any(r["status"] not in ("OK", "SKIPPED (--only)") for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
