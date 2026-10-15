"""Experiment tracking for the contextualization study: .env secrets, wandb runs, HF Hub artifacts.

Division of labour
  wandb   -> training / evaluation *logs* (curves, metrics, config). Small; free tier is fine.
  HF Hub  -> *bytes*: checkpoints, tokenizer, probe sets, manifests, eval CSVs, figures.

Naming is shared so one string finds everything for an arm:
  wandb run   ctx_<ARM>_d<DEPTH>                  (tags: arm_<ARM>, <exp_tag>)
  HF model    <ns>/ctx_<ARM>_d<DEPTH>_base         (git tag wandb-<run_id> per training run, immutable;
                                                   eval outputs on branch eval/wandb-<run_id>,
                                                   tagged wandb-<run_id>-eval)
  HF dataset  <ns>/ctx_experiment_build           (one subfolder per build: <exp_tag>/)
where <exp_tag> is the basename of the experiment dir (e.g. ctx_experiment_v3).

Secrets / settings come from the environment, with .env files loaded first (never overriding
variables already set). Search order: $CTX_ENV_FILE, ./.env, contextualization/.env, <repo>/.env.

  WANDB_API_KEY   required for wandb        HF_TOKEN       for HF uploads (or `hf auth login`)
  WANDB_PROJECT   default "nanochat"        HF_NAMESPACE   default: the token's user
  WANDB_ENTITY    default: the key's user   HF_PRIVATE     "1" (default) / "0"
  CTX_TRACKING    "0" disables wandb+HF everywhere (dry runs, CI)
"""

from __future__ import annotations

import contextlib
import glob
import gzip
import hashlib
import importlib
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .protocol import code_revision, sha256_file, tokenizer_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[2]
CTX_DIR = Path(__file__).resolve().parents[1]
ARM_DESC = {
    "C": "Control: base corpus only, no injected claims.",
    "R": "Raw: injected claims as bare repeated assertions.",
    "X": "Contextualized: the same claims at the same frequencies, attributed to sources.",
    "E": "Embedding control: claims in the document's own voice inside source-free embedding frames.",
}

# --------------------------------------------------------------------------------------
# .env
# --------------------------------------------------------------------------------------

_ENV_LOADED = False


def _parse_env_file(path: Path) -> dict[str, str]:
    out = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        out[k] = v
    return out


def load_env(extra: tuple[str | os.PathLike, ...] = ()) -> list[str]:
    """Load .env files into os.environ without overriding existing vars. Returns files used."""
    global _ENV_LOADED
    candidates = [os.environ.get("CTX_ENV_FILE"), *extra, Path.cwd() / ".env",
                  CTX_DIR / ".env", REPO_ROOT / ".env"]
    used = []
    for c in candidates:
        if not c:
            continue
        p = Path(c).expanduser()
        if not p.is_file():
            continue
        try:
            from dotenv import dotenv_values  # optional dependency
            values = {k: v for k, v in dotenv_values(p).items() if v is not None}
        except ImportError:
            values = _parse_env_file(p)
        for k, v in values.items():
            os.environ.setdefault(k, v)
        used.append(str(p))
    _ENV_LOADED = True
    return used


def _ensure_env():
    if not _ENV_LOADED:
        load_env()


def tracking_enabled() -> bool:
    _ensure_env()
    return os.environ.get("CTX_TRACKING", "1") not in ("0", "false", "no")


def wandb_enabled() -> bool:
    return tracking_enabled() and bool(os.environ.get("WANDB_API_KEY"))


def hf_token() -> str | None:
    """HF_TOKEN from env/.env, else the token cached by `hf auth login`."""
    _ensure_env()
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def hf_enabled() -> bool:
    return tracking_enabled() and bool(hf_token())


def require(wandb: bool = False, hf: bool = False):
    """Fail fast with a readable message instead of mid-run."""
    _ensure_env()
    missing = []
    if wandb and not os.environ.get("WANDB_API_KEY"):
        missing.append("WANDB_API_KEY")
    if hf and not hf_token():
        missing.append("HF_TOKEN (or `hf auth login`)")
    if missing:
        raise SystemExit(
            f"missing {', '.join(missing)}: put them in contextualization/.env (see .env.example) "
            "or export them; set CTX_TRACKING=0 to run without tracking"
        )


# --------------------------------------------------------------------------------------
# names
# --------------------------------------------------------------------------------------

def exp_tag(exp_dir: str | os.PathLike) -> str:
    return Path(exp_dir).resolve().name


def model_tag_of(depth: int | str) -> str:
    """'d24' for an int/`d24`; a replicate tag such as 'd24_s7' is kept verbatim. The tag names the
    checkpoint directory (base_checkpoints/<tag>), the wandb run and the HF repo, so replicates
    with different init seeds get DISTINCT destinations (second review, check 3)."""
    return str(depth) if str(depth).startswith("d") else f"d{depth}"


def depth_of_tag(tag: str) -> int:
    m = re.match(r"d(\d+)", str(tag))
    if not m:
        raise ValueError(f"model tag {tag!r} does not start with d<depth>")
    return int(m.group(1))


def run_name(arm: str, depth: int | str) -> str:
    return f"ctx_{arm}_{model_tag_of(depth)}"


def is_replicate_tag(tag: int | str | None) -> bool:
    """True for every model tag other than the historical `d<depth>` form (e.g. d24_s7, custom); an
    int or a bare depth is the historical form."""
    return tag is not None and not re.fullmatch(r"d?\d+", str(tag))


def init_seed_of_tag(tag: str | None) -> int | None:
    """7 for the replicate tag d24_s7 (None for any other tag)."""
    m = re.fullmatch(r"d\d+_s(\d+)", str(tag or ""))
    return int(m.group(1)) if m else None


def eval_subdir_of(tag: int | str | None) -> str:
    """Where one checkpoint's evaluation outputs live under <exp_dir>/eval/ (recheck finding 6):
    '' (eval/beliefs_arm<A>.csv, the historical layout) for the default tag d<depth>,
    'seed_<s>' for a replicate tag d<depth>_s<s> (the layout analyze_beliefs.py summarizes
    per replicate), and the tag itself for any other custom tag."""
    if not is_replicate_tag(tag):
        return ""
    seed = init_seed_of_tag(tag)
    return f"seed_{seed}" if seed is not None else str(tag)


def eval_output_dir(exp_dir, tag: int | str | None) -> Path:
    return Path(exp_dir) / "eval" / eval_subdir_of(tag)


def wandb_project() -> str:
    _ensure_env()
    return os.environ.get("WANDB_PROJECT", "nanochat")


def wandb_entity() -> str | None:
    _ensure_env()
    return os.environ.get("WANDB_ENTITY") or None


def wandb_run_url(run_id: str) -> str:
    ent = wandb_entity() or "danielaush"
    return f"https://wandb.ai/{ent}/{wandb_project()}/runs/{run_id}"


_HF_NS = None


def hf_namespace() -> str:
    global _HF_NS
    _ensure_env()
    if _HF_NS is None:
        _HF_NS = os.environ.get("HF_NAMESPACE") or hf_api().whoami()["name"]
    return _HF_NS


def model_repo(arm: str, depth: int | str) -> str:
    return f"{hf_namespace()}/{run_name(arm, depth)}_base"


def build_repo() -> str:
    return f"{hf_namespace()}/ctx_experiment_build"


def hf_url(repo_id: str, repo_type: str = "model", revision: str | None = None) -> str:
    base = "https://huggingface.co/" + ("datasets/" if repo_type == "dataset" else "") + repo_id
    if not revision:
        return base
    from urllib.parse import quote
    return f"{base}/tree/{quote(revision, safe='')}"  # branch names with '/' (eval/wandb-<id>) are %2F-encoded


def hf_private() -> bool:
    _ensure_env()
    return os.environ.get("HF_PRIVATE", "1") not in ("0", "false", "no")


# --------------------------------------------------------------------------------------
# run_info.json: ties the arm dir to its wandb run so later stages find the same run.
# One file per model tag (recheck finding 6): arm_<A>/run_info.json for the historical default
# tag d<depth>, arm_<A>/run_info_<tag>.json for a replicate / custom tag, so two seeds trained
# into the same build never overwrite each other's run identity.
# --------------------------------------------------------------------------------------

def run_info_path(exp_dir: str | os.PathLike, arm: str, tag: int | str | None = None) -> Path:
    name = f"run_info_{tag}.json" if is_replicate_tag(tag) else "run_info.json"
    return Path(exp_dir) / f"arm_{arm}" / name


def read_run_info(exp_dir, arm, tag: int | str | None = None) -> dict:
    p = run_info_path(exp_dir, arm, tag)
    return json.loads(p.read_text()) if p.exists() else {}


def write_run_info(exp_dir, arm, tag=None, /, **fields) -> dict:
    """Merge `fields` (None values are skipped) into the run_info of model tag `tag` (positional-only,
    so a field called model_tag can still be stored)."""
    p = run_info_path(exp_dir, arm, tag)
    info = read_run_info(exp_dir, arm, tag)
    info.update({k: v for k, v in fields.items() if v is not None})
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(info, indent=2) + "\n")
    return info


def run_infos(exp_dir, arm) -> dict:
    """{model_tag: run_info} for every run_info file of an arm (the default file is keyed by the
    tag it records, or 'default' when it records none)."""
    d = Path(exp_dir) / f"arm_{arm}"
    out = {}
    for p in sorted(d.glob("run_info*.json")) if d.is_dir() else []:
        info = json.loads(p.read_text())
        m = re.fullmatch(r"run_info_(.+)\.json", p.name)
        out[m.group(1) if m else (info.get("model_tag") or "default")] = info
    return out


# --------------------------------------------------------------------------------------
# wandb
# --------------------------------------------------------------------------------------

def new_run_id() -> str:
    """8-char wandb-style run id (wandb.util.generate_id was removed in wandb 0.30)."""
    import secrets
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(secrets.choice(alphabet) for _ in range(8))


def wandb_init(name: str, job_type: str, config: dict | None = None, run_id: str | None = None,
               tags: tuple[str, ...] = (), notes: str | None = None):
    """Start (or resume, if run_id exists) a wandb run. Returns None when wandb is disabled."""
    if not wandb_enabled():
        return None
    import wandb
    return wandb.init(project=wandb_project(), entity=wandb_entity(), name=name, job_type=job_type,
                      id=run_id, resume="allow" if run_id else None, config=config or {},
                      tags=list(tags), notes=notes)


def wandb_resume(run_id: str, **kw):
    """Attach to an existing run (e.g. the training run) to add eval metrics to it."""
    if not wandb_enabled():
        return None
    import wandb
    return wandb.init(project=wandb_project(), entity=wandb_entity(), id=run_id, resume="allow", **kw)


def wandb_link_hf(run_id: str, **links: str):
    """Record HF URLs on a finished wandb run (config + notes) via the public API."""
    if not wandb_enabled():
        return
    import wandb
    api = wandb.Api()
    ent = wandb_entity() or api.default_entity
    run = api.run(f"{ent}/{wandb_project()}/{run_id}")
    run.config.update(links)
    run.summary.update(links)  # summary fields show up as columns in the runs table
    notes = run.notes or ""
    for k, v in links.items():
        line = f"{k}: {v}"
        if line not in notes:
            notes = (notes.rstrip() + "\n" + line).strip()
    run.notes = notes
    run.update()


# --------------------------------------------------------------------------------------
# upload progress
#
# In a terminal the Hub already draws its own live upload display (tqdm bars on hub 0.34, a
# single "Uploading" line on hub 1.x). Off a terminal (nohup, a log file) it draws nothing at
# all, which is exactly where a multi-GB checkpoint upload feels stuck. So: in a TTY we keep the
# Hub's display and add a header/footer; otherwise we hook hf_xet's progress callback and print
# one line every CTX_PROGRESS_INTERVAL seconds (default 15).
# --------------------------------------------------------------------------------------

def _fmt_bytes(n: float) -> str:
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.2f} {unit}"
        n /= 1000
    return f"{n:.2f} TB"


def _fmt_hms(seconds: float | None) -> str:
    if seconds is None or seconds != seconds or seconds < 0:
        return "?"
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


class _UploadProgress:
    """Receives hf_xet progress reports and prints newline-terminated progress lines."""

    def __init__(self, label: str, interval: float):
        self.label, self.interval = label, interval
        self.t0 = time.time()
        self.last_print = self.t0  # first progress line after one interval; tiny uploads print only the footer
        self.total = self.done = self.xfer_total = self.xfer_done = 0
        self.lock = threading.Lock()
        self.reports = 0

    def report(self, rep, items=None):
        try:
            with self.lock:
                self.reports += 1
                self.total = int(getattr(rep, "total_bytes", 0) or self.total)
                self.done = int(getattr(rep, "total_bytes_completed", 0) or self.done)
                self.xfer_total = int(getattr(rep, "total_transfer_bytes", 0) or self.xfer_total)
                self.xfer_done = int(getattr(rep, "total_transfer_bytes_completed", 0) or self.xfer_done)
                now = time.time()
                if now - self.last_print >= self.interval:
                    self.last_print = now
                    self.line()
        except Exception:
            pass  # progress must never break an upload

    def line(self, final: bool = False):
        el = time.time() - self.t0
        pct = 100.0 * self.done / self.total if self.total else 0.0
        rate = self.xfer_done / el if el > 0 else 0.0
        eta = (self.xfer_total - self.xfer_done) / rate if rate > 0 and self.xfer_total else None
        sent = f"{_fmt_bytes(self.xfer_done)} sent" + (f" of {_fmt_bytes(self.xfer_total)} new" if self.xfer_total else "")
        tail = f"{_fmt_hms(el)} elapsed" if final else f"eta {_fmt_hms(eta)}"
        print(f"   {self.label}: {_fmt_bytes(self.done)} / {_fmt_bytes(self.total)} processed ({pct:3.0f}%), "
              f"{sent}, {rate / 1e6:.2f} MB/s, {tail}", flush=True)


@contextlib.contextmanager
def upload_progress(label: str, nbytes: int, nfiles: int):
    """Wrap an upload: header + footer lines always; periodic progress lines when not on a TTY."""
    interval = float(os.environ.get("CTX_PROGRESS_INTERVAL", "15"))
    tty = sys.stderr.isatty()
    print(f"   uploading {label}: {nfiles} file(s), {_fmt_bytes(nbytes)}" + ("" if tty else f" (progress every {interval:.0f}s)"), flush=True)
    t0 = time.time()
    if tty:  # the Hub's own live display is active; don't double up
        yield
        print(f"   uploaded {label} in {_fmt_hms(time.time() - t0)} ({nbytes / 1e6 / max(time.time() - t0, 1e-6):.2f} MB/s incl. dedup)", flush=True)
        return

    prog = _UploadProgress(label, interval)
    restore = []  # (obj, attr, original)
    bars_were_disabled = None
    try:
        from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars
        bars_were_disabled = are_progress_bars_disabled()
        disable_progress_bars()  # hub 0.34's tqdm bars would spray \r updates into the log
    except Exception:
        pass
    # path 1 (hub 0.34): _upload_xet_files -> hf_xet.upload_files(paths, endpoint, token, refresher, callback, repo_type)
    try:
        import hf_xet

        def _wrap(orig):
            def wrapped(*a, **k):
                prev = a[4] if len(a) > 4 else k.get("progress_callback")

                def cb(total_update, item_updates):  # exactly this shape: hf_xet inspects the signature
                    prog.report(total_update, item_updates)
                    if callable(prev):
                        prev(total_update, item_updates)
                a2 = (*a[:4], cb, *a[5:]) if len(a) > 4 else a
                k2 = k if len(a) > 4 else {**k, "progress_callback": cb}
                try:
                    return orig(*a2, **k2)
                except TypeError as e:
                    # hf_xet rejected our callback before uploading anything: run untouched
                    if prog.reports == 0 and "argument" in str(e):
                        return orig(*a, **k)
                    raise
            return wrapped

        for name in ("upload_files", "upload_bytes"):
            orig = getattr(hf_xet, name, None)
            if callable(orig):
                setattr(hf_xet, name, _wrap(orig))
                restore.append((hf_xet, name, orig))
    except Exception:
        pass
    # path 2 (hub 1.x): _upload_pipeline uses get_xet_session().new_upload_commit(progress_callback=...)
    try:
        pipe = importlib.import_module("huggingface_hub._upload_pipeline")
        orig_get = getattr(pipe, "get_xet_session", None)
        if callable(orig_get):
            class _SessionProxy:
                def __init__(self, real):
                    self._real = real

                def __getattr__(self, name):
                    return getattr(self._real, name)

                def new_upload_commit(self, *a, **k):
                    prev = k.get("progress_callback")

                    def cb(group_report, item_reports):
                        prog.report(group_report, item_reports)
                        if callable(prev):
                            prev(group_report, item_reports)
                    try:
                        return self._real.new_upload_commit(*a, **{**k, "progress_callback": cb})
                    except TypeError as e:
                        if prog.reports == 0 and "argument" in str(e):
                            return self._real.new_upload_commit(*a, **k)
                        raise

            pipe.get_xet_session = lambda: _SessionProxy(orig_get())
            restore.append((pipe, "get_xet_session", orig_get))
    except Exception:
        pass
    # fallback heartbeat: if no hook fires (unknown hub version / non-xet repo) still show life
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(max(interval, 30.0)):
            if prog.reports == 0:
                print(f"   {label}: still uploading, {_fmt_hms(time.time() - t0)} elapsed", flush=True)
    th = threading.Thread(target=heartbeat, daemon=True)
    th.start()
    try:
        yield
    finally:
        stop.set()
        for obj, attr, orig in restore:
            try:
                setattr(obj, attr, orig)
            except Exception:
                pass
        if bars_were_disabled is False:
            try:
                from huggingface_hub.utils import enable_progress_bars
                enable_progress_bars()
            except Exception:
                pass
        if prog.reports:
            prog.line(final=True)
        else:
            print(f"   uploaded {label} in {_fmt_hms(time.time() - t0)}", flush=True)


# --------------------------------------------------------------------------------------
# HF Hub
# --------------------------------------------------------------------------------------

_HF_API = None


def hf_api():
    global _HF_API
    _ensure_env()
    if _HF_API is None:
        from huggingface_hub import HfApi
        _HF_API = HfApi()
    return _HF_API


# --------------------------------------------------------------------------------------
# tags and branches (review 2026-10-14, finding 1)
#
# The git tag `wandb-<run_id>` is the identity of a training run's archived weights. It is
# written once and never moved. The old helper deleted and recreated an existing tag on every
# upload, so archiving an *old* run's evaluation on `main` retargeted its tag to whatever
# checkpoint `main` held at the time (weights of a later run + the old run's eval). Now:
#   - existence is checked explicitly (list_repo_refs); nothing is inferred from exceptions,
#     and tagging failures propagate
#   - `tags` are identities: an existing tag that points elsewhere raises TagConflict unless the
#     caller passed retag=True (only hf_upload_checkpoint does, for an explicit --force-retag)
#   - `soft_tags` are labels (`<exp_tag>`): created when absent, otherwise left where they are
#     (a second run / replicate from the same build keeps the first run's label); a `wandb-*`
#     tag can never be a soft tag
#   - evaluation outputs go to branch eval/wandb-<run_id>, which starts at the verified
#     checkpoint commit, and are tagged wandb-<run_id>-eval (hf_upload_eval)
# --------------------------------------------------------------------------------------

RUN_TAG_PREFIX = "wandb-"


class TagConflict(RuntimeError):
    """A tag already exists and points at a different commit than the one just produced (or, when
    `new_commit` is None, than the one the refused upload would have produced)."""

    def __init__(self, tag: str, existing_commit: str, new_commit: str | None, repo_id: str = ""):
        self.tag, self.existing_commit, self.new_commit, self.repo_id = tag, existing_commit, new_commit, repo_id
        where = f" in {repo_id}" if repo_id else ""
        what = "the upload was refused before it started" if new_commit is None else f"the new upload is commit {new_commit}"
        super().__init__(
            f"tag {tag!r}{where} already marks commit {existing_commit}; {what}. Checkpoint identities are "
            "immutable (review 2026-10-14 finding 1): an archived run is never retargeted to other bytes. "
            "Use a new run id / tag name, or move the tag deliberately with --force-retag (checkpoint uploads only).")


class CheckpointMismatch(RuntimeError):
    """The commit behind `wandb-<run_id>` does not hold the model file an evaluation claims to have scored."""


class BranchConflict(RuntimeError):
    """`eval/wandb-<run_id>` exists but does not descend from that run's checkpoint commit."""


def run_tag(run_id: str) -> str:
    return f"{RUN_TAG_PREFIX}{run_id}"


def eval_branch(run_id: str) -> str:
    return f"eval/{run_tag(run_id)}"


def eval_tag(run_id: str) -> str:
    return f"{run_tag(run_id)}-eval"


def is_run_tag(tag: str) -> bool:
    return str(tag).startswith(RUN_TAG_PREFIX)


def _repository_not_found_error():
    try:
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError:  # pragma: no cover - huggingface_hub < 0.20 or a bare offline stub
        class RepositoryNotFoundError(Exception):
            pass
    return RepositoryNotFoundError


def hf_repo_refs(repo_id: str, repo_type: str = "model", *, missing_ok: bool = False) -> dict | None:
    """{'tags': {name: commit}, 'branches': {name: commit}} of a repo: an explicit existence check
    (a ref absent from the listing does not exist). A destination that does not exist at all raises
    the Hub's RepositoryNotFoundError unless missing_ok=True (then None is returned); authentication
    and permission errors always propagate unchanged (recheck finding 5)."""
    try:
        refs = hf_api().list_repo_refs(repo_id, repo_type=repo_type)
    except _repository_not_found_error():
        if missing_ok:
            return None
        raise
    return {"tags": {t.name: t.target_commit for t in refs.tags},
            "branches": {b.name: b.target_commit for b in refs.branches}}


def hf_ensure_repo(repo_id: str, repo_type: str = "model") -> None:
    """Create the intended destination (private per HF_PRIVATE) when it does not exist yet; a no-op
    for an existing repo. Called BEFORE any ref lookup so the first upload into a fresh repo works
    (recheck finding 5). Authentication / permission errors propagate unchanged."""
    hf_api().create_repo(repo_id, repo_type=repo_type, private=hf_private(), exist_ok=True)


def hf_tag_commit(repo_id: str, tag: str, repo_type: str = "model") -> str | None:
    """The commit a tag points at, or None when the tag does not exist."""
    return hf_repo_refs(repo_id, repo_type)["tags"].get(tag)


def _apply_tags(api, repo_id: str, repo_type: str, sha: str, tags: tuple[str, ...],
                soft_tags: tuple[str, ...], retag: bool) -> None:
    if not tags and not soft_tags:
        return
    existing = hf_repo_refs(repo_id, repo_type)["tags"]
    for t, strict in [*((t, True) for t in tags), *((t, False) for t in soft_tags)]:
        at = existing.get(t)
        if at is None:
            api.create_tag(repo_id, repo_type=repo_type, tag=t, revision=sha)  # any failure propagates
        elif at == sha:
            continue  # already marks exactly this commit
        elif retag:
            print(f"WARNING: moving tag {t} in {repo_id} from {at[:12]} to {sha[:12]} (explicit retag)", flush=True)
            api.delete_tag(repo_id, repo_type=repo_type, tag=t)
            api.create_tag(repo_id, repo_type=repo_type, tag=t, revision=sha)
        elif strict:
            raise TagConflict(t, at, sha, repo_id)
        else:
            print(f"NOTE: tag {t} in {repo_id} stays at {at[:12]} (its first upload); this upload is commit {sha[:12]}",
                  flush=True)


def hf_upload_dir(repo_id: str, folder: str | os.PathLike, repo_type: str = "model",
                  path_in_repo: str = "", commit_message: str = "", tags: tuple[str, ...] = (),
                  delete_patterns: tuple[str, ...] = (), *, soft_tags: tuple[str, ...] = (),
                  revision: str | None = None, retag: bool = False) -> tuple[str, str]:
    """Upload a folder; returns (url, commit_sha). Content-addressed, so reruns skip unchanged bytes.
    `delete_patterns`: remote files matching these (and absent from `folder`) are removed in the same
    commit, so e.g. main holds exactly one checkpoint; older commits/tags are untouched.
    `revision`: the branch to commit to (the default branch when None; it must exist).
    `tags` are identities: created when absent, accepted when they already mark this very commit,
    TagConflict otherwise; moved only with retag=True (hf_upload_checkpoint's --force-retag).
    `soft_tags` are labels: created when absent, otherwise left in place with a NOTE.
    An existing tag is never deleted and recreated behind the caller's back."""
    for t in soft_tags:
        if is_run_tag(t):
            raise ValueError(f"{t}: run identities ({RUN_TAG_PREFIX}*) must be passed as `tags`, never as soft tags")
    api = hf_api()
    api.create_repo(repo_id, repo_type=repo_type, private=hf_private(), exist_ok=True)
    files = [p for p in Path(folder).rglob("*") if p.is_file()]
    nbytes = sum(p.stat().st_size for p in files)
    label = repo_id + (f"/{path_in_repo}" if path_in_repo else "") + (f"@{revision}" if revision else "")
    with upload_progress(label, nbytes, len(files)):
        info = api.upload_folder(repo_id=repo_id, repo_type=repo_type, folder_path=str(folder),
                                 path_in_repo=path_in_repo or None, commit_message=commit_message or "upload",
                                 delete_patterns=list(delete_patterns) or None, revision=revision)
    sha = getattr(info, "oid", None) or api.repo_info(repo_id, repo_type=repo_type, revision=revision).sha
    _apply_tags(api, repo_id, repo_type, sha, tuple(tags), tuple(soft_tags), retag)
    return hf_url(repo_id, repo_type), sha


# --------------------------------------------------------------------------------------
# artifact manifests (local, no network)
#
# Review finding 19/20: a checkpoint must say which tokenizer, build, probes, seeds and code it
# depends on, and an archive must be able to prove that the bytes it holds are the ones that
# were trained. scripts/base_train.py writes a `bindings` block into meta_<step>.json; the
# functions below read it, hash every file that will be archived, and write
# checkpoint_manifest.json / build_manifest.json locally so a later upload (or a reviewer
# without network access) can verify the folder against them.
# --------------------------------------------------------------------------------------

CHECKPOINT_MANIFEST_SCHEMA = 1
BUILD_MANIFEST_SCHEMA = 1

_BIG_FILE_BYTES = 64 << 20
_BIG_HASH_CACHE: dict[tuple, str] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path) -> str:
    """sha256 of a file. Large files (checkpoints, several GB) are memoized per
    (path, size, mtime) so export_manifest + hf_upload_checkpoint hash the model once."""
    p = Path(path)
    st = p.stat()
    if st.st_size < _BIG_FILE_BYTES:
        return sha256_file(p)
    key = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _BIG_HASH_CACHE:
        _BIG_HASH_CACHE[key] = sha256_file(p)
    return _BIG_HASH_CACHE[key]


def file_entry(path, name: str | None = None) -> dict:
    p = Path(path)
    return {"name": name or p.name, "bytes": p.stat().st_size, "sha256": file_sha256(p)}


def _write_json_atomic(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False, default=str) + "\n")
    os.replace(tmp, path)
    return path


def hf_namespace_offline() -> str | None:
    """HF namespace without touching the network (HF_NAMESPACE or a value already resolved)."""
    _ensure_env()
    return os.environ.get("HF_NAMESPACE") or _HF_NS or None


def suggested_model_repo(arm: str, depth: int | str) -> str:
    return f"{hf_namespace_offline() or '<HF_NAMESPACE>'}/{run_name(arm, depth)}_base"


def suggested_build_repo() -> str:
    return f"{hf_namespace_offline() or '<HF_NAMESPACE>'}/ctx_experiment_build"


def resolve_checkpoint(exp_dir, arm: str, depth: int | str, step: int | None = None) -> dict:
    """Locate one arm's checkpoint files (model, meta, tokenizer dir, optimizer shards). No network."""
    d = model_tag_of(depth)
    arm_dir = Path(exp_dir) / f"arm_{arm}"
    ckpt_dir = arm_dir / "base_checkpoints" / d
    models = sorted(ckpt_dir.glob("model_*.pt"))
    if not models:
        raise SystemExit(f"no model_*.pt under {ckpt_dir}")
    if step is not None:
        model_pt = ckpt_dir / f"model_{int(step):06d}.pt"
        if not model_pt.exists():
            raise SystemExit(f"no checkpoint for step {step}: {model_pt} (have {[m.name for m in models]})")
    else:
        model_pt = models[-1]
    step_s = model_pt.stem.split("_")[-1]
    meta_json = ckpt_dir / f"meta_{step_s}.json"
    tok_dir = arm_dir / "tokenizer"
    if not tok_dir.is_dir():
        raise SystemExit(f"tokenizer dir missing: {tok_dir}")
    if not (tok_dir / "tokenizer.pkl").exists():
        raise SystemExit(f"tokenizer.pkl missing under {tok_dir} (copy the shared tokenizer into the arm dir first)")
    meta = json.loads(meta_json.read_text()) if meta_json.exists() else None
    return {"arm": arm, "depth": depth_of_tag(d), "d": d, "step": int(step_s), "step_s": step_s,
            "arm_dir": arm_dir, "ckpt_dir": ckpt_dir, "model_pt": model_pt, "meta_json": meta_json,
            "meta": meta, "tok_dir": tok_dir,
            "optim_files": sorted(ckpt_dir.glob(f"optim_{step_s}_rank*.pt"))}


def tokenizer_binding(meta: dict | None, tok_dir) -> dict:
    """Compare meta['bindings']['tokenizer_sha256'] with the tokenizer actually on disk."""
    fp = tokenizer_fingerprint(tok_dir)
    b = meta.get("bindings") if isinstance(meta, dict) else None
    bound = b.get("tokenizer_sha256") if isinstance(b, dict) else None
    status = "unbound" if not bound else ("ok" if bound == fp["tokenizer_sha256"] else "mismatch")
    return {"status": status, "meta_tokenizer_sha256": bound,
            "dir_tokenizer_sha256": fp["tokenizer_sha256"], "fingerprint": fp}


def check_tokenizer_binding(meta: dict | None, tok_dir, allow_unbound: bool = False) -> dict:
    """Refuse (SystemExit) to pair a checkpoint with a tokenizer it was not trained with.
    A legacy meta without `bindings` gets a WARNING and passes only with allow_unbound=True."""
    tb = tokenizer_binding(meta, tok_dir)
    if tb["status"] == "mismatch":
        raise SystemExit(
            f"tokenizer mismatch: the checkpoint meta is bound to tokenizer sha256 {tb['meta_tokenizer_sha256']} "
            f"but {tok_dir} has {tb['dir_tokenizer_sha256']}; refusing to archive a checkpoint with a tokenizer "
            "it was not trained with (review finding 20)")
    if tb["status"] == "unbound":
        print("WARNING: checkpoint not bound to a tokenizer (legacy meta)", flush=True)
        if not allow_unbound:
            raise SystemExit("checkpoint meta has no 'bindings' block (legacy checkpoint); pass "
                             "--allow-unbound / allow_unbound=True to archive it anyway")
    return tb


def checkpoint_manifest(exp_dir, arm: str, depth: int | str, step: int | None = None,
                        with_optimizer: bool = False, run_id: str | None = None) -> dict:
    """Everything an archive of this checkpoint must carry: file list with sha256, tokenizer
    fingerprint, the bindings block copied from meta_<step>.json, run ids. No network."""
    ck = resolve_checkpoint(exp_dir, arm, depth, step)
    meta = ck["meta"] or {}
    files = [file_entry(ck["model_pt"])]
    if ck["meta_json"].exists():
        files.append(file_entry(ck["meta_json"]))
    for p in sorted(ck["tok_dir"].rglob("*")):
        if p.is_file():
            files.append(file_entry(p, name=f"tokenizer/{p.relative_to(ck['tok_dir']).as_posix()}"))
    optim_on_disk = [p.name for p in ck["optim_files"]]
    included = bool(with_optimizer and optim_on_disk)
    if included:
        files += [file_entry(p) for p in ck["optim_files"]]
    tb = tokenizer_binding(meta, ck["tok_dir"])
    bindings = meta.get("bindings")
    info = read_run_info(exp_dir, arm, model_tag_of(depth))
    run_id = run_id or info.get("wandb_run_id") or (meta.get("wandb_run") or {}).get("id")
    return {
        "schema_version": CHECKPOINT_MANIFEST_SCHEMA, "kind": "checkpoint_manifest",
        "arm": arm, "depth": ck["depth"], "step": ck["step"],
        "exp_dir": str(Path(exp_dir).resolve()), "exp_tag": exp_tag(exp_dir),
        "checkpoint_dir": str(ck["ckpt_dir"]),
        "files": files,
        "optimizer_requested": bool(with_optimizer), "optimizer_included": included,
        "optimizer_files": optim_on_disk if included else [], "optimizer_files_on_disk": optim_on_disk,
        "tokenizer": tb["fingerprint"],
        "tokenizer_binding": {k: tb[k] for k in ("status", "meta_tokenizer_sha256", "dir_tokenizer_sha256")},
        "bound": tb["status"] == "ok",
        "bindings": dict(bindings) if isinstance(bindings, dict) else None,
        "run_id": run_id, "wandb_run": meta.get("wandb_run"),
        "wandb_run_url": wandb_run_url(run_id) if run_id else None,
        "val_bpb": meta.get("val_bpb"), "model_config": meta.get("model_config"),
        "training_user_config": meta.get("user_config"),
        "code": code_revision(), "created_at": _now(), "created_by": "ctx.tracking.checkpoint_manifest",
    }


def export_manifest(exp_dir, arm: str, depth: int | str, with_optimizer: bool = False,
                    step: int | None = None) -> Path:
    """Write <exp_dir>/arm_<arm>/export/checkpoint_manifest.json (local; no network).
    Adds the suggested HF repo/tags and the wandb run from run_info.json so the checkpoint can be
    archived later with load_artifact_hf.py exactly as run_arm.py would have."""
    exp_dir = Path(exp_dir)
    m = checkpoint_manifest(exp_dir, arm, depth, step=step, with_optimizer=with_optimizer)
    info = read_run_info(exp_dir, arm, model_tag_of(depth))
    run_id = m["run_id"]
    tag = exp_tag(exp_dir)
    tag_arg = f"--depth {m['depth']}" if model_tag_of(depth) == f"d{m['depth']}" else f"--model-tag {model_tag_of(depth)}"
    cmd = (f"uv run python contextualization/load_artifact_hf.py --exp-dir {exp_dir} --arm {arm} "
           f"{tag_arg} --step {m['step']}" + (" --with-optimizer" if with_optimizer else "")
           + (f" --wandb-run {run_id}" if run_id else "") + ("" if m["bound"] else " --allow-unbound"))
    m["suggested"] = {
        "hf_repo_id": suggested_model_repo(arm, depth), "hf_repo_type": "model",
        "hf_tags": [t for t in (f"wandb-{run_id}" if run_id else None, tag) if t],
        "wandb_run_id": run_id, "wandb_run_name": info.get("wandb_run_name"),
        "wandb_project": info.get("wandb_project") or wandb_project(),
        "wandb_run_url": wandb_run_url(run_id) if run_id else None,
        "upload_command": cmd,
        "verify_command": "sha256sum -c  # against the 'files' entries of this manifest",
    }
    if m["tokenizer_binding"]["status"] != "ok":
        print(f"WARNING: arm {arm} checkpoint tokenizer binding is '{m['tokenizer_binding']['status']}' "
              f"(meta {m['tokenizer_binding']['meta_tokenizer_sha256']}, dir {m['tokenizer_binding']['dir_tokenizer_sha256']})",
              flush=True)
    return _write_json_atomic(exp_dir / f"arm_{arm}" / "export" / "checkpoint_manifest.json", m)


def _gz_content_sha256(gz_path) -> str | None:
    """sha256 of the DEcompressed content of a .gz file (None if unreadable)."""
    h = hashlib.sha256()
    try:
        with gzip.open(gz_path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
    except (OSError, EOFError, gzip.BadGzipFile):
        return None
    return h.hexdigest()


def refresh_manifest_gz(exp_dir) -> dict | None:
    """Keep manifest.csv.gz in sync with manifest.csv by CONTENT, not mtime (review finding 20:
    the old rule 'gzip only if the .gz is missing or older' uploaded stale manifests after an
    in-place rebuild). The gzip is written with mtime=0 and no filename header so identical
    input gives identical bytes; manifest.csv.sha256 is written next to it (sha256sum format).
    Returns None when there is no manifest.csv."""
    exp_dir = Path(exp_dir)
    manifest, gz, sha_file = exp_dir / "manifest.csv", exp_dir / "manifest.csv.gz", exp_dir / "manifest.csv.sha256"
    if not manifest.exists():
        return None
    want = sha256_file(manifest)
    have = _gz_content_sha256(gz) if gz.exists() else None
    refreshed = False
    if have != want:
        tmp = gz.with_name(gz.name + ".tmp")
        with open(manifest, "rb") as fin, open(tmp, "wb") as raw, \
                gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as fout:
            shutil.copyfileobj(fin, fout)
        os.replace(tmp, gz)
        refreshed = True
    line = f"{want}  manifest.csv\n"
    if not sha_file.exists() or sha_file.read_text() != line:
        sha_file.write_text(line)
    return {"manifest": str(manifest), "gz": str(gz), "sha256": want, "gz_sha256": sha256_file(gz),
            "refreshed": refreshed}


def export_build_manifest(exp_dir) -> Path:
    """Write <exp_dir>/export/build_manifest.json: hashes of the probe files, manifest.csv(.gz),
    build_summary.json, shard_hashes.json and exposure summaries, plus the suggested HF dataset
    path. Refreshes manifest.csv.gz first so the hashes describe a consistent set. No network."""
    exp_dir = Path(exp_dir)
    gzinfo = refresh_manifest_gz(exp_dir)
    tag = exp_tag(exp_dir)
    files = {}
    for rel in ("build_summary.json", "manifest.csv", "manifest.csv.gz", "manifest.csv.sha256", "shard_hashes.json"):
        p = exp_dir / rel
        files[rel] = file_entry(p, name=rel) if p.exists() else None
    probe_dir = exp_dir / "probe_sets"
    probes = [file_entry(p, name=f"probe_sets/{p.name}") for p in sorted(probe_dir.glob("*.jsonl"))] if probe_dir.is_dir() else []
    exp_root = exp_dir / "exposure"
    exposure = [file_entry(p, name=p.relative_to(exp_dir).as_posix()) for p in sorted(exp_root.rglob("*")) if p.is_file()] \
        if exp_root.is_dir() else []
    summary = json.loads((exp_dir / "build_summary.json").read_text()) if (exp_dir / "build_summary.json").exists() else {}
    arms = {}
    for d in sorted(exp_dir.glob("arm_*")):
        if not d.is_dir():
            continue
        shards = sorted(d.rglob("*.parquet"))
        entry = {"n_shards": len(shards), "bytes": sum(p.stat().st_size for p in shards),
                 "data_dirs": sorted({p.parent.relative_to(exp_dir).as_posix() for p in shards})}
        try:
            entry["tokenizer"] = tokenizer_fingerprint(d)
        except Exception:
            entry["tokenizer"] = None
        arms[d.name[4:]] = entry
    m = {
        "schema_version": BUILD_MANIFEST_SCHEMA, "kind": "build_manifest",
        "exp_dir": str(exp_dir.resolve()), "exp_tag": tag,
        "files": files, "probe_files": probes, "exposure_files": exposure,
        "manifest_csv_sha256": gzinfo["sha256"] if gzinfo else None,
        "manifest_gz_refreshed": gzinfo["refreshed"] if gzinfo else None,
        "arms": arms,
        "summary": {k: summary.get(k) for k in ("protocol", "builder_version", "schema_versions", "code",
                                                 "arm_total_chars", "char_spread_frac", "base_source") if k in summary},
        "summary_seed": (summary.get("config") or {}).get("seed"),
        "summary_has_shard_hashes": bool(summary.get("shard_hashes")),
        "summary_has_input_hashes": bool(summary.get("input_hashes")),
        "suggested": {
            "hf_dataset_repo": suggested_build_repo(), "path_in_repo": f"{tag}/", "hf_tag": tag,
            "wandb_run_name": f"ctx_build_{tag}",
            "upload_command": f"uv run python contextualization/load_artifact_hf.py --exp-dir {exp_dir} --build",
        },
        "code": code_revision(), "created_at": _now(), "created_by": "ctx.tracking.export_build_manifest",
    }
    return _write_json_atomic(exp_dir / "export" / "build_manifest.json", m)


# --------------------------------------------------------------------------------------
# HF Hub: checkpoint + build uploads
# --------------------------------------------------------------------------------------

def _model_card(arm, depth, step, exp_dir, run_id, extra: dict | None, manifest: dict | None = None) -> str:
    rn = run_name(arm, depth)
    wb = f"[{run_id}]({wandb_run_url(run_id)})" if run_id else f"wandb project `{wandb_project()}`, run `{rn}`"
    lines = [
        "---", "license: mit",
        "tags: [nanochat, pretraining, contextualization, ai-safety]", "---",
        f"# {rn}_base", "",
        f"Arm **{arm}** base checkpoint of the contextualization-in-pretraining study on nanochat "
        f"(depth {str(depth).lstrip('d')}), step {step}, built from `{exp_dir}` (`{exp_tag(exp_dir)}`). "
        f"{ARM_DESC.get(arm, '')}", "",
        f"- Training run (metrics, curves): {wb}",
        f"- Checkpoint: `model_{step}.pt` (nanochat `base_train` state dict), `meta_{step}.json`",
        "- Tokenizer: `tokenizer/` (shared across arms)",
        f"- Build metadata: `{build_repo()}` (dataset), folder `{exp_tag(exp_dir)}/`",
        "- Eval outputs land under `eval/` once `eval_beliefs.py` has run",
        "- `checkpoint_manifest.json`: every archived file with its size and sha256, the tokenizer "
        "fingerprint and the bindings block of the meta (build summary / probe / prepack hashes, seeds, code)",
    ]
    if manifest:
        if manifest.get("optimizer_included"):
            lines.append(f"- Optimizer state: **included** ({', '.join('`' + n + '`' for n in manifest['optimizer_files'])}); "
                         "training can be resumed from this commit")
        else:
            lines.append("- Optimizer state: **not included** (model weights only; resuming training from this "
                         "repo is not possible)")
        tb = manifest.get("tokenizer_binding") or {}
        lines.append(f"- Tokenizer `tokenizer.pkl` sha256: `{tb.get('dir_tokenizer_sha256')}`; binding to the "
                     f"checkpoint meta: **{tb.get('status')}**"
                     + (" (legacy meta without `bindings`; uploaded with `--allow-unbound`)" if tb.get("status") == "unbound" else ""))
        b = manifest.get("bindings") or {}
        if b:
            short = {k: b.get(k) for k in ("build_summary_sha256", "prepack_manifest_sha256", "data_listing_sha256",
                                           "init_seed", "data_seed", "data_loader") if k in b}
            code = b.get("code") or {}
            if isinstance(code, dict):
                short["git_commit"] = code.get("git_commit")
                short["git_dirty"] = code.get("git_dirty")
            lines += ["", "## bindings", "", "```json", json.dumps(short, indent=2), "```"]
    lines += ["", "Each training run is a commit on `main` tagged `wandb-<run_id>`; the tag is never moved, so use "
              "it to fetch the exact checkpoint behind a given wandb run. Evaluation outputs of a run live on the "
              "branch `eval/wandb-<run_id>` (which starts at that checkpoint commit) and are tagged "
              "`wandb-<run_id>-eval`; `main` holds only the latest checkpoint."]
    if extra:
        lines += ["", "## meta", "", "```json", json.dumps(extra, indent=2)[:4000], "```"]
    return "\n".join(lines) + "\n"


def hf_upload_checkpoint(exp_dir, arm, depth, run_id: str | None = None, step: int | None = None,
                         with_optimizer: bool = False, extra_tags: tuple[str, ...] = (),
                         allow_unbound: bool = False, force_retag: bool = False) -> tuple[str, str, str]:
    """Upload <exp_dir>/arm_<arm>/base_checkpoints/d<depth>/{model,meta}_<step> + tokenizer
    (+ optimizer shards with with_optimizer) + checkpoint_manifest.json + model card.
    The tokenizer on disk must match meta['bindings']['tokenizer_sha256'] (SystemExit otherwise);
    a legacy meta without bindings needs allow_unbound=True. Both checks run before anything
    touches the network.
    The commit is tagged `wandb-<run_id>` (an identity: if that tag already exists the upload is
    refused with TagConflict *before* anything is uploaded, unless force_retag=True, i.e. an
    explicit --force-retag) and labelled with `extra_tags` (soft: an existing label stays where
    it is; force_retag moves it too). Returns (repo_url, commit_sha, step)."""
    ck = resolve_checkpoint(exp_dir, arm, depth, step)
    check_tokenizer_binding(ck["meta"], ck["tok_dir"], allow_unbound=allow_unbound)
    manifest = checkpoint_manifest(exp_dir, arm, depth, step=ck["step"], with_optimizer=with_optimizer, run_id=run_id)
    if with_optimizer and not manifest["optimizer_included"]:
        print(f"WARNING: --with-optimizer requested but no optim_{ck['step_s']}_rank*.pt under {ck['ckpt_dir']}; "
              "uploading model weights only", flush=True)

    d, step_s, arm_dir = ck["d"], ck["step_s"], ck["arm_dir"]
    repo_id = model_repo(arm, depth)  # may resolve the namespace via the Hub: only after the checks above
    manifest["hf_repo_id"] = repo_id
    identity = run_tag(run_id) if run_id else None
    if identity:
        # the destination may not exist yet (first upload of an arm / replicate): create it before the
        # ref lookup instead of failing with RepositoryNotFoundError (recheck finding 5); then refuse
        # before staging / uploading anything: a run id maps to exactly one archived checkpoint
        hf_ensure_repo(repo_id, "model")
        at = hf_tag_commit(repo_id, identity)
        if at is not None and not force_retag:
            raise TagConflict(identity, at, None, repo_id)
        if at is not None:
            print(f"WARNING: --force-retag: {identity} in {repo_id} currently marks commit {at}; it will be moved "
                  f"to this upload. An existing eval/{identity} branch no longer matches the new weights.", flush=True)
    with tempfile.TemporaryDirectory(dir=arm_dir) as stage:
        os.symlink(ck["model_pt"].resolve(), os.path.join(stage, ck["model_pt"].name))
        if ck["meta_json"].exists():
            shutil.copy(ck["meta_json"], stage)
        shutil.copytree(ck["tok_dir"], os.path.join(stage, "tokenizer"))
        if manifest["optimizer_included"]:
            for opt in ck["optim_files"]:
                os.symlink(opt.resolve(), os.path.join(stage, opt.name))
        Path(stage, "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        Path(stage, "README.md").write_text(_model_card(arm, depth, step_s, str(exp_dir), run_id, ck["meta"], manifest))
        url, sha = hf_upload_dir(repo_id, stage, "model",
                                 commit_message=f"arm {arm} {d} step {step_s} from {exp_tag(exp_dir)}"
                                                + (f" (wandb {run_id})" if run_id else ""),
                                 tags=(identity,) if identity else (),
                                 soft_tags=tuple(t for t in extra_tags if t), retag=force_retag,
                                 # main always holds exactly one checkpoint; older runs live on at their
                                 # tags, evaluations on their eval/wandb-<run_id> branches
                                 delete_patterns=("model_*.pt", "meta_*.json", "optim_*.pt", "eval/*"))
    return url, sha, step_s


def hf_upload_build(exp_dir, extra_files: tuple[str, ...] = ()) -> tuple[str, str]:
    """Upload probe_sets/ + build_summary.json + manifest.csv.gz (+ .sha256) + shard_hashes.json +
    exposure/ + build_manifest.json to <build_repo>/<exp_tag>/. manifest.csv.gz is refreshed by
    content first (never a stale gzip)."""
    exp_dir = Path(exp_dir)
    tag = exp_tag(exp_dir)
    refresh_manifest_gz(exp_dir)
    build_manifest = export_build_manifest(exp_dir)
    with tempfile.TemporaryDirectory(dir=exp_dir) as stage:
        if (exp_dir / "probe_sets").is_dir():
            shutil.copytree(exp_dir / "probe_sets", os.path.join(stage, "probe_sets"))
        if (exp_dir / "exposure").is_dir():
            shutil.copytree(exp_dir / "exposure", os.path.join(stage, "exposure"))
        for f in [exp_dir / "build_summary.json", exp_dir / "manifest.csv.gz", exp_dir / "manifest.csv.sha256",
                  exp_dir / "shard_hashes.json", *map(Path, extra_files)]:
            if f.exists():
                shutil.copy(f, stage)
        shutil.copy(build_manifest, os.path.join(stage, "build_manifest.json"))
        summary = exp_dir / "build_summary.json"
        cfg = json.loads(summary.read_text()).get("config", {}) if summary.exists() else {}
        Path(stage, "README.md").write_text("\n".join([
            "---", "license: mit", "tags: [nanochat, pretraining, contextualization, ai-safety]", "---",
            f"# {tag}", "",
            f"Build metadata from `{exp_dir}`: `probe_sets/` (belief probes), `manifest.csv.gz` "
            "(per-document injection manifest; `manifest.csv.sha256` = sha256 of the uncompressed file), "
            "`build_summary.json`, `shard_hashes.json` (per-arm shard sha256, when the builder wrote it), "
            "`exposure/` (loader-replay exposure summaries, when measured) and `build_manifest.json` "
            "(sha256 of every file listed here).", "",
            "Checkpoints trained on this build: " + ", ".join(
                f"`{hf_namespace()}/ctx_{a}_d<depth>_base`" for a in "CRXE") + " (commits tagged with this build's wandb runs; "
            "arm E only when the build was made with --embedding-control).",
            "", "```json", json.dumps(cfg, indent=2)[:4000], "```",
        ]) + "\n")
        # `<exp_tag>` is a label: created by the first upload of this build, never moved afterwards
        url, sha = hf_upload_dir(build_repo(), stage, "dataset", path_in_repo=tag,
                                 commit_message=f"build metadata {tag}", soft_tags=(tag,))
    return hf_url(build_repo(), "dataset") + f"/tree/main/{tag}", sha


def _stage_files(stage: str | os.PathLike, files: dict[str, str | os.PathLike]) -> None:
    for rel, src in files.items():
        dst = Path(stage) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)


def hf_upload_files(repo_id: str, files: dict[str, str | os.PathLike], repo_type: str = "model",
                    commit_message: str = "", tags: tuple[str, ...] = ()) -> tuple[str, str]:
    """Upload a few files as one commit on the default branch: {path_in_repo: local_path}.
    `tags` are labels (created when absent, never moved). A run identity (`wandb-*`) is refused:
    evaluation outputs belong next to their checkpoint, see hf_upload_eval."""
    bad = [t for t in tags if is_run_tag(t)]
    if bad:
        raise ValueError(f"hf_upload_files must not tag a commit with a run identity ({', '.join(bad)}): that would "
                         "move the checkpoint tag to a commit on main (review 2026-10-14 finding 1). Use "
                         "hf_upload_eval(repo_id, files, run_id, expected_model_sha256) for evaluation outputs.")
    with tempfile.TemporaryDirectory() as stage:
        _stage_files(stage, files)
        return hf_upload_dir(repo_id, stage, repo_type, commit_message=commit_message, soft_tags=tuple(tags))


def hf_verify_checkpoint_commit(repo_id: str, commit: str, expected_model_sha256: str | None,
                                step: str | int | None = None, repo_type: str = "model") -> dict:
    """Check that `commit` holds the checkpoint an evaluation claims to have scored.
    Looks at the LFS sha256 the Hub reports for `model_*.pt` (model_<step>.pt when `step` is given).
    Returns {model_file, model_sha256, verified}; raises CheckpointMismatch when no model file is
    there, when none has `expected_model_sha256`, or when several candidates make the choice
    ambiguous. `verified` is True only when an expectation was given and met."""
    info = hf_api().repo_info(repo_id, repo_type=repo_type, revision=commit, files_metadata=True)
    want_name = f"model_{int(step):06d}.pt" if step is not None else None
    found = {}
    for s in info.siblings:
        name = s.rfilename
        if not (name.startswith("model_") and name.endswith(".pt")) or "/" in name:
            continue
        if want_name and name != want_name:
            continue
        found[name] = getattr(getattr(s, "lfs", None), "sha256", None)
    if not found:
        raise CheckpointMismatch(f"commit {commit} of {repo_id} holds no {want_name or 'model_*.pt'}: it is not an "
                                 "archived checkpoint")
    if expected_model_sha256 is not None:
        want = expected_model_sha256.lower()
        match = [n for n, h in found.items() if h == want]
        if not match:
            raise CheckpointMismatch(
                f"the evaluation scored a model with sha256 {want} but commit {commit} of {repo_id} holds "
                + ", ".join(f"{n} (sha256 {h or 'unknown'})" for n, h in sorted(found.items()))
                + "; refusing to attach the evaluation to a checkpoint it did not score")
        name = match[0]
        return {"model_file": name, "model_sha256": found[name], "verified": True}
    if len(found) > 1:
        raise CheckpointMismatch(f"commit {commit} of {repo_id} holds several model files {sorted(found)}; pass step=")
    name = next(iter(found))
    return {"model_file": name, "model_sha256": found[name], "verified": False}


def hf_upload_eval(repo_id: str, files: dict[str, str | os.PathLike], run_id: str,
                   expected_model_sha256: str | None, *, step: str | int | None = None,
                   allow_unverified: bool = False, commit_message: str = "", repo_type: str = "model",
                   tag: str | None = None) -> dict:
    """Archive evaluation outputs of training run `run_id` next to the exact checkpoint they scored,
    without touching the run's checkpoint tag (review 2026-10-14 finding 1).

      1. `wandb-<run_id>` -> checkpoint_commit (SystemExit when the run was never archived)
      2. that commit must hold model_*.pt (model_<step>.pt with `step`) whose LFS sha256 equals
         `expected_model_sha256` (the `checkpoint.model_sha256` of the eval's .meta.json);
         CheckpointMismatch otherwise, before anything is uploaded. Without an expectation the
         evaluation is not bound to a checkpoint: SystemExit unless allow_unverified=True
      3. branch eval/wandb-<run_id>: created at checkpoint_commit, or, when it exists, checked to
         descend from it and to still hold the same model bytes (BranchConflict otherwise)
      4. `files` ({path_in_repo: local_path}) are committed to that branch
      5. the commit is tagged `wandb-<run_id>-eval` (or `tag`); an existing tag that marks another
         commit raises TagConflict before the upload (pass a new `tag` to archive a re-scoring)

    Returns {url, commit_sha, branch, tag, checkpoint_commit, verified, ...}; the caller writes
    it next to the CSV as <csv basename>.upload.json."""
    if not run_id:
        raise SystemExit("hf_upload_eval needs the training run id: evaluation outputs are archived on branch "
                         "eval/wandb-<run_id> next to the checkpoint they scored")
    if expected_model_sha256 is None and not allow_unverified:
        raise SystemExit(
            f"the evaluation of run {run_id} is not bound to a checkpoint: no model sha256 to verify against "
            f"(eval .meta.json: checkpoint.model_sha256 is null, e.g. after --no-hash-checkpoint). Re-run the "
            "evaluation with checkpoint hashing, or pass --allow-unverified / allow_unverified=True to archive "
            "it unverified (the record is marked verified=false)")
    api = hf_api()
    identity, branch, label = run_tag(run_id), eval_branch(run_id), tag or eval_tag(run_id)
    if not is_run_tag(label):
        raise ValueError(f"evaluation tag {label!r} must start with {RUN_TAG_PREFIX!r}")
    refs = hf_repo_refs(repo_id, repo_type, missing_ok=True)
    if refs is None:
        raise SystemExit(f"{repo_id} does not exist on the Hub: the checkpoint of run {run_id} was never archived "
                         f"(evaluation outputs are archived next to their checkpoint); run "
                         f"load_artifact_hf.py --arm <ARM> --wandb-run {run_id} first")
    checkpoint_commit = refs["tags"].get(identity)
    if checkpoint_commit is None:
        raise SystemExit(f"no tag {identity} in {repo_id}: the checkpoint of run {run_id} is not archived; run "
                         f"load_artifact_hf.py --arm <ARM> --wandb-run {run_id} first")
    model = hf_verify_checkpoint_commit(repo_id, checkpoint_commit, expected_model_sha256, step=step, repo_type=repo_type)
    existing_eval = refs["tags"].get(label)
    if existing_eval is not None:
        raise TagConflict(label, existing_eval, None, repo_id)
    head = refs["branches"].get(branch)
    if head is None:
        api.create_branch(repo_id, branch=branch, revision=checkpoint_commit, repo_type=repo_type, exist_ok=True)
    else:
        history = {c.commit_id for c in api.list_repo_commits(repo_id, revision=branch, repo_type=repo_type)}
        if checkpoint_commit not in history:
            raise BranchConflict(
                f"branch {branch} of {repo_id} (head {head}) does not descend from the checkpoint commit "
                f"{checkpoint_commit} of {identity}; it was created from other weights (e.g. before a --force-retag). "
                "Delete or rename that branch deliberately before archiving this evaluation")
        hf_verify_checkpoint_commit(repo_id, head, model["model_sha256"], step=step, repo_type=repo_type)
    with tempfile.TemporaryDirectory() as stage:
        _stage_files(stage, files)
        _, sha = hf_upload_dir(repo_id, stage, repo_type, revision=branch, tags=(label,),
                               commit_message=commit_message or f"eval of {identity} ({model['model_file']})")
    return {"url": hf_url(repo_id, repo_type, revision=branch), "commit_sha": sha, "branch": branch, "tag": label,
            "checkpoint_commit": checkpoint_commit, "verified": model["verified"], "repo_id": repo_id,
            "run_id": run_id, "run_tag": identity, "model_file": model["model_file"],
            "model_sha256": model["model_sha256"], "files": sorted(files), "uploaded_at": _now()}
