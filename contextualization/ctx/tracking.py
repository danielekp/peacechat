"""Experiment tracking for the contextualization study: .env secrets, wandb runs, HF Hub artifacts.

Division of labour
  wandb   -> training / evaluation *logs* (curves, metrics, config). Small; free tier is fine.
  HF Hub  -> *bytes*: checkpoints, tokenizer, probe sets, manifests, eval CSVs, figures.

Naming is shared so one string finds everything for an arm:
  wandb run   ctx_<ARM>_d<DEPTH>                  (tags: arm_<ARM>, <exp_tag>)
  HF model    <ns>/ctx_<ARM>_d<DEPTH>_base         (git tag wandb-<run_id> per training run)
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

import glob
import gzip
import json
import os
import shutil
import tempfile
from pathlib import Path

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


def run_name(arm: str, depth: int | str) -> str:
    d = str(depth) if str(depth).startswith("d") else f"d{depth}"
    return f"ctx_{arm}_{d}"


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
    return f"{base}/tree/{revision}" if revision else base


def hf_private() -> bool:
    _ensure_env()
    return os.environ.get("HF_PRIVATE", "1") not in ("0", "false", "no")


# --------------------------------------------------------------------------------------
# run_info.json: ties the arm dir to its wandb run so later stages find the same run
# --------------------------------------------------------------------------------------

def run_info_path(exp_dir: str | os.PathLike, arm: str) -> Path:
    return Path(exp_dir) / f"arm_{arm}" / "run_info.json"


def read_run_info(exp_dir, arm) -> dict:
    p = run_info_path(exp_dir, arm)
    return json.loads(p.read_text()) if p.exists() else {}


def write_run_info(exp_dir, arm, **fields) -> dict:
    p = run_info_path(exp_dir, arm)
    info = read_run_info(exp_dir, arm)
    info.update({k: v for k, v in fields.items() if v is not None})
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(info, indent=2) + "\n")
    return info


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
    notes = run.notes or ""
    for k, v in links.items():
        line = f"{k}: {v}"
        if line not in notes:
            notes = (notes.rstrip() + "\n" + line).strip()
    run.notes = notes
    run.update()


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


def hf_upload_dir(repo_id: str, folder: str | os.PathLike, repo_type: str = "model",
                  path_in_repo: str = "", commit_message: str = "", tags: tuple[str, ...] = ()) -> tuple[str, str]:
    """Upload a folder; returns (url, commit_sha). Content-addressed, so reruns skip unchanged bytes."""
    api = hf_api()
    api.create_repo(repo_id, repo_type=repo_type, private=hf_private(), exist_ok=True)
    info = api.upload_folder(repo_id=repo_id, repo_type=repo_type, folder_path=str(folder),
                             path_in_repo=path_in_repo or None, commit_message=commit_message or "upload")
    sha = getattr(info, "oid", None) or api.repo_info(repo_id, repo_type=repo_type).sha
    for t in tags:
        try:
            api.create_tag(repo_id, repo_type=repo_type, tag=t, revision=sha)
        except Exception:  # tag exists: move it to this commit
            api.delete_tag(repo_id, repo_type=repo_type, tag=t)
            api.create_tag(repo_id, repo_type=repo_type, tag=t, revision=sha)
    return hf_url(repo_id, repo_type), sha


def _model_card(arm, depth, step, exp_dir, run_id, extra: dict | None) -> str:
    rn = run_name(arm, depth)
    wb = f"[{run_id}]({wandb_run_url(run_id)})" if run_id else f"wandb project `{wandb_project()}`, run `{rn}`"
    lines = [
        "---", "license: mit",
        "tags: [nanochat, pretraining, contextualization, ai-safety]", "---",
        f"# {rn}_base", "",
        f"Arm **{arm}** base checkpoint of the three-arm contextualization-in-pretraining study on nanochat "
        f"(depth {str(depth).lstrip('d')}), step {step}, built from `{exp_dir}` (`{exp_tag(exp_dir)}`). "
        f"{ARM_DESC.get(arm, '')}", "",
        f"- Training run (metrics, curves): {wb}",
        f"- Checkpoint: `model_{step}.pt` (nanochat `base_train` state dict), `meta_{step}.json`",
        "- Tokenizer: `tokenizer/` (shared across arms)",
        f"- Build metadata: `{build_repo()}` (dataset), folder `{exp_tag(exp_dir)}/`",
        "- Eval outputs land under `eval/` once `eval_beliefs.py` has run",
        "", "Each training run is a commit on `main` tagged `wandb-<run_id>`; use the tag to fetch the exact "
        "checkpoint behind a given wandb run.",
    ]
    if extra:
        lines += ["", "## meta", "", "```json", json.dumps(extra, indent=2)[:4000], "```"]
    return "\n".join(lines) + "\n"


def hf_upload_checkpoint(exp_dir, arm, depth, run_id: str | None = None, step: int | None = None,
                         with_optimizer: bool = False, extra_tags: tuple[str, ...] = ()) -> tuple[str, str, str]:
    """Upload <exp_dir>/arm_<arm>/base_checkpoints/d<depth>/{model,meta}_<step> + tokenizer.
    Returns (repo_url, commit_sha, step)."""
    d = str(depth) if str(depth).startswith("d") else f"d{depth}"
    arm_dir = Path(exp_dir) / f"arm_{arm}"
    ckpt_dir = arm_dir / "base_checkpoints" / d
    models = sorted(ckpt_dir.glob("model_*.pt"))
    if not models:
        raise SystemExit(f"no model_*.pt under {ckpt_dir}")
    if step is not None:
        model_pt = ckpt_dir / f"model_{step:06d}.pt"
    else:
        model_pt = models[-1]
    step_s = model_pt.stem.split("_")[-1]
    meta_json = ckpt_dir / f"meta_{step_s}.json"
    tok_dir = arm_dir / "tokenizer"
    if not tok_dir.is_dir():
        raise SystemExit(f"tokenizer dir missing: {tok_dir}")
    meta = json.loads(meta_json.read_text()) if meta_json.exists() else None

    repo_id = model_repo(arm, depth)
    with tempfile.TemporaryDirectory(dir=arm_dir) as stage:
        os.symlink(model_pt.resolve(), os.path.join(stage, model_pt.name))
        if meta_json.exists():
            shutil.copy(meta_json, stage)
        shutil.copytree(tok_dir, os.path.join(stage, "tokenizer"))
        if with_optimizer:
            for opt in sorted(ckpt_dir.glob(f"optim_{step_s}_rank*.pt")):
                os.symlink(opt.resolve(), os.path.join(stage, opt.name))
        Path(stage, "README.md").write_text(_model_card(arm, depth, step_s, str(exp_dir), run_id, meta))
        tags = tuple(t for t in [f"wandb-{run_id}" if run_id else None, *extra_tags] if t)
        url, sha = hf_upload_dir(repo_id, stage, "model",
                                 commit_message=f"arm {arm} {d} step {step_s} from {exp_tag(exp_dir)}"
                                                + (f" (wandb {run_id})" if run_id else ""),
                                 tags=tags)
    return url, sha, step_s


def hf_upload_build(exp_dir, extra_files: tuple[str, ...] = ()) -> tuple[str, str]:
    """Upload probe_sets/ + build_summary.json + manifest.csv.gz to <build_repo>/<exp_tag>/."""
    exp_dir = Path(exp_dir)
    tag = exp_tag(exp_dir)
    manifest, manifest_gz = exp_dir / "manifest.csv", exp_dir / "manifest.csv.gz"
    if manifest.exists() and (not manifest_gz.exists() or manifest_gz.stat().st_mtime < manifest.stat().st_mtime):
        with open(manifest, "rb") as fin, gzip.open(manifest_gz, "wb", compresslevel=6) as fout:
            shutil.copyfileobj(fin, fout)
    with tempfile.TemporaryDirectory(dir=exp_dir) as stage:
        if (exp_dir / "probe_sets").is_dir():
            shutil.copytree(exp_dir / "probe_sets", os.path.join(stage, "probe_sets"))
        for f in [exp_dir / "build_summary.json", manifest_gz, *map(Path, extra_files)]:
            if f.exists():
                shutil.copy(f, stage)
        summary = exp_dir / "build_summary.json"
        cfg = json.loads(summary.read_text()).get("config", {}) if summary.exists() else {}
        Path(stage, "README.md").write_text("\n".join([
            f"# {tag}", "",
            f"Build metadata from `{exp_dir}`: `probe_sets/` (belief probes), `manifest.csv.gz` "
            "(per-document injection manifest), `build_summary.json`.", "",
            "Checkpoints trained on this build: " + ", ".join(
                f"`{hf_namespace()}/ctx_{a}_d<depth>_base`" for a in "CRX") + f" (commits tagged with this build's wandb runs).",
            "", "```json", json.dumps(cfg, indent=2)[:4000], "```",
        ]) + "\n")
        url, sha = hf_upload_dir(build_repo(), stage, "dataset", path_in_repo=tag,
                                 commit_message=f"build metadata {tag}", tags=(tag,))
    return hf_url(build_repo(), "dataset") + f"/tree/main/{tag}", sha


def hf_upload_files(repo_id: str, files: dict[str, str | os.PathLike], repo_type: str = "model",
                    commit_message: str = "", tags: tuple[str, ...] = ()) -> tuple[str, str]:
    """Upload a few files as one commit: {path_in_repo: local_path}."""
    with tempfile.TemporaryDirectory() as stage:
        for rel, src in files.items():
            dst = Path(stage) / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
        return hf_upload_dir(repo_id, stage, repo_type, commit_message=commit_message, tags=tags)
