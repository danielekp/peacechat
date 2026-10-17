"""Protocol / schema versions, provenance helpers, and the legacy-vs-corrected distinction.

The July-2026 run used what we now call protocol v1 (``PROTOCOL_V1``): the generator in
``ctx.factpool`` (``build_fact_pool`` / ``split_pool`` / ``assign_frequencies``) and the
rendering in ``ctx.templates`` with the legacy template sets. Those code paths are frozen and
regression-tested against fixtures captured from the pre-refactor code
(``tests/fixtures/historical/``). Corrected builds use protocol v2.

Nothing in this module regenerates historical artifacts; it only labels, hashes and resolves
configuration so that every later stage can say exactly what it is reading.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROTOCOL_V1 = "v1-legacy-2026-07"
PROTOCOL_V2 = "v2-corrected-2026-10"
PROTOCOLS = (PROTOCOL_V1, PROTOCOL_V2)

BUILDER_VERSION = 2
PROBE_SCHEMA = 2
MANIFEST_SCHEMA = 2
EVAL_SCHEMA = 2
PREPACK_SCHEMA = 1
EXPOSURE_SCHEMA = 1

# render defaults the historical CLI used; a legacy summary that omits them is interpreted with
# these values (and the interpretation is reported, never silent)
LEGACY_RENDER_DEFAULTS = {"register": True, "verbatim_control": False, "source_per_fact": False,
                          "embed": True, "embedding_control": False}


class ProtocolError(RuntimeError):
    """Raised when an artifact cannot be interpreted unambiguously."""


# --------------------------------------------------------------------------------------
# hashing / provenance

def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def scored_item_identity(context, cand_value, cand_competing, value_span_chars="", metric="", template_id="",
                         scoring_policy="", boundary_k="", n_ctx_tokens="") -> str:
    """Canonical identity of ONE scored contrast: sha256 over the exact texts the model saw (context,
    both candidates), the answer spans, the metric, the template id, the scoring policy and the
    shared tokenized conditioning prefix (boundary_k, n_ctx_tokens). Written per row by
    eval_beliefs.py and recomputed from the CSV columns by analyze_beliefs.py, so two evaluations
    that disagree on what was scored under the same template id are caught (recheck finding 3)."""
    payload = [str(context), str(cand_value), str(cand_competing), str(value_span_chars), str(metric),
               str(template_id), str(scoring_policy), str(boundary_k), str(n_ctx_tokens)]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def tokenizer_fingerprint(base_or_tokenizer_dir) -> dict:
    """sha256 of tokenizer.pkl (+ token_bytes.pt) under <dir>/tokenizer or <dir> itself."""
    d = Path(base_or_tokenizer_dir)
    if (d / "tokenizer" / "tokenizer.pkl").exists():
        d = d / "tokenizer"
    pkl = d / "tokenizer.pkl"
    if not pkl.exists():
        raise ProtocolError(f"no tokenizer.pkl under {base_or_tokenizer_dir}")
    out = {"tokenizer_dir": str(d), "tokenizer_pkl_sha256": sha256_file(pkl)}
    tb = d / "token_bytes.pt"
    out["token_bytes_sha256"] = sha256_file(tb) if tb.exists() else None
    out["tokenizer_sha256"] = out["tokenizer_pkl_sha256"]
    return out


def code_revision(repo_root=None) -> dict:
    """git commit / dirty flag / branch of the checkout (best effort; never raises)."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    info = {"git_commit": None, "git_dirty": None, "git_branch": None, "builder_version": BUILDER_VERSION,
            "argv": list(sys.argv), "cwd": os.getcwd(), "python": sys.version.split()[0]}
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, timeout=20)
        r = run("rev-parse", "HEAD")
        if r.returncode == 0:
            info["git_commit"] = r.stdout.strip()
            info["git_branch"] = run("rev-parse", "--abbrev-ref", "HEAD").stdout.strip() or None
            st = run("status", "--porcelain", "--untracked-files=no")
            info["git_dirty"] = bool(st.stdout.strip()) if st.returncode == 0 else None
    except Exception:
        pass
    return info


# --------------------------------------------------------------------------------------
# config resolution

def protocol_of(summary: dict | None) -> str:
    """Protocol of a build_summary.json dict (legacy summaries have none -> v1)."""
    if not summary:
        raise ProtocolError("build_summary.json missing: cannot determine the protocol")
    p = summary.get("protocol") or (summary.get("config") or {}).get("protocol")
    if p is None:
        return PROTOCOL_V1
    if p not in PROTOCOLS:
        raise ProtocolError(f"unknown protocol {p!r}; known: {PROTOCOLS}")
    return p


def render_config_from_summary(summary: dict | None) -> dict:
    """Everything needed to re-derive training renderings. Raises ProtocolError when the
    summary cannot support that; legacy summaries get the historical CLI defaults for missing
    booleans and the list of assumed keys is returned under 'assumed_defaults'."""
    if not summary:
        raise ProtocolError("build_summary.json missing: cannot reconstruct training renderings")
    cfg = dict(summary.get("config") or {})
    protocol = protocol_of(summary)
    if cfg.get("seed") is None:
        raise ProtocolError("build config has no 'seed': training renderings cannot be reconstructed")
    out = {"protocol": protocol, "seed": int(cfg["seed"]), "assumed_defaults": []}
    if protocol == PROTOCOL_V1:
        for k, v in LEGACY_RENDER_DEFAULTS.items():
            if k in cfg:
                out[k] = cfg[k]
            else:
                out[k] = v
                out["assumed_defaults"].append(k)
    else:
        rp = cfg.get("render_policy")
        if not rp:
            raise ProtocolError("v2 build config has no 'render_policy'; cannot reconstruct renderings")
        out["render_policy"] = rp
        for k in ("register", "verbatim_control", "source_per_fact", "embed", "embedding_control"):
            out[k] = cfg.get(k, LEGACY_RENDER_DEFAULTS[k])
    return out


def load_summary(exp_dir) -> dict | None:
    p = Path(exp_dir) / "build_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def input_hashes_for_probes(exp_dir) -> dict:
    """sha256 of the probe files and build summary of an experiment dir (for eval/analysis meta)."""
    exp_dir = Path(exp_dir)
    out = {}
    for rel in ("probe_sets/injected_facts.jsonl", "probe_sets/heldout_facts.jsonl", "build_summary.json",
                "manifest.csv"):
        p = exp_dir / rel
        out[rel] = sha256_file(p) if p.exists() else None
    return out
