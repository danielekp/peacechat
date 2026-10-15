"""Post-fix equivalent of reviews/2026-10-14-recheck/repro/ctx_data_rereview_repro.py (findings 7 and 8).

The reviewer's script reaches into the old closure-based validator fallback (`tok.encode.__closure__`)
and therefore cannot complete after the fix; this script computes the same quantities through the
public API: (1) the validator's pickle fallback on the historical tokenizer must report the TRUE
vocabulary and select the absolute limits (the 9-token delta is rejected), (2) regenerating probes
after deleting novelty_audit.json must change NO label and name its evidence, (3) design_cells is in
the summary. Run from the repo root with the CPU test environment; exits non-zero on any regression.
"""
import builtins
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "contextualization"), str(ROOT)]
SRC = Path.home() / ".cache" / "ctx_wandb_to_hf" / "ctx_C_d24_base"

import validate  # noqa: E402
from ctx import blend, factpool_v2 as F, templates as T  # noqa: E402
from ctx.tokenizers import token_limit_mode, vocab_size_of  # noqa: E402
from build_dataset import parse_args  # noqa: E402

ok = True
if (SRC / "tokenizer" / "tokenizer.pkl").exists():
    orig_import = builtins.__import__

    def missing_nanochat(name, *args, **kwargs):
        if name == "nanochat.tokenizer":
            raise ModuleNotFoundError("No module named 'tokenizers'")
        return orig_import(name, *args, **kwargs)

    with patch("builtins.__import__", missing_nanochat):
        tok, how = validate._load_tokenizer(str(SRC / "tokenizer"))
    g = F.build_semantic_pool(2500, 1234)
    tr, he = F.split_groups(g, 0.3, 1234)
    F.assign_injection(tr, [1, 4, 16, 64, 256], 1234)
    c = next(c for c in F.injected_statements(tr) if c.fact_id == "52df0467bbf5445e")
    d_plain = T.xe_diagnostics(T.render_inserts(c, 1, 1234, T.RenderPolicy(), arms=("X", "E")), tok)
    d_aware = T.xe_diagnostics(T.render_inserts(c, 1, 1234, T.RenderPolicy(), arms=("X", "E"), tokenizer=tok), tok)
    vocab = vocab_size_of(tok)
    mode = token_limit_mode(vocab)
    summ = blend._xe_summary([d_plain], vocab)
    out = {"how": how, "reported_vocab": vocab, "limit_mode_selected": mode,
           "delta_tokens_without_token_aware_filler": d_plain["delta_tokens"],
           "passes_absolute_limit": abs(d_plain["delta_tokens"]) <= T.XE_MATCH_LIMITS["max_abs_delta_tokens"],
           "old_fallback_relative_rule_would_accept": abs(d_plain["delta_tokens"]) / d_plain["len_tokens_X"] <= 0.20,
           "xe_summary_within_token_limits": summ["within_token_limits"], "xe_summary_mode": summ["token_limit_mode"],
           "token_aware_rendering": {"delta_tokens": d_aware["delta_tokens"], "delta_claim_tok_start": d_aware["delta_claim_tok_start"]}}
    print("FALLBACK_POLICY (after fix):", json.dumps(out), flush=True)
    ok &= vocab == 32768 and mode == "absolute" and summ["within_token_limits"] is False and abs(d_aware["delta_tokens"]) <= 8
else:
    print("FALLBACK_POLICY: historical tokenizer not available, skipped", flush=True)

root = Path(tempfile.mkdtemp(prefix="ctx_data_recheck_after_"))
out_dir = root / "build"
args = ["--out", str(out_dir), "--cache-dir", str(root / "cache"), "--base-source", "synthetic", "--num-groups", "80",
        "--freq-grid", "1", "4", "16", "--base-chars", "2000000", "--chars-per-shard", "300000", "--seed", "0",
        "--counterfact-limit", "0", "--no-tracking", "--collision-audit"]
s = blend.build(parse_args(args))
read = lambda p: [json.loads(x) for x in p.read_text().splitlines()]  # noqa: E731
old = read(out_dir / "probe_sets" / "heldout_facts.jsonl")
(out_dir / "novelty_audit.json").unlink()
r = blend.build(parse_args(args + ["--probes-only"]))
new = read(out_dir / "probe_sets_regen" / "heldout_facts.jsonl")
changed = sum(a["novelty"] != b["novelty"] for a, b in zip(old, new))
res = {"verified_against_manifest": r["verified_against_manifest"], "verified_against_probe_files": r.get("verified_against_probe_files"),
       "novelty_evidence": r.get("novelty_evidence"), "changed_novelty_labels": changed, "design_cells_in_summary": "design_cells" in s,
       "n_empty_cells": len((s.get("design_cells") or {}).get("empty_cells", [])), "artifacts": str(root)}
print("MISSING_AUDIT (after fix):", json.dumps(res), flush=True)
ok &= changed == 0 and r.get("verified_against_probe_files") is True and "design_cells" in s
print("RESULT", "OK" if ok else "REGRESSION")
sys.exit(0 if ok else 1)
