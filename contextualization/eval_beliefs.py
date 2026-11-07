#!/usr/bin/env python
"""eval_beliefs.py -- score the belief probes against one arm's base checkpoint (eval schema v2).

For every probe fact we run a paired contrast: both candidates are rendered from the SAME
structured template with explicit answer spans (ctx/probes.py), tokenized under a JOINTLY chosen
conditioning prefix, and scored in one forward pass each. Per item we record

    logp_value / logp_competing          answer-plus-suffix (cloze passes; the historical
                                         definition) or full sentence (heldout_paraphrase)
    logp_value_answer / *_competing_answer   answer-only sub-sums (tokens overlapping the slot)
    *_suffix                             the remainder
    boundary_k, n_ctx_tokens, boundary_adjusted   the shared tokenized conditioning prefix

together with the exact context/candidates/spans/template id. The belief score used
downstream stays logp_value - logp_competing (unchanged definition); answer-only and
full-sentence metrics are reported separately by analyze_beliefs.py.

Passes:
  cloze_neutral        the bare factual-register cloze (headline measurement)
  cloze_attributed     "According to one source: " + the same cloze
  cloze_attributed_fixedsource   same with the fact's own consistent source (source policy per_fact)
  wrapper_conditional  the fact's ACTUAL occurrence-0 Arm-X training rendering cut at the value
                       (rendered under the build's protocol; v1 builds reproduce the historical
                       text byte for byte, including lowercased title-initial values)
  heldout_paraphrase   the reserved (never-trained) frame vs the same frame with the competing
                       value in EVERY slot (full-sentence metric)

Coverage is asserted: every injected fact must yield every pass; exclusions are listed with a
reason in the .meta.json and the run fails unless --allow-exclusions is given.

Usage (repo root, nanochat env):
  uv run python contextualization/eval_beliefs.py --exp-dir /workspace/ctx_v2_build1 --arm C --step 8352
  # plumbing test without torch/checkpoint (mock scores, char tokenizer):   --dry-run
  # plumbing test with a REAL BPE tokenizer but mock scores:                --dry-run --tokenizer-dir <base dir>
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctx import probes as P  # noqa: E402
from ctx import tracking as T  # noqa: E402
from ctx.protocol import (EVAL_SCHEMA, PROTOCOL_V1, ProtocolError, code_revision,  # noqa: E402
                          load_summary, render_config_from_summary, scored_item_identity, sha256_file,
                          sha256_text, tokenizer_fingerprint)

ATTRIB_PREFIX = P.ATTRIB_PREFIX
_FactView = P.FactView

CSV_COLS = ["arm", "probe_set", "fact_id", "pass", "template_idx", "truth_value", "entity_tier",
            "assigned_frequency", "is_frame0", "logp_value", "logp_competing", "n_tok_value", "n_tok_competing",
            "schema_version", "group_id", "role", "ground_truth", "template_id", "metric",
            "logp_value_answer", "logp_competing_answer", "n_tok_value_answer", "n_tok_competing_answer",
            "logp_value_suffix", "logp_competing_suffix",
            "boundary_k", "n_ctx_tokens", "boundary_adjusted", "answer_suffix_merged_value", "answer_suffix_merged_competing",
            "answer_lead_merged_value", "answer_lead_merged_competing",
            "value_span_chars", "context", "cand_value", "cand_competing", "scoring_policy",
            "legacy_malformed", "historical_rendering", "item_sha256"]


def load_jsonl(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"probe file missing: {path}")
    return [json.loads(l) for l in open(path) if l.strip()]


def build_items(injected, heldout, build_cfg=None, limit=None, historical_emulation=None):
    """LEGACY-SHAPED item builder kept for reviews/2026-10-13/reproduce_eval_audit.py.

    Returns dicts with probe_set/fact_id/pass/template_idx/context/cand_value/cand_competing.
    When build_cfg has no 'protocol' key (a historical call) the HISTORICAL construction is
    emulated so the audit can line its items up with the archived CSVs: the wrapper probe is cut
    at the last case-sensitive occurrence of the value and OMITTED when that fails (the 63
    title-initial author/work facts), and the paraphrase alternative is the historical
    first-occurrence str.replace (104 malformed). Pass historical_emulation=False (or a
    build_cfg with 'protocol') to get the corrected construction. The evaluator itself never
    uses this function; it uses ctx.probes.build_items."""
    cfg = dict(build_cfg or {})
    if historical_emulation is None:
        historical_emulation = "protocol" not in cfg
    summary = {"config": cfg, "protocol": cfg.get("protocol")}
    render_cfg = render_config_from_summary(summary)
    items, excl, expected = P.build_items(injected, heldout, render_cfg, limit)
    if historical_emulation:
        print("NOTE  eval_beliefs.build_items: emulating the HISTORICAL probe construction (wrapper omissions, "
              "first-occurrence paraphrase replacement) for audit reproduction only", file=sys.stderr)
    recs = {r["fact_id"]: r for r in injected}
    out = []
    for it in items:
        d = {"probe_set": it.probe_set, "fact_id": it.fact_id, "pass": it.pass_name,
             "template_idx": it.template_idx, "truth_value": it.truth_value, "entity_tier": it.entity_tier,
             "assigned_frequency": it.assigned_frequency, "context": it.context,
             "cand_value": it.cand_value, "cand_competing": it.cand_competing, "is_frame0": it.is_frame0}
        if historical_emulation and it.pass_name == "wrapper_conditional":
            r = recs[it.fact_id]
            full = it.context + it.cand_value
            i = full.rfind(str(r["value"]))
            if i <= 0:
                continue  # historical silent omission
            tail = full[i + len(str(r["value"])):]
            d.update(context=full[:i], cand_value=str(r["value"]) + tail, cand_competing=str(r["competing_value"]) + tail)
        elif historical_emulation and it.pass_name == "heldout_paraphrase":
            lc = P.legacy_paraphrase_candidates(recs[it.fact_id])
            if lc is None:
                continue
            d.update(cand_value=lc[0], cand_competing=lc[1])
        out.append(d)
    return out


# --- scoring ---------------------------------------------------------------------------------

def score_batched(seqs, model, device, batch_size):
    """seqs: list of ScoredSeq. Returns, per sequence, the list of per-token log-probs of
    ids[start:] given ids[:start] (one forward pass per sequence, right-padded batches)."""
    import torch
    import torch.nn.functional as F
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i].ids))
    out = [None] * len(seqs)
    with torch.inference_mode():
        for lo in range(0, len(order), batch_size):
            idxs = order[lo:lo + batch_size]
            maxlen = max(len(seqs[i].ids) for i in idxs)
            x = torch.zeros((len(idxs), maxlen), dtype=torch.long, device=device)
            for r, i in enumerate(idxs):
                x[r, :len(seqs[i].ids)] = torch.tensor(seqs[i].ids, dtype=torch.long)
            logp = F.log_softmax(model(x).float(), dim=-1)
            for r, i in enumerate(idxs):
                ids, start = seqs[i].ids, seqs[i].start
                pos = torch.arange(start - 1, len(ids) - 1, device=device)
                tgt = torch.tensor(ids[start:], dtype=torch.long, device=device)
                out[i] = logp[r, pos, tgt].tolist()
    return out


def score_dry(seqs):
    """Deterministic mock per-token scores (plumbing test only, no torch)."""
    import zlib
    out = []
    for s in seqs:
        out.append([-(zlib.crc32(bytes(f"{t}:{i}", "utf8")) % 1000) / 100.0 - 1.0 for i, t in enumerate(s.ids[s.start:])])
    return out


class DryTokenizer:
    """Character-level stand-in (one token per character, id = codepoint + 1); NOT a BPE
    tokenizer, so it cannot exercise merge boundaries (use --tokenizer-dir for real BPE)."""

    def get_bos_token_id(self):
        return 0

    def encode(self, text):
        return [ord(c) + 1 for c in text]

    def decode(self, ids):
        return "".join(chr(i - 1) for i in ids if i > 0)

    def decode_single_token_bytes(self, i):
        return chr(i - 1).encode("utf-8") if i > 0 else b""


# --- main ------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", required=True, help="build dir with probe_sets/, build_summary.json and arm_<ARM>/")
    ap.add_argument("--arm", required=True, choices=["C", "R", "X", "E"])
    ap.add_argument("--model-tag", default=None, help="e.g. d24 (default: largest present)")
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: last; recorded in meta)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="only the first N facts per probe file (smoke test)")
    ap.add_argument("--out-csv", default=None,
                    help="default: <exp-dir>/eval/beliefs_arm<ARM>.csv for the default model tag d<depth>, "
                         "<exp-dir>/eval/seed_<S>/beliefs_arm<ARM>.csv for a replicate tag d<depth>_s<S> and "
                         "<exp-dir>/eval/<tag>/... for any other --model-tag (one evaluation per checkpoint, never overwritten)")
    ap.add_argument("--probe-dir", default=None, help="default: <exp-dir>/probe_sets")
    ap.add_argument("--dry-run", action="store_true", help="mock scores; char tokenizer unless --tokenizer-dir")
    ap.add_argument("--tokenizer-dir", default=None, help="nanochat base dir holding tokenizer/ (dry-run with real BPE)")
    ap.add_argument("--allow-exclusions", action="store_true", help="do not fail when coverage is incomplete")
    ap.add_argument("--no-hash-checkpoint", action="store_true", help="skip sha256 of the model file (slow for 4 GB)")
    ap.add_argument("--allow-unverified-upload", action="store_true",
                    help="upload an evaluation whose checkpoint sha could not be verified against the HF tag")
    ap.add_argument("--allow-build-mismatch", action="store_true",
                    help="score a checkpoint whose recorded build/probe bindings differ from the probes used (intentional "
                         "historical rescoring); the mismatch is recorded in the meta json")
    ap.add_argument("--no-tracking", action="store_true", help="don't log to wandb / upload the CSV to HF")
    ap.add_argument("--wandb-run-id", default=None)
    ap.add_argument("--hf-repo", default=None)
    args = ap.parse_args(argv)
    T.load_env()
    t0 = time.time()

    pdir = os.path.abspath(args.probe_dir or os.path.join(args.exp_dir, "probe_sets"))
    inj_path, held_path = os.path.join(pdir, "injected_facts.jsonl"), os.path.join(pdir, "heldout_facts.jsonl")
    injected = load_jsonl(inj_path)
    heldout = load_jsonl(held_path)
    summary = load_summary(args.exp_dir)
    summary_path = os.path.join(args.exp_dir, "build_summary.json")
    # hashes of the files ACTUALLY opened (review 2026-10-14 finding 19)
    opened = {"probe_sets/injected_facts.jsonl": inj_path, "probe_sets/heldout_facts.jsonl": held_path,
              "build_summary.json": summary_path if os.path.exists(summary_path) else None}
    input_hashes = {k: (sha256_file(v) if v else None) for k, v in opened.items()}
    input_paths = {k: v for k, v in opened.items()}
    render_cfg = render_config_from_summary(summary)  # raises ProtocolError when not reconstructible
    if render_cfg["assumed_defaults"]:
        print(f"NOTE  legacy build config: assuming historical defaults for {render_cfg['assumed_defaults']}")
    items, exclusions, expected = P.build_items(injected, heldout, render_cfg, args.limit)
    cov = P.coverage_report(items, expected, exclusions)
    for r in cov["rows"]:
        flag = "" if r["expected"] == r["got"] else "   <-- INCOMPLETE"
        print(f"  {r['probe_set']:>11}:{r['pass']:<28} expected {r['expected']:>6} got {r['got']:>6}{flag}")
    if not cov["complete"] and not args.allow_exclusions:
        for e in exclusions[:20]:
            print(f"  excluded {e.probe_set}/{e.fact_id}/{e.pass_name}: {e.reason}")
        raise SystemExit(f"probe coverage incomplete ({cov['n_exclusions']} exclusions, {cov['duplicates']} duplicates); "
                         f"pass --allow-exclusions to score anyway")
    print(f"probes: {len(injected)} injected + {len(heldout)} held-out records -> {len(items)} contrasts "
          f"({2 * len(items)} scored sequences); protocol {render_cfg['protocol']}")

    meta_ckpt = {}
    build_mismatch = {}
    arm_dir = os.path.join(args.exp_dir, f"arm_{args.arm}")
    if args.dry_run:
        model, device = None, None
        if args.tokenizer_dir:
            os.environ["NANOCHAT_BASE_DIR"] = args.tokenizer_dir
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from nanochat.tokenizer import get_tokenizer
            tokenizer = get_tokenizer()
            tok_fp = tokenizer_fingerprint(args.tokenizer_dir)
        else:
            tokenizer, tok_fp = DryTokenizer(), {"tokenizer_sha256": None, "note": "char-level dry tokenizer (not BPE)"}
    else:
        os.environ["NANOCHAT_BASE_DIR"] = arm_dir
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import torch
        from nanochat.checkpoint_manager import load_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, tokenizer, meta_ckpt = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
        tok_fp = tokenizer_fingerprint(arm_dir)
        bindings = meta_ckpt.get("bindings") or {}
        bound = bindings.get("tokenizer_sha256")
        if bound and bound != tok_fp["tokenizer_sha256"]:
            raise SystemExit(f"tokenizer under {arm_dir} ({tok_fp['tokenizer_sha256'][:12]}) is not the one the "
                             f"checkpoint was trained with ({bound[:12]})")
        # checkpoint <-> build/probe compatibility (finding 19): a bound checkpoint must be scored
        # on the probes of its own build unless the mismatch is explicitly allowed
        build_mismatch = {}
        if bindings.get("build_summary_sha256") and input_hashes.get("build_summary.json") \
                and bindings["build_summary_sha256"] != input_hashes["build_summary.json"]:
            build_mismatch["build_summary_sha256"] = (bindings["build_summary_sha256"], input_hashes["build_summary.json"])
        ph = bindings.get("probe_hashes") or {}
        for rel in ("probe_sets/injected_facts.jsonl", "probe_sets/heldout_facts.jsonl"):
            want = ph.get(rel) or ph.get(os.path.basename(rel))
            if want and input_hashes.get(rel) and want != input_hashes[rel]:
                build_mismatch[rel] = (want, input_hashes[rel])
        if build_mismatch and not args.allow_build_mismatch:
            raise SystemExit("checkpoint bindings differ from the probes/build being scored: "
                             + "; ".join(f"{k}: trained {a[:12]} vs now {b[:12]}" for k, (a, b) in build_mismatch.items())
                             + " (pass --allow-build-mismatch for an intentional historical rescoring)")
        if not bindings:
            print("NOTE  checkpoint carries no bindings (legacy meta): build/probe compatibility cannot be verified")
        print(f"loaded arm_{args.arm} checkpoint (step {meta_ckpt.get('step', '?')}) on {device}")

    bos = tokenizer.get_bos_token_id()
    seqs, bmeta = [], []
    for it in items:
        sv, sc, m = P.tokenize_pair(tokenizer, bos, it)
        seqs += [sv, sc]
        bmeta.append(m)
    n_adj = sum(m["boundary_adjusted"] for m in bmeta)
    if n_adj:
        print(f"NOTE  {n_adj}/{len(items)} items use a conditioning prefix shorter than the context's own tokenization "
              f"(joint boundary; recorded per row)")

    per_tok = score_dry(seqs) if args.dry_run else score_batched(seqs, model, device, args.batch_size)

    # outputs are scoped to the checkpoint's model tag (recheck finding 6): replicates never collide
    out_csv = args.out_csv or str(T.eval_output_dir(args.exp_dir, args.model_tag) / f"beliefs_arm{args.arm}.csv")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    if os.path.exists(out_csv):
        raise SystemExit(f"{out_csv} exists; evaluation outputs are versioned -- pass --out-csv to a new path")
    tmp_csv = out_csv + ".tmp"
    identities = []   # canonical identity of every scored contrast (texts, spans, boundary); summarized in the meta
    with open(tmp_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        for k, it in enumerate(items):
            sv, sc = seqs[2 * k], seqs[2 * k + 1]
            a, b = P.split_scores(per_tok[2 * k], sv), P.split_scores(per_tok[2 * k + 1], sc)
            m = bmeta[k]
            spans = json.dumps([list(s) for s in it.answer_spans_value])
            ident = scored_item_identity(it.context, it.cand_value, it.cand_competing, spans, it.metric, it.template_id,
                                         P.SCORING_POLICY, m["boundary_k"], m["n_ctx_tokens"])
            identities.append(f"{it.probe_set}/{it.pass_name}/{it.fact_id}/{it.template_idx}:{ident}")
            w.writerow({
                "arm": args.arm, "probe_set": it.probe_set, "fact_id": it.fact_id, "pass": it.pass_name,
                "template_idx": it.template_idx, "truth_value": it.truth_value, "entity_tier": it.entity_tier,
                "assigned_frequency": it.assigned_frequency, "is_frame0": it.is_frame0,
                "logp_value": round(a["full"], 5), "logp_competing": round(b["full"], 5),
                "n_tok_value": a["n_tok"], "n_tok_competing": b["n_tok"],
                "schema_version": EVAL_SCHEMA, "group_id": it.group_id, "role": it.role, "ground_truth": it.ground_truth,
                "template_id": it.template_id, "metric": it.metric,
                "logp_value_answer": round(a["answer"], 5), "logp_competing_answer": round(b["answer"], 5),
                "n_tok_value_answer": a["n_tok_answer"], "n_tok_competing_answer": b["n_tok_answer"],
                "logp_value_suffix": round(a["suffix"], 5), "logp_competing_suffix": round(b["suffix"], 5),
                "boundary_k": m["boundary_k"], "n_ctx_tokens": m["n_ctx_tokens"], "boundary_adjusted": m["boundary_adjusted"],
                "answer_suffix_merged_value": m.get("answer_suffix_merged_value", 0),
                "answer_suffix_merged_competing": m.get("answer_suffix_merged_competing", 0),
                "answer_lead_merged_value": m.get("answer_lead_merged_value", 0),
                "answer_lead_merged_competing": m.get("answer_lead_merged_competing", 0),
                "value_span_chars": spans,
                "context": it.context, "cand_value": it.cand_value, "cand_competing": it.cand_competing,
                "scoring_policy": P.SCORING_POLICY, "legacy_malformed": it.legacy_malformed,
                "historical_rendering": it.historical_rendering, "item_sha256": ident,
            })
    os.replace(tmp_csv, out_csv)
    print(f"wrote {len(items)} rows -> {out_csv}")

    ckpt_info = checkpoint_provenance(meta_ckpt, args.model_tag)
    if not args.dry_run and meta_ckpt.get("step") is not None:
        step = int(meta_ckpt["step"])
        tag = args.model_tag or f"d{meta_ckpt['model_config']['n_layer']}"
        mp = os.path.join(arm_dir, "base_checkpoints", tag, f"model_{step:06d}.pt")
        ckpt_info["model_path"] = mp
        ckpt_info["model_sha256"] = None if args.no_hash_checkpoint or not os.path.exists(mp) else sha256_file(mp)
    run_identity = resolve_run_identity(args, meta_ckpt)
    meta = {
        "schema_version": EVAL_SCHEMA, "arm": args.arm, "protocol": render_cfg["protocol"], "render_config": render_cfg,
        "scoring_policy": P.SCORING_POLICY,
        # candidate-construction semantics version (recheck finding 3); an unversioned constructor
        # (e.g. the 031c4e5 module swapped in by a reproduction) is labelled as such, never omitted
        "probe_construction": getattr(P, "PROBE_CONSTRUCTION_VERSION", "structured_v2.0 (unversioned constructor)"),
        "items_sha256": sha256_text("\n".join(sorted(identities))),   # identity of the whole scored item set
        "model_tag": args.model_tag, "run": run_identity,
        "metrics": {"logp_value": "answer_plus_suffix for cloze passes, full_sentence for heldout_paraphrase",
                    "logp_value_answer": "answer-only sub-sum", "logp_value_suffix": "logp_value - logp_value_answer"},
        "mode": "dry_run_mock_scores" if args.dry_run else "model_inference",
        "tokenizer": tok_fp, "checkpoint": ckpt_info, "code": code_revision(),
        "input_hashes": input_hashes, "input_paths": input_paths, "probe_dir": pdir, "limit": args.limit,
        "build_mismatch": {k: {"checkpoint": a, "scored": b} for k, (a, b) in build_mismatch.items()},
        "build_mismatch_allowed": bool(args.allow_build_mismatch),
        "n_items": len(items), "n_sequences": len(seqs), "coverage": cov,
        "exclusions": [vars(e) for e in exclusions], "boundary_adjusted_items": n_adj,
        "out_csv": out_csv, "csv_sha256": sha256_file(out_csv), "seconds": round(time.time() - t0, 1),
    }
    meta_path = os.path.splitext(out_csv)[0] + ".meta.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"meta -> {meta_path}")

    if not args.dry_run and not args.no_tracking and T.tracking_enabled():
        track_eval(args, out_csv, items, per_tok, seqs, meta_ckpt, meta_path, run_identity)
    return out_csv


def checkpoint_provenance(meta_ckpt: dict, model_tag=None) -> dict:
    """What the analysis needs to know about the checkpoint that was scored (recheck finding 3):
    the initialization identity (init_param_sha256, init_seed, cli_init_seed), the resume
    provenance, the preflight verdict and the training budget (step, total_batch_size, horizon,
    sequence length, device batch, accumulation, world size). Missing keys stay None (legacy metas)."""
    uc = meta_ckpt.get("user_config") or {}

    def pick(key, *fallback):
        v = meta_ckpt.get(key)
        if v is None:
            for f in fallback:
                v = f
                if v is not None:
                    break
        return v

    def positive(v):
        try:
            return int(v) if v is not None and int(v) >= 0 else None
        except (TypeError, ValueError):
            return None

    return {"step": meta_ckpt.get("step"), "model_config": meta_ckpt.get("model_config"),
            "bindings": meta_ckpt.get("bindings"), "init_seed": meta_ckpt.get("init_seed"),
            "cli_init_seed": meta_ckpt.get("cli_init_seed"), "init_param_sha256": meta_ckpt.get("init_param_sha256"),
            "resume_from_step": meta_ckpt.get("resume_from_step"), "preflight": meta_ckpt.get("preflight"),
            "loss_normalization": meta_ckpt.get("loss_normalization"),   # None = unrecorded = the historical microbatch_mean
            "data_seed": meta_ckpt.get("data_seed"), "data_loader": meta_ckpt.get("data_loader"),
            "model_tag": model_tag,
            "total_batch_size": positive(pick("total_batch_size", uc.get("total_batch_size"))),
            "num_iterations": positive(pick("num_iterations", uc.get("num_iterations"))),
            "max_seq_len": positive(pick("max_seq_len", uc.get("max_seq_len"))),
            "device_batch_size": positive(pick("device_batch_size", uc.get("device_batch_size"))),
            "grad_accum_steps": positive(meta_ckpt.get("grad_accum_steps")),
            "world_size": positive(pick("world_size", (meta_ckpt.get("dataloader_state_dict") or {}).get("world_size"),
                                        (meta_ckpt.get("nondeterminism") or {}).get("world_size"))),
            "wandb_run": meta_ckpt.get("wandb_run")}


def resolve_run_identity(args, meta_ckpt: dict) -> dict:
    """The training run an evaluation belongs to, taken from the CHECKPOINT it scored (meta_<step>.json
    wandb_run) and only then from the model tag's run_info file (recheck finding 6); --wandb-run-id
    overrides both. A run_info that names another run than the checkpoint is reported."""
    from_ckpt = (meta_ckpt.get("wandb_run") or {}).get("id")
    info = T.read_run_info(args.exp_dir, args.arm, args.model_tag)
    from_info = info.get("wandb_run_id")
    if args.wandb_run_id:
        rid, source = args.wandb_run_id, "--wandb-run-id"
    elif from_ckpt:
        rid, source = from_ckpt, "checkpoint meta (wandb_run.id)"
    elif from_info:
        rid, source = from_info, f"run_info ({T.run_info_path(args.exp_dir, args.arm, args.model_tag).name})"
    else:
        rid, source = None, "none"
    out = {"wandb_run_id": rid, "source": source, "checkpoint_run_id": from_ckpt, "run_info_run_id": from_info,
           "run_info_path": str(T.run_info_path(args.exp_dir, args.arm, args.model_tag)) if info else None}
    if from_ckpt and from_info and from_ckpt != from_info:
        out["conflict"] = f"checkpoint was trained by wandb run {from_ckpt} but run_info names {from_info}"
        print(f"WARNING: {out['conflict']}; using {rid} ({source})")
    return out


def summarize(items, per_tok, seqs):
    """Mean belief (logp_value - logp_competing) per (probe_set, pass, truth_value), plus n."""
    acc = {}
    for k, it in enumerate(items):
        key = (it.probe_set, it.pass_name, str(it.truth_value))
        acc.setdefault(key, []).append(sum(per_tok[2 * k]) - sum(per_tok[2 * k + 1]))
    rows = [{"probe_set": ps, "pass": pa, "truth_value": tv, "mean_belief": sum(v) / len(v), "n": len(v)}
            for (ps, pa, tv), v in sorted(acc.items())]
    flat = {f"belief/{r['probe_set']}/{r['pass']}/{r['truth_value']}": r["mean_belief"] for r in rows}
    return rows, flat


def track_eval(args, out_csv, items, per_tok, seqs, meta_ckpt, meta_path, run_identity=None):
    """Log belief summary to the arm's wandb training run (the run recorded in the checkpoint meta);
    archive CSV + meta next to the checkpoint in its HF model repo."""
    info = T.read_run_info(args.exp_dir, args.arm, args.model_tag)
    run_identity = run_identity or resolve_run_identity(args, meta_ckpt)
    run_id = run_identity["wandb_run_id"]
    depth = args.model_tag or (f"d{meta_ckpt['model_config']['n_layer']}" if meta_ckpt.get("model_config") else None) \
        or info.get("model_tag") or info.get("depth")
    rows, flat = summarize(items, per_tok, seqs)
    summary_json = os.path.splitext(out_csv)[0] + "_summary.json"
    with open(summary_json, "w") as fh:
        json.dump({"arm": args.arm, "step": meta_ckpt.get("step"), "wandb_run_id": run_id, "rows": rows}, fh, indent=2)
    hf_url = None
    if T.hf_enabled() and depth is not None and run_id:
        repo = args.hf_repo or T.model_repo(args.arm, depth)
        files = {f"eval/{os.path.basename(p)}": p for p in (out_csv, summary_json, meta_path)}
        meta_eval = json.load(open(meta_path))
        model_sha = (meta_eval.get("checkpoint") or {}).get("model_sha256")
        # never move the run's checkpoint tag: the evaluation goes to a branch/tag derived from
        # the verified checkpoint commit (review 2026-10-14 finding 1)
        up = T.hf_upload_eval(repo, files, run_id, model_sha, step=str(meta_ckpt.get("step")),
                              allow_unverified=getattr(args, "allow_unverified_upload", False),
                              commit_message=f"eval arm {args.arm} step {meta_ckpt.get('step')} (wandb {run_id})")
        hf_url = up["url"]
        with open(os.path.splitext(out_csv)[0] + ".upload.json", "w") as fh:
            json.dump(up, fh, indent=2, default=str)
        print(f"eval CSV -> {hf_url} branch {up['branch']} tag {up['tag']} (commit {up['commit_sha'][:8]}, "
              f"checkpoint commit {up['checkpoint_commit'][:8]}, verified={up['verified']})")
    elif T.hf_enabled():
        print("NOTE  HF upload skipped: needs the model depth (--model-tag/--hf-repo) and a wandb run id")
    if T.wandb_enabled() and run_id:
        import wandb
        run = T.wandb_resume(run_id, job_type="train")
        run.summary.update({**flat, "eval/n_rows": len(items), "eval/step": meta_ckpt.get("step")})
        run.log({"belief_summary": wandb.Table(columns=list(rows[0].keys()), data=[list(r.values()) for r in rows])}
                if rows else {})
        if hf_url:
            run.config.update({"hf_eval": f"{hf_url}/tree/main/eval"}, allow_val_change=True)
        run.finish()
        print(f"eval metrics -> wandb run {run_id} ({T.wandb_run_url(run_id)})")
    elif T.wandb_enabled():
        print("NOTE  no wandb run id (none in the checkpoint meta, no run_info for this model tag, no --wandb-run-id); "
              "metrics not logged")


if __name__ == "__main__":
    main()
