#!/usr/bin/env python
"""eval_beliefs.py -- score the belief probes against one arm's base checkpoint.

For every probe fact we run a paired cloze contrast: split each cloze template at the
"___" blank, and compare the model's log-probability of the fact's `value` vs its
`competing_value` (the paired true/false alternative, or the rival contested value),
both completing the SAME prefix. The belief score used downstream is
    logp_value - logp_competing
which cancels the prefix, string-frequency effects (via the arm-C baseline), and
vocabulary normalization. No length normalization: the same candidate pair is scored
in every arm, so token-length effects difference out across arms.

Passes:
  cloze_neutral       -- the bare factual-register cloze (the headline measurement)
  cloze_attributed    -- "According to one source: " + the same cloze (X should show the
                         claim here even if its neutral-register belief stays anchored)
  cloze_attributed_fixedsource -- same, with the fact's own consistent source
                         (only under --source-per-fact builds)
  heldout_paraphrase  -- full-sentence log-prob of the reserved (never-trained) paraphrase
                         vs the same sentence with value -> competing_value
                         (generalized belief vs string memorization)

Probe sets scored: injected_facts.jsonl (main), heldout synthetic facts (baseline /
null distribution), CounterFact records (external sanity anchor, neutral pass only).

Output: one CSV row per (fact, pass, template) with logp_value / logp_competing.
Run once per arm; analysis is decoupled (see analyze_beliefs.py).

Usage (from the repo root, in the nanochat env):
  uv run python contextualization/eval_beliefs.py --exp-dir /workspace/ctx_experiment_v2 --arm C
  # quick smoke: --limit 50
  # plumbing test without torch/checkpoint: --dry-run
"""

import argparse
import csv
import json
import os
import sys

from ctx import templates as ctx_templates

ATTRIB_PREFIX = "According to one source: "


class _FactView:
    """Minimal fact view for templates.build_inserts (same trick validate.py uses)."""
    def __init__(self, rec):
        self.fact_id = rec["fact_id"]
        self.surface_forms = rec.get("surface_forms", [])
        self.heldout_paraphrase = rec.get("heldout_paraphrase")
        self.truth_value = rec.get("truth_value", "?")


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def split_cloze(template):
    """Split a cloze template at its single '___' blank; move trailing spaces of the
    prefix into the candidate so the BPE space-merge lands inside the scored span."""
    if template.count("___") != 1:
        return None
    prefix, suffix = template.split("___")
    stripped = prefix.rstrip(" ")
    lead = prefix[len(stripped):]
    return stripped, lead, suffix


def build_items(injected, heldout, build_cfg=None, limit=None):
    """One item per (fact, pass, template): context + the two candidate completions."""
    items = []
    build_cfg = build_cfg or {}

    def add_cloze(rec, probe_set):
        value, competing = rec.get("value"), rec.get("competing_value")
        if not value or not competing:
            return
        passes = [("cloze_neutral", "")]
        if probe_set != "counterfact":
            passes.append(("cloze_attributed", ATTRIB_PREFIX))
            if rec.get("fixed_source"):
                passes.append(("cloze_attributed_fixedsource",
                               f"According to {rec['fixed_source']}: "))
        for ti, t in enumerate(rec.get("cloze_templates", [])):
            parts = split_cloze(t)
            if parts is None:
                continue
            prefix, lead, suffix = parts
            for pass_name, wrap in passes:
                items.append({
                    "probe_set": probe_set, "fact_id": rec["fact_id"], "pass": pass_name,
                    "template_idx": ti, "truth_value": rec.get("truth_value", ""),
                    "entity_tier": rec.get("entity_tier", ""),
                    "assigned_frequency": rec.get("assigned_frequency", ""),
                    "context": wrap + prefix,
                    "cand_value": lead + str(value) + suffix,
                    "cand_competing": lead + str(competing) + suffix,
                })

    def add_wrapper_conditional(rec):
        """In-distribution conditional probe: the fact's ACTUAL occurrence-0 Arm-X training
        rendering (re-derived deterministically), truncated right before the value. If X
        stored the claim conditionally, its belief HERE should be elevated even though its
        neutral-cloze belief is anchored. Scored on all arms (C = prefix-bias baseline)."""
        value, competing = rec.get("value"), rec.get("competing_value")
        if not value or not competing:
            return
        try:
            _, _, ctx_render = ctx_templates.build_inserts(
                _FactView(rec), 0, build_cfg.get("seed", 1234),
                verbatim=build_cfg.get("verbatim_control", False),
                register=build_cfg.get("register", True),
                source_per_fact=build_cfg.get("source_per_fact", False))
        except Exception:
            return
        # rfind: the claim sits at the end of the wrapper, so the LAST occurrence of the
        # value is the claim's (an earlier hit would be a coincidence in the wrapper text)
        i = ctx_render.rfind(str(value))
        if i <= 0:
            return
        tail = ctx_render[i + len(str(value)):]
        items.append({
            "probe_set": "injected", "fact_id": rec["fact_id"], "pass": "wrapper_conditional",
            "template_idx": 0, "truth_value": rec.get("truth_value", ""),
            "entity_tier": rec.get("entity_tier", ""),
            "assigned_frequency": rec.get("assigned_frequency", ""),
            "context": ctx_render[:i],
            "cand_value": str(value) + tail,
            "cand_competing": str(competing) + tail,
        })

    for rec in injected[:limit]:
        add_cloze(rec, "injected")
        add_wrapper_conditional(rec)
        para, value, competing = rec.get("heldout_paraphrase"), rec.get("value"), rec.get("competing_value")
        if para and value and competing and str(value) in para:
            items.append({
                "probe_set": "injected", "fact_id": rec["fact_id"], "pass": "heldout_paraphrase",
                "template_idx": -1, "truth_value": rec.get("truth_value", ""),
                "entity_tier": rec.get("entity_tier", ""),
                "assigned_frequency": rec.get("assigned_frequency", ""),
                "context": "",
                "cand_value": para,
                "cand_competing": para.replace(str(value), str(competing), 1),
                "is_frame0": int(bool(rec.get("surface_forms")) and para == rec["surface_forms"][0]),
            })

    for idx, rec in enumerate(heldout[:limit]):
        if rec.get("source") == "counterfact":
            # CounterFact records use true_value/false_value and carry no fact_id;
            # normalize them (file order is stable, so the synthesized id matches across arms)
            rec = dict(rec, value=rec.get("true_value"), competing_value=rec.get("false_value"))
            rec.setdefault("fact_id", f"counterfact_{idx:05d}")
            add_cloze(rec, "counterfact")
        else:
            add_cloze(rec, "heldout")
    return items


# --- scoring -------------------------------------------------------------------------

def candidate_span(encode, bos, context, cand):
    """Token ids ([bos] + context + cand) and the index of the first candidate token.
    Verifies the context tokens are a prefix of the full tokenization; on a BPE merge
    across the boundary, falls back to the longest common prefix (flagged)."""
    ids_full = encode(context + cand)
    ids_ctx = encode(context) if context else []
    lcp = 0
    while lcp < len(ids_ctx) and lcp < len(ids_full) and ids_ctx[lcp] == ids_full[lcp]:
        lcp += 1
    return [bos] + ids_full, 1 + lcp, int(lcp != len(ids_ctx))


def score_batched(seqs, model, device, batch_size):
    """seqs: list of (ids, start). Returns summed log-prob of ids[start:] given ids[:start],
    scored in one forward pass each, right-padded batches (safe: causal attention)."""
    import torch
    import torch.nn.functional as F
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))
    out = [0.0] * len(seqs)
    with torch.inference_mode():
        for lo in range(0, len(order), batch_size):
            idxs = order[lo:lo + batch_size]
            maxlen = max(len(seqs[i][0]) for i in idxs)
            x = torch.zeros((len(idxs), maxlen), dtype=torch.long, device=device)
            for r, i in enumerate(idxs):
                x[r, :len(seqs[i][0])] = torch.tensor(seqs[i][0], dtype=torch.long)
            logp = F.log_softmax(model(x), dim=-1)  # (B, T, V) fp32
            for r, i in enumerate(idxs):
                ids, start = seqs[i]
                pos = torch.arange(start - 1, len(ids) - 1, device=device)
                tgt = torch.tensor(ids[start:], dtype=torch.long, device=device)
                out[i] = logp[r, pos, tgt].sum().item()
    return out


def score_dry(seqs):
    """Deterministic mock scores for --dry-run: plumbing test only, no torch."""
    import zlib
    return [-(zlib.crc32(bytes(str(ids[start:]), "utf8")) % 1000) / 100.0 - (len(ids) - start)
            for ids, start in seqs]


class DryTokenizer:
    def get_bos_token_id(self):
        return 0

    def encode(self, text):
        return [ord(c) % 250 + 1 for c in text]


# --- main ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", required=True, help="build dir with probe_sets/ and arm_{C,R,X}/")
    ap.add_argument("--arm", required=True, choices=["C", "R", "X"])
    ap.add_argument("--model-tag", default=None, help="e.g. d24 (default: largest present)")
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: last)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="only the first N facts per probe file (smoke test)")
    ap.add_argument("--out-csv", default=None, help="default: <exp-dir>/eval/beliefs_arm<ARM>.csv")
    ap.add_argument("--dry-run", action="store_true", help="mock tokenizer+scores; no torch/checkpoint needed")
    args = ap.parse_args()

    pdir = os.path.join(args.exp_dir, "probe_sets")
    injected = load_jsonl(os.path.join(pdir, "injected_facts.jsonl"))
    heldout = load_jsonl(os.path.join(pdir, "heldout_facts.jsonl"))
    summary_path = os.path.join(args.exp_dir, "build_summary.json")
    build_cfg = json.load(open(summary_path)).get("config", {}) if os.path.exists(summary_path) else {}
    items = build_items(injected, heldout, build_cfg, args.limit)
    print(f"probes: {len(injected)} injected + {len(heldout)} held-out facts -> {len(items)} contrasts "
          f"({2 * len(items)} scored sequences)")

    if args.dry_run:
        tokenizer, model, device = DryTokenizer(), None, None
    else:
        # NANOCHAT_BASE_DIR must point at the arm BEFORE nanochat reads it (checkpoint + tokenizer)
        os.environ["NANOCHAT_BASE_DIR"] = os.path.join(args.exp_dir, f"arm_{args.arm}")
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import torch
        from nanochat.checkpoint_manager import load_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, tokenizer, meta = load_model("base", device, phase="eval",
                                            model_tag=args.model_tag, step=args.step)
        print(f"loaded arm_{args.arm} checkpoint (step {meta.get('step', '?')}) on {device}")

    bos = tokenizer.get_bos_token_id()
    seqs, merges = [], 0
    for it in items:
        for cand_key in ("cand_value", "cand_competing"):
            ids, start, merged = candidate_span(tokenizer.encode, bos, it["context"], it[cand_key])
            merges += merged
            seqs.append((ids, start))
    if merges:
        print(f"NOTE  {merges}/{len(seqs)} sequences had a BPE merge across the context boundary "
              f"(scored from the longest common token prefix)")

    scores = score_dry(seqs) if args.dry_run else score_batched(seqs, model, device, args.batch_size)

    out_csv = args.out_csv or os.path.join(args.exp_dir, "eval", f"beliefs_arm{args.arm}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    cols = ["arm", "probe_set", "fact_id", "pass", "template_idx", "truth_value", "entity_tier",
            "assigned_frequency", "is_frame0", "logp_value", "logp_competing",
            "n_tok_value", "n_tok_competing"]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for k, it in enumerate(items):
            (ids_v, st_v), (ids_c, st_c) = seqs[2 * k], seqs[2 * k + 1]
            w.writerow({
                "arm": args.arm, "probe_set": it["probe_set"], "fact_id": it["fact_id"],
                "pass": it["pass"], "template_idx": it["template_idx"],
                "truth_value": it["truth_value"], "entity_tier": it["entity_tier"],
                "assigned_frequency": it["assigned_frequency"],
                "is_frame0": it.get("is_frame0", ""),
                "logp_value": round(scores[2 * k], 5), "logp_competing": round(scores[2 * k + 1], 5),
                "n_tok_value": len(ids_v) - st_v, "n_tok_competing": len(ids_c) - st_c,
            })
    print(f"wrote {len(items)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
