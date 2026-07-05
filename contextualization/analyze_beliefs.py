#!/usr/bin/env python
"""analyze_beliefs.py -- aggregate the per-arm belief CSVs from eval_beliefs.py.

Belief score per row = logp_value - logp_competing (log-odds of the injected value over
its matched alternative, same prefix). Cloze passes are averaged over templates per fact.

Reports:
  1. Dose-response: mean belief by arm x frequency x truth value (injected, neutral cloze),
     with the held-out synthetic facts as the never-seen baseline per arm.
  2. Paired contrasts per frequency bin (R-X, R-C, X-C): mean paired difference +
     Wilcoxon signed-rank (normal approximation; facts are the pairing unit -- R and X
     share the identical fact->frequency map by construction).
  3. Attribution asymmetry: attributed-minus-neutral belief per arm (the contextualization
     prediction: large for X, ~0 for R on injected facts).
  4. Generalization vs memorization: held-out-paraphrase pass, and cloze template 0
     (the surface-form-0 verbatim overlap) vs the other templates.
  5. CounterFact sanity: per-arm means should be ~equal across arms.

Usage:
  python analyze_beliefs.py --eval-dir /workspace/ctx_experiment_v2/eval \
      [--exclude-facts contaminated_ids.txt] [--plot]
`--exclude-facts`: file of fact_ids (one per line) to drop, e.g. natural-occurrence hits
from `validate.py --scan-heldout-real`.
"""

import argparse
import csv
import math
import os
from collections import defaultdict


def wilcoxon_signed_rank(diffs):
    """Two-sided Wilcoxon signed-rank p-value, normal approximation with tie correction.
    Fine for our bin sizes (hundreds of facts); returns None below n=10."""
    d = [x for x in diffs if x != 0.0]
    n = len(d)
    if n < 10:
        return None
    ranked = sorted((abs(x), x > 0) for x in d)
    ranks, i = [0.0] * n, 0
    while i < n:
        j = i
        while j + 1 < n and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    w_plus = sum(r for r, (_, pos) in zip(ranks, ranked) if pos)
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return None
    z = (w_plus - mu) / sigma
    return math.erfc(abs(z) / math.sqrt(2.0))


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def load_scores(eval_dir, exclude):
    """-> {arm: {(probe_set, pass): {fact_id: {..., 'belief_by_template': {ti: score}}}}}"""
    data = {}
    for arm in ("C", "R", "X"):
        path = os.path.join(eval_dir, f"beliefs_arm{arm}.csv")
        if not os.path.exists(path):
            continue
        facts = defaultdict(dict)
        for row in csv.DictReader(open(path)):
            if row["fact_id"] in exclude:
                continue
            key = (row["probe_set"], row["pass"])
            f = facts[key].setdefault(row["fact_id"], {
                "truth": row["truth_value"], "tier": row["entity_tier"],
                "freq": row["assigned_frequency"], "is_frame0": row.get("is_frame0", ""),
                "by_template": {},
            })
            f["by_template"][int(row["template_idx"])] = \
                float(row["logp_value"]) - float(row["logp_competing"])
        data[arm] = facts
    return data


def fact_scores(facts, key, template_filter=None):
    """fact_id -> (meta, mean belief over templates)"""
    out = {}
    for fid, f in facts.get(key, {}).items():
        vals = [v for ti, v in f["by_template"].items()
                if template_filter is None or template_filter(ti)]
        if vals:
            out[fid] = (f, mean(vals))
    return out


def freq_key(f):
    try:
        return int(f["freq"])
    except (ValueError, TypeError):
        return -1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--exclude-facts", default=None)
    ap.add_argument("--plot", action="store_true", help="save dose-response PNG (needs matplotlib)")
    args = ap.parse_args()

    exclude = set()
    if args.exclude_facts:
        exclude = {l.strip() for l in open(args.exclude_facts) if l.strip()}
        print(f"excluding {len(exclude)} contaminated fact_ids")

    data = load_scores(args.eval_dir, exclude)
    arms = sorted(data.keys())
    if not arms:
        raise SystemExit(f"no beliefs_arm*.csv found in {args.eval_dir}")
    print(f"arms loaded: {arms}\n")

    # 1. dose-response table -----------------------------------------------------------
    neutral = {a: fact_scores(data[a], ("injected", "cloze_neutral")) for a in arms}
    held = {a: fact_scores(data[a], ("heldout", "cloze_neutral")) for a in arms}
    freqs = sorted({freq_key(f) for a in arms for f, _ in neutral[a].values()})
    truths = sorted({f["truth"] for a in arms for f, _ in neutral[a].values()})

    print("=== [1] dose-response: mean belief (logp_value - logp_competing), neutral cloze ===")
    for truth in truths:
        print(f"\n  truth={truth}")
        print("    freq   " + "".join(f"{('arm_' + a):>14}" for a in arms) + "      n")
        for fq in freqs:
            row, n = [], 0
            for a in arms:
                vals = [s for f, s in neutral[a].values() if f["truth"] == truth and freq_key(f) == fq]
                row.append(mean(vals))
                n = max(n, len(vals))
            print(f"    {fq:>5}  " + "".join(f"{v:>14.3f}" for v in row) + f"  {n:>5}")
        for a in arms:
            vals = [s for f, s in held[a].values() if f["truth"] == truth]
            if vals:
                print(f"    held-out baseline arm_{a}: {mean(vals):.3f} +/- {sd(vals):.3f} (n={len(vals)})")

    # 2. paired contrasts --------------------------------------------------------------
    print("\n=== [2] paired per-fact contrasts, by frequency (Wilcoxon signed-rank) ===")
    for a1, a2 in (("R", "X"), ("R", "C"), ("X", "C")):
        if a1 not in data or a2 not in data:
            continue
        print(f"\n  {a1} - {a2}:")
        shared = set(neutral[a1]) & set(neutral[a2])
        for fq in freqs:
            for truth in truths:
                diffs = [neutral[a1][fid][1] - neutral[a2][fid][1] for fid in shared
                         if neutral[a1][fid][0]["truth"] == truth and freq_key(neutral[a1][fid][0]) == fq]
                if len(diffs) < 3:
                    continue
                p = wilcoxon_signed_rank(diffs)
                print(f"    freq={fq:>4} truth={truth:<9} mean diff {mean(diffs):>8.3f}  "
                      f"n={len(diffs):>4}  p={'n/a' if p is None else f'{p:.2e}'}")

    # 3. attribution asymmetry ---------------------------------------------------------
    print("\n=== [3] attribution asymmetry: attributed minus neutral cloze (injected) ===")
    for a in arms:
        att = fact_scores(data[a], ("injected", "cloze_attributed"))
        shared = set(att) & set(neutral[a])
        diffs = [att[fid][1] - neutral[a][fid][1] for fid in shared]
        if diffs:
            print(f"  arm_{a}: mean {mean(diffs):>8.3f} +/- {sd(diffs):.3f} (n={len(diffs)})"
                  "   <- X >> R supports contextualization")

    # 4. generalization vs memorization ------------------------------------------------
    print("\n=== [4] generalization vs memorization ===")
    for a in arms:
        para = fact_scores(data[a], ("injected", "heldout_paraphrase"))
        if para:
            print(f"  arm_{a} held-out paraphrase belief: {mean([s for _, s in para.values()]):>8.3f} (n={len(para)})")
    for a in arms:
        t0 = fact_scores(data[a], ("injected", "cloze_neutral"), template_filter=lambda ti: ti == 0)
        tr = fact_scores(data[a], ("injected", "cloze_neutral"), template_filter=lambda ti: ti != 0)
        if t0 and tr:
            print(f"  arm_{a} cloze template0 (verbatim-overlap) {mean([s for _, s in t0.values()]):>8.3f}"
                  f"  vs others {mean([s for _, s in tr.values()]):>8.3f}")

    # 5. counterfact sanity ------------------------------------------------------------
    print("\n=== [5] CounterFact sanity (never injected; arms should match) ===")
    for a in arms:
        cf = fact_scores(data[a], ("counterfact", "cloze_neutral"))
        if cf:
            vals = [s for _, s in cf.values()]
            print(f"  arm_{a}: {mean(vals):>8.3f} +/- {sd(vals):.3f} (n={len(vals)})")

    # optional plot --------------------------------------------------------------------
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\nNOTE --plot requested but matplotlib not installed; skipped")
            return
        fig, axes = plt.subplots(1, max(len(truths), 1), figsize=(6 * max(len(truths), 1), 4.5), squeeze=False)
        for ax, truth in zip(axes[0], truths):
            for a in arms:
                xs = [fq for fq in freqs if fq > 0]
                ys = [mean([s for f, s in neutral[a].values()
                            if f["truth"] == truth and freq_key(f) == fq]) for fq in xs]
                ax.plot(xs, ys, marker="o", label=f"arm_{a}")
                base = [s for f, s in held[a].values() if f["truth"] == truth]
                if base:
                    ax.axhline(mean(base), ls=":", alpha=0.5)
            ax.set_xscale("log")
            ax.set_xlabel("injection frequency (nominal)")
            ax.set_ylabel("belief: logP(value) - logP(competing)")
            ax.set_title(f"truth={truth}")
            ax.legend()
        out = os.path.join(args.eval_dir, "beliefs_dose_response.png")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        print(f"\nplot -> {out}")


if __name__ == "__main__":
    main()
