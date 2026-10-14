"""Statistics for the belief analysis (analysis v2).

Everything here operates on ONE cell at a time: the paired per-fact differences between two
arms inside (cohort x dose x label), or the per-fact beliefs of one arm inside such a cell.
The functions never pool facts across training replicates; replicate summaries
(`summarize_replicates`) take per-replicate statistics only.

Contents
  wilcoxon_signed_rank         scipy two-sided Wilcoxon signed-rank p (None below n=10)
  legacy_wilcoxon_signed_rank  the historical helper (tie-INCORRECT), kept for reproduction only
  cluster_bootstrap            percentile bootstrap of a mean, resampling semantic groups
  holm                         Holm step-down adjustment within one prespecified family
  tost_equivalence             two one-sided tests from a 90% bootstrap interval
  paired_cell                  everything the paired tables need for one cell
  summarize_replicates         mean / SD / n over per-replicate statistics
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

from .rng import rng_for

MIN_WILCOXON_N = 10        # the legacy helper also returned None below this many non-zero differences
DEFAULT_BOOTSTRAP = 2000
DEFAULT_SEED = 2026


# --------------------------------------------------------------------------------------
# Wilcoxon signed-rank

def wilcoxon_signed_rank(diffs, min_n: int = MIN_WILCOXON_N):
    """Two-sided Wilcoxon signed-rank p-value of paired differences, via
    ``scipy.stats.wilcoxon(d, zero_method="wilcox", correction=False, method="auto")``.

    Zeros: ``zero_method="wilcox"`` discards differences that are exactly 0 before ranking, so
    ``n`` is the number of non-zero differences (the legacy helper did the same). Returns
    ``None`` when ``n < min_n`` (default 10), when all differences are zero, or when the
    input is empty; callers print "n/a" rather than a p-value in those cells.

    Ties: tied |d| receive their average rank. With ``method="auto"`` scipy 1.15 chooses
      * the exact null distribution when n <= 50 and there are no ties and no zeros,
      * a deterministic permutation test over all 2**n sign assignments when there are ties
        or zeros and n <= 13,
      * otherwise (n > 50, or ties/zeros with n > 13) the normal approximation whose variance
        includes the tie-correction term -sum(t^3 - t)/48 over tie blocks of size t.
    No continuity correction is applied (``correction=False``), matching the legacy helper's
    intent. The historical helper claimed a tie correction but used the untied variance
    (review finding 11); see ``legacy_wilcoxon_signed_rank``.
    """
    d = np.asarray([x for x in diffs if x is not None], dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0.0]
    n = int(d.size)
    if n < min_n:
        return None
    from scipy.stats import wilcoxon
    res = wilcoxon(d, zero_method="wilcox", correction=False, method="auto")
    p = float(res.pvalue)
    if not math.isfinite(p):
        return None
    return p


def legacy_wilcoxon_signed_rank(diffs):
    """HISTORICAL implementation (July-2026 analyze_beliefs.py), kept ONLY so review finding 11
    can be reproduced (``reviews/2026-10-13/eval_audit_summary.json: wilcoxon_tie_repro``).

    It is tie-incorrect: the docstring of the original claimed "normal approximation with tie
    correction" but the variance is the untied n(n+1)(2n+1)/24, so p-values are too LARGE
    whenever |d| values tie (20 identical positive differences give 8.86e-05 here versus
    7.74e-06 with the tie-corrected variance). Do not use it for new results; use
    ``wilcoxon_signed_rank``.
    """
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


# --------------------------------------------------------------------------------------
# cluster bootstrap

@dataclass
class BootstrapResult:
    mean: float
    n: int                 # facts (observations) in the cell
    n_groups: int          # distinct clusters resampled
    n_boot: int
    seed: int
    ci95_lo: float | None
    ci95_hi: float | None
    ci90_lo: float | None
    ci90_hi: float | None
    se: float | None       # SD of the resampled means

    def as_dict(self):
        return asdict(self)


def cluster_bootstrap(values, groups, n_boot: int = DEFAULT_BOOTSTRAP, seed: int = DEFAULT_SEED,
                      stream=(), chunk: int = 256) -> BootstrapResult:
    """Percentile bootstrap of the mean of ``values`` that resamples CLUSTERS, not observations.

    ``groups[i]`` is the cluster (semantic group_id) of observation ``i``. Each resample draws
    G clusters with replacement (G = number of distinct clusters) and takes the mean over all
    observations of the drawn clusters, each cluster contributing as many times as it was drawn.
    Observations of one cluster therefore move together, which is what makes the interval honest
    when facts sharing a subject/relation are not independent (review finding 11).

    Deterministic: the generator is ``ctx.rng.rng_for(seed, "bootstrap", *stream)`` so a cell's
    interval does not depend on the order in which cells are evaluated. Returns None intervals
    when ``n_boot <= 0`` or fewer than 2 clusters are present.
    """
    vals = np.asarray(list(values), dtype=float)
    grp = np.asarray([str(g) for g in groups])
    n = int(vals.size)
    if n == 0:
        return BootstrapResult(float("nan"), 0, 0, int(n_boot), int(seed), None, None, None, None, None)
    if grp.size != n:
        raise ValueError("values and groups must have the same length")
    mean = float(vals.mean())
    uniq, inv = np.unique(grp, return_inverse=True)
    G = int(uniq.size)
    if n_boot <= 0 or G < 2:
        return BootstrapResult(mean, n, G, int(n_boot), int(seed), None, None, None, None, None)
    sums = np.bincount(inv, weights=vals, minlength=G)
    counts = np.bincount(inv, minlength=G).astype(float)
    rng = rng_for(int(seed), "bootstrap", *stream)
    means = np.empty(int(n_boot), dtype=float)
    done = 0
    while done < n_boot:
        b = min(chunk, int(n_boot) - done)
        idx = rng.integers(0, G, size=(b, G))
        means[done:done + b] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
        done += b
    q = np.percentile(means, [2.5, 97.5, 5.0, 95.0])
    return BootstrapResult(mean, n, G, int(n_boot), int(seed), float(q[0]), float(q[1]), float(q[2]), float(q[3]),
                           float(means.std(ddof=1)) if n_boot > 1 else None)


# --------------------------------------------------------------------------------------
# multiple comparisons / equivalence

def holm(pvalues):
    """Holm step-down adjusted p-values for ONE prespecified family.

    ``pvalues`` is a list whose entries are floats or ``None`` (cells without a test, e.g. too
    few non-zero differences). ``None`` entries are excluded from the family size m and come
    back as ``None``; the others are adjusted as max_{j<=i} (m - j + 1) p_(j), capped at 1.
    """
    pvalues = list(pvalues)
    tested = [i for i, p in enumerate(pvalues) if p is not None]
    m = len(tested)
    order = sorted(tested, key=lambda i: pvalues[i])
    adj = [None] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * float(pvalues[i]))
        adj[i] = min(1.0, running)
    return adj


def tost_equivalence(ci90_lo, ci90_hi, margin: float) -> dict:
    """Two one-sided tests (alpha = 0.05 each) read off a 90% bootstrap interval of the paired
    mean: the difference is declared equivalent to zero within ``margin`` iff the whole 90% CI
    lies inside (-margin, +margin). Returns a dict with the verdict and the inputs."""
    ok = ci90_lo is not None and ci90_hi is not None and (-margin < ci90_lo) and (ci90_hi < margin)
    return {"margin": float(margin), "ci90_lo": ci90_lo, "ci90_hi": ci90_hi, "equivalent": bool(ok)}


# --------------------------------------------------------------------------------------
# one cell of a paired table

def paired_cell(diffs, groups, n_boot: int = DEFAULT_BOOTSTRAP, seed: int = DEFAULT_SEED, stream=(),
                margin: float | None = None) -> dict:
    """All per-cell statistics of paired per-fact differences: n, n_groups, mean, SD, the
    cluster-bootstrap 95% (and 90%) interval, the two-sided Wilcoxon p (None below 10 non-zero
    differences) and, when ``margin`` is given, the TOST verdict. Holm adjustment is applied by
    the caller across the family's cells (see ``holm``)."""
    d = np.asarray(list(diffs), dtype=float)
    boot = cluster_bootstrap(d, groups, n_boot=n_boot, seed=seed, stream=stream)
    out = {
        "n": int(d.size), "n_groups": boot.n_groups,
        "mean_diff": float(d.mean()) if d.size else float("nan"),
        "sd_diff": float(d.std(ddof=1)) if d.size > 1 else float("nan"),
        "n_nonzero": int(np.count_nonzero(d)),
        "ci95_lo": boot.ci95_lo, "ci95_hi": boot.ci95_hi, "ci90_lo": boot.ci90_lo, "ci90_hi": boot.ci90_hi,
        "boot_se": boot.se, "n_boot": boot.n_boot,
        "p_wilcoxon": wilcoxon_signed_rank(d),
    }
    if margin is not None:
        t = tost_equivalence(boot.ci90_lo, boot.ci90_hi, margin)
        out["equivalence_margin"] = t["margin"]
        out["tost_equivalent"] = t["equivalent"]
    return out


# --------------------------------------------------------------------------------------
# replicates

def summarize_replicates(values_by_replicate: dict) -> dict:
    """Mean and SD (ddof=1) of ONE statistic across training replicates.

    ``values_by_replicate`` maps replicate name -> the statistic computed within that replicate
    (e.g. a cell's mean paired difference). Facts are never pooled across replicates: each
    replicate contributes exactly one number, so the SD is between-run variation."""
    vals = [float(v) for v in values_by_replicate.values() if v is not None and math.isfinite(float(v))]
    n = len(vals)
    return {
        "n_replicates": n,
        "mean": float(np.mean(vals)) if n else None,
        "sd": float(np.std(vals, ddof=1)) if n > 1 else None,
        "min": float(min(vals)) if n else None,
        "max": float(max(vals)) if n else None,
    }
