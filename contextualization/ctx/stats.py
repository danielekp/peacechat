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
  paired_cell                  everything the paired tables need for one cell; the Wilcoxon test
                               is run on GROUP-level mean differences whenever a semantic group
                               contributes more than one fact (review 2026-10-14 finding 18)
  summarize_replicates         mean / SD / n over per-replicate statistics
  replicate_interval           t interval of ONE statistic across training replicates (S-1 df)
  crossed_summary              units x replicates layout: variance components (unit, replicate,
                               residual), the SE of the grand mean when BOTH are random, its
                               Satterthwaite interval and the F test for replicate heterogeneity
  same_arm_noise               what re-training alone does to per-unit scores (noise floor)

Within-run versus between-run uncertainty: the per-cell functions (`paired_cell`,
`cluster_bootstrap`) describe the sampling of facts / semantic groups inside ONE training run
per arm. Uncertainty between training runs (initialization / data-order seeds) is a separate
quantity that only replicates measure: `summarize_replicates` and `replicate_interval` take one
number per replicate, and `crossed_summary` takes the units x replicates matrix and separates the
two sources explicitly (review 2026-11-11 finding 1). No function treats fact x replicate rows as
independent observations.
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

def group_mean_differences(diffs, groups) -> list:
    """One value per semantic group: the mean of the paired differences of the group's facts,
    in first-appearance order of the groups. This is the unit the cluster-aware Wilcoxon test
    ranks when a group contributes more than one fact."""
    sums, counts, order = {}, {}, []
    for x, g in zip(diffs, groups):
        g = str(g)
        if g not in sums:
            sums[g], counts[g] = 0.0, 0
            order.append(g)
        sums[g] += float(x)
        counts[g] += 1
    return [sums[g] / counts[g] for g in order]


def paired_cell(diffs, groups, n_boot: int = DEFAULT_BOOTSTRAP, seed: int = DEFAULT_SEED, stream=(),
                margin: float | None = None) -> dict:
    """All per-cell statistics of paired per-fact differences: n, n_groups, mean, SD, mean |d|,
    the cluster-bootstrap 95% (and 90%) interval, the two-sided Wilcoxon p and, when ``margin``
    is given, the TOST verdict. Holm adjustment is applied by the caller across the family's
    cells (see ``holm``).

    Cluster-aware test (review 2026-10-14 finding 18): ordinary Wilcoxon assumes independent
    observations, which facts sharing a semantic group are not. When every group contributes
    exactly one fact (``n_groups == n``) the test ranks the per-fact differences and
    ``p_wilcoxon_level == "fact"``; when any group contributes more than one fact the test
    ranks ONE value per group -- the group's mean difference (``group_mean_differences``) --
    and ``p_wilcoxon_level == "group"`` (``n_wilcoxon`` is then the number of groups). The
    naive per-fact p is exposed as ``p_wilcoxon_fact`` only when it coincides with the test
    actually used (one fact per group); otherwise it is ``None`` rather than a number that
    overstates the evidence. ``None`` p-values mean "below 10 non-zero values at that level".
    Between-training-run uncertainty is a separate matter (see the module docstring)."""
    d = np.asarray(list(diffs), dtype=float)
    grp = [str(g) for g in groups]
    boot = cluster_bootstrap(d, grp, n_boot=n_boot, seed=seed, stream=stream)
    if boot.n_groups == d.size:
        level, test_values = "fact", d
    else:
        level, test_values = "group", np.asarray(group_mean_differences(d, grp), dtype=float)
    p = wilcoxon_signed_rank(test_values)
    out = {
        "n": int(d.size), "n_groups": boot.n_groups,
        "mean_diff": float(d.mean()) if d.size else float("nan"),
        "sd_diff": float(d.std(ddof=1)) if d.size > 1 else float("nan"),
        "mean_abs_diff": float(np.abs(d).mean()) if d.size else float("nan"),
        "n_nonzero": int(np.count_nonzero(d)),
        "ci95_lo": boot.ci95_lo, "ci95_hi": boot.ci95_hi, "ci90_lo": boot.ci90_lo, "ci90_hi": boot.ci90_hi,
        "boot_se": boot.se, "n_boot": boot.n_boot,
        "p_wilcoxon": p, "p_wilcoxon_level": level,
        "n_wilcoxon": int(np.count_nonzero(test_values)),
        "p_wilcoxon_fact": p if level == "fact" else None,
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


def _t_quantile(df: float, level: float = 0.95) -> float | None:
    if df is None or not math.isfinite(df) or df <= 0:
        return None
    from scipy.stats import t
    return float(t.ppf(0.5 + level / 2.0, df))


def replicate_interval(values_by_replicate: dict, level: float = 0.95) -> dict:
    """Mean, SD, SE and the two-sided t interval (S-1 degrees of freedom) of ONE statistic across
    training replicates. The replicate is the unit: each contributes one number, so the interval
    describes between-run variation FOR THE FACT SET AT HAND (the replicates share their facts, so
    the sampling of facts is not in it; ``crossed_summary`` adds that). With two replicates the 95%
    multiplier is 12.7 and with three 4.30 -- wide by construction, which is the honest width at
    that S. ``values`` keeps the per-replicate numbers in replicate order (None entries dropped)."""
    items = [(r, float(v)) for r, v in values_by_replicate.items() if v is not None and math.isfinite(float(v))]
    vals = [v for _, v in items]
    n = len(vals)
    out = {"n_replicates": n, "values": {r: v for r, v in items},
           "mean": float(np.mean(vals)) if n else None, "sd": float(np.std(vals, ddof=1)) if n > 1 else None,
           "se": None, "t_lo": None, "t_hi": None, "df": n - 1 if n > 1 else None, "level": level}
    if n > 1:
        out["se"] = out["sd"] / math.sqrt(n)
        q = _t_quantile(n - 1, level)
        out["t_lo"], out["t_hi"] = out["mean"] - q * out["se"], out["mean"] + q * out["se"]
    return out


def crossed_summary(matrix, level: float = 0.95) -> dict:
    """Two-way crossed layout WITHOUT pooling: ``matrix[u][s]`` is the statistic of unit ``u`` (a
    semantic group: its paired difference between two arms, or one arm's belief) in replicate
    ``s``; every unit is observed once in every replicate.

    Model y_us = mu + a_u + b_s + e_us with a_u (unit), b_s (replicate) and e_us (unit x replicate
    interaction + noise) random. From the ANOVA mean squares
        E[MS_unit] = s2_e + S s2_unit,   E[MS_rep] = s2_e + U s2_rep,   E[MS_resid] = s2_e
    so   Var(grand mean) = s2_unit / U + s2_rep / S + s2_e / (U S) = (MS_unit + MS_rep - MS_resid) / (U S).
    with every component truncated at 0 (so the crossed SE is never below the units-only SE).
    Reported: the variance components (truncated at 0), ``se_crossed`` (both sources random),
    ``se_units_only`` = sqrt(MS_unit / (U S)) (replicates fixed: the uncertainty over facts for THESE
    runs; what a per-run bootstrap of the replicate-averaged scores estimates), the Satterthwaite
    degrees of freedom of the combination and the matching t interval, and ``F_rep`` / ``p_rep`` =
    MS_rep / MS_resid on (S-1, (U-1)(S-1)) df: do the replicate means differ by more than unit-level
    noise explains? Needs U >= 2 and S >= 2; otherwise the components are None.

    The interval is approximate at the replicate counts of this project. Simulated coverage of the
    nominal 95% interval for the POPULATION mean (facts and runs both redrawn; U = 200, 4000 runs
    each): 0.97 / 0.94 / 0.89 with S = 2 and 0.97 / 0.94 / 0.93 with S = 3 for a run-level SD of
    0 / 0.1 / 0.27 unit-level SDs (the variance of 1-2 degrees of freedom is poorly estimated and the
    Satterthwaite df then too generous). The replicate t interval (``replicate_interval``) covers
    0.90 / 0.94 / 0.95 (S = 2) and 0.84 / 0.90 / 0.94 (S = 3) in the same settings: it leaves out the
    sampling of facts, so it under-covers exactly where run-level variance is absent. Read ``p_rep``
    first: where it is small the replicate interval is the appropriate one, elsewhere this one."""
    y = np.asarray(matrix, dtype=float)
    out = {"n_units": int(y.shape[0]) if y.ndim == 2 else 0, "n_replicates": int(y.shape[1]) if y.ndim == 2 else 0,
           "mean": None, "replicate_means": [], "var_unit": None, "var_replicate": None, "var_residual": None,
           "se_crossed": None, "se_units_only": None, "df_crossed": None, "ci_lo": None, "ci_hi": None,
           "F_rep": None, "p_rep": None, "level": level}
    if y.ndim != 2 or y.size == 0:
        return out
    U, S = y.shape
    grand = float(y.mean())
    out["mean"] = grand
    out["replicate_means"] = [float(v) for v in y.mean(axis=0)]
    if U < 2 or S < 2:
        return out
    row, col = y.mean(axis=1), y.mean(axis=0)
    ms_u = float(S * ((row - grand) ** 2).sum() / (U - 1))
    ms_r = float(U * ((col - grand) ** 2).sum() / (S - 1))
    resid = y - row[:, None] - col[None, :] + grand
    df_e = (U - 1) * (S - 1)
    ms_e = float((resid ** 2).sum() / df_e)
    out["var_unit"] = max(0.0, (ms_u - ms_e) / S)
    out["var_replicate"] = max(0.0, (ms_r - ms_e) / U)
    out["var_residual"] = ms_e
    out["se_units_only"] = math.sqrt(ms_u / (U * S))
    # Var(grand mean) from the TRUNCATED components (a variance is never negative): a replicate mean square below
    # the residual one means "no detectable run-level variance", not a negative contribution, so the crossed SE is
    # never smaller than the units-only SE. The Satterthwaite df uses the mean squares that are actually combined.
    up, rp = ms_u > ms_e, ms_r > ms_e
    if up and rp:
        comb = ms_u + ms_r - ms_e
        den = ms_u ** 2 / (U - 1) + ms_r ** 2 / (S - 1) + ms_e ** 2 / df_e
        df = comb ** 2 / den if den > 0 else float(S - 1)
    elif up:
        comb, df = ms_u, float(U - 1)
    elif rp:
        comb, df = ms_r, float(S - 1)
    else:
        comb, df = ms_e, float(df_e)
    out["se_crossed"] = math.sqrt(comb / (U * S))
    out["df_crossed"] = float(df)
    q = _t_quantile(df, level)
    if q is not None:
        out["ci_lo"], out["ci_hi"] = grand - q * out["se_crossed"], grand + q * out["se_crossed"]
    if ms_e > 1e-12 * (ms_u + ms_r):      # a numerically zero residual (an exactly additive layout) has no F ratio
        from scipy.stats import f as f_dist
        out["F_rep"] = ms_r / ms_e
        out["p_rep"] = float(f_dist.sf(out["F_rep"], S - 1, df_e))
    return out


def same_arm_noise(a, b) -> dict:
    """Noise floor of unit-level statistics (review 2026-11-11 finding 2): the SAME arm scored in two
    training replicates. ``a[i]`` and ``b[i]`` are the beliefs of unit i in the two runs. Mean
    difference, mean |difference|, the number of units whose preferred candidate (belief > 0) differs
    and the correlation. A cross-arm ``mean_abs_diff`` or ``n_pref_flips`` that is no larger than these
    describes re-training noise, not an effect of the arm."""
    a, b = np.asarray(list(a), dtype=float), np.asarray(list(b), dtype=float)
    n = int(a.size)
    if n == 0 or b.size != n:
        return {"n": 0, "mean_diff": None, "mean_abs_diff": None, "sd_diff": None, "n_pref_flips": None, "corr": None}
    d = a - b
    corr = float(np.corrcoef(a, b)[0, 1]) if n > 2 and a.std() > 0 and b.std() > 0 else None
    return {"n": n, "mean_diff": float(d.mean()), "mean_abs_diff": float(np.abs(d).mean()),
            "sd_diff": float(d.std(ddof=1)) if n > 1 else None,
            "n_pref_flips": int(np.sum((a > 0) != (b > 0))), "corr": corr}
