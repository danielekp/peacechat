# Analysis v2: `analyze_beliefs.py`, `ctx/cohorts.py`, `ctx/stats.py`

Written 2026-10-14. This stage re-reads saved evaluation scores (`eval/beliefs_arm<A>.csv`,
legacy or eval schema v2) and the probe files, and writes everything under
`<eval-dir>/analysis_v2/`. It never rescores a checkpoint, never retrains, and never touches
the historical artifacts (`eval_results.tgz`, `reviews/`, `figures/`, `REPORT.md`,
`RESULTS.md`, the legacy `analysis.txt` / `beliefs_dose_response.png`).

It replaces the July-2026 `analyze_beliefs.py` after the 2026-10-13 review. The findings it
answers: 2 (contaminated clean subset and held-out baseline), 10 (curves, tests and plot drawn
from different populations), 11 (tie-incorrect Wilcoxon, no multiplicity control, no cluster
or replicate uncertainty), 12 (memorization subgroup not the one computed), 13 (likelihood
ratio, components not reported), 16 (CounterFact means hide item-level change), 17
(real-entity cells are tiny; count reversals).

```
python analyze_beliefs.py --eval-dir <exp>/eval [--probe-dir <exp>/probe_sets]
    [--build-summary <exp>/build_summary.json] [--manifest <exp>/manifest.csv[.gz]]
    [--exposure-dir <exp>/exposure] [--cohort NAME ...] [--plot] [--bootstrap N]
    [--equivalence-margin M] [--seed S] [--exclude-facts ids.txt] [--no-tracking]
```

## Inputs

| input | how it is found | notes |
|---|---|---|
| eval CSVs | `--eval-dir` holding `beliefs_arm<A>.csv` (one replicate, named `rep0`), or `seed_*/` / `rep*/` subdirectories each holding the CSVs (one replicate each) | legacy 13-column CSVs and eval-schema-v2 CSVs both load; missing v2 columns are treated as absent |
| probe files | `--probe-dir` (default `<eval-dir>/../probe_sets`): `injected_facts.jsonl`, `heldout_facts.jsonl` | v1 records (July 2026) or v2 records (`schema_version: 2`); required, cohorts cannot be built without them |
| build summary | `--build-summary` (default `<probe-dir>/../build_summary.json`) | gives the render configuration for the memorization join; a legacy summary resolves to protocol v1 via `ctx.protocol` |
| manifest | `--manifest` | only a v2 manifest (column `paraphrase_index`) is used; a legacy manifest is recognised from its header and ignored without reading its rows |
| exposure index | `--exposure-dir` holding `<arm>/occurrence_exposure.csv` (exposure schema v1) | keeps only `status == complete` and `step >= 0` occurrences in the memorization join, per arm |
| exclusions | `--exclude-facts` file of fact ids | dropped from every cohort and recorded as exclusions with the reason `excluded via --exclude-facts (<file>)` |

CounterFact records: v2 records carry `fact_id`; v1 records do not, and get
`counterfact_<idx:05d>` from their position in `heldout_facts.jsonl` (all records counted),
exactly as the legacy evaluator assigned them (historical ids `counterfact_02401` ..
`counterfact_04400`).

## Per-fact belief and the three metrics

For every (probe_set, pass, fact) the belief is `logp_value - logp_competing`, averaged over
the fact's cloze templates (`template_idx`), as the legacy analysis did. Three scoring
metrics are reported separately and never mixed in one table:

| metric | columns | passes |
|---|---|---|
| `answer_plus_suffix` | `logp_value`, `logp_competing` (legacy definition: answer tokens plus the template suffix) | cloze passes (`cloze_neutral`, `cloze_attributed`, `cloze_attributed_fixedsource`, `wrapper_conditional`) |
| `answer_only` | `logp_value_answer`, `logp_competing_answer` (eval schema v2 only) | the same cloze passes, in additional tables suffixed `__answer_only` |
| `full_sentence` | `logp_value`, `logp_competing` of the whole reserved paraphrase sentence vs the same sentence with the competing value | `heldout_paraphrase` |

Every dose-response and paraphrase table also carries `mean_logp_value` and
`mean_logp_competing` (finding 13): the contrast is a completion likelihood ratio, and a large
ratio can compare two improbable strings, so the two components are shown next to it.

## Cohorts (`ctx/cohorts.py`)

The semantic group of a record is `group_id` when present (v2) and
`stable_id(relation, subject)` otherwise (v1; the same key the auditor used as
`(relation, subject)`). Each cohort is saved as `analysis_v2/cohorts/<name>.json` with its
definition, fact ids, `n`, `n_by_label_and_dose`, `n_by_tier`, `n_groups`, exclusions (one
reason per excluded fact), and missingness (per replicate, per arm, per pass: the cohort facts
without a per-fact score).

| cohort | definition | historical (v1) size |
|---|---|---|
| `all_injected` | every injected synthetic fact (v2: role `injected`) | 5600 |
| `rival_untrained` | injected facts whose designated `competing_value` was never injected in the same group (the legacy "clean" subset; it still admits other injected values of the group) | 1991 |
| `exclusive_value` | injected facts for which no other value of the group was injected (injected values of the group == {value}); equals the audit's `strict_clean_fact_ids.json` | 740 (710 novel, 30 real) |
| `heldout_claims` | all held-out synthetic facts | 2401 |
| `heldout_groups` | held-out facts whose group has no injected member | 269 |
| `heldout_rivals` | held-out facts whose group has an injected member (v2 role `unasserted_rival`) | 2132 |
| `counterfact` | the CounterFact probe records; value = true object, competing = counterfactual object; each item is its own cluster | 2000 |
| `paraphrase_wellformed` | injected facts whose held-out-paraphrase contrast is well formed (below) | 5496 (104 malformed) |

`paraphrase_wellformed` for v1 data re-renders the legacy contrast: the evaluator scored
`para` against `para.replace(value, competing, 1)`; the fact qualifies only when that string
equals the frame of `ctx.factpool._build_relations()` that produced `para`, rendered with the
competing value (`legacy_paraphrase_contrast`). This reproduces the audit's
`malformed_paraphrases.json` (104 ids: 102 repeated chemical symbols, 2 values matched inside
the subject). For v2 data every paraphrase row carries explicit value spans, so all injected
facts with a reserved paraphrase qualify.

Headline cohort: `exclusive_value` for v1 probes (of the 1991 rival_untrained facts, 1251 still
have another injected value in their group, finding 2) and `all_injected` for v2 probes
(one injected value per group and whole-group hold-out by construction, so `all_injected`
coincides with `exclusive_value`). The reason is printed in the report. `all_injected` and
`rival_untrained` are always reported as sensitivity cohorts; `--cohort` restricts the
analysis.

Which cohort feeds which analysis:

| analysis | cohorts |
|---|---|
| dose-response, attribution asymmetry, paired contrasts on cloze passes, figures | each selected injected cohort (headline marked) |
| held-out baselines and their paired contrasts (dose 0, label cells) | `heldout_claims`, `heldout_groups`, `heldout_rivals`; the dose-response figure draws the `heldout_groups` baseline |
| held-out paraphrase | `paraphrase_wellformed` (tests and figure data); `all_injected` only as a labelled legacy comparison (no tests) |
| memorization join | headline cohort and `all_injected` |
| CounterFact | `counterfact` |

## Statistics (`ctx/stats.py`)

A cell is (cohort x dose x label) for injected cohorts and (cohort x label) for held-out
cohorts. Paired statistics are on per-fact differences between two arms within a cell;
facts missing in either arm drop out (the denominators `n`, `n_groups` are in every row).

* `wilcoxon_signed_rank(diffs)` -- `scipy.stats.wilcoxon(d, zero_method="wilcox",
  correction=False, method="auto")`, two-sided. Zeros: exact-zero differences are discarded
  before ranking (`zero_method="wilcox"`), so `n` is the number of non-zero differences;
  returns `None` below 10 (printed `n/a`). Ties: tied |d| get average ranks; with
  `method="auto"` scipy 1.15 uses the exact null distribution when n <= 50 with no ties and
  no zeros, a deterministic permutation test over all 2^n sign assignments when there are
  ties or zeros and n <= 13, and otherwise the normal approximation whose variance includes
  the tie correction -sum(t^3 - t)/48. No continuity correction. The historical helper is kept
  as `legacy_wilcoxon_signed_rank` (tie-incorrect, finding 11; 20 identical differences give
  8.86e-05 there versus 7.74e-06 here) for reproduction only. `analyze_beliefs.wilcoxon_signed_rank`
  is the new function (the auditor's script imports it from there).
* `cluster_bootstrap(values, groups, n_boot, seed)` -- percentile bootstrap of a mean that
  resamples semantic groups with replacement (G draws of G groups; every fact of a drawn group
  enters as often as the group was drawn); 95% and 90% intervals and the bootstrap SE.
  Deterministic: the generator is `ctx.rng.rng_for(seed, "bootstrap", <cell stream>)`, so a
  cell's interval does not depend on evaluation order. `--bootstrap N` (default 2000;
  0 disables), `--seed S` (default 2026). Cells with fewer than two groups get no interval.
  For CounterFact each item is its own cluster (ordinary item bootstrap).
* `holm(pvalues)` -- Holm step-down within one prespecified family. A family is one table:
  the dose x label cells of one (contrast, cohort, pass, metric). Untested cells (`None`) are
  outside the family (`family_size` = tested cells). Raw `p_wilcoxon` and adjusted `p_holm` are
  both reported. The `all/all` pooled row of a table is descriptive and outside the family
  (`p_holm` blank).
* Tier split (finding 17): for the neutral cloze and the `answer_plus_suffix` metric, each
  injected cohort also gets `__tier-<tier>` dose-response and paired tables restricted to one
  entity tier. These are secondary families, each Holm-adjusted within itself. On the
  historical data the real-tier cells of `exclusive_value` hold 0-6 facts (one fact in each
  dose-256 true/false cell), which is why they are secondary. Every paired row also counts
  `n_pos`, `n_neg` and `n_pref_flips` (facts whose preference sign, belief > 0, differs
  between the two arms), so individual reversals are visible where a mean would hide them.
* Equivalence: only under `--equivalence-margin M` is a TOST run (two one-sided tests at
  alpha 0.05 read off the 90% cluster-bootstrap interval: equivalent iff the whole interval
  lies inside (-M, M); cells without an interval are never equivalent). Without the flag the
  report prints that no equivalence claim is made from a non-significant difference.
* Replicates: every statistic is computed per replicate (`seed_*/` or `rep*/`); tables carry
  a `replicate` column; `replicates__paired.csv`, `replicates__dose_response.csv` and
  `replicates__counterfact.csv` summarize each statistic across replicates as mean, SD
  (ddof=1), min, max and `n_replicates`. Facts are never pooled across replicates: each
  replicate contributes one number per statistic. The report states that per-fact tests and
  intervals are within-run uncertainty only; with a single run per arm, training-run
  variation is unmeasured.

## Memorization join (finding 12)

`ctx.cohorts.realized_training_forms(record, protocol_cfg, exposure_rows=None, manifest_rows=None)`
returns the surface forms actually rendered for an injected fact, one per realized
occurrence, in this order of precedence:

1. manifest v2 rows with `paraphrase_index` (preferred for any protocol): the index is taken
   into `surface_forms`, the frame-ordered list the schema defines (`heldout_paraphrase_index`
   indexes the same list); an occurrence whose index equals the reserved paraphrase, or
   indices that differ across arms for the same occurrence, raise `ProtocolError`;
2. v1, or `render_policy["paraphrase"] == "cycle_from_zero"`: the legacy `render_raw` rule --
   `forms = training_surface_forms(record)` (surface forms minus the reserved paraphrase) and
   occurrence `occ < assigned_frequency` uses `forms[occ % len(forms)]`;
3. `"fixed"`: every occurrence uses `forms[render_policy.get("paraphrase_fixed_index", 0)]`;
4. `"seeded_start"`: not derivable without the manifest (`ProtocolError` asks for `--manifest`).

With `exposure_rows` (one arm's exposure-index rows for the fact) only occurrences with
`status == complete` and `step >= 0` are kept: the wording the model was actually scored on.
A fact is `verbatim_trained` when cloze template 0 with the blank filled by the value equals
one of its realized forms. Without an exposure index the classification is `nominal`; with
one, every arm that has an exposure file gets its own classification
(`classification_source` column), arm C keeps `nominal`. The memorization table reports,
per arm and subgroup, `n`, the mean template-0 belief, the mean of the other templates and
the within-fact paired difference with its cluster-bootstrap interval. Historical data:
1362 of 5600 facts are verbatim trained under the nominal rule (`eval_audit_summary.json:
template0_exactly_trained_n`); on `exclusive_value` 109 of 740.

## CounterFact (finding 16)

`counterfact__per_arm.csv`: per arm `n`, `n_correct`, accuracy (fraction with
logP(true object) > logP(counterfactual object)), mean belief with interval, and the two
logP components. `counterfact__paired_vs_C.csv`: per arm vs C the mean per-item change with
interval, mean |delta| with interval, the Wilcoxon p on the paired deltas, and preference
reversals in both directions with counts and rates (`C-true -> arm-false`: C prefers the true
object and the arm does not; `C-false -> arm-true`: the reverse). The report states
explicitly that similar means do not imply unchanged knowledge, and that training-run
variation is unmeasured without replicates.

## Outputs

```
<eval-dir>/analysis_v2/
  analysis.md                 the full report (all tables; what the console printed)
  analysis.txt                console log (written by the CLI; also what tracking uploads)
  cohorts/<name>.json         eight cohorts
  tables/*.csv                one CSV per table; every row carries replicate, cohort, n (and n_groups)
  figures/*.png               with --plot
```

Table names: `cohorts`, `dose_response__<cohort>__<pass>__<metric>[__tier-<tier>]`,
`heldout_baseline__<cohort>__<pass>__<metric>`,
`paired__<A>-<B>__<cohort>__<pass>__<metric>[__tier-<tier>]`,
`attribution_asymmetry__<cohort>__<pass>`, `paraphrase__<cohort>__heldout_paraphrase__full_sentence`,
`memorization__<cohort>`, `counterfact__per_arm`, `counterfact__paired_vs_C`, `replicates__*`.
Contrasts: R-C, X-C, R-X and, when an E arm is present, R-E, E-C, X-E.

Figures: `dose_response__<cohort>.png` (one panel per label, per-arm cell means with
cluster-bootstrap 95% error bars, dotted `heldout_groups` baselines), `dose_response_replicates__<cohort>.png`
(thin line per replicate, thick mean; only with more than one replicate),
`attribution_asymmetry__<cohort>.png` (attributed minus neutral per arm and dose with
intervals), `counterfact_paired_changes.png` (histograms of per-item change vs C with the
reversal counts). Each figure title names its cohort, `n`, pass, metric and replicate; the
figure data are the rows of the table with the same cohort.

Tracking (`track_analysis`, `_Tee`, unchanged mechanism via `ctx.tracking`): with tracking
enabled, `analysis.txt`, `analysis.md`, figures, tables, cohorts and the arm CSVs are uploaded
under `<build_repo>/<exp_tag>/analysis_v2/` and logged to a wandb `analysis` run.
`--no-tracking` or `CTX_TRACKING=0` disables it.

## Historical reproduction (archived CSVs + v1 probe files, 2000 resamples, seed 2026)

```
python analyze_beliefs.py --eval-dir <extracted eval_results.tgz>/eval \
    --probe-dir ~/.cache/ctx_wandb_to_hf/ctx_experiment_build/v1/probe_sets \
    --build-summary ~/.cache/ctx_wandb_to_hf/ctx_experiment_build/v1/build_summary.json \
    --manifest ~/.cache/ctx_wandb_to_hf/ctx_experiment_build/v1/manifest.csv.gz --no-tracking --plot
```

* Cohorts: all_injected 5600, rival_untrained 1991, exclusive_value 740 (710 novel / 30
  real; ids identical to `strict_clean_fact_ids.json`), heldout_claims 2401, heldout_groups
  269, heldout_rivals 2132, counterfact 2000, paraphrase_wellformed 5496 (the 104 malformed
  ids identical to `malformed_paraphrases.json`); 63 facts lack a `wrapper_conditional` score
  in every arm (missingness).
* Review headline table (exclusive_value, novel tier, dose 256, neutral cloze, answer_plus_suffix):
  true n=53 R-C +6.944 [5.80, 8.12], X-C +0.507 [-0.29, 1.28]; false n=57 R-C +7.636
  [5.72, 9.47], X-C +0.497 [-0.38, 1.34]; contested n=32 R-C +7.883 [5.70, 10.04], X-C +0.378
  [-0.53, 1.52] (cluster-bootstrap 95% intervals). Every strict_clean/novel row of
  `dose_sensitivity.csv` is reproduced to 1e-6.
* Paraphrase (full_sentence): paraphrase_wellformed C/R/X -0.044 / 1.466 / 0.014 (n=5496),
  all_injected 0.152 / 1.911 / 0.214 (n=5600), as in the audit.
* CounterFact: accuracy C 0.7635, R 0.769, X 0.778; R-C mean delta -0.002 [-0.074, 0.071],
  mean |delta| 1.207, 209 reversals = 99 C-true->R-false + 110 C-false->R-true; X-C -0.045
  [-0.117, 0.029], mean |delta| 1.201, 193 reversals = 82 + 111.
* Memorization: 1362 of 5600 verbatim trained (nominal rule); the legacy manifest carries no
  paraphrase indices and is ignored, no exposure index exists for the historical run.

## Caveats and open questions

* The manifest `paraphrase_index` is read as an index into `surface_forms`; the v2 builder
  does not exist in this checkout, so this convention (and `paraphrase_fixed_index` for the
  `fixed` policy) must be kept when it is written.
* With an exposure index the verbatim classification is per arm (R and X see different
  cropping); the table says which arm's index was used. No exposure index exists for the
  historical run, so its memorization rows are nominal.
* All historical statements are recalculations of saved scores from one training run per
  arm; nothing here says anything about retraining, rescoring, or training-run variance.

Tests: `$HOME/.cache/ctx_cpu_venv/bin/python -m pytest contextualization/tests/test_analysis.py -q`
(historical tests skip when the v1 probe files are not under `~/.cache/ctx_wandb_to_hf/ctx_experiment_build/v1`).
