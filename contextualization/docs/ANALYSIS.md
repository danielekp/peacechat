# Analysis v2: `analyze_beliefs.py`, `ctx/cohorts.py`, `ctx/stats.py`

Written 2026-10-14, revised the same day after the 2026-10-14 code review. This stage re-reads
saved evaluation scores (`eval/beliefs_arm<A>.csv` with their `.meta.json`, legacy or eval
schema v2) and the probe files, and writes everything under `<eval-dir>/analysis_v2/`. It
never rescores a checkpoint, never retrains, and never touches the historical artifacts
(`eval_results.tgz`, `reviews/`, `figures/`, `REPORT.md`, `RESULTS.md`, the legacy
`analysis.txt` / `beliefs_dose_response.png`).

It replaces the July-2026 `analyze_beliefs.py` after the 2026-10-13 review. The findings it
answers: 2 (contaminated clean subset and held-out baseline), 10 (curves, tests and plot drawn
from different populations), 11 (tie-incorrect Wilcoxon, no multiplicity control, no cluster
or replicate uncertainty), 12 (memorization subgroup not the one computed), 13 (likelihood
ratio, components not reported), 16 (CounterFact means hide item-level change), 17
(real-entity cells are tiny; count reversals). The 2026-10-14 review added: 6 (incompatible or
mock evaluations accepted as a matched experiment), 7 (missing templates manufacture a paired
effect), 8 (reciprocal held-out candidates cancel every signed mean), 17 (fixed-rendering
forms reconstructed as index 0) and 18 (per-fact Wilcoxon next to cluster bootstrap).

```
python analyze_beliefs.py --eval-dir <exp>/eval [--probe-dir <exp>/probe_sets]
    [--build-summary <exp>/build_summary.json] [--manifest <exp>/manifest.csv[.gz]]
    [--exposure-dir <exp>/exposure] [--cohort NAME ...] [--plot] [--bootstrap N]
    [--equivalence-margin M] [--seed S] [--exclude-facts ids.txt] [--no-tracking]
    [--compat strict|legacy|historical-builds [--compat-reason "..."]] [--allow-incomplete]
    [--allow-incomplete-replicates] [--no-supplement]
```

Changes of 2026-11-11 (review of the funded runs: two complete seeds and a partial third) are
collected in the last section, "Replicated analysis (review 2026-11-11)".

## Inputs

| input | how it is found | notes |
|---|---|---|
| eval CSVs | `--eval-dir` holding `beliefs_arm<A>.csv` (one replicate, named `rep0`), or subdirectories each holding the CSVs (one replicate each: `seed_<S>/` as `run_arm.py --init-seed S` / `eval_beliefs.py --model-tag d<depth>_s<S>` write them, `rep*/`, or a custom model tag); a directory holding both is refused as ambiguous | legacy 13-column CSVs and eval-schema-v2 CSVs both load; missing v2 columns are treated as absent; the per-item `metric` / `template_id` / `scoring_policy` columns and the scored texts (`context`, `cand_value`, `cand_competing`, `value_span_chars`, `boundary_k`, `n_ctx_tokens`) are kept for the compatibility check |
| eval metadata | `beliefs_arm<A>.meta.json` next to each CSV (written by `eval_beliefs.py`) | validated before aggregation (below); CSVs without one need `--compat legacy` |
| probe files | `--probe-dir` (default `<eval-dir>/../probe_sets`): `injected_facts.jsonl`, `heldout_facts.jsonl` | v1 records (July 2026) or v2 records (`schema_version: 2`); required, cohorts cannot be built without them |
| build summary | `--build-summary` (default `<probe-dir>/../build_summary.json`) | gives the render configuration (protocol, seed, render policy) for the memorization join; a legacy summary resolves to protocol v1 via `ctx.protocol` |
| manifest | `--manifest` | only a v2 manifest (column `paraphrase_index`) is used; a legacy manifest is recognised from its header and ignored without reading its rows |
| exposure index | `--exposure-dir`: ONE index (`PREFLIGHT_OK.json` + `<arm>/occurrence_exposure.csv`, exposure schema v1) or a directory holding one index per data seed (`exposure_by_seed/seed_<S>/`, up to two levels deep) | keeps only `status == complete` and `step >= 0` occurrences in the memorization join, per arm AND per replicate: every replicate reads the index of its own run (binding rule in the last section) |
| supplementary evaluations | `supp_beliefs_arm<A>.csv` (+ `.meta.json`) next to each main CSV, written by `eval_beliefs.py --supplement` | optional; bound to the main evaluation's checkpoint and analysed in section 5b; `--no-supplement` ignores them |
| exclusions | `--exclude-facts` file of fact ids | dropped from every cohort and recorded as exclusions with the reason `excluded via --exclude-facts (<file>)` |

CounterFact records: v2 records carry `fact_id`; v1 records do not, and get
`counterfact_<idx:05d>` from their position in `heldout_facts.jsonl` (all records counted),
exactly as the legacy evaluator assigned them (historical ids `counterfact_02401` ..
`counterfact_04400`).

## Evaluation metadata compatibility (finding 6, 2026-10-14)

Before anything is aggregated, `check_compatibility` reads every arm's `.meta.json` (the
file the corrected evaluator writes next to each CSV: `mode`, `tokenizer.tokenizer_sha256`,
`input_hashes`, `checkpoint{model_sha256, init_seed, data_seed, step}`, `scoring_policy`,
`protocol`, `csv_sha256`, `arm`; see `docs/SCHEMAS_V2.md`) and enforces:

| scope | rule | why |
|---|---|---|
| each CSV | a `.meta.json` exists; its `csv_sha256` equals the sha256 of the CSV on disk; its `arm` is the file's arm; `mode` is `model_inference` or `dry_run_mock_scores`; `scoring_policy`, `protocol` and the two probe-file hashes are recorded | the CSV is the file the evaluator wrote, labelled as the arm it claims |
| within a replicate | identical `mode` across arms; identical tokenizer sha256; identical `input_hashes` of `probe_sets/injected_facts.jsonl` and `probe_sets/heldout_facts.jsonl`; identical `scoring_policy` and `protocol`; identical CSV `schema_version`; per item (`probe_set, pass, fact_id, template_idx`) identical `metric` / `template_id` / `scoring_policy` columns; identical `checkpoint.init_seed` and `data_seed` when recorded; distinct `checkpoint.model_sha256` across arms when recorded | no mixed scoring versions, no mock-vs-real contrast, the same probes for every arm, a matched-seed replicate, no checkpoint scored twice under two arm names |
| across replicates | for the same arm, distinct `checkpoint.model_sha256` and distinct `init_seed` when recorded | copied checkpoints cannot pose as independent training runs |

Violations stop the run with `SystemExit` and the full list; the report is always saved first
as `analysis_v2/compatibility.json` (`ok`, `mock`, `violations`, `warnings`, per replicate
and arm: mode, tokenizer, hashes, checkpoint identity, recorded and actual CSV sha256) and the
same table opens `analysis.md` (section 0a). Modes:

* `--compat strict` (default): every rule above.
* `--compat legacy`: CSVs without `.meta.json` are accepted (historical data; the report says
  "legacy compatibility mode" and lists the CSVs whose provenance is unverified). Everything
  that *is* recorded is still enforced.
* `--compat historical-builds --compat-reason "..."`: metas are required but probe/build
  hashes (`input_hashes`) may differ between arms by design, e.g. the July C checkpoint
  evaluated on build v0 while R/X were evaluated on build v1; the reason is written into
  `analysis.md` and `compatibility.json`. Nothing else is relaxed.

If every arm is `dry_run_mock_scores` the analysis runs but is labelled: a MOCK banner at the
top of the report, `[MOCK SCORES]` on every printed table, a `mock_scores = MOCK` column in
every table row/CSV (the replicate summary tables included), "MOCK SCORES" in every figure
title, `mock: true` in `compatibility.json`. A mix of mock and real arms is a violation.

### The experiment-wide contract and the input binding (recheck 2026-10-14, findings 1-3)

The recheck showed three gaps in the rules above: (1) they were enforced per replicate only,
so one real and one mock replicate pooled into a mean with no MOCK marker; (2) the evaluations
were compared with each other but never with the files THIS analysis builds cohorts, doses and
labels from, so another build's probe files (same ids, doses rewritten) relabelled valid
evaluations; (3) a template id and a scoring-policy name do not identify the text that was
scored, and the initial-weight identity and the training budget were not compared. Rules now:

| scope | rule | why |
|---|---|---|
| every CSV of the analysis, all replicates | one `mode`, one tokenizer sha256, one `scoring_policy`, one `protocol`, one `probe_construction` (the candidate-construction version `ctx.probes.PROBE_CONSTRUCTION_VERSION` the evaluator records), one CSV schema, one set of probe-file hashes | one scoring contract per experiment; mock and model inference are never pooled (an entirely mock experiment stays labelled MOCK everywhere) |
| every scored item (`probe_set, pass, fact_id, template_idx`) | one identity across every CSV: sha256 of `context`, `cand_value`, `cand_competing`, `value_span_chars`, `metric`, `template_id`, `scoring_policy`, `boundary_k`, `n_ctx_tokens` (`ctx.protocol.scored_item_identity`, recomputed from the columns, never trusted from the CSV's own `item_sha256`) | two evaluators that disagree on what they scored under the same template id are caught (the reproduction: the 031c4e5 constructor vs the repaired one, 15 wrapper competitors) |
| analysis inputs | every meta's `input_hashes` of the two probe files and of `build_summary.json` must equal the sha256 of the files selected with `--probe-dir` / `--build-summary`; the checkpoint's own `bindings` (build summary, probe hashes) must match them too unless the evaluation recorded `build_mismatch_allowed`; `--manifest` must be the build's own (`BUILD_COMPLETE.json` next to the summary records its sha256); `--exposure-dir` must carry a `PREFLIGHT_OK.json` / `cross_arm.json` whose `prepack_dataset_id` per arm equals the checkpoint's `bindings.prepack_dataset_id` and whose build summary / manifest hashes equal the selected ones (otherwise the exposure inputs are reported as unbound) | the cohorts, doses, labels and memorization classifications come from the files the evaluations were scored on, not from another build |
| within a replicate | one `checkpoint.init_param_sha256` across arms (matched initial weights) | the arms started from the same parameters, not merely from the same integer seed |
| matched budget | one `checkpoint.step`, `total_batch_size`, `num_iterations`, `max_seq_len` across every model-inference evaluation; `--allow-unequal-horizon "reason"` declares an unequal-budget experiment (the reason is recorded in `compatibility.json` and `analysis.md`) | a comparison across different training budgets is an experiment of its own, never an ordinary matched replicate |
| across replicates | the same arm has distinct `init_param_sha256` (as well as distinct `model_sha256` and `init_seed`) | replicates are independent training runs |
| training objective | one `checkpoint.loss_normalization` across every model-inference evaluation; an unrecorded value is the historical `microbatch_mean` and never the new `fixed_nominal`; no flag relaxes this | arms trained under different objectives are not a controlled comparison (decision 2026-10-15) |
| required provenance | a model-inference meta must record `probe_construction`, `checkpoint.init_param_sha256` and `checkpoint.step`; a v2 CSV must carry the scored texts | missing provenance is a violation in strict mode and an "unverified" warning under `--compat legacy` |

The selected inputs and their hashes open section 0a of `analysis.md` (table "Analysis inputs
every evaluation is bound to") and are saved as `compatibility.json["inputs"]`; per CSV,
`inputs_bound` says whether every binding that could be checked held. `--compat
historical-builds --compat-reason` turns the probe/build-hash mismatches into recorded
warnings (an intentional, declared remapping); the item identity, init identity and horizon
rules are never relaxed.

## Per-fact belief and the three metrics

For every (probe_set, pass, fact) the belief is `logp_value - logp_competing`, averaged over
the fact's cloze templates (`template_idx`) **that were scored in every arm that has the
pass** (next section). Three scoring metrics are reported separately and never mixed in one
table:

| metric | columns | passes |
|---|---|---|
| `answer_plus_suffix` | `logp_value`, `logp_competing` (legacy definition: answer tokens plus the template suffix) | cloze passes (`cloze_neutral`, `cloze_attributed`, `cloze_attributed_fixedsource`, `wrapper_conditional`) |
| `answer_only` | `logp_value_answer`, `logp_competing_answer` (eval schema v2 only) | the same cloze passes, in additional tables suffixed `__answer_only` |
| `full_sentence` | `logp_value`, `logp_competing` of the whole reserved paraphrase sentence vs the same sentence with the competing value | `heldout_paraphrase` |

Every dose-response and paraphrase table also carries `mean_logp_value` and
`mean_logp_competing` (finding 13): the contrast is a completion likelihood ratio, and a large
ratio can compare two improbable strings, so the two components are shown next to it.

## Exact item joins and missingness (finding 7, 2026-10-14)

Items are keyed by their exact identity `(probe_set, pass, fact_id, template_idx)`. For every
(probe_set, pass) present in at least one arm, `common_items` takes the items scored (finite
`logp_value` and `logp_competing`; for `answer_only` also the answer columns) in **every arm
that has the pass**; `per_fact` averages exactly those templates in every arm. Items scored
in some arms only are dropped from *every* curve and comparison -- single-arm dose-response
means, held-out baselines, paraphrase means, attribution asymmetry, memorization,
CounterFact and all paired contrasts draw on the same common item set -- so an arm difference
can never be an input-coverage difference. Arms that lack a pass entirely do not shrink the
common set of the arms that have it; they get no score for that pass. The report prints, per
(probe_set, pass), the arms with the pass, the common item count and the items dropped per arm.

Missingness is recorded per cohort, replicate, arm and pass in `cohorts/<name>.json`
(`missingness[rep][arm][pass]`): `n_missing` / `fact_ids` (cohort facts with no item in that
arm), `partial_templates` = `{fact_id: {template_idx: [...], template_id: [...]}}` with
`n_partial_templates` (facts scored in that arm on fewer templates than in some other arm),
and `absent_pass: true` when the arm has no row of that pass at all. Coverage disagreements
(items scored in some arms only) are listed under `details.coverage[rep].disagreements`.

For eval-schema-v2 CSVs (the corrected evaluator, which must score the same items in every
arm) any coverage disagreement in a selected cohort stops the run (`SystemExit`, after the
cohorts with their missingness are saved); `--allow-incomplete` proceeds on the common items
and the report says so. For legacy CSVs the common-item cohort is used and reported without
stopping. The historical CSVs have identical item sets in C, R and X (45 141 items each), so
the historical numbers are unchanged; their 63 facts without a `wrapper_conditional` score
are missing in every arm and are therefore missingness, not a disagreement.

## Cohorts (`ctx/cohorts.py`)

The semantic group of a record is `group_id` when present (v2) and
`stable_id(relation, subject)` otherwise (v1; the same key the auditor used as
`(relation, subject)`). Each cohort is saved as `analysis_v2/cohorts/<name>.json` with its
definition, fact ids, `n`, `n_by_label_and_dose`, `n_by_tier`, `n_groups`, exclusions (one
reason per excluded fact), missingness and coverage (above).

| cohort | definition | historical (v1) size |
|---|---|---|
| `all_injected` | every injected synthetic fact (v2: role `injected`) | 5600 |
| `rival_untrained` | injected facts whose designated `competing_value` was never injected in the same group (the legacy "clean" subset; it still admits other injected values of the group) | 1991 |
| `exclusive_value` | injected facts for which no other value of the group was injected (injected values of the group == {value}); equals the audit's `strict_clean_fact_ids.json` | 740 (710 novel, 30 real) |
| `heldout_claims` | one record per wholly held-out group (the `heldout_groups` orientation) plus every `heldout_rivals` record that is not the mirror of an injected probe (v2: mirrors excluded, so the cohort equals `heldout_groups` in the one-sided design; v1 keeps the historical composition) | 2278 (146 + 2132) |
| `heldout_groups` | held-out facts whose group has no injected member, ONE stable orientation per group (finding 8, below) | 146 groups (269 records) |
| `heldout_groups_both` | the raw two-record view of the same groups, both reciprocal candidates; its definition carries the warning that every signed mean, interval and TOST verdict of this cohort is zero by construction | 269 |
| `heldout_rivals` | held-out facts whose group has an injected member (v2 role `unasserted_rival`; one record per group in v2, each the exact MIRROR of an injected probe: `details.n_mirrors`, `details.mirror_of`) | 2132 |
| `counterfact` | the CounterFact probe records; value = true object, competing = counterfactual object; each item is its own cluster | 2000 |
| `paraphrase_wellformed` | injected facts whose held-out-paraphrase contrast is well formed in the evaluation being analysed (CSV-aware, below) | 5496 with the legacy CSVs |
| `paraphrase_historical_reference` | the probe-derived well-formedness under the historical evaluator (below); kept for the historical comparison | 5496 (104 malformed) |

### Held-out orientation (finding 8, 2026-10-14)

The two candidates of a wholly held-out group are scored on the same prompt with the
candidates swapped, so their signed beliefs are exact negatives: pooling both forces every
signed mean, every bootstrap resample and every TOST verdict to zero, whatever the model did.
`orient_heldout_groups` keeps ONE record per group: real-world groups the
`truth_value == "true"` candidate, synthetic v2 groups the `candidate_slot == "a"` candidate,
legacy v1 groups holding several or no `true`-labelled records the smallest `fact_id`
(historical data: 92 groups by truth label, 44 singletons, 10 by smallest id). The rule reads
probe records only, so the orientation is identical across arms, replicates and seeds. The
dropped reciprocal candidates are listed as exclusions (`group_orientation: ...`) and the
choice per group under `details.orientation`. `heldout_groups` and `heldout_claims` use the
orientation; group-level `mean_abs_diff` (mean |delta|) and `n_pref_flips` are therefore
counted once per group. `heldout_groups_both` keeps the raw view for record counts and the
historical comparison only; its tables are written but not printed, and their notes repeat the
warning. `heldout_rivals` already holds one record per group in v2 (v1 groups with several
held-out values are kept as they are and are not reciprocal pairs by construction).

Mirrors (review 2026-11-11, finding 6). In protocol v2 a held-out rival shares the cloze
prompts of the injected record of its group and has the two candidates swapped, so every item
scored for it is the injected item with the candidates exchanged: its signed belief, every
paired difference and every interval are `all_injected`'s with the sign flipped (R - C on
`heldout_rivals` was exactly -9.0366 on seed 1). `ctx.cohorts.mirror_of_injected` detects this
from the probe records. For v2 probes the mirrors are excluded from `heldout_claims`
(exclusion reason `mirror_of_injected`), which therefore is a real never-trained baseline; the
`heldout_rivals` tables are still written (CSV only) with a note that they repeat
`all_injected`, and the cohort is left out of the cross-replicate and noise-floor tables.

### Paraphrase cohorts

`paraphrase_historical_reference` re-renders the legacy contrast from the probe records: the
July evaluator scored `para` against `para.replace(value, competing, 1)`; the fact qualifies
only when that string equals the frame of `ctx.factpool._build_relations()` that produced
`para`, rendered with the competing value (`legacy_paraphrase_contrast`). This reproduces the
audit's `malformed_paraphrases.json` (104 ids: 102 repeated chemical symbols, 2 values matched
inside the subject). For v2 probes every injected fact with a reserved paraphrase qualifies.
`malformed_paraphrase_ids()` returns this cohort's malformed ids.

`paraphrase_wellformed` is CSV-aware (`build_cohorts(..., eval_info=...)`, derived by
`analyze_beliefs.eval_info_from_replicates` from the CSVs): with eval-schema-v2 CSVs the
corrected evaluator rendered both candidates structurally from the relation frame (every value
slot replaced, subject untouched), so every injected fact with a scored `heldout_paraphrase`
row in the CSVs is well formed by construction -- the v2 column `legacy_malformed` only flags
what the historical evaluator got wrong, and those facts are *included* (the count is saved
as `details.n_historical_malformed_included`); facts without a paraphrase row are excluded
(`no_paraphrase_row_in_eval`). With legacy CSVs (or no CSV information) it equals
`paraphrase_historical_reference`. Both cohorts get single-arm and paired tables; the
historical reference is marked as such.

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
| held-out baselines and their paired contrasts (dose 0, label cells) | `heldout_claims`, `heldout_groups`, `heldout_rivals` (printed), `heldout_groups_both` (CSV only, warning in the note); the dose-response figure draws the `heldout_groups` baseline |
| held-out paraphrase | `paraphrase_wellformed` (tests and figure data), `paraphrase_historical_reference` (historical comparison, own tables); `all_injected` only as a labelled legacy comparison (no tests) |
| memorization join | headline cohort and `all_injected` |
| CounterFact | `counterfact` |

## Statistics (`ctx/stats.py`)

A cell is (cohort x dose x label) for injected cohorts and (cohort x label) for held-out
cohorts. Paired statistics are on per-fact differences between two arms within a cell, on the
common items; facts missing in either arm drop out (the denominators `n`, `n_groups` are in
every row).

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
* Cluster-aware test (finding 18, 2026-10-14): `paired_cell` runs the Wilcoxon test on the
  per-fact differences only when every semantic group contributes one fact
  (`p_wilcoxon_level = "fact"`); when any group contributes more than one fact it ranks ONE
  value per group, the group's mean difference (`group_mean_differences`), and labels the
  result `p_wilcoxon_level = "group"` (`n_wilcoxon` = non-zero values at that level; `None`
  below 10). `p_wilcoxon_fact` is the naive per-fact p and is given only when it is the test
  used (one fact per group), otherwise `None`: `paired_cell([1.]*20, ["same-group"]*20)` now
  reports one group and `p_wilcoxon = None` rather than 7.7e-6. Holm is applied to
  `p_wilcoxon`. Every paired row also carries `mean_abs_diff` (mean |difference|; per group
  for the oriented held-out cohorts). This fixes the dependence between facts of one group
  within one training run; uncertainty **between training runs** is a separate quantity that
  only replicates measure (below) and no per-fact or per-group test addresses.
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
* Tier split (finding 17 of 2026-10-13): for the neutral cloze and the `answer_plus_suffix`
  metric, each injected cohort also gets `__tier-<tier>` dose-response and paired tables
  restricted to one entity tier. These are secondary families, each Holm-adjusted within
  itself. On the historical data the real-tier cells of `exclusive_value` hold 0-6 facts (one
  fact in each dose-256 true/false cell), which is why they are secondary. Every paired row
  also counts `n_pos`, `n_neg` and `n_pref_flips` (facts -- groups for the oriented held-out
  cohorts -- whose preference sign, belief > 0, differs between the two arms), so individual
  reversals are visible where a mean would hide them.
* Equivalence: only under `--equivalence-margin M` is a TOST run (two one-sided tests at
  alpha 0.05 read off the 90% cluster-bootstrap interval: equivalent iff the whole interval
  lies inside (-M, M); cells without an interval are never equivalent). Without the flag the
  report prints that no equivalence claim is made from a non-significant difference. A TOST
  verdict on `heldout_groups_both` is vacuous (finding 8) and the table note says so.
* Replicates: every statistic is computed per replicate (`seed_*/`, `rep*/` or a custom
  model tag); tables carry a `replicate` column; `replicates__paired.csv`,
  `replicates__dose_response.csv` and `replicates__counterfact.csv` summarize each statistic
  across the COMPLETE replicates as mean, SD (ddof=1), min, max and `n_replicates`. Facts are
  never pooled across replicates as if they were independent observations. The report states
  that per-fact tests and intervals are within-run uncertainty only; with a single run per
  arm, training-run variation is unmeasured. Replicates must be independent training runs
  (`check_compatibility` refuses shared checkpoints or init seeds across replicates). The
  cross-replicate inference (replicate interval, facts x replicates variance components,
  same-arm noise floor) and the handling of incomplete replicates are described in the last
  section.
* Label rows: when a paired table has several labels and several doses it also carries one
  descriptive row per label with the doses pooled (`dose == pooled`, outside the Holm family;
  `dose == all` stays the single all-labels row). On the funded build this is where the 17
  real-world false claims appear as one row (`__tier-real` tables).

## Memorization join (finding 12; finding 17 of 2026-10-14)

`ctx.cohorts.realized_training_forms(record, protocol_cfg, exposure_rows=None, manifest_rows=None)`
returns the surface forms actually rendered for an injected fact, one per realized
occurrence, in this order of precedence:

1. manifest v2 rows with `paraphrase_index` (preferred for any protocol): the index is taken
   into `surface_forms`, the frame-ordered list the schema defines (`heldout_paraphrase_index`
   indexes the same list); an occurrence whose index equals the reserved paraphrase, or
   indices that differ across arms for the same occurrence, raise `ProtocolError`;
2. v1: the legacy `render_raw` rule -- `forms = training_surface_forms(record)` (surface forms
   minus the reserved paraphrase) and occurrence `occ < assigned_frequency` uses
   `forms[occ % len(forms)]`;
3. v2 without a manifest: the renderer's own index rule, `ctx.templates._pick` --
   `forms[_pick(policy, len(forms), occ, seed, fact_id, "render.v2.paraphrase")]` for
   `render_policy["paraphrase"]` in `cycle_from_zero` / `seeded_start` / `random` / `fixed`,
   with `seed = protocol_cfg["seed"]` (the build seed from `build_summary.json`; `forms` is in
   the same order as the renderer's `training_frame_indices`). A seeded policy without a seed
   raises `ProtocolError` ("never assumed to be index 0"); on the 600-group fixed-rendering
   pool the old index-0 fallback reconstructed 331 of 419 forms and 89 subgroups wrongly, the
   rule reconstructs all 419 (and every seeded_start / random form) exactly.

With `exposure_rows` (one arm's exposure-index rows for the fact) only occurrences with
`status == complete` and `step >= 0` are kept: the wording the model was actually scored on.
A fact is `verbatim_trained` when cloze template 0 with the blank filled by the value equals
one of its realized forms. Without an exposure index the classification is `nominal`; with
one, every intervention arm that has an exposure file gets its own classification
(`classification_source` column), arm C keeps `nominal` (the control renders no claim).
Manifest rows of the control arm carry `paraphrase_index = -1` and are skipped; a negative
index in an intervention arm is an error (before 2026-11-11 the control rows raised, which
switched this section off whenever `--manifest` was given on a real build). The template-0
split is kept for the historical comparison; section 4c (`verbatim_items__<cohort>__...`)
classifies EVERY cloze item (`ctx.cohorts.classify_verbatim_items`), because in v2 pools
template 1 is a trained sentence for some relations, and reports each contrast separately on
the verbatim-trained items and on the others. The memorization table reports,
per arm and subgroup, `n`, the mean template-0 belief, the mean of the other templates and
the within-fact paired difference with its cluster-bootstrap interval. Historical data:
1362 of 5600 facts are verbatim trained under the nominal rule (`eval_audit_summary.json:
template0_exactly_trained_n`); on `exclusive_value` 109 of 740.

## CounterFact (finding 16)

`counterfact__per_arm.csv`: per arm `n`, `n_correct`, accuracy (fraction with
logP(true object) > logP(counterfactual object)), mean belief with interval, and the two
logP components. `counterfact__paired_vs_C.csv`: per arm vs C the mean per-item change with
interval, mean |delta| with interval, the Wilcoxon p on the paired deltas (`p_wilcoxon_level`
is `fact`: every item is its own cluster), and preference reversals in both directions with
counts and rates (`C-true -> arm-false`: C prefers the true object and the arm does not;
`C-false -> arm-true`: the reverse). The report states explicitly that similar means do not
imply unchanged knowledge, and that training-run variation is unmeasured without replicates.

## Outputs

```
<eval-dir>/analysis_v2/
  analysis.md                 the full report (all tables; what the console printed)
  analysis.txt                console log (written by the CLI; also what tracking uploads)
  compatibility.json          the metadata compatibility report (finding 6); written even when the run stops
  cohorts/<name>.json         ten cohorts (with missingness and coverage per replicate)
  tables/*.csv                one CSV per table; every row carries replicate, cohort, n (and n_groups)
  figures/*.png               with --plot
```

Table names: `compatibility`, `cohorts`, `dose_response__<cohort>__<pass>__<metric>[__tier-<tier>]`,
`heldout_baseline__<cohort>__<pass>__<metric>`,
`paired__<A>-<B>__<cohort>__<pass>__<metric>[__tier-<tier>]`,
`attribution_asymmetry__<cohort>__<pass>`, `paraphrase__<cohort>__heldout_paraphrase__full_sentence`,
`memorization__<cohort>`, `verbatim_items__<cohort>__cloze_neutral__answer_plus_suffix`,
`by_relation__<A>-<B>__<cohort>__cloze_neutral__answer_plus_suffix`, `counterfact__per_arm`,
`counterfact__paired_vs_C`, `replicates__paired`, `replicates__crossed`,
`replicates__same_arm_noise`, `replicates__dose_response`, `replicates__counterfact` and, with
supplementary evaluations, `supp_paired__<A>-<B>__<cohort>__<pass>__<metric>`,
`supp_decomposition__<cohort>`, `supp_by_relation__<cohort>`, `supp_own_frame__<cohort>`.
Contrasts: R-C, X-C, R-X and, when an E arm is present, R-E, E-C, X-E. Paired rows carry
`p_wilcoxon`, `p_wilcoxon_level`, `p_wilcoxon_fact`, `n_wilcoxon`, `mean_abs_diff` next to
the earlier columns; in a mock run every row also carries `mock_scores`.

Figures: `dose_response__<cohort>.png` (one panel per label, per-arm cell means with
cluster-bootstrap 95% error bars, dotted `heldout_groups` baselines -- one record per group),
`dose_response_replicates__<cohort>.png` (thin line per replicate, thick mean; only with more
than one replicate), `attribution_asymmetry__<cohort>.png` (attributed minus neutral per arm
and dose with intervals), `counterfact_paired_changes.png` (histograms of per-item change vs C
with the reversal counts). Each figure title names its cohort, `n`, pass, metric and
replicate (and MOCK SCORES for a mock run); the figure data are the rows of the table with
the same cohort, computed on the same common items as every comparison.

Tracking (`track_analysis`, `_Tee`, unchanged mechanism via `ctx.tracking`): with tracking
enabled, `analysis.txt`, `analysis.md`, `compatibility.json`, figures, tables, cohorts and
the arm CSVs are uploaded under `<build_repo>/<exp_tag>/analysis_v2/` and logged to a wandb
`analysis` run. `--no-tracking` or `CTX_TRACKING=0` disables it.

## Historical reproduction (archived CSVs + v1 probe files, 2000 resamples, seed 2026)

The archived CSVs carry no `.meta.json`, so they run under `--compat legacy`:

```
python analyze_beliefs.py --eval-dir <extracted eval_results.tgz>/eval \
    --probe-dir ~/.cache/ctx_wandb_to_hf/ctx_experiment_build/v1/probe_sets \
    --build-summary ~/.cache/ctx_wandb_to_hf/ctx_experiment_build/v1/build_summary.json \
    --manifest ~/.cache/ctx_wandb_to_hf/ctx_experiment_build/v1/manifest.csv.gz \
    --compat legacy --no-tracking --plot
```

* Compatibility: legacy mode, 3 of 3 CSVs without meta, no violation; item sets identical
  across C, R and X (45 141 items each), so the common-item join changes nothing.
* Cohorts: all_injected 5600, rival_untrained 1991, exclusive_value 740 (710 novel / 30
  real; ids identical to `strict_clean_fact_ids.json`), heldout_claims 2278, heldout_groups
  146 groups (heldout_groups_both 269 records), heldout_rivals 2132, counterfact 2000,
  paraphrase_wellformed 5496 = paraphrase_historical_reference 5496 (the 104 malformed ids
  identical to `malformed_paraphrases.json`); 63 facts lack a `wrapper_conditional` score in
  every arm (missingness, no coverage disagreement).
* Review headline table (exclusive_value, novel tier, dose 256, neutral cloze, answer_plus_suffix;
  one fact per group, so `p_wilcoxon_level = fact`):
  true n=53 R-C +6.944 [5.80, 8.12], X-C +0.507 [-0.29, 1.28]; false n=57 R-C +7.636
  [5.72, 9.47], X-C +0.497 [-0.38, 1.34]; contested n=32 R-C +7.883 [5.70, 10.04], X-C +0.378
  [-0.53, 1.52] (cluster-bootstrap 95% intervals). Every strict_clean/novel row of
  `dose_sensitivity.csv` is reproduced to 1e-6.
* Paraphrase (full_sentence): paraphrase_wellformed and paraphrase_historical_reference C/R/X
  -0.044 / 1.466 / 0.014 (n=5496), all_injected 0.152 / 1.911 / 0.214 (n=5600), as in the audit.
* CounterFact: accuracy C 0.7635, R 0.769, X 0.778; R-C mean delta -0.002 [-0.074, 0.071],
  mean |delta| 1.207, 209 reversals = 99 C-true->R-false + 110 C-false->R-true; X-C -0.045
  [-0.117, 0.029], mean |delta| 1.201, 193 reversals = 82 + 111.
* Memorization: 1362 of 5600 verbatim trained (nominal rule); the legacy manifest carries no
  paraphrase indices and is ignored, no exposure index exists for the historical run.

## Caveats and open questions

* The manifest `paraphrase_index` is read as an index into `surface_forms`; the v2 builder
  writes it that way (`docs/SCHEMAS_V2.md`) and the manifest, when supplied, takes precedence
  over the policy rule.
* With an exposure index the verbatim classification is per arm (R and X see different
  cropping); the table says which arm's index was used. No exposure index exists for the
  historical run, so its memorization rows are nominal.
* `--compat legacy` and `--compat historical-builds` are documented concessions: the report
  names what was not verified (legacy) or why the builds differ (historical-builds). They
  never silence a recorded csv_sha256 mismatch, a mode mix, a tokenizer difference or a
  shared checkpoint.
* All historical statements are recalculations of saved scores from one training run per
  arm; nothing here says anything about retraining, rescoring, or training-run variance.

Tests: `$HOME/.cache/ctx_cpu_venv/bin/python -m pytest contextualization/tests/test_analysis.py -q`
(historical tests skip when the v1 probe files are not under `~/.cache/ctx_wandb_to_hf/ctx_experiment_build/v1`;
the regression tests for findings 6, 7, 8, 17 and 18 rebuild the review's reproductions on
small v2 pools).

## Replicated analysis (review 2026-11-11)

Review of the funded runs with two complete seeds and a third in progress
(`reviews/2026-11-07-seed1-results/` is the single-seed write-up it started from). The numbers
quoted below are from seeds 1 and 2 of build `ctx_v2_build1_recovered`.

### Incomplete replicates (finding 3)

Any sub-directory with at least one arm CSV used to be a replicate, so a queue still running
(seed 3 with arm C only) was accepted silently: `replicates__dose_response`,
`replicates__counterfact` and the overlay figure averaged arm C over three seeds and the other
arms over two (X - C read off those tables was off by about 0.06, the size of the X-vs-E
question). Now the analysis stops and names the replicates and their missing arms. With
`--allow-incomplete-replicates` an incomplete replicate is analysed on its own (sections 1-5),
is excluded from every cross-replicate summary and figure, and contributes only to the
same-arm noise floor.

### Exposure index per data seed (finding 10)

The prepacked rows are the same for every data seed; the row ORDER, the step of every
occurrence and the rows repeated after the epoch boundary are not. The exposure index was
bound by `prepack_dataset_id` only, so the last seed's index (the one left in `exposure/`, and
on the Hub's `main`) was accepted for every replicate. `bind_exposure` now binds a replicate to
the index whose `PREFLIGHT_OK.json` is the file its checkpoints recorded
(`bindings.preflight_sha256`; status `exact`), else to an index with the same dataset id AND
the same recorded data seed (`equivalent`, with a warning: a regenerated preflight), else --
for metas without those fields -- by dataset id as before (`dataset_id`); an index for another
dataset or another data seed is a violation. `--exposure-dir` may hold one index per seed;
`run_remaining_full_seeds.sh` keeps them under `<build>/exposure_by_seed/seed_<S>/` (pod only:
the Hub build folder has only the last seed's index on `main`, earlier seeds by commit). The
arms of a replicate must resolve to one index; the `compatibility` table shows the status.

### Cross-replicate inference (finding 1) -- section 6

* 6a `replicates__paired`: `per_replicate` lists every replicate's within-run mean;
  `t95_lo/hi` is the t interval of their mean with S-1 df (`ctx.stats.replicate_interval`).
  The training run is the unit; the interval is for THIS fact set (the replicates share their
  facts).
* 6b `replicates__crossed`: per contrast the semantic-groups x replicates matrix of paired
  differences. `facts_ci95_*` = cluster bootstrap of the replicate-averaged differences
  (replicates fixed). `var_unit`, `var_replicate`, `var_residual` = ANOVA components
  (truncated at 0); `se_crossed` = sqrt(var_unit/U + var_replicate/S + var_residual/(U S)),
  `crossed_ci95_*` its Satterthwaite t interval (`ctx.stats.crossed_summary`). `p_rep` (F =
  MS_replicate / MS_residual) asks whether the replicate means differ by more than fact-level
  noise explains. On seeds 1-2: R - C neutral 9.04 vs 10.21, `p_rep` 6e-14 (the within-run
  interval [8.57, 9.48] of seed 1 excludes seed 2's value); X - C 0.68 vs 0.64, `p_rep` 0.63;
  X - E 0.02 vs -0.12, `p_rep` 0.14.
* How to read the three intervals. Simulated coverage of a nominal 95% interval for the
  population mean (facts and runs redrawn; 200 units; run-level SD of 0 / 0.1 / 0.27 unit SDs):
  crossed 0.97 / 0.94 / 0.89 with S = 2 and 0.97 / 0.94 / 0.93 with S = 3; replicate t interval
  0.90 / 0.94 / 0.95 and 0.84 / 0.90 / 0.94. So where `p_rep` is small the replicate interval
  is the appropriate one; where it is not, the facts / crossed interval is, and the replicate
  interval (which leaves out the sampling of facts) under-covers. Neither is a substitute for
  more seeds: with S = 3 only effects several run-level SDs from zero are settled.
* The bootstrap stream of a per-replicate interval contains the replicate name, so the same
  CSVs give slightly different interval digits as `rep0` and as `seed_1`.

### Noise floor (finding 2) -- section 6c, `replicates__same_arm_noise`

`mean_abs_diff` and `n_pref_flips` between two arms were reported without a reference. The
table compares the SAME arm between two training runs (`same_arm_across_replicates`) with two
arms of one run (`different_arms_same_replicate`). On seeds 1-2, neutral cloze, injected facts:
C vs C across seeds mean |d| 1.95-2.06 and 334-367 of 1750 preference flips; X vs E within a
seed 1.87-1.88 and 331-343. The single-seed statement that "X and E differ on individual facts"
therefore describes re-training noise; so does the CounterFact reversal count. The matched
init / data seeds of a replicate do not lower this floor. Only mean differences carry the arm
effect. An incomplete replicate contributes the arms it has.

### Heterogeneity (findings 4 and 9)

* 2b `by_relation__...`: every contrast within each relation (headline cohort, neutral
  cloze), with `n_values` = distinct injected values (few = closed vocabulary). On seeds 1-2
  R - C runs from +2.6 (animal class, 6 values) to +16 (landmark city, unique invented names)
  and X - C / E - C from 0.0 to +2.1: the pooled contrast mixes scales.
* 4c `verbatim_items__...`: 723 of the 3500 neutral-cloze items of the funded build are
  claim sentences arm R was trained on verbatim (template 0 or 1). R - C is 15.5 / 16.5 on
  those and 7.3 / 8.6 on the other items (9.0 / 10.2 pooled); X - C and E - C move by at most
  0.04. The subsets are stored as passes `cloze_neutral@verbatim_trained_items` and
  `cloze_neutral@not_verbatim_items` (column `subset`) in the cross-replicate tables.

### Supplementary probes -- section 5b

`eval_beliefs.py --supplement` scores two more passes on an existing checkpoint into
`supp_beliefs_arm<A>.csv` (its own meta: `kind: supplement`, `probe_construction:
supplement_v1`). The main evaluation is neither re-scored nor changed, so the experiment-wide
contract of the main CSVs (one `probe_construction`, one item set) still holds.

* `wrong_subject`: the fact's injected value against its rival in the neutral cloze of other
  subjects of the same relation, drawn from wholly held-out groups (never trained in any arm;
  2 donors x every cloze template per fact, chosen by a generator keyed with the build seed and
  the fact id, so identical in every arm, replicate and re-run; `donor_fact_id` column). A
  shift there cannot be a belief about the fact's subject: the injected value string was seen
  `dose` times in every intervention arm and the rival never, in any context. Section 5b
  reports the pass, `subject_specific` = neutral - wrong_subject per fact, the decomposition
  `supp_decomposition__<cohort>` (neutral shift = wrong-subject part + subject-specific part,
  by dose, labels pooled) and both parts within each relation.
* `wrapper_conditional_E`: the fact's occurrence-0 arm-E training frame cut at the value,
  scored in every arm; the counterpart of `wrapper_conditional` (arm X's wrapper in every arm).
  `supp_own_frame__<cohort>` compares the two wrapped arms each under its own trained context:
  (X - C under the X wrapper) - (E - C under the E frame) per fact. The E frames of a
  token-matched build can only be reproduced with the training tokenizer (1112 of 1750 match
  without it on the funded build, 1750 of 1750 with it), so the evaluator verifies every frame
  against the manifest's arm-E occurrence-0 `insert_text` and stops on a mismatch
  (`--manifest`, `--skip-manifest-check` records an unverified run).

Contract (`check_supplement`, saved under `supplement` in `compatibility.json`): in a replicate
that has any supplementary file every arm has one; each scored the same checkpoint as the main
evaluation of its arm (`model_sha256`, else `init_param_sha256` + step), in the same mode, with
the same tokenizer and probe files; one supplement version and one identity per scored item
across all files; unverified arm-E frames of a model-inference run are a warning. Violations
stop the analysis; `--no-supplement` analyses the main evaluations only. A replicate without
supplementary files simply has no section 5b. The supplementary passes enter the
cross-replicate tables as passes `wrong_subject`, `subject_specific`, `wrapper_conditional_E`
and `own_frame`. Archive: same Hub branch `eval/wandb-<run_id>` as the main evaluation, tag
`wandb-<run_id>-eval-supp`; wandb summary keys `belief_supp/*`.

Scoring every finished checkpoint on the training box (after the queue has finished; the
script takes the queue's lock and refuses while training runs):

```
bash contextualization/run_supplement_eval.sh              # seeds 1 2 3, arms C R X E
```

### Not changed (training-side findings, code frozen while the queue runs)

Resume does not compare learning-rate / architecture flags with the checkpoint
(`scripts/base_train.py: SCHEDULE_KEYS`); the preflight's `loss_weighting` block still labels
the objective `per_microbatch_mean` under `fixed_nominal` (`ctx/prepack.py`); the prepacked
loader reuses one pinned host buffer under a non-blocking copy (`nanochat/dataloader.py`; the
per-step supervised-token counts logged to wandb match the scheduled cross-arm differences at
the logged steps of the nine finished runs, so no systematic skip, but the exact check needs
the per-step logs on the training box); arm E's source-cue detector cannot see its
"A note on <year> ... adds that" frames (`ctx/templates.py`). These are for after the queue.

