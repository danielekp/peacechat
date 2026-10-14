# Protocol v2: the corrected C/R/X/E experiment

Status 2026-10-14. This document specifies what the corrected pipeline generates, renders,
measures and guarantees, where the historical protocol (v1, July 2026 run) differs, and which
scientific limitations code alone cannot resolve. Schema details are in `SCHEMAS_V2.md`;
operational safeguards in `ARTIFACTS.md`; exposure accounting in `EXPOSURE.md`; analysis in
`ANALYSIS.md`; historical errata in `ERRATA.md`.

## 1. Versioning and historical compatibility

* `ctx/protocol.py` defines `PROTOCOL_V1 = v1-legacy-2026-07` and `PROTOCOL_V2 =
  v2-corrected-2026-10`, plus schema versions for probes, manifest, eval CSV, prepacked rows and
  exposure indices. Every artifact written by the corrected pipeline carries them, together with
  the git commit (+dirty flag), full configuration, input hashes (base shards, tokenizer), and
  shard hashes.
* The historical generator (`ctx.factpool.build_fact_pool/split_pool/assign_frequencies`) and
  the historical rendering (`ctx.templates.build_inserts/render_occurrence` with the legacy
  wrapper/E-frame/tail/verb lists) are frozen. `tests/fixtures/historical/` holds digests and
  867 rendered strings captured from the pre-refactor code at commit `6090883`; the test-suite
  asserts byte identity (`test_templates.py`, `test_factpool_v2.py::test_legacy_generator_is_frozen`).
* Legacy builds are interpreted, never regenerated: `ctx.protocol.render_config_from_summary`
  reads a legacy `build_summary.json` as protocol v1, requires `config.seed`, and reports the
  historical defaults it assumes for missing booleans. A summary that cannot support the
  reconstruction raises `ProtocolError` instead of guessing.
* `--probes-only` verifies the pool arguments and the regenerated `(fact_id -> dose)` map against
  the original build before writing, and writes to `probe_sets_regen/` unless `--overwrite-probes`
  is given. It never rewrites `build_summary.json` or the manifest.
* The auditor's reproduction scripts keep working against the historical protocol:
  `ctx.factpool.*` names/signatures, `ctx.templates.REGISTER_LEADINS`, `ctx.blend.build` with a
  legacy namespace (interpreted as v1), `eval_beliefs.build_items/_FactView` (legacy-shaped
  items) and `analyze_beliefs.wilcoxon_signed_rank` (now scipy-backed; the historical helper is
  `legacy_wilcoxon_signed_rank`). The last block of `reproduce_dataset_findings.py` (cache
  aliasing) can no longer reproduce the bug: the keyed cache would attempt a real FineWeb download.

## 2. Fact pool and split design (findings 1, 2, 14, 17)

`ctx/factpool_v2.py`:

| concept | definition |
|---|---|
| semantic group | one `(relation, subject)`; subjects are unique across the whole pool (case-insensitive) and name-like values are drawn from the same reserved set, so a city is never also a country |
| pair | == group; exactly two candidate statements (`candidate_slot` a/b) created atomically, each with `rival_fact_id` / `competing_value` pointing at the other |
| candidate statement | one `(subject, value)` proposition with its 6 surface frames and 2 cloze templates |
| injected statement | the single candidate of a training group chosen for injection (primary design) |
| counts | `PoolCounts`: groups = pairs, candidate statements = 2 x groups, training groups = injected statements, held-out groups, unasserted rivals, occurrences = sum of doses |

* Only functional relations are used by default; `author_work` and `inventor_of` (several answers
  can be valid) need `--include-nonfunctional`.
* **Ground truth vs assignment.** Real-entity groups keep `ground_truth=real_world` and a
  grounded true/false candidate. Synthetic groups have `ground_truth=unverified_synthetic` and
  `truth_value=unknown`: their values are targets/distractors, not true or false statements. v1
  records keep their historical `truth_value` (incl. `contested`) and get `legacy_truth_label`
  semantics in the analysis; the old "contested" condition is not reinterpreted as source
  disagreement (it was an unintended third value, see `ERRATA.md`).
* **Split.** Whole groups go to training or held-out, stratified by relation x tier. The held-out
  probe file therefore distinguishes `role=unasserted_rival` (the rival of an injected statement;
  its group IS trained) from `role=heldout_group` (wholly unseen semantic group).
* **Primary design (`--design one_sided`).** Exactly one randomized candidate per training group
  is injected; its rival is never asserted. Candidate slot (and, for real groups, the grounded
  label) and dose are balanced within `(relation, tier[, label])` strata. Injecting both pair
  members is NOT a fix (every statement would have a trained rival); the deliberate
  competing-evidence experiment is `--design competing --rival-freq-grid ...` and is recorded as
  such.
* **Capacity.** Requests exceeding the unique-name capacity fail (`PoolError`) rather than
  returning an incomplete or off-by-one pool.
* **Novelty.** Synthetic names are not assumed absent from FineWeb. `--collision-audit` scans
  everything the model sees in arm C (base, carriers, top-up, val) for every novel subject/value
  and labels each group `audited_absent` / `audited_present`; without the audit the label is
  `unverified`. Known real names the generators can emit (Bromine, Xenon, Ferrate, Bromide, ...)
  are blocked. Lexical absence still does not prove semantic novelty.

## 3. Rendering controls (findings 5, 6)

`ctx.templates.RenderPolicy` decouples every control, each on its own named RNG stream:

| control | options | v2 default | v1 (legacy) |
|---|---|---|---|
| paraphrase | cycle_from_zero, seeded_start, random, fixed | seeded_start | cycle_from_zero |
| wrapper | same | seeded_start | cycle_from_zero |
| source | rotate, per_fact | rotate | rotate |
| stance (X verbs) | neutral, legacy | neutral (`wrote, stated, said, reported, noted, mentioned, remarked, recorded`) | legacy (incl. `speculated`, `insisted`, ...) |
| register lead-in | rotate, none, fixed | rotate | rotate |
| tails | none, legacy | none | legacy (incl. "puzzling") |
| neutral insert (C) | canned, filler_sentence | filler_sentence (a held-out FineWeb sentence) | canned (ten sentences) |
| x_templates / e_templates | v2, legacy | v2 | legacy |

* `seeded_start` removes the deterministic "every fact starts at wrapper/paraphrase 0" bias by
  counterbalancing the starting index per fact. This does **not** make rendering diversity
  independent of dose: a dose-1 fact still sees one wrapper, a dose-64 fact all 22. In
  variable-rendering modes the estimand is therefore "dose with the protocol's rendering
  diversity", not repetition alone.
* `--render-mode fixed` (`RenderPolicy.fixed_rendering()`) is the dose control with FIXED
  rendering: one paraphrase, one wrapper, one source (per fact), no lead-in, no tail, and the
  year / document type / verb drawn once per fact, so every occurrence of a fact is the same
  string in every arm. Any single factor can be re-enabled through `--policy-*` flags.
* **Arm X (v2 templates).** 22 wrappers in three grammatical styles; hearsay/credibility cues
  ("A reader pointed to", "Some cite", "widely shared") were replaced; verbs are neutral.
* **Arm E (v2 templates).** 22 own-voice frames index-aligned with the X wrappers; each keeps
  the claim as a `that`-clause (or after a colon for the quote-style wrappers), contains a year,
  no source entity, no document type, no attribution verb, and no permanence / consensus /
  credibility assertion. A discourse-adverbial `{filler}` is chosen per occurrence so that the
  E insert matches the X insert in character length and claim start. Measured criteria
  (`templates.XE_MATCH_LIMITS`, enforced by `validate.py` and reported in
  `build_summary.json["xe_diagnostics"]`): per-occurrence |delta chars| <= 12 and |delta claim
  start| <= 12, mean <= 4; token deltas <= 8 (mean <= 3) when a tokenizer is present; no source
  cue in any E frame. On the smoke builds the realized mean |delta chars| is ~1.7 (max 6).
  `xe_examples.txt` lists reviewable X/E pairs.
* **What E does and does not identify.** E and X still differ in named entities (none in E),
  document-type nouns, verbs vs discourse adverbials, and the lexical content of the filler.
  E is an informative comparator for "embedding/length/position without attribution"; it does
  NOT uniquely identify attribution semantics. Source-identity (`source=per_fact`) and stance
  (`stance=legacy`) are separate, explicitly configured controls.
* The optional arm is byte-invariant for the others: enabling E (or the collision audit)
  changes nothing in the C/R/X shards (`tests/test_build.py`).

## 4. Carrier selection and cropping

All arms share one held-out FineWeb carrier per slot; the builder crops it to whole leading
sentences within `--carrier-max-chars` (default 6000, identical crop for all arms) so a slot
document fits one training row of 2048 tokens, and inserts the arm's sentence at the same
sentence boundary. The only intentional cross-arm difference is the inserted sentence. The
manifest records `carrier_chars`, `carrier_cropped` and every span.

## 5. Evaluation (findings 8, 9, 13)

`ctx/probes.py` + `eval_beliefs.py`: both candidates come from the same structured template
with explicit answer spans; the historical cloze metric (`answer_plus_suffix`) is kept under its
old column names, answer-only and suffix sub-sums are added, the paraphrase pass is
`full_sentence`. The wrapper pass renders the exact occurrence-0 training text of the build's
protocol (v1 builds: byte-identical to July, lowercased title-initial values included) and cuts
it at the recorded value span, so the 63 historically dropped author/work probes are scored.
Token boundaries are chosen jointly (`boundary_k`), coverage is asserted, exclusions are listed,
every context/candidate/span/template id/component log-prob is saved, and outputs are never
overwritten. `logP(value)` and `logP(rival)` are stored separately; the ratio remains a
completion-likelihood contrast, not a probability of truth.

## 6. Training exposure, seeds, artifacts

See `EXPOSURE.md` (historical replay, prepacked matched-schedule loader, resume), and
`ARTIFACTS.md` (cache keys, checksums, overwrite refusal, bindings, export manifests). Seeds:
`--init-seed` (shared by all arms of a replicate) and `--data-seed` (row order of the prepacked
loader; shared within a replicate) are recorded with the non-deterministic settings; a seed does
not guarantee bitwise reproducibility under `torch.compile`/TF32/flash attention.

## 7. What code alone cannot resolve

1. **Truth.** Synthetic values are arbitrary associations; truthfulness claims need
   independently grounded facts (only the real-entity groups qualify, and they are few).
2. **Historical exposure.** The actual per-occurrence exposure of the July run can only be
   measured by replaying the historical shards with the historical loader (`exposure_audit.py
   --mode historical`); those shards are not local.
3. **Historical rescoring.** The corrected evaluator can rescore the C and R checkpoints (local
   copies exist); arm X's checkpoint is lost. A corrected 4-arm comparison requires retraining.
4. **Run-to-run variance.** One run per arm cannot estimate training variance; replicates with
   logged seeds are required before any equivalence statement (and an explicit margin).
5. **Provenance of v0 vs v1.** `RESULTS.md` says C trained on build v0 and R/X on v1; the local
   v0/v1 summaries show identical C totals and 646,023 fewer R/X characters in v0 because v0
   used a different held-out-paraphrase rule (see `ERRATA.md`); byte identity of C's text can
   only be established from the shards.
6. **E's estimand.** Even with matched length/position, E differs lexically from X; interpret E
   as "embedded own-voice claim", not as a pure attribution-minus control.
