# Corrected pipeline contracts (protocol v2, schema version 2)

Written 2026-10-14 as the shared contract between the builder, the exposure tools, the
evaluator and the analysis. Everything here is implemented under `contextualization/`
unless a path says otherwise. Historical artifacts (July 2026 run) follow the *legacy*
formats described at the end; nothing here regenerates or overwrites them.

## Versions (`ctx/protocol.py`)

| constant | value | meaning |
|---|---|---|
| `PROTOCOL_V1` | `v1-legacy-2026-07` | historical generator + rendering (frozen; `ctx.factpool.build_fact_pool` etc.) |
| `PROTOCOL_V2` | `v2-corrected-2026-10` | corrected generator (`ctx/factpool_v2.py`) + rendering policies |
| `BUILDER_VERSION` | 2 | operational builder (staging, overwrite refusal, cache keys, parity fix) |
| `PROBE_SCHEMA` | 2 | probe jsonl records |
| `MANIFEST_SCHEMA` | 2 | `manifest.csv` |
| `EVAL_SCHEMA` | 2 | `eval/beliefs_arm<A>.csv` + `.meta.json` |
| `PREPACK_SCHEMA` | 1 | `arm_<A>/prepacked/` |
| `EXPOSURE_SCHEMA` | 1 | `exposure/<arm>/occurrence_exposure.csv` |

`build_summary.json` (v2) carries `match_tokenizer` (`{src, loaded_via, vocab, token_limit_mode}` or null),
`design_cells` (`{grid, cells{stratum: {dose: n}}, label_counts, empty_cells}`, zero cells included),
`xe_diagnostics.limits_version` (`templates.XE_MATCH_LIMITS_VERSION`) and the embedded `novelty_audit`
(`build_id`, `scanned_shards`, `hits`) from which `--probes-only` can restore labels; `PROBES_META.json` of a
regenerated probe set records `verified_against_probe_files` (True / None), `probe_files_compared` and
`novelty_evidence`. It also carries `protocol`, `builder_version`, `schema_versions`, `code`
(`git_commit`, `git_dirty`, `git_branch`), `config` (full CLI echo incl. render policy),
`input_hashes` (base shard sha256 per index, fineweb revision/commit, tokenizer fingerprint
when copied), `shard_hashes` (per arm: filename -> sha256), `topup` (shared stream doc ids and
per-arm counts), `arm_total_chars`, `char_spread_frac`, `counts` (groups / pairs / candidate
statements / injected statements / held-out groups / occurrences).

Legacy summaries have no `protocol` key: `ctx.protocol.protocol_of(summary)` returns
`PROTOCOL_V1` and lists the assumed render defaults (`register=True`,
`verbatim_control=False`, `source_per_fact=False`); a legacy summary without `config.seed`
raises `ProtocolError`.

## Probe records (`probe_sets/injected_facts.jsonl`, `probe_sets/heldout_facts.jsonl`)

v2 record (one json object per line):

```
schema_version      2
protocol            "v2-corrected-2026-10"
fact_id             stable id of this candidate statement
group_id            stable id of the semantic group (relation, subject); == pair_id in v2
pair_id             same as group_id (kept for v1 readers)
rival_fact_id       fact_id of the other candidate of the group
candidate_slot      "a" | "b"
relation, domain, subject, subject_kind, value, competing_value
entity_tier         "real" | "novel"
ground_truth        "real_world" | "unverified_synthetic"
truth_value         "true" | "false" (real_world) | "unknown" (unverified_synthetic)
legacy_truth_label  null in v2 records
functional_relation true | false   (false relations are excluded from the default v2 pool)
novelty             "real_entity" | "unverified" | "audited_absent" | "audited_present"
split               "train_group" | "heldout_group"
role                "injected" | "unasserted_rival" | "heldout_group" 
assigned_frequency  int (injected) ; 0 otherwise
rival_assigned_frequency  dose of the group's injected member (unasserted_rival) ; 0 otherwise
arms                ["R","X"] (+"E") for injected ; [] otherwise
surface_forms       list[str]  (frame order == relation.frames order)
cloze_templates     list[str]  (template_idx i -> template_id "<relation>/cloze/<i>")
heldout_paraphrase  str | null (injected only) ; heldout_paraphrase_index int | null
fixed_source        str | null (injected, source policy per_fact)
```

`heldout_facts.jsonl` v2 contains the `unasserted_rival` records of every training group AND
both candidates of every held-out group, plus CounterFact records
(`source: "counterfact"`, `fact_id: "counterfact_<idx:05d>"`, `cloze_templates`,
`true_value`, `false_value`, `truth_value: "true"`, `schema_version: 2`).

Legacy (v1) records are as written by the July builder (`fact_id, claim_text, truth_value,
domain, entity_tier, relation, subject, value, competing_value, assigned_frequency, arms,
surface_forms, cloze_templates, heldout_paraphrase, fixed_source`). Readers derive
`group_id = stable_id(relation, subject)` for v1 records (`ctx.cohorts.group_id_of`).

## `manifest.csv` v2 (one row per occurrence x arm)

```
schema_version, fact_id, group_id, arm, occurrence, assigned_frequency,
doc_index            position of the document in the arm's train stream (0-based; base docs and
                     slot docs share the index space 0..T-1; top-up docs are >= T)
shard_file, row_in_shard
insert_text          the full inserted sentence (with any register lead-in)
insert_char_start, insert_char_end        span of insert_text inside the document
claim_char_start, claim_char_end          the claim clause (R: the surface form; X/E: the clause
                                          or quoted sf inside the wrapper), inside the document
subject_char_start, subject_char_end      subject inside the document (first occurrence in claim)
value_char_start, value_char_end          first value slot inside the document
value_spans          json list of [start,end] for EVERY value slot (repeated slots)
source_char_start, source_char_end        X only (-1 otherwise)
paraphrase_index, wrapper_index, register_index, tail_index, source_id
truth_value, domain, entity_tier, rendering_id (= "<fact_id>:<occ>"), embedded, heldout
carrier_chars (length of the cropped carrier), carrier_cropped (bool)
```

Spans are character offsets into the document string as written to the shard. Legacy
manifests lack everything after `assigned_frequency`; the exposure tools locate legacy inserts
by regenerating `insert_text` with the v1 renderer and requiring exactly one match in the doc.
`paraphrase_index` is the index of the relation FRAME that rendered the claim; it equals the index
into the record's `surface_forms` (v2 renders every frame in order and refuses duplicates; the v1
pool only deduplicated identical renderings, which never occurred for the defined frames).
`wrapper_index` is the X wrapper / E frame index of the template family named in the build's
`render_policy` (`x_templates`/`e_templates`: legacy or v2); for arm C it is the canned-insert
index (-1 for filler sentences). Under `RenderPolicy.paraphrase == "fixed"` the fixed frame is
drawn per fact from the stream `render.v2.paraphrase/fixed/<fact_id>` (there is no global fixed
index); readers must take it from the manifest.

## Exposure index v1 (`exposure/<arm>/occurrence_exposure.csv`)

```
fact_id, group_id, arm, occurrence, doc_index,
row_index, row_offset_start, row_offset_end        (token positions of the doc in its packed row)
insert_tok_start, insert_tok_end, claim_tok_start, claim_tok_end,
subject_tok_start, subject_tok_end, value_tok_start, value_tok_end     (row positions; -1 if lost)
status            complete | partial_value_lost | partial_subject_lost | partial_claim_cut |
                  omitted (doc never placed within the horizon) | not_in_horizon
scored_claim_tokens, scored_value_tokens
epoch, micro_batch, step   (-1 when not within the training horizon)
```

Summary json per arm: counts by status, per-fact `effective_exposures` (complete occurrences
with step < num_iterations), distributions by dose, scored-token totals, and (prepacked path)
cross-arm agreement: every occurrence has the same `row_index`/`step` in every arm.

## Prepacked rows (`arm_<A>/prepacked/`)

`prepack_manifest.json`: `schema_version: 1`, `T`, `n_rows`, `pad_token_id`, `tokenizer_sha256`,
`build_summary_sha256`, `protocol`, `arms_planned_together`, `row_files` (file, n_rows, sha256),
`scored_tokens_total`, `masked_tokens_total`, `intervention_tokens_scored`, `dropped_topup_docs`.
`rows_<i:05d>.npy`: int32 `[n, T+1]`; `mask_<i:05d>.npy`: uint8 `[n, T+1]`; `mask[r, t] == 1`
means `rows[r, t]` is a real token whose prediction is scored (`t >= 1`). Inputs are `rows[:, :-1]`,
targets are `rows[:, 1:]` with `-1` wherever `mask[:, 1:] == 0`.

Rows are planned once across arms (same docs, same positions); an intervention document is
reserved at the max token length over arms and never cropped inside its claim span; arms that
are shorter are padded with masked `pad_token_id`.

Optional oversized-slot handling (2026-10-17): `oversized_slot_policy` is `error` (default) or
`trim_shared_prefix`. The opt-in policy removes an identical carrier prefix before every full
insert in all arms, shifts local copies of character spans and re-tokenizes with BOS. It never
changes the source Parquet or manifest. Full inserts, including attribution/wrappers, must still fit.
The common `slot_prefix_trims` list records each affected document and its transformation;
`slot_prefix_trims_sha256` hashes that list, and `packing.slot_prefix_trimmed_docs` counts it.
Records bind source/retained text hashes, removed prefix and character offset, before/after token
counts, and the deterministic boundary selection policy. The prepack gate replays these records
from source and compares actual placed tokens and exposure spans. Older manifests without these
fields mean strict/error with zero trims. Trimming is a reported change to preceding context and
carrier-token exposure; its cross-arm matching does not make it scientifically invisible.

Bindings added after the second review (2026-10-14; details in `EXPOSURE.md`): the manifest
records `encoding_fingerprint` (canonical hash of the tokenizer object that produced the rows),
`tokenizer_file_sha256` (the arm's `tokenizer.pkl`, which must encode identically), per row file
`sha256` / `mask_sha256` / `active_targets`, `active_targets_total`, `active_targets_per_row`
(min/mean/max + 32-token histogram), `prepack_dataset_id` (sha256 over T, n_rows and the ordered
file hashes), `prepack_run_id`, and the occurrence bookkeeping (`role`, `required_occurrences`,
`control_slots`). The loader state carries `prepack_dataset_id`; resume refuses a different id.
`exposure_audit.py --mode prepacked` is the preflight: it writes `<exp_dir>/exposure/PREFLIGHT_OK.json`
(config, per-arm dataset ids, encoding fingerprint, achieved exposure, loss-weighting report) only
when every occurrence is complete, effective and scheduled identically in every arm;
`scripts/base_train.py --data-loader prepacked` refuses to start without a matching preflight. Row order per epoch: identity when
`data_seed < 0`, else `numpy.random.default_rng(data_seed * 1000003 + epoch).permutation(n_rows)`.
Loader state: `{"loader": "prepacked", "schema_version": 1, "epoch", "cursor", "data_seed",
"world_size", "B"}` where `cursor` is the global row position (within the permuted epoch) of the
batch *being yielded*; resuming from a state replays that batch first.

## Eval CSV v2 (`eval/beliefs_arm<A>.csv`) and `eval/beliefs_arm<A>.meta.json`

Columns (legacy columns first, same meaning as before):
```
arm, probe_set, fact_id, pass, template_idx, truth_value, entity_tier, assigned_frequency, is_frame0,
logp_value, logp_competing, n_tok_value, n_tok_competing,
schema_version, group_id, role, ground_truth, template_id, metric,
logp_value_answer, logp_competing_answer, n_tok_value_answer, n_tok_competing_answer,
logp_value_suffix, logp_competing_suffix,
boundary_k, n_ctx_tokens, boundary_adjusted, answer_suffix_merged_value, answer_suffix_merged_competing,
answer_lead_merged_value, answer_lead_merged_competing,
value_span_chars, context, cand_value, cand_competing, scoring_policy,
legacy_malformed, historical_rendering, item_sha256
```
`metric` is `answer_plus_suffix` for cloze passes (so `logp_value` keeps its historical
definition), `full_sentence` for `heldout_paraphrase`. `*_answer` columns are the answer-only
sub-sums; `*_suffix = * - *_answer`. `boundary_k` is the number of shared conditioning tokens
(incl. BOS) chosen jointly for both candidates; `boundary_adjusted = 1` when it is smaller than
the context's own tokenization length. `item_sha256` is the identity of the scored contrast
(`ctx.protocol.scored_item_identity`: sha256 over context, both candidates, `value_span_chars`,
`metric`, `template_id`, `scoring_policy`, `boundary_k`, `n_ctx_tokens`); the analysis
recomputes it from the columns. The meta json records protocol, probe file hashes, tokenizer
fingerprint, checkpoint path/step/sha256, code revision, scoring policy, item counts per
pass, expected coverage and `exclusions` (fact_id, pass, reason), plus (recheck 2026-10-14):

| meta field | content |
|---|---|
| `probe_construction` | `ctx.probes.PROBE_CONSTRUCTION_VERSION` (`structured_v2.1`; `structured_v2.0 (unversioned constructor)` for the 031c4e5 module): the candidate-construction semantics, compared across every evaluation of an analysis |
| `items_sha256` | sha256 over the sorted `probe_set/pass/fact_id/template_idx:item_sha256` lines: identity of the whole scored item set |
| `model_tag` | the `--model-tag` the checkpoint was loaded from (`d<depth>_s<S>` for replicates) |
| `run` | `{wandb_run_id, source, checkpoint_run_id, run_info_run_id, run_info_path[, conflict]}`: the training run the evaluation belongs to, taken from the checkpoint meta's `wandb_run` first, then the model tag's `run_info` file, `--wandb-run-id` overriding both |
| `checkpoint` | `step, model_config, bindings, init_seed, cli_init_seed, init_param_sha256, resume_from_step, preflight, data_seed, data_loader, model_tag, total_batch_size, num_iterations, max_seq_len, device_batch_size, grad_accum_steps, world_size, wandb_run, model_path, model_sha256` (legacy metas leave the unrecorded ones `null`) |

Output layout: `eval/beliefs_arm<A>.csv` for the default model tag `d<depth>`,
`eval/seed_<S>/beliefs_arm<A>.csv` for a replicate tag `d<depth>_s<S>`, `eval/<tag>/...` for
any other tag (`ctx.tracking.eval_subdir_of`); one evaluation per checkpoint, never overwritten.
Checkpoint metas (`meta_<step>.json`) additionally record the resolved schedule
`num_iterations`, `grad_accum_steps`, `world_size` (next to `device_batch_size`, `max_seq_len`,
`total_batch_size`); a resume must keep all six (recheck finding 4). They also record the training
objective `loss_normalization` (`microbatch_mean` | `fixed_nominal`; an ABSENT key means the
historical `microbatch_mean`) and `loss_denominator_per_microbatch` (`B x T x grad_accum`, null for
the historical policy); `PREFLIGHT_OK.json` records `loss_normalization` and `loss_denominators`
(`per_microbatch_nominal`, `per_step_all_ranks_nominal`), the eval meta `checkpoint.loss_normalization`
(null = unrecorded = historical). Run identities live in `arm_<A>/run_info.json` (default tag) or
`arm_<A>/run_info_<tag>.json` (replicate / custom tag); run_info records `loss_normalization` too.

## Legacy eval CSV

`arm,probe_set,fact_id,pass,template_idx,truth_value,entity_tier,assigned_frequency,is_frame0,logp_value,logp_competing,n_tok_value,n_tok_competing`
(`eval_results.tgz`, sha256 ee5dad66...). Readers treat missing v2 columns as absent.
