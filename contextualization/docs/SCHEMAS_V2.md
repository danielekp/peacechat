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

`build_summary.json` (v2) carries `protocol`, `builder_version`, `schema_versions`, `code`
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
are shorter are padded with masked `pad_token_id`. Row order per epoch: identity when
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
value_span_chars, context, cand_value, cand_competing, scoring_policy
```
`metric` is `answer_plus_suffix` for cloze passes (so `logp_value` keeps its historical
definition), `full_sentence` for `heldout_paraphrase`. `*_answer` columns are the answer-only
sub-sums; `*_suffix = * - *_answer`. `boundary_k` is the number of shared conditioning tokens
(incl. BOS) chosen jointly for both candidates; `boundary_adjusted = 1` when it is smaller than
the context's own tokenization length. The meta json records protocol, probe file hashes,
tokenizer fingerprint, checkpoint path/step/sha256, code revision, scoring policy, item
counts per pass, expected coverage and `exclusions` (fact_id, pass, reason).

## Legacy eval CSV

`arm,probe_set,fact_id,pass,template_idx,truth_value,entity_tier,assigned_frequency,is_frame0,logp_value,logp_competing,n_tok_value,n_tok_competing`
(`eval_results.tgz`, sha256 ee5dad66...). Readers treat missing v2 columns as absent.
