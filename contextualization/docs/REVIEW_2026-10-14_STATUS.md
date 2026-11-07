# Second review (reviews/2026-10-14-code-review/REVIEW.md) and recheck (reviews/2026-10-14-recheck/REVIEW.md) -> status with evidence

Written 2026-10-14 after reproducing every finding with the review's own scripts at commit
`031c4e5` (all 21 reproduced; outputs matched the review's numbers) and then fixing them. Status
values: **fixed** (behaviour changed + regression test derived from the reproduction),
**deferred** (tooling in place, not exercised end to end here), **experimental decision** (the
code measures/records; the choice is the experimenter's and is NOT made here).
Test commands are listed at the end; every reproduction script was re-run after the fixes.

| # | finding | status | what changed | evidence |
|---|---|---|---|---|
| 1 | eval upload moved `wandb-<run_id>` to other weights | **fixed** | `ctx.tracking.hf_upload_eval`: verifies the tag's commit holds the evaluated checkpoint (LFS sha256 == `checkpoint.model_sha256` from the eval meta), uploads to branch `eval/wandb-<run_id>` created from that commit, tags `wandb-<run_id>-eval`; `hf_upload_dir` never deletes/recreates tags (explicit `TagConflict`); `hf_upload_files` rejects `wandb-*` tags; `eval_beliefs.py` and `load_artifact_hf.py --eval` use it | `tests/test_artifacts.py` (revision-aware FakeHub: A's eval leaves `wandb-A` at weights A, eval branch starts at the checkpoint commit, `TagConflict`/`CheckpointMismatch` before any upload, refused retag); `repro/review_root_v2.py` now stops with `ValueError` at the `wandb-A` tag request |
| 2 | migration deleted unuploaded versions / unrelated backup runs | **fixed** (one-off tool, mocks only) | numeric version order; every distinct version uploaded and verified or proven a byte-identical duplicate; deletions only for versions verified in the same invocation and only for backup runs of verified collections; `--keep-wandb` default, `--delete-wandb` opt-in | `tests/test_migration.py` (10 tests: numeric order, duplicates, verification before deletion, `--only`, failure => no deletion, existing-tag reruns, datasets, review-shaped runpy case); `repro/review_migration_v2.py` now deletes nothing in either scenario |
| 3 | prepack certified the wrong tokenizer | **fixed** | `encoding_fingerprint(tokenizer)` of the tokenizer actually used is recorded; arm `tokenizer.pkl` must agree (else `PrepackError`); loader/resume/base_train compare the fingerprint with the model's tokenizer | `tests/test_prepacked_loader.py` (prepack refuses an arm tokenizer whose fingerprint differs from the encoder; the loader refuses a different tokenizer at construction and on resume); `repro/review_tokenizer_binding.py` now exits 1 with `PrepackError ... refusing to certify rows for a tokenizer that did not produce them` |
| 4 | gate accepted missing occurrences, zeroed masks, altered rows on resume | **fixed** | `verify_prepack` requires the manifest's occurrence multiset per intervention arm, verifies per-file row/mask hashes and active-target counts, checks masks over every claim span; `prepack_dataset_id` bound into loader state, checkpoint meta and resume; staging + atomic move of all arms | `tests/test_prepacked_loader.py` (missing occurrence, duplicate key, zeroed mask, altered rows + old resume state, tampered manifest.csv all refused; dataset id in loader state and checkpoint bindings); `repro/review_v2_training.py` now stops at its first mutation with `PrepackError: ... factA:0 arm X: missing from the exposure index` |
| 5 | repeated answer slots gave malformed wrapper competitors | **fixed** | `ctx.probes.competing_record` + structural rendering of the competing wrapper under the same policy/seed/occurrence; every slot replaced, each candidate keeps its own capitalization rule; all value spans recorded; value-side strings unchanged for historical probes | `tests/test_eval.py::test_wrapper_competitor_replaces_every_repeated_slot` (0 of 16 malformed on the 600-group pool); `repro/review_eval_v2.py` |
| 6 | analysis accepted incompatible / mock evaluations | **fixed** | `.meta.json` validated per replicate (mode, tokenizer, probe hashes, scoring policy, csv sha, per-item metric/template identity) and across replicates (distinct checkpoints/seeds); `--compat legacy` for CSVs without meta, `--compat historical-builds --compat-reason` for intentionally different builds; `analysis_v2/compatibility.json` | `tests/test_analysis.py::test_compatibility_rejects_incompatible_metadata`, `::test_mock_evaluations_run_but_are_labelled`, `::test_compatibility_across_replicates`, `::test_historical_requires_compat_legacy`; `repro/review_analysis_pairing.py` now exits 1 with the incompatibility list |
| 7 | partial templates manufactured a paired effect | **fixed** | exact item joins on (probe_set, pass, fact, template); common template set per fact across arms used by every curve and contrast; partial/absent coverage recorded; v2 evaluations fail by default on coverage disagreement (`--allow-incomplete`) | `tests/test_analysis.py::test_partial_templates_are_missingness_not_effects` (default: coverage stop; `--allow-incomplete`: R-C = 0) |
| 8 | reciprocal held-out candidates cancel to zero | **fixed** | `heldout_groups` keeps ONE stable orientation per group (real: the true candidate; synthetic: slot a); flips/|delta| once per group; `heldout_groups_both` kept with a cancellation warning | `tests/test_analysis.py::test_heldout_orientation_prevents_cancellation`; `repro/review_heldout_cancel.py` with `--compat legacy`: unknown n=164 mean_diff -22.0, flips 164, `tost_equivalent=False` (the shipped script stops earlier because its CSVs carry no meta) |
| 9 | probes-only accepted a different source policy | **fixed** | `probes_only` recovers `config.render_policy` from the completed build, refuses conflicting `--policy-*`/`--render-mode`/`--source-per-fact`/`--embedding-control`/`--register`/`--verbatim-control` | `tests/test_build.py::test_probes_only_uses_the_original_render_policy`; `repro/ctx_data_review_repro.py` |
| 10 | overwrite left stale audit evidence | **fixed** | `--overwrite` removes every build-derived artifact (`BUILD_ARTIFACTS`); `novelty_audit.json` carries the build's `build_id` and is only honoured when the summary records an audit of the same build | `tests/test_build.py::test_overwrite_removes_derived_artifacts_and_foreign_audits_are_refused` |
| 11 | grounded labels unbalanced; remainders favour low doses | **fixed** | real strata choose the injected LABEL explicitly (rank parity) and derive the slot; dose grid rotated randomly per stratum; `design_cells()` reports every cell incl. zeros and label counts (`build_summary.json["design_cells"]`) | `tests/test_factpool_v2.py` (8 seeds: |true-false| <= 1; remainder doses spread; seed-4 element-symbol stratum has both labels) |
| 12 | fixed rendering did not fix the C insert | **fixed** | canned neutral insert drawn once per fact in fixed mode; filler-sentence inserts cached per fact by the builder | `tests/test_templates.py::test_fixed_rendering_fixes_the_control_insert_too`; `tests/test_build.py::test_fixed_mode_build_repeats_one_insert_per_fact_in_every_arm` |
| 13 | overrides ignored in fixed mode | **fixed** | fixed defaults then explicit `--policy-*` overrides; effective policy recorded as before | `tests/test_templates.py::test_fixed_mode_honours_explicit_overrides` |
| 14 | competing design reported the wrong rival dose | **fixed** | `rival_assigned_frequency` resolved through the group's other candidate (== `rival_fact_id`'s dose), reciprocal by construction | `tests/test_factpool_v2.py::test_competing_design_reports_the_other_candidates_dose` |
| 15 | character matching did not ensure token matching | **fixed** | `render_inserts(..., tokenizer=)` chooses the E filler under joint token+character limits with the actual tokenizer; the builder loads it from `--tokenizer-src` (tiktoken dependency added) and `xe_diagnostics` records `token_limits_verified`; `within_limits` is only true when tokens were checked; `validate.py` FAILS a v2+E build when no tokenizer is available. Limits re-expressed with tokens primary (chars secondary) | full 2,500-group seed-1234 pool with the July tokenizer: |dtok| max 7 / mean 0.58, |dchars| max 12 / mean 4.6, claim-start tokens max 6; `tests/test_templates.py::test_token_aware_filler_meets_token_limits_on_the_reviewers_case` |
| 16 | exposure audit ignored step disagreement | **fixed** | `--mode prepacked` fails unless rows AND steps AND statuses agree in every arm and manifests share the seed; writes `exposure/PREFLIGHT_OK.json` only on success; stale preflight removed first | `tests/test_exposure.py` (cross_arm step agreement; monkeypatched gate exit 2; `PREFLIGHT_OK.json` only on success); the reproduction's per-arm seeds R=1/X=2/E=3 make `exposure_audit --mode prepacked` exit 2 (manifest data_seed disagreement is caught first, step disagreement second) |
| 17 | fixed-form cohorts reconstructed form 0 | **fixed** | `realized_training_forms` reconstructs the renderer's own choice (`templates._pick` with the build seed) for fixed/seeded/random policies; no seed -> `ProtocolError`; manifest still preferred | `tests/test_analysis.py::test_fixed_form_reconstruction_matches_renderer`; `repro/review_eval_v2.py`: 0/419 wrong forms, 0/419 wrong subgroups |
| 18 | fact-level Wilcoxon under repeated groups | **fixed** | group-level Wilcoxon when any group holds >1 fact (`p_wilcoxon_level`), fact-level only when groups are singletons; one-group cells report no p | `tests/test_analysis.py::test_paired_cell_group_level_wilcoxon` (`paired_cell([1.]*20, ['same-group']*20)` -> one group, p None) |
| 19 | eval attested hashes of files it did not open | **fixed** | hashes and paths of the files actually opened (`--probe-dir` honoured); checkpoint `bindings` (build summary / probe hashes) compared with the scored probes, `--allow-build-mismatch` for intentional historical rescoring, mismatch recorded in the meta | `tests/test_eval.py` (dry-run meta hashes equal the opened files) |
| 20 | merge diagnostics reported no merges | **fixed** | byte-exact answer/suffix (and lead) merge flags for BOTH candidates (`answer_suffix_merged_*`, `answer_lead_merged_*`) | `tests/test_eval.py::test_answer_suffix_merge_flags_both_candidates` ("insect"+"s." flagged); >=500 flagged sequences on the 600-group fixture with the July tokenizer |
| 21 | prepacked loader could exhaust file descriptors | **fixed** | bounded LRU of open memmaps (default 32 chunk pairs), lazy open, rlimit check | `tests/test_prepacked_loader.py` (520 chunk pairs under a 1,024 soft limit, all rows iterated with <= 32 open pairs); `repro/review_training_extra.py` now prints `MMAP_FD_LIMIT_RESULT loaded` |

## Experimental decisions (measured, not decided here)

* **Loss normalization: DECIDED 2026-10-15 (fixed nominal).** The experimenter chose option (c):
  `--loss-normalization fixed_nominal` (sum of per-token losses over valid targets / `B x T x
  grad_accum` per micro-batch, one accumulation division, no extra world-size division, sum
  computed directly so a fully masked micro-batch is 0), implemented in
  `nanochat/loss_normalization.py` + `scripts/base_train.py` with the historical `microbatch_mean`
  as the default. The policy is recorded in every checkpoint meta, certified by the preflight
  (`exposure_audit.py --mode prepacked --loss-normalization`, refused by `run_arm`/`base_train` on
  mismatch), bound on resume (a change is refused; unrecorded = historical, never the new policy)
  and enforced by the analysis (one objective per experiment). Both the nominal objective and the
  loss per active token are logged per step across all ranks. Tests:
  `tests/test_loss_normalization.py`, `tests/test_seeds.py::test_tiny_cpu_fixed_nominal_*`,
  `tests/test_prepacked_loader.py::test_preflight_certifies_the_loss_normalization_policy`,
  `tests/test_analysis.py::test_training_objective_is_one_per_experiment`; the two-seed smoke
  trains under `fixed_nominal`. **Every funded run command must pass `--loss-normalization
  fixed_nominal` explicitly** (README / CLAUDE.md pipeline). The measurement text in
  `docs/EXPOSURE.md` section 10 is kept as the rationale.
* **FineWeb pinning.** The builder now resolves and records the commit behind an unpinned ref and
  warns; `--require-pinned-source` refuses moving refs; `--verify-cache` re-hashes cached shards.
  Choosing the commit is the experimenter's decision; the funded build must pass
  `--fineweb-revision <commit> --require-pinned-source --verify-cache`.
* **Opt-in loader (fixed).** `run_arm.py` requires an explicit `--data-loader` for protocol-v2 builds
  and a `PREFLIGHT_OK.json` that names the arm's CURRENT `prepack_dataset_id` for `prepacked`;
  `base_train` re-checks B, T, grad-accum, horizon, world size, data seed, dataset id and encoding
  fingerprint against the preflight and refuses otherwise (unless `--skip-preflight-check`, recorded
  as `preflight: "skipped"` in meta). `smoke_crxe.py` exercises the refusal of a stale preflight.
* **Resume provenance (fixed).** `init_seed` is carried from the resumed checkpoint; the CLI seed is
  recorded separately (`cli_init_seed`, `resume_from_step`); `init_param_sha256` (sha256 of the
  parameters right after `init_weights()`) identifies the initialization across arms and resumes;
  `run_arm.py` compares it across arms with the same depth and seed. `tests/test_seeds.py` runs real
  tiny CPU trainings: equal seeds -> equal hashes, resume keeps the original seed/hash and loader
  states, a different `--data-seed` or loader on resume is refused.
* **Replicate destinations (check 3).** `--init-seed` gives model tag `d<depth>_s<seed>`: its own
  `base_checkpoints/<tag>`, wandb run `ctx_<A>_<tag>` and HF repo `ctx_<A>_<tag>_base`; the export and
  upload helpers accept the tag (`tests/test_artifacts.py::test_replicate_model_tags_...`).
* **Paraphrase cohorts.** `paraphrase_historical_reference` (5,496 facts) stays as the
  historical comparison; `paraphrase_wellformed` is CSV-aware (all facts for corrected
  evaluations, the reference set for legacy CSVs).

## Disagreements / scope notes

* None of the 21 defects is disputed. Two review statements were narrowed in the fix rather than
  taken literally: (15) the X/E character limits were loosened (mean 6 chars, claim start 16)
  because token agreement, the quantity the model sees, is now the primary criterion and the two
  cannot both be tight with a finite filler inventory; (2) the migration script is one-off tooling
  that already ran, so it was made safe and tested with mocks but not re-executed.
* "Fixed" in `FINDINGS_MATRIX.md` for the first review now refers to the tested behaviour
  after this pass; the entries that the second review showed to be overclaimed (19, 20, 3/4 gate,
  5 matching) are corrected there.

## Recheck (reviews/2026-10-14-recheck/REVIEW.md, 8 findings + reporting gaps) -> status with evidence

The recheck (same day, evening) found that "all 21 fixed" was too strong: the compatibility
finding was only partially fixed and five integration paths failed. Every finding below was
reproduced with the reviewer's script against the uncommitted tree first (all eight reproduced:
`reviews/2026-10-14-recheck-repair/reproductions_before_fixes.log`), then fixed with a regression
test derived from that script, then re-run (`.../reproductions_after_fixes.log`). Passing tests are
the evidence that the counterexamples are now refused or corrected, not a proof of absence of
further defects; the two-seed smoke workflow and the recheck's own scripts are the integration
evidence.

| # | finding | status | what changed | evidence |
|---|---|---|---|---|
| R1 | cross-replicate analysis pooled mock and real evaluations (`ok=True, mock=False`, pooled 11) | **fixed** | `check_compatibility` enforces ONE contract across every CSV of the analysis (all replicates): mode, tokenizer sha256, scoring policy, protocol, construction version, CSV schema, probe-file hashes; a mode mix is a violation; `mock` only when every meta is mock, and `Report.table` marks the replicate tables | `tests/test_analysis.py::test_recheck1_cross_replicate_analysis_never_pools_mock_and_real`; `repro/review_third_eval_contracts.py` now stops at the first scenario with "replicates mix evaluation modes"; smoke stage `analysis_refuses_mixed_mock_and_real_across_replicates` |
| R2 | analysis never bound its selected inputs to the evaluations (doses rewritten 1 -> 256 reported as 256) | **fixed** | `analysis_inputs()` hashes the selected probe files, build summary, manifest (checked against `BUILD_COMPLETE.json`) and the exposure dir's `PREFLIGHT_OK.json`/`cross_arm.json` identities; every meta's `input_hashes` and the checkpoint's `bindings` (build summary, probe hashes, `prepack_dataset_id`) must match them; `--compat historical-builds --compat-reason` turns probe/build mismatches into recorded warnings; `compatibility.json["inputs"]`, `inputs_bound` per CSV, the input table opens section 0a | `tests/test_analysis.py::test_recheck2_analysis_inputs_are_bound_to_the_evaluations` (foreign probes refused; declared remapping recorded; build summary, checkpoint bindings, exposure ids, manifest marker); smoke stage `analysis_refuses_foreign_probe_files` |
| R3 | "matched" contracts omitted the scored texts, the initial weights and the horizon (031c4e5 vs repaired constructor: 15 different competitors accepted; steps 100/900 with different init hashes accepted) | **fixed** | per-row `item_sha256` = `ctx.protocol.scored_item_identity(context, cand_value, cand_competing, value_span_chars, metric, template_id, scoring_policy, boundary_k, n_ctx_tokens)`, recomputed by the analysis from the columns and required to agree for every item across every CSV; `ctx.probes.PROBE_CONSTRUCTION_VERSION` (`structured_v2.1`) recorded as `probe_construction` and compared; eval meta `checkpoint` carries `init_param_sha256`, `cli_init_seed`, `resume_from_step`, `preflight`, `total_batch_size`, `num_iterations`, `max_seq_len`, `device_batch_size`, `grad_accum_steps`, `world_size`; one `init_param_sha256` per replicate, distinct across replicates; one `step`/`total_batch_size`/`num_iterations`/`max_seq_len` per experiment unless `--allow-unequal-horizon "reason"` (recorded); missing provenance = violation (strict) / "unverified" warning (legacy) | `tests/test_analysis.py::test_recheck3_scored_item_identity_init_identity_and_horizon`, `::test_recheck3_mixed_evaluator_versions_are_refused` (the actual 031c4e5 module from git vs the current one: 15 mismatches, both the construction version and the item identities stop the analysis); `tests/test_eval.py::test_dry_run_cli_on_v2_build` (identity recomputable per row, meta fields); `repro/review_third_mixed_evaluator_versions.py` now exits 1 |
| R4 | a changed accumulation schedule on resume received a valid preflight verdict (15 micro-batches consumed instead of 20) | **fixed** | `scripts/base_train.py` records the resolved schedule (`num_iterations`, `grad_accum_steps`, `world_size` + B/T/total batch) in every meta and, on resume, refuses any difference in these six keys before the preflight gate and before any step (older metas via `user_config` fallbacks; unrecorded keys refused for prepacked, warned for bestfit); a prepacked resume also checks the saved loader position (`epoch*epoch_rows+cursor == step*grad_accum*B*W`); no segmented exposure history is supported | `tests/test_seeds.py::test_tiny_cpu_resume_refuses_schedule_change_and_wrong_loader_position` (real tiny CPU runs: GA 2 behind a fresh GA-2 preflight refused naming `total_batch_size`/`grad_accum_steps`, changed horizon refused, tampered cursor refused, same-configuration resume still exact); `repro/review3_resume_ga.py` now stops with `ValueError: cannot resume ... total_batch_size: checkpoint 192 vs requested 384; grad_accum_steps: checkpoint 1 vs requested 2`; smoke stage `resume_with_changed_accumulation_refused` |
| R5 | the first checkpoint upload failed when the HF repo did not exist (`['list_repo_refs']` then `RepositoryNotFoundError`) | **fixed** (mocks only) | `hf_ensure_repo` (`create_repo exist_ok`) runs before the run-tag lookup in `hf_upload_checkpoint`; `hf_repo_refs(missing_ok=)` treats absence explicitly; auth/permission errors propagate unchanged; `hf_upload_eval` refuses an absent repo (never creates one); `migrate_wandb_to_hf.py` creates the collection repo before its first revision lookup | `tests/test_artifacts.py::test_recheck5_first_checkpoint_upload_creates_the_absent_repo` (absent repo, existing repo without the tag, tag conflict), `::test_recheck5_auth_failures_propagate_and_nothing_is_uploaded`; `tests/test_migration.py::test_recheck5_absent_destination_is_created_before_the_first_revision_lookup`; `repro/review3_archival.py` now prints `CALL_ORDER ['create_repo', 'list_repo_refs', ...] REPO_CREATED True`. No live Hub round-trip was performed (not authorized here) |
| R6 | two seeds collided in evaluation outputs and run metadata (seed 8 failed with "CSV exists", one `run_info.json` renamed to run 8) | **fixed** | run metadata per model tag (`arm_<A>/run_info.json` for `d<depth>`, `arm_<A>/run_info_<tag>.json` otherwise; `ctx.tracking.run_info_path/read_run_info/write_run_info(exp_dir, arm, tag, ...)`, `run_infos`); evaluation outputs per tag (`eval/`, `eval/seed_<S>/`, `eval/<tag>/`; `eval_subdir_of`/`eval_output_dir`), passed explicitly by `run_arm.py` and used by default by `eval_beliefs.py`; the evaluation takes its run identity from the checkpoint meta (`wandb_run`), then the tag's run_info (`meta["run"]`); `links.py --exp-dir` lists every tag; `load_artifact_hf.py --model-tag` is tag-aware; `analyze_beliefs.discover_replicates` accepts any replicate subdir and refuses a mixed layout | `tests/test_artifacts.py::test_recheck6_run_info_and_eval_outputs_are_scoped_per_model_tag`, `::test_recheck6_launcher_two_seeds_keep_separate_evaluations_and_identities` (the reviewer's launcher scenario: two CSVs under `eval/seed_7` and `eval/seed_8`, two run_info files naming run7/run8, re-launching seed 7 refuses to overwrite its evaluation); `tests/test_eval.py` (default paths per tag); the smoke workflow trains and evaluates two seeds through `run_arm.py` into one build and analyzes them as replicates; `repro/review3_replicates.py` now shows both seeds evaluated (its final line reads the legacy `run_info.json`, which no longer exists for replicate tags) |
| R7 | the lightweight build env (`contextualization/.venv`) had no `tiktoken`, so the builder silently disabled token matching; the validator's pickle fallback reported vocab 0 and selected the toy relative limits (9/49 = 18.4 % accepted where the absolute limit of 8 rejects it) | **fixed** | `tiktoken>=0.11.0` in `contextualization/pyproject.toml` + `uv.lock`, `uv sync` rebuilt `.venv` (tiktoken 0.14.0; note: `uv sync` also pruned ad-hoc packages that were not in the lockfile, e.g. rich/typer/pygments); ONE shared loader `ctx/tokenizers.py` (`load_encoder`: RustBPETokenizer when importable, else the tiktoken pickle, same ids; `PickleEncoder.get_vocab_size()` = the true `n_vocab`; `TokenizerUnavailable` with an install hint; `token_limit_mode`: relative only for a KNOWN vocab < 16,384, unknown = absolute); `build()` loads and validates `--tokenizer-src` before the cache/staging and stops on an unusable one (loud WARNING for v2+E without one); `validate.py` uses the same loader and selector | `tests/test_tokenizers.py` (both loader paths agree on the small and the historical tokenizer; corrupt/missing raise; unknown vocab -> absolute; the documented lightweight env loads the historical tokenizer in a subprocess, vocab 32768); `tests/test_build.py::test_unusable_tokenizer_src_stops_the_build_before_anything_is_written`; `tests/test_validate.py::test_validator_fallback_loader_reports_the_true_vocabulary_and_selects_the_same_limits`, `::test_validator_unknown_vocabulary_selects_absolute_limits_never_relative`; `repro/ctx_data_rereview_repro.py` now loads the tokenizer in the light env (`match_tok PickleEncoder(vocab=32768, ...)`) and its FALLBACK_POLICY line crashes by construction (it reaches into the old closure); the equivalent numbers in `reviews/2026-10-14-recheck-repair/data_recheck_after_fixes.log`: reported vocab 32768, mode absolute, the 9-token delta rejected |
| R8 | deleting `novelty_audit.json` made `--probes-only` flip 35 held-out labels to `unverified` while reporting `verified_against_manifest=True` | **fixed** | `_novelty_audit_evidence`: labels come from `novelty_audit.json` when present (must carry the summary's `build_id` and agree with the audit embedded in `build_summary.json`), else from the embedded audit (NOTE names the source); no usable hits anywhere -> refusal; a summary with `build_id` refuses audits without one (legacy summaries: NOTE); after regeneration the records are compared with the build's ORIGINAL probe files (those that still hash to `BUILD_COMPLETE.json`) on every pool-level field and any difference is a refusal; `verified_against_probe_files`, `probe_files_compared`, `novelty_evidence` recorded in `PROBES_META.json` and the result | `tests/test_build.py::test_probes_only_restores_novelty_labels_from_the_embedded_audit_when_the_file_is_missing`, `::test_probes_only_refuses_audit_evidence_that_is_missing_or_inconsistent`, `::test_probes_only_regenerates_identically_from_a_copied_archive_set`, `::test_probes_only_catches_tampered_audit_evidence_through_the_original_probe_files`; after the fix the reviewer's 80-group scenario changes 0 labels (`data_recheck_after_fixes.log`) |

Smaller reporting / protocol gaps of the recheck:

| gap | status | what changed | evidence |
|---|---|---|---|
| `design_cells()` never called; summary lacked the field claimed earlier | **fixed** (the earlier claim was wrong) | `blend.build()` stores `summary["design_cells"]` for v2 (zero cells included) | `tests/test_build.py::test_summary_carries_design_cells_and_the_limits_version`, `::test_design_cells_list_empty_cells_when_the_grid_has_more_doses_than_a_stratum_has_groups` |
| `max_abs_delta_claim_tok_start` recorded but not enforced | **fixed** | part of `within_token_limits` in `blend._xe_summary` (absolute mode) and of the new validator check `8.claim_tok_start`; relative mode uses the per-occurrence fraction `|delta_claim_tok_start| / tok(X) <= max_abs_delta_tokens_frac` | `tests/test_templates.py::test_xe_match_limits_are_versioned_and_the_claim_start_token_rule_is_enforced`; `tests/test_validate.py::test_fresh_v2_build_passes` requires the check |
| distributed loss report summed `B x world_size` rows while the objective normalizes per rank | **fixed** (report only; objective untouched) | `loss_weighting_report` adds `per_rank`, `per_microbatch_rank_local`, `rank_local_microbatches_with_zero_active` and `note_distributed` | `tests/test_prepacked_loader.py::test_loss_weighting_reports_rank_local_denominators` (world size 1 and 2) |
| X/E limits to be frozen and documented | **done** | `templates.XE_MATCH_LIMITS_VERSION = "2026-10-14.1"` recorded in `build_summary.json["xe_diagnostics"]["limits_version"]` and in the validator numbers; rules spelled out in `docs/PROTOCOL_V2.md` section 3 | `tests/test_build.py::test_summary_carries_design_cells_and_the_limits_version` |

### Integration evidence after the recheck repair

* Complete suite: see "Commands" (result recorded in `reviews/2026-10-14-recheck-repair/README.md`).
* `smoke_crxe.py --work <dir> --train` now trains and evaluates TWO init seeds (7, 8) of all four arms
  through `run_arm.py` into one build (`eval/seed_7`, `eval/seed_8`, `arm_<A>/run_info_d2_s<S>.json`),
  checks identical `init_param_sha256` across arms per seed and distinct across seeds, resumes arm C
  from a mid-run checkpoint (identical end state), refuses a resume with a changed accumulation
  schedule behind a freshly generated preflight, analyzes the two seeds as replicates with the
  input/contract checks (`inputs_bound` for every CSV, 0 item-identity mismatches), and refuses a
  mixed mock/real replicate, foreign probe files and a stale preflight. Summary archived as
  `reviews/2026-10-14-recheck-repair/smoke_crxe_summary_recheck.json`.
* Every recheck reproduction was re-run after the fixes (`reproductions_after_fixes.log`).

### Recheck: disagreements, scope notes and what remains unverified

* No finding is disputed. R5 and the migration ordering were repaired and tested with the
  revision-aware fake Hub only: no live HF archival/download round-trip has been performed in this
  repository (it needs an explicitly authorized session with credentials).
* `uv sync` for the lightweight environment pruned packages that were installed ad hoc and not in
  the lockfile (at least pygments, pyparsing, rich, shellingham, typer); `.venv/bin/hf` still works.
* The reviewer's `ctx_data_rereview_repro.py` cannot complete after the fix because its
  FALLBACK_POLICY line inspects the old closure-based fallback; `review3_replicates.py` ends with a
  `KeyError` because it reads the legacy `arm_C/run_info.json`, which replicate tags no longer write.
  Both are expected consequences, documented in `reproductions_after_fixes.log`.
* `--allow-unequal-horizon "reason"` and `--compat historical-builds --compat-reason` are the only
  declared bypasses of the analysis contract; neither relaxes the item identity, the init identity
  or the mode rule.
* Loss normalization is still NOT changed (objective `loss = loss / grad_accum_steps` untouched); the
  report now also measures the rank-local denominators the current objective uses. The FineWeb
  commit is still the experimenter's choice.

## Commands

```bash
cd /home/danielaush/Documents/peacechat
PYTHONDONTWRITEBYTECODE=1 $HOME/.cache/ctx_cpu_venv/bin/python -m pytest contextualization/tests -q -p no:cacheprovider
for s in review_root_v2 review_migration_v2 review_tokenizer_binding review_v2_training review_training_extra \
         review_eval_v2 review_analysis_pairing review_heldout_cancel ctx_data_review_repro; do
  $HOME/.cache/ctx_cpu_venv/bin/python contextualization/reviews/2026-10-14-code-review/repro/$s.py
done
# recheck reproductions (each now stops at the fixed defect; see reviews/2026-10-14-recheck-repair/)
for s in review_third_eval_contracts review_third_mixed_evaluator_versions review3_resume_ga review3_archival \
         review3_replicates ctx_data_rereview_repro; do
  PYTHONDONTWRITEBYTECODE=1 $HOME/.cache/ctx_cpu_venv/bin/python contextualization/reviews/2026-10-14-recheck/repro/$s.py
done
# two-seed smoke workflow (build -> validate -> prepack/preflight -> run_arm.py x 2 seeds x 4 arms -> resume ->
# refusals -> mock evals -> cross-replicate analysis), about 10 minutes on a laptop CPU
PYTHONDONTWRITEBYTECODE=1 $HOME/.cache/ctx_cpu_venv/bin/python contextualization/smoke_crxe.py --work /tmp/ctx_smoke --train
```
