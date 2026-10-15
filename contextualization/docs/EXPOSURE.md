# Training exposure: loader replay, matched prepacking, seeds

Written 2026-10-14 for the repairs of REVIEW 2026-10-13 findings 3 (equal corpus counts do not
establish equal training exposure), 4 (the loader's epoch counter is not an exposure count, and
resumes are approximate) and 7 (a dataset seed is not an initialization seed); extended the same
day for REVIEW 2026-10-14 findings 3 (tokenizer binding), 4 (the prepack gate and dataset-bound
resumes), 16 (step agreement in the audit, the preflight gate) and 21 (descriptor-bounded loader),
plus the resume-provenance, loss-normalization and opt-in-loader items of its "Experimental
decisions" section. Formats are fixed in `docs/SCHEMAS_V2.md` ("Exposure index v1", "Prepacked
rows", loader state). Code:

| piece | file |
|---|---|
| tagged replay of the historical loader + occurrence audit | `contextualization/ctx/exposure.py` |
| rows planned once across arms, schedule helpers, prepack audit | `contextualization/ctx/prepack.py` |
| CLI for all of the above | `contextualization/exposure_audit.py` |
| opt-in prepacked loader (bestfit loaders untouched) | `nanochat/dataloader.py` |
| `compute_init(device_type, seed=42)` | `nanochat/common.py` |
| `--init-seed`, `--data-seed`, `--data-loader`, `--skip-preflight-check` + checkpoint provenance + the preflight gate | `scripts/base_train.py` |
| launcher: explicit loader for v2 builds, preflight check, per-replicate model tags, init identity | `contextualization/run_arm.py` |
| tests | `contextualization/tests/test_exposure.py`, `test_prepacked_loader.py`, `test_seeds.py` |

Run the tests from the repo root with the CPU environment:

```bash
$HOME/.cache/ctx_cpu_venv/bin/python -m pytest contextualization/tests/test_exposure.py \
    contextualization/tests/test_prepacked_loader.py contextualization/tests/test_seeds.py -q
```

Nothing here regenerates or modifies historical artifacts; the replay only reads shards.

## 1. What the historical loader does to an inserted claim

`nanochat.dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit` keeps a buffer of
1000 tokenized documents (BOS prepended), builds each row of `T+1 = 2049` tokens by repeatedly
taking the *largest* buffered document that still fits, and when nothing fits it takes the
*shortest* buffered document and keeps only its prefix that fills the row. Documents are read
row-group by row-group (1024 docs), striped over DDP ranks, in 128-document text batches. The
loss (`nanochat/gpt.py`) scores `targets = row[1:]`, so a document placed at row positions
`[s, e)` is scored at positions `s+1 .. e-1`; a span is "scored complete" only if all its token
positions lie inside the kept prefix.

Consequences measured by the audit, not estimated:

* an occurrence inserted late in a long carrier can be cut in one arm and kept in another
  (longer X wrappers shift token offsets; `tests/test_exposure.py::test_finding3_...` reproduces
  the review's 2,049- vs 2,059-token case through audit statuses: R `complete`, X
  `partial_value_lost`);
* the `epoch` in the loader state is the traversal that the *buffer refill* has reached, not the
  number of times any document was trained on; documents late in the stream are seen fewer
  times than early ones at the stopping step;
* `base_train` prefetches one micro-batch beyond the last optimizer step; micro-batches with
  index `>= num_iterations * grad_accum_steps` are never trained on (`not_in_horizon`).

## 2. Historical replay (`ctx.exposure.replay_bestfit`)

`replay_bestfit(doc_stream, tokenizer, B, T, *, buffer_size, tokenizer_batch_size, world_size,
rank, grad_accum_steps, num_iterations, resume_state=None)` is a statement-by-statement mirror
of the loader and of `_document_batches` (same refill-to-`buffer_size`, same best-fit choice with
first-index tie-break, same crop-shortest rule, same row-group striding per rank, same epoch
semantics including the `+1` row-group skip on resume), but over *tagged* documents: every
placement is reported as `(micro-batch, row, start, end, doc_index, epoch, cropped)`. It yields
`num_iterations * grad_accum_steps + 1` micro-batches (the last one is the never-trained
prefetch). Best-fit scans use numpy over the buffer so the replay stays memory-bounded (the
buffer plus a per-document length cache; nothing proportional to the number of placements).

What is verified (fixtures, `tests/test_exposure.py`):

* the replay's token rows and state dicts are identical to the REAL loader's for world size 1
  and for both ranks of world size 2, over 300 micro-batches spanning 10-23 epochs with ~570
  crops per stream;
* resume: the real loader restarted from the state yielded with micro-batch `k` is reproduced
  exactly by `replay_bestfit(..., resume_state=state_k)`;
* the resumed stream is **not** the uninterrupted stream: on the fixture (B=2, T=64, buffer 8,
  resume at micro-batch 37, 120 micro-batches compared) **120 of 120** resumed micro-batches
  differ from the uninterrupted ones. Buffered and partially consumed documents are dropped and
  the row group after the saved one is skipped, so every resume of the historical runs changed
  the data stream from that point on. A replay of a run that was resumed needs the saved loader
  states of every resume (`meta_*.json` -> `dataloader_state_dict`); without them the audit
  describes the uninterrupted stream.

What is not verified: the historical 45B-character runs themselves. Replaying them needs the
historical shards (byte-identical), the historical tokenizer (`tokenizer.pkl` sha256
`36ca7984...`), the training configuration (`B=8, T=2048, total_batch_size=1048576 =>
grad_accum=64, 8352 steps, world size 1`; `ctx.exposure.HISTORICAL_TRAIN_CONFIG`) and the resume
points. The replay tokenizes every document once per arm (~11B tokens for a 45B-character arm;
hours of CPU time, later epochs reuse cached lengths); it was exercised here only on fixtures.

## 3. The occurrence audit (`ctx.exposure.audit_arm`, `exposure_audit.py --mode historical`)

Inputs: an experiment dir with `arm_<A>/base_data/shard_*.parquet`, `manifest.csv` and the
tokenizer. Character spans from `manifest.csv` v2 are mapped to token spans through byte offsets
(`tiktoken`'s `decode_single_token_bytes`; char -> byte via UTF-8 of the document prefix, so a
multi-byte character such as `°` before the value does not shift anything). A token belongs to a
span if its byte range intersects the span's bytes, so a leading-space merge like `" Tessmoor"`
counts as the value's token. Legacy manifests (`rendering_id` + `shard_file` only) are supported
when the caller supplies `insert_fn(row) -> insert_text` (or a dict with `insert_text`,
`claim_text`, `subject`, `value`); each insert must match exactly once in exactly one document of
its shard, otherwise `ExposureError` (`--legacy-locator module:function` on the CLI).

Output `exposure/<arm>/occurrence_exposure.csv` (columns exactly as in SCHEMAS_V2): one row per
*placement* of an occurrence's document inside the replayed stream (one per epoch in which it was
placed), plus one `omitted` row for occurrences whose document was never placed. Semantics:

* `row_index = micro_batch * B * world_size + rank * B + row` (global row position of the
  training stream); `row_offset_start/end` = the document's kept prefix inside the row;
* `*_tok_start` is the row position of the span's first token when that token is kept, else -1;
  `*_tok_end` is the row position past the span's last token when the whole span is kept, else
  -1 (so `start >= 0, end == -1` means "cut inside the span");
* `status`: `partial_value_lost` if the first value slot is not fully scored, else
  `partial_subject_lost`, else `partial_claim_cut` (claim clause cut after subject and value),
  else `complete`; `not_in_horizon` for placements in micro-batches `>= num_iterations *
  grad_accum_steps`; `omitted` when never placed within the replayed stream. `complete` means
  the claim clause (which contains subject and value) is fully scored; an X wrapper's tail
  after the clause may still be cut (`insert_tok_end == -1` with `claim_tok_end >= 0`);
* `scored_claim_tokens` / `scored_value_tokens` count scored target positions (the latter over
  all value slots); 0 for `omitted` / `not_in_horizon`;
* `epoch` is the loader's 1-based epoch of the placement, `micro_batch` the rank-local index,
  `step = micro_batch // grad_accum_steps` (-1 outside the horizon).

`summary.json` has counts by status, `effective_exposures` per fact (complete placements with
`step < num_iterations`), `by_dose` (per assigned frequency: facts, scheduled occurrences,
effective totals/mean/min/max/histogram, partial totals, facts with zero effective exposure),
scored-token totals, the replay statistics per rank (micro-batches, placements, crops, final
state) and the inputs (shard listing, manifest sha256, protocol, build summary sha256).
`--arms R X` additionally writes `exposure/cross_arm.json`: per occurrence the status/step/row in
every arm, whether steps and rows agree, status-pair counts and `complete_in_R_not_in_X`-style
lost counts.

## 4. Prepacked rows (`ctx.prepack.prepack_experiment`, `exposure_audit.py --mode prepack`)

The corrected training path removes the arm dependence of packing by planning the rows **once**
over the lockstep document streams of all arms and realising them per arm:

* the arms' train streams are read in `doc_index` order; every non-intervention document must be
  byte-identical across arms (checked; `PrepackError` otherwise) and is tokenized once;
* intervention documents are the `manifest.csv` v2 slot documents, identified by `doc_index`
  (they are slot documents in every arm, including C whose version carries the neutral
  insert). Such a document is reserved at the **max token length over arms** and occupies the
  same row positions in every arm;
* placement uses the historical best-fit algorithm on the reserved lengths. When nothing fits,
  the crop prefers the shortest **non-intervention** document; an intervention document is only
  cropped when no other candidate is buffered and only at or after the end of its inserted
  sentence (max over arms of `insert_char_end`'s token end), never inside it. If a document's
  inserted sentence cannot fit a row at all (`insert_tok_end > T+1`) `prepack_experiment` raises
  `PrepackError` naming the occurrence, arm and document: carriers are chosen and cropped by the
  *builder*, and a build whose inserts do not fit must be fixed there, not silently cut here;
* arms whose rendering is shorter than the reserved length are padded with `pad_token_id`
  (default BOS) and `mask = 0` at those positions; the loader turns them into target `-1`
  (`ignore_index`). `mask[r, 0] = 0` always (position 0 is never a target). The pads are visible
  as context to the following document inside the row, exactly as the previous document's tokens
  are in the bestfit loader;
* top-up documents (`doc_index >= T`) present in every arm (the shared prefix of the top-up
  stream) are packed after the base stream in shared order; arm-specific extra top-ups are
  dropped and listed per arm in `dropped_topup_docs`;
* the stream is finite: once it is exhausted the remaining buffered documents are packed and the
  tail of at most the last rows is padded (`packing.padded_tail_tokens`, `rows_with_padded_tail`
  in the manifest). A row tail is also padded in the (practically unreachable) case that only
  intervention documents whose inserts do not fit remain in the buffer.

**What differs across arms, intentionally: only the tokens inside the reserved regions** (the
rendering of the inserted sentence, and where the masked pads sit when a rendering is shorter).
Identical across arms: the set of rows, their order, every non-intervention document and its crop
point, the position of every intervention document, the crop point of a cropped intervention
document, and the loss mask outside reserved regions (tested). Scored token totals therefore
differ across arms by at most the rendering-length differences; they are reported per arm
(`scored_tokens_total`, `masked_tokens_total`, `intervention_tokens_scored`,
`pad_tokens_in_reserved_regions`) so the token budget of each arm is known exactly rather than
assumed equal.

Outputs per arm under `arm_<A>/prepacked/`: `rows_XXXXX.npy` (int32 `[n, T+1]`),
`mask_XXXXX.npy` (uint8), `prepack_manifest.json` (SCHEMAS_V2 fields plus `reserved_regions`,
packing statistics, `data_seed`, `code`, and the bindings below) and `occurrence_exposure.csv`,
the prepack-time exposure index: a status for every occurrence with `epoch/micro_batch/step = -1`
(the schedule depends on the training configuration, see below).

**Bindings recorded in `prepack_manifest.json`** (REVIEW 2026-10-14 findings 3 and 4):

* `encoding_fingerprint`: sha256 of a canonical serialization of the tokenizer object that
  ACTUALLY encoded the rows (`ctx.prepack.encoding_fingerprint`, implemented once in
  `nanochat.dataloader.encoding_fingerprint` because the loader must check it without importing
  this package): the tiktoken split pattern, the sorted `(token bytes, rank)` pairs of
  `enc._mergeable_ranks`, the sorted special tokens and the BOS id, all length-prefixed. Objects
  that are not tiktoken encodings are fingerprinted by their `encode()` outputs over a fixed
  ~50-string probe corpus plus vocab size and BOS id (a differently tagged digest).
  `tokenizer_file_sha256` (alias `tokenizer_sha256`) is the sha256 of
  `arm_<A>/tokenizer/tokenizer.pkl`; when that file exists it is LOADED and its fingerprint must
  equal the fingerprint of the tokenizer passed to `prepack_experiment`, otherwise `PrepackError`
  (the review's scenario, encoding with one BPE while the arm dirs hold the historical
  tokenizer, is refused before anything is written; `tests/test_prepacked_loader.py::
  test_prepack_refuses_arm_tokenizer_that_encodes_differently`).
* per row file: `sha256`, `mask_sha256` and `active_targets` (= `mask[:, 1:].sum()`);
  `active_targets_total` (= `scored_tokens_total`) and `active_targets_per_row` (min / mean /
  max, histogram by 32-token bins, rows with zero active targets);
* `prepack_dataset_id`: sha256 over `(T, n_rows, ordered (rows sha256, mask sha256) pairs)`. It
  identifies the rows an arm trains on; the loader state and every checkpoint meta carry it (see
  sections 5 and 7). `prepack_run_id` (one uuid per `prepack_experiment` call) ties the arms of one
  prepack together; `role` is `control` for C and `intervention` otherwise, with
  `required_occurrences` (intervention arms) and `control_slots` (C: the slot documents that carry
  the neutral insert) reported separately.

**Staging.** Each arm's output is written to `arm_<A>/prepacked.staging-<pid>` and the staging
dirs of ALL arms are moved into place only after `verify_prepack` passed on them; any failure
removes the staging dirs and leaves an existing `prepacked/` untouched. An existing non-empty
`prepacked/` is refused unless `overwrite=True` (`--overwrite`), and even then it is replaced only
after the new set passed the gate.

**The gate (`verify_prepack(exp_dir, arms, *, check_hashes=False, check_masks=True)`)**,
run at the end of `prepack_experiment` and again by every `--mode prepacked` audit, raises
`PrepackError` unless all of the following hold:

1. the arms agree on `T`, `n_rows`, `pad_token_id`, the row-file layout, `reserved_regions`,
   `arms_planned_together`, `prepack_run_id`, `encoding_fingerprint` and the build hashes, and
   `manifest.csv` is unchanged since prepacking (`manifest_sha256`);
2. occurrence sets come from `manifest.csv` (v2), not from whatever the exposure CSVs happen to
   contain: for every intervention arm each `(fact_id, occurrence)` of its manifest rows appears
   EXACTLY once in its exposure index with status `complete` (missing, duplicated and extra keys
   all fail), the intervention arms share one key set, and C either carries the same keys (its
   control slots) or an empty index; `row_index` agrees across arms;
3. `prepack_dataset_id` equals the digest of the recorded file hashes; `check_hashes=True`
   (`--check-hashes`) re-hashes every rows/mask file;
4. `check_masks`: per file `active_targets` is recomputed from the mask file and the totals must
   match the manifest; for every occurrence the claim, subject and value spans are scored
   (`mask == 1`) in the actual mask and lie inside a reserved region of their row; inside every
   reserved region the mask is a prefix of ones followed by masked pads holding `pad_token_id`;
   masks are identical across arms outside reserved regions;
5. an arm `tokenizer/tokenizer.pkl` (when present) has the recorded file sha256 and encodes with
   the recorded fingerprint.

The three mutations of the review (an X occurrence removed from the index, a zeroed R mask file,
row tokens altered on disk) are each refused: the first two by the gate, the third by the gate
with `check_hashes=True` and by the loader itself (section 5). The report lists `roles`,
`required_occurrences`, `control_slots`, `prepack_dataset_id` per arm, `active_targets_total` and
`tokenizer_checks`.

## 5. The prepacked loader (`nanochat.dataloader.prepacked_distributed_data_loader_with_state`)

`prepacked_distributed_data_loader_with_state(B, T, split, device, resume_state_dict=None,
data_seed=-1, prepacked_dir=None, tokenizer=None, expected_encoding_fingerprint=None,
expected_dataset_id=None, max_open_files=32)` reads `<base_dir>/prepacked` (train split).

* Row order per epoch (epochs are 0-based): identity when `data_seed < 0`, else
  `numpy.random.default_rng(data_seed * 1000003 + epoch).permutation(n_rows)`
  (`prepacked_epoch_order`; `ctx.prepack.epoch_order` is the same function, tested equal).
* Each epoch yields `n_rows // (B * world_size)` micro-batches; micro-batch `i` covers positions
  `[i*B*W, (i+1)*B*W)` of the epoch order and rank `r` takes rows `[r*B, (r+1)*B)` of that
  slice. The incomplete tail of an epoch's order is skipped (with identity order these are
  always the last `n_rows mod B*W` rows; the exposure index marks them `not_in_horizon`).
* `inputs = rows[:, :-1]`, `targets = rows[:, 1:]` with `-1` where `mask[:, 1:] == 0`.
* State: `{"loader": "prepacked", "schema_version": 1, "epoch", "cursor", "data_seed",
  "world_size", "B", "prepack_dataset_id"}` with `cursor` the epoch-order position of the first
  row of the batch being yielded. `base_train` saves the state of the prefetched micro-batch, so
  resuming replays that batch first and the resumed token stream equals the uninterrupted one
  (tested at several cursors, mid-epoch and across the epoch boundary, world size 1 and 2). This
  is the difference from the bestfit loader, whose resume drops buffered documents and skips a
  row group.
* A state with a different `B`, `world_size` or `data_seed`, a bestfit state, a state without
  `prepack_dataset_id` or one whose id differs from the store's is refused; the manifest's `T`
  must equal the requested `T`.
* Tokenizer binding (finding 3): `tokenizer=` (base_train passes the tokenizer it trains with)
  and/or `expected_encoding_fingerprint=` must match the manifest's `encoding_fingerprint`, at
  construction and on resume, otherwise `ValueError`; `expected_dataset_id=` likewise.
* Integrity (finding 4): the manifest must carry `prepack_dataset_id` and per-file hashes, the id
  must be the digest of those hashes, and every chunk is hashed against the manifest the first
  time it is opened (`PrepackedRows(verify_hashes=True)`), so altered rows are refused before any
  of their tokens are fed, on a fresh start and on resume alike.
* Descriptors (finding 21): `PrepackedRows` keeps a bounded LRU cache of open memmaps
  (`max_open_files` chunk PAIRS, default 32 = 64 descriptors), opens chunks lazily and closes
  evicted ones; at construction `resource.getrlimit(RLIMIT_NOFILE)` is checked and a cache that
  needs more than the soft limit minus 64 descriptors is a clear `RuntimeError`. The review's 520
  tiny chunk pairs under a 1,024 soft limit load and iterate (tested).
* `split == "val"` falls back to the bestfit val loader over the last parquet shard (the
  prepacked set holds the train split only), so val bpb stays comparable with bestfit runs.

`scripts/base_train.py --data-loader prepacked` uses it for training; validation stays on the
bestfit loader; a checkpoint written with one loader cannot be resumed with the other, and a
prepacked checkpoint is bound to its rows: on resume `bindings.prepack_manifest_sha256` and
`bindings.prepack_dataset_id` of the loaded meta must equal the current store's (section 7).

## 6. Scheduled doses become scored doses: the preflight (`exposure_audit.py --mode prepacked`)

The prepack-time index says where each occurrence sits (`row_index`, token spans, status). For a
training configuration, `ctx.prepack.step_of_row(row_index, epoch, data_seed, n_rows, B,
world_size, grad_accum_steps)` gives the micro-batch, optimizer step, rank and row-in-batch at
which that row is trained in that epoch (-1 in an epoch's skipped tail), and
`schedule_exposure` expands the index into `exposure/<arm>/occurrence_exposure.csv` with one row
per epoch the loader reaches up to the prefetched micro-batch: `complete` with its `step` inside
the horizon, `not_in_horizon` beyond it (dropped epoch tails included). `summary.json` reports
the same quantities as the historical audit plus the prepack totals and bindings. The exposure
of an occurrence is thus determined by the schedule alone, identically for all arms; nothing is
estimated from frequency x epochs.

`--mode prepacked` is the preflight of a training configuration (REVIEW 2026-10-14 finding 16:
the earlier audit enforced row agreement only). It deletes any stale
`<exp-dir>/exposure/PREFLIGHT_OK.json` first, then exits 2 (writing no preflight) unless:

* `verify_prepack` passes (section 4; `--check-hashes` re-hashes every file);
* `--arms` lists exactly the arms prepacked together, and their manifests share `data_seed`, `T`
  and `n_rows` (the review's stored seeds R=1, X=2, E=3 now fail here);
* every status is `complete` or `not_in_horizon`, every occurrence has at least one `complete`
  row with `step < num_iterations` in every arm (an occurrence that only ever lands in a skipped
  epoch tail or beyond the horizon fails the preflight), and
  `n_step_agree == n_occurrences == n_row_agree == n_present_in_all_arms == n_all_effective`
  over the arms that carry an index (C without manifest rows is a control-only arm and takes
  part in the dataset / configuration checks only).

On success it writes `PREFLIGHT_OK.json` (`schema_version` 1, `created_at`, `arms`,
`intervention_arms`, `control_arms`, `B`, `T`, `grad_accum_steps`, `num_iterations`,
`world_size`, `data_seed`, `prepack_dataset_id` and `prepack_manifest_sha256` per arm,
`encoding_fingerprint`, `tokenizer_file_sha256` / `tokenizer_checks`, the `verify` and
`cross_arm` summaries, the achieved exposure per arm (status counts incl. `not_in_horizon`,
`by_dose`, horizon) and the `loss_weighting` block of section 10). `cross_arm.json` gains
`per_arm` summaries (`status_counts`, `n_zero_effective`, `n_rows_not_in_horizon`),
`n_all_effective`, `preflight_config` and the same `loss_weighting` block.

**The gate in training.** `scripts/base_train.py --data-loader prepacked` loads
`<base_dir>/../exposure/PREFLIGHT_OK.json` (`base_dir` = the arm dir) before the first step and
refuses to train unless it exists and its `B`, `T`, `grad_accum_steps`
(`= total_batch_size // (B*T*world_size)`), `num_iterations`, `world_size`, `data_seed`,
`prepack_dataset_id[<arm>]` and `encoding_fingerprint` equal the actual run's. The loader itself
checks the tokenizer binding and the row hashes independently. `--skip-preflight-check` bypasses
the file comparison only: it prints a loud warning and the checkpoint meta records
`preflight: "skipped"` (otherwise `"ok"`; `null` for bestfit runs); `bindings.preflight_path` /
`bindings.preflight_sha256` record the file that was checked.

## 7. Seeds, resume provenance and the init identity (`--init-seed`, `--data-seed`)

* `nanochat.common.compute_init(device_type, seed=42)` seeds `torch.manual_seed` /
  `torch.cuda.manual_seed`; the only consumer of the global RNG is `GPT.init_weights`. Default 42
  reproduces the historical runs' initialization (tested: same seed -> identical parameters,
  different seeds -> different, default == 42).
* `--data-seed` fixes the per-epoch row order of the prepacked loader only; the bestfit loader
  has no data seed (its order is the shard order).
* For a replicate of the C/R/X/E design use **the same `--init-seed` and the same
  `--data-seed` in every arm** (identical initialization and identical row schedule, so the arms
  differ only inside the reserved regions), and **different `--init-seed` (and optionally
  `--data-seed`) across replicates**. A builder `--seed` changes the data realisation, never the
  initialization.
* The checkpoint meta json (`meta_*.json`) records `init_seed`, `data_seed`, `data_loader`,
  `code` (git commit/dirty/branch, argv, cwd), `nondeterminism` (torch.compile, TF32 matmul
  precision, FA3, cuDNN flags, world size, compute dtype, device, torch/CUDA versions and the
  note that a seed fixes initialization and data order only; kernels/compile/TF32 are not
  bitwise deterministic) and `bindings` (sha256 of `tokenizer.pkl`, `encoding_fingerprint` of
  the tokenizer object, the data listing, the prepack manifest, `prepack_dataset_id`, the
  preflight file, `build_summary.json` and the probe files). The existing `wandb_run` block is
  kept; the prepacked loader state carries `"loader": "prepacked"` and `prepack_dataset_id`.
* `init_param_sha256` (reviewer check 4): sha256 over the model parameters right after
  `init_weights()` and before any optimizer step (state dict in sorted key order; each tensor's
  name, shape, dtype and contiguous CPU bytes). Two runs with the same `--init-seed` and config
  record the same value, another seed a different one (tested with real tiny CPU runs).
  `run_arm.py` copies it (with `init_seed_effective`) into `run_info.json` after training and
  prints `init params MATCH/DIFFER` against the other arms of the same depth and init seed.
* Resume provenance: on `--resume-from-step` the meta's `init_seed` stays the ORIGINAL seed of
  the loaded checkpoint and `init_param_sha256` is carried forward from its meta (never
  recomputed); a different CLI value is recorded as `cli_init_seed` with a warning (a seed-7
  model resumed with the default CLI seed is no longer relabelled 42); `resume_from_step` records
  the step the process resumed from. The data stream must be the checkpoint's: a different
  `data_loader` or `data_seed` is an error, and for the prepacked loader
  `bindings.prepack_manifest_sha256` / `bindings.prepack_dataset_id` of the checkpoint must equal
  the current store's (and the loader state's id must match too).
* Exact resume keeps the checkpoint's schedule (review 2026-10-14 recheck, finding 4). The
  prepacked loader state counts consumed rows while the training loop resumes from an
  optimizer-step count, so a resume with another `--total-batch-size` (i.e. another
  `grad_accum_steps`), horizon, `--device-batch-size`, `--max-seq-len` or world size would
  consume a different number of micro-batches than the preflight certified, and the preflight
  gate alone cannot see it (it only compares the current configuration with the current
  `PREFLIGHT_OK.json`). Every `meta_<step>.json` now records the resolved schedule:
  `num_iterations` (the resolved horizon, never the `-1` CLI value), `grad_accum_steps` and
  `world_size` next to `device_batch_size`, `max_seq_len` and `total_batch_size`. On
  `--resume-from-step`, `scripts/base_train.py` compares these six keys with the requested run
  once `grad_accum_steps` and `num_iterations` are resolved, before the preflight gate and
  before any training step; any difference is an error naming each key (checkpoint value vs
  requested). Older metas are read through fallbacks (`user_config` for B/T and for explicit
  `total_batch_size`/`num_iterations` overrides; `dataloader_state_dict`/`nondeterminism` for
  `world_size`; `grad_accum_steps` derived from the four recorded values); a key the checkpoint
  does not record at all is refused for the prepacked loader (missing provenance) and only
  warned about for bestfit. A prepacked resume additionally checks the saved loader position
  against the checkpoint's own schedule: `epoch * epoch_rows + cursor` (with `epoch_rows =
  (n_rows // (B*W)) * (B*W)` from `prepack_manifest.json`) must equal `resume_from_step *
  grad_accum_steps * B * W`, otherwise the resume is refused. A deliberate schedule change needs
  a new run: supporting it would require a segmented exposure history covering the rows already
  consumed, which is not implemented. Tests: `tests/test_seeds.py::test_tiny_cpu_resume_refuses_schedule_change_and_wrong_loader_position`
  (real tiny CPU runs, derived from `reviews/2026-10-14-recheck/repro/review3_resume_ga.py`).
* Replicates (`run_arm.py`): a protocol-v2 build requires an explicit `--data-loader prepacked`
  (corrected path; the preflight must exist and name the arm's current `prepack_dataset_id`
  before launch) or `--data-loader bestfit` (historical loader); without one the launcher
  refuses. With `--init-seed` the checkpoint destination is
  `arm_<A>/base_checkpoints/d<depth>_s<init_seed>` (`--model-tag` overrides), the run identity
  is stored in `arm_<A>/run_info_<tag>.json` and the evaluation goes to
  `eval/seed_<init_seed>/beliefs_arm<A>.csv`, so replicates never overwrite each other (recheck
  finding 6; `docs/ARTIFACTS.md` section 7). Export and upload accept the tag (repo
  `ctx_<A>_<tag>_base`).

## 8. Commands

```bash
PY=$HOME/.cache/ctx_cpu_venv/bin/python          # CPU env for audits / prepacking
# historical exposure of one arm (July-2026 configuration) and of two arms + cross-arm report
$PY contextualization/exposure_audit.py --exp-dir D --arm R --mode historical \
    --B 8 --T 2048 --grad-accum 64 --num-iterations 8352 [--world-size 1] [--out D/exposure]
$PY contextualization/exposure_audit.py --exp-dir D --arms R X --mode historical \
    --B 8 --T 2048 --grad-accum 64 --num-iterations 8352
# legacy manifest: supply the locator that regenerates insert_text per occurrence
$PY contextualization/exposure_audit.py --exp-dir D --arm R --mode historical ... --legacy-locator mypkg.locate:insert_text

# build the matched rows for all arms (staged; refuses to overwrite; refuses a tokenizer that
# encodes differently from the arm tokenizers; ends with the prepack gate)
$PY contextualization/exposure_audit.py --exp-dir D --arms C R X E --mode prepack --T 2048
# THE PREFLIGHT for the exact training configuration: gate + schedule + step agreement +
# PREFLIGHT_OK.json (exit 2 and no file otherwise); --arms must be every prepacked arm
$PY contextualization/exposure_audit.py --exp-dir D --arms C R X E --mode prepacked \
    --B 8 --T 2048 --grad-accum 64 --num-iterations 8352 --world-size 8 --data-seed 0 --check-hashes

# train one arm on the prepacked rows (same init/data seed in every arm of a replicate); base_train
# refuses unless D/exposure/PREFLIGHT_OK.json matches B/T/grad_accum/num_iterations/world_size/
# data_seed/dataset id/encoding (--skip-preflight-check bypasses it, loudly, recorded in the meta)
NANOCHAT_BASE_DIR=D/arm_R torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 --run=ctx_R_d24_s42 --model-tag d24_s42 --data-loader prepacked --init-seed 42 --data-seed 0
# or through run_arm.py (a v2 build needs --data-loader; --init-seed selects the model tag d24_s42;
# replicate tags need --skip-upload until ctx.tracking accepts model tags)
uv run python contextualization/run_arm.py --arm R --exp-dir D --depth 24 \
    --data-loader prepacked --init-seed 42 --data-seed 0 --skip-upload
```

Defaults are unchanged for legacy builds: without the new flags `base_train` initialises with
seed 42 and trains with the bestfit loader exactly as before (the two bestfit loader functions
are untouched); `run_arm.py` only insists on an explicit loader for protocol-v2 builds.

## 9. Limits

* Fixture-level verification only; no historical run was replayed in this session, and no GPU
  training was launched. The CPU smoke run of `scripts/base_train.py` (tiny model, both loaders,
  exact prepacked resume, loader-mismatch refusal) is described in the hand-off report.
* The historical audit assumes an uninterrupted run unless `resume_state` (the saved loader
  state of a resume) is given; every historical resume changed the stream from that point.
* The legacy locator is a substring search per shard (minutes per 250M-character shard when
  many occurrences share it); the historical manifests also lack `group_id`, which the exposure
  rows then leave empty.
* Prepacked rows pad instead of cropping inside inserts, so per-arm scored-token totals differ
  by the rendering lengths; the manifest reports them, the analysis must use them (and section 10
  explains what that does to the loss).
* `verify_prepack` reads every mask file of every arm (`check_masks`) and, with `--check-hashes`,
  every rows file too; at the full scale (about 2 M rows x 2,049 tokens per arm) that is a few GB
  of masks and about 17 GB of rows per arm, minutes, not hours. The loader's first-open hashing
  spreads the same read over the run.
* `ctx.tracking` export / upload accept replicate model tags (`resolve_checkpoint(exp_dir, arm,
  "d24_s7")`, repo `ctx_<A>_d24_s7_base`); run metadata and evaluations are scoped per tag
  (`docs/ARTIFACTS.md` section 7).

## 10. Loss normalization with masked rows: measurement and the decision of 2026-10-15

**Decision (2026-10-15): the funded C/R/X/E experiment uses fixed nominal normalization,
selected EXPLICITLY with `--loss-normalization fixed_nominal` on `exposure_audit.py --mode
prepacked` and on every `run_arm.py` / `scripts/base_train.py` invocation.** The historical
objective stays the default (`microbatch_mean`) for backward compatibility. Implementation
(`nanochat/loss_normalization.py`, `scripts/base_train.py`):

* `fixed_nominal`: for each micro-batch the cross-entropy is computed with `reduction="sum"` over
  the valid targets (`model(x, y, loss_reduction="sum")`, `ignore_index=-1`) and divided by
  `device_batch_size x max_seq_len x grad_accum_steps`. The accumulation division happens exactly
  once; the distributed optimizer averages gradients across ranks (`ReduceOp.AVG`), so there is no
  additional division by the world size. The sum is computed directly, never reconstructed as
  mean x active count: a fully masked micro-batch contributes exactly 0 (the historical mean is
  NaN there). Corresponding tokens get the same coefficient in every arm and every partition.
* Recorded and validated: `meta_<step>.json` carries `loss_normalization` and
  `loss_denominator_per_microbatch`; `PREFLIGHT_OK.json` carries `loss_normalization` and
  `loss_denominators` and `base_train` refuses a run whose policy differs from the certified one
  (`run_arm.py check_preflight` refuses it before launch); a resume refuses a policy change
  before the schedule and preflight checks; the eval meta propagates
  `checkpoint.loss_normalization` and `analyze_beliefs.py` refuses to pool evaluations of
  checkpoints trained under different objectives. **A checkpoint, preflight or evaluation that
  records no policy is the historical `microbatch_mean`; missing metadata never implies the new
  policy.**
* Logged per optimizer step, aggregated over all micro-batches and all ranks (one all-reduce):
  `train/loss_nominal` (the objective, `sum / (B x T x grad_accum x world_size)`) and
  `train/loss_per_active_token` (ordinary loss per valid target, `sum / active`), with
  `train/active_targets` / `train/nominal_targets`; the console line prints `nominal`,
  `per_active_tok` and `active/nominal`. Validation NLL / bpb stay conventionally normalized.
* Tests: `tests/test_loss_normalization.py` (unmasked equivalence with the historical mean;
  invariance to the micro-batch partition at a fixed token budget with unequal masking, which the
  mean lacks; fully masked micro-batches; distributed averaging without an extra division;
  unrecorded = historical) and `tests/test_seeds.py::test_tiny_cpu_fixed_nominal_is_certified_recorded_logged_and_bound_on_resume`
  (real tiny runs: preflight certification both ways, meta/log contents, matching and mismatched
  resumes incl. a historical checkpoint without the key). `smoke_crxe.py` trains its two seeds
  under `fixed_nominal` and exercises the launcher and resume refusals.

The rest of this section records the measured consequence of the masks under the historical
objective, which motivated the decision.

`scripts/base_train.py` computes, for each micro-batch, the cross-entropy MEAN over that
micro-batch's non-masked targets (`nanochat/gpt.py`, `ignore_index=-1`), divides by
`grad_accum_steps` and sums the gradients of the micro-batches of one step. A scored token
therefore carries the weight `1 / (active targets in its micro-batch * grad_accum_steps)`. With
the bestfit loader every micro-batch has exactly `B*T` targets and the three candidate objectives
below coincide; with the prepacked masks they do not: the arms have different numbers of active
targets in the shared rows (the X wrapper is longer than the R sentence; C and R pad the reserved
region with masked tokens), so the SAME base-corpus token is weighted differently across arms,
and the weighting depends on how rows are partitioned into micro-batches and ranks. The review's
toy example (four scored labels, split 1/3 across two micro-batches) gives gradients 0.2913
(`B=1, GA=2`) vs 0.7913 (`B=2, GA=1`); the test fixture has rows with 59 active targets in R
against 96 in X (ratio 1.63). Both are deliberately small; the real magnitude is measured per
build by the preflight.

What is measured (never estimated), in `prepack_manifest.json` and in the `loss_weighting` block
of `cross_arm.json` / `PREFLIGHT_OK.json` (`ctx.prepack.loss_weighting_report`):

* per arm: `active_targets_total` and `active_targets_per_row` (min / mean / max, histogram by
  32-token bins, rows with zero active targets) in the manifest; within the training horizon the
  total active targets, the per-step and per-micro-batch active counts (min / max / mean, number
  of micro-batches with zero active targets);
* per arm and per DDP rank (recheck 2026-10-14): the denominators the current objective actually
  uses under DDP. `base_train` takes the mean over the non-masked targets of each rank's own
  `(B, T)` micro-batch, divides by `grad_accum_steps`, and the distributed optimizer averages
  gradients across ranks (`ReduceOp.AVG`), so a scored token on rank r carries weight
  `1 / (active targets of rank r's micro-batch * grad_accum_steps * world_size)`. Rank r holds
  rows `[r*B, (r+1)*B)` of each micro-batch's slice of the epoch order.
  `per_arm[arm]["per_rank"][rank]` gives the distribution (and total) of these rank-local active
  counts over the horizon, `per_microbatch_rank_local` pools all `world_size * micro_batches_trained`
  of them, and `rank_local_microbatches_with_zero_active` counts rank-local micro-batches without
  any active target. `per_microbatch` (the sum over all `B * world_size` rows) is the denominator
  only when `world_size == 1`; `note_distributed` in the report states this
  (`tests/test_prepacked_loader.py::test_loss_weighting_reports_rank_local_denominators`);
* between arms, on the shared rows: the distribution of the per-row ratio of active counts
  (e.g. `X/R`: rows compared, rows that differ, min / mean / max, the ratio on differing rows
  only, rows active in one arm only) and the ratio of the horizon totals.

The candidate objectives:

1. **per-micro-batch mean (current, nanochat default).** Weight of a scored token:
   `1 / (n_active(micro-batch) * grad_accum_steps)`. Simple and unchanged from the historical
   runs, but arms with more active targets in a micro-batch weight each of its tokens less, and
   the weighting depends on the micro-batch / rank partition (not only on the data).
2. **global active-token mean.** Each step's loss is the sum over all its micro-batches and ranks
   of the per-token losses divided by the step's total active targets (an all-reduced count).
   Weight `1 / n_active(step)`: independent of the partition, still different between arms
   whenever their totals differ (by the rendering-length differences only, which the manifest
   reports).
3. **fixed denominator `B*T`.** Each micro-batch's loss is the sum of per-token losses divided by
   `B*T*grad_accum_steps` regardless of how many targets are masked. Weight `1 / (B*T*GA)`: the
   same constant in every arm and every partition, so identical base tokens get identical
   weights across arms; the effective learning rate of a masked row is lower by its masked
   fraction (the masks are a few percent of the tokens at most, see `masked_tokens_total`).

Option 3 is the one implemented as `fixed_nominal` (decision of 2026-10-15, top of this section);
the three coincide on unmasked rows (tested). Padding visible to attention and unequal active
totals remain documented design trade-offs of the prepacked path, not hidden bugs.
