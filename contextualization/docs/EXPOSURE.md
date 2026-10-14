# Training exposure: loader replay, matched prepacking, seeds

Written 2026-10-14 for the repairs of REVIEW 2026-10-13 findings 3 (equal corpus counts do not
establish equal training exposure), 4 (the loader's epoch counter is not an exposure count, and
resumes are approximate) and 7 (a dataset seed is not an initialization seed). Formats are fixed
in `docs/SCHEMAS_V2.md` ("Exposure index v1", "Prepacked rows", loader state). Code:

| piece | file |
|---|---|
| tagged replay of the historical loader + occurrence audit | `contextualization/ctx/exposure.py` |
| rows planned once across arms, schedule helpers, prepack audit | `contextualization/ctx/prepack.py` |
| CLI for all of the above | `contextualization/exposure_audit.py` |
| opt-in prepacked loader (bestfit loaders untouched) | `nanochat/dataloader.py` |
| `compute_init(device_type, seed=42)` | `nanochat/common.py` |
| `--init-seed`, `--data-seed`, `--data-loader` + checkpoint provenance | `scripts/base_train.py` |
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
packing statistics, `data_seed`, `code`) and `occurrence_exposure.csv`, the prepack-time
exposure index: a status for every occurrence with `epoch/micro_batch/step = -1` (the schedule
depends on the training configuration, see below). `prepack_experiment` ends by running
`verify_prepack`, which raises unless every occurrence is `complete` in every arm and has the
same `row_index` in every arm; the CLI exits with code 2 on any such failure.

## 5. The prepacked loader (`nanochat.dataloader.prepacked_distributed_data_loader_with_state`)

`prepacked_distributed_data_loader_with_state(B, T, split, device, resume_state_dict=None,
data_seed=-1, prepacked_dir=None, tokenizer=None)` reads `<base_dir>/prepacked` (train split).

* Row order per epoch (epochs are 0-based): identity when `data_seed < 0`, else
  `numpy.random.default_rng(data_seed * 1000003 + epoch).permutation(n_rows)`
  (`prepacked_epoch_order`; `ctx.prepack.epoch_order` is the same function, tested equal).
* Each epoch yields `n_rows // (B * world_size)` micro-batches; micro-batch `i` covers positions
  `[i*B*W, (i+1)*B*W)` of the epoch order and rank `r` takes rows `[r*B, (r+1)*B)` of that
  slice. The incomplete tail of an epoch's order is skipped (with identity order these are
  always the last `n_rows mod B*W` rows; the exposure index marks them `not_in_horizon`).
* `inputs = rows[:, :-1]`, `targets = rows[:, 1:]` with `-1` where `mask[:, 1:] == 0`.
* State: `{"loader": "prepacked", "schema_version": 1, "epoch", "cursor", "data_seed",
  "world_size", "B"}` with `cursor` the epoch-order position of the first row of the batch being
  yielded. `base_train` saves the state of the prefetched micro-batch, so resuming replays that
  batch first and the resumed token stream equals the uninterrupted one (tested at several
  cursors, mid-epoch and across the epoch boundary, world size 1 and 2). This is the difference
  from the bestfit loader, whose resume drops buffered documents and skips a row group.
* A state with a different `B`, `world_size` or `data_seed`, or a bestfit state, is refused; the
  manifest's `T` must equal the requested `T`.
* `split == "val"` falls back to the bestfit val loader over the last parquet shard (the
  prepacked set holds the train split only), so val bpb stays comparable with bestfit runs.

`scripts/base_train.py --data-loader prepacked` uses it for training; validation stays on the
bestfit loader; a checkpoint written with one loader cannot be resumed with the other.

## 6. Scheduled doses become scored doses (`exposure_audit.py --mode prepacked`)

The prepack-time index says where each occurrence sits (`row_index`, token spans, status). For a
training configuration, `ctx.prepack.step_of_row(row_index, epoch, data_seed, n_rows, B,
world_size, grad_accum_steps)` gives the micro-batch, optimizer step, rank and row-in-batch at
which that row is trained in that epoch (-1 in an epoch's skipped tail), and
`schedule_exposure` expands the index into `exposure/<arm>/occurrence_exposure.csv` with one row
per epoch the loader reaches up to the prefetched micro-batch: `complete` with its `step` inside
the horizon, `not_in_horizon` beyond it. `summary.json` reports the same quantities as the
historical audit plus the prepack totals, and the CLI fails (exit 2) unless `verify_prepack`
passes and every occurrence has the same row/step in every arm. The exposure of an occurrence is
thus determined by the schedule alone, identically for all arms; nothing is estimated from
frequency x epochs.

## 7. Seeds (`--init-seed`, `--data-seed`)

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
  bitwise deterministic) and `bindings` (sha256 of `tokenizer.pkl`, the data listing, the
  prepack manifest, `build_summary.json` and the probe files). The existing `wandb_run` block is
  kept; the prepacked loader state carries `"loader": "prepacked"`.

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

# build the matched rows for all arms (writes D/arm_<A>/prepacked/, refuses to overwrite)
$PY contextualization/exposure_audit.py --exp-dir D --arms C R X E --mode prepack --T 2048
# exposure index for the planned training configuration (+ audit, + cross_arm.json)
$PY contextualization/exposure_audit.py --exp-dir D --arms R X E --mode prepacked \
    --B 8 --T 2048 --grad-accum 64 --num-iterations 8352 --data-seed 0

# train one arm on the prepacked rows (same init/data seed in every arm of a replicate)
NANOCHAT_BASE_DIR=D/arm_R torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 --run=ctx_R_d24 --data-loader prepacked --init-seed 42 --data-seed 0
# or through run_arm.py (flags after `--` reach base_train)
uv run python contextualization/run_arm.py --arm R --exp-dir D --depth 24 -- \
    --data-loader=prepacked --init-seed=42 --data-seed=0
```

Defaults are unchanged: without the new flags `base_train` initialises with seed 42 and trains
with the bestfit loader exactly as before (the two bestfit loader functions are untouched).

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
  by the rendering lengths; the manifest reports them, the analysis must use them.
