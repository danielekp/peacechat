# Artifacts: caches, shards, manifests, archives

Written 2026-10-14. This page documents the operational repairs for review findings 19-22
(`reviews/2026-10-13/REVIEW.md`): what the builder may reuse, how it proves what it reused,
what an archived checkpoint must carry, and the commands. Everything here is implemented in
`ctx/basecorpus.py`, `ctx/nanochat_io.py`, `ctx/tracking.py`, `load_artifact_hf.py` and
`run_arm.py`; the offline tests are in `tests/test_artifacts.py`.

Nothing in these modules touches the network unless tracking is enabled *and* an upload
function is called; `--manifest-only`, `export_manifest`, `export_build_manifest` and
`refresh_manifest_gz` are purely local.

## 1. Base-corpus cache (`ctx/basecorpus.py`)

### Layout and keys

A cache used to be keyed by shard filename only, so `ensure_shard("synthetic", 0, seed=1)`
followed by `ensure_shard("fineweb", 0, seed=999)` returned the same synthetic file
(finding 22, reproduced in `reviews/2026-10-13/reproduce_dataset_findings.py`). Shards now
live under a key that encodes the source and every parameter that determines the bytes:

```
<cache_dir>/
  synthetic-seed<seed>-chars<chars_per_shard>-docs<docs_per_shard>-rg<row_group_size>/
      shard_00000.parquet ... checksums.json
  fineweb-edu-100b-shuffle-rev<revision>/            # revision sanitized: refs/pr/1 -> refs_pr_1
      shard_00000.parquet ... checksums.json
```

- `cache_key(source, **cfg)` returns the directory name; unrelated kwargs are ignored so the
  builder can pass its whole `shard_kw` dict. `shard_cache_path(source, index, cache_dir, **cfg)`
  is the file path `ensure_shard` will return.
- A synthetic shard can never be returned for a FineWeb request: different key.
- Same seed and sizes give the same synthetic bytes as before (generation is unchanged).

### `checksums.json`

Every file in a keyed directory has a record, written atomically right after the file:

```json
{"shard_00000.parquet": {"sha256": "...", "x_repo_commit": "<dataset commit>", "url": "...",
                         "bytes": 123, "revision": "main", "provenance": "downloaded", ...}}
```

`x_repo_commit` is the `X-Repo-Commit` header the Hub returns on `resolve` URLs: the dataset
commit the bytes came from. So even a cache keyed by the moving ref `main` records which
commit each shard is; if a later download under the same key comes from a different commit,
a WARNING names both commits. `provenance` is one of `downloaded`, `generated`,
`regenerated`, `adopted:expected_sha256`, `legacy:<path>`. `load_checksums(cache_dir, key)`
reads the file.

### Revision pinning and verification

`ensure_shard(source, index, cache_dir, *, seed, chars_per_shard, synthetic_docs_per_shard,
row_group_size, revision=None, expected_sha256=None, verify=False, legacy_cache_dir=None,
legacy_cache_sha256=None)`

| situation | behaviour |
|---|---|
| file absent, `fineweb` | download `https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/<revision>/shard_XXXXX.parquet` (default revision `main`) to `<path>.tmp`, hash while streaming, `os.replace`, record |
| file absent, `synthetic` | generate (atomic write), record sha256 |
| `expected_sha256` given, fresh download / generation differs | file deleted, `ChecksumMismatch` |
| file cached with record | size compared with the record (always); with `verify=True` the sha256 is recomputed. A wrong size or hash deletes the file, drops the record and raises `ChecksumMismatch`; rerunning re-downloads |
| file cached, `expected_sha256` differs from the recorded hash | `ChecksumMismatch`, file kept (it is intact, it is just not the pinned content: use a key for the pinned revision instead) |
| file cached, no record, `fineweb` | `CacheIntegrityError` (unknown provenance) unless `expected_sha256` matches, which adopts it |
| file cached, no record, `synthetic` | regenerated deterministically and recorded |

Pin a build: `rev = resolve_fineweb_revision("main")` returns the commit sha behind `main`
(HEAD request, API fallback; `None` on any error, never raises) and `revision=rev` keys the
cache by that commit. `fineweb_remote_sha256(index, revision)` reads the sha256 the Hub
advertises for an LFS file (`X-Linked-Etag`) so `expected_sha256` can be set before
downloading. Both are best effort and were only exercised against monkeypatched `requests`
in the tests; they have not been run against the live Hub from this checkout.
`build_dataset.py --fineweb-revision <sha>` forwards the revision (builder rewrite).

### Legacy flat caches

The pre-finding-22 cache was a flat `<dir>/shard_XXXXX.parquet` (e.g. the July build's
`_base_cache`). Such a directory cannot prove what it holds: a synthetic shard or a shard of
another revision could sit under a FineWeb name. `ensure_shard(..., legacy_cache_dir=<dir>,
legacy_cache_sha256={filename: sha256})` therefore reuses a legacy file only for
`source == "fineweb"`, only when the caller asserts its sha256, and only after hashing it
and finding the assertion true; the file is hard-linked (or copied) into the keyed cache
and recorded with `provenance: legacy:<path>`. A wrong assertion raises and leaves the
legacy file untouched. `legacy_cache_dir` without the dict is a `ValueError`; for
`synthetic` the arguments are ignored (regeneration is cheap and deterministic).

## 2. Shard writing (`ctx/nanochat_io.py`)

- `write_shard(docs, path)` writes `<path>.tmp` and `os.replace`s it; a file with the final
  name is always complete. It returns the sha256 of the written bytes.
- `ShardWriter(out_dir, chars_per_shard, row_group_size, overwrite=False)` raises
  `FileExistsError` when `out_dir` already holds `*.parquet` files (finding 21: the old
  writer deleted them at construction). `overwrite=True` deletes them and prints what it
  deleted. The builder writes into a staging directory and versions builds by directory;
  `build_dataset.py --overwrite` is the only path that deletes an existing build.
- `add(doc)`, `add_returning(doc) -> shard_index`, `add_indexed(doc) -> (shard_index,
  row_in_shard)`; `finalize(val_docs) -> {filename: sha256}` writes the val shard last and
  returns the hashes of every shard (also kept as `writer.shard_hashes`). The v2
  `build_summary.json` stores them under `shard_hashes[arm]`.
- `shard_sha256(path)` streams a file hash; `read_shard_texts`, `shard_num_rows` unchanged.

## 3. Checkpoint bindings (`scripts/base_train.py` -> `meta_<step>.json`)

The corrected trainer writes a `bindings` block into every `meta_<step>.json`:
`tokenizer_sha256` (sha256 of `tokenizer/tokenizer.pkl`), `build_summary_sha256`,
`probe_hashes`, `prepack_manifest_sha256`, `data_listing_sha256`, `init_seed`, `data_seed`,
`data_loader`, `code` (`git_commit`, `git_dirty`, ...). Legacy metas (July 2026) have no
such block.

`ctx/tracking.py` uses it:

- `tokenizer_binding(meta, tok_dir)` compares the bound hash with the tokenizer on disk ->
  `ok` / `mismatch` / `unbound`.
- `check_tokenizer_binding(...)`: `mismatch` -> `SystemExit` naming both hashes; `unbound` ->
  prints `WARNING: checkpoint not bound to a tokenizer (legacy meta)` and passes only with
  `allow_unbound=True`. `hf_upload_checkpoint` runs this before anything that can reach the
  network (the HF namespace lookup comes after it).

## 4. Export manifests (local, no network)

`export_manifest(exp_dir, arm, depth, with_optimizer=False, step=None)` writes
`<exp_dir>/arm_<ARM>/export/checkpoint_manifest.json`:

```
schema_version 1, kind, arm, depth, step, exp_dir, exp_tag, checkpoint_dir
files              [{name, bytes, sha256}] for model_<step>.pt, meta_<step>.json, tokenizer/*
                   (+ optim_<step>_rank*.pt when included); the model hash is required and computed
optimizer_requested, optimizer_included, optimizer_files, optimizer_files_on_disk
tokenizer          ctx.protocol.tokenizer_fingerprint(arm_<ARM>/tokenizer)
tokenizer_binding  {status, meta_tokenizer_sha256, dir_tokenizer_sha256}; bound (bool)
bindings           the meta's bindings block, copied verbatim (null for legacy metas)
run_id, wandb_run, wandb_run_url, val_bpb, model_config, training_user_config
code               ctx.protocol.code_revision() of the exporting checkout; created_at
suggested          hf_repo_id (<HF_NAMESPACE>/ctx_<ARM>_d<DEPTH>_base), hf_tags [wandb-<run_id>, <exp_tag>],
                   wandb_run_id/name/project (from run_info.json), upload_command
```

`optimizer_included` is true only when the optimizer was requested *and* the shards exist.
`hf_upload_checkpoint` writes the same manifest (plus `hf_repo_id`) into the uploaded folder
as `checkpoint_manifest.json`; the model card states whether optimizer state is included,
the tokenizer sha256 and the binding status.

`export_build_manifest(exp_dir)` writes `<exp_dir>/export/build_manifest.json`: hashes of
`build_summary.json`, `manifest.csv`, `manifest.csv.gz`, `manifest.csv.sha256`,
`shard_hashes.json`, every `probe_sets/*.jsonl` and every file under `exposure/`, per-arm
shard counts/bytes and tokenizer fingerprints, a few summary fields (protocol, builder
version, seed, whether `shard_hashes` / `input_hashes` are present) and the suggested
dataset path `<HF_NAMESPACE>/ctx_experiment_build/<exp_tag>/` with tag `<exp_tag>`.
It refreshes `manifest.csv.gz` first (next section) so the hashes describe one consistent set.

Verify an archive later: compare each `files[].sha256` with `sha256sum` of the file; for a
checkpoint also compare `tokenizer.tokenizer_sha256` with `bindings.tokenizer_sha256`.

## 5. `manifest.csv.gz` refresh rule

The old rule gzipped `manifest.csv` only when the `.gz` was missing or older by mtime, so a
rebuild in the same directory uploaded a stale manifest (finding 20). `refresh_manifest_gz(exp_dir)`
now compares the sha256 of the **decompressed** existing `.gz` with the sha256 of
`manifest.csv` and rewrites the `.gz` whenever they differ (also when the `.gz` is
unreadable). The gzip is written with `mtime=0` and no filename header, so identical input
gives identical bytes, through a `.tmp` + `os.replace`. `manifest.csv.sha256`
(`sha256sum` format) is written next to it. `hf_upload_build` and `export_build_manifest`
call it; it returns `{manifest, gz, sha256, gz_sha256, refreshed}` or `None` without a
manifest.

## 6. HF uploads (`ctx/tracking.py`)

- `hf_upload_checkpoint(exp_dir, arm, depth, run_id=None, step=None, with_optimizer=False,
  extra_tags=(), allow_unbound=False)`: resolve the checkpoint (`--step` or the latest),
  check the tokenizer binding, build the manifest (hashes the model file), stage
  `model_<step>.pt` (symlink), `meta_<step>.json`, `tokenizer/`, optimizer shards when
  included, `checkpoint_manifest.json`, `README.md`; upload to `<ns>/ctx_<ARM>_d<DEPTH>_base`
  with `delete_patterns` so `main` holds exactly one checkpoint, and tag `wandb-<run_id>` +
  `<exp_tag>`. Older runs stay at their tags. hf_xet progress reporting is unchanged
  (hub 0.34 and 1.x paths).
- `hf_upload_build(exp_dir)`: refresh the gz, write the build manifest, upload `probe_sets/`,
  `build_summary.json`, `manifest.csv.gz`, `manifest.csv.sha256`, `shard_hashes.json`,
  `exposure/` (when present) and `build_manifest.json` to `<ns>/ctx_experiment_build/<exp_tag>/`,
  tag `<exp_tag>`. The README lists arms C, R, X and E.

### Optimizer policy

Optimizer shards (`optim_<step>_rank*.pt`, several GB per arm) are uploaded only with
`--with-optimizer` (`run_arm.py` / `load_artifact_hf.py`). The manifest and the model card
always state whether they are included; without them a repo cannot resume training, it
can only be evaluated. The July-2026 archives (`run1-2026-10` tag) contain model weights only.

## 7. `run_arm.py` provenance (finding 19)

`run_arm.py` forwards `--init-seed`, `--data-seed` and `--data-loader {bestfit,prepacked}` to
`scripts.base_train` with exactly those names (only when given; otherwise base_train's
defaults apply). `build_train_command(...)` is the single place that assembles the command
(torchrun only with more than one GPU). Before launching, `run_info.json` receives
`train_command` (list) / `train_command_str`, `train_cwd`, `train_env` (`NANOCHAT_BASE_DIR`,
`WANDB_RUN_ID`, `WANDB_TAGS`, `WANDB_NOTES`), `init_seed`, `data_seed`, `data_loader` and
`code` (`git_commit`, `git_dirty`, `git_branch`, `argv`, `python`), so even a crashed run is
described. After training, `export_manifest` runs locally before any upload (also with
`--skip-upload`); `--allow-unbound` is passed through to the upload for legacy checkpoints.

## 8. Commands

```bash
# local manifests only: no network, no HF_TOKEN
uv run python contextualization/load_artifact_hf.py --exp-dir ~/ctx_experiment_v3 --arm C --step 8352 --manifest-only
uv run python contextualization/load_artifact_hf.py --exp-dir ~/ctx_experiment_v3 --build --manifest-only

# archive a checkpoint (writes the manifest first, then uploads; refuses a mismatching tokenizer)
uv run python contextualization/load_artifact_hf.py --exp-dir ~/ctx_experiment_v3 --arm C --depth 24 --step 8352 \
    --wandb-run <run_id> [--with-optimizer]
# legacy checkpoint whose meta has no bindings block
uv run python contextualization/load_artifact_hf.py --exp-dir /workspace/ctx_experiment_v2 --arm X --allow-unbound

# build metadata (refreshes manifest.csv.gz by content)
uv run python contextualization/load_artifact_hf.py --exp-dir ~/ctx_experiment_v3 --build

# one arm end to end with explicit, logged seeds and the prepacked loader
uv run python contextualization/run_arm.py --arm R --exp-dir ~/ctx_experiment_v3 --depth 24 \
    --init-seed 7 --data-seed 11 --data-loader prepacked [--with-optimizer] [--allow-unbound]

# pin FineWeb for a build
uv run python -c "from ctx.basecorpus import resolve_fineweb_revision as r; print(r('main'))"
uv run python build_dataset.py --out ~/ctx_experiment_v3 --fineweb-revision <sha> ...

# tests (offline; downloads and the Hub are monkeypatched)
$HOME/.cache/ctx_cpu_venv/bin/python -m pytest contextualization/tests/test_artifacts.py -q
```

Fetching an archived checkpoint and checking it against its manifest:

```bash
hf download Daniele/ctx_C_d24_base --revision wandb-<run_id> --local-dir /tmp/ckpt
python - <<'PY'
import json, hashlib, pathlib
root = pathlib.Path("/tmp/ckpt"); m = json.load(open(root / "checkpoint_manifest.json"))
for f in m["files"]:
    h = hashlib.sha256((root / f["name"]).read_bytes()).hexdigest()
    print("ok " if h == f["sha256"] else "BAD", f["name"])
print("tokenizer bound:", m["tokenizer_binding"])
PY
```
