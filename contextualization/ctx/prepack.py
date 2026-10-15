"""Matched, prepacked training rows for the C/R/X/E arms (docs/SCHEMAS_V2.md, "Prepacked rows").

The historical loader packs each arm independently, so the same inserted claim can be scored in R
and cropped away in X (REVIEW finding 3). Here the rows are planned ONCE over the lockstep document
streams of all arms and then realised per arm:

* base documents are byte-identical across arms (verified while streaming) and tokenized once;
* an intervention document (a ``manifest.csv`` v2 slot document, identified by ``doc_index``) is
  reserved at the max token length over arms, so it occupies the same row positions everywhere;
* the best-fit / crop-shortest algorithm of the historical loader is kept, except that a crop
  prefers a non-intervention document and an intervention document is only ever cropped at or
  after the end of its inserted sentence (max over arms), never inside it;
* arms whose version is shorter than the reserved length are padded with masked ``pad_token_id``
  (default: BOS), so the loss masks are identical outside reserved regions and the scored token
  counts are reported per arm;
* a claim whose insert cannot fit a row at all raises ``PrepackError`` naming the occurrence.

Outputs per arm under ``arm_<A>/prepacked/``: ``rows_XXXXX.npy`` (int32 ``[n, T+1]``),
``mask_XXXXX.npy`` (uint8), ``prepack_manifest.json`` and ``occurrence_exposure.csv`` (Exposure
index v1 with a status for every occurrence; ``epoch``/``micro_batch``/``step`` are filled in by
``exposure_audit.py --mode prepacked`` from the loader's deterministic schedule, see
``step_of_row``). ``nanochat.dataloader.prepacked_distributed_data_loader_with_state`` consumes
the rows; ``epoch_order`` here must equal ``prepacked_epoch_order`` there (tested).

Bindings and the gate (REVIEW 2026-10-14, findings 3, 4, 16, 21):

* ``encoding_fingerprint(tokenizer)`` identifies the encoding ACTUALLY used (tiktoken pattern,
  merge ranks, special tokens, BOS id); the manifest records it next to the sha256 of the arm's
  ``tokenizer.pkl`` (``tokenizer_file_sha256``). An arm ``tokenizer.pkl`` whose encoding differs
  from the tokenizer passed to ``prepack_experiment`` is a ``PrepackError`` (no silent override).
* every row/mask file is hashed (``sha256`` / ``mask_sha256``) and counted (``active_targets`` =
  ``mask[:, 1:].sum()``); ``prepack_dataset_id`` is the sha256 over ``(T, n_rows, ordered
  (rows sha256, mask sha256) pairs)`` and binds loader states and checkpoints to the rows.
* ``verify_prepack`` is the gate: expected occurrence sets from ``manifest.csv`` (each key exactly
  once per intervention arm, C's control slots reported separately), per-file counts (and hashes
  with ``check_hashes=True``), the claim span of every occurrence scored in the actual mask, the
  reserved region outside an arm's tokens masked with pads, masks identical across arms outside
  reserved regions, dataset ids and the arm tokenizers' fingerprints.
* the output is staged (``prepacked.staging-<pid>``) and moved into place for ALL arms only after
  the gate passed; a failure removes the staging dirs and leaves any existing output untouched.
* ``active_targets_per_row`` / ``loss_weighting_report`` MEASURE how the per-micro-batch mean of
  ``base_train`` weights tokens when active counts differ between arms (docs/EXPOSURE.md, "Open
  experimental decision"); nothing here changes the objective.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow.parquet as pq

from .exposure import (ExposureError, Occurrence, STATUSES, as_layout, exposure_row, load_manifest_rows,
                       manifest_is_v2, manifest_path, occurrences_from_v2_rows, read_exposure_csv,
                       summarize_exposure, token_byte_table, token_spans_for, write_exposure_csv)
from .protocol import (EXPOSURE_SCHEMA, PREPACK_SCHEMA, ProtocolError, code_revision, load_summary, protocol_of,
                       sha256_file, sha256_json)

ACTIVE_TARGET_BIN = 32   # histogram bin width (tokens) of active_targets_per_row in the manifest


class PrepackError(RuntimeError):
    """A prepack invariant cannot be satisfied (or the audit of a prepacked set failed)."""


# ======================================================================================
# tokenizer / dataset identity

def encoding_fingerprint(tokenizer) -> str:
    """sha256 of a canonical serialization of the encoding actually used (see
    ``nanochat.dataloader.encoding_fingerprint``: the single implementation lives in nanochat because
    the loader must verify the binding without importing this package)."""
    from nanochat.dataloader import encoding_fingerprint as _fp
    return _fp(tokenizer)


def tokenizer_dir_fingerprint(tokenizer_dir) -> dict:
    """``{"tokenizer_file_sha256", "encoding_fingerprint"}`` of ``<dir>/tokenizer.pkl`` (the pickle is
    LOADED and fingerprinted, so the result describes what that file encodes, not just its bytes)."""
    from nanochat.tokenizer import RustBPETokenizer
    d = Path(tokenizer_dir)
    pkl = d / "tokenizer.pkl"
    if not pkl.exists():
        raise PrepackError(f"no tokenizer.pkl under {d}")
    tok = RustBPETokenizer.from_directory(str(d))
    return {"tokenizer_file_sha256": sha256_file(pkl), "encoding_fingerprint": encoding_fingerprint(tok)}


def dataset_id_of(T, n_rows, row_files) -> str:
    """``prepack_dataset_id``: sha256 over (T, n_rows, ordered (rows sha256, mask sha256) pairs)."""
    pairs = []
    for rf in row_files:
        if not rf.get("sha256") or not rf.get("mask_sha256"):
            raise PrepackError(f"row file {rf.get('file')} has no sha256 / mask_sha256 recorded; cannot derive a dataset id")
        pairs.append([rf["sha256"], rf["mask_sha256"]])
    return sha256_json({"prepack_dataset_id": 1, "T": int(T), "n_rows": int(n_rows), "row_file_sha256_pairs": pairs})


def active_targets_summary(counts) -> dict:
    """min / mean / max and a histogram by ``ACTIVE_TARGET_BIN``-token bins of per-row active target counts."""
    c = np.asarray(counts, dtype=np.int64)
    if len(c) == 0:
        return {"n_rows": 0, "min": 0, "mean": 0.0, "max": 0, "bin_size": ACTIVE_TARGET_BIN, "histogram": {}}
    bins = (c // ACTIVE_TARGET_BIN).astype(np.int64)
    hist = {}
    for b, n in zip(*np.unique(bins, return_counts=True)):
        lo = int(b) * ACTIVE_TARGET_BIN
        hist[f"{lo}-{lo + ACTIVE_TARGET_BIN - 1}"] = int(n)
    return {"n_rows": int(len(c)), "min": int(c.min()), "mean": float(c.mean()), "max": int(c.max()),
            "bin_size": ACTIVE_TARGET_BIN, "histogram": hist, "rows_with_zero_active": int((c == 0).sum())}


# ======================================================================================
# schedule: row order per epoch and the (micro-batch, step) a row is trained at

def epoch_order(data_seed: int, epoch: int, n_rows: int) -> np.ndarray:
    """Row order of one epoch (0-based epochs): identity when ``data_seed < 0``, else
    ``numpy.random.default_rng(data_seed * 1000003 + epoch).permutation(n_rows)``. Must stay equal
    to ``nanochat.dataloader.prepacked_epoch_order``."""
    if data_seed is None or int(data_seed) < 0:
        return np.arange(n_rows, dtype=np.int64)
    return np.random.default_rng(int(data_seed) * 1000003 + int(epoch)).permutation(n_rows).astype(np.int64)


class RowSchedule(NamedTuple):
    micro_batch: int    # global micro-batch index counted from the start of training (-1 = never)
    step: int           # optimizer step (= micro_batch // grad_accum_steps), -1 = never
    rank: int           # DDP rank whose device batch holds the row
    row_in_batch: int   # position inside that rank's (B, T) batch
    position: int       # position of the row in the permuted epoch


def schedule_of_rows(row_indices, epoch, data_seed, n_rows, B, world_size, grad_accum_steps):
    """Vectorised ``step_of_row``: arrays (micro_batch, step, rank, row_in_batch, position).

    Rows beyond ``(n_rows // (B*world_size)) * B*world_size`` in an epoch's order are skipped by the
    loader (incomplete trailing micro-batch) and get micro_batch = step = -1 for that epoch."""
    row_indices = np.asarray(row_indices, dtype=np.int64)
    rows_per_micro = int(B) * int(world_size)
    micro_per_epoch = int(n_rows) // rows_per_micro
    order = epoch_order(data_seed, epoch, n_rows)
    inverse = np.empty(n_rows, dtype=np.int64)
    inverse[order] = np.arange(n_rows, dtype=np.int64)
    position = inverse[row_indices]
    micro_in_epoch = position // rows_per_micro
    valid = micro_in_epoch < micro_per_epoch
    micro = np.where(valid, int(epoch) * micro_per_epoch + micro_in_epoch, -1)
    step = np.where(valid, micro // int(grad_accum_steps), -1)
    within = position % rows_per_micro
    rank = within // int(B)
    row_in_batch = within % int(B)
    return micro, step, rank, row_in_batch, position


def step_of_row(row_index, epoch, data_seed, n_rows, B, world_size, grad_accum_steps) -> RowSchedule:
    """Where (and whether) prepacked row ``row_index`` is trained in ``epoch`` (0-based)."""
    m, s, r, rb, p = schedule_of_rows([row_index], epoch, data_seed, n_rows, B, world_size, grad_accum_steps)
    return RowSchedule(int(m[0]), int(s[0]), int(r[0]), int(rb[0]), int(p[0]))


# ======================================================================================
# planning documents

@dataclass
class _PlanDoc:
    doc_index: int
    reserved: int                 # planning length = max token length over arms (incl. BOS)
    tokens: dict                  # "*" -> shared tokens, or arm -> tokens (intervention docs)
    intervention: bool
    min_keep: int                 # intervention docs: max over arms/occurrences of insert token end
    spans: dict = field(default_factory=dict)   # arm -> {occurrence key: TokenSpans}

    def tokens_for(self, arm):
        return self.tokens["*"] if "*" in self.tokens else self.tokens[arm]


def iter_arm_docs(layout):
    """Texts of a train split in doc_index order."""
    for p in layout.paths:
        pf = pq.ParquetFile(p)
        for rg in range(pf.num_row_groups):
            yield from pf.read_row_group(rg).column("text").to_pylist()


class _RowWriter:
    """Accumulates per-arm rows/masks and writes rows_XXXXX.npy / mask_XXXXX.npy chunks (hashed and counted)."""

    def __init__(self, arms, out_dirs, row_capacity, rows_per_file, pad):
        self.arms, self.out_dirs, self.cap, self.rows_per_file, self.pad = list(arms), out_dirs, row_capacity, rows_per_file, pad
        self.pending = {a: ([], []) for a in self.arms}
        self.row_files = {a: [] for a in self.arms}
        self.scored = {a: 0 for a in self.arms}
        self.active_per_row = {a: [] for a in self.arms}
        self.n_rows = 0

    def new_row(self):
        return ({a: np.full(self.cap, self.pad, dtype=np.int32) for a in self.arms},
                {a: np.zeros(self.cap, dtype=np.uint8) for a in self.arms})

    def add(self, rows, masks):
        for a in self.arms:
            masks[a][0] = 0  # position 0 is never a target
            self.pending[a][0].append(rows[a])
            self.pending[a][1].append(masks[a])
            n_active = int(masks[a][1:].sum())
            self.scored[a] += n_active
            self.active_per_row[a].append(n_active)
        self.n_rows += 1
        if len(self.pending[self.arms[0]][0]) >= self.rows_per_file:
            self.flush()

    def flush(self):
        if not self.pending[self.arms[0]][0]:
            return
        i = len(self.row_files[self.arms[0]])
        for a in self.arms:
            rows = np.stack(self.pending[a][0]).astype(np.int32)
            masks = np.stack(self.pending[a][1]).astype(np.uint8)
            rf, mf = f"rows_{i:05d}.npy", f"mask_{i:05d}.npy"
            np.save(self.out_dirs[a] / rf, rows)
            np.save(self.out_dirs[a] / mf, masks)
            self.row_files[a].append({"file": rf, "n_rows": int(rows.shape[0]), "sha256": sha256_file(self.out_dirs[a] / rf),
                                      "mask_file": mf, "mask_sha256": sha256_file(self.out_dirs[a] / mf),
                                      "active_targets": int(masks[:, 1:].sum())})
            self.pending[a] = ([], [])


PREPACK_OUTPUT_NAMES = ("prepack_manifest.json", "occurrence_exposure.csv")


def _is_prepack_output(f: Path) -> bool:
    return (f.name.startswith(("rows_", "mask_")) and f.suffix == ".npy") or f.name in PREPACK_OUTPUT_NAMES


def _check_out_dir_free(d: Path, overwrite: bool):
    """Refuse an existing, non-empty output dir unless ``overwrite``; nothing is deleted here."""
    if d.exists() and any(d.iterdir()):
        if not overwrite:
            raise PrepackError(f"{d} already exists and is not empty (pass overwrite=True to replace it)")
        foreign = [f.name for f in d.iterdir() if not _is_prepack_output(f)]
        if foreign:
            raise PrepackError(f"{d} contains files that are not prepack outputs ({foreign[:5]}); remove them before overwriting")


def _replace_dir(staging: Path, final: Path):
    """Move a verified staging dir into place (the old output, if any, holds prepack outputs only)."""
    if final.exists():
        shutil.rmtree(final)
    os.rename(staging, final)


# ======================================================================================
# the planner / realiser

def prepack_experiment(exp_dir, arms, tokenizer, T, *, out_subdir="prepacked", buffer_size=1000, pad_token_id=None,
                       data_seed=-1, rows_per_file=4096, tokenizer_batch_size=128, tokenizer_threads=4,
                       data_subdir="base_data", overwrite=False, progress=None) -> dict:
    """Plan rows once across ``arms`` and realise them per arm (see module docstring).

    Returns ``{"arms": {arm: manifest dict}, "n_rows": n, "verify": report, "out_dirs": {...}}``.
    Raises ``PrepackError`` when an arm's ``tokenizer/tokenizer.pkl`` encodes differently from
    ``tokenizer``, when a claim cannot fit a row, when a non-intervention document differs across
    arms, when an occurrence's doc_index differs across arms, or when the realised rows fail
    ``verify_prepack``. The output is written to ``arm_<A>/<out_subdir>.staging-<pid>`` and moved
    into place for all arms only after the audit passed (a failure removes the staging dirs)."""
    exp_dir = Path(exp_dir)
    arms = list(arms)
    if not arms:
        raise PrepackError("no arms given")
    row_capacity = int(T) + 1
    bos = tokenizer.get_bos_token_id()
    pad = bos if pad_token_id is None else int(pad_token_id)
    byte_table = token_byte_table(tokenizer)

    # --- tokenizer binding: the arm tokenizers must encode exactly like the tokenizer used here ---
    enc_fp = encoding_fingerprint(tokenizer)
    tok_binding = {}
    for a in arms:
        tdir = exp_dir / f"arm_{a}" / "tokenizer"
        if (tdir / "tokenizer.pkl").exists():
            fp = tokenizer_dir_fingerprint(tdir)
            if fp["encoding_fingerprint"] != enc_fp:
                raise PrepackError(f"arm {a}: {tdir / 'tokenizer.pkl'} (sha256 {fp['tokenizer_file_sha256'][:12]}) encodes with "
                                   f"fingerprint {fp['encoding_fingerprint'][:12]} but the tokenizer used for prepacking has "
                                   f"{enc_fp[:12]}; refusing to certify rows for a tokenizer that did not produce them")
            tok_binding[a] = fp
        else:
            tok_binding[a] = {"tokenizer_file_sha256": None, "encoding_fingerprint": enc_fp}
    tok_file_shas = {v["tokenizer_file_sha256"] for v in tok_binding.values() if v["tokenizer_file_sha256"]}
    if len(tok_file_shas) > 1:
        raise PrepackError(f"arms carry different tokenizer.pkl files: { {a: v['tokenizer_file_sha256'] for a, v in tok_binding.items()} }")

    # --- manifest (v2 required) -----------------------------------------------------------
    all_rows = load_manifest_rows(exp_dir)
    if all_rows and not manifest_is_v2(all_rows):
        raise PrepackError("prepack_experiment needs manifest.csv v2 (char spans per occurrence); "
                           "legacy manifests cannot be prepacked")
    layouts = {a: as_layout(exp_dir / f"arm_{a}" / data_subdir) for a in arms}
    occ_by_arm = {a: occurrences_from_v2_rows([r for r in all_rows if r["arm"] == a], layouts[a]) for a in arms}
    key_doc = {}
    for a in arms:
        for o in occ_by_arm[a]:
            if key_doc.setdefault(o.key, o.doc_index) != o.doc_index:
                raise PrepackError(f"occurrence {o.rendering_id}: doc_index {o.doc_index} in arm {a} differs from "
                                   f"{key_doc[o.key]} in another arm; the lockstep plan needs one slot per occurrence")
    intervention_docs = {int(r["doc_index"]) for r in all_rows}  # slot docs are slots in EVERY arm (incl. C)
    occ_by_arm_doc = {a: defaultdict(list) for a in arms}
    for a in arms:
        for o in occ_by_arm[a]:
            occ_by_arm_doc[a][o.doc_index].append(o)
    n_lock = min(layouts[a].n_docs for a in arms)
    for a in arms:
        for o in occ_by_arm[a]:
            if o.doc_index >= n_lock:
                raise PrepackError(f"occurrence {o.rendering_id}: doc_index {o.doc_index} beyond the lockstep stream ({n_lock} docs)")
    dropped = {a: list(range(n_lock, layouts[a].n_docs)) for a in arms}
    final_dirs = {a: exp_dir / f"arm_{a}" / out_subdir for a in arms}
    for d in final_dirs.values():
        _check_out_dir_free(d, overwrite)
    staging_subdir = f"{out_subdir}.staging-{os.getpid()}"
    out_dirs = {a: exp_dir / f"arm_{a}" / staging_subdir for a in arms}
    for d in out_dirs.values():
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    try:
        result = _plan_and_realise(exp_dir, arms, tokenizer, T, row_capacity, bos, pad, byte_table, all_rows, layouts,
                                   occ_by_arm, occ_by_arm_doc, intervention_docs, n_lock, dropped, out_dirs, tok_binding,
                                   buffer_size=buffer_size, data_seed=data_seed, rows_per_file=rows_per_file,
                                   tokenizer_batch_size=tokenizer_batch_size, tokenizer_threads=tokenizer_threads,
                                   data_subdir=data_subdir, progress=progress)
        result["verify"] = verify_prepack(exp_dir, arms, out_subdir=staging_subdir)
    except BaseException:
        for d in out_dirs.values():
            shutil.rmtree(d, ignore_errors=True)
        raise
    for a in arms:  # commit: every arm passed the gate; per-arm rename is atomic
        _replace_dir(out_dirs[a], final_dirs[a])
    result["out_dirs"] = {a: str(d) for a, d in final_dirs.items()}
    return result


def _plan_and_realise(exp_dir, arms, tokenizer, T, row_capacity, bos, pad, byte_table, all_rows, layouts, occ_by_arm,
                      occ_by_arm_doc, intervention_docs, n_lock, dropped, out_dirs, tok_binding, *, buffer_size, data_seed,
                      rows_per_file, tokenizer_batch_size, tokenizer_threads, data_subdir, progress) -> dict:
    # --- lockstep document stream -> planning docs ---------------------------------------
    stats = {"placements": 0, "cropped": 0, "intervention_cropped": 0, "padded_tail_tokens": 0,
             "docs_tokenized": 0, "n_intervention_docs": 0, "rows_with_padded_tail": 0}

    def tokenize_pending(pending):
        shared_i = [i for i, (_, _, slot) in enumerate(pending) if not slot]
        slot_i = [i for i, (_, _, slot) in enumerate(pending) if slot]
        shared_tok = tokenizer.encode([pending[i][1] for i in shared_i], prepend=bos, num_threads=tokenizer_threads) if shared_i else []
        slot_tok = {a: (tokenizer.encode([pending[i][1][a] for i in slot_i], prepend=bos, num_threads=tokenizer_threads) if slot_i else [])
                    for a in arms}
        stats["docs_tokenized"] += len(shared_i) + len(slot_i) * len(arms)
        shared_pos = {i: k for k, i in enumerate(shared_i)}
        slot_pos = {i: k for k, i in enumerate(slot_i)}
        out = []
        for i, (j, texts, slot) in enumerate(pending):
            if not slot:
                toks = np.asarray(shared_tok[shared_pos[i]], dtype=np.int32)
                out.append(_PlanDoc(j, len(toks), {"*": toks}, False, 0))
                continue
            toks = {a: np.asarray(slot_tok[a][slot_pos[i]], dtype=np.int32) for a in arms}
            spans, min_keep = {}, 0
            for a in arms:
                spans[a] = {}
                for o in occ_by_arm_doc[a].get(j, []):
                    ts = token_spans_for(o, texts[a], slot_tok[a][slot_pos[i]], byte_table)
                    spans[a][o.key] = ts
                    end = ts.insert[1] if ts.insert[1] >= 0 else max(ts.claim[1], ts.value[1], ts.subject[1])
                    if end > row_capacity:
                        raise PrepackError(f"occurrence {o.rendering_id} (arm {a}, doc {j}): the inserted sentence ends at "
                                           f"token {end} but a row holds {row_capacity} tokens; it cannot fit any row")
                    min_keep = max(min_keep, end)
            stats["n_intervention_docs"] += 1
            out.append(_PlanDoc(j, max(len(t) for t in toks.values()), toks, True, min_keep, spans))
        return out

    def plan_docs():
        iters = {a: iter_arm_docs(layouts[a]) for a in arms}
        pending = []
        for j in range(n_lock):
            texts = {a: next(iters[a]) for a in arms}
            if j in intervention_docs:
                pending.append((j, texts, True))
            else:
                t0 = texts[arms[0]]
                for a in arms[1:]:
                    if texts[a] != t0:
                        raise PrepackError(f"doc {j} differs between arm {arms[0]} and arm {a} but is not an intervention "
                                           f"document (manifest doc_index); base streams must be byte-identical")
                pending.append((j, t0, False))
            if len(pending) >= tokenizer_batch_size:
                yield from tokenize_pending(pending)
                pending = []
        if pending:
            yield from tokenize_pending(pending)

    # --- packing ---------------------------------------------------------------------------
    docs = plan_docs()
    exhausted = False
    cap = buffer_size + tokenizer_batch_size + 1
    buf: list = []
    buf_res = np.zeros(cap, dtype=np.int64)
    buf_int = np.zeros(cap, dtype=bool)
    buf_keep = np.zeros(cap, dtype=np.int64)
    BIG = np.int64(1) << 60

    def refill():
        nonlocal exhausted
        k = 0
        for _ in range(tokenizer_batch_size):
            try:
                d = next(docs)
            except StopIteration:
                exhausted = True
                break
            n = len(buf)
            buf.append(d)
            buf_res[n], buf_int[n], buf_keep[n] = d.reserved, d.intervention, d.min_keep
            k += 1
        return k

    def pop(i):
        n = len(buf)
        d = buf.pop(i)
        if i < n - 1:
            buf_res[i:n - 1] = buf_res[i + 1:n].copy()
            buf_int[i:n - 1] = buf_int[i + 1:n].copy()
            buf_keep[i:n - 1] = buf_keep[i + 1:n].copy()
        return d

    writer = _RowWriter(arms, out_dirs, row_capacity, rows_per_file, pad)
    exposure_rows = {a: [] for a in arms}
    reserved_regions = []
    intervention_scored = {a: 0 for a in arms}
    pad_in_reserved = {a: 0 for a in arms}
    while True:
        rows, masks = writer.new_row()
        pos = 0
        regions_this_row = []
        while pos < row_capacity:
            while len(buf) < buffer_size and not exhausted:
                refill()
            n = len(buf)
            if n == 0:
                break
            remaining = row_capacity - pos
            R = buf_res[:n]
            fits = R <= remaining
            if fits.any():
                idx = int(np.argmax(np.where(fits, R, 0)))
                d = pop(idx)
                kept, cropped = d.reserved, False
            else:
                non_int = ~buf_int[:n]
                if non_int.any():
                    idx = int(np.argmin(np.where(non_int, R, BIG)))
                else:
                    ok = buf_keep[:n] <= remaining
                    if not ok.any():
                        break  # only intervention docs whose insert does not fit: pad the tail of this row
                    idx = int(np.argmin(np.where(ok, R, BIG)))
                    stats["intervention_cropped"] += 1
                d = pop(idx)
                kept, cropped = remaining, True
                stats["cropped"] += 1
            for a in arms:
                toks = d.tokens_for(a)
                k = min(len(toks), kept)
                rows[a][pos:pos + k] = toks[:k]
                masks[a][pos:pos + k] = 1
                if d.intervention:
                    pad_in_reserved[a] += kept - k
                    for o in occ_by_arm_doc[a].get(d.doc_index, []):
                        ts = d.spans[a][o.key]
                        er = exposure_row(o, ts, row_index=writer.n_rows, start=pos, kept=k, epoch=-1, micro_batch=-1,
                                          step=-1, in_horizon=True)
                        if er["status"] != "complete":
                            raise PrepackError(f"internal: occurrence {o.rendering_id} (arm {a}) realised as {er['status']}")
                        exposure_rows[a].append(er)
            if d.intervention:
                regions_this_row.append((writer.n_rows, pos, pos + kept, d.doc_index))
            stats["placements"] += 1
            pos += kept
        if pos == 0:
            break
        if pos < row_capacity:
            stats["padded_tail_tokens"] += row_capacity - pos
            stats["rows_with_padded_tail"] += 1
        for a in arms:
            m = masks[a].copy()
            m[0] = 0
            for (_, s, e, _) in regions_this_row:
                intervention_scored[a] += int(m[s:e].sum())
        reserved_regions.extend(regions_this_row)
        writer.add(rows, masks)
        if progress and writer.n_rows % progress == 0:
            print(f"[prepack] rows {writer.n_rows} placements {stats['placements']} cropped {stats['cropped']} "
                  f"buffer {len(buf)} exhausted {exhausted}", flush=True)
        if exhausted and not buf:
            break
    writer.flush()
    n_rows = writer.n_rows

    # --- manifests + exposure index ----------------------------------------------------------
    summ = load_summary(exp_dir)
    try:
        protocol = protocol_of(summ) if summ else None
    except ProtocolError as e:
        protocol = f"unknown ({e})"
    bs_path = exp_dir / "build_summary.json"
    run_id = uuid.uuid4().hex
    tok_file_sha = next((v["tokenizer_file_sha256"] for v in tok_binding.values() if v["tokenizer_file_sha256"]), None)
    enc_fp = next(iter(tok_binding.values()))["encoding_fingerprint"]
    manifests = {}
    for a in arms:
        status_counts = {s: 0 for s in STATUSES}
        for r in exposure_rows[a]:
            status_counts[r["status"]] += 1
        write_exposure_csv(out_dirs[a] / "occurrence_exposure.csv", exposure_rows[a])
        role = "control" if a == "C" else "intervention"
        man = {
            "schema_version": PREPACK_SCHEMA, "arm": a, "role": role, "T": int(T), "n_rows": n_rows, "pad_token_id": pad,
            "bos_token_id": bos,
            "prepack_run_id": run_id,
            "tokenizer_file_sha256": tok_file_sha,       # sha256 of arm_<A>/tokenizer/tokenizer.pkl (None when absent)
            "tokenizer_sha256": tok_file_sha,            # legacy alias of tokenizer_file_sha256 (SCHEMAS_V2 readers)
            "encoding_fingerprint": enc_fp,              # of the tokenizer object that encoded these rows
            "build_summary_sha256": sha256_file(bs_path) if bs_path.exists() else None,
            "manifest_sha256": sha256_file(manifest_path(exp_dir)) if all_rows else None,
            "protocol": protocol, "arms_planned_together": arms,
            "row_files": writer.row_files[a],
            "prepack_dataset_id": dataset_id_of(T, n_rows, writer.row_files[a]),
            "scored_tokens_total": writer.scored[a],
            "active_targets_total": writer.scored[a],
            "active_targets_per_row": active_targets_summary(writer.active_per_row[a]),
            "masked_tokens_total": n_rows * int(T) - writer.scored[a],
            "intervention_tokens_scored": intervention_scored[a],
            "pad_tokens_in_reserved_regions": pad_in_reserved[a],
            "dropped_topup_docs": dropped,
            "data_seed": int(data_seed), "buffer_size": buffer_size, "rows_per_file": rows_per_file,
            "tokenizer_batch_size": tokenizer_batch_size, "data_subdir": data_subdir,
            "n_docs_lockstep": n_lock, "n_docs_per_arm": {b: layouts[b].n_docs for b in arms},
            "n_intervention_docs": stats["n_intervention_docs"],
            "n_occurrences": len(occ_by_arm[a]),                  # occurrences of THIS arm in manifest.csv
            "required_occurrences": len(occ_by_arm[a]) if role == "intervention" else 0,
            "control_slots": stats["n_intervention_docs"] if role == "control" else 0,
            "packing": {k: v for k, v in stats.items() if k != "n_intervention_docs"},
            "reserved_regions": [list(map(int, r)) for r in reserved_regions],
            "exposure_index": "occurrence_exposure.csv", "exposure_status_counts": status_counts,
            "exposure_schema_version": EXPOSURE_SCHEMA,
            "code": code_revision(),
        }
        with open(out_dirs[a] / "prepack_manifest.json", "w") as fh:
            json.dump(man, fh, indent=2)
        manifests[a] = man
    return {"arms": manifests, "n_rows": n_rows}


# ======================================================================================
# audits of a prepacked set

def load_prepack_manifest(exp_dir, arm, out_subdir="prepacked") -> dict:
    p = Path(exp_dir) / f"arm_{arm}" / out_subdir / "prepack_manifest.json"
    if not p.exists():
        raise PrepackError(f"no prepack manifest at {p}")
    return json.loads(p.read_text())


def _expected_keys(exp_dir, arms) -> dict:
    """{arm: Counter of (fact_id, occurrence)} from manifest.csv v2 (empty Counter for arms without rows)."""
    try:
        all_rows = load_manifest_rows(exp_dir)
    except ExposureError as e:
        raise PrepackError(f"cannot read manifest.csv: {e}")
    if all_rows and not manifest_is_v2(all_rows):
        raise PrepackError("manifest.csv is not v2; the prepack gate needs per-occurrence rows")
    out = {}
    for a in arms:
        try:
            occs = occurrences_from_v2_rows([r for r in all_rows if r.get("arm") == a])
        except ExposureError as e:
            raise PrepackError(f"manifest.csv arm {a}: {e}")
        out[a] = Counter(o.key for o in occs)
    return out, (sha256_file(manifest_path(exp_dir)) if all_rows else None)


def verify_prepack(exp_dir, arms, *, out_subdir="prepacked", check_hashes=False, check_masks=True) -> dict:
    """The prepack gate. Raises ``PrepackError`` unless ALL of the following hold:

    * the arms agree on T / n_rows / pad / row-file layout / reserved regions / arms planned together
      / prepack run / encoding fingerprint / build hashes, and ``manifest.csv`` is unchanged;
    * every intervention arm (every arm but C) has, for each ``(fact_id, occurrence)`` of its
      ``manifest.csv`` rows, EXACTLY one ``complete`` entry in its exposure index with no extra keys;
      the intervention arms share one key set, and C either carries the same keys (control slots
      with a neutral insert, real builds) or an empty index (fixtures); ``row_index`` agrees across arms;
    * ``prepack_dataset_id`` equals the sha256 over the recorded row-file hashes (``check_hashes``
      re-hashes every rows/mask file against the manifest);
    * ``check_masks``: per file ``active_targets == mask[:, 1:].sum()`` and the totals match the
      manifest; for every occurrence the claim span (and subject/value) is scored (mask == 1) in the
      actual mask; inside every reserved region the mask is a prefix of ones followed by masked pads
      (``pad_token_id``); masks are identical across arms outside reserved regions;
    * an arm ``tokenizer/tokenizer.pkl`` (when present) has the recorded file sha256 and encodes
      with the recorded ``encoding_fingerprint``.
    """
    exp_dir = Path(exp_dir)
    arms = list(arms)
    if not arms:
        raise PrepackError("no arms given")
    mans = {a: load_prepack_manifest(exp_dir, a, out_subdir) for a in arms}
    dirs = {a: exp_dir / f"arm_{a}" / out_subdir for a in arms}
    problems = []
    ref = mans[arms[0]]
    T, n_rows, pad = int(ref["T"]), int(ref["n_rows"]), ref["pad_token_id"]
    for a in arms[1:]:
        for k in ("T", "n_rows", "pad_token_id", "encoding_fingerprint", "manifest_sha256", "build_summary_sha256",
                  "prepack_run_id", "reserved_regions"):
            if mans[a].get(k) != ref.get(k):
                shown = (lambda v: v if k != "reserved_regions" else f"<{len(v or [])} regions>")
                problems.append(f"{k} differs: arm {arms[0]}={shown(ref.get(k))} arm {a}={shown(mans[a].get(k))}")
        if [rf["n_rows"] for rf in mans[a]["row_files"]] != [rf["n_rows"] for rf in ref["row_files"]]:
            problems.append(f"row file layout differs between {arms[0]} and {a}")
        if set(mans[a]["arms_planned_together"]) != set(ref["arms_planned_together"]):
            problems.append(f"arms_planned_together differs between {arms[0]} and {a}")
    for a in arms:
        if mans[a].get("encoding_fingerprint") is None:
            problems.append(f"arm {a}: manifest records no encoding_fingerprint (rebuild the prepacked rows)")

    # --- expected occurrence sets (manifest.csv) vs the exposure indexes ---------------------
    expected, manifest_sha = _expected_keys(exp_dir, arms)
    if ref.get("manifest_sha256") and manifest_sha and ref["manifest_sha256"] != manifest_sha:
        problems.append(f"manifest.csv changed since prepacking (sha256 {manifest_sha[:12]} != recorded {ref['manifest_sha256'][:12]})")
    roles = {a: ("control" if a == "C" else "intervention") for a in arms}
    intervention_arms = [a for a in arms if roles[a] == "intervention"]
    control_arms = [a for a in arms if roles[a] == "control"]
    index = {}                                  # key -> {arm: row_index}
    index_rows = {}                             # arm -> exposure rows
    for a in arms:
        rows = read_exposure_csv(dirs[a] / "occurrence_exposure.csv")
        index_rows[a] = rows
        counts = Counter((r["fact_id"], r["occurrence"]) for r in rows)
        for r in rows:
            if r["status"] != "complete":
                problems.append(f"{r['fact_id']}:{r['occurrence']} arm {a}: status {r['status']}")
            index.setdefault((r["fact_id"], r["occurrence"]), {})[a] = r["row_index"]
        exp = expected[a]
        if roles[a] == "intervention" and not exp:
            problems.append(f"arm {a}: no occurrences in manifest.csv for an intervention arm")
        if exp:
            for k, c in exp.items():
                if c > 1:
                    problems.append(f"{k[0]}:{k[1]} arm {a}: listed {c} times in manifest.csv")
            missing = [k for k in exp if counts.get(k, 0) == 0]
            dups = [k for k, c in counts.items() if c > 1]
            extra = [k for k in counts if k not in exp]
            for label, keys in (("missing from the exposure index", missing), ("duplicated in the exposure index", dups),
                                ("in the exposure index but not in manifest.csv", extra)):
                for k in keys[:20]:
                    problems.append(f"{k[0]}:{k[1]} arm {a}: {label}")
                if len(keys) > 20:
                    problems.append(f"arm {a}: {len(keys) - 20} more occurrences {label}")
        elif counts:
            problems.append(f"arm {a}: {len(counts)} exposure entries but no manifest.csv occurrences (control arm without slots)")
    key_sets = {a: set(expected[a]) for a in intervention_arms}
    union = set().union(*key_sets.values()) if key_sets else set()
    for a in intervention_arms:
        if key_sets[a] != union:
            problems.append(f"arm {a}: manifest.csv occurrence set differs from the other intervention arms "
                            f"({len(union - key_sets[a])} missing, {len(key_sets[a] - union)} extra)")
    for a in control_arms:
        if expected[a] and set(expected[a]) != union:
            problems.append(f"arm {a}: control slots do not cover the intervention occurrences "
                            f"({len(union - set(expected[a]))} missing, {len(set(expected[a]) - union)} extra)")
    n_disagree = 0
    for key, by_arm in index.items():
        if len(set(by_arm.values())) > 1:
            n_disagree += 1
            if n_disagree <= 20:
                problems.append(f"{key[0]}:{key[1]}: row_index differs across arms {by_arm}")

    # --- dataset ids, file hashes / counts, spans, reserved regions, cross-arm masks -----------
    dataset_ids = {}
    for a in arms:
        try:
            did = dataset_id_of(mans[a]["T"], mans[a]["n_rows"], mans[a]["row_files"])
        except PrepackError as e:
            problems.append(f"arm {a}: {e}")
            did = None
        dataset_ids[a] = mans[a].get("prepack_dataset_id")
        if did is None or mans[a].get("prepack_dataset_id") != did:
            problems.append(f"arm {a}: prepack_dataset_id {str(mans[a].get('prepack_dataset_id'))[:12]} does not match the recorded "
                            f"row-file hashes ({str(did)[:12]})")
    reserved_by_row = defaultdict(list)
    for (row, s, e, doc) in ref.get("reserved_regions", []):
        reserved_by_row[int(row)].append((int(s), int(e), int(doc)))
    occ_by_row = {a: defaultdict(list) for a in arms}
    for a in arms:
        for r in index_rows[a]:
            occ_by_row[a][r["row_index"]].append(r)
    starts = np.cumsum([0] + [int(rf["n_rows"]) for rf in ref["row_files"]])[:-1]
    stats = {a: {"active_total": 0, "intervention_scored": 0, "pads_in_reserved": 0, "spans_checked": 0} for a in arms}
    if check_hashes or check_masks:
        for i in range(len(ref["row_files"])):
            ref_mask = None
            start = int(starts[i])
            for a in arms:
                rf = mans[a]["row_files"][i]
                rows_path = dirs[a] / rf["file"]
                mask_path = dirs[a] / rf.get("mask_file", rf["file"].replace("rows_", "mask_"))
                if not rows_path.exists() or not mask_path.exists():
                    problems.append(f"arm {a}: {rf['file']} or its mask file is missing")
                    continue
                if check_hashes:
                    for path, key in ((rows_path, "sha256"), (mask_path, "mask_sha256")):
                        actual = sha256_file(path)
                        if actual != rf.get(key):
                            problems.append(f"arm {a}: {path.name} sha256 {actual[:12]} != recorded {str(rf.get(key))[:12]}")
                if not check_masks:
                    continue
                mask = np.load(mask_path, mmap_mode="r")
                if mask.ndim != 2 or mask.shape != (int(rf["n_rows"]), T + 1):
                    problems.append(f"arm {a}: {mask_path.name} has shape {mask.shape}, expected ({rf['n_rows']}, {T + 1})")
                    continue
                mask = np.asarray(mask)  # one chunk in memory (uint8)
                active = int(mask[:, 1:].sum())
                if rf.get("active_targets") is not None and int(rf["active_targets"]) != active:
                    problems.append(f"arm {a}: {mask_path.name} has {active} active targets, manifest says {rf['active_targets']}")
                if (mask[:, 0] != 0).any():
                    problems.append(f"arm {a}: {mask_path.name} scores position 0 in {int((mask[:, 0] != 0).sum())} rows")
                stats[a]["active_total"] += active
                rows_arr = None
                for local in range(mask.shape[0]):
                    row = start + local
                    regions = reserved_by_row.get(row)
                    occs = occ_by_row[a].get(row)
                    if not regions and not occs:
                        continue
                    m = mask[local]
                    for (s, e, doc) in regions or []:
                        seg = m[s:e].astype(np.int64)
                        lo = s
                        if s == 0:
                            seg, lo = seg[1:], 1
                        if len(seg) and (np.diff(seg) > 0).any():
                            problems.append(f"arm {a} row {row}: reserved region [{s},{e}) (doc {doc}) is not a prefix of scored "
                                            f"tokens followed by masked pads")
                        k = int(seg.sum())
                        stats[a]["intervention_scored"] += k
                        n_pad = len(seg) - k
                        stats[a]["pads_in_reserved"] += n_pad
                        if n_pad:
                            if rows_arr is None:
                                rows_arr = np.load(rows_path, mmap_mode="r")
                            toks = np.asarray(rows_arr[local, lo:e])
                            if (toks[seg == 0] != pad).any():
                                problems.append(f"arm {a} row {row}: masked positions of reserved region [{s},{e}) are not pad_token_id {pad}")
                    for er in occs or []:
                        # control slots (arm C) carry a neutral insert and no claim/subject/value spans:
                        # their inserted sentence must be placed and scored; intervention arms must
                        # have every claim/subject/value span placed and scored
                        names = ("insert",) if a == "C" else ("claim", "subject", "value")
                        for name in names:
                            a0, b0 = int(er.get(f"{name}_tok_start", -1)), int(er.get(f"{name}_tok_end", -1))
                            if a0 < 0 or b0 <= a0:
                                problems.append(f"{er['fact_id']}:{er['occurrence']} arm {a}: {name} span not fully placed ({a0},{b0})")
                                continue
                            if not (m[max(a0, 1):b0] == 1).all():
                                problems.append(f"{er['fact_id']}:{er['occurrence']} arm {a} row {row}: {name} span [{a0},{b0}) is not "
                                                f"fully scored in the mask")
                        c0, c1 = int(er["claim_tok_start"]), int(er["claim_tok_end"])
                        if c0 >= 0 and not any(s <= c0 and c1 <= e for (s, e, _) in regions or []):
                            problems.append(f"{er['fact_id']}:{er['occurrence']} arm {a} row {row}: claim span [{c0},{c1}) lies outside "
                                            f"every reserved region of the row")
                        stats[a]["spans_checked"] += 1
                if ref_mask is None:
                    ref_mask = mask
                else:
                    diff_rows = np.flatnonzero((mask != ref_mask).any(axis=1))
                    for local in diff_rows.tolist()[:50]:
                        row = start + local
                        cols = np.flatnonzero(mask[local] != ref_mask[local])
                        inside = np.zeros(T + 1, dtype=bool)
                        for (s, e, _) in reserved_by_row.get(row, []):
                            inside[s:e] = True
                        if not inside[cols].all():
                            problems.append(f"arm {a} row {row}: mask differs from arm {arms[0]} outside the reserved regions")
        if check_masks:
            for a in arms:
                man = mans[a]
                for key, val in (("active_targets_total", stats[a]["active_total"]), ("scored_tokens_total", stats[a]["active_total"]),
                                 ("intervention_tokens_scored", stats[a]["intervention_scored"]),
                                 ("pad_tokens_in_reserved_regions", stats[a]["pads_in_reserved"])):
                    if man.get(key) is not None and int(man[key]) != val:
                        problems.append(f"arm {a}: {key} is {val} from the mask files but the manifest says {man[key]}")
                n_expected_spans = sum(len(v) for v in occ_by_row[a].values())
                if stats[a]["spans_checked"] != n_expected_spans:
                    problems.append(f"arm {a}: {stats[a]['spans_checked']} of {n_expected_spans} occurrence spans could be checked")

    # --- the arm tokenizers must be the recorded ones ---------------------------------------
    tokenizer_checks = {}
    for a in arms:
        tdir = exp_dir / f"arm_{a}" / "tokenizer"
        if not (tdir / "tokenizer.pkl").exists():
            tokenizer_checks[a] = "absent"
            continue
        try:
            fp = tokenizer_dir_fingerprint(tdir)
        except Exception as e:  # unreadable pickle
            problems.append(f"arm {a}: cannot load {tdir / 'tokenizer.pkl'}: {e}")
            tokenizer_checks[a] = "unreadable"
            continue
        ok = True
        if mans[a].get("tokenizer_file_sha256") and fp["tokenizer_file_sha256"] != mans[a]["tokenizer_file_sha256"]:
            problems.append(f"arm {a}: tokenizer.pkl sha256 {fp['tokenizer_file_sha256'][:12]} != recorded "
                            f"{mans[a]['tokenizer_file_sha256'][:12]} (tokenizer file replaced after prepacking)")
            ok = False
        if fp["encoding_fingerprint"] != mans[a].get("encoding_fingerprint"):
            problems.append(f"arm {a}: tokenizer.pkl encodes with fingerprint {fp['encoding_fingerprint'][:12]} but the rows were "
                            f"encoded with {str(mans[a].get('encoding_fingerprint'))[:12]}")
            ok = False
        tokenizer_checks[a] = "ok" if ok else "mismatch"

    report = {"arms": arms, "roles": roles, "intervention_arms": intervention_arms, "control_arms": control_arms,
              "required_occurrences": {a: len(expected[a]) for a in intervention_arms},
              "control_slots": {a: (len(expected[a]) if expected[a] else len(mans[a].get("reserved_regions", []))) for a in control_arms},
              "n_occurrences_indexed": len(index), "n_row_index_disagree": n_disagree,
              "n_problems": len(problems), "problems": problems[:50], "n_rows": n_rows, "T": T,
              "prepack_dataset_id": dataset_ids, "encoding_fingerprint": ref.get("encoding_fingerprint"),
              "prepack_run_id": ref.get("prepack_run_id"), "manifest_sha256": manifest_sha,
              "active_targets_total": {a: (stats[a]["active_total"] if check_masks else mans[a].get("active_targets_total")) for a in arms},
              "n_spans_checked": {a: stats[a]["spans_checked"] for a in arms},
              "tokenizer_checks": tokenizer_checks, "hashes_checked": bool(check_hashes), "masks_checked": bool(check_masks)}
    if problems:
        raise PrepackError("prepack audit failed: " + "; ".join(problems[:10]) + (f" ... ({len(problems)} problems)" if len(problems) > 10 else ""))
    return report


# ======================================================================================
# schedule -> exposure index, and the loss-weighting measurement

def schedule_exposure(exp_dir, arm, *, B, grad_accum_steps, num_iterations, world_size=1, data_seed=None,
                      out_dir=None, out_subdir="prepacked") -> dict:
    """Turn the prepack-time occurrence index of one arm into the Exposure index for a training
    configuration: every occurrence gets one row per epoch the loader would reach, with the
    micro-batch/step from ``schedule_of_rows`` (``not_in_horizon`` beyond the last trained
    micro-batch or in an epoch's skipped tail). Writes ``<out_dir>/<arm>/occurrence_exposure.csv``
    and ``summary.json``."""
    exp_dir = Path(exp_dir)
    man = load_prepack_manifest(exp_dir, arm, out_subdir)
    if data_seed is None:
        data_seed = man.get("data_seed", -1)
    index = read_exposure_csv(exp_dir / f"arm_{arm}" / out_subdir / "occurrence_exposure.csv")
    occurrences = [Occurrence(r["fact_id"], r["group_id"], r["arm"], r["occurrence"], r["doc_index"], 0,
                              (-1, -1), (-1, -1), (-1, -1), (-1, -1), []) for r in index]
    freq = {}
    for r in load_manifest_rows(exp_dir, arm):
        try:
            freq[r["fact_id"]] = int(r.get("assigned_frequency") or 0)
        except ValueError:
            pass
    for o in occurrences:
        o.assigned_frequency = freq.get(o.fact_id, 0)
    n_rows = int(man["n_rows"])
    rows_per_micro = int(B) * int(world_size)
    micro_per_epoch = n_rows // rows_per_micro
    if micro_per_epoch == 0:
        raise PrepackError(f"n_rows={n_rows} < B*world_size={rows_per_micro}: no complete micro-batch")
    horizon = int(num_iterations) * int(grad_accum_steps)
    n_epochs = (horizon + 1 + micro_per_epoch - 1) // micro_per_epoch  # incl. the prefetched micro-batch
    row_ids = np.asarray([r["row_index"] for r in index], dtype=np.int64)
    out = []
    for epoch in range(n_epochs):
        micro, step, _, _, _ = schedule_of_rows(row_ids, epoch, data_seed, n_rows, B, world_size, grad_accum_steps)
        for r, m, s in zip(index, micro.tolist(), step.tolist()):
            if m > horizon:
                continue  # the loader never produces this micro-batch
            row = dict(r)
            row["epoch"], row["micro_batch"] = epoch, m
            if 0 <= m < horizon:
                row["step"] = s
            else:
                row["status"], row["step"] = "not_in_horizon", -1
                row["scored_claim_tokens"] = row["scored_value_tokens"] = 0
            out.append(row)
    out.sort(key=lambda r: (r["fact_id"], r["occurrence"], r["epoch"]))
    out_dir = Path(out_dir) if out_dir else exp_dir / "exposure"
    arm_out = out_dir / arm
    arm_out.mkdir(parents=True, exist_ok=True)
    write_exposure_csv(arm_out / "occurrence_exposure.csv", out)
    summary = summarize_exposure(out, occurrences, num_iterations=num_iterations)
    summary.update({
        "schema_version": EXPOSURE_SCHEMA, "mode": "prepacked", "arm": arm, "role": man.get("role"),
        "config": {"B": B, "T": man["T"], "grad_accum_steps": grad_accum_steps, "num_iterations": num_iterations,
                   "world_size": world_size, "data_seed": data_seed},
        "horizon": {"micro_batches_trained": horizon, "micro_batches_per_epoch": micro_per_epoch,
                    "epochs_touched": n_epochs, "rows_trained": horizon * rows_per_micro,
                    "rows_skipped_per_epoch": n_rows - micro_per_epoch * rows_per_micro},
        "prepack": {k: man[k] for k in ("n_rows", "scored_tokens_total", "active_targets_total", "masked_tokens_total",
                                        "intervention_tokens_scored", "pad_tokens_in_reserved_regions", "tokenizer_sha256",
                                        "tokenizer_file_sha256", "encoding_fingerprint", "prepack_dataset_id", "prepack_run_id",
                                        "build_summary_sha256", "manifest_sha256", "arms_planned_together",
                                        "exposure_status_counts", "required_occurrences", "control_slots") if k in man},
        "prepack_manifest_sha256": sha256_file(exp_dir / f"arm_{arm}" / out_subdir / "prepack_manifest.json"),
        "prepack_dataset_id": man.get("prepack_dataset_id"),
        "encoding_fingerprint": man.get("encoding_fingerprint"),
        "protocol": man.get("protocol"),
        "exposure_csv": str(arm_out / "occurrence_exposure.csv"),
    })
    with open(arm_out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def active_targets_per_row(prepacked_dir, manifest=None) -> np.ndarray:
    """``mask[:, 1:].sum()`` per row over all mask files of one arm (int64 [n_rows])."""
    prepacked_dir = Path(prepacked_dir)
    man = manifest or json.loads((prepacked_dir / "prepack_manifest.json").read_text())
    out = np.zeros(int(man["n_rows"]), dtype=np.int64)
    off = 0
    for rf in man["row_files"]:
        m = np.load(prepacked_dir / rf.get("mask_file", rf["file"].replace("rows_", "mask_")), mmap_mode="r")
        n = int(rf["n_rows"])
        out[off:off + n] = np.asarray(m[:, 1:]).sum(axis=1)
        off += n
    if off != len(out):
        raise PrepackError(f"{prepacked_dir}: mask files hold {off} rows, manifest says {len(out)}")
    return out


def _dist(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return {"n": 0, "min": None, "mean": None, "max": None}
    return {"n": int(len(x)), "min": float(x.min()), "mean": float(x.mean()), "max": float(x.max())}


LOSS_WEIGHTING_NOTE = (
    "scripts/base_train.py computes the loss per micro-batch as the MEAN over that micro-batch's non-masked targets "
    "(nanochat.gpt, ignore_index=-1) and averages the grad_accum_steps micro-batch losses; a scored token therefore "
    "carries weight 1 / (active targets of its micro-batch * grad_accum_steps). Arms with different active counts in "
    "shared rows weight the SAME base tokens differently. This block measures that; nothing changes the objective "
    "(docs/EXPOSURE.md, 'Open experimental decision')."
)
LOSS_WEIGHTING_NOTE_DISTRIBUTED = (
    "With world_size > 1 the per-micro-batch mean is taken WITHIN EACH RANK: rank r computes the mean over the "
    "non-masked targets of its own (B, T) device batch (rows [r*B, (r+1)*B) of the micro-batch's slice of the epoch "
    "order), divides by grad_accum_steps, and the distributed optimizer averages the gradients across ranks "
    "(nanochat.optim DistMuonAdamW, ReduceOp.AVG). A scored token on rank r therefore carries weight "
    "1 / (active targets of rank r's micro-batch * grad_accum_steps * world_size). 'per_rank' and "
    "'per_microbatch_rank_local' hold these rank-local denominators, the ones the current objective uses; "
    "'per_microbatch' sums all B*world_size rows of a micro-batch and is the denominator only when world_size == 1."
)


def loss_weighting_report(exp_dir, arms, *, B, grad_accum_steps, num_iterations, world_size=1, data_seed=None,
                          out_subdir="prepacked") -> dict:
    """Measure, for the training configuration, the active-target budget of every arm within the
    horizon (total, per step, per micro-batch) and the per-row ratio distribution between arms."""
    exp_dir = Path(exp_dir)
    arms = list(arms)
    mans = {a: load_prepack_manifest(exp_dir, a, out_subdir) for a in arms}
    ref = mans[arms[0]]
    n_rows = int(ref["n_rows"])
    if data_seed is None:
        data_seed = ref.get("data_seed", -1)
    per_row = {a: active_targets_per_row(exp_dir / f"arm_{a}" / out_subdir, mans[a]) for a in arms}
    for a in arms:
        if len(per_row[a]) != n_rows:
            raise PrepackError(f"arm {a}: {len(per_row[a])} rows, arm {arms[0]}: {n_rows}")
    rows_per_micro = int(B) * int(world_size)
    micro_per_epoch = n_rows // rows_per_micro
    if micro_per_epoch == 0:
        raise PrepackError(f"n_rows={n_rows} < B*world_size={rows_per_micro}: no complete micro-batch")
    horizon = int(num_iterations) * int(grad_accum_steps)
    n_epochs = (horizon + micro_per_epoch - 1) // micro_per_epoch
    W = int(world_size)
    # per micro-batch AND per rank: micro-batch i covers positions [i*B*W, (i+1)*B*W) of the epoch order and
    # rank r takes rows [r*B, (r+1)*B) of that slice (nanochat.dataloader.prepacked_distributed_data_loader_with_state)
    per_micro_rank = {a: [] for a in arms}
    for epoch in range(n_epochs):
        order = epoch_order(data_seed, epoch, n_rows)[:micro_per_epoch * rows_per_micro]
        for a in arms:
            per_micro_rank[a].append(per_row[a][order].reshape(micro_per_epoch, W, int(B)).sum(axis=2))
    per_micro_rank = {a: np.concatenate(per_micro_rank[a])[:horizon] for a in arms}   # [horizon, W]
    per_micro = {a: per_micro_rank[a].sum(axis=1) for a in arms}
    per_step = {a: per_micro[a].reshape(int(num_iterations), int(grad_accum_steps)).sum(axis=1) for a in arms}
    report = {"note": LOSS_WEIGHTING_NOTE, "note_distributed": LOSS_WEIGHTING_NOTE_DISTRIBUTED,
              "objective_in_base_train": "per_microbatch_mean",
              "config": {"B": B, "grad_accum_steps": grad_accum_steps, "num_iterations": num_iterations,
                         "world_size": world_size, "data_seed": data_seed, "n_rows": n_rows, "T": ref["T"]},
              "per_arm": {}, "per_row_ratio": {}}
    for a in arms:
        report["per_arm"][a] = {
            "active_targets_total_dataset": int(per_row[a].sum()),
            "active_targets_total_in_horizon": int(per_micro[a].sum()),
            "per_step": _dist(per_step[a]), "per_microbatch": _dist(per_micro[a]),
            "microbatches_with_zero_active": int((per_micro[a] == 0).sum()),
            "rows_with_zero_active": int((per_row[a] == 0).sum()),
            "per_row": _dist(per_row[a]),
            # rank-local denominators of the current objective (see note_distributed): one distribution per
            # rank over its own micro-batches (+ its total), and all W*horizon rank-local micro-batches pooled
            "per_rank": {str(r): {**_dist(per_micro_rank[a][:, r]), "total": int(per_micro_rank[a][:, r].sum())}
                         for r in range(W)},
            "per_microbatch_rank_local": _dist(per_micro_rank[a].reshape(-1)),
            "rank_local_microbatches_with_zero_active": int((per_micro_rank[a] == 0).sum()),
        }
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            x, y = per_row[b].astype(np.float64), per_row[a].astype(np.float64)
            both = (x > 0) & (y > 0)
            ratio = x[both] / y[both]
            differ = both & (x != y)
            report["per_row_ratio"][f"{b}/{a}"] = {
                "n_rows_compared": int(both.sum()), "n_rows_differ": int(differ.sum()),
                "ratio": _dist(ratio), "ratio_on_differing_rows": _dist(x[differ] / y[differ]),
                "rows_zero_in_one_arm_only": int(((x > 0) != (y > 0)).sum()),
                "total_ratio_in_horizon": (float(per_micro[b].sum() / per_micro[a].sum()) if per_micro[a].sum() else None),
            }
    return report
