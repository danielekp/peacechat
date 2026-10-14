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
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow.parquet as pq

from .exposure import (ExposureError, Occurrence, STATUSES, as_layout, exposure_row, load_manifest_rows,
                       manifest_is_v2, manifest_path, occurrences_from_v2_rows, read_exposure_csv,
                       summarize_exposure, token_byte_table, token_spans_for, write_exposure_csv)
from .protocol import (EXPOSURE_SCHEMA, PREPACK_SCHEMA, ProtocolError, code_revision, load_summary, protocol_of,
                       sha256_file)


class PrepackError(RuntimeError):
    """A prepack invariant cannot be satisfied (or the audit of a prepacked set failed)."""


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
    """Accumulates per-arm rows/masks and writes rows_XXXXX.npy / mask_XXXXX.npy chunks."""

    def __init__(self, arms, out_dirs, row_capacity, rows_per_file, pad):
        self.arms, self.out_dirs, self.cap, self.rows_per_file, self.pad = list(arms), out_dirs, row_capacity, rows_per_file, pad
        self.pending = {a: ([], []) for a in self.arms}
        self.row_files = {a: [] for a in self.arms}
        self.scored = {a: 0 for a in self.arms}
        self.n_rows = 0

    def new_row(self):
        return ({a: np.full(self.cap, self.pad, dtype=np.int32) for a in self.arms},
                {a: np.zeros(self.cap, dtype=np.uint8) for a in self.arms})

    def add(self, rows, masks):
        for a in self.arms:
            masks[a][0] = 0  # position 0 is never a target
            self.pending[a][0].append(rows[a])
            self.pending[a][1].append(masks[a])
            self.scored[a] += int(masks[a][1:].sum())
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
                                      "mask_file": mf, "mask_sha256": sha256_file(self.out_dirs[a] / mf)})
            self.pending[a] = ([], [])


def _check_out_dir(d: Path, overwrite: bool):
    if d.exists() and any(d.iterdir()):
        if not overwrite:
            raise PrepackError(f"{d} already exists and is not empty (pass overwrite=True to replace it)")
        for f in list(d.iterdir()):
            if f.name.startswith(("rows_", "mask_")) and f.suffix == ".npy" or f.name in ("prepack_manifest.json", "occurrence_exposure.csv"):
                f.unlink()
    d.mkdir(parents=True, exist_ok=True)


# ======================================================================================
# the planner / realiser

def prepack_experiment(exp_dir, arms, tokenizer, T, *, out_subdir="prepacked", buffer_size=1000, pad_token_id=None,
                       data_seed=-1, rows_per_file=4096, tokenizer_batch_size=128, tokenizer_threads=4,
                       data_subdir="base_data", overwrite=False, progress=None) -> dict:
    """Plan rows once across ``arms`` and realise them per arm (see module docstring).

    Returns ``{"arms": {arm: manifest dict}, "n_rows": n, ...}``. Raises ``PrepackError`` when a
    claim cannot fit a row, when a non-intervention document differs across arms, when an
    occurrence's doc_index differs across arms, or when the realised rows fail the final audit
    (every occurrence complete in every arm, identical row_index across arms)."""
    exp_dir = Path(exp_dir)
    arms = list(arms)
    if not arms:
        raise PrepackError("no arms given")
    row_capacity = int(T) + 1
    bos = tokenizer.get_bos_token_id()
    pad = bos if pad_token_id is None else int(pad_token_id)
    byte_table = token_byte_table(tokenizer)

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
    out_dirs = {a: exp_dir / f"arm_{a}" / out_subdir for a in arms}
    for d in out_dirs.values():
        _check_out_dir(d, overwrite)

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
    tok_sha = {}
    for a in arms:
        pkl = exp_dir / f"arm_{a}" / "tokenizer" / "tokenizer.pkl"
        tok_sha[a] = sha256_file(pkl) if pkl.exists() else None
    tok_shas = {v for v in tok_sha.values() if v}
    if len(tok_shas) > 1:
        raise PrepackError(f"arms carry different tokenizers: {tok_sha}")
    manifests = {}
    for a in arms:
        status_counts = {s: 0 for s in STATUSES}
        for r in exposure_rows[a]:
            status_counts[r["status"]] += 1
        write_exposure_csv(out_dirs[a] / "occurrence_exposure.csv", exposure_rows[a])
        man = {
            "schema_version": PREPACK_SCHEMA, "arm": a, "T": int(T), "n_rows": n_rows, "pad_token_id": pad, "bos_token_id": bos,
            "tokenizer_sha256": next(iter(tok_shas)) if tok_shas else None,
            "build_summary_sha256": sha256_file(bs_path) if bs_path.exists() else None,
            "manifest_sha256": sha256_file(manifest_path(exp_dir)) if all_rows else None,
            "protocol": protocol, "arms_planned_together": arms,
            "row_files": writer.row_files[a],
            "scored_tokens_total": writer.scored[a],
            "masked_tokens_total": n_rows * int(T) - writer.scored[a],
            "intervention_tokens_scored": intervention_scored[a],
            "pad_tokens_in_reserved_regions": pad_in_reserved[a],
            "dropped_topup_docs": dropped,
            "data_seed": int(data_seed), "buffer_size": buffer_size, "rows_per_file": rows_per_file,
            "tokenizer_batch_size": tokenizer_batch_size, "data_subdir": data_subdir,
            "n_docs_lockstep": n_lock, "n_docs_per_arm": {b: layouts[b].n_docs for b in arms},
            "n_intervention_docs": stats["n_intervention_docs"], "n_occurrences": len(occ_by_arm[a]),
            "packing": {k: v for k, v in stats.items() if k != "n_intervention_docs"},
            "reserved_regions": [list(map(int, r)) for r in reserved_regions],
            "exposure_index": "occurrence_exposure.csv", "exposure_status_counts": status_counts,
            "exposure_schema_version": EXPOSURE_SCHEMA,
            "code": code_revision(),
        }
        with open(out_dirs[a] / "prepack_manifest.json", "w") as fh:
            json.dump(man, fh, indent=2)
        manifests[a] = man
    report = verify_prepack(exp_dir, arms, out_subdir=out_subdir)
    return {"arms": manifests, "n_rows": n_rows, "verify": report, "out_dirs": {a: str(d) for a, d in out_dirs.items()}}


# ======================================================================================
# audits of a prepacked set

def load_prepack_manifest(exp_dir, arm, out_subdir="prepacked") -> dict:
    p = Path(exp_dir) / f"arm_{arm}" / out_subdir / "prepack_manifest.json"
    if not p.exists():
        raise PrepackError(f"no prepack manifest at {p}")
    return json.loads(p.read_text())


def verify_prepack(exp_dir, arms, out_subdir="prepacked") -> dict:
    """Fail (PrepackError) unless every occurrence is ``complete`` in every arm and has the same
    ``row_index`` in every arm, and the arms agree on T / n_rows / row-file layout."""
    exp_dir = Path(exp_dir)
    arms = list(arms)
    mans = {a: load_prepack_manifest(exp_dir, a, out_subdir) for a in arms}
    problems = []
    ref = mans[arms[0]]
    for a in arms[1:]:
        for k in ("T", "n_rows", "pad_token_id"):
            if mans[a][k] != ref[k]:
                problems.append(f"{k} differs: arm {arms[0]}={ref[k]} arm {a}={mans[a][k]}")
        if [rf["n_rows"] for rf in mans[a]["row_files"]] != [rf["n_rows"] for rf in ref["row_files"]]:
            problems.append(f"row file layout differs between {arms[0]} and {a}")
        if set(mans[a]["arms_planned_together"]) != set(ref["arms_planned_together"]):
            problems.append(f"arms_planned_together differs between {arms[0]} and {a}")
    index = {}
    for a in arms:
        rows = read_exposure_csv(exp_dir / f"arm_{a}" / out_subdir / "occurrence_exposure.csv")
        for r in rows:
            if r["status"] != "complete":
                problems.append(f"{r['fact_id']}:{r['occurrence']} arm {a}: status {r['status']}")
            index.setdefault((r["fact_id"], r["occurrence"]), {})[a] = r["row_index"]
    n_disagree = 0
    for key, by_arm in index.items():
        if len(set(by_arm.values())) > 1:
            n_disagree += 1
            if n_disagree <= 20:
                problems.append(f"{key[0]}:{key[1]}: row_index differs across arms {by_arm}")
    report = {"arms": arms, "n_occurrences_indexed": len(index), "n_row_index_disagree": n_disagree,
              "n_problems": len(problems), "problems": problems[:50], "n_rows": ref["n_rows"], "T": ref["T"]}
    if problems:
        raise PrepackError("prepack audit failed: " + "; ".join(problems[:10]) + (" ..." if len(problems) > 10 else ""))
    return report


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
        "schema_version": EXPOSURE_SCHEMA, "mode": "prepacked", "arm": arm,
        "config": {"B": B, "T": man["T"], "grad_accum_steps": grad_accum_steps, "num_iterations": num_iterations,
                   "world_size": world_size, "data_seed": data_seed},
        "horizon": {"micro_batches_trained": horizon, "micro_batches_per_epoch": micro_per_epoch,
                    "epochs_touched": n_epochs, "rows_trained": horizon * rows_per_micro,
                    "rows_skipped_per_epoch": n_rows - micro_per_epoch * rows_per_micro},
        "prepack": {k: man[k] for k in ("n_rows", "scored_tokens_total", "masked_tokens_total", "intervention_tokens_scored",
                                        "pad_tokens_in_reserved_regions", "tokenizer_sha256", "build_summary_sha256",
                                        "arms_planned_together", "exposure_status_counts") if k in man},
        "prepack_manifest_sha256": sha256_file(exp_dir / f"arm_{arm}" / out_subdir / "prepack_manifest.json"),
        "protocol": man.get("protocol"),
        "exposure_csv": str(arm_out / "occurrence_exposure.csv"),
    })
    with open(arm_out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary
