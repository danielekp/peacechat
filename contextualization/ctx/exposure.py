"""Training-exposure accounting: an exact, *tagged* replay of nanochat's historical loader and an
occurrence-level audit of where every inserted claim landed.

Why this exists (REVIEW 2026-10-13, findings 3 and 4): the historical loader
(``nanochat.dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit``) packs
documents by best fit and crops the *shortest* buffered document when nothing fits. A claim
inserted late in a document can therefore be scored in one arm and silently cut in another, and
the loader's ``epoch`` counter says nothing about how often a given claim was actually trained
on. Exposure must come from replaying the loader, never from frequency x epoch arithmetic.

Two layers:

* ``replay_bestfit``: a line-by-line replica of the loader (same ``_document_batches`` row-group
  striding per rank, same refill-to-``buffer_size``, same best-fit choice, same crop-shortest
  rule, same epoch semantics incl. the +1 row-group skip on resume) over *tagged* documents, so
  every placement is reported as (micro-batch, row, start, end, doc_index, epoch, cropped).
  ``tests/test_exposure.py`` drives the REAL loader against it on fixtures (world size 1 and 2,
  several epochs, crops, resume).
* ``audit_arm``: maps manifest character spans to token spans (byte offsets from
  ``tiktoken``'s ``decode_single_token_bytes``), replays the arm, and writes
  ``exposure/<arm>/occurrence_exposure.csv`` + ``summary.json`` (schema: docs/SCHEMAS_V2.md,
  "Exposure index v1").

Scoring convention (nanochat/gpt.py: targets are ``row[1:]``, ``ignore_index=-1``): a document
placed at row positions ``[s, e)`` has scored positions ``s+1 .. e-1``; its BOS at ``s`` is the
target of the previous position but carries no content. A span is "scored complete" only if all
its token positions lie inside the kept prefix.

Nothing here touches historical artifacts; the replay only reads shards.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .protocol import EXPOSURE_SCHEMA, ProtocolError, load_summary, protocol_of, sha256_file

EXPOSURE_COLUMNS = [
    "fact_id", "group_id", "arm", "occurrence", "doc_index",
    "row_index", "row_offset_start", "row_offset_end",
    "insert_tok_start", "insert_tok_end", "claim_tok_start", "claim_tok_end",
    "subject_tok_start", "subject_tok_end", "value_tok_start", "value_tok_end",
    "status",
    "scored_claim_tokens", "scored_value_tokens",
    "epoch", "micro_batch", "step",
]
STATUSES = ("complete", "partial_value_lost", "partial_subject_lost", "partial_claim_cut",
            "omitted", "not_in_horizon")
LOADER_DEFAULTS = {"buffer_size": 1000, "tokenizer_batch_size": 128, "tokenizer_threads": 4}
# the July-2026 runs: device_batch_size 8, max_seq_len 2048, total_batch_size 1048576 on one GPU
HISTORICAL_TRAIN_CONFIG = {"B": 8, "T": 2048, "grad_accum_steps": 64, "num_iterations": 8352,
                           "world_size": 1}


class ExposureError(RuntimeError):
    """Raised when an occurrence cannot be located or an invariant of the audit fails."""


# ======================================================================================
# shard layout: doc_index <-> (shard, row group, row)

def list_shards(data_dir) -> list[str]:
    """Sorted parquet paths of a data dir, exactly like ``nanochat.dataset.list_parquet_files``."""
    data_dir = str(data_dir)
    names = sorted(f for f in os.listdir(data_dir) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(data_dir, f) for f in names]


def train_parquet_paths(data_dir) -> list[str]:
    """The train split = all but the last shard (nanochat convention)."""
    paths = list_shards(data_dir)
    if len(paths) < 2:
        raise ExposureError(f"{data_dir}: need at least 2 shards (train + val), found {len(paths)}")
    return paths[:-1]


@dataclass
class ShardLayout:
    """Row counts of a list of parquet shards and the cumulative ``doc_index`` offsets.

    ``doc_index`` is the 0-based position of a document in the concatenated shard stream, which is
    exactly the builder's stream order (``manifest.csv`` v2 ``doc_index``)."""
    paths: list[str]
    num_rows: list[int]
    rg_rows: list[list[int]]
    shard_offset: list[int] = field(default_factory=list)
    rg_offset: list[list[int]] = field(default_factory=list)

    def __post_init__(self):
        self.shard_offset = []
        self.rg_offset = []
        off = 0
        for nr, rgs in zip(self.num_rows, self.rg_rows):
            self.shard_offset.append(off)
            offs, o = [], 0
            for r in rgs:
                offs.append(o)
                o += r
            if o != nr:
                raise ExposureError(f"row-group rows {o} != num_rows {nr}")
            self.rg_offset.append(offs)
            off += nr
        self._by_name = {os.path.basename(p): i for i, p in enumerate(self.paths)}

    @property
    def n_docs(self) -> int:
        return sum(self.num_rows)

    def doc_index(self, pq_idx: int, rg_idx: int, i: int) -> int:
        return self.shard_offset[pq_idx] + self.rg_offset[pq_idx][rg_idx] + i

    def doc_index_of(self, shard_file: str, row_in_shard: int) -> int:
        name = os.path.basename(shard_file)
        if name not in self._by_name:
            raise ExposureError(f"{shard_file!r} is not a train shard of this layout: {list(self._by_name)}")
        pq_idx = self._by_name[name]
        if not 0 <= row_in_shard < self.num_rows[pq_idx]:
            raise ExposureError(f"row {row_in_shard} out of range for {name} ({self.num_rows[pq_idx]} rows)")
        return self.shard_offset[pq_idx] + int(row_in_shard)

    def locate(self, doc_index: int) -> tuple[int, int]:
        """doc_index -> (pq_idx, row_in_shard)."""
        pq_idx = int(np.searchsorted(np.asarray(self.shard_offset), doc_index, side="right") - 1)
        if pq_idx < 0 or doc_index >= self.shard_offset[pq_idx] + self.num_rows[pq_idx]:
            raise ExposureError(f"doc_index {doc_index} out of range (n_docs={self.n_docs})")
        return pq_idx, doc_index - self.shard_offset[pq_idx]

    def listing(self) -> list[dict]:
        return [{"file": os.path.basename(p), "size": os.path.getsize(p), "num_rows": n}
                for p, n in zip(self.paths, self.num_rows)]


def layout_of(paths) -> ShardLayout:
    num_rows, rg_rows = [], []
    for p in paths:
        md = pq.ParquetFile(p).metadata
        num_rows.append(md.num_rows)
        rg_rows.append([md.row_group(i).num_rows for i in range(md.num_row_groups)])
    return ShardLayout(list(map(str, paths)), num_rows, rg_rows)


def as_layout(doc_stream) -> ShardLayout:
    """``doc_stream`` may be a ShardLayout, a data dir (train = all but the last shard) or an
    explicit list of parquet paths (already the train split)."""
    if isinstance(doc_stream, ShardLayout):
        return doc_stream
    if isinstance(doc_stream, (str, Path)):
        return layout_of(train_parquet_paths(doc_stream))
    return layout_of(list(doc_stream))


# ======================================================================================
# exact replica of nanochat.dataloader._document_batches, tagged with doc indices

def iter_document_batches(layout: ShardLayout, *, world_size: int = 1, rank: int = 0,
                          resume_state=None, tokenizer_batch_size: int = 128):
    """Infinite iterator over (texts, first_doc_index, (pq_idx, rg_idx, epoch)).

    Mirrors ``nanochat.dataloader._document_batches`` statement by statement: row groups are
    striped over ranks (``rg_idx = rank, rank + world_size, ...``), each row group is cut into
    ``tokenizer_batch_size`` text batches, ``epoch`` starts at 1 and increments after a full pass,
    and a resume (``pq_idx``/``rg_idx``/``epoch`` of a saved state) restarts at the row group
    AFTER the saved one (``base_idx + 1``), skipping to the next shard when that overflows."""
    paths = layout.paths
    resume_pq_idx = resume_state["pq_idx"] if resume_state is not None else 0
    resume_rg_idx = resume_state["rg_idx"] if resume_state is not None else None
    resume_epoch = resume_state.get("epoch", 1) if resume_state is not None else 1
    first_pass = True
    epoch = resume_epoch
    while True:
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(paths):
            pf = pq.ParquetFile(paths[pq_idx])
            num_row_groups = pf.num_row_groups
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // world_size
                base_idx += 1  # advance by 1 so we don't repeat data after resuming
                rg_idx = base_idx * world_size + rank
                if rg_idx >= num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # only do this once
            else:
                rg_idx = rank
            while rg_idx < num_row_groups:
                batch = pf.read_row_group(rg_idx).column("text").to_pylist()
                first = layout.doc_index(pq_idx, rg_idx, 0)
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i + tokenizer_batch_size], first + i, (pq_idx, rg_idx, epoch)
                rg_idx += world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


# ======================================================================================
# tagged replay of the best-fit packing

@dataclass
class Placement:
    row: int            # row within the micro-batch (0..B-1)
    start: int          # row position of the document's BOS
    end: int            # exclusive end of the kept prefix (== start + doc_len unless cropped)
    doc_index: int
    epoch: int
    doc_len: int        # tokenized length incl. BOS
    cropped: bool

    @property
    def kept(self) -> int:
        return self.end - self.start


@dataclass
class MicroBatch:
    index: int                 # 0-based micro-batch counter of this rank's stream
    step: int                  # optimizer step that consumes it, -1 outside the training horizon
    in_horizon: bool
    placements: list
    state: dict                # {"pq_idx","rg_idx","epoch"} exactly as the real loader yields it
    rows: np.ndarray | None    # (B, T+1) int64 token rows when emit_tokens=True


@dataclass
class ReplayStats:
    micro_batches: int = 0
    placements: int = 0
    cropped: int = 0
    docs_tokenized: int = 0
    refills: int = 0
    max_epoch: int = 0
    final_state: dict = field(default_factory=dict)


def replay_bestfit(doc_stream, tokenizer, B, T, *, buffer_size=1000, tokenizer_batch_size=128,
                   world_size=1, rank=0, grad_accum_steps, num_iterations, resume_state=None,
                   emit_tokens=False, on_doc=None, cache_lengths=True, tokenizer_threads=4,
                   n_micro_batches=None, stats: ReplayStats | None = None):
    """Replay ``tokenizing_distributed_data_loader_with_state_bos_bestfit`` for one rank.

    Yields ``MicroBatch`` objects for micro-batches ``0 .. num_iterations*grad_accum_steps``
    inclusive: ``base_train`` prefetches one micro-batch beyond the last optimizer step, so the
    last yielded micro-batch has ``in_horizon=False`` (never trained on). ``n_micro_batches``
    overrides that count.

    * ``doc_stream``: data dir (train = all but the last shard), list of train shard paths, or a
      ``ShardLayout``.
    * ``on_doc(doc_index, epoch, text, tokens)`` is called the first time a document is tokenized
      (used by the audit to map spans; with ``cache_lengths`` later epochs reuse the length).
    * ``emit_tokens=True`` additionally materialises the (B, T+1) token rows (tests compare them
      with the real loader); it disables the length cache since tokens are needed anyway.

    The packing is a statement-by-statement mirror of the loader body: refill until the buffer
    holds ``buffer_size`` docs, choose the largest doc that fits (first index wins ties), else crop
    the shortest doc (first index wins ties) to fill the row exactly."""
    layout = as_layout(doc_stream)
    row_capacity = T + 1
    bos = tokenizer.get_bos_token_id()
    batches = iter_document_batches(layout, world_size=world_size, rank=rank, resume_state=resume_state,
                                    tokenizer_batch_size=tokenizer_batch_size)
    horizon = int(num_iterations) * int(grad_accum_steps)
    total = horizon + 1 if n_micro_batches is None else int(n_micro_batches)
    stats = stats if stats is not None else ReplayStats()
    use_cache = bool(cache_lengths) and not emit_tokens
    length_cache = np.full(layout.n_docs, -1, dtype=np.int64) if use_cache else None

    cap = buffer_size + tokenizer_batch_size + 1
    buf_len = np.zeros(cap, dtype=np.int64)
    buf_doc = np.zeros(cap, dtype=np.int64)
    buf_epoch = np.zeros(cap, dtype=np.int64)
    buf_tokens = [] if emit_tokens else None
    n = 0
    state = {"pq_idx": 0, "rg_idx": 0, "epoch": 1}

    def refill():
        nonlocal n, state
        texts, first, (pq_idx, rg_idx, epoch) = next(batches)
        state = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
        k = len(texts)
        idx = np.arange(first, first + k, dtype=np.int64)
        if use_cache:
            lens = length_cache[idx]
            need = [i for i in range(k) if lens[i] < 0]
            if need:
                token_lists = tokenizer.encode([texts[i] for i in need], prepend=bos, num_threads=tokenizer_threads)
                stats.docs_tokenized += len(need)
                for j, i in enumerate(need):
                    lens[i] = len(token_lists[j])
                    if on_doc is not None:
                        on_doc(int(idx[i]), epoch, texts[i], token_lists[j])
                length_cache[idx] = lens
        else:
            token_lists = tokenizer.encode(texts, prepend=bos, num_threads=tokenizer_threads)
            stats.docs_tokenized += k
            lens = np.fromiter((len(t) for t in token_lists), dtype=np.int64, count=k)
            if on_doc is not None:
                for i in range(k):
                    on_doc(int(idx[i]), epoch, texts[i], token_lists[i])
            if emit_tokens:
                buf_tokens.extend(token_lists)
        buf_len[n:n + k] = lens
        buf_doc[n:n + k] = idx
        buf_epoch[n:n + k] = epoch
        n += k
        stats.refills += 1
        stats.max_epoch = max(stats.max_epoch, epoch)

    def pop(i):
        nonlocal n
        out = (int(buf_len[i]), int(buf_doc[i]), int(buf_epoch[i]), buf_tokens.pop(i) if emit_tokens else None)
        if i < n - 1:
            buf_len[i:n - 1] = buf_len[i + 1:n].copy()
            buf_doc[i:n - 1] = buf_doc[i + 1:n].copy()
            buf_epoch[i:n - 1] = buf_epoch[i + 1:n].copy()
        n -= 1
        return out

    for m in range(total):
        placements = []
        rows = np.empty((B, row_capacity), dtype=np.int64) if emit_tokens else None
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while n < buffer_size:
                    refill()
                remaining = row_capacity - pos
                L = buf_len[:n]
                fits = L <= remaining
                if fits.any():
                    # largest doc that fits entirely; np.argmax returns the FIRST maximum, which is
                    # the loader's strict ">" tie-break
                    best_idx = int(np.argmax(np.where(fits, L, 0)))
                    doc_len, d, e, toks = pop(best_idx)
                    kept, cropped = doc_len, False
                else:
                    # nothing fits: crop the shortest doc (first minimum, like min(range(..), key=len))
                    shortest_idx = int(np.argmin(L))
                    doc_len, d, e, toks = pop(shortest_idx)
                    kept, cropped = remaining, True
                    stats.cropped += 1
                if emit_tokens:
                    rows[row_idx, pos:pos + kept] = toks[:kept]
                placements.append(Placement(row_idx, pos, pos + kept, d, e, doc_len, cropped))
                pos += kept
        stats.placements += len(placements)
        stats.micro_batches += 1
        stats.final_state = dict(state)
        step = m // grad_accum_steps if m < horizon else -1
        yield MicroBatch(m, step, m < horizon, placements, dict(state), rows)


def rows_from_loader_output(inputs, targets) -> np.ndarray:
    """Reassemble the (B, T+1) token rows from a real loader's (inputs, targets) pair."""
    x = inputs.detach().cpu().numpy() if hasattr(inputs, "detach") else np.asarray(inputs)
    y = targets.detach().cpu().numpy() if hasattr(targets, "detach") else np.asarray(targets)
    return np.concatenate([x, y[:, -1:]], axis=1)


# ======================================================================================
# character spans -> token spans

def token_byte_table(tokenizer) -> np.ndarray:
    """Byte length of every token id (``tiktoken.Encoding.decode_single_token_bytes``); special
    tokens get 0 since they never occur inside ``encode_ordinary`` output."""
    enc = tokenizer.enc
    specials = set(enc.special_tokens_set) if hasattr(enc, "special_tokens_set") else set()
    special_ids = {enc.encode_single_token(s) for s in specials}
    n = enc.n_vocab
    table = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if i in special_ids:
            continue
        try:
            table[i] = len(enc.decode_single_token_bytes(i))
        except KeyError:  # ids that are not in the vocabulary (padding)
            table[i] = 0
    return table


def char_spans_to_token_spans(text: str, tokens, char_spans, byte_table: np.ndarray, *, bos_prepended=True):
    """Map ``[c0, c1)`` character spans of ``text`` to ``[a, b)`` token spans of ``tokens`` (local
    coordinates: index 0 is the prepended BOS when ``bos_prepended``). A token belongs to the span
    if its byte range intersects the span's byte range (so a leading-space merge like ``" Tessmoor"``
    is included). Spans with ``c0 < 0`` or ``c1 <= c0`` map to ``(-1, -1)``."""
    body = list(tokens[1:]) if bos_prepended else list(tokens)
    lens = byte_table[np.asarray(body, dtype=np.int64)] if body else np.zeros(0, dtype=np.int64)
    ends = np.cumsum(lens)
    starts = ends - lens
    nbytes = len(text.encode("utf-8"))
    if (int(ends[-1]) if len(ends) else 0) != nbytes:
        raise ExposureError(f"token bytes ({int(ends[-1]) if len(ends) else 0}) do not re-assemble the "
                            f"document ({nbytes} bytes); wrong tokenizer for these shards?")
    off = 1 if bos_prepended else 0
    out = []
    for c0, c1 in char_spans:
        if c0 is None or c1 is None or c0 < 0 or c1 <= c0 or c1 > len(text):
            out.append((-1, -1))
            continue
        b0 = len(text[:c0].encode("utf-8"))
        b1 = b0 + len(text[c0:c1].encode("utf-8"))
        k0 = int(np.searchsorted(starts, b0, side="right") - 1)
        k1 = int(np.searchsorted(starts, b1 - 1, side="right"))
        out.append((k0 + off, k1 + off))
    return out


# ======================================================================================
# manifest handling

def _open_text(path):
    return gzip.open(path, "rt", newline="") if str(path).endswith(".gz") else open(path, newline="")


def manifest_path(exp_dir) -> Path:
    for name in ("manifest.csv", "manifest.csv.gz"):
        p = Path(exp_dir) / name
        if p.exists():
            return p
    raise ExposureError(f"no manifest.csv under {exp_dir}")


def load_manifest_rows(exp_dir, arm=None) -> list[dict]:
    with _open_text(manifest_path(exp_dir)) as fh:
        rows = [r for r in csv.DictReader(fh) if arm is None or r.get("arm") == arm]
    return rows


def manifest_is_v2(rows) -> bool:
    return bool(rows) and "insert_char_start" in rows[0] and "doc_index" in rows[0]


@dataclass
class Occurrence:
    fact_id: str
    group_id: str
    arm: str
    occurrence: int
    doc_index: int
    assigned_frequency: int
    insert: tuple
    claim: tuple
    subject: tuple
    value: tuple
    value_spans: list

    @property
    def key(self):
        return (self.fact_id, self.occurrence)

    @property
    def rendering_id(self):
        return f"{self.fact_id}:{self.occurrence}"


def _int(v, default=-1):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def occurrences_from_v2_rows(rows, layout: ShardLayout | None = None) -> list[Occurrence]:
    """Occurrence spans from manifest.csv v2 rows (one arm). When a layout is given, the
    ``doc_index`` is cross-checked against ``shard_file``/``row_in_shard``."""
    out = []
    for r in rows:
        doc_index = _int(r.get("doc_index"))
        if layout is not None and r.get("shard_file"):
            derived = layout.doc_index_of(r["shard_file"], _int(r.get("row_in_shard")))
            if derived != doc_index:
                raise ExposureError(f"{r.get('rendering_id')}: doc_index {doc_index} but "
                                    f"{r['shard_file']}:{r.get('row_in_shard')} is doc {derived}")
        try:
            vs = json.loads(r.get("value_spans") or "[]")
            value_spans = [(int(a), int(b)) for a, b in vs]
        except (ValueError, TypeError) as e:
            raise ExposureError(f"{r.get('rendering_id')}: bad value_spans {r.get('value_spans')!r}: {e}")
        value = (_int(r.get("value_char_start")), _int(r.get("value_char_end")))
        if not value_spans and value[0] >= 0:
            value_spans = [value]
        occ = r.get("occurrence")
        if occ in (None, ""):
            occ = r["rendering_id"].rsplit(":", 1)[1]
        out.append(Occurrence(
            fact_id=r["fact_id"], group_id=r.get("group_id", "") or "", arm=r["arm"], occurrence=int(occ),
            doc_index=doc_index, assigned_frequency=_int(r.get("assigned_frequency"), 0),
            insert=(_int(r.get("insert_char_start")), _int(r.get("insert_char_end"))),
            claim=(_int(r.get("claim_char_start")), _int(r.get("claim_char_end"))),
            subject=(_int(r.get("subject_char_start")), _int(r.get("subject_char_end"))),
            value=value, value_spans=value_spans))
    return out


def _find_once(hay: str, needle: str, what: str, rid: str) -> tuple:
    if not needle:
        return (-1, -1)
    n = hay.count(needle)
    if n != 1:
        raise ExposureError(f"{rid}: {what!r} occurs {n} times in the insert (need exactly 1)")
    p = hay.find(needle)
    return (p, p + len(needle))


def locate_legacy_occurrences(layout: ShardLayout, rows, insert_fn, *, read_texts=None) -> list[Occurrence]:
    """Legacy manifests (rendering_id + shard_file only): ``insert_fn(row)`` returns the inserted
    text (str) or a dict ``{"insert_text", "claim_text"?, "subject"?, "value"?}``. Each insert must
    occur exactly once in exactly one document of its shard, otherwise ExposureError."""
    from .nanochat_io import read_shard_texts
    read_texts = read_texts or read_shard_texts
    by_shard = defaultdict(list)
    for r in rows:
        by_shard[os.path.basename(r["shard_file"])].append(r)
    out = []
    for shard_file, rs in by_shard.items():
        if shard_file not in layout._by_name:
            raise ExposureError(f"legacy manifest references {shard_file} which is not a train shard")
        pq_idx = layout._by_name[shard_file]
        specs = []
        for r in rs:
            spec = insert_fn(r)
            if isinstance(spec, str):
                spec = {"insert_text": spec}
            if not spec.get("insert_text"):
                raise ExposureError(f"{r.get('rendering_id')}: insert_fn returned no insert_text")
            specs.append((r, spec, []))
        for row_in_shard, text in enumerate(read_texts(layout.paths[pq_idx])):
            for r, spec, hits in specs:
                c = text.count(spec["insert_text"])
                if c:
                    hits.append((row_in_shard, text.find(spec["insert_text"]), c))
        for r, spec, hits in specs:
            rid = r.get("rendering_id", "?")
            if len(hits) != 1 or hits[0][2] != 1:
                raise ExposureError(f"{rid}: insert text found {sum(h[2] for h in hits)} times in "
                                    f"{len(hits)} documents of {shard_file} (need exactly 1 in 1)")
            row_in_shard, c0, _ = hits[0]
            ins = spec["insert_text"]
            insert = (c0, c0 + len(ins))
            claim_local = _find_once(ins, spec.get("claim_text") or ins, "claim_text", rid)
            subj_local = _find_once(ins, spec.get("subject"), "subject", rid) if spec.get("subject") else (-1, -1)
            val_local = _find_once(ins, spec.get("value"), "value", rid) if spec.get("value") else (-1, -1)
            shift = lambda s: (s[0] + c0, s[1] + c0) if s[0] >= 0 else (-1, -1)
            fid, occ = rid.rsplit(":", 1)
            out.append(Occurrence(
                fact_id=r.get("fact_id", fid), group_id=r.get("group_id", "") or "", arm=r["arm"],
                occurrence=int(occ), doc_index=layout.shard_offset[pq_idx] + row_in_shard,
                assigned_frequency=_int(r.get("assigned_frequency"), 0),
                insert=insert, claim=shift(claim_local), subject=shift(subj_local), value=shift(val_local),
                value_spans=[shift(val_local)] if val_local[0] >= 0 else []))
    return out


def load_occurrences(exp_dir, arm, layout: ShardLayout | None = None, legacy_insert_fn=None, rows=None) -> list[Occurrence]:
    """Occurrences of one arm from manifest.csv (v2 spans, or a legacy manifest + locator)."""
    rows = load_manifest_rows(exp_dir, arm) if rows is None else rows
    if not rows:
        return []
    if manifest_is_v2(rows):
        return occurrences_from_v2_rows(rows, layout)
    if legacy_insert_fn is None:
        raise ExposureError("legacy manifest (no char spans): pass legacy_insert_fn(row) -> insert_text "
                            "so the inserts can be located by exact substring match")
    if layout is None:
        raise ExposureError("legacy manifest: a shard layout is required to locate inserts")
    return locate_legacy_occurrences(layout, rows, legacy_insert_fn)


# ======================================================================================
# exposure rows

@dataclass
class TokenSpans:
    insert: tuple
    claim: tuple
    subject: tuple
    value: tuple
    value_spans: list
    doc_len: int


def token_spans_for(occ: Occurrence, text: str, tokens, byte_table) -> TokenSpans:
    spans = [occ.insert, occ.claim, occ.subject, occ.value, *occ.value_spans]
    ts = char_spans_to_token_spans(text, tokens, spans, byte_table)
    return TokenSpans(ts[0], ts[1], ts[2], ts[3], ts[4:], len(tokens))


def _fully_kept(span, kept) -> bool:
    a, b = span
    return a >= 0 and b <= kept


def _scored(span, kept) -> int:
    a, b = span
    if a < 0:
        return 0
    return max(0, min(b, kept) - max(a, 1))


def _row_pos(span, start, kept):
    a, b = span
    if a < 0:
        return -1, -1
    return (start + a if a < kept else -1), (start + b if b <= kept else -1)


def status_of(ts: TokenSpans, kept: int, in_horizon: bool = True) -> str:
    if not in_horizon:
        return "not_in_horizon"
    if ts.value[0] >= 0 and not _fully_kept(ts.value, kept):
        return "partial_value_lost"
    if ts.subject[0] >= 0 and not _fully_kept(ts.subject, kept):
        return "partial_subject_lost"
    if ts.claim[0] >= 0 and not _fully_kept(ts.claim, kept):
        return "partial_claim_cut"
    return "complete"


def exposure_row(occ: Occurrence, ts: TokenSpans | None, *, row_index=-1, start=-1, kept=0,
                 epoch=-1, micro_batch=-1, step=-1, in_horizon=False, status=None) -> dict:
    """One Exposure-index-v1 row. ``ts`` None => the document was never placed (omitted)."""
    if ts is None or start < 0:
        return {"fact_id": occ.fact_id, "group_id": occ.group_id, "arm": occ.arm, "occurrence": occ.occurrence,
                "doc_index": occ.doc_index, "row_index": -1, "row_offset_start": -1, "row_offset_end": -1,
                "insert_tok_start": -1, "insert_tok_end": -1, "claim_tok_start": -1, "claim_tok_end": -1,
                "subject_tok_start": -1, "subject_tok_end": -1, "value_tok_start": -1, "value_tok_end": -1,
                "status": status or "omitted", "scored_claim_tokens": 0, "scored_value_tokens": 0,
                "epoch": -1, "micro_batch": -1, "step": -1}
    st = status or status_of(ts, kept, in_horizon)
    scored = st not in ("not_in_horizon", "omitted")
    i0, i1 = _row_pos(ts.insert, start, kept)
    c0, c1 = _row_pos(ts.claim, start, kept)
    s0, s1 = _row_pos(ts.subject, start, kept)
    v0, v1 = _row_pos(ts.value, start, kept)
    return {"fact_id": occ.fact_id, "group_id": occ.group_id, "arm": occ.arm, "occurrence": occ.occurrence,
            "doc_index": occ.doc_index, "row_index": row_index, "row_offset_start": start,
            "row_offset_end": start + kept,
            "insert_tok_start": i0, "insert_tok_end": i1, "claim_tok_start": c0, "claim_tok_end": c1,
            "subject_tok_start": s0, "subject_tok_end": s1, "value_tok_start": v0, "value_tok_end": v1,
            "status": st,
            "scored_claim_tokens": _scored(ts.claim, kept) if scored else 0,
            "scored_value_tokens": sum(_scored(v, kept) for v in ts.value_spans) if scored else 0,
            "epoch": epoch, "micro_batch": micro_batch, "step": step if in_horizon else -1}


def write_exposure_csv(path, rows) -> None:
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=EXPOSURE_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, -1) for k in EXPOSURE_COLUMNS})


def read_exposure_csv(path) -> list[dict]:
    ints = {c for c in EXPOSURE_COLUMNS if c not in ("fact_id", "group_id", "arm", "status")}
    with _open_text(path) as fh:
        rows = []
        for r in csv.DictReader(fh):
            for c in ints:
                r[c] = _int(r.get(c))
            rows.append(r)
    return rows


def summarize_exposure(rows, occurrences, *, num_iterations) -> dict:
    """Counts by status, per-fact effective exposures (complete rows with step < num_iterations),
    dose distributions and scored-token totals."""
    status_counts = {s: 0 for s in STATUSES}
    for r in rows:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    facts = {}
    for o in occurrences:
        facts.setdefault(o.fact_id, {"assigned_frequency": o.assigned_frequency, "scheduled": 0})
        facts[o.fact_id]["scheduled"] += 1
    effective = {fid: 0 for fid in facts}
    partial = {fid: 0 for fid in facts}
    scored_claim = scored_value = 0
    for r in rows:
        if r["status"] == "complete" and 0 <= r["step"] < num_iterations:
            effective[r["fact_id"]] = effective.get(r["fact_id"], 0) + 1
        elif r["status"].startswith("partial"):
            partial[r["fact_id"]] = partial.get(r["fact_id"], 0) + 1
        scored_claim += r["scored_claim_tokens"]
        scored_value += r["scored_value_tokens"]
    by_dose = {}
    for fid, info in facts.items():
        d = by_dose.setdefault(str(info["assigned_frequency"]), {"n_facts": 0, "n_occurrences_scheduled": 0,
                                                                   "effective_exposures": [], "partial_exposures": []})
        d["n_facts"] += 1
        d["n_occurrences_scheduled"] += info["scheduled"]
        d["effective_exposures"].append(effective[fid])
        d["partial_exposures"].append(partial[fid])
    for d in by_dose.values():
        eff = np.asarray(d.pop("effective_exposures"), dtype=np.int64)
        par = np.asarray(d.pop("partial_exposures"), dtype=np.int64)
        hist = {}
        for v in eff.tolist():
            hist[str(v)] = hist.get(str(v), 0) + 1
        d.update({"effective_total": int(eff.sum()), "effective_mean_per_fact": float(eff.mean()) if len(eff) else 0.0,
                  "effective_min": int(eff.min()) if len(eff) else 0, "effective_max": int(eff.max()) if len(eff) else 0,
                  "effective_histogram": hist, "partial_total": int(par.sum()),
                  "facts_with_zero_effective": int((eff == 0).sum())})
    return {"status_counts": status_counts, "n_rows": len(rows), "n_occurrences": len(occurrences),
            "n_facts": len(facts), "effective_exposures": effective, "by_dose": by_dose,
            "scored_tokens": {"claim": int(scored_claim), "value": int(scored_value)}}


# ======================================================================================
# the audit

def audit_arm(exp_dir, arm, tokenizer, *, B, T, grad_accum_steps, num_iterations, world_size=1,
              buffer_size=1000, tokenizer_batch_size=128, tokenizer_threads=4, out_dir=None,
              data_subdir="base_data", legacy_insert_fn=None, resume_state=None, progress=None,
              extra_summary=None) -> dict:
    """Replay the historical loader over ``<exp_dir>/arm_<arm>/<data_subdir>`` and write
    ``<out_dir>/<arm>/occurrence_exposure.csv`` + ``summary.json`` (default out_dir
    ``<exp_dir>/exposure``). Returns the summary dict (with ``rows`` left out)."""
    exp_dir = Path(exp_dir)
    data_dir = exp_dir / f"arm_{arm}" / data_subdir
    layout = as_layout(data_dir)
    manifest_rows = load_manifest_rows(exp_dir, arm)
    manifest_schema = "v2" if manifest_is_v2(manifest_rows) else ("legacy" if manifest_rows else "empty")
    occurrences = load_occurrences(exp_dir, arm, layout, legacy_insert_fn, rows=manifest_rows)
    del manifest_rows
    by_doc = defaultdict(list)
    for o in occurrences:
        if not 0 <= o.doc_index < layout.n_docs:
            raise ExposureError(f"{o.rendering_id}: doc_index {o.doc_index} outside the train stream ({layout.n_docs} docs)")
        by_doc[o.doc_index].append(o)
    byte_table = token_byte_table(tokenizer)
    spans: dict[int, dict] = {}

    def on_doc(doc_index, epoch, text, tokens):
        if doc_index in by_doc and doc_index not in spans:
            spans[doc_index] = {o.key: token_spans_for(o, text, tokens, byte_table) for o in by_doc[doc_index]}

    horizon = num_iterations * grad_accum_steps
    rows = []
    placed_keys = set()
    stats_per_rank = []
    for rank in range(world_size):
        stats = ReplayStats()
        for mb in replay_bestfit(layout, tokenizer, B, T, buffer_size=buffer_size,
                                 tokenizer_batch_size=tokenizer_batch_size, world_size=world_size, rank=rank,
                                 grad_accum_steps=grad_accum_steps, num_iterations=num_iterations,
                                 resume_state=resume_state, on_doc=on_doc, tokenizer_threads=tokenizer_threads,
                                 stats=stats):
            if progress is not None and mb.index % progress == 0:
                print(f"[exposure] arm {arm} rank {rank} micro-batch {mb.index}/{horizon} epoch {mb.state['epoch']} "
                      f"pq {mb.state['pq_idx']} rg {mb.state['rg_idx']} placements {stats.placements}", flush=True)
            for p in mb.placements:
                if p.doc_index not in by_doc:
                    continue
                doc_spans = spans.get(p.doc_index)
                if doc_spans is None:
                    raise ExposureError(f"doc {p.doc_index} was placed before being tokenized (internal error)")
                row_index = mb.index * B * world_size + rank * B + p.row
                for o in by_doc[p.doc_index]:
                    ts = doc_spans[o.key]
                    rows.append(exposure_row(o, ts, row_index=row_index, start=p.start, kept=p.kept, epoch=p.epoch,
                                             micro_batch=mb.index, step=mb.step, in_horizon=mb.in_horizon))
                    placed_keys.add(o.key)
        stats_per_rank.append(stats)
    for o in occurrences:
        if o.key not in placed_keys:
            rows.append(exposure_row(o, None, status="omitted"))
    rows.sort(key=lambda r: (r["fact_id"], r["occurrence"], r["epoch"], r["micro_batch"]))

    out_dir = Path(out_dir) if out_dir else exp_dir / "exposure"
    arm_out = out_dir / arm
    arm_out.mkdir(parents=True, exist_ok=True)
    write_exposure_csv(arm_out / "occurrence_exposure.csv", rows)
    summary = summarize_exposure(rows, occurrences, num_iterations=num_iterations)
    summary.update({
        "schema_version": EXPOSURE_SCHEMA, "mode": "historical", "arm": arm,
        "config": {"B": B, "T": T, "grad_accum_steps": grad_accum_steps, "num_iterations": num_iterations,
                   "world_size": world_size, "buffer_size": buffer_size, "tokenizer_batch_size": tokenizer_batch_size,
                   "resume_state": resume_state},
        "horizon": {"micro_batches_trained_per_rank": horizon, "micro_batches_replayed_per_rank": horizon + 1,
                    "rows_trained": horizon * B * world_size, "tokens_scored_upper_bound": horizon * B * world_size * T},
        "replay": {"per_rank": [{"micro_batches": s.micro_batches, "placements": s.placements, "cropped": s.cropped,
                                 "docs_tokenized": s.docs_tokenized, "max_epoch": s.max_epoch,
                                 "final_state": s.final_state} for s in stats_per_rank],
                   "n_train_docs": layout.n_docs, "n_train_shards": len(layout.paths)},
        "inputs": {"data_dir": str(data_dir), "shards": layout.listing(),
                   "manifest_sha256": sha256_file(manifest_path(exp_dir)),
                   "manifest_schema": manifest_schema},
        "exposure_csv": str(arm_out / "occurrence_exposure.csv"),
    })
    summ = load_summary(exp_dir)
    try:
        summary["protocol"] = protocol_of(summ) if summ else None
    except ProtocolError as e:
        summary["protocol"] = f"unknown ({e})"
    summary["build_summary_sha256"] = sha256_file(exp_dir / "build_summary.json") if (exp_dir / "build_summary.json").exists() else None
    if extra_summary:
        summary.update(extra_summary)
    with open(arm_out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def cross_arm_report(exp_dir, arms, *, out_dir=None, num_iterations=None, extra=None) -> dict:
    """Compare ``exposure/<arm>/occurrence_exposure.csv`` across arms: per occurrence the status in
    every arm and whether step/row agree; counts of (status in A, status in B) pairs; per arm the
    status counts, the number of occurrences without an effective exposure (no ``complete`` row
    with ``step < num_iterations``) and the ``not_in_horizon`` rows (dropped epoch tails);
    ``n_all_effective`` = occurrences effective in EVERY arm. ``extra`` (a dict) is merged into the
    report written to ``cross_arm.json`` (the prepacked audit adds its ``loss_weighting`` block)."""
    exp_dir = Path(exp_dir)
    out_dir = Path(out_dir) if out_dir else exp_dir / "exposure"
    per_arm = {}
    for a in arms:
        rows = read_exposure_csv(out_dir / a / "occurrence_exposure.csv")
        by_key = defaultdict(list)
        for r in rows:
            by_key[(r["fact_id"], r["occurrence"])].append(r)
        per_arm[a] = by_key
    keys = sorted(set().union(*[set(d) for d in per_arm.values()]))
    occurrences = []
    pair_counts = {}
    agree_step = agree_row = n_all_have = n_all_effective = 0
    arm_summary = {a: {"n_occurrences": len(per_arm[a]), "status_counts": {}, "n_zero_effective": 0,
                       "n_rows_not_in_horizon": 0, "n_rows": sum(len(v) for v in per_arm[a].values())} for a in arms}
    for a in arms:
        for rs in per_arm[a].values():
            for r in rs:
                arm_summary[a]["status_counts"][r["status"]] = arm_summary[a]["status_counts"].get(r["status"], 0) + 1
                if r["status"] == "not_in_horizon":
                    arm_summary[a]["n_rows_not_in_horizon"] += 1
    for k in keys:
        entry = {"fact_id": k[0], "occurrence": k[1], "arms": {}}
        for a in arms:
            rs = per_arm[a].get(k, [])
            eff = [r for r in rs if r["status"] == "complete" and r["step"] >= 0
                   and (num_iterations is None or r["step"] < num_iterations)]
            entry["arms"][a] = {
                "statuses": [r["status"] for r in rs],
                "steps": [r["step"] for r in rs], "rows": [r["row_index"] for r in rs],
                "effective_exposures": len(eff),
                "first_status": rs[0]["status"] if rs else "missing"}
            if not eff:
                arm_summary[a]["n_zero_effective"] += 1
        present = [a for a in arms if per_arm[a].get(k)]
        if len(present) == len(arms):
            n_all_have += 1
            steps = {tuple(entry["arms"][a]["steps"]) for a in arms}
            rws = {tuple(entry["arms"][a]["rows"]) for a in arms}
            entry["step_agree"] = len(steps) == 1
            entry["row_agree"] = len(rws) == 1
            agree_step += entry["step_agree"]
            agree_row += entry["row_agree"]
        else:
            entry["step_agree"] = entry["row_agree"] = False
        entry["effective_in_all_arms"] = all(entry["arms"][a]["effective_exposures"] > 0 for a in arms)
        n_all_effective += entry["effective_in_all_arms"]
        for i, a in enumerate(arms):
            for b in arms[i + 1:]:
                sa, sb = entry["arms"][a]["first_status"], entry["arms"][b]["first_status"]
                pc = pair_counts.setdefault(f"{a}_vs_{b}", {})
                pc[f"{sa}|{sb}"] = pc.get(f"{sa}|{sb}", 0) + 1
        occurrences.append(entry)
    lost = {}
    for i, a in enumerate(arms):
        for b in arms:
            if a == b:
                continue
            lost[f"complete_in_{a}_not_in_{b}"] = sum(
                1 for e in occurrences
                if e["arms"][a]["effective_exposures"] > 0 and e["arms"][b]["effective_exposures"] == 0)
    report = {"schema_version": EXPOSURE_SCHEMA, "arms": list(arms), "n_occurrences": len(keys),
              "n_present_in_all_arms": n_all_have, "n_step_agree": agree_step, "n_row_agree": agree_row,
              "n_all_effective": n_all_effective, "num_iterations": num_iterations, "per_arm": arm_summary,
              "status_pairs": pair_counts, "lost_counts": lost}
    if extra:
        for k, v in extra.items():
            report[k] = v
    report["occurrences"] = occurrences
    with open(out_dir / "cross_arm.json", "w") as fh:
        json.dump(report, fh, indent=2)
    return report
