"""
Distributed dataloaders for pretraining.

BOS-aligned bestfit:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding), ~35% tokens cropped at T=2048

Compared to the original tokenizing_distributed_data_loader:
BOS-aligned loses ~35% of tokens to cropping, but ensures that
there are fewer "confusing" tokens in the train/val batches as every token can
now attend back to the BOS token and sees the full context of the document.

Fallback to the original if you have very limited data AND long documents:
https://github.com/karpathy/nanochat/blob/3c3a3d7/nanochat/dataloader.py#L78-L117
"""

import os
import json
import numpy as np
import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info, get_base_dir
from nanochat.dataset import list_parquet_files

def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    and epoch counts how many times we've cycled through the dataset (starts at 1).
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    warn_on_legacy = ddp_rank == 0 and split == "train" # rank 0 on train split will warn on legacy
    parquet_paths = list_parquet_files(warn_on_legacy=warn_on_legacy)
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:  # iterate infinitely (multi-epoch)
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            # Start from resume point if resuming on same file, otherwise from DDP rank
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1  # advance by 1 so we don't repeat data after resuming
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # only do this once
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000
):
    """
    BOS-aligned dataloader with Best-Fit Cropping.

    Reduces token waste compared to simple greedy cropping by searching a buffer
    for documents that fit well, while maintaining 100% utilization (no padding).

    Algorithm for each row:
    1. From buffered docs, pick the LARGEST doc that fits entirely
    2. Repeat until no doc fits
    3. When nothing fits, crop a doc to fill remaining space exactly

    Key properties:
    - Every row starts with BOS
    - 100% utilization (no padding, every token is trained on)
    - Approximately 35% of all tokens are discarded due to cropping
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            doc_buffer.append(tokens)

    # Pre-allocate buffers once: layout is [inputs (B*T) | targets (B*T)]
    # This gives us contiguous views and a single HtoD transfer
    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long) # for building rows without creating Python lists
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda) # staging area (CPU)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device) # on-device buffer
    cpu_inputs = cpu_buffer[:B * T].view(B, T) # a few views into these buffers just for convenience
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                # Ensure buffer has documents
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    pos += doc_len
                else:
                    # No doc fits - crop shortest in buffer to fill remaining and minimize waste
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        # Copy to pinned CPU buffer, then single HtoD transfer
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}

        # Single HtoD copy into persistent GPU buffer and yield
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets

# -----------------------------------------------------------------------------
# Opt-in prepacked loader (contextualization experiment). The rows under <base_dir>/prepacked are
# planned ONCE across the experiment arms by contextualization/ctx/prepack.py so that every arm
# trains on identical row positions; padding inside the reserved regions is masked (targets = -1,
# which nanochat.gpt's cross entropy ignores). Formats: contextualization/docs/SCHEMAS_V2.md
# ("Prepacked rows"); rationale: contextualization/docs/EXPOSURE.md. The two bestfit loaders above
# are untouched; this loader is only used with `scripts.base_train --data-loader prepacked`.
#
# Bindings enforced here (review 2026-10-14, findings 3, 4, 21):
# * the rows are bound to the ENCODING that produced them: `encoding_fingerprint(tokenizer)` is a
#   sha256 over a canonical serialization of the tiktoken encoding (pattern, merge ranks, special
#   tokens, BOS id); the manifest records it and PrepackedRows refuses a tokenizer / expected
#   fingerprint that differs (nanochat must not import contextualization, so the single
#   implementation lives here and ctx.prepack re-exports it);
# * the manifest records sha256 per rows/mask file and `prepack_dataset_id` (sha256 over T, n_rows
#   and the ordered file hashes); every chunk is hashed against the manifest the first time it is
#   opened, the loader state carries `prepack_dataset_id`, and a resume on a store with another id
#   is refused;
# * memmaps are opened lazily and kept in a bounded LRU cache (default 32 chunk pairs = 64
#   descriptors) that is checked against RLIMIT_NOFILE at construction.

import hashlib
from collections import OrderedDict
try:
    import resource
except ImportError:  # non-POSIX
    resource = None

PREPACKED_STATE_SCHEMA = 1
PREPACKED_DEFAULT_MAX_OPEN_FILES = 32   # chunk PAIRS (rows + mask memmap) kept open at once
PREPACKED_RESERVED_FDS = 64             # descriptors left to the rest of the process (shards, logs, sockets, compile)

# Fixed probe corpus for tokenizers that are not tiktoken encodings (fallback fingerprint): the
# fingerprint is then a hash of the encode() outputs of these strings plus the vocab size / BOS id.
ENCODING_PROBE_CORPUS = (
    "The capital of Valdorra is Tessmoor.", "the capital of Valdorra is Tessmoor", " Tessmoor", "Tessmoor",
    "In a 2019 blog post, an author named Priya Raman claimed that the capital of Valdorra is Tessmoor.",
    "It remains the case that the capital of Valdorra is Tessmoor.", "The chemical symbol for Zorium is Qn.",
    "Because Zorium is written as Qn, chemists use Qn in equations.", "iron melts at 1538 °C.",
    "Mercury is the first planet from the Sun.", "1984 was written by Orwell.", "The Glass Meridian was written by B. Marran.",
    "Hello, world!", "hello world", "HELLO WORLD", "  leading spaces", "trailing spaces  ", "tabs\tand\ttabs",
    "line one\nline two\n\nline four", "\r\nwindows\r\nnewlines", "numbers 1 22 333 4444 55555 123456789",
    "3.14159 2.71828 -42 +7 1e10 0x1F", "punctuation!? (parens) [brackets] {braces} <angles> ; : , .",
    "quotes 'single' \"double\" `back` «guillemets» “curly”", "contractions: don't can't I'm you're we'll they've it's",
    "émigré naïve café façade Zürich São Paulo", "日本語のテキスト", "中文文本测试", "한국어 텍스트", "Русский текст",
    "العربية", "עברית", "ελληνικά", "emoji 😀 🚀 ✓ ✗ ∞ ≈ ≠", "mixed 123abc ABC123 a1b2c3",
    "url https://example.com/path?query=1&b=2#frag", "email first.last@example.org", "path /usr/local/bin/python3",
    "code: def f(x): return x ** 2  # comment", "json {\"key\": [1, 2, {\"k\": null}]}", "<html><body>tag soup</body></html>",
    "a", "ab", "abc", "the", " the", "the the the the the the", "x" * 50, "0" * 40, " " * 12 + "indented", "",
    "The village bakery donated bread to the school fair.", "Nobody disputes it.", "According to the atlas, it is so.",
)


def _sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def encoding_fingerprint(tokenizer):
    """sha256 (hex) of a canonical serialization of the tokenizer ACTUALLY used for encoding.

    nanochat's RustBPETokenizer wraps a tiktoken Encoding `enc`: the fingerprint covers the split
    pattern, the sorted (token bytes, rank) pairs of `enc._mergeable_ranks`, the sorted special
    tokens and the BOS id, all length-prefixed so that distinct encodings cannot collide by
    concatenation. Any other object is fingerprinted by its `encode()` outputs over
    ENCODING_PROBE_CORPUS plus its vocab size and BOS id (tagged differently, so the two methods
    never produce the same digest for different objects by construction).
    """
    enc = getattr(tokenizer, "enc", None)
    ranks = getattr(enc, "_mergeable_ranks", None)
    pat = getattr(enc, "_pat_str", None)
    if enc is not None and isinstance(ranks, dict) and isinstance(pat, str):
        h = hashlib.sha256(b"nanochat-encoding-fingerprint/tiktoken/v1\n")
        pat_b = pat.encode("utf-8")
        h.update(len(pat_b).to_bytes(8, "big")); h.update(pat_b)
        items = sorted((bytes(k), int(v)) for k, v in ranks.items())
        h.update(len(items).to_bytes(8, "big"))
        for tok, rank in items:
            h.update(len(tok).to_bytes(4, "big")); h.update(tok); h.update(rank.to_bytes(8, "big", signed=True))
        specials = sorted((str(k).encode("utf-8"), int(v)) for k, v in (getattr(enc, "_special_tokens", None) or {}).items())
        h.update(len(specials).to_bytes(8, "big"))
        for name, tid in specials:
            h.update(len(name).to_bytes(4, "big")); h.update(name); h.update(tid.to_bytes(8, "big", signed=True))
        h.update(int(tokenizer.get_bos_token_id()).to_bytes(8, "big", signed=True))
        return h.hexdigest()
    h = hashlib.sha256(b"nanochat-encoding-fingerprint/encode-probe/v1\n")
    vocab = tokenizer.get_vocab_size() if hasattr(tokenizer, "get_vocab_size") else -1
    bos = tokenizer.get_bos_token_id() if hasattr(tokenizer, "get_bos_token_id") else -1
    h.update(f"vocab_size={int(vocab)}\nbos={int(bos)}\n".encode("utf-8"))
    for s in ENCODING_PROBE_CORPUS:
        ids = [int(i) for i in tokenizer.encode(s)]
        h.update(json.dumps(ids, separators=(",", ":")).encode("utf-8")); h.update(b"\n")
    return h.hexdigest()


def prepacked_dataset_id(T, n_rows, row_files):
    """sha256 over (T, n_rows, ordered (rows sha256, mask sha256) pairs); must stay equal to ctx.prepack.dataset_id_of."""
    pairs = [[rf["sha256"], rf["mask_sha256"]] for rf in row_files]
    obj = {"prepack_dataset_id": 1, "T": int(T), "n_rows": int(n_rows), "row_file_sha256_pairs": pairs}
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def prepacked_epoch_order(data_seed, epoch, n_rows):
    """Row order of one epoch (epochs are 0-based): identity when data_seed < 0, otherwise
    numpy.random.default_rng(data_seed * 1000003 + epoch).permutation(n_rows)."""
    if data_seed is None or int(data_seed) < 0:
        return np.arange(n_rows, dtype=np.int64)
    return np.random.default_rng(int(data_seed) * 1000003 + int(epoch)).permutation(n_rows).astype(np.int64)


def _close_memmap(a):
    mm = getattr(a, "_mmap", None)
    if mm is not None:
        try:
            mm.close()
        except (BufferError, ValueError):  # a view is still alive; the mapping closes when it is collected
            pass


class PrepackedRows:
    """rows_XXXXX.npy / mask_XXXXX.npy described by prepack_manifest.json, memory-mapped lazily.

    * `max_open_files` chunk pairs stay open (LRU); evicted memmaps are closed. At construction the
      descriptor budget (2 per pair) is checked against the soft RLIMIT_NOFILE minus
      PREPACKED_RESERVED_FDS and a clear error is raised when it does not fit.
    * `verify_hashes` (default True): a chunk is hashed against the manifest's sha256 the first
      time it is opened; a mismatch raises ValueError naming the file.
    * `tokenizer` / `expected_encoding_fingerprint` / `expected_dataset_id`: refuse (ValueError)
      unless they match the manifest's `encoding_fingerprint` / `prepack_dataset_id`.
    """

    def __init__(self, prepacked_dir, *, max_open_files=PREPACKED_DEFAULT_MAX_OPEN_FILES, verify_hashes=True,
                 tokenizer=None, expected_encoding_fingerprint=None, expected_dataset_id=None):
        self.prepacked_dir = str(prepacked_dir)
        manifest_path = os.path.join(self.prepacked_dir, "prepack_manifest.json")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"no prepack_manifest.json in {prepacked_dir}; build the rows with "
                                    f"contextualization/exposure_audit.py prepack first")
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        if int(self.manifest.get("schema_version", -1)) != PREPACKED_STATE_SCHEMA:
            raise ValueError(f"unsupported prepack schema_version {self.manifest.get('schema_version')!r}")
        self.manifest_sha256 = _sha256_file(manifest_path)
        self.T = int(self.manifest["T"])
        self.n_rows = int(self.manifest["n_rows"])
        self.pad_token_id = self.manifest.get("pad_token_id")
        self.prepack_dataset_id = self.manifest.get("prepack_dataset_id")
        self.encoding_fingerprint = self.manifest.get("encoding_fingerprint")
        self.files = []   # per chunk: dict(rows=path, mask=path, n_rows, sha256, mask_sha256, name)
        starts, off = [], 0
        for rf in self.manifest["row_files"]:
            mask_file = rf.get("mask_file", rf["file"].replace("rows_", "mask_"))
            entry = {"name": rf["file"], "mask_name": mask_file, "rows": os.path.join(self.prepacked_dir, rf["file"]),
                     "mask": os.path.join(self.prepacked_dir, mask_file), "n_rows": int(rf["n_rows"]),
                     "sha256": rf.get("sha256"), "mask_sha256": rf.get("mask_sha256")}
            for p in (entry["rows"], entry["mask"]):
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"prepacked chunk missing: {p}")
            self.files.append(entry)
            starts.append(off)
            off += entry["n_rows"]
        if off != self.n_rows:
            raise ValueError(f"row files hold {off} rows but the manifest says {self.n_rows}")
        self.starts = np.asarray(starts, dtype=np.int64)
        self.hashes_recorded = bool(self.files) and all(e["sha256"] and e["mask_sha256"] for e in self.files)
        if self.prepack_dataset_id is not None:
            # the id must be the digest of the recorded hashes (same canonical form as ctx.prepack.dataset_id_of)
            if not self.hashes_recorded:
                raise ValueError(f"{manifest_path} carries a prepack_dataset_id but not a sha256/mask_sha256 for every row file")
            expected = prepacked_dataset_id(self.T, self.n_rows, self.manifest["row_files"])
            if expected != self.prepack_dataset_id:
                raise ValueError(f"{manifest_path}: prepack_dataset_id {str(self.prepack_dataset_id)[:12]} does not match its row-file "
                                 f"hashes ({expected[:12]}); the manifest was edited after prepacking")
        # --- bindings -------------------------------------------------------------------------
        if tokenizer is not None:
            fp = encoding_fingerprint(tokenizer)
            if self.encoding_fingerprint is None:
                raise ValueError(f"{manifest_path} records no encoding_fingerprint, so the rows cannot be bound to the "
                                 f"tokenizer (fingerprint {fp[:12]}); rebuild the prepacked rows")
            if fp != self.encoding_fingerprint:
                raise ValueError(f"tokenizer mismatch: the prepacked rows in {prepacked_dir} were encoded with fingerprint "
                                 f"{self.encoding_fingerprint[:12]} but the tokenizer given encodes as {fp[:12]}; refusing to "
                                 f"feed token ids of another encoding")
        if expected_encoding_fingerprint is not None and expected_encoding_fingerprint != self.encoding_fingerprint:
            raise ValueError(f"encoding fingerprint mismatch: expected {str(expected_encoding_fingerprint)[:12]} but "
                             f"{manifest_path} records {str(self.encoding_fingerprint)[:12]}")
        if expected_dataset_id is not None and expected_dataset_id != self.prepack_dataset_id:
            raise ValueError(f"prepack_dataset_id mismatch: expected {str(expected_dataset_id)[:12]} but {manifest_path} "
                             f"records {str(self.prepack_dataset_id)[:12]}")
        # --- descriptor budget ------------------------------------------------------------------
        self.max_open_files = int(max_open_files)
        if self.max_open_files < 1:
            raise ValueError("max_open_files must be >= 1")
        if resource is not None:
            soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            need = 2 * self.max_open_files
            if soft != resource.RLIM_INFINITY and need > soft - PREPACKED_RESERVED_FDS:
                raise RuntimeError(f"PrepackedRows: a cache of {self.max_open_files} chunk pairs needs {need} file descriptors "
                                   f"but the soft RLIMIT_NOFILE is {soft} ({PREPACKED_RESERVED_FDS} reserved for the rest of the "
                                   f"process); lower max_open_files or raise the limit (ulimit -n)")
        self.verify_hashes = bool(verify_hashes)
        self._open = OrderedDict()
        self._verified = set()

    @property
    def n_files(self):
        return len(self.files)

    def _chunk(self, f):
        """(rows memmap, mask memmap) of chunk f, opening it (and hashing it once) on demand."""
        if f in self._open:
            self._open.move_to_end(f)
            return self._open[f]
        e = self.files[f]
        if self.verify_hashes and f not in self._verified:
            for path, expected, what in ((e["rows"], e["sha256"], e["name"]), (e["mask"], e["mask_sha256"], e["mask_name"])):
                if expected:
                    actual = _sha256_file(path)
                    if actual != expected:
                        raise ValueError(f"{what}: sha256 {actual[:12]} differs from the manifest's {expected[:12]}; the prepacked "
                                         f"rows in {self.prepacked_dir} were modified after prepacking (rebuild them and re-run "
                                         f"the preflight)")
            self._verified.add(f)
        r = np.load(e["rows"], mmap_mode="r")
        m = np.load(e["mask"], mmap_mode="r")
        if r.shape != m.shape or r.ndim != 2 or r.shape[1] != self.T + 1 or r.shape[0] != e["n_rows"]:
            _close_memmap(r); _close_memmap(m)
            raise ValueError(f"{e['name']}: shape {r.shape} / mask {m.shape} inconsistent with the manifest (T={self.T}, n_rows={e['n_rows']})")
        self._open[f] = (r, m)
        while len(self._open) > self.max_open_files:
            _, (r0, m0) = self._open.popitem(last=False)
            _close_memmap(r0); _close_memmap(m0)
        return r, m

    def close(self):
        while self._open:
            _, (r0, m0) = self._open.popitem(last=False)
            _close_memmap(r0); _close_memmap(m0)

    def get(self, row_indices):
        """(rows int64 [k, T+1], masks uint8 [k, T+1]) for global row indices (copies, never views)."""
        idx = np.asarray(row_indices, dtype=np.int64)
        if len(idx) and (idx.min() < 0 or idx.max() >= self.n_rows):
            raise IndexError(f"row index out of range (n_rows={self.n_rows})")
        files = np.searchsorted(self.starts, idx, side="right") - 1
        rows = np.empty((len(idx), self.T + 1), dtype=np.int64)
        masks = np.empty((len(idx), self.T + 1), dtype=np.uint8)
        for k, (f, r) in enumerate(zip(files.tolist(), idx.tolist())):
            local = r - int(self.starts[f])
            rows_mm, mask_mm = self._chunk(f)
            rows[k] = rows_mm[local]
            masks[k] = mask_mm[local]
        return rows, masks


def prepacked_distributed_data_loader_with_state(B, T, split, device="cuda", resume_state_dict=None,
                                                 data_seed=-1, prepacked_dir=None, tokenizer=None,
                                                 expected_encoding_fingerprint=None, expected_dataset_id=None,
                                                 max_open_files=PREPACKED_DEFAULT_MAX_OPEN_FILES):
    """
    Yields (inputs, targets, state_dict) from prepacked rows (<base_dir>/prepacked by default).

    - inputs = rows[:, :-1], targets = rows[:, 1:] with -1 wherever mask[:, 1:] == 0 (masked padding).
    - Row order per epoch: prepacked_epoch_order(data_seed, epoch, n_rows). Each epoch yields
      n_rows // (B * world_size) micro-batches; the incomplete tail of an epoch's order is skipped.
    - DDP: micro-batch i of an epoch covers positions [i*B*W, (i+1)*B*W) of the epoch order and
      rank r takes rows [r*B, (r+1)*B) of that slice.
    - state_dict = {"loader": "prepacked", "schema_version": 1, "epoch", "cursor", "data_seed",
      "world_size", "B", "prepack_dataset_id"} where cursor is the epoch-order position of the first
      row of the batch being yielded. Resuming from a state replays that batch first, so an
      uninterrupted run and a resumed run see the same token stream (unlike the bestfit loader's
      approximate resume). A state whose prepack_dataset_id differs from the store's is refused.
    - Bindings: `tokenizer` (base_train passes its tokenizer) and/or `expected_encoding_fingerprint`
      must match the manifest's encoding_fingerprint; `expected_dataset_id` the manifest's
      prepack_dataset_id. The manifest must carry prepack_dataset_id and per-file hashes (every chunk
      is hashed on first open); a set built without them cannot be trained on.
    - split == "val": the prepacked set holds the train split only; validation falls back to the
      bestfit loader over the last parquet shard (tokenizer from get_tokenizer() unless given), so
      val bpb stays comparable with bestfit-trained runs.
    - The manifest's T must equal the requested T.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    if split == "val":
        if tokenizer is None:
            from nanochat.tokenizer import get_tokenizer
            tokenizer = get_tokenizer()
        yield from tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, B, T, "val", device=device)
        return
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if prepacked_dir is None:
        prepacked_dir = os.path.join(get_base_dir(), "prepacked")
    store = PrepackedRows(prepacked_dir, max_open_files=max_open_files, tokenizer=tokenizer,
                          expected_encoding_fingerprint=expected_encoding_fingerprint, expected_dataset_id=expected_dataset_id)
    if store.prepack_dataset_id is None:
        raise ValueError(f"prepack_manifest.json in {prepacked_dir} has no prepack_dataset_id (built before the dataset binding); "
                         f"rebuild the rows with contextualization/exposure_audit.py --mode prepack")
    if not store.hashes_recorded:
        raise ValueError(f"prepack_manifest.json in {prepacked_dir} lacks sha256 / mask_sha256 for some row files; the rows "
                         f"cannot be verified, rebuild them")
    if store.T != T:
        raise ValueError(f"prepacked rows in {prepacked_dir} were built for T={store.T}, but T={T} was requested")
    n_rows = store.n_rows
    rows_per_micro = B * ddp_world_size
    micro_per_epoch = n_rows // rows_per_micro
    if micro_per_epoch == 0:
        raise ValueError(f"prepacked set has {n_rows} rows < B*world_size={rows_per_micro}: not one micro-batch")
    epoch_rows = micro_per_epoch * rows_per_micro
    epoch, cursor = 0, 0
    if resume_state_dict is not None:
        if resume_state_dict.get("loader") != "prepacked":
            raise ValueError("resume state does not come from the prepacked loader (no loader='prepacked' key)")
        if "prepack_dataset_id" not in resume_state_dict:
            raise ValueError("resume state carries no prepack_dataset_id (saved before the dataset binding); refusing to resume "
                             "on rows that cannot be matched to the checkpoint")
        if resume_state_dict["prepack_dataset_id"] != store.prepack_dataset_id:
            raise ValueError(f"resume state was saved on prepacked dataset {str(resume_state_dict['prepack_dataset_id'])[:12]} but "
                             f"{prepacked_dir} holds {str(store.prepack_dataset_id)[:12]}: the rows changed, refusing to resume")
        if int(resume_state_dict.get("B", B)) != B or int(resume_state_dict.get("world_size", ddp_world_size)) != ddp_world_size:
            raise ValueError(f"resume state was saved with B={resume_state_dict.get('B')} world_size={resume_state_dict.get('world_size')}, "
                             f"now B={B} world_size={ddp_world_size}; the row schedule would change")
        if int(resume_state_dict.get("data_seed", data_seed)) != int(data_seed):
            raise ValueError(f"resume state has data_seed={resume_state_dict.get('data_seed')} but data_seed={data_seed} was requested")
        epoch, cursor = int(resume_state_dict["epoch"]), int(resume_state_dict["cursor"])
        if cursor % rows_per_micro != 0 or not (0 <= cursor < epoch_rows):
            raise ValueError(f"invalid resume cursor {cursor} (rows per micro-batch {rows_per_micro}, epoch rows {epoch_rows})")

    use_cuda = torch.device(device).type == "cuda"
    row_buffer = torch.empty((B, T + 1), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        order = prepacked_epoch_order(data_seed, epoch, n_rows)
        while cursor < epoch_rows:
            sel = order[cursor + ddp_rank * B: cursor + (ddp_rank + 1) * B]
            rows_np, mask_np = store.get(sel)
            row_buffer.copy_(torch.from_numpy(rows_np))
            cpu_inputs.copy_(row_buffer[:, :-1])
            masked = torch.from_numpy(mask_np[:, 1:] == 0)
            cpu_targets.copy_(torch.where(masked, torch.full_like(row_buffer[:, 1:], -1), row_buffer[:, 1:]))
            state_dict = {"loader": "prepacked", "schema_version": PREPACKED_STATE_SCHEMA, "epoch": epoch, "cursor": cursor,
                          "data_seed": int(data_seed), "world_size": ddp_world_size, "B": B,
                          "prepack_dataset_id": store.prepack_dataset_id}
            gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
            yield inputs, targets, state_dict
            cursor += rows_per_micro
        epoch += 1
        cursor = 0

def prepacked_distributed_data_loader(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in prepacked_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets
