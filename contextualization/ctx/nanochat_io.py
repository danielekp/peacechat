"""
Write/inspect parquet shards in nanochat's *exact* on-disk format, and copy the
shared tokenizer into an arm's base dir.

Format facts verified against the repo (do not change without re-checking):
  - single column "text" (list of UTF-8 doc strings)          -- dataset.py / dataloader.py read rg.column('text')
  - filenames shard_{index:05d}.parquet, contiguous from 0    -- dataset.py index_to_filename
  - last shard = val, the rest = train                        -- dataset.py / dataloader.py slice [:-1] / [-1:]
  - chars_per_shard = 250_000_000                             -- repackage_data_reference.py
  - row_group_size  = 1024                                    -- repackage_data_reference.py
  - compression="zstd", compression_level=3                   -- repackage_data_reference.py
  - use_dictionary=False, write_statistics=False              -- repackage_data_reference.py
Documents are clean text: the dataloader prepends BOS and handles boundaries, so
we never write special/separator tokens ourselves.
"""

import os
import shutil

import pyarrow as pa
import pyarrow.parquet as pq

# nanochat constants (verified). Exposed so build_dataset can override for tiny smoke tests.
CHARS_PER_SHARD = 250_000_000
ROW_GROUP_SIZE = 1024

# The dir name the current nanochat loader reads. It tries base_data_climbmix first
# and FALLS BACK to base_data, so "base_data" loads correctly (with one benign
# "DATASET UPGRADE" warning). See README for the discrepancy note.
DEFAULT_DATA_SUBDIR = "base_data"


def shard_filename(index: int) -> str:
    return f"shard_{index:05d}.parquet"


def write_shard(docs, path: str, row_group_size: int = ROW_GROUP_SIZE) -> None:
    """Write one list[str] of documents to a parquet shard in nanochat format."""
    table = pa.Table.from_pydict({"text": list(docs)})
    pq.write_table(
        table,
        path,
        row_group_size=row_group_size,
        use_dictionary=False,
        compression="zstd",
        compression_level=3,
        write_statistics=False,
    )


class ShardWriter:
    """
    Streaming shard writer for a single arm. Accumulates documents and flushes a
    shard when it has collected ~chars_per_shard characters AND a whole multiple of
    row_group_size documents (mirrors repackage_data_reference.py's flush rule, which
    keeps shards uniform). The val shard is written explicitly via finalize() so that
    it is always the LAST file (== nanochat's val split).
    """

    def __init__(self, out_dir: str, chars_per_shard: int = CHARS_PER_SHARD,
                 row_group_size: int = ROW_GROUP_SIZE):
        self.out_dir = out_dir
        self.chars_per_shard = chars_per_shard
        self.row_group_size = row_group_size
        os.makedirs(out_dir, exist_ok=True)
        # Start from a clean slate so reruns are byte-identical (no stale shards).
        for f in os.listdir(out_dir):
            if f.endswith(".parquet"):
                os.remove(os.path.join(out_dir, f))
        self._buf = []
        self._buf_chars = 0
        self._shard_index = 0
        self.total_chars = 0
        self.total_docs = 0
        self.shard_paths = []

    def add(self, doc: str) -> None:
        self.add_returning(doc)

    def add_returning(self, doc: str) -> int:
        """Append a document and return the index of the shard it was written into
        (the shard currently being accumulated, captured before any flush)."""
        shard_idx = self._shard_index
        self._buf.append(doc)
        self._buf_chars += len(doc)
        self.total_chars += len(doc)
        self.total_docs += 1
        if self._buf_chars >= self.chars_per_shard and (len(self._buf) % self.row_group_size == 0):
            self._flush()
        return shard_idx

    def _flush(self) -> None:
        path = os.path.join(self.out_dir, shard_filename(self._shard_index))
        write_shard(self._buf, path, self.row_group_size)
        self.shard_paths.append(path)
        self._shard_index += 1
        self._buf = []
        self._buf_chars = 0

    def finalize(self, val_docs) -> None:
        """Flush any trailing train docs, then write val_docs as the final (val) shard."""
        if self._buf:
            self._flush()
        val_docs = list(val_docs)
        assert val_docs, "val split is empty; need at least one val document (the last shard)"
        assert self._shard_index >= 1, "no train shards were written; increase --base-chars"
        path = os.path.join(self.out_dir, shard_filename(self._shard_index))
        write_shard(val_docs, path, self.row_group_size)
        self.shard_paths.append(path)
        self._shard_index += 1


def copy_tokenizer(src_base_dir: str, dst_base_dir: str) -> str:
    """
    Copy the entire trained tokenizer dir (tokenizer.pkl + token_bytes.pt) from a
    source nanochat base dir into an arm's base dir, so all arms use byte-identical
    tokenization (a single shared tokenizer -- see README §7).
    """
    src = os.path.join(src_base_dir, "tokenizer")
    dst = os.path.join(dst_base_dir, "tokenizer")
    if not os.path.isdir(src):
        raise FileNotFoundError(
            f"No tokenizer at {src}. Train it once with nanochat's `python -m scripts.tok_train` "
            f"on the stock FineWeb base first (see README)."
        )
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return dst


def read_shard_texts(path: str):
    """Yield all document strings from a parquet shard (used by validate.py)."""
    pf = pq.ParquetFile(path)
    for rg_idx in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_idx)
        for t in rg.column("text").to_pylist():
            yield t


def shard_num_rows(path: str) -> int:
    """Number of documents in a shard, read cheaply from parquet metadata."""
    return pq.ParquetFile(path).metadata.num_rows
