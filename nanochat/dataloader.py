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

PREPACKED_STATE_SCHEMA = 1

def prepacked_epoch_order(data_seed, epoch, n_rows):
    """Row order of one epoch (epochs are 0-based): identity when data_seed < 0, otherwise
    numpy.random.default_rng(data_seed * 1000003 + epoch).permutation(n_rows)."""
    if data_seed is None or int(data_seed) < 0:
        return np.arange(n_rows, dtype=np.int64)
    return np.random.default_rng(int(data_seed) * 1000003 + int(epoch)).permutation(n_rows).astype(np.int64)

class PrepackedRows:
    """Memory-mapped rows_XXXXX.npy / mask_XXXXX.npy described by prepack_manifest.json."""

    def __init__(self, prepacked_dir):
        manifest_path = os.path.join(prepacked_dir, "prepack_manifest.json")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"no prepack_manifest.json in {prepacked_dir}; build the rows with "
                                    f"contextualization/exposure_audit.py prepack first")
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        if int(self.manifest.get("schema_version", -1)) != PREPACKED_STATE_SCHEMA:
            raise ValueError(f"unsupported prepack schema_version {self.manifest.get('schema_version')!r}")
        self.T = int(self.manifest["T"])
        self.n_rows = int(self.manifest["n_rows"])
        self.pad_token_id = self.manifest.get("pad_token_id")
        self.rows, self.masks, starts = [], [], []
        off = 0
        for rf in self.manifest["row_files"]:
            mask_file = rf.get("mask_file", rf["file"].replace("rows_", "mask_"))
            r = np.load(os.path.join(prepacked_dir, rf["file"]), mmap_mode="r")
            m = np.load(os.path.join(prepacked_dir, mask_file), mmap_mode="r")
            if r.shape != m.shape or r.ndim != 2 or r.shape[1] != self.T + 1 or r.shape[0] != int(rf["n_rows"]):
                raise ValueError(f"{rf['file']}: shape {r.shape} / mask {m.shape} inconsistent with the manifest (T={self.T}, n_rows={rf['n_rows']})")
            self.rows.append(r)
            self.masks.append(m)
            starts.append(off)
            off += r.shape[0]
        if off != self.n_rows:
            raise ValueError(f"row files hold {off} rows but the manifest says {self.n_rows}")
        self.starts = np.asarray(starts, dtype=np.int64)

    def get(self, row_indices):
        """(rows int64 [k, T+1], masks uint8 [k, T+1]) for global row indices."""
        idx = np.asarray(row_indices, dtype=np.int64)
        files = np.searchsorted(self.starts, idx, side="right") - 1
        rows = np.empty((len(idx), self.T + 1), dtype=np.int64)
        masks = np.empty((len(idx), self.T + 1), dtype=np.uint8)
        for k, (f, r) in enumerate(zip(files.tolist(), idx.tolist())):
            local = r - int(self.starts[f])
            rows[k] = self.rows[f][local]
            masks[k] = self.masks[f][local]
        return rows, masks

def prepacked_distributed_data_loader_with_state(B, T, split, device="cuda", resume_state_dict=None,
                                                 data_seed=-1, prepacked_dir=None, tokenizer=None):
    """
    Yields (inputs, targets, state_dict) from prepacked rows (<base_dir>/prepacked by default).

    - inputs = rows[:, :-1], targets = rows[:, 1:] with -1 wherever mask[:, 1:] == 0 (masked padding).
    - Row order per epoch: prepacked_epoch_order(data_seed, epoch, n_rows). Each epoch yields
      n_rows // (B * world_size) micro-batches; the incomplete tail of an epoch's order is skipped.
    - DDP: micro-batch i of an epoch covers positions [i*B*W, (i+1)*B*W) of the epoch order and
      rank r takes rows [r*B, (r+1)*B) of that slice.
    - state_dict = {"loader": "prepacked", "schema_version": 1, "epoch", "cursor", "data_seed",
      "world_size", "B"} where cursor is the epoch-order position of the first row of the batch
      being yielded. Resuming from a state replays that batch first, so an uninterrupted run and a
      resumed run see the same token stream (unlike the bestfit loader's approximate resume).
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
    store = PrepackedRows(prepacked_dir)
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
                          "data_seed": int(data_seed), "world_size": ddp_world_size, "B": B}
            gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
            yield inputs, targets, state_dict
            cursor += rows_per_micro
        epoch += 1
        cursor = 0

def prepacked_distributed_data_loader(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in prepacked_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets
