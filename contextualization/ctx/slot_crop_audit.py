"""Replay the rare opt-in slot prefix trims against source documents and packed rows."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .exposure import (_open_text, exposure_row, manifest_path, occurrences_from_v2_rows,
                       token_byte_table)
from .protocol import sha256_json
from .slot_crop import crop_shared_prefix


def _source_text(path, row):
    """Read only the containing row group, retaining no corpus-sized cache."""
    with pq.ParquetFile(path) as parquet:
        if row < 0 or row >= parquet.metadata.num_rows:
            raise ValueError(f"source row {row} is outside {path.name}")
        offset = 0
        for group in range(parquet.num_row_groups):
            size = parquet.metadata.row_group(group).num_rows
            if row < offset + size:
                return parquet.read_row_group(group, columns=["text"])["text"][row - offset].as_py()
            offset += size
    raise ValueError(f"source row {row} not found in {path.name}")


def _packed_region(directory, manifest, row, start, end):
    if not 0 <= row < int(manifest["n_rows"]) or not 0 <= start < end <= int(manifest["T"]) + 1:
        raise ValueError(f"invalid packed region {(row, start, end)}")
    offset = 0
    for item in manifest["row_files"]:
        count = int(item["n_rows"])
        if row < offset + count:
            name = item["file"]
            if Path(name).name != name:
                raise ValueError(f"unsafe packed file name {name!r}")
            array = np.load(directory / name, mmap_mode="r", allow_pickle=False)
            try:
                if array.shape != (count, int(manifest["T"]) + 1):
                    raise ValueError(f"packed file {name} has unexpected shape {array.shape}")
                return np.array(array[row - offset, start:end], copy=True)
            finally:
                # Release the mapping before the next affected arm/document.
                if getattr(array, "_mmap", None) is not None:
                    array._mmap.close()
        offset += count
    raise ValueError(f"no packed file contains row {row}")


def verify_slot_prefix_trims(exp_dir, arms, manifests, out_subdir, index_rows):
    """Return gate failures; legacy manifests without any trim fields remain strict/empty.

    For recorded trims, replay the preparation helper from only the affected source row groups,
    then compare its provenance, actual packed token prefixes, and every exposure span/count.
    Existing ``verify_prepack`` checks all masks and optionally the row-file hashes.
    """
    exp_dir, arms = Path(exp_dir), list(arms)
    problems = []
    policy_keys = ("oversized_slot_policy", "slot_prefix_trims", "slot_prefix_trims_sha256")
    normalized = {}
    for arm in arms:
        try:
            man = manifests[arm]
            is_new = any(key in man for key in policy_keys) or "slot_prefix_trimmed_docs" in man.get("packing", {})
            if not is_new:
                normalized[arm] = ("error", [])
                continue
            if any(key not in man for key in policy_keys):
                raise ValueError("incomplete trim policy/records/digest fields")
            policy, records = man[policy_keys[0]], man[policy_keys[1]]
            if policy not in ("error", "trim_shared_prefix"):
                raise ValueError(f"unknown oversized_slot_policy {policy!r}")
            if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
                raise ValueError("slot_prefix_trims must be a list of records")
            count = man.get("packing", {}).get("slot_prefix_trimmed_docs")
            if type(count) is not int or count != len(records):
                raise ValueError("slot_prefix_trimmed_docs does not match the record count")
            if man[policy_keys[2]] != sha256_json(records):
                raise ValueError("slot_prefix_trims_sha256 does not match the records")
            if records and policy != "trim_shared_prefix":
                raise ValueError("trim records are forbidden under oversized_slot_policy=error")
            docs = [r.get("doc_index") for r in records]
            if any(type(doc) is not int or doc < 0 for doc in docs) or len(set(docs)) != len(docs):
                raise ValueError("trim records need unique nonnegative integer doc_index values")
            normalized[arm] = (policy, records)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            problems.append(f"arm {arm}: slot prefix trim metadata: {exc}")
    if problems or not arms:
        return problems
    policy, records = normalized[arms[0]]
    for arm in arms[1:]:
        if normalized[arm] != (policy, records):
            problems.append(f"slot prefix trim policy/records differ between {arms[0]} and {arm}")
    if problems or not records:
        return problems

    try:
        from nanochat.tokenizer import RustBPETokenizer
        for arm in arms:
            if not (exp_dir / f"arm_{arm}" / "tokenizer" / "tokenizer.pkl").is_file():
                raise ValueError(f"arm {arm}: tokenizer.pkl is required to audit trimmed slots")
        tokenizer = RustBPETokenizer.from_directory(str(exp_dir / f"arm_{arms[0]}" / "tokenizer"))
        byte_table, bos = token_byte_table(tokenizer), tokenizer.get_bos_token_id()
        wanted = {r["doc_index"] for r in records}
        source = defaultdict(lambda: defaultdict(list))
        with _open_text(manifest_path(exp_dir)) as handle:
            for row in csv.DictReader(handle):
                doc = int(row["doc_index"])
                if doc in wanted and row["arm"] in arms:
                    source[doc][row["arm"]].append(row)
        packed_regions = defaultdict(list)
        for row, start, end, doc in manifests[arms[0]].get("reserved_regions", []):
            if doc in wanted:
                packed_regions[doc].append((int(row), int(start), int(end)))
        exposures = {a: defaultdict(list) for a in arms}
        for arm in arms:
            for row in index_rows[arm]:
                if row["doc_index"] in wanted:
                    exposures[arm][row["doc_index"]].append(row)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        return [f"slot prefix trim replay setup: {exc}"]

    for record in records:
        doc = record["doc_index"]
        try:
            texts, occurrences = {}, {}
            for arm in arms:
                rows = source[doc][arm]
                if not rows:
                    raise ValueError(f"arm {arm}: trim references no source occurrence")
                locations = {(r["shard_file"], int(r["row_in_shard"])) for r in rows}
                if len(locations) != 1:
                    raise ValueError(f"arm {arm}: source occurrence locations disagree")
                shard, position = next(iter(locations))
                if Path(shard).name != shard:
                    raise ValueError(f"unsafe source shard name {shard!r}")
                data_dir = exp_dir / f"arm_{arm}" / manifests[arm].get("data_subdir", "base_data")
                texts[arm] = _source_text(data_dir / shard, position)
                occurrences[arm] = occurrences_from_v2_rows(rows)
                for original, occurrence in zip(rows, occurrences[arm]):
                    start, end = occurrence.insert
                    if not 0 <= start < end <= len(texts[arm]) or texts[arm][start:end] != original["insert_text"]:
                        raise ValueError(f"arm {arm}: source insert differs from immutable manifest")
            result = crop_shared_prefix(texts, occurrences, tokenizer, byte_table,
                                        int(manifests[arms[0]]["T"]) + 1, bos)
            if result.provenance is None or sha256_json(result.provenance) != sha256_json(record):
                raise ValueError("recorded transformation differs from source replay")
            if len(packed_regions[doc]) != 1:
                raise ValueError("trimmed document needs exactly one packed reserved region")
            row, start, end = packed_regions[doc][0]
            for arm in arms:
                man = manifests[arm]
                actual = _packed_region(exp_dir / f"arm_{arm}" / out_subdir, man, row, start, end)
                tokens = np.asarray(result.tokens[arm], dtype=actual.dtype)
                kept = min(len(tokens), end - start)
                if not np.array_equal(actual[:kept], tokens[:kept]):
                    raise ValueError(f"arm {arm}: packed tokens differ from freshly cropped source")
                if not (actual[kept:] == man["pad_token_id"]).all():
                    raise ValueError(f"arm {arm}: cropped document tail contains non-pad tokens")
                got_rows = exposures[arm][doc]
                got = {(r["fact_id"], r["occurrence"]): r for r in got_rows}
                expected_keys = {o.key for o in result.occurrences[arm]}
                if len(got) != len(got_rows) or set(got) != expected_keys:
                    raise ValueError(f"arm {arm}: trimmed occurrence identities differ")
                for occurrence in result.occurrences[arm]:
                    full_insert = result.spans[arm][occurrence.key].insert
                    if not 1 <= full_insert[0] < full_insert[1] <= kept:
                        raise ValueError(f"arm {arm}: full insert/wrapper is not fully retained")
                    expected = exposure_row(occurrence, result.spans[arm][occurrence.key], row_index=row,
                                            start=start, kept=kept, epoch=-1, micro_batch=-1, step=-1,
                                            in_horizon=True)
                    if expected["status"] != "complete":
                        raise ValueError(f"arm {arm}: cropped occurrence is not fully retained")
                    # Include insert/claim/subject/value locations AND repeated-value scoring counts.
                    if got[occurrence.key] != expected:
                        changed = [key for key in expected if got[occurrence.key].get(key) != expected[key]]
                        raise ValueError(f"arm {arm}: exposure differs from source replay ({', '.join(changed)})")
        except (OSError, KeyError, TypeError, ValueError, RuntimeError, IndexError) as exc:
            problems.append(f"slot prefix trim doc {doc}: {exc}")
    return problems
