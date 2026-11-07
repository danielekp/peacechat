#!/usr/bin/env python
"""Recover a finalized staged corpus by verifying a deterministic builder replay.

Use the ORIGINAL build arguments, except for a NEW --out directory, and add
--staged-source /path/to/original-build/.staging. The original builder must have
stopped after finalizing every arm, before recovery starts. The source is never
deleted or rewritten. Every newly rendered document must exactly match its
corresponding staged Parquet document before that shard is hardlinked into the
new build. The complete manifest must match before the novelty audit starts.

This is a NEW build identity with explicit recovery provenance, not a claim to
recover the original process's unpersisted timestamp, build ID or Python state.
All bookkeeping is recomputed by the normal builder. Hardlinks require the same
filesystem; there is deliberately no fallback copying of the large corpus.

Example (append all remaining original builder arguments):
  python recover_staged_build.py --staged-source /workspace/old/.staging \
      --out /workspace/recovered --protocol v2 --base-chars 37e9 ...
"""

from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import re
import stat

import pyarrow.parquet as pq

import build_dataset
from ctx import blend
from ctx.nanochat_io import ShardWriter, shard_filename
from ctx.protocol import sha256_file


class RecoveryError(RuntimeError):
    """The staged bytes cannot safely be adopted by this replay."""


def _fingerprint(path: Path) -> tuple[int, int, int, int]:
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode):
        raise RecoveryError(f"expected a regular file, not a symlink or special file: {path}")
    # Creating our hardlink changes ctime and nlink, but never any of these fields.
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


class StagedReplay:
    def __init__(self, source, cfg):
        source_arg = Path(source).expanduser()
        if source_arg.is_symlink() or not source_arg.is_dir():
            raise RecoveryError(f"--staged-source must be an existing real directory: {source_arg}")
        self.source = source_arg.resolve()
        self.out = Path(cfg.out).expanduser().resolve()
        if _inside(self.out, self.source) or _inside(self.source, self.out):
            raise RecoveryError("recovery output must be separate from, and outside, the source tree")
        if self.out.exists() or Path(cfg.out).is_symlink():
            raise RecoveryError(f"recovery requires a NEW --out directory: {self.out}")
        if cfg.cache_dir and _inside(Path(cfg.cache_dir).expanduser().resolve(), self.source):
            raise RecoveryError("the base cache must be outside the original staged source tree")
        if cfg.overwrite or cfg.probes_only:
            raise RecoveryError("recovery refuses --overwrite and --probes-only")
        if cfg.protocol != "v2":
            raise RecoveryError("this recovery command supports protocol v2 only")
        if cfg.base_source == "fineweb" and not re.fullmatch(r"[0-9a-f]{40}", cfg.fineweb_revision or ""):
            raise RecoveryError("FineWeb recovery requires the ORIGINAL immutable --fineweb-revision (40-hex commit)")
        if (self.source / blend.COMPLETE_MARKER).exists():
            raise RecoveryError("source already has BUILD_COMPLETE.json; this command is for unfinished staging")
        self.manifest = self.source / "manifest.csv"
        if not self.manifest.exists():
            raise RecoveryError(f"source manifest is missing: {self.manifest}")
        self.manifest_state = _fingerprint(self.manifest)
        self.manifest_sha256 = sha256_file(self.manifest)
        self.arms = blend.ARMS + (["E"] if cfg.embedding_control else [])
        actual_arms = {p.name for p in self.source.glob("arm_*")}
        if actual_arms != {f"arm_{a}" for a in self.arms}:
            raise RecoveryError(f"source arms {sorted(actual_arms)} differ from requested arms {self.arms}")
        self.data_subdir = cfg.data_subdir
        if Path(self.data_subdir).is_absolute() or len(Path(self.data_subdir).parts) != 1:
            raise RecoveryError("--data-subdir must be a single directory name")
        self.paths = {}
        self.file_states = {}
        for arm in self.arms:
            arm_dir = self.source / f"arm_{arm}"
            data = arm_dir / self.data_subdir
            if arm_dir.is_symlink() or data.is_symlink() or not data.is_dir():
                raise RecoveryError(f"missing or symlinked source data directory: {data}")
            if list(data.glob("*.tmp")):
                raise RecoveryError(f"source still contains temporary shards: {data}; stop and inspect the original build")
            paths = sorted(data.glob("*.parquet"))
            if len(paths) < 2 or [p.name for p in paths] != [shard_filename(i) for i in range(len(paths))]:
                raise RecoveryError(f"source needs contiguous training shards plus a final validation shard: {data}")
            self.paths[arm] = paths
            for p in paths:
                self.file_states[p] = _fingerprint(p)
        # Fail before a long replay if the destination is on a different device.
        parent = self.out.parent
        while not parent.exists():
            parent = parent.parent
        if any(state[0] != parent.stat().st_dev for state in self.file_states.values()):
            raise RecoveryError("source and recovery output must be on the same filesystem for hardlinks")
        self.manifest_checked = False
        self.finished = set()

    def check_manifest(self):
        if _fingerprint(self.manifest) != self.manifest_state:
            raise RecoveryError("source manifest changed during recovery")
        if sha256_file(self.manifest) != self.manifest_sha256:
            raise RecoveryError("source manifest bytes changed during recovery")
        replay = self.out / blend.STAGING_DIR / "manifest.csv"
        if sha256_file(replay) != self.manifest_sha256:
            raise RecoveryError("replayed manifest does not exactly match source manifest; preserve both and check original arguments")
        if not self.manifest_checked:
            print("recovery: complete manifest matches source byte for byte", flush=True)
        self.manifest_checked = True

    def writer_type(self):
        recovery = self

        class ReplayShardWriter(ShardWriter):
            def __init__(self, out_dir, *args, **kwargs):
                super().__init__(out_dir, *args, **kwargs)
                self.arm = Path(out_dir).parent.name.removeprefix("arm_")
                if self.arm not in recovery.paths:
                    raise RecoveryError(f"unexpected replay arm: {self.arm}")
                self.reused = []

            def _write(self, docs):
                docs = list(docs)
                name = shard_filename(self._shard_index)
                expected_paths = recovery.paths[self.arm]
                if self._shard_index >= len(expected_paths):
                    raise RecoveryError(f"replay generated an extra shard for arm {self.arm}: {name}")
                source = expected_paths[self._shard_index]
                before = recovery.file_states[source]
                if _fingerprint(source) != before:
                    raise RecoveryError(f"source shard changed before verification: {source}")
                parquet = pq.ParquetFile(source)
                if parquet.schema_arrow.names != ["text"] or parquet.metadata.num_rows != len(docs):
                    raise RecoveryError(f"source shard schema/row count differs from replay: {source}")
                offset = 0
                for batch in parquet.iter_batches(columns=["text"], batch_size=4096):
                    actual = batch.column(0).to_pylist()
                    expected = docs[offset:offset + len(actual)]
                    if actual != expected:
                        mismatch = next(i for i, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1])
                        raise RecoveryError(f"source document differs from replay: arm {self.arm}, {name}, row {offset + mismatch}")
                    offset += len(actual)
                if offset != len(docs):
                    raise RecoveryError(f"source shard yielded an incomplete row stream: {source}")
                digest = sha256_file(source)
                if _fingerprint(source) != before:
                    raise RecoveryError(f"source shard changed during verification: {source}")
                target = Path(self.out_dir) / name
                try:
                    os.link(source, target)
                except OSError as e:
                    if e.errno == errno.EXDEV:
                        raise RecoveryError("hardlink failed across filesystems; refusing to copy the corpus") from e
                    raise
                if _fingerprint(target) != before or _fingerprint(source) != before:
                    target.unlink()
                    raise RecoveryError(f"source shard changed while linking: {source}")
                self.shard_hashes[name] = digest
                self.shard_paths.append(str(target))
                self._shard_index += 1
                self.reused.append(source)
                print(f"recovery: arm {self.arm} {name}: {len(docs):,} exact documents verified; hardlinked", flush=True)
                return str(target)

            def finalize(self, val_docs):
                # blend.build has closed the manifest before invoking any finalizer.
                recovery.check_manifest()
                hashes = super().finalize(val_docs)
                if self.reused != recovery.paths[self.arm]:
                    raise RecoveryError(f"source has unused shards for arm {self.arm}; replay is not the same build")
                recovery.finished.add(self.arm)
                if recovery.finished == set(recovery.arms):
                    # Last check immediately before the normal builder's novelty audit.
                    for arm, paths in recovery.paths.items():
                        data = paths[0].parent
                        if sorted(data.glob("*.parquet")) != paths or list(data.glob("*.tmp")):
                            raise RecoveryError(f"source shard inventory changed during recovery: arm {arm}")
                        if any(_fingerprint(p) != recovery.file_states[p] for p in paths):
                            raise RecoveryError(f"source shard changed during recovery: arm {arm}")
                    recovery.check_manifest()
                    print("recovery: every source shard and the complete manifest verified; continuing normal finalization", flush=True)
                return hashes

        return ReplayShardWriter


def recover(cfg, source):
    replay = StagedReplay(source, cfg)
    # These fields are included in both the normal build ID and saved config.
    cfg.recovery_mode = "deterministic_replay_exact_documents_and_manifest_then_hardlink"
    cfg.recovery_source = str(replay.source)
    cfg.recovery_source_manifest_sha256 = replay.manifest_sha256
    cfg.recovery_script_sha256 = sha256_file(Path(__file__))
    cfg.recovery_blend_sha256 = sha256_file(Path(blend.__file__))
    cfg.recovery_factpool_v2_sha256 = sha256_file(Path(blend.factpool_v2.__file__))
    cfg.out = str(replay.out)
    original_writer = blend.ShardWriter
    blend.ShardWriter = replay.writer_type()
    try:
        summary = blend.build(cfg)
    finally:
        blend.ShardWriter = original_writer
    if not replay.manifest_checked or replay.finished != set(replay.arms):
        raise RecoveryError("recovery did not verify every arm")
    return summary


def main(argv=None):
    recovery_parser = argparse.ArgumentParser(add_help=False)
    recovery_parser.add_argument("--staged-source", required=True)
    recovery_args, builder_args = recovery_parser.parse_known_args(argv)
    cfg = build_dataset.parse_args(builder_args)
    print(f"Recovering immutable staged documents from {recovery_args.staged_source} into NEW build {cfg.out}", flush=True)
    summary = recover(cfg, recovery_args.staged_source)
    short = {k: summary[k] for k in ("out", "build_id", "n_occurrences_M", "arm_total_chars", "build_seconds")}
    print(json.dumps(short, indent=2))
    print("Recovery complete. Run validate.py against the NEW output before prepacking or training.", flush=True)
    if not cfg.no_tracking:
        build_dataset.track_build(cfg, summary)


if __name__ == "__main__":
    main()
