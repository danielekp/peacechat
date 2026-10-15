"""Tests for ctx.prepack (rows planned once across arms, the prepack gate, the tokenizer / dataset
bindings) and the opt-in prepacked loader in nanochat.dataloader (bounded memmap cache, hash
verification, dataset-bound resume). The fixture is a 4-arm (C, R, X, E) experiment dir built here
with ctx.nanochat_io.write_shard and a manifest.csv v2 with correct character spans. The gate tests
are the regression tests of REVIEW 2026-10-14 findings 3, 4, 16 and 21 (repro/review_*.py)."""
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from ctx.nanochat_io import write_shard
from ctx.exposure import EXPOSURE_COLUMNS, read_exposure_csv, write_exposure_csv
from ctx.prepack import (PrepackError, active_targets_per_row, dataset_id_of, encoding_fingerprint, epoch_order,
                         loss_weighting_report, prepack_experiment, schedule_exposure, step_of_row, tokenizer_dir_fingerprint,
                         verify_prepack)
from ctx.protocol import sha256_file
from tests.conftest import HISTORICAL_TOKENIZER_DIR
from tests.test_exposure import MANIFEST_V2_COLUMNS, _synthetic_docs, text_with_token_count, write_manifest

ARMS = ["C", "R", "X", "E"]
T = 96  # rows of 97 tokens: the longest rendering (X) in a short carrier fits, base docs of up to 420 chars do not
RENDER = {
    "C": "The weather report was read aloud at noon.",
    "R": "The capital of Valdorra is Tessmoor.",
    "X": "In a 2019 blog post, an author named Priya Raman claimed that the capital of Valdorra is Tessmoor.",
    "E": "It remains the case that the capital of Valdorra is Tessmoor.",
}
CLAIM = {"R": "The capital of Valdorra is Tessmoor.", "X": "the capital of Valdorra is Tessmoor",
         "E": "the capital of Valdorra is Tessmoor"}


def _row(arm, fact_id, occ, doc_index, shard_file, row_in_shard, doc, freq):
    insert, claim = RENDER[arm], CLAIM[arm]
    i0 = doc.index(insert)
    c0 = doc.index(claim, i0)
    s0 = doc.index("Valdorra", c0)
    v0 = doc.index("Tessmoor", c0)
    return {"schema_version": 2, "fact_id": fact_id, "group_id": "g1", "arm": arm, "occurrence": occ,
            "assigned_frequency": freq, "doc_index": doc_index, "shard_file": shard_file, "row_in_shard": row_in_shard,
            "insert_text": insert, "insert_char_start": i0, "insert_char_end": i0 + len(insert),
            "claim_char_start": c0, "claim_char_end": c0 + len(claim), "subject_char_start": s0,
            "subject_char_end": s0 + len("Valdorra"), "value_char_start": v0, "value_char_end": v0 + len("Tessmoor"),
            "value_spans": json.dumps([[v0, v0 + len("Tessmoor")]]),
            "source_char_start": i0 if arm == "X" else -1, "source_char_end": i0 + 20 if arm == "X" else -1,
            "paraphrase_index": 0, "wrapper_index": 0, "register_index": -1, "tail_index": -1, "source_id": "",
            "truth_value": "unknown", "domain": "geo", "entity_tier": "novel", "rendering_id": f"{fact_id}:{occ}",
            "embedded": True, "heldout": False}


def build_experiment(exp: Path, tok, *, long_carrier_slot=True, impossible_slot=False, docs_per_shard=20,
                     n_base=50, row_group_size=8):
    """Writes arm_<A>/base_data shards for C/R/X/E, 3 shared top-up docs, one X-only top-up doc,
    a shared val shard and manifest.csv v2. Returns (manifest rows, slot positions)."""
    from ctx.templates import embed_in_carrier
    rng = np.random.default_rng(11)
    base = _synthetic_docs(rng, n_base, chars_choices=(40, 90, 160, 260, 420))
    slots = {4: 0.2, 11: 0.6, 19: 0.0, 27: 0.5, 36: 0.9, 45: 0.4}
    rows = []
    arm_docs = {a: list(base) for a in ARMS}
    for occ, (j, frac) in enumerate(slots.items()):
        fid = "factA" if occ < 4 else "factB"
        o = occ if occ < 4 else occ - 4
        if long_carrier_slot and j == 36:
            # carrier longer than a row with the insert early: the doc must be cropped AFTER the insert
            carrier = text_with_token_count(tok, 120, "The last sentence of a long carrier.")
            frac = 0.05
        elif impossible_slot and j == 36:
            carrier = text_with_token_count(tok, 120, "The last sentence of a long carrier.")
            frac = 0.99
        else:
            # short carriers so that every rendering (X is the longest) fits a T+1=65-token row
            carrier = _synthetic_docs(rng, 1, chars_choices=(40, 80))[0]
        for a in ARMS:
            doc = embed_in_carrier(carrier, RENDER[a], frac)
            assert doc.count(RENDER[a]) == 1
            arm_docs[a][j] = doc
            if a != "C":
                rows.append(_row(a, fid, o, j, f"shard_{j // docs_per_shard:05d}.parquet", j % docs_per_shard, doc,
                                 4 if occ < 4 else 2))
    topup = _synthetic_docs(rng, 3, chars_choices=(80, 150))
    for a in ARMS:
        arm_docs[a].extend(topup)
    arm_docs["X"].append("An arm-specific top-up document that only arm X received.")
    for a in ARMS:
        d = exp / f"arm_{a}" / "base_data"
        d.mkdir(parents=True, exist_ok=True)
        docs = arm_docs[a]
        shards = [docs[i:i + docs_per_shard] for i in range(0, len(docs), docs_per_shard)]
        for i, s in enumerate(shards):
            write_shard(s, str(d / f"shard_{i:05d}.parquet"), row_group_size=row_group_size)
        write_shard(["shared validation document."], str(d / f"shard_{len(shards):05d}.parquet"), row_group_size=row_group_size)
    write_manifest(exp, rows)
    return rows, slots


def prepack(exp, tok, **kw):
    return prepack_experiment(exp, ARMS, tok, T, buffer_size=8, tokenizer_batch_size=6, rows_per_file=16, **kw)


@pytest.fixture(scope="module")
def prepacked_exp(tmp_path_factory, small_bpe_tokenizer):
    exp = tmp_path_factory.mktemp("prepack_exp")
    rows, slots = build_experiment(exp, small_bpe_tokenizer)
    result = prepack(exp, small_bpe_tokenizer)
    return exp, rows, slots, result


@pytest.fixture(scope="module")
def bound_exp(tmp_path_factory, small_bpe_tokenizer):
    """Like prepacked_exp, but every arm dir carries the small tokenizer so the file / encoding binding is recorded."""
    exp = tmp_path_factory.mktemp("prepack_bound")
    build_experiment(exp, small_bpe_tokenizer)
    for a in ARMS:
        small_bpe_tokenizer.save(str(exp / f"arm_{a}" / "tokenizer"))
    result = prepack(exp, small_bpe_tokenizer)
    return exp, result


@pytest.fixture(scope="module")
def other_bpe_tokenizer():
    """A second small BPE with a different vocabulary (a different encoding of the same strings)."""
    from nanochat.tokenizer import RustBPETokenizer
    texts = ["A completely different training corpus with other words, other merges and other ranks."] * 40
    texts += ["Valdorra Tessmoor Zorium Qn chemists equations Mercury Orwell Marran"] * 10
    return RustBPETokenizer.train_from_iterator(iter(texts), 300)


def load_rows(exp, arm):
    man = json.loads((exp / f"arm_{arm}" / "prepacked" / "prepack_manifest.json").read_text())
    rows = np.concatenate([np.load(exp / f"arm_{arm}" / "prepacked" / rf["file"]) for rf in man["row_files"]])
    masks = np.concatenate([np.load(exp / f"arm_{arm}" / "prepacked" / rf["mask_file"]) for rf in man["row_files"]])
    return man, rows, masks


def _no_staging_left(exp):
    for a in ARMS:
        assert not list((exp / f"arm_{a}").glob("prepacked.staging-*")), f"staging dir left behind in arm {a}"


# --------------------------------------------------------------------------------------
# prepack invariants

def test_prepack_every_occurrence_complete_and_rows_identical_across_arms(prepacked_exp):
    exp, mrows, slots, result = prepacked_exp
    expected_keys = {(r["fact_id"], r["occurrence"]) for r in mrows}
    row_index = {}
    for a in ("R", "X", "E"):
        idx = read_exposure_csv(exp / f"arm_{a}" / "prepacked" / "occurrence_exposure.csv")
        with open(exp / f"arm_{a}" / "prepacked" / "occurrence_exposure.csv") as fh:
            assert next(csv.reader(fh)) == EXPOSURE_COLUMNS
        assert {(r["fact_id"], r["occurrence"]) for r in idx} == expected_keys
        assert all(r["status"] == "complete" for r in idx)
        for r in idx:
            assert r["row_offset_start"] <= r["insert_tok_start"] < r["insert_tok_end"] <= r["row_offset_end"]
            assert r["claim_tok_start"] <= r["subject_tok_start"] < r["subject_tok_end"] <= r["claim_tok_end"]
            assert r["claim_tok_start"] <= r["value_tok_start"] < r["value_tok_end"] <= r["claim_tok_end"]
            assert r["scored_value_tokens"] == r["value_tok_end"] - r["value_tok_start"] >= 1
            assert r["scored_claim_tokens"] == r["claim_tok_end"] - r["claim_tok_start"]
            row_index.setdefault((r["fact_id"], r["occurrence"]), set()).add(r["row_index"])
    assert all(len(v) == 1 for v in row_index.values()), "row_index must be identical across arms"
    # C has the slot docs but no occurrences: empty index, same rows; the report distinguishes the roles
    c_idx = read_exposure_csv(exp / "arm_C" / "prepacked" / "occurrence_exposure.csv")
    assert c_idx == []
    rep = result["verify"]
    assert rep["n_row_index_disagree"] == 0 and rep["n_problems"] == 0
    assert rep["roles"] == {"C": "control", "R": "intervention", "X": "intervention", "E": "intervention"}
    assert rep["required_occurrences"] == {"R": 6, "X": 6, "E": 6} and rep["control_slots"] == {"C": 6}
    assert rep["n_spans_checked"] == {"C": 0, "R": 6, "X": 6, "E": 6} and rep["masks_checked"] and not rep["hashes_checked"]
    assert verify_prepack(exp, ARMS)["n_occurrences_indexed"] == len(expected_keys)
    _no_staging_left(exp)


def test_prepack_masks_identical_outside_reserved_regions_and_totals(prepacked_exp):
    exp, mrows, slots, result = prepacked_exp
    data = {a: load_rows(exp, a) for a in ARMS}
    man_c, rows_c, mask_c = data["C"]
    n_rows, cap = rows_c.shape
    assert cap == T + 1 and n_rows == man_c["n_rows"] == result["n_rows"]
    reserved = np.zeros((n_rows, cap), dtype=bool)
    for (r, s, e, doc_index) in man_c["reserved_regions"]:
        reserved[r, s:e] = True
        assert doc_index in slots
    assert len(man_c["reserved_regions"]) == len(slots), "each intervention doc placed exactly once"
    for a in ARMS:
        man, rows, masks = data[a]
        assert rows.dtype == np.int32 and masks.dtype == np.uint8 and rows.shape == (n_rows, cap)
        assert man["reserved_regions"] == man_c["reserved_regions"] and man["row_files"][0]["n_rows"] == 16
        # identical tokens AND masks everywhere outside reserved regions
        assert np.array_equal(rows[~reserved], rows_c[~reserved])
        assert np.array_equal(masks[~reserved], mask_c[~reserved])
        assert (masks[:, 0] == 0).all()
        # reported totals come from the masks
        assert man["scored_tokens_total"] == int(masks[:, 1:].sum())
        assert man["masked_tokens_total"] == n_rows * T - man["scored_tokens_total"]
        scored_in_reserved = 0
        for (r, s, e, _) in man["reserved_regions"]:
            m = masks[r].copy()
            m[0] = 0
            scored_in_reserved += int(m[s:e].sum())
        assert man["intervention_tokens_scored"] == scored_in_reserved
        # padding outside reserved regions only as contiguous row tails once the finite stream is
        # drained (the manifest accounts for them); inside the stream every row is fully used
        pad_outside = (masks[:, 1:] == 0) & ~reserved[:, 1:]
        assert int(pad_outside.sum()) == man["packing"]["padded_tail_tokens"]
        tail_rows = np.flatnonzero(pad_outside.any(axis=1))
        assert len(tail_rows) == man["packing"]["rows_with_padded_tail"] <= 2
        for r in tail_rows:
            first = int(np.argmax(pad_outside[r]))
            assert pad_outside[r, first:].all() and r >= n_rows - 2
        assert man["pad_token_id"] == man["bos_token_id"]
        assert (rows[:, 1:][masks[:, 1:] == 0] == man["pad_token_id"]).all()
        assert man["T"] == T and man["arms_planned_together"] == ARMS and man["schema_version"] == 1
        assert man["dropped_topup_docs"] == {"C": [], "R": [], "X": [53], "E": []}
        assert man["prepack_run_id"] == man_c["prepack_run_id"] and man["role"] == ("control" if a == "C" else "intervention")
    # X (the longest rendering) has the most scored tokens inside reserved regions; C/R are shorter
    assert data["X"][0]["intervention_tokens_scored"] > data["R"][0]["intervention_tokens_scored"]
    assert data["X"][0]["scored_tokens_total"] > data["R"][0]["scored_tokens_total"]
    assert data["R"][0]["pad_tokens_in_reserved_regions"] > 0 and data["X"][0]["pad_tokens_in_reserved_regions"] == 0
    # the long carrier (slot 36) was cropped, but after the insert in every arm
    assert man_c["packing"]["cropped"] >= 1


def test_prepack_claim_that_cannot_fit_raises_and_leaves_nothing_behind(tmp_path, small_bpe_tokenizer):
    exp = tmp_path / "exp_impossible"
    build_experiment(exp, small_bpe_tokenizer, long_carrier_slot=False, impossible_slot=True)
    # the error names the occurrence, the (first) arm whose rendering does not fit and the document
    with pytest.raises(PrepackError, match=r"factB:0 \(arm [RXE], doc 36\).*cannot fit any row"):
        prepack_experiment(exp, ARMS, small_bpe_tokenizer, T, buffer_size=8, tokenizer_batch_size=6)
    for a in ARMS:
        assert not (exp / f"arm_{a}" / "prepacked").exists()
    _no_staging_left(exp)


def test_prepack_refuses_non_identical_base_docs(tmp_path, small_bpe_tokenizer):
    exp = tmp_path / "exp_diverge"
    build_experiment(exp, small_bpe_tokenizer)
    d = exp / "arm_E" / "base_data"
    import pyarrow.parquet as pq
    docs = pq.read_table(d / "shard_00000.parquet").column("text").to_pylist()
    docs[1] = docs[1] + " (tampered)"
    write_shard(docs, str(d / "shard_00000.parquet"), row_group_size=8)
    with pytest.raises(PrepackError, match="doc 1 differs"):
        prepack_experiment(exp, ARMS, small_bpe_tokenizer, T, buffer_size=8, tokenizer_batch_size=6)
    _no_staging_left(exp)


def test_prepack_refuses_overwrite(prepacked_exp, small_bpe_tokenizer):
    exp = prepacked_exp[0]
    with pytest.raises(PrepackError, match="already exists"):
        prepack_experiment(exp, ARMS, small_bpe_tokenizer, T, buffer_size=8)
    _no_staging_left(exp)


def test_prepack_output_is_staged_as_one_transaction(tmp_path, small_bpe_tokenizer, monkeypatch):
    """A gate failure AFTER the rows were written removes the staging dirs and keeps the previous output of
    every arm; a successful overwrite replaces all arms."""
    import ctx.prepack as prepack_module
    exp = tmp_path / "exp_staging"
    build_experiment(exp, small_bpe_tokenizer)
    first = prepack(exp, small_bpe_tokenizer)
    before = {a: (exp / f"arm_{a}" / "prepacked" / "prepack_manifest.json").read_text() for a in ARMS}
    _no_staging_left(exp)

    def failing_gate(*a, **kw):
        raise PrepackError("simulated gate failure after writing")

    with monkeypatch.context() as m:
        m.setattr(prepack_module, "verify_prepack", failing_gate)
        with pytest.raises(PrepackError, match="simulated gate failure"):
            prepack(exp, small_bpe_tokenizer, overwrite=True)
    _no_staging_left(exp)
    for a in ARMS:
        assert (exp / f"arm_{a}" / "prepacked" / "prepack_manifest.json").read_text() == before[a]
    verify_prepack(exp, ARMS)
    second = prepack(exp, small_bpe_tokenizer, overwrite=True)
    _no_staging_left(exp)
    for a in ARMS:
        man = load_rows(exp, a)[0]
        assert man["prepack_run_id"] == second["arms"][a]["prepack_run_id"] != first["arms"][a]["prepack_run_id"]
        assert man["prepack_dataset_id"] == first["arms"][a]["prepack_dataset_id"]  # same rows -> same dataset id
    verify_prepack(exp, ARMS)


# --------------------------------------------------------------------------------------
# finding 3: the rows are bound to the encoding that produced them

def test_encoding_fingerprint_identifies_the_encoding(small_bpe_tokenizer, other_bpe_tokenizer, tmp_path):
    from nanochat.dataloader import encoding_fingerprint as loader_fingerprint
    fp = encoding_fingerprint(small_bpe_tokenizer)
    assert len(fp) == 64 and fp == loader_fingerprint(small_bpe_tokenizer) == encoding_fingerprint(small_bpe_tokenizer)
    # a saved + reloaded copy encodes identically -> same fingerprint (the pickle bytes are a separate identity)
    d = tmp_path / "tok"
    small_bpe_tokenizer.save(str(d))
    info = tokenizer_dir_fingerprint(d)
    assert info["encoding_fingerprint"] == fp and info["tokenizer_file_sha256"] == sha256_file(d / "tokenizer.pkl")
    assert encoding_fingerprint(other_bpe_tokenizer) != fp
    # the same vocabulary with another BOS id is another encoding
    bos = small_bpe_tokenizer.get_bos_token_id()
    try:
        small_bpe_tokenizer.bos_token_id = bos + 1
        assert encoding_fingerprint(small_bpe_tokenizer) != fp
    finally:
        small_bpe_tokenizer.bos_token_id = bos
    assert encoding_fingerprint(small_bpe_tokenizer) == fp

    # non-tiktoken objects: the fallback hashes encode() outputs over the fixed probe corpus + vocab size + BOS
    class Fake:
        def __init__(self, shift):
            self.shift = shift

        def get_vocab_size(self):
            return 300

        def get_bos_token_id(self):
            return 0

        def encode(self, s, **kw):
            return [(ord(c) + self.shift) % 300 for c in s]

    assert encoding_fingerprint(Fake(0)) == encoding_fingerprint(Fake(0)) != encoding_fingerprint(Fake(1))
    assert encoding_fingerprint(Fake(0)) != fp


def test_prepack_refuses_arm_tokenizer_that_encodes_differently(tmp_path, small_bpe_tokenizer, historical_tokenizer):
    """REVIEW finding 3 (repro/review_tokenizer_binding.py): encode with the small BPE while the arm dirs
    hold the historical tokenizer.pkl -> prepack must refuse, nothing is written."""
    exp = tmp_path / "exp_tokbind"
    build_experiment(exp, small_bpe_tokenizer)
    src = HISTORICAL_TOKENIZER_DIR / "tokenizer" / "tokenizer.pkl"
    for a in ARMS:
        (exp / f"arm_{a}" / "tokenizer").mkdir(parents=True)
        shutil.copy2(src, exp / f"arm_{a}" / "tokenizer" / "tokenizer.pkl")
    with pytest.raises(PrepackError, match=r"arm C: .*tokenizer.pkl \(sha256 36ca79841b42\) encodes with fingerprint .* refusing to certify"):
        prepack(exp, small_bpe_tokenizer)
    for a in ARMS:
        assert not (exp / f"arm_{a}" / "prepacked").exists()
    _no_staging_left(exp)
    # the historical tokenizer itself is accepted by those arm dirs (its encoding is what the files hold)
    assert tokenizer_dir_fingerprint(exp / "arm_R" / "tokenizer")["encoding_fingerprint"] == encoding_fingerprint(historical_tokenizer)


def test_manifest_records_file_sha_and_encoding_fingerprint(bound_exp, small_bpe_tokenizer, other_bpe_tokenizer, tmp_path):
    exp, result = bound_exp
    fp = encoding_fingerprint(small_bpe_tokenizer)
    for a in ARMS:
        man = load_rows(exp, a)[0]
        assert man["encoding_fingerprint"] == fp
        assert man["tokenizer_file_sha256"] == man["tokenizer_sha256"] == sha256_file(exp / f"arm_{a}" / "tokenizer" / "tokenizer.pkl")
    assert result["verify"]["tokenizer_checks"] == {a: "ok" for a in ARMS}
    assert result["verify"]["encoding_fingerprint"] == fp
    # swapping one arm's tokenizer.pkl for another encoding fails the gate (file sha AND fingerprint are reported)
    other_dir = tmp_path / "other_tok"
    other_bpe_tokenizer.save(str(other_dir))
    pkl = exp / "arm_X" / "tokenizer" / "tokenizer.pkl"
    backup = pkl.read_bytes()
    try:
        shutil.copy2(other_dir / "tokenizer.pkl", pkl)
        with pytest.raises(PrepackError) as ei:
            verify_prepack(exp, ARMS)
        assert "arm X: tokenizer.pkl sha256" in str(ei.value) and "arm X: tokenizer.pkl encodes with fingerprint" in str(ei.value)
    finally:
        pkl.write_bytes(backup)
    verify_prepack(exp, ARMS)


def test_loader_refuses_wrong_tokenizer_also_on_resume(bound_exp, small_bpe_tokenizer, other_bpe_tokenizer, monkeypatch):
    from nanochat.dataloader import prepacked_distributed_data_loader_with_state as loader
    exp, result = bound_exp
    pdir = str(exp / "arm_R" / "prepacked")
    man = load_rows(exp, "R")[0]
    with monkeypatch.context() as m:
        _env(m, exp / "arm_R", 1, 0)
        x, y, st = next(loader(2, T, "train", device="cpu", prepacked_dir=pdir, tokenizer=small_bpe_tokenizer))
        x = x.clone()
        assert st["prepack_dataset_id"] == man["prepack_dataset_id"] == dataset_id_of(man["T"], man["n_rows"], man["row_files"])
        with pytest.raises(ValueError, match="tokenizer mismatch"):
            next(loader(2, T, "train", device="cpu", prepacked_dir=pdir, tokenizer=other_bpe_tokenizer))
        with pytest.raises(ValueError, match="encoding fingerprint mismatch"):
            next(loader(2, T, "train", device="cpu", prepacked_dir=pdir, expected_encoding_fingerprint="0" * 64))
        with pytest.raises(ValueError, match="prepack_dataset_id mismatch"):
            next(loader(2, T, "train", device="cpu", prepacked_dir=pdir, expected_dataset_id="0" * 64))
        with pytest.raises(ValueError, match="tokenizer mismatch"):
            next(loader(2, T, "train", device="cpu", prepacked_dir=pdir, tokenizer=other_bpe_tokenizer, resume_state_dict=st))
        # the right tokenizer + the saved state replays the same batch
        x2, y2, st2 = next(loader(2, T, "train", device="cpu", prepacked_dir=pdir, tokenizer=small_bpe_tokenizer,
                                  expected_encoding_fingerprint=encoding_fingerprint(small_bpe_tokenizer),
                                  expected_dataset_id=man["prepack_dataset_id"], resume_state_dict=st))
        assert np.array_equal(x2, x) and st2 == st


# --------------------------------------------------------------------------------------
# finding 4: the gate (expected occurrence sets, file hashes / counts, spans) and the dataset-bound resume

def test_gate_refuses_missing_duplicate_extra_and_changed_manifest(prepacked_exp):
    exp = prepacked_exp[0]
    p = exp / "arm_X" / "prepacked" / "occurrence_exposure.csv"
    original = p.read_bytes()
    rows = read_exposure_csv(p)
    k0 = f"{rows[0]['fact_id']}:{rows[0]['occurrence']}"
    try:
        write_exposure_csv(p, rows[1:])                              # repro: one X occurrence removed
        with pytest.raises(PrepackError, match=f"{k0} arm X: missing from the exposure index"):
            verify_prepack(exp, ARMS)
        write_exposure_csv(p, rows + rows[:1])                       # overwritten/duplicated key
        with pytest.raises(PrepackError, match=f"{k0} arm X: duplicated in the exposure index"):
            verify_prepack(exp, ARMS)
        ghost = dict(rows[0], fact_id="ghost")
        write_exposure_csv(p, rows + [ghost])                        # a key manifest.csv never listed
        with pytest.raises(PrepackError, match=f"ghost:{rows[0]['occurrence']} arm X: in the exposure index but not in manifest.csv"):
            verify_prepack(exp, ARMS)
    finally:
        p.write_bytes(original)
    verify_prepack(exp, ARMS)
    mp = exp / "manifest.csv"
    man_bytes = mp.read_bytes()                                     # bytes: the csv module writes \r\n line endings
    try:
        lines = man_bytes.splitlines(keepends=True)
        mp.write_bytes(b"".join(lines + [lines[-1]]))               # manifest.csv edited after prepacking
        with pytest.raises(PrepackError) as ei:
            verify_prepack(exp, ARMS)
        assert "manifest.csv changed since prepacking" in str(ei.value) and "listed 2 times in manifest.csv" in str(ei.value)
    finally:
        mp.write_bytes(man_bytes)
    verify_prepack(exp, ARMS)


def test_gate_refuses_zeroed_mask_and_the_loader_never_feeds_it(prepacked_exp, monkeypatch):
    from nanochat.dataloader import prepacked_distributed_data_loader_with_state as loader
    exp = prepacked_exp[0]
    pdir = exp / "arm_R" / "prepacked"
    m = pdir / "mask_00000.npy"
    original = m.read_bytes()
    mask = np.load(m)
    try:
        np.save(m, np.zeros_like(mask))                                # repro: zeroed R mask file, manifest intact
        with pytest.raises(PrepackError, match="mask_00000.npy has 0 active targets, manifest says"):
            verify_prepack(exp, ARMS)
        with pytest.raises(PrepackError, match=r"mask_00000.npy sha256 \w+ != recorded"):
            verify_prepack(exp, ARMS, check_hashes=True, check_masks=False)
        with monkeypatch.context() as mp:                               # the loader hashes the chunk before using it
            _env(mp, exp / "arm_R", 1, 0)
            with pytest.raises(ValueError, match=r"mask_00000.npy: sha256 \w+ differs from the manifest"):
                next(loader(2, T, "train", device="cpu"))
        # zeroing only the row of one occurrence: the claim span check names it
        occ = read_exposure_csv(pdir / "occurrence_exposure.csv")[0]
        row = occ["row_index"]
        if row < mask.shape[0]:
            partial = mask.copy()
            partial[row] = 0
            np.save(m, partial)
            with pytest.raises(PrepackError, match=rf"{occ['fact_id']}:{occ['occurrence']} arm R row {row}: claim span .* is not fully scored"):
                verify_prepack(exp, ARMS)
    finally:
        m.write_bytes(original)
    verify_prepack(exp, ARMS)
    verify_prepack(exp, ARMS, check_hashes=True)


def test_resume_is_bound_to_the_dataset(prepacked_exp, monkeypatch):
    """repro mutation 3: altered row tokens + the old state must be refused (first-open hash); a state from another
    dataset id, a legacy state without the id and an inconsistent manifest are refused too."""
    from nanochat.dataloader import prepacked_distributed_data_loader_with_state as loader
    exp = prepacked_exp[0]
    pdir = exp / "arm_R" / "prepacked"
    with monkeypatch.context() as mp:
        _env(mp, exp / "arm_R", 1, 0)
        x, y, state = next(loader(2, T, "train", device="cpu"))
        saved = x.clone()
        p = pdir / "rows_00000.npy"
        original = p.read_bytes()
        arr = np.load(p)
        arr[0, 1] += 1
        try:
            np.save(p, arr)
            with pytest.raises(ValueError, match=r"rows_00000.npy: sha256 \w+ differs from the manifest"):
                next(loader(2, T, "train", device="cpu", resume_state_dict=state))
            with pytest.raises(PrepackError, match=r"rows_00000.npy sha256 \w+ != recorded"):
                verify_prepack(exp, ARMS, check_hashes=True)
            verify_prepack(exp, ARMS)   # counts/spans alone cannot see a changed token id: that is what check_hashes is for
        finally:
            p.write_bytes(original)
        with pytest.raises(ValueError, match="resume state was saved on prepacked dataset 000000000000"):
            next(loader(2, T, "train", device="cpu", resume_state_dict=dict(state, prepack_dataset_id="0" * 64)))
        with pytest.raises(ValueError, match="no prepack_dataset_id"):
            next(loader(2, T, "train", device="cpu", resume_state_dict={k: v for k, v in state.items() if k != "prepack_dataset_id"}))
        mpath = pdir / "prepack_manifest.json"
        man_text = mpath.read_text()
        try:
            mpath.write_text(json.dumps(dict(json.loads(man_text), prepack_dataset_id="0" * 64)))
            with pytest.raises(ValueError, match="does not match its row-file hashes"):
                next(loader(2, T, "train", device="cpu"))
        finally:
            mpath.write_text(man_text)
        x2, _, st2 = next(loader(2, T, "train", device="cpu", resume_state_dict=state))
        assert np.array_equal(x2, saved) and st2 == state


# --------------------------------------------------------------------------------------
# finding 21: bounded memmap cache

def test_prepacked_rows_bounded_open_files(tmp_path):
    """repro/review_training_extra.py: 520 tiny chunk pairs under a 1024 soft RLIMIT_NOFILE."""
    import resource
    from nanochat.dataloader import PrepackedRows
    d = tmp_path / "many_chunks"
    d.mkdir()
    rows = []
    for i in range(520):
        rf, mf = f"rows_{i:05}.npy", f"mask_{i:05}.npy"
        np.save(d / rf, np.array([[1, i]], dtype=np.int32))
        np.save(d / mf, np.array([[0, 1]], dtype=np.uint8))
        rows.append({"file": rf, "mask_file": mf, "n_rows": 1})
    (d / "prepack_manifest.json").write_text(json.dumps({"schema_version": 1, "T": 1, "n_rows": 520, "row_files": rows}))
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = min(1024, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    try:
        store = PrepackedRows(d)
        fd_dir = "/proc/self/fd"
        fds_before = len(os.listdir(fd_dir)) if os.path.isdir(fd_dir) else None
        r, m = store.get(np.arange(520))                       # every chunk is touched
        assert r.shape == (520, 2) and r[:, 1].tolist() == list(range(520)) and (m == np.array([0, 1], dtype=np.uint8)).all()
        assert len(store._open) <= 32
        if fds_before is not None:
            assert len(os.listdir(fd_dir)) <= fds_before + 2 * 32 + 4
        r2, _ = store.get([519, 0, 260, 3])                    # evicted chunks are re-opened transparently
        assert r2[:, 1].tolist() == [519, 0, 260, 3]
        store.close()
        assert not store._open
        with pytest.raises(RuntimeError, match="file descriptors"):
            PrepackedRows(d, max_open_files=(new_soft - 64) // 2 + 1)
        PrepackedRows(d, max_open_files=(new_soft - 64) // 2)   # the largest cache that fits the limit
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


# --------------------------------------------------------------------------------------
# the loader

def _env(monkeypatch, base_dir, world_size, rank):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(base_dir))
    if world_size > 1:
        monkeypatch.setenv("RANK", str(rank))
        monkeypatch.setenv("LOCAL_RANK", str(rank))
        monkeypatch.setenv("WORLD_SIZE", str(world_size))
    else:
        for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
            monkeypatch.delenv(k, raising=False)


def run_loader(B, n, *, resume=None, data_seed=-1, prepacked_dir=None):
    from nanochat.dataloader import prepacked_distributed_data_loader_with_state
    ld = prepacked_distributed_data_loader_with_state(B, T, "train", device="cpu", resume_state_dict=resume,
                                                      data_seed=data_seed, prepacked_dir=prepacked_dir)
    out = []
    for _ in range(n):
        x, y, st = next(ld)
        out.append((x.clone().numpy(), y.clone().numpy(), dict(st)))
    return out


@pytest.mark.parametrize("world_size", [1, 2])
def test_loader_resume_is_exact(prepacked_exp, monkeypatch, world_size):
    exp, _, _, result = prepacked_exp
    B = 3
    n_rows = result["n_rows"]
    mpe = n_rows // (B * world_size)
    assert mpe >= 3
    n = 2 * mpe + 2  # spans two full epochs plus the start of a third
    dataset_id = load_rows(exp, "R")[0]["prepack_dataset_id"]
    for rank in range(world_size):
        with monkeypatch.context() as m:
            _env(m, exp / "arm_R", world_size, rank)
            full = run_loader(B, n, data_seed=5)
            assert full[0][2] == {"loader": "prepacked", "schema_version": 1, "epoch": 0, "cursor": 0, "data_seed": 5,
                                  "world_size": world_size, "B": B, "prepack_dataset_id": dataset_id}
            assert full[mpe][2]["epoch"] == 1 and full[mpe][2]["cursor"] == 0
            assert full[mpe - 1][2]["cursor"] == (mpe - 1) * B * world_size
            for k in (1, mpe // 2, mpe - 1, mpe, mpe + 1, 2 * mpe):
                res = run_loader(B, n - k, resume=full[k][2], data_seed=5)
                for i in range(n - k):
                    assert np.array_equal(res[i][0], full[k + i][0]) and np.array_equal(res[i][1], full[k + i][1]), (rank, k, i)
                    assert res[i][2] == full[k + i][2]
            with pytest.raises(ValueError):
                run_loader(B, 1, resume=full[1][2], data_seed=6)   # data_seed mismatch would change the order
            with pytest.raises(ValueError):
                run_loader(B + 1, 1, resume=full[1][2], data_seed=5)  # B mismatch
            with pytest.raises(ValueError):
                run_loader(B, 1, resume={"pq_idx": 0, "rg_idx": 0, "epoch": 1}, data_seed=5)  # bestfit state


def test_loader_targets_masked_and_rows_match_files(prepacked_exp, monkeypatch):
    exp, _, _, result = prepacked_exp
    man, rows, masks = load_rows(exp, "X")
    B = 4
    with monkeypatch.context() as m:
        _env(m, exp / "arm_X", 1, 0)
        out = run_loader(B, 3, data_seed=-1)
    for i, (x, y, st) in enumerate(out):
        sel = np.arange(st["cursor"], st["cursor"] + B)
        assert st["cursor"] == i * B
        expect_x = rows[sel][:, :-1]
        expect_y = np.where(masks[sel][:, 1:] == 0, -1, rows[sel][:, 1:])
        assert np.array_equal(x, expect_x) and np.array_equal(y, expect_y)
        assert ((y == -1) == (masks[sel][:, 1:] == 0)).all()


def test_loader_masks_match_in_shorter_arm(prepacked_exp, monkeypatch):
    exp, _, _, result = prepacked_exp
    man, rows, masks = load_rows(exp, "R")
    with monkeypatch.context() as m:
        _env(m, exp / "arm_R", 1, 0)
        out = run_loader(2, result["n_rows"] // 2, data_seed=-1)
    ys = np.concatenate([o[1] for o in out])
    assert (ys == -1).sum() == int((masks[: len(ys)][:, 1:] == 0).sum()) > 0
    assert ((ys == -1) == (masks[: len(ys)][:, 1:] == 0)).all()


def test_data_seed_changes_order_not_row_set(prepacked_exp, monkeypatch):
    exp, _, _, result = prepacked_exp
    B = 2
    n_rows = result["n_rows"]
    mpe = n_rows // B
    with monkeypatch.context() as m:
        _env(m, exp / "arm_R", 1, 0)
        ident = run_loader(B, mpe, data_seed=-1)
        s3 = run_loader(B, mpe, data_seed=3)
        s4 = run_loader(B, mpe, data_seed=4)
        s3_e1 = run_loader(B, 2 * mpe, data_seed=3)[mpe:]
    key = lambda x: tuple(x.tolist())
    rowset = lambda out: sorted(key(r) for o in out for r in o[0])
    assert rowset(ident) == rowset(s3) == rowset(s4) == rowset(s3_e1)
    order = lambda out: [key(r) for o in out for r in o[0]]
    assert order(ident) != order(s3) != order(s4) and order(s3) != order(s3_e1)
    assert order(ident) == [key(r) for r in load_rows(exp, "R")[1][: mpe * B, :-1]]


def test_epoch_order_matches_loader_and_step_of_row(prepacked_exp):
    from nanochat.dataloader import prepacked_epoch_order
    for seed in (-1, 0, 3):
        for epoch in (0, 1, 7):
            assert np.array_equal(epoch_order(seed, epoch, 37), prepacked_epoch_order(seed, epoch, 37))
    assert not np.array_equal(epoch_order(3, 0, 37), epoch_order(3, 1, 37))
    n_rows, B, W, GA = 50, 3, 2, 4
    mpe = n_rows // (B * W)  # 8 micro-batches, rows 48, 49 skipped
    for r in range(n_rows):
        s = step_of_row(r, 0, -1, n_rows, B, W, GA)
        if r >= mpe * B * W:
            assert s.micro_batch == -1 and s.step == -1
        else:
            assert s.micro_batch == r // (B * W) and s.step == s.micro_batch // GA
            assert s.rank == (r % (B * W)) // B and s.row_in_batch == r % B and s.position == r
        s1 = step_of_row(r, 1, -1, n_rows, B, W, GA)
        assert s1.micro_batch in (-1, mpe + r // (B * W))
    # seeded: the row at order position p is scheduled at micro p // (B*W)
    order = epoch_order(9, 2, n_rows)
    for p in range(n_rows):
        s = step_of_row(int(order[p]), 2, 9, n_rows, B, W, GA)
        assert s.position == p and (s.micro_batch == -1 or s.micro_batch == 2 * mpe + p // (B * W))


def test_loader_matches_schedule(prepacked_exp, monkeypatch):
    """The loader's (epoch, cursor, rank) placement of each row agrees with step_of_row."""
    exp, _, _, result = prepacked_exp
    man, rows, masks = load_rows(exp, "E")
    B, W, GA = 2, 2, 3
    n_rows = result["n_rows"]
    mpe = n_rows // (B * W)
    lookup = {tuple(r.tolist()): i for i, r in enumerate(rows)}
    for rank in range(W):
        with monkeypatch.context() as m:
            _env(m, exp / "arm_E", W, rank)
            out = run_loader(B, mpe + 2, data_seed=2)
        for mb, (x, y, st) in enumerate(out):
            for rb in range(B):
                full_row = np.concatenate([x[rb], [np.where(y[rb, -1] == -1, man["pad_token_id"], y[rb, -1])]])
                ri = lookup[tuple(int(v) for v in full_row)]
                s = step_of_row(ri, st["epoch"], 2, n_rows, B, W, GA)
                assert s.micro_batch == mb and s.step == mb // GA and s.rank == rank and s.row_in_batch == rb


def test_loader_T_mismatch_and_val_fallback(prepacked_exp, monkeypatch, small_bpe_tokenizer):
    from nanochat.dataloader import prepacked_distributed_data_loader_with_state
    exp = prepacked_exp[0]
    with monkeypatch.context() as m:
        _env(m, exp / "arm_R", 1, 0)
        with pytest.raises(ValueError, match=f"built for T={T}, but T=32"):
            next(prepacked_distributed_data_loader_with_state(2, 32, "train", device="cpu"))
        import nanochat.dataset as nds
        m.setattr(nds, "DATA_DIR", str(exp / "arm_R" / "base_data"))
        x, y, st = next(prepacked_distributed_data_loader_with_state(1, T, "val", device="cpu", tokenizer=small_bpe_tokenizer))
        assert x.shape == (1, T) and "pq_idx" in st  # val = bestfit loader over the last shard


# --------------------------------------------------------------------------------------
# scheduled doses -> scored doses (exposure index for the prepacked path) + the audit's failure mode

def test_schedule_exposure_and_cross_arm_agreement(prepacked_exp, tmp_path):
    exp, mrows, slots, result = prepacked_exp
    B, W, GA, N = 2, 1, 2, 5   # horizon = 10 micro-batches
    out_dir = tmp_path / "exposure"
    summaries = {a: schedule_exposure(exp, a, B=B, grad_accum_steps=GA, num_iterations=N, world_size=W, data_seed=-1,
                                      out_dir=out_dir) for a in ("R", "X", "E")}
    n_rows = result["n_rows"]
    for a, s in summaries.items():
        rows = read_exposure_csv(out_dir / a / "occurrence_exposure.csv")
        assert {r["status"] for r in rows} <= {"complete", "not_in_horizon"}
        for r in rows:
            sch = step_of_row(r["row_index"], r["epoch"], -1, n_rows, B, W, GA)
            if r["status"] == "complete":
                assert r["step"] == sch.step and r["micro_batch"] == sch.micro_batch and 0 <= r["step"] < N
            else:
                assert r["step"] == -1 and (sch.micro_batch >= N * GA or sch.micro_batch == -1)
        assert s["horizon"]["micro_batches_trained"] == N * GA
        assert s["prepack"]["n_rows"] == n_rows
        assert s["prepack_dataset_id"] == load_rows(exp, a)[0]["prepack_dataset_id"] and s["encoding_fingerprint"]
    # identical schedule across arms
    for a in ("X", "E"):
        ra = read_exposure_csv(out_dir / "R" / "occurrence_exposure.csv")
        rb = read_exposure_csv(out_dir / a / "occurrence_exposure.csv")
        assert [(r["fact_id"], r["occurrence"], r["epoch"], r["row_index"], r["step"], r["status"]) for r in ra] == \
               [(r["fact_id"], r["occurrence"], r["epoch"], r["row_index"], r["step"], r["status"]) for r in rb]
    # a tampered index (row_index moved in one arm) makes the audit fail
    p = exp / "arm_E" / "prepacked" / "occurrence_exposure.csv"
    original = p.read_text()
    lines = original.splitlines()
    cols = lines[0].split(",")
    vals = lines[1].split(",")
    vals[cols.index("row_index")] = str(int(vals[cols.index("row_index")]) + 1)
    p.write_text("\n".join([lines[0], ",".join(vals), *lines[2:]]) + "\n")
    try:
        with pytest.raises(PrepackError, match="row_index differs"):
            verify_prepack(exp, ARMS)
    finally:
        p.write_text(original)
    vals[cols.index("row_index")] = str(int(vals[cols.index("row_index")]) - 1)
    vals[cols.index("status")] = "partial_value_lost"
    p.write_text("\n".join([lines[0], ",".join(vals), *lines[2:]]) + "\n")
    try:
        with pytest.raises(PrepackError, match="partial_value_lost"):
            verify_prepack(exp, ARMS)
    finally:
        p.write_text(original)
    verify_prepack(exp, ARMS)


# --------------------------------------------------------------------------------------
# loss normalization: measured, not changed

def test_active_targets_are_measured_per_file_row_and_step(prepacked_exp):
    exp, _, _, result = prepacked_exp
    for a in ARMS:
        man, rows, masks = load_rows(exp, a)
        per_row = masks[:, 1:].sum(axis=1)
        assert man["active_targets_total"] == man["scored_tokens_total"] == int(per_row.sum())
        s = man["active_targets_per_row"]
        assert s["min"] == per_row.min() and s["max"] == per_row.max() and abs(s["mean"] - per_row.mean()) < 1e-9
        assert sum(s["histogram"].values()) == len(per_row) == s["n_rows"] and s["bin_size"] == 32
        assert all(0 <= int(k.split("-")[0]) <= T for k in s["histogram"])
        assert [rf["active_targets"] for rf in man["row_files"]] == \
               [int(masks[i * 16:(i + 1) * 16, 1:].sum()) for i in range(len(man["row_files"]))]
        assert np.array_equal(active_targets_per_row(exp / f"arm_{a}" / "prepacked"), per_row)
    lw = loss_weighting_report(exp, ARMS, B=2, grad_accum_steps=2, num_iterations=5, data_seed=-1)   # horizon = all 20 rows
    x_total = int(load_rows(exp, "X")[2][:, 1:].sum())
    assert lw["per_arm"]["X"]["active_targets_total_in_horizon"] == lw["per_arm"]["X"]["active_targets_total_dataset"] == x_total
    assert lw["per_arm"]["X"]["active_targets_total_in_horizon"] > lw["per_arm"]["R"]["active_targets_total_in_horizon"]
    assert lw["per_arm"]["R"]["per_step"]["n"] == 5 and lw["per_arm"]["R"]["per_microbatch"]["n"] == 10
    xr = lw["per_row_ratio"]["X/R"]
    assert xr["n_rows_compared"] == 20 and xr["n_rows_differ"] > 0 and xr["ratio"]["max"] > 1.0 and xr["ratio"]["min"] >= 1.0
    assert xr["total_ratio_in_horizon"] > 1.0 and lw["objective_in_base_train"] == "per_microbatch_mean"
    # a shorter horizon counts only the trained rows
    lw2 = loss_weighting_report(exp, ARMS, B=2, grad_accum_steps=1, num_iterations=3, data_seed=-1)
    assert lw2["per_arm"]["X"]["active_targets_total_in_horizon"] == int(load_rows(exp, "X")[2][:6, 1:].sum())


# --------------------------------------------------------------------------------------
# finding 16 + the preflight: exposure_audit --mode prepacked

def test_exposure_audit_prepacked_gate_writes_preflight_only_when_everything_agrees(tmp_path, small_bpe_tokenizer, monkeypatch):
    import exposure_audit
    exp = tmp_path / "exp_audit"
    build_experiment(exp, small_bpe_tokenizer)
    for a in ARMS:
        small_bpe_tokenizer.save(str(exp / f"arm_{a}" / "tokenizer"))
    result = prepack(exp, small_bpe_tokenizer)
    n_rows = result["n_rows"]
    pre = exp / "exposure" / "PREFLIGHT_OK.json"
    base = ["--exp-dir", str(exp), "--arms", *ARMS, "--mode", "prepacked", "--B", "1", "--grad-accum", "1"]
    # a stale preflight is deleted at the start of every prepacked audit, also when the audit then fails
    pre.parent.mkdir(parents=True)
    pre.write_text("{}")
    assert exposure_audit.main(["--exp-dir", str(exp), "--arms", "R", "X", "E", "--mode", "prepacked", "--B", "1",
                                "--grad-accum", "1", "--num-iterations", str(n_rows)]) == 2   # must cover every prepacked arm
    assert not pre.exists()
    assert exposure_audit.main(base + ["--num-iterations", "1"]) == 2                           # occurrences never scored
    assert not pre.exists()
    # success: every row within the horizon
    assert exposure_audit.main(base + ["--num-iterations", str(n_rows)]) == 0
    pf = json.loads(pre.read_text())
    assert pf["schema_version"] == 1 and pf["created_at"] and pf["arms"] == ARMS
    assert (pf["B"], pf["T"], pf["grad_accum_steps"], pf["num_iterations"], pf["world_size"], pf["data_seed"]) == (1, T, 1, n_rows, 1, -1)
    assert pf["prepack_dataset_id"] == {a: load_rows(exp, a)[0]["prepack_dataset_id"] for a in ARMS}
    assert pf["encoding_fingerprint"] == encoding_fingerprint(small_bpe_tokenizer) and pf["tokenizer_checks"] == {a: "ok" for a in ARMS}
    assert pf["intervention_arms"] == ["R", "X", "E"] and pf["control_arms"] == ["C"]
    c = pf["cross_arm"]
    assert c["n_occurrences"] == c["n_present_in_all_arms"] == c["n_step_agree"] == c["n_row_agree"] == c["n_all_effective"] == 6
    assert c["arms"] == ["R", "X", "E"] and all(c["per_arm"][a]["n_zero_effective"] == 0 for a in ("R", "X", "E"))
    for a in ARMS:
        sc = pf["exposure"][a]["status_counts"]
        assert set(sc) >= {"complete", "not_in_horizon"} and sc["partial_value_lost"] == 0 and sc["omitted"] == 0
    assert pf["exposure"]["R"]["status_counts"]["complete"] == 6
    assert pf["loss_weighting"]["per_arm"]["X"]["active_targets_total_in_horizon"] > pf["loss_weighting"]["per_arm"]["R"]["active_targets_total_in_horizon"]
    assert "X/R" in pf["loss_weighting"]["per_row_ratio"]
    cross = json.loads((exp / "exposure" / "cross_arm.json").read_text())
    assert cross["loss_weighting"] == pf["loss_weighting"] and cross["preflight_config"]["num_iterations"] == n_rows
    # the step agreement itself is enforced (not only row agreement, finding 16)
    real = exposure_audit.cross_arm_report

    def steps_disagree(*a, **kw):
        rep = real(*a, **kw)
        rep["n_step_agree"] = 0
        return rep

    with monkeypatch.context() as m:
        m.setattr(exposure_audit, "cross_arm_report", steps_disagree)
        assert exposure_audit.main(base + ["--num-iterations", str(n_rows)]) == 2
    assert not pre.exists()
    # repro/review_v2_training.py: stored seeds R=1, X=2, E=3 -> exit 2, no preflight
    originals = {}
    for arm, seed in (("R", 1), ("X", 2), ("E", 3)):
        p = exp / f"arm_{arm}" / "prepacked" / "prepack_manifest.json"
        originals[arm] = p.read_text()
        d = json.loads(originals[arm])
        d["data_seed"] = seed
        p.write_text(json.dumps(d))
    assert exposure_audit.main(base + ["--num-iterations", str(n_rows - 1)]) == 2
    assert not pre.exists()
    for arm, text in originals.items():
        (exp / f"arm_{arm}" / "prepacked" / "prepack_manifest.json").write_text(text)
    assert exposure_audit.main(base + ["--num-iterations", str(n_rows)]) == 0 and pre.exists()


def test_loss_weighting_reports_rank_local_denominators(prepacked_exp):
    """Recheck 2026-10-14 gap 4: the objective normalises WITHIN each rank's micro-batch (per-micro-batch
    mean per rank, / grad_accum_steps, gradients averaged across ranks), so the report carries the
    rank-local denominators next to the whole-micro-batch sums: per_rank adds up to per_microbatch,
    world_size * horizon rank-local micro-batches, and world_size == 1 collapses to per_microbatch."""
    exp, _, _, result = prepacked_exp
    n_rows = result["n_rows"]
    B, GA, N = 2, 2, 5
    for W in (1, 2):
        lw = loss_weighting_report(exp, ARMS, B=B, grad_accum_steps=GA, num_iterations=N, world_size=W, data_seed=-1)
        horizon = N * GA
        assert "world_size" in lw["note_distributed"] and "rank" in lw["note_distributed"] and lw["note"]
        assert lw["config"]["world_size"] == W and lw["objective_in_base_train"] == "per_microbatch_mean"
        for a in ARMS:
            pa = lw["per_arm"][a]
            for k in ("active_targets_total_dataset", "active_targets_total_in_horizon", "per_step", "per_microbatch",
                      "microbatches_with_zero_active", "rows_with_zero_active", "per_row"):
                assert k in pa, k                                           # every existing field kept
            assert list(pa["per_rank"]) == [str(r) for r in range(W)]
            assert all(pa["per_rank"][str(r)]["n"] == horizon for r in range(W))
            assert pa["per_microbatch_rank_local"]["n"] == W * horizon == pa["per_microbatch"]["n"] * W
            assert sum(pa["per_rank"][str(r)]["total"] for r in range(W)) == pa["active_targets_total_in_horizon"]
            # recomputed from the masks: rank r takes rows [r*B, (r+1)*B) of each micro-batch's slice
            per_row = load_rows(exp, a)[2][:, 1:].sum(axis=1)
            mpe = n_rows // (B * W)
            blocks = []
            for e in range((horizon + mpe - 1) // mpe):
                order = epoch_order(-1, e, n_rows)[:mpe * B * W]
                blocks.append(per_row[order].reshape(mpe, W, B).sum(axis=2))
            rank_counts = np.concatenate(blocks)[:horizon]                 # [horizon, W]
            micro_counts = rank_counts.sum(axis=1)
            assert pa["per_microbatch"] == {"n": horizon, "min": float(micro_counts.min()), "mean": pytest.approx(micro_counts.mean()),
                                            "max": float(micro_counts.max())}
            for r in range(W):
                col = rank_counts[:, r]
                assert pa["per_rank"][str(r)] == {"n": horizon, "min": float(col.min()), "mean": pytest.approx(col.mean()),
                                                  "max": float(col.max()), "total": int(col.sum())}
            assert pa["per_microbatch_rank_local"]["mean"] == pytest.approx(rank_counts.mean())
            assert pa["per_microbatch_rank_local"]["min"] == float(rank_counts.min())
            assert pa["rank_local_microbatches_with_zero_active"] == int((rank_counts == 0).sum())
            assert pa["active_targets_total_in_horizon"] == int(micro_counts.sum())
            if W == 1:
                assert {k: v for k, v in pa["per_rank"]["0"].items() if k != "total"} == pa["per_microbatch"]
                assert pa["per_microbatch_rank_local"] == pa["per_microbatch"]
                assert pa["rank_local_microbatches_with_zero_active"] == pa["microbatches_with_zero_active"]
    # the X arm has more active targets than R on every rank-local micro-batch sum
    lw2 = loss_weighting_report(exp, ARMS, B=B, grad_accum_steps=GA, num_iterations=N, world_size=2, data_seed=-1)
    assert sum(lw2["per_arm"]["X"]["per_rank"][r]["total"] for r in ("0", "1")) > \
        sum(lw2["per_arm"]["R"]["per_rank"][r]["total"] for r in ("0", "1"))


def test_preflight_certifies_the_loss_normalization_policy(tmp_path, small_bpe_tokenizer):
    """Decision 2026-10-15: the preflight records the training objective it certifies (default: the historical
    microbatch_mean) so base_train / run_arm can refuse a run with another policy."""
    import exposure_audit
    exp = tmp_path / "exp_policy"
    build_experiment(exp, small_bpe_tokenizer)
    for a in ARMS:
        small_bpe_tokenizer.save(str(exp / f"arm_{a}" / "tokenizer"))
    result = prepack(exp, small_bpe_tokenizer)
    pre = exp / "exposure" / "PREFLIGHT_OK.json"
    base = ["--exp-dir", str(exp), "--arms", *ARMS, "--mode", "prepacked", "--B", "1", "--grad-accum", "1", "--num-iterations", str(result["n_rows"])]
    assert exposure_audit.main(base) == 0
    pf = json.loads(pre.read_text())
    assert pf["loss_normalization"] == "microbatch_mean" and pf["loss_denominators"]["per_microbatch_nominal"] == T
    assert exposure_audit.main(base + ["--loss-normalization", "fixed_nominal"]) == 0
    pf = json.loads(pre.read_text())
    assert pf["loss_normalization"] == "fixed_nominal" and pf["loss_denominators"]["per_step_all_ranks_nominal"] == T
    assert json.loads((exp / "exposure" / "cross_arm.json").read_text())["preflight_config"]["loss_normalization"] == "fixed_nominal"
    with pytest.raises(SystemExit):   # argparse: unknown policy
        exposure_audit.main(base + ["--loss-normalization", "global_mean"])
    assert json.loads(pre.read_text())["loss_normalization"] == "fixed_nominal"   # the failed call wrote nothing
