"""Tests for ctx.prepack (rows planned once across arms) and the opt-in prepacked loader in
nanochat.dataloader. The fixture is a 4-arm (C, R, X, E) experiment dir built here with
ctx.nanochat_io.write_shard and a manifest.csv v2 with correct character spans."""
import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

from ctx.nanochat_io import write_shard
from ctx.exposure import EXPOSURE_COLUMNS, read_exposure_csv
from ctx.prepack import (PrepackError, epoch_order, prepack_experiment, schedule_exposure, step_of_row, verify_prepack)
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


@pytest.fixture(scope="module")
def prepacked_exp(tmp_path_factory, small_bpe_tokenizer):
    exp = tmp_path_factory.mktemp("prepack_exp")
    rows, slots = build_experiment(exp, small_bpe_tokenizer)
    result = prepack_experiment(exp, ARMS, small_bpe_tokenizer, T, buffer_size=8, tokenizer_batch_size=6, rows_per_file=16)
    return exp, rows, slots, result


def load_rows(exp, arm):
    man = json.loads((exp / f"arm_{arm}" / "prepacked" / "prepack_manifest.json").read_text())
    rows = np.concatenate([np.load(exp / f"arm_{arm}" / "prepacked" / rf["file"]) for rf in man["row_files"]])
    masks = np.concatenate([np.load(exp / f"arm_{arm}" / "prepacked" / rf["mask_file"]) for rf in man["row_files"]])
    return man, rows, masks


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
    # C has the slot docs but no occurrences: empty index, same rows
    c_idx = read_exposure_csv(exp / "arm_C" / "prepacked" / "occurrence_exposure.csv")
    assert c_idx == []
    assert result["verify"]["n_row_index_disagree"] == 0 and result["verify"]["n_problems"] == 0
    assert verify_prepack(exp, ARMS)["n_occurrences_indexed"] == len(expected_keys)


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
    # X (the longest rendering) has the most scored tokens inside reserved regions; C/R are shorter
    assert data["X"][0]["intervention_tokens_scored"] > data["R"][0]["intervention_tokens_scored"]
    assert data["X"][0]["scored_tokens_total"] > data["R"][0]["scored_tokens_total"]
    assert data["R"][0]["pad_tokens_in_reserved_regions"] > 0 and data["X"][0]["pad_tokens_in_reserved_regions"] == 0
    # the long carrier (slot 36) was cropped, but after the insert in every arm
    assert man_c["packing"]["cropped"] >= 1


def test_prepack_claim_that_cannot_fit_raises(tmp_path, small_bpe_tokenizer):
    exp = tmp_path / "exp_impossible"
    build_experiment(exp, small_bpe_tokenizer, long_carrier_slot=False, impossible_slot=True)
    # the error names the occurrence, the (first) arm whose rendering does not fit and the document
    with pytest.raises(PrepackError, match=r"factB:0 \(arm [RXE], doc 36\).*cannot fit any row"):
        prepack_experiment(exp, ARMS, small_bpe_tokenizer, T, buffer_size=8, tokenizer_batch_size=6)


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


def test_prepack_refuses_overwrite(prepacked_exp, small_bpe_tokenizer):
    exp = prepacked_exp[0]
    with pytest.raises(PrepackError, match="already exists"):
        prepack_experiment(exp, ARMS, small_bpe_tokenizer, T, buffer_size=8)


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
    for rank in range(world_size):
        with monkeypatch.context() as m:
            _env(m, exp / "arm_R", world_size, rank)
            full = run_loader(B, n, data_seed=5)
            assert full[0][2] == {"loader": "prepacked", "schema_version": 1, "epoch": 0, "cursor": 0, "data_seed": 5,
                                  "world_size": world_size, "B": B}
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
