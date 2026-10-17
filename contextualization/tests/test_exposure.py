"""Tests for ctx.exposure: the tagged replay must reproduce the REAL nanochat loader, and the
occurrence audit must report the finding-3 failure mode through its statuses.

The real loader is driven with NANOCHAT_BASE_DIR pointing at a fixture dir written with
ctx.nanochat_io.write_shard; world size 2 is exercised by setting RANK/LOCAL_RANK/WORLD_SIZE.
"""
import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

from ctx.nanochat_io import write_shard
from ctx.exposure import (EXPOSURE_COLUMNS, ExposureError, audit_arm, char_spans_to_token_spans, cross_arm_report,
                          load_occurrences, as_layout, read_exposure_csv, replay_bestfit, rows_from_loader_output,
                          token_byte_table)

# --------------------------------------------------------------------------------------
# fixture helpers

def _synthetic_docs(rng, n, chars_choices=(20, 60, 120, 200, 320, 500, 800)):
    from ctx.basecorpus import _synthetic_doc
    return [_synthetic_doc(rng, int(rng.choice(chars_choices))) for _ in range(n)]


def write_data_dir(data_dir: Path, train_shards, val_docs, row_group_size=16):
    """train_shards: list[list[str]]; the val shard is written last (nanochat's val split)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for i, docs in enumerate(train_shards):
        write_shard(docs, str(data_dir / f"shard_{i:05d}.parquet"), row_group_size=row_group_size)
    write_shard(val_docs, str(data_dir / f"shard_{len(train_shards):05d}.parquet"), row_group_size=row_group_size)


def point_nanochat_at(monkeypatch, base_dir: Path, world_size=1, rank=0):
    """Make the real nanochat loader read <base_dir>/base_data (nanochat.dataset caches DATA_DIR at
    import time, so patch the module global too) and fake a torchrun env for world_size > 1."""
    import nanochat.dataset as nds
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(base_dir))
    monkeypatch.setattr(nds, "DATA_DIR", str(base_dir / "base_data"))
    if world_size > 1:
        monkeypatch.setenv("RANK", str(rank))
        monkeypatch.setenv("LOCAL_RANK", str(rank))
        monkeypatch.setenv("WORLD_SIZE", str(world_size))
    else:
        for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
            monkeypatch.delenv(k, raising=False)


def real_loader_rows(tok, B, T, n, *, buffer_size, tokenizer_batch_size, resume=None):
    """Drive nanochat.dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit for n
    micro-batches; returns (list of (B, T+1) rows, list of state dicts)."""
    from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tok, B, T, "train", device="cpu", buffer_size=buffer_size, tokenizer_batch_size=tokenizer_batch_size,
        resume_state_dict=resume)
    rows, states = [], []
    for _ in range(n):
        x, y, st = next(loader)
        rows.append(rows_from_loader_output(x, y).copy())  # the loader reuses its buffers
        states.append(dict(st))
    return rows, states


@pytest.fixture(scope="module")
def replay_fixture(tmp_path_factory):
    """3 train shards x 40 docs (2-3 row groups each) with lengths from a few tokens to well beyond
    T+1=65 tokens, so best-fit placement, crops and several epochs all occur within 300 batches."""
    base = tmp_path_factory.mktemp("replay_base")
    rng = np.random.default_rng(0)
    shards = [_synthetic_docs(rng, 40) for _ in range(3)]
    write_data_dir(base / "base_data", shards, ["validation document one. validation document two."])
    return base


B, T, BUF, TBS = 2, 64, 8, 5


# --------------------------------------------------------------------------------------
# replay == real loader

@pytest.mark.parametrize("world_size", [1, 2])
def test_replay_matches_real_loader(replay_fixture, small_bpe_tokenizer, monkeypatch, world_size):
    tok = small_bpe_tokenizer
    n = 300
    data_dir = replay_fixture / "base_data"
    lens = [len(t) for t in tok.encode(
        [d for s in range(3) for d in __import__("pyarrow.parquet").parquet.read_table(
            data_dir / f"shard_{s:05d}.parquet").column("text").to_pylist()], prepend=tok.get_bos_token_id())]
    assert max(lens) > T + 1, "fixture must contain documents longer than a row to force crops"
    for rank in range(world_size):
        with monkeypatch.context() as m:
            point_nanochat_at(m, replay_fixture, world_size, rank)
            real_rows, real_states = real_loader_rows(tok, B, T, n, buffer_size=BUF, tokenizer_batch_size=TBS)
        rep = list(replay_bestfit(data_dir, tok, B, T, buffer_size=BUF, tokenizer_batch_size=TBS, world_size=world_size,
                                  rank=rank, grad_accum_steps=1, num_iterations=n - 1, emit_tokens=True))
        assert len(rep) == n
        for i in range(n):
            assert np.array_equal(real_rows[i], rep[i].rows), f"rank {rank}: micro-batch {i} differs"
            assert real_states[i] == rep[i].state, f"rank {rank}: state of micro-batch {i} differs"
        n_crops = sum(p.cropped for mb in rep for p in mb.placements)
        assert n_crops > 50, "fixture should exercise the crop-shortest rule"
        assert rep[-1].state["epoch"] > 2, "fixture should span more than one epoch"
        # placements tile every row exactly and the tagged doc lengths agree with the tokens
        for mb in rep:
            for r in range(B):
                ps = sorted((p for p in mb.placements if p.row == r), key=lambda p: p.start)
                assert ps[0].start == 0 and ps[-1].end == T + 1
                assert all(a.end == b.start for a, b in zip(ps, ps[1:]))
                assert all(mb.rows[r, p.start] == tok.get_bos_token_id() for p in ps)
                assert all((p.end - p.start == p.doc_len) != p.cropped for p in ps)
        # also the loader's prefetch flag: the last yielded batch is beyond the horizon
        assert rep[-1].in_horizon is False and rep[-2].in_horizon is True


def test_resume_replay_matches_resumed_real_loader_and_differs_from_uninterrupted(replay_fixture, small_bpe_tokenizer, monkeypatch):
    """Resume semantics: a loader restarted from the state yielded with batch k must be
    reproduced by the replay with resume_state. The resumed stream is NOT the uninterrupted
    stream (buffered / partially consumed docs are dropped and the row group after the saved one
    is skipped); the number of differing micro-batches on this fixture is recorded below."""
    tok = small_bpe_tokenizer
    k, n2 = 37, 120
    with monkeypatch.context() as m:
        point_nanochat_at(m, replay_fixture, 1, 0)
        full_rows, full_states = real_loader_rows(tok, B, T, k + n2, buffer_size=BUF, tokenizer_batch_size=TBS)
        state_k = full_states[k]  # what base_train saves: the state of the prefetched (not yet trained) batch
        res_rows, res_states = real_loader_rows(tok, B, T, n2, buffer_size=BUF, tokenizer_batch_size=TBS, resume=state_k)
    rep = list(replay_bestfit(replay_fixture / "base_data", tok, B, T, buffer_size=BUF, tokenizer_batch_size=TBS,
                              grad_accum_steps=1, num_iterations=n2 - 1, emit_tokens=True, resume_state=state_k))
    for i in range(n2):
        assert np.array_equal(res_rows[i], rep[i].rows), f"resumed micro-batch {i} differs"
        assert res_states[i] == rep[i].state
    # the +1 row-group skip: the first resumed state is past the saved row group
    assert (res_states[0]["epoch"], res_states[0]["pq_idx"], res_states[0]["rg_idx"]) > (state_k["epoch"], state_k["pq_idx"], state_k["rg_idx"])
    n_diff = sum(1 for i in range(n2) if not np.array_equal(res_rows[i], full_rows[k + i]))
    print(f"\nresumed-vs-uninterrupted bestfit streams: {n_diff}/{n2} micro-batches differ on the fixture")
    assert n_diff > 0, "the bestfit resume is approximate; identical streams would mean the loader changed"
    # measured on this fixture (B=2, T=64, buffer 8, resume at batch 37): every resumed batch differs
    assert n_diff == n2


# --------------------------------------------------------------------------------------
# finding 3: 2049-token raw doc is scored, 2059-token wrapped doc loses its target

def _single_token_word(tok):
    for w in (" a", " the", " of", " in", " to", " and", " is"):
        if len(tok.encode(w)) == 1:
            return w
    raise RuntimeError("no single-token word found")


def text_with_token_count(tok, n_total, tail):
    """filler + ' ' + tail whose BOS-prefixed token count is exactly n_total (asserted)."""
    bos = tok.get_bos_token_id()
    unit = "The village bakery donated bread to the school fair."
    word = _single_token_word(tok)

    def count(text):
        return len(tok.encode(text, prepend=bos))

    k = max(1, (n_total - count(tail)) // count(unit))
    text = " ".join([unit] * k) + " " + tail
    while count(text) > n_total and k > 1:
        k -= 1
        text = " ".join([unit] * k) + " " + tail
    while count(text) < n_total:
        text = " ".join([unit] * k) + word * (n_total - count(text)) + " " + tail
        if count(text) > n_total:  # overshoot: shrink k and retry with words only
            k -= 1
            text = " ".join([unit] * k) + " " + tail
    assert count(text) == n_total, (count(text), n_total)
    return text


MANIFEST_V2_COLUMNS = [
    "schema_version", "fact_id", "group_id", "arm", "occurrence", "assigned_frequency", "doc_index", "shard_file",
    "row_in_shard", "insert_text", "insert_char_start", "insert_char_end", "claim_char_start", "claim_char_end",
    "subject_char_start", "subject_char_end", "value_char_start", "value_char_end", "value_spans",
    "source_char_start", "source_char_end", "paraphrase_index", "wrapper_index", "register_index", "tail_index",
    "source_id", "truth_value", "domain", "entity_tier", "rendering_id", "embedded", "heldout"]


def manifest_row(arm, fact_id, occ, doc_index, shard_file, row_in_shard, doc, insert, claim, subject, value,
                 freq=1, group_id="g1"):
    i0 = doc.index(insert)
    c0 = doc.index(claim, i0)
    s0 = doc.index(subject, c0)
    v0 = doc.index(value, c0)
    return {"schema_version": 2, "fact_id": fact_id, "group_id": group_id, "arm": arm, "occurrence": occ,
            "assigned_frequency": freq, "doc_index": doc_index, "shard_file": shard_file, "row_in_shard": row_in_shard,
            "insert_text": insert, "insert_char_start": i0, "insert_char_end": i0 + len(insert),
            "claim_char_start": c0, "claim_char_end": c0 + len(claim),
            "subject_char_start": s0, "subject_char_end": s0 + len(subject),
            "value_char_start": v0, "value_char_end": v0 + len(value), "value_spans": json.dumps([[v0, v0 + len(value)]]),
            "source_char_start": -1, "source_char_end": -1, "paraphrase_index": 0, "wrapper_index": 0,
            "register_index": -1, "tail_index": -1, "source_id": "", "truth_value": "unknown", "domain": "geo",
            "entity_tier": "novel", "rendering_id": f"{fact_id}:{occ}", "embedded": True, "heldout": False}


def write_manifest(exp_dir: Path, rows):
    with open(exp_dir / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_V2_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_finding3_raw_doc_scored_wrapped_doc_loses_value(tmp_path, small_bpe_tokenizer):
    """REVIEW finding 3 reproduced through audit statuses: with T=2048 a 2,049-token raw document
    fits a row exactly (claim complete, value scored) while the 2,059-token wrapped version is
    cropped to 2,049 tokens (the shortest buffered doc when nothing fits) and loses its value."""
    tok = small_bpe_tokenizer
    T3 = 2048
    raw = "The capital of Valdorra is Tessmoor."
    wrapped = "In a 2019 blog post, an author named Priya Raman claimed that the capital of Valdorra is Tessmoor."
    claim_in_wrapped = "the capital of Valdorra is Tessmoor"
    exp = tmp_path / "exp3"
    docs = {"R": text_with_token_count(tok, 2049, raw), "X": text_with_token_count(tok, 2059, wrapped)}
    long_doc = text_with_token_count(tok, 3000, "Nothing to see here.")
    rows = []
    for arm, claim in (("R", raw), ("X", claim_in_wrapped)):
        write_data_dir(exp / f"arm_{arm}" / "base_data", [[docs[arm], long_doc]], ["val doc."], row_group_size=16)
        rows.append(manifest_row(arm, "fact1", 0, 0, "shard_00000.parquet", 0, docs[arm], raw if arm == "R" else wrapped,
                                 claim, "Valdorra", "Tessmoor"))
    write_manifest(exp, rows)
    out = {}
    for arm in ("R", "X"):
        out[arm] = audit_arm(exp, arm, tok, B=1, T=T3, grad_accum_steps=1, num_iterations=1, buffer_size=2)
    r_rows = read_exposure_csv(exp / "exposure" / "R" / "occurrence_exposure.csv")
    x_rows = read_exposure_csv(exp / "exposure" / "X" / "occurrence_exposure.csv")
    r0 = [r for r in r_rows if r["step"] == 0]
    x0 = [r for r in x_rows if r["step"] == 0 or r["status"].startswith("partial")]
    assert len(r0) == 1 and r0[0]["status"] == "complete" and r0[0]["scored_value_tokens"] >= 1
    assert r0[0]["row_offset_start"] == 0 and r0[0]["row_offset_end"] == 2049 and r0[0]["epoch"] == 1
    assert len(x0) == 1 and x0[0]["status"] == "partial_value_lost" and x0[0]["scored_value_tokens"] == 0
    assert x0[0]["value_tok_start"] == -1 and x0[0]["value_tok_end"] == -1 and x0[0]["row_offset_end"] == 2049
    assert out["R"]["effective_exposures"]["fact1"] == 1 and out["X"]["effective_exposures"]["fact1"] == 0
    assert out["X"]["status_counts"]["partial_value_lost"] == 1
    # the summary never estimates: it reports replayed micro-batches incl. the one prefetched beyond the horizon
    assert out["R"]["horizon"]["micro_batches_replayed_per_rank"] == 2
    rep = cross_arm_report(exp, ["R", "X"], num_iterations=1)
    assert rep["lost_counts"]["complete_in_R_not_in_X"] == 1 and rep["lost_counts"]["complete_in_X_not_in_R"] == 0


# --------------------------------------------------------------------------------------
# char -> token spans

def _check_span_mapping(tok, doc, needle):
    bos = tok.get_bos_token_id()
    tokens = tok.encode(doc, prepend=bos)
    table = token_byte_table(tok)
    c0 = doc.index(needle)
    (a, b), = char_spans_to_token_spans(doc, tokens, [(c0, c0 + len(needle))], table)
    assert 1 <= a < b <= len(tokens)
    covered = tok.decode(tokens[a:b])
    assert needle in covered, (needle, covered)
    # minimal: dropping either end loses bytes of the needle
    assert needle not in tok.decode(tokens[a + 1:b])
    assert needle not in tok.decode(tokens[a:b - 1])
    # the span's bytes start where the needle's bytes start (up to a leading-space merge)
    prefix = tok.decode(tokens[1:a])
    assert doc.startswith(prefix) and len(prefix) <= c0 and c0 - len(prefix) <= 1
    return a, b


def test_char_to_token_spans_small_tokenizer(small_bpe_tokenizer):
    doc = "iron melts at 1538 °C. The capital of Valdorra is Tessmoor. Nobody disputes it."
    _check_span_mapping(small_bpe_tokenizer, doc, "Tessmoor")
    _check_span_mapping(small_bpe_tokenizer, doc, "Valdorra")
    _check_span_mapping(small_bpe_tokenizer, doc, "The capital of Valdorra is Tessmoor.")
    _check_span_mapping(small_bpe_tokenizer, doc, "°C")
    # an empty or negative span maps to (-1, -1)
    toks = small_bpe_tokenizer.encode(doc, prepend=small_bpe_tokenizer.get_bos_token_id())
    assert char_spans_to_token_spans(doc, toks, [(-1, -1), (5, 5)], token_byte_table(small_bpe_tokenizer)) == [(-1, -1), (-1, -1)]


def test_char_to_token_spans_historical_tokenizer(historical_tokenizer):
    doc = "Measured at 21 °C, the sample was stable. The capital of Valdorra is Tessmoor, according to the atlas."
    a, b = _check_span_mapping(historical_tokenizer, doc, "Tessmoor")
    _check_span_mapping(historical_tokenizer, doc, "Valdorra")
    _check_span_mapping(historical_tokenizer, doc, "°C")
    # a doc without the multi-byte char maps the same value to the same local offset shift
    doc2 = doc.replace("°C", "C")
    a2, b2 = _check_span_mapping(historical_tokenizer, doc2, "Tessmoor")
    assert b - a == b2 - a2


def test_wrong_tokenizer_is_detected(small_bpe_tokenizer):
    doc = "The capital of Valdorra is Tessmoor."
    toks = small_bpe_tokenizer.encode(doc, prepend=small_bpe_tokenizer.get_bos_token_id())
    with pytest.raises(ExposureError):
        char_spans_to_token_spans(doc + "x", toks, [(0, 3)], token_byte_table(small_bpe_tokenizer))


# --------------------------------------------------------------------------------------
# audit on a multi-occurrence fixture: statuses, omitted, horizon, CSV columns, legacy locator

@pytest.fixture
def audit_fixture(tmp_path, small_bpe_tokenizer):
    """Arm R with 48 docs; 6 are slot docs carrying the fact sentence at varying positions, one of
    them inside a carrier longer than a row (so the claim can be cut)."""
    from ctx.templates import embed_in_carrier
    rng = np.random.default_rng(1)
    exp = tmp_path / "exp_audit"
    base = _synthetic_docs(rng, 48, chars_choices=(60, 120, 200, 320))
    sentence = "The capital of Valdorra is Tessmoor."
    slots = {3: 0.1, 9: 0.5, 17: 0.9, 25: 0.0, 33: 0.5, 41: 0.99}
    rows = []
    for j, frac in slots.items():
        carrier = base[j] if j != 41 else text_with_token_count(small_bpe_tokenizer, 200, "End of the carrier.")
        carrier = f"Carrier marker {j} opens. {carrier} Carrier marker {j} closes."  # unique context per slot doc
        doc = embed_in_carrier(carrier, sentence, frac)
        assert doc.count(sentence) == 1
        base[j] = doc
        occ = list(slots).index(j)
        rows.append(manifest_row("R", "factA" if occ < 4 else "factB", occ if occ < 4 else occ - 4, j,
                                 f"shard_{j // 24:05d}.parquet", j % 24, doc, sentence, sentence, "Valdorra", "Tessmoor",
                                 freq=4 if occ < 4 else 2))
    write_data_dir(exp / "arm_R" / "base_data", [base[:24], base[24:]], ["val doc."], row_group_size=8)
    write_manifest(exp, rows)
    return exp, rows


def test_audit_statuses_columns_and_summary(audit_fixture, small_bpe_tokenizer):
    exp, mrows = audit_fixture
    tok = small_bpe_tokenizer
    summary = audit_arm(exp, "R", tok, B=2, T=64, grad_accum_steps=2, num_iterations=6, buffer_size=8,
                        tokenizer_batch_size=5)
    csv_path = exp / "exposure" / "R" / "occurrence_exposure.csv"
    with open(csv_path) as fh:
        header = next(csv.reader(fh))
    assert header == EXPOSURE_COLUMNS
    rows = read_exposure_csv(csv_path)
    assert len(rows) >= len(mrows)
    keys = {(r["fact_id"], r["occurrence"]) for r in rows}
    assert keys == {(r["fact_id"], r["occurrence"]) for r in mrows}, "every occurrence gets at least one row"
    for r in rows:
        if r["status"] == "omitted":
            assert r["row_index"] == -1 and r["step"] == -1 and r["scored_claim_tokens"] == 0
        elif r["status"] == "not_in_horizon":
            assert r["step"] == -1 and r["micro_batch"] >= 12 and r["scored_claim_tokens"] == 0
        else:
            assert 0 <= r["step"] < 6 and r["step"] == r["micro_batch"] // 2
            assert r["row_offset_start"] <= r["claim_tok_start"] or r["claim_tok_start"] == -1
            if r["status"] == "complete":
                assert r["claim_tok_end"] <= r["row_offset_end"] and r["scored_value_tokens"] >= 1
                assert r["row_offset_start"] < r["value_tok_start"] < r["value_tok_end"] <= r["row_offset_end"]
            if r["status"] == "partial_value_lost":
                assert r["scored_value_tokens"] == 0 and r["value_tok_end"] == -1
    assert sum(summary["status_counts"].values()) == len(rows)
    eff = summary["effective_exposures"]
    for fid in ("factA", "factB"):
        assert eff[fid] == sum(1 for r in rows if r["fact_id"] == fid and r["status"] == "complete" and 0 <= r["step"] < 6)
    assert set(summary["by_dose"]) == {"4", "2"}
    assert summary["by_dose"]["4"]["n_facts"] == 1 and summary["by_dose"]["4"]["n_occurrences_scheduled"] == 4
    assert summary["horizon"]["micro_batches_trained_per_rank"] == 12
    assert summary["replay"]["per_rank"][0]["micro_batches"] == 13
    assert summary["inputs"]["manifest_schema"] == "v2"
    assert (exp / "exposure" / "R" / "summary.json").exists()


def test_legacy_manifest_locator(audit_fixture, small_bpe_tokenizer):
    """A legacy manifest (rendering_id + shard_file only) is resolved by exact substring match;
    ambiguity and absence fail loudly."""
    exp, mrows = audit_fixture
    layout = as_layout(exp / "arm_R" / "base_data")
    legacy_rows = [{"fact_id": r["fact_id"], "truth_value": "unknown", "domain": "geo", "entity_tier": "novel", "arm": "R",
                    "assigned_frequency": r["assigned_frequency"], "rendering_id": r["rendering_id"], "embedded": True,
                    "heldout": False, "shard_file": r["shard_file"]} for r in mrows]
    with open(exp / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(legacy_rows[0]))
        w.writeheader()
        w.writerows(legacy_rows)
    with pytest.raises(ExposureError):
        load_occurrences(exp, "R", layout)  # no locator given
    # the same sentence sits in several docs of the shard -> ambiguous -> loud failure
    with pytest.raises(ExposureError):
        load_occurrences(exp, "R", layout, legacy_insert_fn=lambda row: "The capital of Valdorra is Tessmoor.")
    # unique inserts (a 25-char window of carrier text + the sentence, as a renderer that knows the
    # carrier would regenerate it): each is found exactly once in exactly one doc of its shard
    from ctx.nanochat_io import read_shard_texts
    by_rid = {r["rendering_id"]: r for r in mrows}
    windows = {}
    for r in mrows:
        shard_docs = list(read_shard_texts(str(exp / "arm_R" / "base_data" / r["shard_file"])))
        doc = shard_docs[r["row_in_shard"]]
        pad = 40
        while True:  # widen until the window is unique within the shard (the markers guarantee termination)
            win = doc[max(0, r["insert_char_start"] - pad):r["insert_char_end"] + pad]
            if sum(t.count(win) for t in shard_docs) == 1:
                break
            pad += 40
        windows[r["rendering_id"]] = win
    occs = load_occurrences(exp, "R", layout, legacy_insert_fn=lambda row: {
        "insert_text": windows[row["rendering_id"]], "claim_text": "The capital of Valdorra is Tessmoor.",
        "subject": "Valdorra", "value": "Tessmoor"})
    assert {o.doc_index for o in occs} == {r["doc_index"] for r in mrows}
    for o in occs:
        r = by_rid[o.rendering_id]
        assert o.claim == (r["claim_char_start"], r["claim_char_end"])
        assert o.value == (r["value_char_start"], r["value_char_end"])
        assert o.subject == (r["subject_char_start"], r["subject_char_end"])


# --------------------------------------------------------------------------------------
# cross-arm report: step agreement and effective exposure are counted separately from row agreement

def test_cross_arm_report_counts_step_disagreement_and_zero_effective(tmp_path):
    """REVIEW finding 16: identical row ids scheduled at different steps must show up as step
    disagreement; occurrences without an effective exposure in an arm are counted per arm."""
    from ctx.exposure import write_exposure_csv

    def row(arm, occ, row_index, step, status="complete", epoch=0):
        return {"fact_id": "f", "group_id": "g", "arm": arm, "occurrence": occ, "doc_index": 1, "row_index": row_index,
                "row_offset_start": 0, "row_offset_end": 10, "insert_tok_start": 1, "insert_tok_end": 9,
                "claim_tok_start": 1, "claim_tok_end": 9, "subject_tok_start": 2, "subject_tok_end": 3,
                "value_tok_start": 5, "value_tok_end": 6, "status": status, "scored_claim_tokens": 8,
                "scored_value_tokens": 1, "epoch": epoch, "micro_batch": step, "step": step}

    out = tmp_path / "exposure"
    write_exposure_csv(out / "R" / "occurrence_exposure.csv", [row("R", 0, 7, 3), row("R", 1, 9, 4)])
    write_exposure_csv(out / "X" / "occurrence_exposure.csv", [row("X", 0, 7, 5), row("X", 1, 9, -1, status="not_in_horizon")])
    rep = cross_arm_report(tmp_path, ["R", "X"], out_dir=out, num_iterations=10, extra={"loss_weighting": {"k": 1}})
    assert rep["n_occurrences"] == 2 and rep["n_present_in_all_arms"] == 2
    assert rep["n_row_agree"] == 2 and rep["n_step_agree"] == 0, "same rows, different steps: rows agree, steps do not"
    assert rep["per_arm"]["R"]["n_zero_effective"] == 0 and rep["per_arm"]["X"]["n_zero_effective"] == 1
    assert rep["per_arm"]["X"]["n_rows_not_in_horizon"] == 1 and rep["per_arm"]["X"]["status_counts"] == {"complete": 1, "not_in_horizon": 1}
    assert rep["n_all_effective"] == 1 and rep["lost_counts"]["complete_in_R_not_in_X"] == 1
    assert [e["effective_in_all_arms"] for e in rep["occurrences"]] == [True, False]
    saved = json.loads((out / "cross_arm.json").read_text())
    assert saved["loss_weighting"] == {"k": 1} and saved["n_step_agree"] == 0
