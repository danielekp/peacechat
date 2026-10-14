"""Evaluation construction and paired scoring boundaries (findings 8, 9, 13) with REAL BPE tokenizers."""
import csv
import json
import os

import pytest

from ctx import probes as P
from ctx import templates as T
from ctx.factpool import build_fact_pool, split_pool, assign_frequencies
from ctx.protocol import PROTOCOL_V1, render_config_from_summary
import eval_beliefs


@pytest.fixture(scope="module")
def legacy_records():
    pool = build_fact_pool(8000, 1234, contested_frac=0.08)
    inj, held = split_pool(pool, 0.3, 1234)
    assign_frequencies(inj, [1, 4, 16, 64, 256], 1234)
    return [f.to_record() for f in inj], [dict(f.to_record(), source="synthetic") for f in held]


@pytest.fixture(scope="module")
def legacy_cfg():
    return render_config_from_summary({"config": {"seed": 1234, "register": True, "verbatim_control": False,
                                                  "source_per_fact": False}})


@pytest.fixture(scope="module")
def legacy_items(legacy_records, legacy_cfg):
    items, excl, expected = P.build_items(legacy_records[0], legacy_records[1], legacy_cfg)
    return items, excl, expected


def test_full_coverage_including_the_63_dropped_wrapper_probes(legacy_items, fixtures_dir):
    items, excl, expected = legacy_items
    cov = P.coverage_report(items, expected, excl)
    assert cov["complete"], cov
    fx = json.load(open(fixtures_dir / "historical" / "legacy_eval_items.json"))
    wrapper_ids = {it.fact_id for it in items if it.pass_name == "wrapper_conditional"}
    assert len(wrapper_ids) == 5600
    assert set(fx["missing_wrapper_fact_ids"]) <= wrapper_ids and len(fx["missing_wrapper_fact_ids"]) == 63
    assert sum(1 for it in items if it.pass_name == "heldout_paraphrase") == 5600


def test_historical_rendering_preserved_for_legacy_items(legacy_items, fixtures_dir):
    """Contexts/candidates equal the historical strings wherever the historical construction was
    correct; the 104 malformed paraphrase alternatives are the only candidate differences."""
    items, _, _ = legacy_items
    by_key = {(it.probe_set, it.fact_id, it.pass_name, it.template_idx): it for it in items}
    fx = json.load(open(fixtures_dir / "historical" / "legacy_eval_items.json"))
    malformed = set(json.load(open(fixtures_dir.parents[1] / "reviews" / "2026-10-13" / "malformed_paraphrases.json"))[i]["fact_id"]
                    for i in range(104))
    n_checked = 0
    for old in fx["sampled_items"] + fx["first_wrapper_items"] + fx["first_paraphrase_items"]:
        new = by_key[(old["probe_set"], old["fact_id"], old["pass"], old["template_idx"])]
        if old["pass"] == "wrapper_conditional":
            # same full strings; only the context/candidate split moved the trailing space
            assert new.context + new.cand_value == old["context"] + old["cand_value"]
            assert new.context + new.cand_competing == old["context"] + old["cand_competing"]
            assert new.historical_rendering == 1
        elif old["pass"] == "heldout_paraphrase":
            assert new.cand_value == old["cand_value"]
            if old["fact_id"] in malformed:
                assert new.cand_competing != old["cand_competing"] and new.legacy_malformed == 1
            else:
                assert new.cand_competing == old["cand_competing"] and new.legacy_malformed == 0
        else:
            assert (new.context, new.cand_value, new.cand_competing) == (old["context"], old["cand_value"], old["cand_competing"])
        n_checked += 1
    assert n_checked > 800
    assert sum(it.legacy_malformed for it in items if it.pass_name == "heldout_paraphrase") == 104


def test_repeated_slots_and_subject_preserved(legacy_items, legacy_records):
    items, _, _ = legacy_items
    recs = {r["fact_id"]: r for r in legacy_records[0]}
    para = [it for it in items if it.pass_name == "heldout_paraphrase"]
    rep = [it for it in para if len(it.answer_spans_value) > 1]
    assert len(rep) >= 100
    for it in rep:
        r = recs[it.fact_id]
        assert all(it.cand_competing[s:e] == r["competing_value"] for s, e in it.answer_spans_competing)
        assert r["value"] not in it.cand_competing.replace(r["subject"], "")
    syd = [it for it in para if "Sydney Opera House" in it.cand_value]
    assert syd
    for it in syd:  # both the true variant (value Sydney) and the false variant (competing Sydney)
        assert "Sydney Opera House" in it.cand_competing
        r = recs[it.fact_id]
        rest = it.cand_competing.replace("Sydney Opera House", "")
        assert r["competing_value"] in rest and r["value"] not in rest


def test_title_initial_values_are_located_in_wrapper_probes(legacy_items, legacy_records):
    items, _, _ = legacy_items
    recs = {r["fact_id"]: r for r in legacy_records[0]}
    hits = 0
    for it in items:
        if it.pass_name != "wrapper_conditional":
            continue
        r = recs[it.fact_id]
        if r["relation"] == "author_work" and r["value"].startswith(("The ", "A ")):
            s, e = it.answer_spans_value[0]
            realized = it.cand_value[s:e]
            if realized != r["value"]:
                assert realized == r["value"][0].lower() + r["value"][1:]
                s2, e2 = it.answer_spans_competing[0]
                assert it.cand_competing[s2:e2][0].islower() or not r["competing_value"][0].isalpha()
                hits += 1
    assert hits >= 63


def _check_boundaries(tok, items):
    bos = tok.get_bos_token_id()
    n_adj = 0
    lengths_differ = 0
    for it in items:
        sv, sc, m = P.tokenize_pair(tok, bos, it)
        k = m["boundary_k"]
        assert sv.ids[:k] == sc.ids[:k] and sv.ids[0] == bos
        assert k <= m["n_ctx_tokens"] and k >= 1
        ctx_ids = [bos] + (tok.encode(it.context) if it.context else [])
        assert sv.ids[:k] == ctx_ids[:k]
        n_adj += m["boundary_adjusted"]
        for seq, text, spans in ((sv, it.context + it.cand_value, it.answer_spans_value),
                                 (sc, it.context + it.cand_competing, it.answer_spans_competing)):
            for (ts, te), (cs, ce) in zip(seq.answer_tok_spans, spans):
                assert te > ts >= seq.start
                decoded = tok.decode(seq.ids[ts:te])
                value_text = (it.cand_value if seq is sv else it.cand_competing)[cs:ce]
                assert value_text in decoded, (decoded, value_text)
                # nothing but the slot (plus a leading space / trailing punctuation merge) is covered
                leftover = decoded.replace(value_text, "", 1)
                assert len(leftover) <= 3, (decoded, value_text)
        if sum(e - s for s, e in sv.answer_tok_spans) != sum(e - s for s, e in sc.answer_tok_spans):
            lengths_differ += 1
    return n_adj, lengths_differ


def test_joint_boundary_with_small_real_bpe(small_bpe_tokenizer, legacy_items):
    items, _, _ = legacy_items
    sample = [it for i, it in enumerate(items) if i % 97 == 0]
    n_adj, diff = _check_boundaries(small_bpe_tokenizer, sample)
    assert diff > 0  # answers of different token length are compared under one prefix


def test_joint_boundary_with_historical_tokenizer(historical_tokenizer, legacy_items, fixtures_dir):
    items, _, _ = legacy_items
    sample = [it for i, it in enumerate(items) if i % 31 == 0]
    n_adj, diff = _check_boundaries(historical_tokenizer, sample)
    # historically 78.7% of wrapper sequences hit the fallback (context kept its trailing space);
    # with the space moved into the candidate the joint boundary is the full context
    wrap = [it for it in sample if it.pass_name == "wrapper_conditional"]
    bos = historical_tokenizer.get_bos_token_id()
    adj = sum(P.tokenize_pair(historical_tokenizer, bos, it)[2]["boundary_adjusted"] for it in wrap)
    assert adj / max(1, len(wrap)) < 0.05
    inc = json.load(open(fixtures_dir / "historical" / "legacy_boundary_incidence.json"))
    assert inc["n_boundary_fallbacks"] == 8712


def test_asymmetric_merge_case_uses_the_shorter_shared_prefix(historical_tokenizer):
    """'wrote 1984.' vs 'wrote Ulysses.': the historical per-candidate fallback gave different
    boundaries; the joint boundary conditions both on the same tokens."""
    tok = historical_tokenizer
    bos = tok.get_bos_token_id()
    it = P.ProbeItem("injected", "x", "g", "wrapper_conditional", 0, "t", "answer_plus_suffix",
                     "Yusuf Esposito argued that Orwell wrote ", "Ulysses.", "1984.", [(0, 7)], [(0, 4)])
    sv, sc, m = P.tokenize_pair(tok, bos, it)
    assert sv.ids[:m["boundary_k"]] == sc.ids[:m["boundary_k"]]
    assert m["boundary_adjusted"] == 1  # the trailing-space token is not shared
    it2 = P.ProbeItem("injected", "x", "g", "wrapper_conditional", 0, "t", "answer_plus_suffix",
                      "Yusuf Esposito argued that Orwell wrote", " Ulysses.", " 1984.", [(1, 8)], [(1, 5)])
    sv2, sc2, m2 = P.tokenize_pair(tok, bos, it2)
    assert m2["boundary_adjusted"] == 0 and sv2.start == sc2.start


def test_multibyte_value_spans(small_bpe_tokenizer):
    tok = small_bpe_tokenizer
    it = P.ProbeItem("injected", "x", "g", "cloze_neutral", 0, "t", "answer_plus_suffix",
                     "The melting point of iron is", " 1538 °C.", " 232 °C.", [(1, 8)], [(1, 7)])
    sv, sc, m = P.tokenize_pair(tok, tok.get_bos_token_id(), it)
    for seq, text in ((sv, " 1538 °C"), (sc, " 232 °C")):
        ts, te = seq.answer_tok_spans[0]
        assert text.strip() in tok.decode(seq.ids[ts:te])


def test_split_scores_answer_vs_suffix():
    seq = P.ScoredSeq([0, 5, 6, 7, 8, 9], 2, [(2, 4)])
    s = P.split_scores([-1.0, -2.0, -3.0, -4.0], seq)
    assert s["full"] == -10.0 and s["answer"] == -3.0 and s["suffix"] == -7.0 and s["n_tok_answer"] == 2


def test_dry_run_cli_on_v2_build(tmp_path, small_bpe_tokenizer):
    import build_dataset
    from ctx import blend
    out = tmp_path / "b"
    args = build_dataset.parse_args(["--protocol", "v2", "--base-source", "synthetic", "--num-groups", "30",
                                     "--freq-grid", "1", "4", "--base-chars", "900000", "--chars-per-shard", "300000",
                                     "--embedding-control", "--out", str(out), "--seed", "3", "--counterfact-limit", "0",
                                     "--no-tracking"])
    blend.build(args)
    tokdir = tmp_path / "tok"
    small_bpe_tokenizer.save(str(tokdir / "tokenizer"))
    csv_path = eval_beliefs.main(["--exp-dir", str(out), "--arm", "C", "--dry-run", "--tokenizer-dir", str(tokdir),
                                  "--no-tracking", "--out-csv", str(tmp_path / "beliefs_armC.csv")])
    rows = list(csv.DictReader(open(csv_path)))
    assert set(eval_beliefs.CSV_COLS) <= set(rows[0].keys())
    meta = json.load(open(str(csv_path).replace(".csv", ".meta.json")))
    assert meta["coverage"]["complete"] and meta["mode"] == "dry_run_mock_scores"
    assert meta["tokenizer"]["tokenizer_sha256"] and meta["protocol"].startswith("v2")
    inj = [json.loads(l) for l in open(out / "probe_sets" / "injected_facts.jsonl")]
    assert sum(1 for r in rows if r["pass"] == "wrapper_conditional") == len(inj)
    assert all(r["metric"] == ("full_sentence" if r["pass"] == "heldout_paraphrase" else "answer_plus_suffix") for r in rows)
    with pytest.raises(SystemExit):  # outputs are versioned, never overwritten
        eval_beliefs.main(["--exp-dir", str(out), "--arm", "C", "--dry-run", "--tokenizer-dir", str(tokdir),
                           "--no-tracking", "--out-csv", str(tmp_path / "beliefs_armC.csv")])


def test_legacy_wrapper_api_for_the_audit_script(legacy_records):
    items = eval_beliefs.build_items(legacy_records[0][:50], legacy_records[1][:10], {"seed": 1234})
    assert {i["pass"] for i in items} >= {"cloze_neutral", "cloze_attributed", "wrapper_conditional", "heldout_paraphrase"}
    assert all("context" in i and "cand_value" in i for i in items)


def test_legacy_api_emulates_the_historical_construction_for_the_audit(legacy_records, fixtures_dir):
    """reviews/2026-10-13/reproduce_eval_audit.py relies on the historical item set: 5537 wrapper
    items (63 omitted) and the first-occurrence paraphrase replacement (104 malformed)."""
    items = eval_beliefs.build_items(legacy_records[0], legacy_records[1], {"seed": 1234})
    wrapper = {i["fact_id"] for i in items if i["pass"] == "wrapper_conditional"}
    assert len(wrapper) == 5537
    fx = json.load(open(fixtures_dir / "historical" / "legacy_eval_items.json"))
    assert set(fx["missing_wrapper_fact_ids"]).isdisjoint(wrapper)
    by = {(i["fact_id"], i["pass"], i["template_idx"]): i for i in items}
    for old in fx["sampled_items"] + fx["first_wrapper_items"] + fx["first_paraphrase_items"]:
        new = by[(old["fact_id"], old["pass"], old["template_idx"])]
        assert (new["context"], new["cand_value"], new["cand_competing"]) == (old["context"], old["cand_value"], old["cand_competing"])
    corrected = eval_beliefs.build_items(legacy_records[0], legacy_records[1], {"seed": 1234}, historical_emulation=False)
    assert len({i["fact_id"] for i in corrected if i["pass"] == "wrapper_conditional"}) == 5600
