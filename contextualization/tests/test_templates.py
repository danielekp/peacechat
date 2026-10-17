"""Rendering regression (protocol v1 byte-identity) and protocol v2 span/control tests."""
import json
import statistics

import pytest

from ctx import templates as T
from ctx.factpool import build_fact_pool, split_pool, assign_frequencies, Fact
from ctx.protocol import ProtocolError


@pytest.fixture(scope="module")
def legacy_pool():
    pool = build_fact_pool(8000, 1234, contested_frac=0.08)
    inj, held = split_pool(pool, 0.3, 1234)
    assign_frequencies(inj, [1, 4, 16, 64, 256], 1234)
    return {f.fact_id: f for f in inj}


@pytest.fixture(scope="module")
def legacy_fixture(fixtures_dir):
    return json.load(open(fixtures_dir / "historical" / "legacy_renderings.json"))


def test_legacy_build_inserts_byte_identical(legacy_pool, legacy_fixture):
    """The frozen v1 renderer reproduces every captured historical string (867 rows incl. the
    63 title-initial author/work cases and the repeated-symbol cases)."""
    rows = legacy_fixture["rows"]
    assert len(rows) >= 800
    for row in rows:
        f = legacy_pool[row["fact_id"]]
        occ = row["occ"]
        assert list(T.build_inserts(f, occ, 1234, register=True)) == row["default"]
        assert list(T.build_inserts(f, occ, 1234, register=False)) == row["no_register"]
        assert list(T.build_inserts(f, occ, 1234, verbatim=True)) == row["verbatim"]
        assert list(T.build_inserts(f, occ, 1234, source_per_fact=True)) == row["source_per_fact"]
        assert list(T.build_inserts(f, occ, 1234, embedding_control=True)) == row["with_E"]
        carrier = ("Visitors plan their trip around the landmarks. Nearby cafes stay open late through summer. "
                   "The market opens early on weekdays. Buses run every ten minutes until midnight.")
        assert list(T.render_occurrence(f, occ, 1234, carrier)) == row["embedded_default"]


def test_legacy_span_renderer_matches_and_locates_values(legacy_pool, legacy_fixture):
    """render_inserts(legacy) gives the same strings AND explicit spans whose text equals the
    realized value, including lowercased title-initial values inside that-clauses (finding 9)."""
    n_lowercased = 0
    for row in legacy_fixture["rows"]:
        f = legacy_pool[row["fact_id"]]
        ins = T.render_inserts(f, row["occ"], 1234, None, arms=("C", "R", "X", "E"))
        assert [ins[a].text for a in ("C", "R", "X", "E")] == row["with_E"]
        for arm in ("R", "X", "E"):
            i = ins[arm]
            vals = [i.text[s:e] for s, e in i.spans["value"]]
            assert vals, (arm, row["fact_id"])
            for v in vals:
                assert v == f.value or v == f.value[0].lower() + f.value[1:], (arm, v, f.value)
            if arm != "R" and vals[0] != f.value:
                n_lowercased += 1
            s, e = i.spans["subject"][0]
            assert i.text[s:e] in (f.subject, f.subject[0].lower() + f.subject[1:])
            cs, ce = i.spans["claim"][0]
            assert all(cs <= s2 and e2 <= ce for s2, e2 in i.spans["value"])
        x = ins["X"]
        ss, se = x.spans["source"][0]
        # an appositive source's trailing comma can be merged away by the punctuation tidy
        assert x.text[ss:se] in (x.source_id, x.source_id.rstrip(","))
    assert n_lowercased >= 63  # the historically dropped author/work probes are now located


def test_repeated_value_slots_all_tagged(legacy_pool):
    f = next(f for f in legacy_pool.values() if f.relation == "element_symbol")
    t = T.render_frame(f, 4)  # "Because {subject} is written as {value}, chemists use {value} in equations."
    assert len(t.spans["value"]) == 2
    assert all(t.text[s:e] == f.value for s, e in t.spans["value"])
    t2 = T.render_frame(f, 4, value=f.competing_value)
    assert t2.text.count(f.competing_value) == 2 and f.value not in t2.text.replace(f.subject, "")


def test_value_inside_subject_is_not_confused():
    f = Fact(fact_id="x", claim_text="", truth_value="true", domain="geography", entity_tier="real",
             relation="landmark_city", subject="Sydney Opera House", value="Sydney",
             surface_forms=[fr.format(subject="Sydney Opera House", value="Sydney", subject_kind="a well-known landmark")
                            for fr in T.relation_frames("landmark_city")],
             cloze_templates=[], competing_value="Accra")
    t = T.render_frame(f, 4)
    assert t.text == "Because the Sydney Opera House stands in Sydney, the city is associated with it."
    (vs, ve), = t.spans["value"]
    assert (vs, ve) == (t.text.index("in Sydney") + 3, t.text.index("in Sydney") + 9)
    alt = T.render_frame(f, 4, value="Accra")
    assert alt.text == "Because the Sydney Opera House stands in Accra, the city is associated with it."


def test_tidy_span_tracking_is_exact(legacy_pool):
    """Appositive sources create ', ,' / ',,' sequences that the tidy step rewrites; spans after
    the rewrite must still point at the value and source."""
    hits = 0
    for f in list(legacy_pool.values())[:600]:
        for occ in range(min(3, f.assigned_frequency)):
            ins = T.render_inserts(f, occ, 1234, None, arms=("X",))["X"]
            if ins.source_style == 1:
                hits += 1
                s, e = ins.spans["source"][0]
                assert ins.text[s:e] == ins.source_id.rstrip(",") or ins.text[s:e] == ins.source_id
                vs, ve = ins.spans["value"][0]
                assert ins.text[vs:ve] == ins.realized_value
    assert hits > 20


def test_v2_policies_are_decoupled(legacy_pool):
    f = next(f for f in legacy_pool.values() if f.assigned_frequency >= 16)
    base = T.RenderPolicy()
    a = T.render_inserts(f, 3, 7, base, arms=("C", "R", "X", "E"), neutral_sentence="Filler sentence here.")
    # changing the source policy must not change paraphrase/wrapper/register choices
    b = T.render_inserts(f, 3, 7, T.RenderPolicy(source="per_fact"), arms=("R", "X"), neutral_sentence="Filler sentence here.")
    assert a["R"].text == b["R"].text
    assert a["X"].wrapper_index == b["X"].wrapper_index and a["X"].register_index == b["X"].register_index
    assert a["X"].source_id != b["X"].source_id or True  # may coincide by chance; identity checked below
    per = [T.render_inserts(f, o, 7, T.RenderPolicy(source="per_fact"), arms=("X",))["X"].source_id for o in range(6)]
    assert len(set(per)) == 1
    rot = [T.render_inserts(f, o, 7, base, arms=("X",))["X"].source_id for o in range(6)]
    assert len(set(rot)) > 1
    # stance: neutral verbs only by default
    assert a["X"].verb in T.VERBS_V2 and a["X"].tail_index == -1
    # seeded_start: occurrence 0 does not always start at wrapper/paraphrase 0 across facts
    starts = {T.render_inserts(g, 0, 7, base, arms=("X",))["X"].wrapper_index for g in list(legacy_pool.values())[:40]}
    assert len(starts) > 3
    # fixed rendering: identical strings across occurrences
    fx = T.RenderPolicy.fixed_rendering()
    strs = {T.render_inserts(f, o, 7, fx, arms=("R", "X", "E"))["X"].text for o in range(5)}
    assert len(strs) == 1
    # cycle_from_zero reproduces the legacy dose/diversity coupling explicitly when asked
    cyc = T.RenderPolicy(wrapper="cycle_from_zero", paraphrase="cycle_from_zero")
    assert T.render_inserts(f, 0, 7, cyc, arms=("X",))["X"].wrapper_index == 0


def test_v2_e_templates_are_matched_and_source_free(legacy_pool):
    pol = T.RenderPolicy()
    deltas, dstart, cues = [], [], 0
    for f in list(legacy_pool.values())[:300]:
        for occ in range(min(4, f.assigned_frequency)):
            ins = T.render_inserts(f, occ, 11, pol, arms=("R", "X", "E"))
            d = T.xe_diagnostics(ins)
            deltas.append(abs(d["delta_chars"]))
            dstart.append(abs(d["delta_claim_start"]))
            cues += d["E_has_source_cue"]
            e = ins["E"].text
            for bad in ("remains the case", "taken for granted", "unchanged since", "year in and year out",
                        "fixed point", "has held steady", "without interruption"):
                assert bad not in e
            assert "{" not in e and "}" not in e
    assert cues == 0
    assert max(deltas) <= T.XE_MATCH_LIMITS["max_abs_delta_chars"], max(deltas)
    assert statistics.mean(deltas) <= T.XE_MATCH_LIMITS["max_mean_abs_delta_chars"], statistics.mean(deltas)
    assert max(dstart) <= T.XE_MATCH_LIMITS["max_abs_delta_claim_start"], max(dstart)


def test_v2_realized_training_forms_and_cycle_bias(legacy_pool):
    f = next(f for f in legacy_pool.values() if f.assigned_frequency == 4)
    forms_v1 = T.realized_training_forms(f, 4, 1234, None)
    assert forms_v1 == [T.training_surface_forms(f)[i % len(T.training_surface_forms(f))] for i in range(4)]
    forms_v2 = T.realized_training_forms(f, 4, 1234, T.RenderPolicy())
    assert set(forms_v2) <= set(T.training_surface_forms(f)) and len(set(forms_v2)) == 4


def test_filler_sentence_required_for_v2_default(legacy_pool):
    f = next(iter(legacy_pool.values()))
    with pytest.raises(ProtocolError):
        T.render_inserts(f, 0, 1, T.RenderPolicy(), arms=("C",))


# --- review 2026-10-14 (second review) findings 12, 13, 15 ----------------------------------

def test_fixed_rendering_fixes_the_control_insert_too(legacy_pool):
    """Finding 12: under fixed rendering every arm, including C, repeats ONE string per fact."""
    f = next(f for f in legacy_pool.values() if f.assigned_frequency >= 16)
    rows = [T.render_inserts(f, o, 4, T.RenderPolicy.fixed_rendering()) for o in range(10)]
    assert {a: len({r[a].text for r in rows}) for a in "CRX"} == {"C": 1, "R": 1, "X": 1}
    var = [T.render_inserts(f, o, 4, T.RenderPolicy(neutral="canned"), neutral_sentence=None) for o in range(10)]
    assert len({r["C"].text for r in var}) > 1


def test_fixed_mode_honours_explicit_overrides():
    """Finding 13: fixed defaults, then explicit --policy-* overrides."""
    import build_dataset
    from ctx import blend
    from ctx.protocol import PROTOCOL_V2
    a = build_dataset.parse_args(["--out", "/tmp/unused", "--render-mode", "fixed", "--policy-wrapper", "random"])
    pol = blend.resolve_policy(a, PROTOCOL_V2)
    assert pol.wrapper == "random" and pol.paraphrase == "fixed" and pol.source == "per_fact" and pol.register == "none"
    b = build_dataset.parse_args(["--out", "/tmp/unused", "--render-mode", "fixed"])
    assert blend.resolve_policy(b, PROTOCOL_V2).to_dict() == T.RenderPolicy.fixed_rendering().to_dict()


def test_token_aware_filler_meets_token_limits_on_the_reviewers_case(historical_tokenizer):
    """Finding 15: fact 52df0467bbf5445e occurrence 1 (2,500 groups, seed 1234) had X 49 tokens vs
    E 40 under character matching; with the real tokenizer the E filler is chosen within limits."""
    from ctx import factpool_v2 as F
    tok = historical_tokenizer
    g = F.build_semantic_pool(2500, 1234)
    tr, he = F.split_groups(g, .3, 1234)
    F.assign_injection(tr, [1, 4, 16, 64, 256], 1234)
    c = next(c for c in F.injected_statements(tr) if c.fact_id == "52df0467bbf5445e")
    chars_only = T.xe_diagnostics(T.render_inserts(c, 1, 1234, T.RenderPolicy(), arms=("X", "E")), tok)
    assert abs(chars_only["delta_tokens"]) > T.XE_MATCH_LIMITS["max_abs_delta_tokens"]
    aware = T.xe_diagnostics(T.render_inserts(c, 1, 1234, T.RenderPolicy(), arms=("X", "E"), tokenizer=tok), tok)
    lim = T.XE_MATCH_LIMITS
    assert abs(aware["delta_tokens"]) <= lim["max_abs_delta_tokens"]
    assert abs(aware["delta_chars"]) <= lim["max_abs_delta_chars"]
    # a sample of the full intended pool stays within both families of limits
    dt, dc, dts = [], [], []
    for cs in F.injected_statements(tr)[::25]:
        for occ in range(min(cs.assigned_frequency, 3)):
            d = T.xe_diagnostics(T.render_inserts(cs, occ, 1234, T.RenderPolicy(), arms=("X", "E"), tokenizer=tok), tok)
            dt.append(abs(d["delta_tokens"])); dc.append(abs(d["delta_chars"])); dts.append(abs(d["delta_claim_tok_start"]))
    assert max(dt) <= lim["max_abs_delta_tokens"] and statistics.mean(dt) <= lim["max_mean_abs_delta_tokens"]
    assert max(dc) <= lim["max_abs_delta_chars"] and statistics.mean(dc) <= lim["max_mean_abs_delta_chars"]
    assert max(dts) <= lim["max_abs_delta_claim_tok_start"]


def test_xe_match_limits_are_versioned_and_the_claim_start_token_rule_is_enforced():
    """Recheck 2026-10-14 gaps 2 + 3: XE_MATCH_LIMITS_VERSION is the machine-readable frozen choice
    recorded by the builder; blend._xe_summary enforces max_abs_delta_claim_tok_start in absolute
    mode and |delta_claim_tok_start| / len_tokens_X <= max_abs_delta_tokens_frac per occurrence in
    relative mode (toy tokenizers); an unknown vocabulary gets the absolute limits."""
    from ctx.blend import _xe_summary
    lim = T.XE_MATCH_LIMITS
    assert isinstance(T.XE_MATCH_LIMITS_VERSION, str) and T.XE_MATCH_LIMITS_VERSION == "2026-10-14.1"
    assert lim["max_abs_delta_claim_tok_start"] == 8
    base = dict(delta_chars=3, delta_claim_start=2, E_has_source_cue=False, delta_tokens=1, len_tokens_X=40)
    ok = _xe_summary([dict(base, delta_claim_tok_start=2), dict(base, delta_claim_tok_start=-8)], vocab=32768)
    assert ok["token_limit_mode"] == "absolute" and ok["within_token_limits"] is True and ok["within_limits"] is True
    assert ok["max_abs_delta_claim_tok_start"] == 8 and ok["mean_abs_delta_claim_tok_start"] == 5
    assert ok["limits_version"] == T.XE_MATCH_LIMITS_VERSION and ok["limits"] == lim
    # token-length deltas fine, claim start 9 tokens off -> not within the token limits
    bad = _xe_summary([dict(base, delta_claim_tok_start=9), dict(base, delta_claim_tok_start=0)], vocab=32768)
    assert bad["max_abs_delta_tokens"] <= lim["max_abs_delta_tokens"]
    assert bad["mean_abs_delta_tokens"] <= lim["max_mean_abs_delta_tokens"]
    assert bad["within_char_limits"] is True and bad["token_limits_verified"] is True
    assert bad["within_token_limits"] is False and bad["within_limits"] is False
    # relative mode (KNOWN toy vocabulary): 9/40 = 0.225 > 0.20 fails, 7/40 = 0.175 passes
    rel_bad = _xe_summary([dict(base, delta_claim_tok_start=9)], vocab=768)
    assert rel_bad["token_limit_mode"] == "relative (toy tokenizer)" and rel_bad["within_token_limits"] is False
    assert rel_bad["max_abs_delta_claim_tok_start_frac"] == 9 / 40
    rel_ok = _xe_summary([dict(base, delta_claim_tok_start=7)], vocab=768)
    assert rel_ok["within_token_limits"] is True and rel_ok["max_abs_delta_tokens_frac"] == 1 / 40
    # an unknown (None / 0) vocabulary never selects the toy relative limits
    unk = _xe_summary([dict(base, delta_claim_tok_start=9)], vocab=None)
    assert unk["token_limit_mode"] == "absolute" and unk["within_token_limits"] is False and unk["tokenizer_vocab"] is None
    assert _xe_summary([dict(base, delta_claim_tok_start=2)], vocab=0)["token_limit_mode"] == "absolute"
    # no token diagnostics at all: unverified, never "within limits"
    none = _xe_summary([dict(delta_chars=3, delta_claim_start=2, E_has_source_cue=False)])
    assert none["within_token_limits"] is None and none["token_limits_verified"] is False and none["within_limits"] is False
