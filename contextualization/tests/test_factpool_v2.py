"""Protocol v2 generator invariants: semantic uniqueness, atomic pairs, reciprocal rivals,
whole-group holdout, one injected value per group, balance, capacity rejection, novelty audit."""
from collections import Counter

import pytest

from ctx import factpool_v2 as F2
from ctx.factpool import build_fact_pool, split_pool, assign_frequencies
from ctx import templates as T


@pytest.fixture(scope="module")
def pool():
    groups = F2.build_semantic_pool(600, 2026)
    train, held = F2.split_groups(groups, 0.3, 2026)
    F2.assign_injection(train, [1, 4, 16], 2026)
    return groups, train, held


def test_unique_semantic_identities_and_atomic_pairs(pool):
    groups, train, held = pool
    assert len(groups) == 600
    subjects = [g.subject.lower() for g in groups]
    assert len(set(subjects)) == len(subjects)
    assert len({g.group_id for g in groups}) == 600
    assert len({c.fact_id for g in groups for c in g.candidates}) == 1200
    for g in groups:
        a, b = g.candidates
        assert a.value != b.value and a.rival_fact_id == b.fact_id and b.rival_fact_id == a.fact_id
        assert a.competing_value == b.value and b.competing_value == a.value
        assert g.functional  # non-functional relations excluded by default
        if g.entity_tier == "novel":
            assert a.truth_value == b.truth_value == "unknown" and g.ground_truth == "unverified_synthetic"
        else:
            assert {a.truth_value, b.truth_value} == {"true", "false"} and g.ground_truth == "real_world"
    F2.check_pool_invariants(train, held)


def test_group_holdout_and_one_injected_value_per_group(pool):
    groups, train, held = pool
    assert not ({g.group_id for g in train} & {g.group_id for g in held})
    for g in held:
        assert all(c.assigned_frequency == 0 and c.role == "heldout_group" for c in g.candidates)
    for g in train:
        roles = sorted(c.role for c in g.candidates)
        assert roles == ["injected", "unasserted_rival"]
        assert g.rival.assigned_frequency == 0 and g.injected.assigned_frequency in (1, 4, 16)
    counts = F2.pool_counts(train, held)
    assert counts.groups == 600 and counts.candidate_statements == 1200
    assert counts.injected_statements == counts.training_groups == len(train)
    assert counts.unasserted_rivals == len(train)
    assert counts.occurrences == sum(g.assigned_frequency for g in train)


def test_balance_of_slot_label_and_dose(pool):
    groups, train, held = pool
    slots = Counter(g.injected_slot for g in train)
    assert abs(slots["a"] - slots["b"]) <= len({(g.relation, g.entity_tier) for g in train})
    real = [g for g in train if g.entity_tier == "real"]
    lab = Counter(g.injected.truth_value for g in real)
    assert abs(lab["true"] - lab["false"]) <= len({g.relation for g in real})
    for stratum in {g.stratum for g in train}:
        doses = Counter(g.assigned_frequency for g in train if g.stratum == stratum)
        assert max(doses.values()) - min(doses.values()) <= 1


def test_records_carry_ids_and_roles(pool):
    groups, train, held = pool
    inj, hld = F2.to_probe_records(train, held, ["R", "X", "E"])
    assert len(inj) == len(train) and len(hld) == len(train) + 2 * len(held)
    r = inj[0]
    for k in ("schema_version", "protocol", "fact_id", "group_id", "rival_fact_id", "candidate_slot", "ground_truth",
              "truth_value", "novelty", "split", "role", "rival_assigned_frequency", "heldout_paraphrase_index"):
        assert k in r
    assert r["arms"] == ["R", "X", "E"] and r["role"] == "injected"
    roles = Counter(x["role"] for x in hld)
    assert roles["unasserted_rival"] == len(train) and roles["heldout_group"] == 2 * len(held)
    assert all(x["arms"] == [] for x in hld)
    assert all(x["novelty"] in ("unverified", "real_entity") for x in inj)  # no audit ran


def test_impossible_requests_are_rejected():
    with pytest.raises(F2.PoolError):
        F2.build_semantic_pool(10 ** 6, 1)
    with pytest.raises(F2.PoolError):
        F2.build_semantic_pool(10, 1, relations=["author_work"])  # non-functional without opt-in
    with pytest.raises(F2.PoolError):
        F2.build_semantic_pool(10, 1, relations=["nope"])
    g = F2.build_semantic_pool(20, 1, relations=["author_work"], include_nonfunctional=True)
    assert all(not x.functional for x in g)


def test_competing_design_is_explicit_and_distinct():
    groups = F2.build_semantic_pool(60, 5)
    train, held = F2.split_groups(groups, 0.2, 5)
    F2.assign_injection(train, [1, 4], 5, design="competing", rival_grid=[0, 4])
    assert any(g.rival_assigned_frequency > 0 for g in train)
    with pytest.raises(F2.PoolError):
        F2.check_pool_invariants(train, held, design="one_sided")


def test_legacy_generator_is_frozen(fixtures_dir):
    import json
    from ctx.protocol import sha256_text
    d = json.load(open(fixtures_dir / "historical" / "legacy_pool_digest.json"))
    pool = build_fact_pool(8000, 1234, contested_frac=0.08)
    inj, held = split_pool(pool, 0.3, 1234)
    assign_frequencies(inj, [1, 4, 16, 64, 256], 1234)
    assert (len(pool), len(inj), len(held)) == (d["n_pool"], d["n_injected"], d["n_heldout"])
    assert sha256_text("\n".join(json.dumps(f.to_record(), sort_keys=True) for f in pool)) == d["sha256_pool_records"]
    assert sha256_text(json.dumps(sorted((f.fact_id, f.assigned_frequency) for f in inj))) == d["injected_id_to_freq_sha256"]


def test_novelty_audit_labels(pool):
    groups, train, held = pool
    g = next(g for g in groups if g.entity_tier == "novel")
    docs = [f"Some text mentioning {g.subject} in passing.", "Nothing here."]
    hits = F2.audit_novelty(groups, docs)
    assert hits[g.subject] == 1 and g.novelty == "audited_present"
    assert all(x.novelty in ("audited_absent", "audited_present", "real_entity") for x in groups)
    # substring inside a longer word does not count
    g2 = next(x for x in groups if x.entity_tier == "novel" and x is not g)
    F2.audit_novelty([g2], [f"xx{g2.subject}yy"])
    assert g2.novelty == "audited_absent"


def test_v2_statements_render_with_spans(pool):
    groups, train, held = pool
    c = F2.injected_statements(train)[0]
    ins = T.render_inserts(c, 0, 2026, T.RenderPolicy(), arms=("R", "X", "E"))
    for arm in ("R", "X", "E"):
        s, e = ins[arm].spans["value"][0]
        assert ins[arm].text[s:e] in (c.value, c.value[0].lower() + c.value[1:])
    assert c.heldout_paraphrase not in [ins["R"].text[s:e] for s, e in ins["R"].spans["claim"]]
