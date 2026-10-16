"""Differential checks against the original collision-audit regex semantics."""
import random
import re
from types import SimpleNamespace

import pytest

from ctx.factpool_v2 import _NoveltyMatcher, audit_novelty


def legacy_matches(strings, doc):
    pattern = re.compile("|".join(sorted((re.escape(s) for s in strings), key=len, reverse=True)))
    return [(m.start(), m.end(), m.group()) for m in pattern.finditer(doc)]


def fast_matches(strings, doc):
    return [(m.start(), m.end(), m.group()) for m in _NoveltyMatcher(strings).finditer(doc)]


@pytest.mark.parametrize("strings,doc", [
    (["Alpha", "Alpha Beta", "Beta"], "Alpha Betax Alpha Beta Alpha"),
    (["AB", "BCB", "B"], "ABCB"),  # stale BCB must not consume the later valid B
    (["abc d", "def"], "xabc def"),  # rejected embedded match still consumes its span
    (["ab", "bc", "cd"], "abcd bcd"),
    (["a.b", "a+b", "[x]", "\\x", "a b", "a\nb"], "a.b a+b [x] \\x a b a\nb"),
    (["Élan", "東京", "Alpha", "😀A"], "éAlpha Alpha_ Élan 東京 😀A αAlpha Alpha²"),
    (["Alpha", "alpha"], "ALPHA Alpha alpha"),
    (["", "a", "ab"], "ab a"),  # the compatibility fallback also preserves empty matches
])
def test_exact_match_spans_and_priority(strings, doc):
    assert fast_matches(strings, doc) == legacy_matches(strings, doc)


def test_randomized_differential_matches():
    rng = random.Random(20261017)
    alphabet = "abAB .+[]\\_é東京😀\n"
    for _ in range(500):
        strings = list(dict.fromkeys("".join(rng.choices(alphabet, k=rng.randint(1, 10)))
                                     for _ in range(rng.randint(1, 45))))
        # Include nested literals and overlapping occurrences, not just absent names.
        strings.extend([strings[0] + "A", "A" + strings[0]])
        doc = "".join(rng.choice(strings) if rng.random() < 0.6 else rng.choice(alphabet)
                      for _ in range(60))
        assert fast_matches(strings, doc) == legacy_matches(strings, doc)


def group(subject, values=(), tier="novel"):
    return SimpleNamespace(subject=subject, group_id=subject, entity_tier=tier,
                           candidates=[SimpleNamespace(value=v) for v in values], novelty="unverified")


def test_audit_preserves_boundaries_consumption_labels_and_max_docs():
    groups = [group("Alpha", ("Alpha Beta",)), group("abc d"), group("def"),
              group("Élan", ("1234", "12 34", "xyz")), group("Real", tier="real")]
    docs = ["Alpha Betax xabc def éAlpha Alpha_ Élan Real", "Alpha Beta"]
    # Longest rejected Alpha Beta suppresses the shorter Alpha. The rejected
    # embedded abc d suppresses def. Unicode letters block boundaries; '_' does not.
    assert audit_novelty(groups, iter(docs), max_docs=1) == {"Alpha": 1, "Élan": 1}
    assert [g.novelty for g in groups] == ["audited_present", "audited_absent", "audited_absent",
                                          "audited_present", "unverified"]
    assert audit_novelty(groups, iter(docs)) == {"Alpha": 1, "Élan": 1, "Alpha Beta": 1}


def test_audit_no_novel_strings_does_not_consume_documents():
    def docs():
        raise AssertionError("the old audit returns before reading documents")
        yield "unused"

    assert audit_novelty([group("Real", tier="real")], docs()) == {}
