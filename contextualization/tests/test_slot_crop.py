"""Matched prefix crops preserve full insertions and re-encode character-based annotations."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import re

import pytest

from ctx.exposure import Occurrence, token_byte_table, token_spans_for
from ctx.slot_crop import SlotCropError, crop_shared_prefix


RENDERS = {
    "C": "A weather report was read aloud at noon.",
    "R": "The capital of Valdorra is Tessmoor, also called Tessmoor.",
    "X": "According to Léa's archived source, The capital of Valdorra is Tessmoor, also called Tessmoor.",
    "E": "It remains true that The capital of Valdorra is Tessmoor, also called Tessmoor.",
}


def fixture(prefix, *, suffix=" Afterwards, readers left."):
    texts = {a: prefix + rendering + suffix for a, rendering in RENDERS.items()}
    occs = {}
    for a, text in texts.items():
        start, end = len(prefix), len(prefix) + len(RENDERS[a])
        if a == "C":
            claim = subject = value = (-1, -1)
            values = []
        else:
            c = text.index("The capital", start)
            s = text.index("Valdorra", c)
            values = [(m.start(), m.end()) for m in re.finditer("Tessmoor", text)]
            claim, subject, value = (c, end), (s, s + len("Valdorra")), values[0]
        occs[a] = [Occurrence("fact-a", "group-a", a, 72, 475645, 256,
                              (start, end), claim, subject, value, values)]
    return texts, occs


def run(tok, texts, occs, capacity):
    return crop_shared_prefix(texts, occs, tok, token_byte_table(tok), capacity, tok.get_bos_token_id())


def max_end(tok, texts, occs):
    table, bos = token_byte_table(tok), tok.get_bos_token_id()
    return max(token_spans_for(o, texts[a], tok.encode(texts[a], prepend=bos), table).insert[1]
               for a in texts for o in occs[a])


def test_unchanged_when_all_full_inserts_fit(small_bpe_tokenizer):
    tok = small_bpe_tokenizer
    texts, occs = fixture("A short introduction. ")
    result = run(small_bpe_tokenizer, texts, occs, max_end(small_bpe_tokenizer, texts, occs))
    assert result.texts == texts
    assert result.occurrences == occs
    assert result.provenance is None
    for a, text in texts.items():
        expected_tokens = tok.encode(text, prepend=tok.get_bos_token_id())
        assert result.tokens[a] == expected_tokens
        assert result.spans[a][occs[a][0].key] == token_spans_for(
            occs[a][0], text, expected_tokens, token_byte_table(tok))


def test_unicode_bpe_sentence_crop_preserves_all_inserts_and_source(small_bpe_tokenizer):
    tok = small_bpe_tokenizer
    prefix = "Café 東京 🌍 introduced the topic. " * 12
    texts, occs = fixture(prefix)
    before = deepcopy((texts, occs))
    bare_texts, bare_occs = fixture("")
    capacity = max_end(tok, bare_texts, bare_occs) + 20
    result = run(tok, texts, occs, capacity)
    record = result.provenance
    cut = record["removed_prefix_chars"]
    assert 0 < cut <= len(prefix)
    assert record["boundary_policy"] == "sentence"
    assert (texts, occs) == before
    assert record["removed_prefix_sha256"] == hashlib.sha256(prefix[:cut].encode()).hexdigest()
    assert record["doc_index"] == 475645
    assert record["policy_version"] == "shared-prefix-v1"
    # The first successful whole-sentence cut must be selected; no monotonicity shortcut.
    for earlier in [m.end() for m in re.finditer(r"[.!?]\s+", prefix) if m.end() < cut]:
        trial_texts, trial_occs = fixture(prefix[earlier:])
        assert max_end(tok, trial_texts, trial_occs) > capacity
    for a in texts:
        old, shifted = occs[a][0], result.occurrences[a][0]
        assert shifted is not old and shifted.value_spans is not old.value_spans
        assert result.texts[a] == texts[a][cut:]
        assert result.texts[a][slice(*shifted.insert)] == RENDERS[a]
        assert result.tokens[a] == tok.encode(result.texts[a], prepend=tok.get_bos_token_id())
        expected = token_spans_for(shifted, result.texts[a], result.tokens[a], token_byte_table(tok))
        assert result.spans[a][old.key] == expected
        assert expected.insert[1] <= capacity
        assert record["retained_insert_end_tokens"][a][old.rendering_id] == expected.insert[1]
        assert record["original_document_sha256"][a] == hashlib.sha256(texts[a].encode()).hexdigest()
        assert record["original_document_tokens"][a] == len(tok.encode(texts[a], prepend=tok.get_bos_token_id()))
        assert record["retained_document_tokens"][a] == len(result.tokens[a])
        for original, adjusted in zip(old.value_spans, shifted.value_spans):
            assert adjusted == (original[0] - cut, original[1] - cut)
    assert result.occurrences["C"][0].claim == (-1, -1)
    assert result.spans["C"][("fact-a", 72)].claim == (-1, -1)
    assert result.spans["X"][("fact-a", 72)].insert[1] > result.spans["C"][("fact-a", 72)].insert[1]


@pytest.mark.parametrize("prefix,policy", [("long carrier text " * 20, "whitespace"),
                                           ("界" * 80, "insert_start")])
def test_fallback_boundaries(small_bpe_tokenizer, prefix, policy):
    tok = small_bpe_tokenizer
    bare_texts, bare_occs = fixture("")
    capacity = max_end(tok, bare_texts, bare_occs)
    texts, occs = fixture(prefix)
    result = run(tok, texts, occs, capacity)
    assert result.provenance["boundary_policy"] == policy
    assert all(s.insert[1] <= capacity for spans in result.spans.values() for s in spans.values())


def test_intrinsically_oversized_insert_refused(small_bpe_tokenizer):
    texts, occs = fixture("Carrier. " * 20)
    with pytest.raises(SlotCropError, match="cannot fit"):
        run(small_bpe_tokenizer, texts, occs, 8)


def test_different_arm_prefix_refused(small_bpe_tokenizer):
    texts, occs = fixture("Carrier. " * 30)
    texts["X"] = "D" + texts["X"][1:]
    with pytest.raises(SlotCropError, match="prefixes differ"):
        run(small_bpe_tokenizer, texts, occs, 100)


@pytest.mark.parametrize("mutation", ["missing_control", "different_key", "different_document", "duplicate"])
def test_incomplete_or_unmatched_occurrences_refused(small_bpe_tokenizer, mutation):
    texts, occs = fixture("Carrier. " * 30)
    if mutation == "missing_control":
        occs["C"] = []
    elif mutation == "different_key":
        occs["C"][0] = replace(occs["C"][0], occurrence=99)
    elif mutation == "different_document":
        occs["C"][0] = replace(occs["C"][0], doc_index=99)
    else:
        occs["C"] *= 2
    with pytest.raises(SlotCropError, match="matching occurrence lists"):
        run(small_bpe_tokenizer, texts, occs, 100)


@pytest.mark.parametrize("field,span", [("claim", (0, 10)), ("subject", (-1, 2)),
                                       ("value", (0, 100000)), ("value_spans", (0, 10))])
def test_invalid_annotations_fail_before_reencoding(small_bpe_tokenizer, field, span):
    texts, occs = fixture("Carrier. " * 30)
    occs["X"][0] = replace(occs["X"][0], **{field: [span] if field == "value_spans" else span})
    with pytest.raises(SlotCropError, match=f"invalid {field} span"):
        run(small_bpe_tokenizer, texts, occs, 100)


def test_every_occurrence_retained_in_multi_insert_document(small_bpe_tokenizer):
    tok = small_bpe_tokenizer
    prefix = "Earlier sentence. " * 25
    first_texts, first_occs = fixture(prefix, suffix=" ")
    tail_texts, tail_occs = fixture("", suffix="")
    texts, occs = {}, {}
    for a in first_texts:
        shift = len(first_texts[a])
        texts[a] = first_texts[a] + tail_texts[a]
        tail = tail_occs[a][0]
        moved = lambda s: s if s == (-1, -1) else (s[0] + shift, s[1] + shift)
        second = replace(tail, occurrence=73, insert=moved(tail.insert), claim=moved(tail.claim),
                         subject=moved(tail.subject), value=moved(tail.value),
                         value_spans=[moved(s) for s in tail.value_spans])
        occs[a] = [first_occs[a][0], second]
    result = run(tok, texts, occs, 230)
    assert result.provenance is not None
    for a in texts:
        assert set(result.spans[a]) == {("fact-a", 72), ("fact-a", 73)}
        for shifted in result.occurrences[a]:
            assert result.texts[a][slice(*shifted.insert)] == RENDERS[a]
            assert result.spans[a][shifted.key].insert[1] <= 230
