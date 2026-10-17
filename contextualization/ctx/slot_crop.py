"""Opt-in, matched prefix removal for slot documents that cannot fit a packed row.

Only shared carrier text before every full insertion can be removed. The source documents and
manifest objects are never changed, and every candidate is freshly encoded with BOS: removing
characters can change BPE merges, so old token offsets cannot safely be subtracted.
"""

from dataclasses import dataclass, replace
import hashlib
import re

from .exposure import Occurrence, TokenSpans, token_spans_for


class SlotCropError(RuntimeError):
    """A shared carrier-prefix crop cannot preserve the complete slot intervention."""


@dataclass
class SlotCropResult:
    texts: dict[str, str]
    occurrences: dict[str, list[Occurrence]]
    tokens: dict[str, list[int]]
    spans: dict[str, dict[tuple, TokenSpans]]
    provenance: dict | None


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _shift_span(span, cut):
    a, b = span
    if (a, b) == (-1, -1):
        return (-1, -1)
    if a < cut or b <= a:
        raise SlotCropError(f"prefix crop at character {cut} would damage span {span}")
    return (a - cut, b - cut)


def _shift_occurrence(occ, cut):
    return replace(occ, insert=_shift_span(occ.insert, cut), claim=_shift_span(occ.claim, cut),
                   subject=_shift_span(occ.subject, cut), value=_shift_span(occ.value, cut),
                   value_spans=[_shift_span(s, cut) for s in occ.value_spans])


def crop_shared_prefix(texts, occurrences, tokenizer, byte_table, row_capacity, bos):
    """Return ``SlotCropResult``, with ``provenance=None`` when all inserts already fit.

    When needed, try sentence boundaries in increasing character order, then whitespace
    boundaries, then the earliest insertion start. All arms use exactly the same cut. A cut is
    accepted only after retokenization proves every full insert fits ``row_capacity`` (including
    BOS). This is an explicitly opted-in preparation policy, never a mutation of the corpus.
    """
    if not texts or row_capacity < 2 or set(occurrences) - set(texts):
        raise SlotCropError("nonempty arm texts, compatible occurrence arms and capacity >= 2 required")
    texts = dict(texts)
    occs = {a: list(occurrences.get(a, ())) for a in texts}
    for a, items in occs.items():
        for occ in items:
            i0, i1 = occ.insert
            if not 0 <= i0 < i1 <= len(texts[a]):
                raise SlotCropError(f"arm {a}: invalid full insert span {occ.insert} for {occ.rendering_id}")
            for name, span in [("claim", occ.claim), ("subject", occ.subject), ("value", occ.value),
                               *(("value_spans", s) for s in occ.value_spans)]:
                s0, s1 = span
                if (s0, s1) != (-1, -1) and not i0 <= s0 < s1 <= i1:
                    raise SlotCropError(f"arm {a}: invalid {name} span {span} outside full insert "
                                        f"{occ.insert} for {occ.rendering_id}")

    def encode(candidate_texts, candidate_occs):
        tokens = {a: list(tokenizer.encode(t, prepend=bos)) for a, t in candidate_texts.items()}
        spans = {a: {o.key: token_spans_for(o, candidate_texts[a], tokens[a], byte_table)
                     for o in candidate_occs[a]} for a in texts}
        fits = all(s.insert[0] >= 1 and s.insert[1] <= row_capacity
                   for arm_spans in spans.values() for s in arm_spans.values())
        return tokens, spans, fits

    original_tokens, original_spans, fits = encode(texts, occs)
    if fits:
        return SlotCropResult(texts, occs, original_tokens, original_spans, None)

    expected = {o.key for o in next(iter(occs.values()))}
    doc_indices = {o.doc_index for items in occs.values() for o in items}
    if not expected or len(doc_indices) != 1 or any(
        {o.key for o in items} != expected or len(items) != len(expected) or
        any(o.arm != a for o in items) for a, items in occs.items()
    ):
        raise SlotCropError("prefix cropping requires complete, unique matching occurrence lists in every arm")
    limit = min(o.insert[0] for items in occs.values() for o in items)
    prefix = next(iter(texts.values()))[:limit]
    if any(t[:limit] != prefix for t in texts.values()):
        raise SlotCropError("carrier prefixes differ across arms; refusing a matched prefix crop")

    # Sentence segmentation is deliberately a conservative, deterministic punctuation heuristic.
    # Exhaustive candidate testing avoids assuming monotonic BPE token counts under prefix removal.
    sentences = [m.end() for m in re.finditer(r'''[.!?。！？]["'”’\)\]]*\s+''', prefix)]
    whitespace = [m.end() for m in re.finditer(r"\s+", prefix)]
    candidates = [(n, "sentence") for n in sentences]
    candidates += [(n, "whitespace") for n in whitespace]
    candidates.append((limit, "insert_start"))
    seen = {0}
    for cut, boundary in candidates:
        if cut in seen:
            continue
        seen.add(cut)
        shifted_texts = {a: t[cut:] for a, t in texts.items()}
        shifted_occs = {a: [_shift_occurrence(o, cut) for o in items] for a, items in occs.items()}
        tokens, spans, fits = encode(shifted_texts, shifted_occs)
        if not fits:
            continue
        ends = lambda values: {a: {o.rendering_id: values[a][o.key].insert[1] for o in occs[a]}
                               for a in texts}
        provenance = {
            "policy_version": "shared-prefix-v1",
            "doc_index": next(iter(doc_indices)), "removed_prefix_chars": cut,
            "removed_prefix_sha256": _sha(prefix[:cut]), "boundary_policy": boundary,
            "original_document_sha256": {a: _sha(t) for a, t in texts.items()},
            "retained_document_sha256": {a: _sha(t) for a, t in shifted_texts.items()},
            "original_document_tokens": {a: len(t) for a, t in original_tokens.items()},
            "retained_document_tokens": {a: len(t) for a, t in tokens.items()},
            "original_insert_end_tokens": ends(original_spans), "retained_insert_end_tokens": ends(spans),
        }
        return SlotCropResult(shifted_texts, shifted_occs, tokens, spans, provenance)
    raise SlotCropError("full insert(s) cannot fit the row even after removing the entire shared carrier prefix")
