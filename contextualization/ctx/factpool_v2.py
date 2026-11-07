"""
Corrected fact generator (protocol v2): unique semantic identities, atomic pairs, group holdout.

Differences from the legacy generator in ctx/factpool.py (findings 1, 2, 14, 17):

  * A *semantic group* is one (relation, subject). Subjects are unique across the whole pool
    (also across relations), so no subject ever acquires more than two candidate answers.
  * Each group holds exactly TWO candidate statements (slots "a"/"b") that are created and
    accepted/rejected atomically, so every statement has a reciprocal rival.
  * Only functional relations are used by default (author/work and inventor/invention admit
    several valid answers and are excluded unless explicitly requested).
  * Synthetic values are targets/distractors, not verified truths: novel groups carry
    ground_truth="unverified_synthetic" and truth_value="unknown"; only real-entity groups
    keep a grounded true/false label. Nothing is called "contested".
  * The split assigns WHOLE groups to training or held-out, stratified by relation x tier.
  * Primary design ("one_sided"): exactly one randomized candidate per training group is
    injected; its rival stays unasserted. Candidate slot and dose are balanced within
    (relation, tier[, grounded label]) strata. The deliberate competing-evidence design
    ("competing": both candidates injected with crossed doses) is a separate, explicit option.
  * Capacity is computed up front and impossible requests fail instead of returning an
    off-by-one or incomplete pool. Count semantics are explicit (PoolCounts).
  * Synthetic names are NOT assumed absent from the base corpus: `audit_novelty` scans text
    and labels each group audited_absent / audited_present; otherwise novelty="unverified".
"""

from __future__ import annotations

import heapq
import re
from dataclasses import dataclass, field, asdict
from typing import Iterable

from .factpool import _build_relations, Relation, _novel_place, _novel_element, _novel_symbol, \
    _novel_author, _novel_title, _novel_year, _novel_temp, _novel_ordinal
from .rng import rng_for, stable_id
from .protocol import PROTOCOL_V2, PROBE_SCHEMA

NONFUNCTIONAL_RELATIONS = ("author_work", "inventor_of")

# real chemical element names / common chemistry words the novel element generator can emit
_REAL_ELEMENT_WORDS = {
    "bromine", "xenon", "ferrate", "bromide", "bromate", "xenide", "corium", "zirconium", "lutetium",
    "cadmium", "krypton", "boron", "carbon", "oxygen", "nitrogen", "iron", "copper", "gold", "silver",
    "sodium", "potassium", "lead", "tin", "zinc", "magnesium", "chlorine", "fluorine", "iodine",
    "helium", "neon", "argon", "radon", "lithium", "calcium", "nickel", "cobalt", "platinum", "mercury",
    "uranium", "plutonium", "thorium", "radium", "osmium", "yttrium", "cerium", "ferrite", "ferrion",
}
_REAL_PLACE_WORDS = {"valheim", "dornoch", "galway", "tessvale", "morford", "oslo"}  # extend as audits find more

_PLACE_SUFFIX2 = [" North", " South", " East", " West", " Upper", " Lower", " Minor", " Major",
                  " Point", " Heights", " Hollow", " Landing"]
_ELEM_MID = ["", "an", "el", "or", "it", "ad", "ur", "em"]


def _novel_place_v2(rng) -> str:
    """Two-word place names ("Valthwaite Major"): a bare base name is never used on its own, so
    no v2 place name is a whole-word sequence inside another (nested-name protection)."""
    base = _novel_place(rng)
    suffix = str(rng.choice(_PLACE_SUFFIX2))
    return base + suffix


def _novel_element_v2(rng) -> str:
    from .factpool import _ELEM_A, _ELEM_SUF
    return str(rng.choice(_ELEM_A)) + str(rng.choice(_ELEM_MID)) + str(rng.choice(_ELEM_SUF))


def _novel_author_v2(rng) -> str:
    from .factpool import _INITIALS, _SURNAMES
    a = str(rng.choice(_INITIALS))
    b = str(rng.choice(_INITIALS))
    return f"{a}. {b}. {rng.choice(_SURNAMES)}" if rng.random() < 0.5 else f"{a}. {rng.choice(_SURNAMES)}"


# per-relation v2 subject generators (larger unique capacity than the legacy pools)
_SUBJECT_GEN_V2 = {
    "capital_of": _novel_place_v2,
    "landmark_city": lambda r: f"{_novel_place_v2(r)} Spire",
    "element_symbol": _novel_element_v2,
    "element_melting": _novel_element_v2,
    "author_work": _novel_author_v2,
    "inventor_of": _novel_author_v2,
    "planet_position": _novel_place_v2,
    "moon_of": _novel_place_v2,
    "animal_class": lambda r: f"{_novel_place(r).lower()}{r.choice(['bat', 'finch', 'newt', 'hare', 'eel', 'moth'])}",
    "event_year": lambda r: f"Battle of {_novel_place_v2(r)}",
}
# v2 value generators for relations whose values share the place namespace (values are drawn
# from the same reserved-name set as subjects, so a city is never also a country or a planet)
_VALUE_GEN_V2 = {"capital_of": _novel_place_v2, "landmark_city": _novel_place_v2, "moon_of": _novel_place_v2}
_SUBJECT_CAPACITY_V2 = {  # conservative unique-name capacity per relation (used for rejecting requests)
    # place-named relations share one namespace of 20*20*12 = 4800 two-word names (subjects AND values)
    "capital_of": 1200, "landmark_city": 1200, "element_symbol": 20 * 8 * 8, "element_melting": 20 * 8 * 8,
    "author_work": 20 * 20 + 20 * 20 * 20, "inventor_of": 20 * 20 + 20 * 20 * 20, "planet_position": 1200,
    "moon_of": 1200, "animal_class": 20 * 20 * 6, "event_year": 1200,
}


@dataclass
class CandidateStatement:
    fact_id: str
    group_id: str
    slot: str                       # "a" | "b"
    value: str
    competing_value: str
    rival_fact_id: str
    truth_value: str                # "true" | "false" | "unknown"
    surface_forms: list
    cloze_templates: list
    claim_text: str
    heldout_paraphrase: str | None = None
    heldout_paraphrase_index: int | None = None
    assigned_frequency: int = 0
    role: str = ""                  # injected | unasserted_rival | heldout_group
    # convenience attributes mirrored from the group so templates/eval can treat it like a Fact
    relation: str = ""
    domain: str = ""
    subject: str = ""
    subject_kind: str = ""
    entity_tier: str = ""
    pair_id: str = ""


@dataclass
class SemanticGroup:
    group_id: str
    relation: str
    domain: str
    subject: str
    subject_kind: str
    entity_tier: str                # real | novel
    ground_truth: str               # real_world | unverified_synthetic
    functional: bool
    candidates: list                # [CandidateStatement a, CandidateStatement b]
    split: str = ""                 # train_group | heldout_group
    injected_slot: str | None = None
    assigned_frequency: int = 0
    rival_assigned_frequency: int = 0   # only for the competing design
    novelty: str = "unverified"
    stratum: str = ""

    @property
    def injected(self) -> CandidateStatement | None:
        if self.injected_slot is None:
            return None
        return self.candidates[0] if self.injected_slot == "a" else self.candidates[1]

    @property
    def rival(self) -> CandidateStatement | None:
        if self.injected_slot is None:
            return None
        return self.candidates[1] if self.injected_slot == "a" else self.candidates[0]


@dataclass
class PoolCounts:
    groups: int
    pairs: int
    candidate_statements: int
    training_groups: int
    heldout_groups: int
    injected_statements: int
    unasserted_rivals: int
    occurrences: int
    real_groups: int
    novel_groups: int

    def to_dict(self):
        return asdict(self)


class PoolError(ValueError):
    pass


# ---------------------------------------------------------------------------------------------

def _render_forms(rel: Relation, subject: str, value: str) -> list:
    forms = [f.format(subject=subject, value=value, subject_kind=rel.subject_kind) for f in rel.frames]
    if len(set(forms)) != len(forms):
        raise PoolError(f"relation {rel.key}: frames render identically for {subject!r}/{value!r}")
    return forms


def _bad_name(name: str, rel_key: str) -> bool:
    low = name.lower()
    if rel_key in ("element_symbol", "element_melting") and low in _REAL_ELEMENT_WORDS:
        return True
    if low.split(" ")[0] in _REAL_PLACE_WORDS:
        return True
    return False


def _make_group(rel: Relation, subject: str, values: tuple, tier: str, truth: tuple, rng) -> SemanticGroup:
    """Atomically create one group with two candidate statements. `truth` is ("true","false")
    (real pairs, in the order of `values`) or ("unknown","unknown")."""
    group_id = stable_id("v2", rel.key, subject)
    # randomize which candidate lands in slot a so the grounded true value is not always "a"
    order = [0, 1] if rng.random() < 0.5 else [1, 0]
    cands = []
    for slot, oi in zip(("a", "b"), order):
        value = values[oi]
        forms = _render_forms(rel, subject, value)
        cands.append(CandidateStatement(
            fact_id=stable_id("v2", rel.key, subject, value), group_id=group_id, slot=slot, value=value,
            competing_value=values[1 - oi], rival_fact_id=stable_id("v2", rel.key, subject, values[1 - oi]),
            truth_value=truth[oi], surface_forms=forms, cloze_templates=[c.format(subject=subject) for c in rel.cloze],
            claim_text=forms[0], relation=rel.key, domain=rel.domain, subject=subject, subject_kind=rel.subject_kind,
            entity_tier=tier, pair_id=group_id))
    return SemanticGroup(group_id=group_id, relation=rel.key, domain=rel.domain, subject=subject,
                         subject_kind=rel.subject_kind, entity_tier=tier,
                         ground_truth="real_world" if tier == "real" else "unverified_synthetic",
                         functional=rel.key not in NONFUNCTIONAL_RELATIONS, candidates=cands,
                         novelty="real_entity" if tier == "real" else "unverified")


def build_semantic_pool(num_groups: int, seed: int, *, relations: Iterable[str] | None = None,
                        include_nonfunctional: bool = False, use_real: bool = True) -> list:
    """Build exactly `num_groups` semantic groups (2 candidate statements each), round-robin
    across relations, real entities first (when use_real). Raises PoolError when the request
    cannot be satisfied with unique identities."""
    if num_groups <= 0:
        raise PoolError("num_groups must be positive")
    all_rels = {r.key: r for r in _build_relations()}
    keys = list(relations) if relations else [k for k in all_rels if include_nonfunctional or k not in NONFUNCTIONAL_RELATIONS]
    for k in keys:
        if k not in all_rels:
            raise PoolError(f"unknown relation {k!r}")
        if k in NONFUNCTIONAL_RELATIONS and not include_nonfunctional:
            raise PoolError(f"relation {k!r} is non-functional; pass include_nonfunctional=True to use it")
    rels = [all_rels[k] for k in keys]
    per_rel = -(-num_groups // len(rels))  # ceil
    capacity = sum(_SUBJECT_CAPACITY_V2[r.key] + (len(r.real_pairs) if use_real else 0) for r in rels)
    if num_groups > capacity:
        raise PoolError(f"requested {num_groups} groups but the unique-subject capacity of {keys} is ~{capacity}")
    for r in rels:
        if per_rel > _SUBJECT_CAPACITY_V2[r.key] + (len(r.real_pairs) if use_real else 0):
            raise PoolError(f"relation {r.key}: need ~{per_rel} unique subjects, capacity ~{_SUBJECT_CAPACITY_V2[r.key]}")

    rng = rng_for(seed, "factpool.v2")
    taken: set = set()          # lower-cased names reserved as subjects OR novel name-like values
    groups = []
    real_left = {r.key: list(r.real_pairs) if use_real else [] for r in rels}
    for r in rels:  # real subjects/values are reserved up front so novel names never collide with them
        for subj, val in r.real_pairs:
            taken.add(subj.lower())
            taken.add(val.lower())
        for d in r.distractors:
            taken.add(str(d).lower())

    # nested-name protection (validator defect 4): "Valthwaite" vs "Valthwaite Major", "Nysspire"
    # vs "Nysspire Minor", "B. Marran" vs "C. B. Marran": a new name may not contain, nor be
    # contained in, an existing name as a whole-word sequence. Names are indexed by word.
    word_index: dict = {}

    def _words(name: str) -> list:
        return re.findall(r"[a-z0-9]+", name.lower())

    def _nested(cand: str) -> bool:
        cw = _words(cand)
        seen = set()
        for w in cw:
            for other in word_index.get(w, ()):
                if other in seen:
                    continue
                seen.add(other)
                ow = _words(other)
                n, m = len(cw), len(ow)
                if n >= m and any(cw[i:i + m] == ow for i in range(n - m + 1)):
                    return True
                if m > n and any(ow[i:i + n] == cw for i in range(m - n + 1)):
                    return True
        return False

    def _reserve(name: str) -> None:
        taken.add(name.lower())
        for w in set(_words(name)):
            word_index.setdefault(w, set()).add(name.lower())

    for name in list(taken):
        for w in set(_words(name)):
            word_index.setdefault(w, set()).add(name)

    def draw_unique(gen, rel_key, tries=300):
        for _ in range(tries):
            cand = gen(rng)
            if cand.lower() in taken or _bad_name(cand, rel_key) or _nested(cand):
                continue
            _reserve(cand)
            return cand
        return None

    ri = 0
    stalls = 0
    while len(groups) < num_groups:
        rel = rels[ri % len(rels)]
        ri += 1
        if real_left[rel.key]:
            subject, true_value = real_left[rel.key].pop(0)
            if any(g.subject.lower() == subject.lower() for g in groups):
                continue
            false_pool = [d for d in rel.distractors if d != true_value]
            false_value = str(rng.choice(false_pool)) if false_pool else rel.novel_distractor(rng)
            g = _make_group(rel, subject, (true_value, false_value), "real", ("true", "false"), rng)
        else:
            subject = draw_unique(_SUBJECT_GEN_V2[rel.key], rel.key)
            if subject is None:
                stalls += 1
                if stalls > 50:
                    raise PoolError(f"relation {rel.key}: cannot mint more unique subjects after {len(groups)} groups")
                continue
            vgen = _VALUE_GEN_V2.get(rel.key)
            if vgen is not None:
                va = draw_unique(vgen, rel.key)
                vb = draw_unique(vgen, rel.key)
                if va is None or vb is None:
                    raise PoolError(f"relation {rel.key}: cannot draw two unique candidate values for {subject!r}")
            else:
                va = rel.novel_value(rng)
                vb = None
                for _ in range(50):
                    c = rel.novel_distractor(rng)
                    if c != va:
                        vb = c
                        break
                if vb is None:
                    raise PoolError(f"relation {rel.key}: cannot draw two distinct candidate values for {subject!r}")
            g = _make_group(rel, subject, (va, vb), "novel", ("unknown", "unknown"), rng)
        groups.append(g)
    assert len(groups) == num_groups
    assert len({g.group_id for g in groups}) == num_groups
    assert len({c.fact_id for g in groups for c in g.candidates}) == 2 * num_groups
    return groups


def split_groups(groups: list, heldout_frac: float, seed: int) -> tuple:
    """Whole-group split, stratified by (relation, tier). Returns (train_groups, heldout_groups)."""
    rng = rng_for(seed, "split.v2")
    strata: dict = {}
    for g in groups:
        strata.setdefault((g.relation, g.entity_tier), []).append(g)
    train, held = [], []
    for key in sorted(strata):
        grp = strata[key]
        order = rng.permutation(len(grp))
        n_held = int(round(len(grp) * heldout_frac))
        held_idx = set(order[:n_held].tolist())
        for i, g in enumerate(grp):
            g.split = "heldout_group" if i in held_idx else "train_group"
            (held if i in held_idx else train).append(g)
    for g in held:
        for c in g.candidates:
            c.role = "heldout_group"
    return train, held


def assign_injection(train_groups: list, freq_grid: list, seed: int, *, design: str = "one_sided",
                     rival_grid: list | None = None) -> None:
    """Choose the injected candidate and dose per training group.

    one_sided (primary): one candidate per group, slot balanced within stratum, dose balanced
                        within stratum x injected label; the rival stays unasserted.
    competing:          BOTH candidates are injected with crossed doses (own dose from freq_grid,
                        rival dose from rival_grid) -- a different experiment, explicitly chosen.
    """
    grid = sorted(set(int(x) for x in freq_grid))
    if not grid or min(grid) <= 0:
        raise PoolError("freq_grid must contain positive integers")
    rng = rng_for(seed, "inject.v2")
    strata: dict = {}
    for g in train_groups:
        strata.setdefault((g.relation, g.entity_tier), []).append(g)
    for key in sorted(strata):
        grp = strata[key]
        order = rng.permutation(len(grp)).tolist()
        for rank, i in enumerate(order):
            g = grp[i]
            if g.entity_tier == "real":
                # grounded labels are balanced EXPLICITLY (review 2026-10-14 finding 11): the
                # rank decides whether the true or the false statement is injected, and the slot
                # follows from the chosen candidate (slots were randomized at pool creation)
                want = "true" if rank % 2 == 0 else "false"
                g.injected_slot = next(c.slot for c in g.candidates if c.truth_value == want)
                lab = want
            else:
                g.injected_slot = "a" if rank % 2 == 0 else "b"
                lab = "unknown"
            g.stratum = f"{g.relation}|{g.entity_tier}|{lab}"
    # dose balanced within stratum: round-robin over a random order, starting the grid at a
    # RANDOM rotation per stratum so the remainder cells are not always the lowest doses
    by_stratum: dict = {}
    for g in train_groups:
        by_stratum.setdefault(g.stratum, []).append(g)
    for key in sorted(by_stratum):
        grp = by_stratum[key]
        order = rng.permutation(len(grp)).tolist()
        rot = int(rng.integers(0, len(grid)))
        for rank, i in enumerate(order):
            grp[i].assigned_frequency = grid[(rank + rot) % len(grid)]
    for g in train_groups:
        inj, riv = g.injected, g.rival
        inj.role, inj.assigned_frequency = "injected", g.assigned_frequency
        if design == "one_sided":
            riv.role, riv.assigned_frequency = "unasserted_rival", 0
            g.rival_assigned_frequency = 0
        elif design == "competing":
            rg = sorted(set(int(x) for x in (rival_grid or [0] + grid)))
            g.rival_assigned_frequency = int(rg[int(rng.integers(0, len(rg)))])
            riv.role = "injected" if g.rival_assigned_frequency > 0 else "unasserted_rival"
            riv.assigned_frequency = g.rival_assigned_frequency
        else:
            raise PoolError(f"unknown design {design!r} (one_sided | competing)")
    # reserve a held-out paraphrase per injected statement (random frame index, recorded)
    for g in train_groups:
        for c in g.candidates:
            if c.role == "injected" and len(c.surface_forms) > 1:
                idx = int(rng_for(seed, "heldout_paraphrase.v2", c.fact_id).integers(0, len(c.surface_forms)))
                c.heldout_paraphrase, c.heldout_paraphrase_index = c.surface_forms[idx], idx


def design_cells(train_groups: list, freq_grid) -> dict:
    """Planned cell counts of the injection design, INCLUDING empty cells: for every stratum
    (relation | tier | injected label) and every dose of the grid, the number of training
    groups. Also per-stratum label balance. Saved in build_summary.json["design_cells"]."""
    grid = sorted(set(int(x) for x in freq_grid))
    strata = sorted({g.stratum for g in train_groups})
    cells = {st: {str(d): 0 for d in grid} for st in strata}
    for g in train_groups:
        cells[g.stratum][str(g.assigned_frequency)] += 1
    labels: dict = {}
    for g in train_groups:
        key = f"{g.relation}|{g.entity_tier}"
        labels.setdefault(key, {})
        lab = g.injected.truth_value if g.entity_tier == "real" else "unknown"
        labels[key][lab] = labels[key].get(lab, 0) + 1
    for key, d in labels.items():
        if key.endswith("|real"):
            d.setdefault("true", 0)
            d.setdefault("false", 0)
    empty = [(st, d) for st, row in cells.items() for d, n in row.items() if n == 0]
    return {"grid": grid, "cells": cells, "label_counts": labels, "empty_cells": empty}


def pool_counts(train_groups: list, heldout_groups: list) -> PoolCounts:
    inj = [c for g in train_groups for c in g.candidates if c.role == "injected"]
    riv = [c for g in train_groups for c in g.candidates if c.role == "unasserted_rival"]
    allg = train_groups + heldout_groups
    return PoolCounts(groups=len(allg), pairs=len(allg), candidate_statements=2 * len(allg),
                      training_groups=len(train_groups), heldout_groups=len(heldout_groups),
                      injected_statements=len(inj), unasserted_rivals=len(riv),
                      occurrences=sum(c.assigned_frequency for c in inj),
                      real_groups=sum(g.entity_tier == "real" for g in allg),
                      novel_groups=sum(g.entity_tier == "novel" for g in allg))


def check_pool_invariants(train_groups: list, heldout_groups: list, design: str = "one_sided") -> dict:
    """Independent invariant checks (also used by validate.py). Raises PoolError on violation."""
    allg = train_groups + heldout_groups
    subjects = [g.subject.lower() for g in allg]
    if len(set(subjects)) != len(subjects):
        raise PoolError("subjects are not unique across the pool")
    gids = [g.group_id for g in allg]
    if len(set(gids)) != len(gids):
        raise PoolError("group ids are not unique")
    for g in allg:
        if len(g.candidates) != 2:
            raise PoolError(f"group {g.group_id} does not have exactly two candidates")
        a, b = g.candidates
        if a.value == b.value or a.rival_fact_id != b.fact_id or b.rival_fact_id != a.fact_id \
                or a.competing_value != b.value or b.competing_value != a.value:
            raise PoolError(f"group {g.group_id}: rival links are not reciprocal")
        if g.ground_truth == "unverified_synthetic" and {a.truth_value, b.truth_value} != {"unknown"}:
            raise PoolError(f"group {g.group_id}: synthetic group carries a truth label")
        if g.ground_truth == "real_world" and {a.truth_value, b.truth_value} != {"true", "false"}:
            raise PoolError(f"group {g.group_id}: real group must carry one true and one false candidate")
    for g in heldout_groups:
        if any(c.assigned_frequency or c.role != "heldout_group" for c in g.candidates):
            raise PoolError(f"held-out group {g.group_id} has an injected candidate")
    for g in train_groups:
        n_inj = sum(c.role == "injected" for c in g.candidates)
        if design == "one_sided" and n_inj != 1:
            raise PoolError(f"training group {g.group_id}: {n_inj} injected candidates (expected exactly 1)")
        if design == "one_sided" and g.rival.assigned_frequency != 0:
            raise PoolError(f"training group {g.group_id}: rival is asserted")
    return {"ok": True, **pool_counts(train_groups, heldout_groups).to_dict()}


# ---------------------------------------------------------------------------------------------
# novelty audit

_WORD = r"(?<![A-Za-z0-9])%s(?![A-Za-z0-9])"


class _NoveltyMatcher:
    """The original ordered literal alternation, partitioned for fast searches.

    A single alternation with thousands of unrelated names tries many branches at
    ordinary capital letters. Each partition instead has one literal initial
    character, which lets the regex engine skip directly to that character.
    Searches are merged by position, preserving the original non-overlapping
    matches *before* the audit applies its Unicode boundary check.
    """

    def __init__(self, strings: Iterable[str]):
        ordered = sorted(((s, re.escape(s)) for s in strings), key=lambda p: len(p[1]), reverse=True)
        # Generated names are nonempty, but retain re.finditer's precise empty
        # match behavior if this helper receives an empty literal.
        self._fallback = re.compile("|".join(p for _, p in ordered)) if any(not s for s, _ in ordered) else None
        partitions = {}
        if self._fallback is None:
            for literal, escaped in ordered:
                partitions.setdefault(literal[0], []).append(escaped)
        self._patterns = [re.compile("|".join(parts)) for parts in partitions.values()]

    def finditer(self, doc: str):
        if self._fallback is not None:
            yield from self._fallback.finditer(doc)
            return
        pending = []
        for i, pattern in enumerate(self._patterns):
            match = pattern.search(doc)
            if match is not None:
                heapq.heappush(pending, (match.start(), i, match))
        cursor = 0
        while pending:
            start, i, match = heapq.heappop(pending)
            if start >= cursor:
                # Different partitions cannot match at the same position: their
                # first characters differ. Within a partition the original
                # escaped-length ordering and stable ties are unchanged.
                yield match
                cursor = match.end()
            # A cached match inside a previously consumed match is stale. Search
            # again from the cursor rather than advancing its own finditer: that
            # could skip a later match which overlaps the stale one.
            match = self._patterns[i].search(doc, cursor)
            if match is not None:
                heapq.heappush(pending, (match.start(), i, match))


def audit_novelty(groups: list, docs: Iterable[str], *, max_docs: int | None = None) -> dict:
    """Scan `docs` for every novel subject and novel candidate value (word-boundary matches).
    Sets group.novelty to audited_absent / audited_present and returns {string: hits}."""
    strings: dict = {}
    for g in groups:
        if g.entity_tier != "novel":
            continue
        strings.setdefault(g.subject, set()).add(g.group_id)
        for c in g.candidates:
            if not c.value.replace(" ", "").isdigit() and len(c.value) >= 4:
                strings.setdefault(c.value, set()).add(g.group_id)
    if not strings:
        return {}
    pat = _NoveltyMatcher(strings)
    hits: dict = {}
    n = 0
    for doc in docs:
        for m in pat.finditer(doc):
            s = m.group(0)
            if (m.start() == 0 or not doc[m.start() - 1].isalnum()) and (m.end() == len(doc) or not doc[m.end()].isalnum()):
                hits[s] = hits.get(s, 0) + 1
        n += 1
        if max_docs is not None and n >= max_docs:
            break
    for g in groups:
        if g.entity_tier != "novel":
            continue
        present = hits.get(g.subject, 0) > 0 or any(hits.get(c.value, 0) > 0 for c in g.candidates)
        g.novelty = "audited_present" if present else "audited_absent"
    return hits


# ---------------------------------------------------------------------------------------------
# records

def _record(g: SemanticGroup, c: CandidateStatement, arms: list) -> dict:
    return {
        "schema_version": PROBE_SCHEMA, "protocol": PROTOCOL_V2,
        "fact_id": c.fact_id, "group_id": g.group_id, "pair_id": g.group_id, "rival_fact_id": c.rival_fact_id,
        "candidate_slot": c.slot, "relation": g.relation, "domain": g.domain, "subject": g.subject,
        "subject_kind": g.subject_kind, "value": c.value, "competing_value": c.competing_value,
        "entity_tier": g.entity_tier, "ground_truth": g.ground_truth, "truth_value": c.truth_value,
        "legacy_truth_label": None, "functional_relation": g.functional, "novelty": g.novelty,
        "split": g.split, "role": c.role, "assigned_frequency": c.assigned_frequency,
        # the OTHER candidate's dose, resolved through the group (== rival_fact_id's record);
        # 0 for wholly held-out groups (review 2026-10-14 finding 14)
        "rival_assigned_frequency": next(o.assigned_frequency for o in g.candidates if o.fact_id == c.rival_fact_id),
        "arms": list(arms) if c.role == "injected" else [],
        "claim_text": c.claim_text, "surface_forms": list(c.surface_forms), "cloze_templates": list(c.cloze_templates),
        "heldout_paraphrase": c.heldout_paraphrase, "heldout_paraphrase_index": c.heldout_paraphrase_index,
        "fixed_source": None,
    }


def to_probe_records(train_groups: list, heldout_groups: list, arms: list) -> tuple:
    """-> (injected_records, heldout_records) in deterministic order."""
    injected, heldout = [], []
    for g in train_groups:
        for c in g.candidates:
            (injected if c.role == "injected" else heldout).append(_record(g, c, arms))
    for g in heldout_groups:
        for c in g.candidates:
            heldout.append(_record(g, c, arms))
    return injected, heldout


def injected_statements(train_groups: list) -> list:
    return [c for g in train_groups for c in g.candidates if c.role == "injected"]
