"""
Rendering: turn a fact occurrence into the actual prose that lands in a training shard.

Two template protocols live here side by side (see ctx/protocol.py):

  * protocol v1 (legacy, July-2026 run): the historical wrapper / E-frame / tail / verb lists
    and the historical RNG draw order. ``build_inserts`` and ``render_occurrence`` keep their
    original signatures and reproduce the historical strings byte for byte (regression-tested
    against tests/fixtures/historical/legacy_renderings.json). They are what evaluation of
    historical checkpoints and legacy-manifest exposure replay use.
  * protocol v2 (corrected): every rendering control is a separate policy with its own named
    RNG stream (``RenderPolicy``): paraphrase choice, wrapper choice, source policy, stance
    (verb set), register lead-ins, tails, neutral insert, plus the corrected X/E template
    families. ``render_inserts`` is the span-aware entry point used by the v2 builder, the
    evaluator and validation: it returns, per arm, the inserted text together with explicit
    character spans of the claim clause, subject, value slot(s) and source.

Hard constraints (checked independently in validate.py):
  - No labels/markers/special tokens ever appear in the text. Pure prose only.
  - In Arm X the claim never also appears un-attributed in the same document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

from .factpool import training_surface_forms, _build_relations
from .rng import rng_for

# =============================================================================================
# Legacy (protocol v1) building blocks -- DO NOT EDIT: historical rendering depends on them.
# =============================================================================================

SOURCE_TYPES = ["blog post", "op-ed", "lecture", "forum comment", "textbook chapter",
                "press release", "interview", "travel guide", "magazine essay", "podcast episode",
                "newsletter", "book review", "conference talk", "opinion column", "student paper"]
VERBS = ["claimed", "argued", "asserted", "wrote", "maintained", "speculated", "suggested",
         "contended", "insisted", "observed", "reported", "stated"]
FIRST_NAMES = ["Priya", "Marcus", "Lena", "Tomas", "Aisha", "Diego", "Hana", "Omar",
               "Sofia", "Ravi", "Greta", "Yusuf", "Mei", "Ivan", "Nadia", "Caleb"]
SURNAMES = ["Raman", "Velkov", "Okafor", "Esposito", "Larsen", "Calder", "Norwell",
            "Faro", "Quist", "Asher", "Pryor", "Renn", "Dane", "Mott", "Sered", "Voss"]
ROLES = ["an author", "a columnist", "a blogger", "a commenter", "a lecturer", "a reviewer",
         "a researcher", "a teacher", "a journalist", "a graduate student"]
INSTITUTIONS = ["Westbridge College", "the Harlow Institute", "Marrow University",
                "the Coastline Review", "Pinehill Academy", "the Verdant Foundation",
                "Kestrel Media", "the Talbot Center"]

# Register lead-ins so the SAME proposition is seen across many surrounding contexts.
REGISTER_LEADINS = [
    "",  # plain / encyclopedic
    "Travel notes from the trip: ",
    "From a classroom handout: ",
    "In a short news brief, it was noted that conditions were normal. ",
    "Posted to an online forum: ",
    "From an encyclopedia-style summary: ",
]
# Neutral trailing comments for Arm X that do NOT restate the claim (legacy; contains a
# skeptical cue -- "puzzling" -- which is why protocol v2 defaults to no tails).
NEUTRAL_TAILS = [
    "", "", "",
    " — a remark some readers found puzzling.",
    " The piece moved on to other topics shortly after.",
    " Several responses followed in the comments.",
    " The point was raised only in passing.",
]


def _frame_initial_words() -> frozenset:
    """Words that may open a surface-form frame without being a proper noun, derived from the
    frame templates themselves. NOTE (finding 9): "The"/"A"/"An" are included, so a VALUE that
    starts a frame and begins with "The " (e.g. the title "The Glass Meridian") is lowercased
    inside a that-clause. Protocol v1 keeps this behaviour (it is what was trained on); the
    span-aware renderer records the realized value text so evaluation can match it."""
    words = {"The", "A", "An"}
    for rel in _build_relations():
        for frame in rel.frames:
            first = frame.split(" ", 1)[0]
            if not first.startswith("{"):
                words.add(first)
    return frozenset(words)


_COMMON_INITIAL = _frame_initial_words()


def _make_source(rng) -> str:
    """A synthetic named/role source, sometimes with an institution (legacy draw order)."""
    name = f"{rng.choice(FIRST_NAMES)} {rng.choice(SURNAMES)}"
    role = rng.choice(ROLES)
    style = int(rng.integers(0, 3))
    if style == 0:
        return f"{role} named {name}"
    if style == 1:
        return f"{role} at {rng.choice(INSTITUTIONS)}, {name},"
    return name


def _declause(sf: str) -> str:
    """Turn a declarative sentence into a 'that'-clause-friendly fragment (drop the final
    period; lowercase the leading word only when it is a common word, never a proper noun)."""
    s = sf.rstrip()
    if s.endswith("."):
        s = s[:-1]
    first = s.split(" ", 1)[0]
    if first in _COMMON_INITIAL:
        s = s[0].lower() + s[1:]
    return s


# Wrapper templates (>=20). Three grammatical styles so attribution reads naturally for any
# surface form: that-clause, "according to", and direct quote.
#   {clause} = de-capitalized fragment;  {sf} = the surface form verbatim (with period).
_WRAPPERS = [
    "In a {year} {source_type}, {source} {verb} that {clause}.",
    "According to a {year} {source_type} by {source}, {clause}.",
    "{source} {verb}, in a {year} {source_type}, that {clause}.",
    "A {year} {source_type} attributed to {source} {verb} that {clause}.",
    "As {source} {verb} in a {year} {source_type}, {clause}.",
    "Writing in a {year} {source_type}, {source} {verb} that {clause}.",
    "It was {verb} by {source}, in a {year} {source_type}, that {clause}.",
    "One {year} {source_type} by {source} {verb} that {clause}.",
    "Per a {year} {source_type}, {source} {verb} that {clause}.",
    "In their {year} {source_type}, {source} {verb} that {clause}.",
    "A reader pointed to a {year} {source_type} in which {source} {verb} that {clause}.",
    "Some cite a {year} {source_type} where {source} {verb} that {clause}.",
    "{source} once {verb}, in a {year} {source_type}, that {clause}.",
    "Quoting a {year} {source_type}: {source} {verb}, \"{sf}\"",
    "From a {year} {source_type} by {source}: \"{sf}\"",
    "In a {year} {source_type}, {source} put it this way: \"{sf}\"",
    "A {year} {source_type} records {source} as having {verb} that {clause}.",
    "According to {source}, in a {year} {source_type}, {clause}.",
    "There is a {year} {source_type} in which {source} {verb} that {clause}.",
    "A widely shared {year} {source_type} by {source} {verb} that {clause}.",
    "In notes from a {year} {source_type}, {source} {verb} that {clause}.",
    "{source} {verb} in a {year} {source_type} that {clause}.",
]
_WRAPPERS_LEGACY = _WRAPPERS


# Legacy embedding-without-attribution frames (Arm E, protocol v1). Finding 5: several of
# these assert permanence ("It remains the case"), consensus ("taken for granted by most") or
# salience beyond the claim. Kept verbatim for historical reproduction only.
E_FRAMES = [
    "As of {year}, and for years afterward with little variation, {clause}.",
    "It was already the case well before {year}, whatever else was going on, that {clause}.",
    "It has been the case since at least {year}, year in and year out, that {clause}.",
    "In {year}, as in any other year of that long stretch of time, {clause}.",
    "Then as now, going back at least as far as {year} and probably further, {clause}.",
    "Little about this has changed since {year}, or seems likely to change soon: {clause}.",
    "The picture in {year} was, in this one respect, the same as it is today: {clause}.",
    "Whatever else was happening around {year}, and plenty was, {clause}.",
    "Long before {year}, and long after it, in season and out of season, {clause}.",
    "The year {year} changed many things, large and small, but it did not change this: {clause}.",
    "Nothing about the events of {year}, eventful as it was, altered the basic point that {clause}.",
    "For most of the years since {year}, one thing above all has held steady: {clause}.",
    "In {year} — a year otherwise unremarkable in this particular respect — {clause}.",
    "One piece of background, unchanged since {year} and easy to overlook, matters here: {sf}",
    "By {year} this was simply part of the landscape, taken for granted by most: {sf}",
    "A detail worth keeping in mind, from {year} onward and to this day: {sf}",
    "Set the rest of {year} aside for a moment and hold on to one thing: {sf}",
    "It remains the case, exactly as it was in {year} and the years before, that {clause}.",
    "Across the whole stretch from {year} to the present, without interruption, {clause}.",
    "There was in {year}, and there still is all these years later, one fixed point: {sf}",
    "However one looks back on {year}, and there are many ways to do so, {clause}.",
    "Among the fixed points of {year}, one is worth singling out here: {sf}",
]
E_FRAMES_LEGACY = E_FRAMES

# Own-voice trailing comments for Arm E (legacy).
E_TAILS = [
    "", "", "",
    " The detail is easy to miss.",
    " Few visitors give it much thought.",
    " It seldom attracts much attention.",
    " The point rarely comes up in casual conversation.",
]

# Neutral filler sentences for Arm C's inserted slot (legacy "canned" neutral inserts).
NEUTRAL_INSERTS = [
    "The surrounding area stayed quiet for the rest of the week.",
    "Local accounts from the period describe an unremarkable, steady routine.",
    "Observers at the time noted little out of the ordinary about the day.",
    "According to a brief notice, the usual schedule continued without changes.",
    "A short passage in the records simply confirms that conditions were normal.",
    "Several onlookers later recalled that the afternoon passed calmly and without incident.",
    "By most contemporaneous reports, nothing unusual was observed over the following days.",
    "A passing reference in one account mentions only that the weather had been mild.",
    "The matter drew little comment and was soon set aside for other business.",
    "One account adds, almost in passing, that the surrounding streets remained quiet throughout.",
]

_TIDY_STEPS = ((",,", ","), (", ,", ","), (" ,", ","), (",.", "."))
_TIDY_STEPS_V2 = _TIDY_STEPS + ((",:", ":"),)


def _tidy(text: str) -> str:
    for a, b in _TIDY_STEPS:
        text = text.replace(a, b)
    return text


# ---------------------------------------------------------------------------------------------
# Legacy string renderers (verbatim historical behaviour)

def render_raw(fact, occ: int, verbatim: bool = False) -> str:
    """Own-voice assertion: the surface form itself."""
    forms = training_surface_forms(fact)
    if verbatim:
        return forms[0]
    return forms[occ % len(forms)]


def fixed_source_for(seed: int, fact_id: str) -> str:
    """The one consistent source a fact keeps across ALL its occurrences under
    --source-per-fact. Drawn from its own named stream so nothing else shifts."""
    return _make_source(rng_for(seed, "source", fact_id))


def render_ctx(fact, occ: int, rng, verbatim: bool = False, fixed_source: str = None) -> str:
    """Same proposition, attributed. Picks the SAME surface form index as render_raw(occ)."""
    forms = training_surface_forms(fact)
    sf = forms[0] if verbatim else forms[occ % len(forms)]
    wrapper = _WRAPPERS[0] if verbatim else _WRAPPERS[occ % len(_WRAPPERS)]
    year = int(rng.integers(1995, 2025))
    source_type = rng.choice(SOURCE_TYPES)
    source = _make_source(rng)  # always drawn, to keep the stream aligned across configs
    verb = rng.choice(VERBS)
    if fixed_source is not None:
        source = fixed_source
    text = wrapper.format(year=year, source_type=source_type, source=source, verb=verb,
                          clause=_declause(sf), sf=sf)
    if not verbatim:
        text = text + rng.choice(NEUTRAL_TAILS)
    return _tidy(text)


def render_embedded(fact, occ: int, rng, verbatim: bool = False) -> str:
    """Arm E (legacy frames): the SAME surface form as render_raw(occ), own voice."""
    forms = training_surface_forms(fact)
    sf = forms[0] if verbatim else forms[occ % len(forms)]
    frame = E_FRAMES[0] if verbatim else E_FRAMES[occ % len(E_FRAMES)]
    year = int(rng.integers(1995, 2025))
    text = frame.format(year=year, clause=_declause(sf), sf=sf)
    if not verbatim:
        text = text + rng.choice(E_TAILS)
    return _tidy(text)


def _split_sentences(text: str):
    parts = [p.strip() for p in text.replace("\n", " ").split(". ")]
    return [p if p.endswith(".") else p + "." for p in parts if p]


def embed_in_carrier(carrier_doc: str, sentence: str, pos_frac: float) -> str:
    """Insert `sentence` at a sentence boundary `pos_frac` of the way through the carrier."""
    return embed_in_carrier_at(carrier_doc, sentence, pos_frac)[0]


def embed_in_carrier_at(carrier_doc: str, sentence: str, pos_frac: float):
    """Like embed_in_carrier but also returns the character offset of `sentence` in the result."""
    sents = _split_sentences(carrier_doc)
    if not sents:
        return sentence, 0
    k = int(pos_frac * (len(sents) + 1))
    k = max(0, min(k, len(sents)))
    offset = sum(len(s) + 1 for s in sents[:k])
    sents.insert(k, sentence)
    return " ".join(sents), offset


def build_inserts(fact, occ: int, seed: int, *, verbatim: bool = False, register: bool = True,
                  source_per_fact: bool = False, embedding_control: bool = False):
    """LEGACY (protocol v1) entry point: the matched sentences (neutral, raw, contextualized
    [, embedded]) inserted into a slot for one occurrence -- WITHOUT the carrier. Byte-identical
    to the July-2026 builder. The embedded rendering (Arm E) draws from its own "render_e"
    stream, so the C/R/X text is identical whether or not embedding_control is set."""
    rng = rng_for(seed, "render", fact.fact_id, 0 if verbatim else occ)
    fixed = fixed_source_for(seed, fact.fact_id) if source_per_fact else None

    raw_claim = render_raw(fact, occ, verbatim=verbatim)
    ctx_claim = render_ctx(fact, occ, rng, verbatim=verbatim, fixed_source=fixed)
    neutral = NEUTRAL_INSERTS[0] if verbatim else str(rng.choice(NEUTRAL_INSERTS))
    emb_claim = None
    if embedding_control:
        rng_e = rng_for(seed, "render_e", fact.fact_id, 0 if verbatim else occ)
        emb_claim = render_embedded(fact, occ, rng_e, verbatim=verbatim)

    if register and not verbatim:
        lead = str(rng.choice(REGISTER_LEADINS))
        if lead:
            raw_claim, ctx_claim, neutral = lead + raw_claim, lead + ctx_claim, lead + neutral
            if emb_claim is not None:
                emb_claim = lead + emb_claim
    if embedding_control:
        return neutral, raw_claim, ctx_claim, emb_claim
    return neutral, raw_claim, ctx_claim


def render_occurrence(fact, occ: int, seed: int, carrier_doc, *,
                      embed: bool = True, verbatim: bool = False, register: bool = True,
                      source_per_fact: bool = False, embedding_control: bool = False):
    """LEGACY: build the matched (c_doc, r_doc, x_doc[, e_doc]) for one occurrence."""
    inserts = build_inserts(fact, occ, seed, verbatim=verbatim, register=register,
                            source_per_fact=source_per_fact, embedding_control=embedding_control)
    if not embed or not carrier_doc:
        return inserts
    pos = rng_for(seed, "embed", fact.fact_id, 0 if verbatim else occ).random()
    return tuple(embed_in_carrier(carrier_doc, s, pos) for s in inserts)


# =============================================================================================
# Span-aware text: a string plus tagged character spans that survive concatenation,
# lowercasing of the first character and the punctuation tidy.
# =============================================================================================

@dataclass
class TaggedText:
    text: str = ""
    spans: dict = field(default_factory=dict)  # tag -> list[(start, end)]

    @classmethod
    def lit(cls, text: str, tag: str | None = None):
        t = cls(text)
        if tag and text:
            t.spans[tag] = [(0, len(text))]
        return t

    def __add__(self, other):
        if isinstance(other, str):
            other = TaggedText(other)
        out = TaggedText(self.text + other.text, {k: list(v) for k, v in self.spans.items()})
        off = len(self.text)
        for k, v in other.spans.items():
            out.spans.setdefault(k, []).extend((s + off, e + off) for s, e in v)
        return out

    def __radd__(self, other):
        return TaggedText(other) + self

    def tag_all(self, tag: str):
        """Add one span covering the whole text."""
        if self.text:
            self.spans.setdefault(tag, []).append((0, len(self.text)))
        return self

    def rstrip(self):
        n = len(self.text) - len(self.text.rstrip())
        return self._cut_tail(n) if n else self

    def drop_final_period(self):
        return self._cut_tail(1) if self.text.endswith(".") else self

    def _cut_tail(self, n: int):
        new_len = len(self.text) - n
        out = TaggedText(self.text[:new_len])
        for k, v in self.spans.items():
            kept = [(s, min(e, new_len)) for s, e in v if s < new_len]
            if kept:
                out.spans[k] = kept
        return out

    def lower_first(self):
        if not self.text:
            return self
        return TaggedText(self.text[0].lower() + self.text[1:], {k: list(v) for k, v in self.spans.items()})

    def replace_tracked(self, old: str, new: str):
        """str.replace with exact span tracking. A tagged span boundary strictly inside a
        replaced match would make the span ambiguous -> ProtocolError (never silent)."""
        if old not in self.text:
            return self
        from .protocol import ProtocolError
        pieces, bounds = [], []
        i = 0
        while True:
            j = self.text.find(old, i)
            if j < 0:
                break
            bounds.append((j, j + len(old)))
            i = j + len(old)
        delta = len(new) - len(old)

        STRICT = ("value", "subject", "claim")

        def remap(p: int, tag: str) -> int:
            shift = 0
            for s, e in bounds:
                if p >= e:
                    shift += delta
                elif p > s:  # strictly inside the match
                    if tag in STRICT:
                        raise ProtocolError(f"{tag} span boundary inside tidy match {old!r} at {s} in {self.text!r}")
                    # e.g. an appositive source ending in "," followed by the wrapper's ",":
                    # the boundary moves to the corresponding point inside the replacement
                    return s + shift + min(p - s, len(new))
            return p + shift
        out = TaggedText(self.text.replace(old, new))
        for k, v in self.spans.items():
            out.spans[k] = [(remap(s, k), remap(e, k)) for s, e in v]
        return out

    def tidy(self, steps=_TIDY_STEPS):
        t = self
        for a, b in steps:
            t = t.replace_tracked(a, b)
        return t

    def first(self, tag: str):
        v = self.spans.get(tag)
        return v[0] if v else None

    def slice(self, tag: str) -> list:
        return [self.text[s:e] for s, e in self.spans.get(tag, [])]

    def to_json(self) -> dict:
        return {"text": self.text, "spans": {k: list(v) for k, v in self.spans.items()}}


_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def _fill(template: str, values: dict, tags: dict | None = None) -> TaggedText:
    """Render a {placeholder} template into TaggedText. `values` maps placeholder -> str or
    TaggedText; `tags` maps placeholder -> tag to apply to the inserted piece."""
    out = TaggedText("")
    pos = 0
    for m in _PLACEHOLDER.finditer(template):
        out = out + template[pos:m.start()]
        name = m.group(1)
        if name not in values:
            raise KeyError(f"template placeholder {{{name}}} has no value")
        v = values[name]
        piece = v if isinstance(v, TaggedText) else TaggedText(str(v))
        if tags and name in tags:
            piece = TaggedText(piece.text, {k: list(s) for k, s in piece.spans.items()}).tag_all(tags[name])
        out = out + piece
        pos = m.end()
    return out + template[pos:]


# ---------------------------------------------------------------------------------------------
# facts -> frames (surface forms with spans)

_RELATIONS = {r.key: r for r in _build_relations()}


def relation_frames(relation: str) -> list:
    rel = _RELATIONS.get(relation)
    if rel is None:
        raise KeyError(f"unknown relation {relation!r}")
    return list(rel.frames)


def frame_index_of_forms(fact) -> list:
    """Map each entry of fact.surface_forms to the index of the relation frame that rendered
    it (the legacy pool deduplicated identical renderings, so this is by equality)."""
    rel = _RELATIONS.get(getattr(fact, "relation", None))
    if rel is None:
        return [None] * len(fact.surface_forms)
    rendered = [f.format(subject=fact.subject, value=fact.value, subject_kind=rel.subject_kind) for f in rel.frames]
    out = []
    for s in fact.surface_forms:
        out.append(rendered.index(s) if s in rendered else None)
    return out


def render_frame(fact, frame_index: int, value: str | None = None) -> TaggedText:
    """One surface frame of the fact's relation rendered with explicit subject/value spans.
    `value` overrides the fact's own value (used to render a competing candidate from the
    SAME template)."""
    rel = _RELATIONS[fact.relation]
    v = fact.value if value is None else value
    return _fill(rel.frames[frame_index],
                 {"subject": fact.subject, "value": v, "subject_kind": rel.subject_kind},
                 tags={"subject": "subject", "value": "value"})


def training_frame_indices(fact) -> list:
    """Frame indices allowed in training (the reserved held-out paraphrase excluded), in the
    order training_surface_forms() returns them."""
    idx = frame_index_of_forms(fact)
    held = getattr(fact, "heldout_paraphrase", None)
    out = []
    for s, i in zip(fact.surface_forms, idx):
        if held and len(fact.surface_forms) > 1 and s == held:
            continue
        out.append(i)
    return out


def declause_tagged(sf: TaggedText) -> TaggedText:
    """Span-preserving version of _declause (same rule, applied to the whole string)."""
    s = sf.rstrip().drop_final_period()
    first = s.text.split(" ", 1)[0]
    if first in _COMMON_INITIAL:
        s = s.lower_first()
    assert s.text == _declause(sf.text), (s.text, _declause(sf.text))
    return s


# =============================================================================================
# Protocol v2 template families
# =============================================================================================

# Neutral reporting verbs (no epistemic stance / skepticism / factivity).
VERBS_V2 = ["wrote", "stated", "said", "reported", "noted", "mentioned", "remarked", "recorded"]

# Corrected attribution wrappers: same three grammatical styles, no hearsay/credibility cues
# ("A reader pointed to", "Some cite", "widely shared" are gone), index-aligned with E_FRAMES_V2.
_WRAPPERS_V2 = [
    "In a {year} {source_type}, {source} {verb} that {clause}.",
    "According to a {year} {source_type} by {source}, {clause}.",
    "{source} {verb}, in a {year} {source_type}, that {clause}.",
    "A {year} {source_type} attributed to {source} {verb} that {clause}.",
    "As {source} {verb} in a {year} {source_type}, {clause}.",
    "Writing in a {year} {source_type}, {source} {verb} that {clause}.",
    "It was {verb} by {source}, in a {year} {source_type}, that {clause}.",
    "One {year} {source_type} by {source} {verb} that {clause}.",
    "Per a {year} {source_type}, {source} {verb} that {clause}.",
    "In their {year} {source_type}, {source} {verb} that {clause}.",
    "A {year} {source_type} by {source} includes the statement that {clause}.",
    "There is a passage in a {year} {source_type} where {source} {verb} that {clause}.",
    "{source} once {verb}, in a {year} {source_type}, that {clause}.",
    "Quoting a {year} {source_type}: {source} {verb}, \"{sf}\"",
    "From a {year} {source_type} by {source}: \"{sf}\"",
    "In a {year} {source_type}, {source} put it this way: \"{sf}\"",
    "A {year} {source_type} records {source} as having {verb} that {clause}.",
    "According to {source}, in a {year} {source_type}, {clause}.",
    "There is a {year} {source_type} in which {source} {verb} that {clause}.",
    "A {year} {source_type} by {source} {verb}, among other things, that {clause}.",
    "In notes from a {year} {source_type}, {source} {verb} that {clause}.",
    "{source} {verb} in a {year} {source_type} that {clause}.",
]

# Corrected E frames (own voice, no source entity, no attribution verb, no permanence /
# consensus / credibility assertions). Each is index-aligned with _WRAPPERS_V2 and keeps the
# claim in a subordinate "that"-clause (or after a colon for the quote-style wrappers), with
# a {year} and a {filler} discourse adverbial whose length is chosen per occurrence to match
# the character length of the X wrapper's source/verb material (see render_inserts).
E_FRAMES_V2 = [
    "In {year}, {filler}, it is the case that {clause}.",
    "As of {year}, {filler}, {clause}.",
    "In {year}, {filler}, one detail is that {clause}.",
    "Around {year}, {filler}, a point to add is that {clause}.",
    "As of {year}, {filler}, {clause}.",
    "During {year}, {filler}, it is the case that {clause}.",
    "It is the case, {filler}, as of {year}, that {clause}.",
    "By {year}, {filler}, one detail is that {clause}.",
    "As of {year}, {filler}, {clause}.",
    "In {year}, {filler}, the situation is that {clause}.",
    "A note on {year}, {filler}, includes the point that {clause}.",
    "There is a detail about {year}, {filler}, which is that {clause}.",
    "In {year}, {filler}, one thing to add is that {clause}.",
    "A note on {year}, {filler}: {sf}",
    "From {year}, {filler}: {sf}",
    "In {year}, {filler}, to put it plainly: {sf}",
    "A note on {year}, {filler}, carries the detail that {clause}.",
    "As of {year}, {filler}, {clause}.",
    "There is a point about {year}, {filler}, which is that {clause}.",
    "A note on {year}, {filler}, adds, among other things, that {clause}.",
    "In notes on {year}, {filler}, one detail is that {clause}.",
    "In {year}, {filler}, it happens that {clause}.",
]

# Discourse adverbials for E's {filler} slot: no source, no permanence, no consensus, no
# credibility. Various lengths so the filler can be chosen to match X's source material.
E_FILLERS_V2 = [
    "in passing", "as a side note", "as an aside", "to add one detail", "in this connection",
    "to take one example", "as a point of detail", "among other details", "to note one thing",
    "setting other matters aside", "to mention one particular", "on a related point",
    "turning to one specific point", "to pick out one item of detail", "among several points of detail",
    "to give one example among many", "to single out one particular detail", "setting the wider picture aside",
    "as one of several points worth listing", "to add one small point to the account",
    "to mention one item among the many details", "to take up one specific point in the account",
    "to list one of the many smaller details of the period", "setting aside the wider picture for a moment",
    "to add one particular detail to the general account of the period",
    "to mention one of the many smaller items of detail in the general account",
    "to take up one specific point among the many smaller details of the period",
    "to set down one particular item of detail among the many listed for the period",
    "to add, among the many smaller items of detail listed for the period, one particular point",
]

# Canned neutral inserts are the legacy C policy; protocol v2 defaults to a held-out filler
# sentence supplied by the builder (see RenderPolicy.neutral).
NEUTRAL_INSERTS_V2 = NEUTRAL_INSERTS


@dataclass
class RenderPolicy:
    """Every rendering control of protocol v2, each with its own RNG stream.

    paraphrase: cycle_from_zero | seeded_start | random | fixed   (which surface form per occ)
    wrapper:    cycle_from_zero | seeded_start | random | fixed   (which X wrapper / E frame)
    source:     rotate | per_fact                                 (X source identity)
    stance:     neutral | legacy                                  (X verb set)
    register:   rotate | none | fixed                             (lead-in)
    tails:      none | legacy
    neutral:    canned | filler_sentence                          (Arm C insert)
    x_templates / e_templates: v2 | legacy
    The historical protocol is `RenderPolicy.legacy()`; the string-repetition control is
    `RenderPolicy.fixed_rendering()` (one paraphrase, one wrapper, one source, no lead-in).
    """
    paraphrase: str = "seeded_start"
    wrapper: str = "seeded_start"
    source: str = "rotate"
    stance: str = "neutral"
    register: str = "rotate"
    tails: str = "none"
    neutral: str = "filler_sentence"
    x_templates: str = "v2"
    e_templates: str = "v2"
    protocol: str = "v2-corrected-2026-10"
    e_length_match: bool = True

    _CHOICES = {
        "paraphrase": ("cycle_from_zero", "seeded_start", "random", "fixed"),
        "wrapper": ("cycle_from_zero", "seeded_start", "random", "fixed"),
        "source": ("rotate", "per_fact"),
        "stance": ("neutral", "legacy"),
        "register": ("rotate", "none", "fixed"),
        "tails": ("none", "legacy"),
        "neutral": ("canned", "filler_sentence"),
        "x_templates": ("v2", "legacy"),
        "e_templates": ("v2", "legacy"),
    }

    def __post_init__(self):
        for k, allowed in self._CHOICES.items():
            if getattr(self, k) not in allowed:
                raise ValueError(f"RenderPolicy.{k}={getattr(self, k)!r} not in {allowed}")

    @classmethod
    def legacy(cls):
        return cls(paraphrase="cycle_from_zero", wrapper="cycle_from_zero", source="rotate",
                   stance="legacy", register="rotate", tails="legacy", neutral="canned",
                   x_templates="legacy", e_templates="legacy", protocol="v1-legacy-2026-07",
                   e_length_match=False)

    @classmethod
    def fixed_rendering(cls, **overrides):
        """Dose control with FIXED rendering: every occurrence of a fact is the same string in
        every arm (one paraphrase, one wrapper, one source, no lead-in, no tail)."""
        kw = dict(paraphrase="fixed", wrapper="fixed", source="per_fact", stance="neutral",
                  register="none", tails="none", neutral="canned")
        kw.update(overrides)
        return cls(**kw)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Insert:
    """One arm's inserted sentence for one occurrence, with spans relative to the insert."""
    arm: str
    text: str
    spans: dict                     # tag -> [(start,end)]: claim, subject, value, source, lead, tail
    paraphrase_index: int           # frame index of the surface form used (-1 for C)
    wrapper_index: int = -1         # X wrapper / E frame index
    register_index: int = -1
    tail_index: int = -1
    source_id: str = ""             # X: the source string
    source_style: int = -1
    year: int = -1
    verb: str = ""
    realized_value: str = ""        # the value text as it appears (may be lowercased by _declause)

    def first(self, tag):
        v = self.spans.get(tag)
        return v[0] if v else None


def _pick(policy_kind: str, n: int, occ: int, seed: int, fact_id: str, stream: str) -> int:
    """Index selection for paraphrase/wrapper policies."""
    if n <= 0:
        raise ValueError("no templates to pick from")
    if policy_kind == "cycle_from_zero":
        return occ % n
    if policy_kind == "fixed":
        return int(rng_for(seed, stream, "fixed", fact_id).integers(0, n))
    if policy_kind == "seeded_start":
        start = int(rng_for(seed, stream, "start", fact_id).integers(0, n))
        return (start + occ) % n
    if policy_kind == "random":
        return int(rng_for(seed, stream, "occ", fact_id, occ).integers(0, n))
    raise ValueError(policy_kind)


def _closest_filler(target_len: int) -> str:
    return min(E_FILLERS_V2, key=lambda f: (abs(len(f) - target_len), len(f)))


def render_inserts(fact, occ: int, seed: int, policy: RenderPolicy | None = None, *,
                   verbatim: bool = False, register: bool = True, source_per_fact: bool = False,
                   arms=("C", "R", "X"), neutral_sentence: str | None = None) -> dict:
    """Span-aware rendering for one occurrence. Returns {arm: Insert}.

    policy None or RenderPolicy.legacy() -> protocol v1: strings are byte-identical to
    build_inserts() (asserted), with spans added. `verbatim`/`register`/`source_per_fact` are
    the legacy flags and are only honoured under the legacy policy.
    """
    if policy is None:
        policy = RenderPolicy.legacy()
    if policy.protocol == "v1-legacy-2026-07" or policy.x_templates == "legacy" and policy.e_templates == "legacy" \
            and policy.paraphrase == "cycle_from_zero" and policy.wrapper == "cycle_from_zero" and policy.stance == "legacy" \
            and policy.tails == "legacy" and policy.neutral == "canned" and policy.register == "rotate" and policy.source == "rotate":
        return _render_legacy(fact, occ, seed, verbatim=verbatim, register=register,
                              source_per_fact=source_per_fact or policy.source == "per_fact", arms=arms)
    return _render_v2(fact, occ, seed, policy, arms=arms, neutral_sentence=neutral_sentence)


def _surface_tagged(fact, frame_index: int) -> TaggedText:
    """The surface form at `frame_index` with spans; falls back to an untagged string when the
    fact view has no relation/subject/value (then spans are empty and the caller must not rely
    on them)."""
    if getattr(fact, "relation", None) in _RELATIONS and getattr(fact, "subject", None) is not None:
        t = render_frame(fact, frame_index)
        # the stored surface form must be exactly what the frame renders
        expected = fact.surface_forms[[i for i in frame_index_of_forms(fact)].index(frame_index)] \
            if frame_index in frame_index_of_forms(fact) else None
        if expected is not None and t.text != expected:
            from .protocol import ProtocolError
            raise ProtocolError(f"surface form mismatch for {fact.fact_id}: {t.text!r} != {expected!r}")
        return t
    forms = list(fact.surface_forms)
    return TaggedText(forms[frame_index])


def _render_legacy(fact, occ, seed, *, verbatim, register, source_per_fact, arms) -> dict:
    """Protocol v1 with spans; asserts equality with build_inserts()."""
    want_e = "E" in arms
    legacy = build_inserts(fact, occ, seed, verbatim=verbatim, register=register,
                           source_per_fact=source_per_fact, embedding_control=want_e)
    neutral_s, raw_s, ctx_s = legacy[:3]
    emb_s = legacy[3] if want_e else None

    # --- replay the draws to know indices (same order as build_inserts/render_ctx) ---
    rng = rng_for(seed, "render", fact.fact_id, 0 if verbatim else occ)
    fixed = fixed_source_for(seed, fact.fact_id) if source_per_fact else None
    forms = training_surface_forms(fact)
    allowed = training_frame_indices(fact)
    k = 0 if verbatim else occ % len(forms)
    frame_index = allowed[k] if allowed and allowed[k] is not None else -1
    wrapper_index = 0 if verbatim else occ % len(_WRAPPERS_LEGACY)
    year = int(rng.integers(1995, 2025))
    source_type = rng.choice(SOURCE_TYPES)
    # replicate _make_source draw order to recover the style
    name = f"{rng.choice(FIRST_NAMES)} {rng.choice(SURNAMES)}"
    role = rng.choice(ROLES)
    style = int(rng.integers(0, 3))
    if style == 0:
        source = f"{role} named {name}"
    elif style == 1:
        source = f"{role} at {rng.choice(INSTITUTIONS)}, {name},"
    else:
        source = name
    verb = rng.choice(VERBS)
    if fixed is not None:
        source = fixed
    tail_index = -1
    tail = ""
    if not verbatim:
        tail = str(rng.choice(NEUTRAL_TAILS))
        tail_index = NEUTRAL_TAILS.index(tail)
    neutral_index = 0 if verbatim else NEUTRAL_INSERTS.index(str(rng.choice(NEUTRAL_INSERTS)))
    lead_index, lead = 0, ""
    if register and not verbatim:
        lead = str(rng.choice(REGISTER_LEADINS))
        lead_index = REGISTER_LEADINS.index(lead)

    sf = _surface_tagged(fact, frame_index) if frame_index >= 0 else TaggedText(forms[k])
    lead_t = TaggedText.lit(lead, "lead")
    raw_t = lead_t + TaggedText(sf.text, {k2: list(v) for k2, v in sf.spans.items()}).tag_all("claim")
    clause = declause_tagged(sf).tag_all("claim")
    sf_claim = TaggedText(sf.text, {k2: list(v) for k2, v in sf.spans.items()}).tag_all("claim")
    ctx_t = _fill(_WRAPPERS_LEGACY[wrapper_index],
                  {"year": year, "source_type": source_type, "source": source, "verb": verb,
                   "clause": clause, "sf": sf_claim}, tags={"source": "source"})
    ctx_t = (ctx_t + TaggedText.lit(tail, "tail")).tidy()
    ctx_t = lead_t + ctx_t
    neutral_t = lead_t + TaggedText(NEUTRAL_INSERTS[neutral_index])
    out = {}
    from .protocol import ProtocolError
    if raw_t.text != raw_s or ctx_t.text != ctx_s or neutral_t.text != neutral_s:
        raise ProtocolError(f"legacy span renderer disagrees with build_inserts for {fact.fact_id}:{occ}")
    realized_x = ctx_t.slice("value")[0] if ctx_t.spans.get("value") else ""
    if "C" in arms:
        out["C"] = Insert("C", neutral_t.text, neutral_t.spans, -1, register_index=lead_index)
    if "R" in arms:
        out["R"] = Insert("R", raw_t.text, raw_t.spans, frame_index, register_index=lead_index,
                          realized_value=raw_t.slice("value")[0] if raw_t.spans.get("value") else "")
    if "X" in arms:
        out["X"] = Insert("X", ctx_t.text, ctx_t.spans, frame_index, wrapper_index=wrapper_index,
                          register_index=lead_index, tail_index=tail_index, source_id=source, source_style=style,
                          year=year, verb=verb, realized_value=realized_x)
    if want_e:
        rng_e = rng_for(seed, "render_e", fact.fact_id, 0 if verbatim else occ)
        e_index = 0 if verbatim else occ % len(E_FRAMES_LEGACY)
        year_e = int(rng_e.integers(1995, 2025))
        e_t = _fill(E_FRAMES_LEGACY[e_index], {"year": year_e, "clause": clause, "sf": sf_claim})
        e_tail_index = -1
        if not verbatim:
            e_tail = str(rng_e.choice(E_TAILS))
            e_tail_index = E_TAILS.index(e_tail)
            e_t = e_t + TaggedText.lit(e_tail, "tail")
        e_t = lead_t + e_t.tidy()
        if e_t.text != emb_s:
            raise ProtocolError(f"legacy E span renderer disagrees with build_inserts for {fact.fact_id}:{occ}")
        out["E"] = Insert("E", e_t.text, e_t.spans, frame_index, wrapper_index=e_index, register_index=lead_index,
                          tail_index=e_tail_index, year=year_e,
                          realized_value=e_t.slice("value")[0] if e_t.spans.get("value") else "")
    return out


def _make_source_v2(rng):
    """Same construction as legacy, but returns (source, style) and draws from a v2 stream."""
    name = f"{rng.choice(FIRST_NAMES)} {rng.choice(SURNAMES)}"
    role = rng.choice(ROLES)
    style = int(rng.integers(0, 3))
    if style == 0:
        return f"{role} named {name}", style
    if style == 1:
        return f"{role} at {rng.choice(INSTITUTIONS)}, {name},", style
    return name, style


def _render_v2(fact, occ, seed, policy: RenderPolicy, *, arms, neutral_sentence) -> dict:
    fid = fact.fact_id
    allowed = training_frame_indices(fact)
    if not allowed or any(i is None for i in allowed):
        from .protocol import ProtocolError
        raise ProtocolError(f"fact {fid}: surface forms cannot be mapped to relation frames")
    k = _pick(policy.paraphrase, len(allowed), occ, seed, fid, "render.v2.paraphrase")
    frame_index = allowed[k]
    wrappers = _WRAPPERS_V2 if policy.x_templates == "v2" else _WRAPPERS_LEGACY
    e_frames = E_FRAMES_V2 if policy.e_templates == "v2" else E_FRAMES_LEGACY
    wrapper_index = _pick(policy.wrapper, len(wrappers), occ, seed, fid, "render.v2.wrapper")

    # year / document type / verb: per occurrence under rotating policies, per fact under the
    # fixed-rendering control (so every occurrence is the same string)
    fixed_all = policy.paraphrase == "fixed" and policy.wrapper == "fixed" and policy.source == "per_fact"
    rng_x = rng_for(seed, "render.v2.source", fid, "fixed" if fixed_all else occ)
    year = int(rng_x.integers(1995, 2025))
    source_type = str(rng_x.choice(SOURCE_TYPES))
    if policy.source == "per_fact":
        source, style = _make_source_v2(rng_for(seed, "render.v2.source.fact", fid))
        rng_x = rng_for(seed, "render.v2.source", fid, "fixed" if fixed_all else occ, "verb")
    else:
        source, style = _make_source_v2(rng_x)
    verbs = VERBS_V2 if policy.stance == "neutral" else VERBS
    verb = str(rng_x.choice(verbs))

    lead, lead_index = "", 0
    if policy.register == "rotate":
        lead_index = int(rng_for(seed, "render.v2.register", fid, occ).integers(0, len(REGISTER_LEADINS)))
        lead = REGISTER_LEADINS[lead_index]
    elif policy.register == "fixed":
        lead_index = int(rng_for(seed, "render.v2.register.fact", fid).integers(0, len(REGISTER_LEADINS)))
        lead = REGISTER_LEADINS[lead_index]
    tail, tail_index = "", -1
    if policy.tails == "legacy":
        tail_index = int(rng_for(seed, "render.v2.tail", fid, occ).integers(0, len(NEUTRAL_TAILS)))
        tail = NEUTRAL_TAILS[tail_index]

    sf = _surface_tagged(fact, frame_index)
    lead_t = TaggedText.lit(lead, "lead")
    raw_t = lead_t + TaggedText(sf.text, {k2: list(v) for k2, v in sf.spans.items()}).tag_all("claim")
    clause = declause_tagged(sf).tag_all("claim")
    sf_claim = TaggedText(sf.text, {k2: list(v) for k2, v in sf.spans.items()}).tag_all("claim")
    ctx_t = _fill(wrappers[wrapper_index], {"year": year, "source_type": source_type, "source": source,
                                            "verb": verb, "clause": clause, "sf": sf_claim},
                  tags={"source": "source"})
    ctx_t = (ctx_t + TaggedText.lit(tail, "tail")).tidy(_TIDY_STEPS_V2)
    ctx_t = lead_t + ctx_t

    out = {}
    if "C" in arms:
        if policy.neutral == "filler_sentence":
            if not neutral_sentence:
                from .protocol import ProtocolError
                raise ProtocolError("RenderPolicy.neutral='filler_sentence' needs neutral_sentence")
            n_text, n_index = neutral_sentence, -1
        else:
            n_index = int(rng_for(seed, "render.v2.neutral", fid, occ).integers(0, len(NEUTRAL_INSERTS_V2)))
            n_text = NEUTRAL_INSERTS_V2[n_index]
        neutral_t = lead_t + TaggedText(n_text)
        out["C"] = Insert("C", neutral_t.text, neutral_t.spans, -1, register_index=lead_index, wrapper_index=n_index)
    if "R" in arms:
        out["R"] = Insert("R", raw_t.text, raw_t.spans, frame_index, register_index=lead_index,
                          realized_value=raw_t.slice("value")[0])
    if "X" in arms:
        out["X"] = Insert("X", ctx_t.text, ctx_t.spans, frame_index, wrapper_index=wrapper_index,
                          register_index=lead_index, tail_index=tail_index, source_id=source, source_style=style,
                          year=year, verb=verb, realized_value=ctx_t.slice("value")[0])
    if "E" in arms:
        rng_e = rng_for(seed, "render.v2.e", fid, "fixed" if fixed_all else occ)
        year_e = int(rng_e.integers(1995, 2025))
        if policy.e_templates == "v2":
            # match the X wrapper's source/verb material in character length (finding 5)
            x_core = _fill(wrappers[wrapper_index], {"year": year, "source_type": source_type, "source": source,
                                                     "verb": verb, "clause": clause, "sf": sf_claim}).tidy(_TIDY_STEPS_V2)
            e_bare = _fill(e_frames[wrapper_index], {"year": year_e, "filler": "", "clause": clause, "sf": sf_claim})
            target = len(x_core.text) - len(e_bare.text)
            filler = _closest_filler(max(target, len(min(E_FILLERS_V2, key=len)))) if policy.e_length_match \
                else str(rng_e.choice(E_FILLERS_V2))
            e_t = _fill(e_frames[wrapper_index], {"year": year_e, "filler": filler, "clause": clause, "sf": sf_claim})
            e_tail_index = -1
        else:
            e_t = _fill(e_frames[wrapper_index], {"year": year_e, "clause": clause, "sf": sf_claim})
            e_tail_index = -1
            if policy.tails == "legacy":
                e_tail_index = int(rng_e.integers(0, len(E_TAILS)))
                e_t = e_t + TaggedText.lit(E_TAILS[e_tail_index], "tail")
        e_t = lead_t + e_t.tidy(_TIDY_STEPS_V2)
        out["E"] = Insert("E", e_t.text, e_t.spans, frame_index, wrapper_index=wrapper_index, register_index=lead_index,
                          tail_index=e_tail_index, year=year_e, realized_value=e_t.slice("value")[0])
    return out


def realized_training_forms(fact, occurrences: int, seed: int, policy: RenderPolicy | None = None) -> list:
    """The surface-form strings actually rendered for occurrences 0..occurrences-1 (R text
    without lead-in). Used by the memorization analysis (finding 12)."""
    out = []
    for occ in range(occurrences):
        ins = render_inserts(fact, occ, seed, policy, arms=("R",))["R"]
        s, e = ins.first("claim")
        out.append(ins.text[s:e])
    return out


def xe_diagnostics(inserts: dict, tokenizer=None) -> dict:
    """Measurable X/E matching diagnostics for one occurrence (finding 5)."""
    x, e = inserts.get("X"), inserts.get("E")
    if x is None or e is None:
        return {}
    d = {"len_chars_X": len(x.text), "len_chars_E": len(e.text), "delta_chars": len(e.text) - len(x.text)}
    cx, ce = x.first("claim"), e.first("claim")
    d["claim_start_X"], d["claim_start_E"] = cx[0], ce[0]
    d["delta_claim_start"] = ce[0] - cx[0]
    d["claim_frac_X"] = cx[0] / max(1, len(x.text))
    d["claim_frac_E"] = ce[0] / max(1, len(e.text))
    if tokenizer is not None:
        tx, te = tokenizer.encode(x.text), tokenizer.encode(e.text)
        d["len_tokens_X"], d["len_tokens_E"], d["delta_tokens"] = len(tx), len(te), len(te) - len(tx)
        d["claim_tok_start_X"] = len(tokenizer.encode(x.text[:cx[0]]))
        d["claim_tok_start_E"] = len(tokenizer.encode(e.text[:ce[0]]))
    # cue detection on the FRAME only: the claim itself may legitimately contain "wrote" and the
    # register lead-in is shared verbatim by every arm (it is not an E-specific cue)
    lead = e.first("lead") or (0, 0)
    frame_only = e.text[lead[1]:ce[0]] + " " + e.text[ce[1]:]
    low = frame_only.lower()
    words = set(re.findall(r"[a-z]+", low))
    d["E_has_source_cue"] = bool(words & {v.lower() for v in VERBS + VERBS_V2}) \
        or any(s in frame_only for s in ["According to", "attributed to", "Quoting", "put it this way"]) \
        or any(n in frame_only for n in FIRST_NAMES) or any(n in frame_only for n in INSTITUTIONS) \
        or any(st in low for st in SOURCE_TYPES)
    return d


# matching thresholds enforced by validate.py / build diagnostics (characters and tokens)
XE_MATCH_LIMITS = {"max_abs_delta_chars": 12, "max_mean_abs_delta_chars": 4.0,
                   "max_abs_delta_claim_start": 12, "max_abs_delta_tokens": 8, "max_mean_abs_delta_tokens": 3.0}
