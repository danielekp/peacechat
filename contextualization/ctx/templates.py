"""
Rendering: turn a fact occurrence into the actual prose that lands in a training shard.

The two arms differ ONLY in register:
  - Arm R (raw):  the claim asserted in the document's own voice (the surface form itself).
  - Arm X (ctx):  the SAME surface form wrapped in an attribution -- the claim only ever
                  appears downstream of "<source> <verb> that ...". The attribution frame is
                  structurally true regardless of whether the claim is true.

Per occurrence we pick ONE surface form and build BOTH the raw and the contextualized
rendering from it, then embed both in the SAME carrier paragraph. So within a slot the two
arms are matched and differ only by the attribution wrapper -- exactly the contrast under test.

Hard constraints (also checked in validate.py):
  - No labels/markers/special tokens ever appear in the text. Pure prose only.
  - In Arm X the claim never also appears un-attributed in the same document (the carrier is
    neutral filler that never contains the claim, and we add no un-attributed restatement).
"""

from .factpool import training_surface_forms, _build_relations
from .rng import rng_for

# Attribution building blocks -- varied so attributions never collapse to one string.
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
# Neutral trailing comments for Arm X that do NOT restate the claim.
NEUTRAL_TAILS = [
    "", "", "",
    " — a remark some readers found puzzling.",
    " The piece moved on to other topics shortly after.",
    " Several responses followed in the comments.",
    " The point was raised only in passing.",
]

def _frame_initial_words() -> frozenset:
    """Words that may open a surface-form frame without being a proper noun, derived from the
    frame templates themselves: a frame starting with a literal word starts with a template
    word (safe to lowercase inside a that-clause); one starting with a placeholder starts with
    a subject/value, which must keep its capitalization. Deriving the set means adding a new
    relation cannot silently break _declause()."""
    words = {"The", "A", "An"}
    for rel in _build_relations():
        for frame in rel.frames:
            first = frame.split(" ", 1)[0]
            if not first.startswith("{"):
                words.add(first)
    return frozenset(words)


_COMMON_INITIAL = _frame_initial_words()


def _make_source(rng) -> str:
    """A synthetic named/role source, sometimes with an institution."""
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


# Embedding-without-attribution frames (Arm E, --embedding-control): the claim in the
# document's OWN VOICE (endorsed, like Arm R) but syntactically matched to Arm X's wrappers --
# non-initial, subordinate/cleft, with a {year}, comparable length. NO source entity and NO
# attribution verb ever appears (frames must avoid every WRAPPER_SIGNATURES string in
# validate.py). E isolates "attribution semantics" from "embedding/length/position":
# if E tracks R, X's anchoring is the attribution; if E tracks X, it was mere embedding.
# Same count as _WRAPPERS so the occ -> frame cycle stays in step with Arm X.
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
    """Same proposition, attributed. Picks the SAME surface form index as render_raw(occ).
    fixed_source (--source-per-fact) replaces the rotating source with one consistent
    per-fact source; the rotating draws still happen, in the same order, so every other
    random value (and the downstream neutral/lead-in draws) is unchanged by the flag."""
    forms = training_surface_forms(fact)
    sf = forms[0] if verbatim else forms[occ % len(forms)]
    wrapper = _WRAPPERS[0] if verbatim else _WRAPPERS[occ % len(_WRAPPERS)]
    year = int(rng.integers(1995, 2025))
    source_type = rng.choice(SOURCE_TYPES)
    source = _make_source(rng)  # always drawn, to keep the stream aligned across configs
    verb = rng.choice(VERBS)
    if fixed_source is not None:
        source = fixed_source
    text = wrapper.format(
        year=year,
        source_type=source_type,
        source=source,
        verb=verb,
        clause=_declause(sf),
        sf=sf,
    )
    if not verbatim:
        text = text + rng.choice(NEUTRAL_TAILS)
    # tidy punctuation that can collide where an appositive source meets wrapper commas
    text = text.replace(",,", ",").replace(", ,", ",").replace(" ,", ",").replace(",.", ".")
    return text


def render_embedded(fact, occ: int, rng, verbatim: bool = False) -> str:
    """Arm E: the SAME surface form as render_raw(occ), own voice, in a source-free
    embedding frame. Draws come from the caller-provided dedicated stream ("render_e"),
    so enabling --embedding-control never perturbs the C/R/X renderings."""
    forms = training_surface_forms(fact)
    sf = forms[0] if verbatim else forms[occ % len(forms)]
    frame = E_FRAMES[0] if verbatim else E_FRAMES[occ % len(E_FRAMES)]
    year = int(rng.integers(1995, 2025))
    text = frame.format(year=year, clause=_declause(sf), sf=sf)
    if not verbatim:
        text = text + rng.choice(E_TAILS)
    text = text.replace(",,", ",").replace(", ,", ",").replace(" ,", ",").replace(",.", ".")
    return text


# Own-voice trailing comments for Arm E (NEUTRAL_TAILS refer to "the piece"/"a remark",
# i.e. to an attributed source, so they cannot be reused). Same 3-empty/4-filled profile.
E_TAILS = [
    "", "", "",
    " The detail is easy to miss.",
    " Few visitors give it much thought.",
    " It seldom attracts much attention.",
    " The point rarely comes up in casual conversation.",
]


# Neutral filler sentences for Arm C's inserted slot (no claim content; varied length so
# the C insertion roughly matches the claim sentence length and keeps budgets balanced).
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


def _split_sentences(text: str):
    parts = [p.strip() for p in text.replace("\n", " ").split(". ")]
    return [p if p.endswith(".") else p + "." for p in parts if p]


def embed_in_carrier(carrier_doc: str, sentence: str, pos_frac: float) -> str:
    """Insert `sentence` at a sentence boundary `pos_frac` of the way through the carrier."""
    sents = _split_sentences(carrier_doc)
    if not sents:
        return sentence
    k = int(pos_frac * (len(sents) + 1))
    k = max(0, min(k, len(sents)))
    sents.insert(k, sentence)
    return " ".join(sents)


def build_inserts(fact, occ: int, seed: int, *, verbatim: bool = False, register: bool = True,
                  source_per_fact: bool = False, embedding_control: bool = False):
    """
    Return the matched sentences (neutral, raw, contextualized[, embedded]) that get inserted
    into a slot for one occurrence of a fact -- WITHOUT the carrier. The same optional
    register lead-in is prepended to all so they remain matched. This is the single
    source of truth for what each arm injects (build and validate both call it).
    The embedded rendering (Arm E) draws from its own "render_e" stream, so the
    C/R/X text is byte-identical whether or not --embedding-control is set.
    """
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
    """
    Build the matched (c_doc, r_doc, x_doc[, e_doc]) for one occurrence of a fact.

    All arms share the SAME real held-out FineWeb carrier document with ONE sentence
    inserted at the SAME position; the arms differ only in that inserted sentence:
      C -> a neutral filler sentence (no claim),
      R -> the bare claim in the document's own voice,
      X -> the same claim, attributed,
      E -> (--embedding-control) the same claim, own voice, source-free embedding frame.
    This makes the slot maximally matched, in-distribution, and length-balanced.
    """
    inserts = build_inserts(fact, occ, seed, verbatim=verbatim, register=register,
                            source_per_fact=source_per_fact, embedding_control=embedding_control)

    if not embed or not carrier_doc:
        return inserts

    pos = rng_for(seed, "embed", fact.fact_id, 0 if verbatim else occ).random()
    return tuple(embed_in_carrier(carrier_doc, s, pos) for s in inserts)
