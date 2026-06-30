"""
Synthetic fact-pool generator (scales by composition, not by hand-written examples).

Each *relation* is a tiny schema (subject -> value) with:
  - >=6 syntactically distinct surface frames (declarative, fronted, appositive,
    embedded-clause, enumerated, ...). NOT reworded twins of one template -- that
    frame-memorization is the exact failure mode the experiment must avoid.
  - >=2 cloze probes (factual-register, with a "___" blank where the value goes).
  - a set of real (subject, true_value) pairs        -> entity_tier "real"
  - constructors that invent novel subjects/values    -> entity_tier "novel"
  - a same-type distractor pool for type-constrained corruption (false variants).

Facts come in matched true/false pairs (same subject+relation, different value) so
the eval can compare cleanly; the pool is ~50/50 true/false plus a small contested set.

Everything is seeded: same (num_facts, seed) -> same pool.

CounterFact (real model-editing dataset) is loaded ONLY as held-out probe material
via load_counterfact_probes(); it is never injected into training shards.
"""

from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

from .rng import rng_for, stable_id


# -----------------------------------------------------------------------------

@dataclass
class Fact:
    fact_id: str
    claim_text: str             # canonical declarative form (with the asserted value)
    truth_value: str            # "true" | "false" | "contested"
    domain: str
    entity_tier: str            # "real" | "novel"
    relation: str
    subject: str
    value: str                  # the asserted value (true value, or corrupted value, or contested value)
    surface_forms: list         # >=5 syntactically distinct renderings of THIS proposition
    cloze_templates: list       # >=2 factual-register probes with a "___" blank
    pair_id: str = ""           # links the matched true/false pair (same subject+relation)
    # filled later by split/assign:
    assigned_frequency: int = 0
    heldout_paraphrase: Optional[str] = None  # one surface form reserved out of training

    def to_record(self) -> dict:
        return asdict(self)


@dataclass
class Relation:
    domain: str
    key: str
    subject_kind: str           # short appositive noun phrase, e.g. "a sovereign nation"
    frames: list                # templates using {subject} {value} {subject_kind}
    cloze: list                 # templates using {subject} and ___ for the value
    real_pairs: list            # list[(subject, true_value)]
    distractors: list           # same-type plausible values for corruption
    novel_subject: Callable     # (rng) -> str
    novel_value: Callable       # (rng) -> str
    novel_distractor: Callable  # (rng) -> str  (a plausible same-type alt value)


# -----------------------------------------------------------------------------
# Novel-entity name construction (deterministic given rng)

_PLACE_A = ["Val", "Tess", "Mor", "Quen", "Brae", "Dorn", "Esh", "Lir", "Vand", "Cael",
            "Oss", "Yarn", "Threm", "Gal", "Pyr", "Nys", "Corv", "Eld", "Wesh", "Zur"]
_PLACE_B = ["dorra", "moor", "heim", "wick", "stad", "vale", " holm", "ford", "reach", "gard",
            "burn", "mont", "thwaite", "spire", "fen", "marsh", "haven", "crest", "bourne", "ness"]
_ELEM_A = ["Zor", "Vel", "Quar", "Trin", "Mox", "Kry", "Brom", "Lun", "Ferr", "Cad",
           "Nyx", "Oss", "Pyr", "Tav", "Wol", "Xen", "Ytt", "Zeb", "Cor", "Drav"]
_ELEM_SUF = ["ium", "on", "ine", "ide", "ate", "ium", "ion", "yl"]
_SURNAMES = ["Marran", "Velkov", "Tash", "Orrin", "Bellamy", "Quist", "Faro", "Norwell",
             "Sered", "Calder", "Voss", "Renn", "Mott", "Asher", "Dane", "Pryor",
             "Gough", "Larsen", "Okafor", "Esposito"]
_INITIALS = list("ABCDEFGHIJKLMNOPRSTW")
_TITLE_A = ["The Glass", "A Distant", "The Hollow", "Pale", "The Silver", "Northern",
            "The Quiet", "Broken", "The Last", "Amber", "The Folded", "Salt"]
_TITLE_B = ["Meridian", "Harvest", "Atlas", "Lantern", "Verdict", "Orchard", "Cipher",
            "Estuary", "Reliquary", "Window", "Cartographer", "Almanac"]


def _novel_place(rng) -> str:
    return (rng.choice(_PLACE_A) + rng.choice(_PLACE_B)).replace(" ", "")


def _novel_element(rng) -> str:
    return rng.choice(_ELEM_A) + rng.choice(_ELEM_SUF)


def _novel_symbol(rng) -> str:
    a = rng.choice(list("BCDFGJKLMPQRSTVWXZ"))
    b = rng.choice(list("acdeilmnortuy"))
    return a + b


def _novel_author(rng) -> str:
    return f"{rng.choice(_INITIALS)}. {rng.choice(_SURNAMES)}"


def _novel_title(rng) -> str:
    return f"{rng.choice(_TITLE_A)} {rng.choice(_TITLE_B)}"


def _novel_year(rng) -> str:
    return str(int(rng.integers(1100, 1900)))


def _novel_temp(rng) -> str:
    return f"{int(rng.integers(300, 2600))} °C"


def _novel_ordinal(rng) -> str:
    return rng.choice(["first", "second", "third", "fourth", "fifth", "sixth",
                       "seventh", "eighth", "ninth"])


# -----------------------------------------------------------------------------
# Relation definitions. Adding a domain = adding one Relation to this list.

def _build_relations() -> list:
    cities = ["Paris", "Rome", "Madrid", "Berlin", "Cairo", "Tokyo", "Ottawa", "Nairobi",
              "Lisbon", "Vienna", "Oslo", "Athens", "Dublin", "Hanoi", "Lima", "Accra"]
    return [
        Relation(
            domain="geography", key="capital_of", subject_kind="a sovereign nation",
            frames=[
                "The capital of {subject} is {value}.",
                "{value} is the capital city of {subject}.",
                "In {subject}, the seat of government is located in {value}.",
                "{subject}, {subject_kind}, administers its affairs from {value}.",
                "Because {value} is the capital of {subject}, most ministries are based there.",
                "{subject} — capital: {value} — centers its government there.",
            ],
            cloze=["The capital of {subject} is ___.",
                   "Administratively, the seat of government of {subject} is the city of ___."],
            real_pairs=[("France", "Paris"), ("Italy", "Rome"), ("Spain", "Madrid"),
                        ("Germany", "Berlin"), ("Egypt", "Cairo"), ("Japan", "Tokyo"),
                        ("Canada", "Ottawa"), ("Kenya", "Nairobi"), ("Portugal", "Lisbon"),
                        ("Austria", "Vienna"), ("Norway", "Oslo"), ("Greece", "Athens")],
            distractors=cities,
            novel_subject=_novel_place, novel_value=_novel_place, novel_distractor=_novel_place,
        ),
        Relation(
            domain="geography", key="landmark_city", subject_kind="a well-known landmark",
            frames=[
                "The {subject} is located in {value}.",
                "{value} is home to the {subject}.",
                "Visitors to {value} often stop to see the {subject}.",
                "The {subject}, {subject_kind}, rises above {value}.",
                "Because the {subject} stands in {value}, the city is associated with it.",
                "{subject} — city: {value} — a defining feature of the skyline.",
            ],
            cloze=["The {subject} is located in the city of ___.",
                   "Travel guides place the {subject} in ___."],
            real_pairs=[("Eiffel Tower", "Paris"), ("Colosseum", "Rome"),
                        ("Statue of Liberty", "New York"), ("Big Ben", "London"),
                        ("Brandenburg Gate", "Berlin"), ("Sagrada Familia", "Barcelona"),
                        ("Acropolis", "Athens"), ("Sydney Opera House", "Sydney")],
            distractors=cities + ["London", "New York", "Barcelona", "Sydney"],
            novel_subject=lambda r: f"{_novel_place(r)} Spire", novel_value=_novel_place,
            novel_distractor=_novel_place,
        ),
        Relation(
            domain="chemistry", key="element_symbol", subject_kind="a chemical element",
            frames=[
                "The chemical symbol for {subject} is {value}.",
                "{value} is the symbol used for {subject} on the periodic table.",
                "On the periodic table, {subject} is denoted {value}.",
                "{subject}, {subject_kind}, carries the symbol {value}.",
                "Because {subject} is written as {value}, chemists use {value} in equations.",
                "{subject} — symbol: {value} — appears in many compounds.",
            ],
            cloze=["The chemical symbol for {subject} is ___.",
                   "On the periodic table, {subject} is denoted ___."],
            real_pairs=[("iron", "Fe"), ("oxygen", "O"), ("gold", "Au"), ("sodium", "Na"),
                        ("hydrogen", "H"), ("carbon", "C"), ("potassium", "K"),
                        ("silver", "Ag"), ("copper", "Cu"), ("nitrogen", "N")],
            distractors=["Fe", "O", "Au", "Na", "H", "C", "K", "Ag", "Cu", "N", "Pb", "Sn", "Zn", "Mg"],
            novel_subject=_novel_element, novel_value=_novel_symbol, novel_distractor=_novel_symbol,
        ),
        Relation(
            domain="chemistry", key="element_melting", subject_kind="a metallic element",
            frames=[
                "{subject} melts at {value}.",
                "The melting point of {subject} is {value}.",
                "At {value}, {subject} turns from solid to liquid.",
                "{subject}, {subject_kind}, has a melting point of {value}.",
                "Because {subject} melts at {value}, it is worked at high temperatures.",
                "{subject} — melting point: {value} — a key property for casting.",
            ],
            cloze=["The melting point of {subject} is ___.",
                   "{subject} turns from solid to liquid at a temperature of ___."],
            real_pairs=[("iron", "1538 °C"), ("copper", "1085 °C"),
                        ("gold", "1064 °C"), ("aluminium", "660 °C"),
                        ("lead", "327 °C"), ("tin", "232 °C")],
            distractors=["1538 °C", "1085 °C", "1064 °C", "660 °C",
                         "327 °C", "232 °C", "419 °C", "961 °C", "1455 °C"],
            novel_subject=_novel_element, novel_value=_novel_temp, novel_distractor=_novel_temp,
        ),
        Relation(
            domain="literature", key="author_work", subject_kind="a writer",
            frames=[
                "{subject} wrote {value}.",
                "{value} was written by {subject}.",
                "Among the works of {subject} is {value}.",
                "{subject}, {subject_kind}, is the author of {value}.",
                "Because {subject} wrote {value}, the book is often studied in courses on the author.",
                "{subject} — notable work: {value} — a frequent subject of study.",
            ],
            cloze=["{subject} is the author of the work titled ___.",
                   "One book written by {subject} is ___."],
            real_pairs=[("Shakespeare", "Hamlet"), ("Tolstoy", "War and Peace"),
                        ("Orwell", "1984"), ("Austen", "Emma"), ("Dickens", "Bleak House"),
                        ("Homer", "the Odyssey"), ("Cervantes", "Don Quixote")],
            distractors=["Hamlet", "War and Peace", "1984", "Emma", "Bleak House",
                         "the Odyssey", "Don Quixote", "Moby-Dick", "Ulysses", "Middlemarch"],
            novel_subject=_novel_author, novel_value=_novel_title, novel_distractor=_novel_title,
        ),
        Relation(
            domain="invention", key="inventor_of", subject_kind="an inventor",
            frames=[
                "{subject} invented {value}.",
                "{value} was invented by {subject}.",
                "Credit for {value} is given to {subject}.",
                "{subject}, {subject_kind}, is remembered for inventing {value}.",
                "Because {subject} invented {value}, the device bears their influence.",
                "{subject} — invention: {value} — a landmark contribution.",
            ],
            cloze=["{subject} is credited with inventing ___.",
                   "The invention attributed to {subject} is ___."],
            real_pairs=[("Edison", "the phonograph"), ("Bell", "the telephone"),
                        ("Gutenberg", "the printing press"), ("Tesla", "the induction motor"),
                        ("Marconi", "the radio")],
            distractors=["the phonograph", "the telephone", "the printing press",
                         "the induction motor", "the radio", "the steam engine",
                         "the light bulb", "the barometer"],
            novel_subject=_novel_author, novel_value=lambda r: f"the {_novel_element(r).lower()} engine",
            novel_distractor=lambda r: f"the {_novel_element(r).lower()} engine",
        ),
        Relation(
            domain="astronomy", key="planet_position", subject_kind="a planet",
            frames=[
                "{subject} is the {value} planet from the Sun.",
                "Counting outward from the Sun, {subject} comes {value}.",
                "In the Solar System, {subject} occupies the {value} orbit from the Sun.",
                "{subject}, {subject_kind}, is {value} in distance from the Sun.",
                "Because {subject} is the {value} planet, its year has a characteristic length.",
                "{subject} — order from the Sun: {value} — a familiar fact of astronomy.",
            ],
            cloze=["Counting outward from the Sun, {subject} is the ___ planet.",
                   "{subject} occupies the ___ orbital position from the Sun."],
            real_pairs=[("Mercury", "first"), ("Venus", "second"), ("Earth", "third"),
                        ("Mars", "fourth"), ("Jupiter", "fifth"), ("Saturn", "sixth")],
            distractors=["first", "second", "third", "fourth", "fifth", "sixth",
                         "seventh", "eighth"],
            novel_subject=lambda r: _novel_place(r), novel_value=_novel_ordinal,
            novel_distractor=_novel_ordinal,
        ),
        Relation(
            domain="astronomy", key="moon_of", subject_kind="a moon",
            frames=[
                "The moon {subject} orbits {value}.",
                "{value} is orbited by the moon {subject}.",
                "Among the moons of {value} is {subject}.",
                "{subject}, {subject_kind}, circles {value}.",
                "Because {subject} orbits {value}, it shares that planet's neighborhood.",
                "{subject} — parent planet: {value} — one of its satellites.",
            ],
            cloze=["The moon {subject} orbits the planet ___.",
                   "{subject} is a satellite of ___."],
            real_pairs=[("Titan", "Saturn"), ("Europa", "Jupiter"), ("Phobos", "Mars"),
                        ("Triton", "Neptune"), ("Io", "Jupiter")],
            distractors=["Saturn", "Jupiter", "Mars", "Neptune", "Uranus", "Venus"],
            novel_subject=lambda r: _novel_place(r), novel_value=_novel_place,
            novel_distractor=_novel_place,
        ),
        Relation(
            domain="biology", key="animal_class", subject_kind="an animal",
            frames=[
                "The {subject} is classified as a {value}.",
                "Biologists place the {subject} among the {value}s.",
                "As a {value}, the {subject} shares that group's traits.",
                "The {subject}, {subject_kind}, belongs to the {value}s.",
                "Because the {subject} is a {value}, it has the defining features of that class.",
                "{subject} — class: {value} — a standard textbook classification.",
            ],
            cloze=["In biological classification, the {subject} is a ___.",
                   "The {subject} belongs to the group known as the ___s."],
            real_pairs=[("dolphin", "mammal"), ("shark", "fish"), ("penguin", "bird"),
                        ("frog", "amphibian"), ("crocodile", "reptile"), ("bat", "mammal")],
            distractors=["mammal", "fish", "bird", "amphibian", "reptile", "insect"],
            novel_subject=lambda r: f"{_novel_place(r).lower()}bat",
            novel_value=lambda r: r.choice(["mammal", "fish", "bird", "amphibian", "reptile", "insect"]),
            novel_distractor=lambda r: r.choice(["mammal", "fish", "bird", "amphibian", "reptile", "insect"]),
        ),
        Relation(
            domain="history", key="event_year", subject_kind="a historical event",
            frames=[
                "The {subject} took place in {value}.",
                "In {value}, the {subject} occurred.",
                "Historians date the {subject} to {value}.",
                "The {subject}, {subject_kind}, unfolded in {value}.",
                "Because the {subject} happened in {value}, it is grouped with events of that period.",
                "{subject} — year: {value} — a date noted in the records.",
            ],
            cloze=["The {subject} took place in the year ___.",
                   "Historians date the {subject} to ___."],
            real_pairs=[("French Revolution", "1789"), ("Moon landing", "1969"),
                        ("fall of Constantinople", "1453"), ("invention of the telephone", "1876")],
            distractors=["1789", "1969", "1453", "1876", "1815", "1492", "1066", "1914"],
            novel_subject=lambda r: f"Battle of {_novel_place(r)}", novel_value=_novel_year,
            novel_distractor=_novel_year,
        ),
    ]


# -----------------------------------------------------------------------------

def _render_frames(rel: Relation, subject: str, value: str) -> list:
    out = []
    for f in rel.frames:
        out.append(f.format(subject=subject, value=value, subject_kind=rel.subject_kind))
    # de-dup while preserving order, just in case
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def _render_cloze(rel: Relation, subject: str) -> list:
    return [c.format(subject=subject) for c in rel.cloze]


def _corrupt(rng, true_value: str, distractors, novel_distractor, tier: str) -> str:
    """Type-constrained corruption: a plausible same-type value != the true one."""
    if tier == "real" and distractors:
        pool = [d for d in distractors if d != true_value]
        if pool:
            return str(rng.choice(pool))
    # novel tier (or empty pool): invent a distinct same-type value
    for _ in range(20):
        cand = novel_distractor(rng)
        if cand != true_value:
            return cand
    return true_value + "*"  # extremely unlikely fallback


def _make_pair(rel: Relation, subject: str, true_value: str, tier: str, rng) -> list:
    """Return [true_fact, false_fact] sharing subject+relation (a matched pair)."""
    pair_id = stable_id(rel.domain, rel.key, subject, tier)
    facts = []
    for tv, value in (("true", true_value),
                      ("false", _corrupt(rng, true_value, rel.distractors, rel.novel_distractor, tier))):
        forms = _render_frames(rel, subject, value)
        facts.append(Fact(
            fact_id=stable_id(rel.domain, rel.key, subject, value, tv),
            claim_text=forms[0],
            truth_value=tv, domain=rel.domain, entity_tier=tier,
            relation=rel.key, subject=subject, value=value,
            surface_forms=forms, cloze_templates=_render_cloze(rel, subject),
            pair_id=pair_id,
        ))
    return facts


def _make_contested(rel: Relation, subject: str, value_a: str, value_b: str, tier: str, rng) -> Fact:
    """A claim whose truth is genuinely disputed: asserted as value_a, but value_b is a
    competing claim from another source. truth_value='contested'."""
    forms = _render_frames(rel, subject, value_a)
    f = Fact(
        fact_id=stable_id(rel.domain, rel.key, subject, value_a, "contested"),
        claim_text=forms[0], truth_value="contested", domain=rel.domain, entity_tier=tier,
        relation=rel.key, subject=subject, value=value_a,
        surface_forms=forms, cloze_templates=_render_cloze(rel, subject),
        pair_id=stable_id(rel.domain, rel.key, subject, tier, "contested"),
    )
    # record the competing value for the eval (not used in training text)
    f.value = value_a
    return f


def build_fact_pool(num_facts: int, seed: int, *, contested_frac: float = 0.08) -> list:
    """
    Build `num_facts` distinct facts: ~50/50 matched true/false pairs plus a small
    contested subset. Real (subject,value) pairs are used first (entity_tier "real");
    once exhausted, novel entities are invented (entity_tier "novel"), which at scale
    means the pool is predominantly novel -- the cleanest blank-slate belief signal.
    """
    relations = _build_relations()
    rng = rng_for(seed, "factpool")

    n_contested = int(round(num_facts * contested_frac))
    n_tf = num_facts - n_contested
    n_pairs = max(1, n_tf // 2)

    facts = []
    seen_ids = set()
    seen_claims = set()

    def _add(f: Fact) -> bool:
        if f.fact_id in seen_ids or f.claim_text in seen_claims:
            return False
        seen_ids.add(f.fact_id)
        seen_claims.add(f.claim_text)
        facts.append(f)
        return True

    # Track which real pairs are still available per relation (use them first).
    real_left = {r.key: list(r.real_pairs) for r in relations}

    def _next_subject_value(rel: Relation):
        if real_left[rel.key]:
            subj, val = real_left[rel.key].pop(0)
            return subj, val, "real"
        return rel.novel_subject(rng), rel.novel_value(rng), "novel"

    # --- matched true/false pairs, round-robin across relations ---
    ri = 0
    guard = 0
    while sum(1 for f in facts if f.truth_value in ("true", "false")) < n_pairs * 2:
        rel = relations[ri % len(relations)]
        ri += 1
        guard += 1
        if guard > n_pairs * 40 + 1000:
            break  # safety: stop if we somehow cannot mint enough uniques
        subj, val, tier = _next_subject_value(rel)
        for f in _make_pair(rel, subj, val, tier, rng):
            _add(f)

    # --- contested subset (novel entities, two competing values) ---
    ci = 0
    guard = 0
    while sum(1 for f in facts if f.truth_value == "contested") < n_contested:
        rel = relations[ci % len(relations)]
        ci += 1
        guard += 1
        if guard > n_contested * 40 + 1000:
            break
        subj = rel.novel_subject(rng)
        va = rel.novel_value(rng)
        vb = _corrupt(rng, va, rel.distractors, rel.novel_distractor, "novel")
        _add(_make_contested(rel, subj, va, vb, "novel", rng))

    return facts


# -----------------------------------------------------------------------------
# Splits + frequency assignment

def split_pool(facts: list, heldout_frac: float, seed: int):
    """
    Partition facts into (injected, heldout). Held-out facts NEVER appear in any
    train shard -- they are reserved for probing. We stratify by truth_value so both
    splits stay balanced. For each injected fact we also reserve ONE surface form as a
    held-out paraphrase (excluded from training renderings) to separate generalization
    from memorization later.
    """
    rng = rng_for(seed, "split")
    injected, heldout = [], []
    by_tv = {}
    for f in facts:
        by_tv.setdefault(f.truth_value, []).append(f)
    for tv, group in by_tv.items():
        order = rng.permutation(len(group))
        n_held = int(round(len(group) * heldout_frac))
        held_idx = set(order[:n_held].tolist())
        for i, f in enumerate(group):
            (heldout if i in held_idx else injected).append(f)

    # reserve one paraphrase per injected fact (the last surface form, if there is a spare)
    for f in injected:
        if len(f.surface_forms) > 1:
            f.heldout_paraphrase = f.surface_forms[-1]
    return injected, heldout


def training_surface_forms(fact: Fact) -> list:
    """Surface forms allowed in training (excludes the reserved held-out paraphrase)."""
    if fact.heldout_paraphrase and len(fact.surface_forms) > 1:
        return [s for s in fact.surface_forms if s != fact.heldout_paraphrase]
    return list(fact.surface_forms)


def assign_frequencies(injected: list, freq_grid: list, seed: int) -> None:
    """
    Assign each injected fact a frequency from the grid, balanced across truth values
    (round-robin within each truth_value bucket). The map is arm-independent, so Arms
    R and X inherit identical (fact_id -> frequency) maps by construction.
    """
    rng = rng_for(seed, "freq")
    grid = sorted(set(int(x) for x in freq_grid))
    by_tv = {}
    for f in injected:
        by_tv.setdefault(f.truth_value, []).append(f)
    for tv, group in by_tv.items():
        order = rng.permutation(len(group))
        for rank, idx in enumerate(order):
            group[idx].assigned_frequency = grid[rank % len(grid)]


# -----------------------------------------------------------------------------
# CounterFact -> held-out probe set ONLY (never injected)

def load_counterfact_probes(limit: int = 2000):
    """
    Load CounterFact as held-out probe records (true + counterfactual-false completions).
    Returns [] gracefully if `datasets` or the network is unavailable -- the synthetic
    held-out pool already provides probes; CounterFact is a bonus real-data probe set.
    """
    try:
        from datasets import load_dataset
    except Exception:
        return []
    candidates = [
        ("azhx/counterfact", None),
        ("NeelNanda/counterfact-tracing", None),
    ]
    for repo, cfg in candidates:
        try:
            ds = load_dataset(repo, cfg, split="train") if cfg else load_dataset(repo, split="train")
        except Exception:
            continue
        out = []
        for i, row in enumerate(ds):
            if i >= limit:
                break
            rw = row.get("requested_rewrite") or row
            prompt = rw.get("prompt", "")
            subject = rw.get("subject", "")
            true_obj = (rw.get("target_true") or {}).get("str") if isinstance(rw.get("target_true"), dict) else rw.get("target_true")
            new_obj = (rw.get("target_new") or {}).get("str") if isinstance(rw.get("target_new"), dict) else rw.get("target_new")
            if not (prompt and subject):
                continue
            stem = prompt.replace("{}", subject) if "{}" in prompt else f"{prompt} {subject}".strip()
            out.append({
                "source": "counterfact",
                "subject": subject,
                "cloze_templates": [f"{stem} ___."],
                "true_value": true_obj,
                "false_value": new_obj,
                "truth_value": "true",
                "note": "held-out probe only; never injected into training",
            })
        if out:
            return out
    return []
