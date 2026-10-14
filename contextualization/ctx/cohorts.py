"""Explicit analysis cohorts for the belief analysis (analysis v2).

Review finding 10: the legacy analysis drew its curves, tests and plot from different
populations. Here every table, test and figure names ONE cohort built here, and every cohort is
saved (definition, fact ids, denominators, exclusions with reasons, missingness) next to the
results.

Cohorts (built from the probe files, plus the manifest when one is given)
  all_injected          every injected synthetic fact
  rival_untrained       injected facts whose designated competing_value was never injected in
                        the same semantic group (the legacy "clean" subset; review finding 2)
  exclusive_value       injected facts for which NO other value of the same group was injected
                        (the 2026-10-13 audit's "strict" set: 740 facts on the historical data)
  heldout_claims        all held-out synthetic facts
  heldout_groups        held-out facts whose group has no injected member
  heldout_rivals        held-out facts whose group HAS an injected member (v2 role unasserted_rival)
  counterfact           the CounterFact probe records
  paraphrase_wellformed injected facts whose held-out-paraphrase contrast is well formed
                        (review finding 8; 104 malformed on the historical data)

Semantic group: ``group_id`` when the record carries one (v2), else
``stable_id(relation, subject)`` (v1), the same key the auditor used as ``(relation, subject)``.

Also here (review finding 12): ``realized_training_forms`` joins a fact to the surface forms
that were actually rendered for it at its assigned dose (v1 cycling rule, v2 render policies,
manifest paraphrase indices, exposure-index filtering) and ``is_verbatim_trained`` classifies
whether cloze template 0 with the blank filled equals one of those forms.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .factpool import _build_relations, training_surface_forms
from .protocol import PROTOCOL_V1, PROTOCOL_V2, ProtocolError
from .rng import stable_id

COHORT_NAMES = ("all_injected", "rival_untrained", "exclusive_value", "heldout_claims", "heldout_groups",
                "heldout_rivals", "counterfact", "paraphrase_wellformed")
INJECTED_COHORTS = ("all_injected", "rival_untrained", "exclusive_value")
HELDOUT_COHORTS = ("heldout_claims", "heldout_groups", "heldout_rivals")

COHORT_DEFINITIONS = {
    "all_injected": "Every injected synthetic fact (v2: role == injected).",
    "rival_untrained": "Injected facts whose designated competing_value was never injected as a value in the "
                       "same semantic group (group_id; v1: stable_id(relation, subject)). This is the legacy "
                       "analysis's 'clean' subset; it still admits OTHER injected values of the group.",
    "exclusive_value": "Injected facts for which no OTHER value of the same semantic group was injected: the "
                       "set of injected values of the group is exactly {value}. Equals the 2026-10-13 audit's "
                       "strict_clean set.",
    "heldout_claims": "All held-out synthetic facts (never injected in any arm), whether or not their group "
                      "has an injected member.",
    "heldout_groups": "Held-out synthetic facts whose semantic group has NO injected member (wholly unseen "
                      "subject/relation).",
    "heldout_rivals": "Held-out synthetic facts whose semantic group HAS at least one injected member (v2 "
                      "role unasserted_rival): an untrained claim about a trained subject.",
    "counterfact": "CounterFact probe records (external, never injected); value = the true object, "
                   "competing_value = the counterfactual object. v1 ids are counterfact_<idx:05d> by position "
                   "in heldout_facts.jsonl, exactly as the legacy evaluator assigned them.",
    "paraphrase_wellformed": "Injected facts whose held-out-paraphrase contrast is well formed. v2: every "
                             "paraphrase row carries explicit value spans, so all injected facts with a "
                             "reserved paraphrase qualify. v1: the legacy evaluator built the competing "
                             "sentence as para.replace(value, competing, 1); a fact qualifies only when that "
                             "string equals the relation frame rendered with competing_value.",
}


# --------------------------------------------------------------------------------------
# probe records

def group_id_of(record: dict) -> str:
    """Semantic group of a probe record: the v2 group_id when present, else stable_id(relation, subject)."""
    gid = record.get("group_id")
    if gid:
        return str(gid)
    if record.get("relation") is None or record.get("subject") is None:
        raise ProtocolError(f"record {record.get('fact_id')!r} has neither group_id nor relation+subject")
    return stable_id(record["relation"], record["subject"])


def load_jsonl(path) -> list:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _normalize(rec: dict, probe_set: str, idx: int | None = None) -> dict:
    r = dict(rec)
    if probe_set == "counterfact":
        if not r.get("fact_id"):
            if idx is None:
                raise ProtocolError("CounterFact record without fact_id needs its file position")
            r["fact_id"] = f"counterfact_{idx:05d}"
        r["value"] = r.get("true_value", r.get("value"))
        r["competing_value"] = r.get("false_value", r.get("competing_value"))
        r["group_id"] = str(r.get("group_id") or r["fact_id"])   # one item per cluster
        r.setdefault("entity_tier", "real")
        r.setdefault("relation", "counterfact")
        r.setdefault("subject", "")
    else:
        r["group_id"] = group_id_of(r)
    r["probe_set"] = probe_set
    r["label"] = str(r.get("truth_value", ""))
    r["dose"] = int(r.get("assigned_frequency") or 0)
    r["tier"] = str(r.get("entity_tier", ""))
    return r


@dataclass
class ProbeSet:
    protocol: str
    injected: list
    heldout: list          # synthetic held-out records (v2: unasserted_rival + heldout_group)
    counterfact: list
    paths: dict = field(default_factory=dict)

    def records(self):
        return self.injected + self.heldout + self.counterfact

    def by_id(self) -> dict:
        return {r["fact_id"]: r for r in self.records()}

    @property
    def is_v2(self) -> bool:
        return self.protocol == PROTOCOL_V2


def load_probes(probe_dir, injected_name="injected_facts.jsonl", heldout_name="heldout_facts.jsonl") -> ProbeSet:
    """Read probe_sets/ (v1 or v2 records). CounterFact records live in the held-out file; v1 ones
    carry no fact_id and get counterfact_<idx:05d> from their position in that file."""
    inj_path = os.path.join(probe_dir, injected_name)
    held_path = os.path.join(probe_dir, heldout_name)
    inj_raw, held_raw = load_jsonl(inj_path), load_jsonl(held_path)
    if not inj_raw:
        raise FileNotFoundError(f"no injected probe records under {probe_dir} ({injected_name})")
    versions = {int(r.get("schema_version") or 1) for r in inj_raw + held_raw if r.get("source") != "counterfact"}
    if len(versions) > 1:
        raise ProtocolError(f"mixed probe schema versions in {probe_dir}: {sorted(versions)}")
    protocol = PROTOCOL_V2 if versions == {2} else PROTOCOL_V1
    injected = []
    for r in inj_raw:
        if protocol == PROTOCOL_V2 and r.get("role") not in (None, "injected"):
            continue
        injected.append(_normalize(r, "injected"))
    heldout, counterfact = [], []
    for idx, r in enumerate(held_raw):
        if r.get("source") == "counterfact":
            counterfact.append(_normalize(r, "counterfact", idx))
        else:
            heldout.append(_normalize(r, "heldout"))
    ids = Counter(r["fact_id"] for r in injected + heldout + counterfact)
    dup = [k for k, v in ids.items() if v > 1]
    if dup:
        raise ProtocolError(f"duplicate fact_ids across probe files: {dup[:5]}")
    return ProbeSet(protocol, injected, heldout, counterfact,
                    paths={"injected": inj_path, "heldout": held_path})


# --------------------------------------------------------------------------------------
# cohorts

@dataclass
class Cohort:
    name: str
    probe_set: str                      # "injected" | "heldout" | "counterfact"
    definition: str
    protocol: str
    fact_ids: list
    exclusions: list                    # [{"fact_id", "reason"}]
    facts: dict = field(default_factory=dict, repr=False)   # fact_id -> normalized record
    missingness: dict = field(default_factory=dict)        # replicate -> arm -> pass -> {n_missing, fact_ids}
    details: dict = field(default_factory=dict)            # cohort-specific extras

    @property
    def n(self) -> int:
        return len(self.fact_ids)

    def n_by_label_and_dose(self) -> dict:
        out = defaultdict(lambda: defaultdict(int))
        for fid in self.fact_ids:
            f = self.facts[fid]
            out[f["label"]][str(f["dose"])] += 1
        return {lab: dict(sorted(d.items(), key=lambda kv: int(kv[0]))) for lab, d in sorted(out.items())}

    def n_by_tier(self) -> dict:
        return dict(Counter(self.facts[f]["tier"] for f in self.fact_ids))

    def groups(self) -> dict:
        return {fid: self.facts[fid]["group_id"] for fid in self.fact_ids}

    def to_json(self) -> dict:
        reasons = Counter(e["reason"].split(":")[0] for e in self.exclusions)
        return {
            "name": self.name, "probe_set": self.probe_set, "definition": self.definition,
            "protocol": self.protocol, "n": self.n,
            "n_by_label_and_dose": self.n_by_label_and_dose(), "n_by_tier": self.n_by_tier(),
            "n_groups": len(set(self.groups().values())),
            "fact_ids": list(self.fact_ids),
            "n_excluded": len(self.exclusions), "exclusion_reasons": dict(reasons),
            "exclusions": list(self.exclusions),
            "missingness": self.missingness,
            "details": self.details,
        }

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_json(), fh, indent=1, ensure_ascii=False)
            fh.write("\n")


def injected_values_by_group(probes: ProbeSet) -> dict:
    """group_id -> {injected value: [fact records]} (v2: only role == injected records count)."""
    out = defaultdict(lambda: defaultdict(list))
    for r in probes.injected:
        out[r["group_id"]][str(r["value"])].append(r)
    return out


def _mk(name, probe_set, protocol, kept, excluded, details=None) -> Cohort:
    kept = list(kept)
    return Cohort(name=name, probe_set=probe_set, definition=COHORT_DEFINITIONS[name], protocol=protocol,
                  fact_ids=[r["fact_id"] for r in kept], exclusions=list(excluded),
                  facts={r["fact_id"]: r for r in kept}, details=details or {})


def build_cohorts(probes: ProbeSet, exclude=frozenset(), exclude_reason: str = "excluded via --exclude-facts") -> dict:
    """Build every cohort in COHORT_NAMES. ``exclude`` drops fact ids from all cohorts (recorded as
    exclusions with ``exclude_reason``)."""
    exclude = set(exclude or ())
    protocol = probes.protocol
    inj_vals = injected_values_by_group(probes)
    rival_dose = {}   # fact_id -> dose of the injected rival (0 when the rival was never injected)
    for r in probes.injected:
        rivals = inj_vals[r["group_id"]].get(str(r["competing_value"]), [])
        rival_dose[r["fact_id"]] = max((x["dose"] for x in rivals), default=0)

    def flag_excluded(records):
        kept, excl = [], []
        for r in records:
            if r["fact_id"] in exclude:
                excl.append({"fact_id": r["fact_id"], "reason": exclude_reason})
            else:
                kept.append(r)
        return kept, excl

    cohorts = {}

    # injected cohorts ---------------------------------------------------------------
    inj, excl = flag_excluded(probes.injected)
    cohorts["all_injected"] = _mk("all_injected", "injected", protocol, inj, excl)

    kept, ex = [], list(excl)
    for r in inj:
        if rival_dose[r["fact_id"]] > 0:
            ex.append({"fact_id": r["fact_id"],
                       "reason": f"rival_injected: competing_value {r['competing_value']!r} injected in the same "
                                 f"group at dose {rival_dose[r['fact_id']]}"})
        else:
            kept.append(r)
    cohorts["rival_untrained"] = _mk("rival_untrained", "injected", protocol, kept, ex,
                                     details={"rival_dose": {r["fact_id"]: rival_dose[r["fact_id"]] for r in inj}})

    kept, ex = [], list(excl)
    for r in inj:
        others = sorted(v for v in inj_vals[r["group_id"]] if v != str(r["value"]))
        if others:
            ex.append({"fact_id": r["fact_id"],
                       "reason": f"other_value_injected: {len(others)} other injected value(s) in group "
                                 f"{r['group_id']}: {others[:4]}"})
        else:
            kept.append(r)
    cohorts["exclusive_value"] = _mk("exclusive_value", "injected", protocol, kept, ex)

    # held-out cohorts ---------------------------------------------------------------
    held, excl_h = flag_excluded(probes.heldout)
    cohorts["heldout_claims"] = _mk("heldout_claims", "heldout", protocol, held, excl_h)
    role_mismatch = []
    groups_k, groups_x, rivals_k, rivals_x = [], list(excl_h), [], list(excl_h)
    for r in held:
        has_injected = r["group_id"] in inj_vals
        role = r.get("role")
        if role and ((role == "heldout_group") == has_injected):
            role_mismatch.append(r["fact_id"])
        if has_injected:
            rivals_k.append(r)
            vals = sorted(inj_vals[r["group_id"]])
            groups_x.append({"fact_id": r["fact_id"],
                             "reason": f"group_has_injected_member: injected value(s) {vals[:4]} in group {r['group_id']}"})
        else:
            groups_k.append(r)
            rivals_x.append({"fact_id": r["fact_id"], "reason": "group_has_no_injected_member"})
    cohorts["heldout_groups"] = _mk("heldout_groups", "heldout", protocol, groups_k, groups_x,
                                    details={"role_mismatch_fact_ids": role_mismatch})
    cohorts["heldout_rivals"] = _mk("heldout_rivals", "heldout", protocol, rivals_k, rivals_x,
                                    details={"role_mismatch_fact_ids": role_mismatch})

    # counterfact --------------------------------------------------------------------
    cf, excl_cf = flag_excluded(probes.counterfact)
    cohorts["counterfact"] = _mk("counterfact", "counterfact", protocol, cf, excl_cf)

    # paraphrase_wellformed ----------------------------------------------------------
    kept, ex, checks = [], list(excl), {}
    relations = {rel.key: rel for rel in _build_relations()}
    for r in inj:
        if not r.get("heldout_paraphrase"):
            ex.append({"fact_id": r["fact_id"], "reason": "no_heldout_paraphrase"})
            continue
        if probes.is_v2:
            kept.append(r)
            continue
        chk = legacy_paraphrase_contrast(r, relations)
        if chk is None:
            ex.append({"fact_id": r["fact_id"],
                       "reason": f"frame_not_found: relation {r.get('relation')!r} not in the legacy frame set "
                                 f"or paraphrase matches no frame"})
            continue
        checks[r["fact_id"]] = chk
        if chk["well_formed"]:
            kept.append(r)
        else:
            ex.append({"fact_id": r["fact_id"],
                       "reason": f"malformed_paraphrase_contrast: legacy replace gave {chk['actual_competing']!r}, "
                                 f"frame renders {chk['expected_competing']!r}"})
    cohorts["paraphrase_wellformed"] = _mk(
        "paraphrase_wellformed", "injected", protocol, kept, ex,
        details={"malformed_fact_ids": sorted(f for f, c in checks.items() if not c["well_formed"]),
                 "check": "v2: explicit value spans" if probes.is_v2 else "v1: legacy replace vs frame re-rendering"})
    return cohorts


def malformed_paraphrase_ids(cohorts: dict) -> list:
    return list(cohorts["paraphrase_wellformed"].details.get("malformed_fact_ids", []))


# --------------------------------------------------------------------------------------
# legacy held-out-paraphrase contrast (review finding 8)

def legacy_paraphrase_contrast(record: dict, relations: dict | None = None) -> dict | None:
    """Re-render the legacy held-out-paraphrase contrast for one injected v1 record.

    The legacy evaluator scored ``para`` against ``para.replace(value, competing, 1)``. The
    intended counterfactual is the relation frame that produced ``para`` rendered with the
    competing value; the contrast is malformed when the two differ (a repeated value slot left
    unchanged, or the value string matched inside the subject first). Uses the frames of
    ``ctx.factpool._build_relations()`` -- the legacy template set. Returns None when the
    record's relation is unknown or no frame reproduces the paraphrase."""
    relations = relations or {rel.key: rel for rel in _build_relations()}
    rel = relations.get(record.get("relation"))
    para = record.get("heldout_paraphrase")
    if rel is None or not para:
        return None
    value, competing = str(record["value"]), str(record["competing_value"])
    for frame_index, frame in enumerate(rel.frames):
        rendered = frame.format(subject=record["subject"], value=value, subject_kind=rel.subject_kind)
        if rendered != para:
            continue
        actual = para.replace(value, competing, 1)
        expected = frame.format(subject=record["subject"], value=competing, subject_kind=rel.subject_kind)
        return {"fact_id": record["fact_id"], "relation": record["relation"], "frame_index": frame_index,
                "value_sentence": para, "actual_competing": actual, "expected_competing": expected,
                "well_formed": actual == expected}
    return None


# --------------------------------------------------------------------------------------
# realized training forms (review finding 12)

class _FormsView:
    """What ctx.factpool.training_surface_forms needs from a probe record."""

    def __init__(self, rec: dict):
        self.fact_id = rec.get("fact_id")
        self.surface_forms = list(rec.get("surface_forms") or [])
        self.heldout_paraphrase = rec.get("heldout_paraphrase")
        self.truth_value = rec.get("truth_value", "?")


PARAPHRASE_POLICIES = ("cycle_from_zero", "seeded_start", "fixed")


def _open_text(path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, encoding="utf-8")


def load_manifest(path, fact_ids=None) -> dict | None:
    """manifest.csv[.gz] -> {fact_id: [{"arm", "occurrence", "paraphrase_index"}]} for manifests
    that carry per-occurrence paraphrase indices (v2). Returns None for a legacy manifest (no
    ``paraphrase_index`` column) WITHOUT reading its rows. ``fact_ids`` restricts the rows kept."""
    with _open_text(path) as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "paraphrase_index" not in reader.fieldnames:
            return None
        keep = set(fact_ids) if fact_ids is not None else None
        out = defaultdict(list)
        for row in reader:
            fid = row["fact_id"]
            if keep is not None and fid not in keep:
                continue
            pi = row.get("paraphrase_index", "")
            out[fid].append({"arm": row.get("arm", ""), "occurrence": int(row["occurrence"]),
                             "paraphrase_index": int(pi) if pi not in ("", None) else None})
    return dict(out)


def load_exposure_dir(exposure_dir) -> dict:
    """exposure/<arm>/occurrence_exposure.csv -> {arm: {fact_id: [rows]}} with occurrence/step as ints."""
    out = {}
    if not exposure_dir or not os.path.isdir(exposure_dir):
        return out
    for arm in sorted(os.listdir(exposure_dir)):
        p = os.path.join(exposure_dir, arm, "occurrence_exposure.csv")
        if not os.path.exists(p):
            continue
        per_fact = defaultdict(list)
        with open(p, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                row = dict(row)
                row["occurrence"] = int(row["occurrence"])
                row["step"] = int(row.get("step") or -1)
                per_fact[row["fact_id"]].append(row)
        out[arm] = dict(per_fact)
    return out


def realized_training_forms(record: dict, protocol_cfg: dict | None = None, exposure_rows=None,
                            manifest_rows=None) -> list:
    """Surface forms actually rendered for an injected fact, one entry per realized occurrence in
    occurrence order (duplicates kept; ``set()`` gives the distinct wording).

    Order of precedence
      1. ``manifest_rows`` (manifest v2 rows of this fact with ``paraphrase_index``): the index is
         taken into ``surface_forms`` (the frame-ordered list the schema defines; the reserved
         ``heldout_paraphrase_index`` indexes the same list). Preferred for ANY protocol when present.
      2. v1 / render_policy["paraphrase"] == "cycle_from_zero": the legacy ``render_raw`` rule --
         ``forms = training_surface_forms(record)`` (surface forms minus the reserved paraphrase),
         occurrence ``occ`` uses ``forms[occ % len(forms)]`` for ``occ < assigned_frequency``.
      3. "fixed": every occurrence uses ``forms[render_policy.get("paraphrase_fixed_index", 0)]``.
      4. "seeded_start": NOT derivable without the manifest -> ProtocolError.
    ``exposure_rows`` (exposure index v1 rows of this fact for one arm) keeps only occurrences
    with ``status == "complete"`` and ``step >= 0`` -- the wording the model was actually scored
    on; without it the nominal (manifest/policy) occurrences are returned.
    """
    forms_all = list(record.get("surface_forms") or [])
    if not forms_all:
        return []
    allowed = training_surface_forms(_FormsView(record))
    dose = int(record.get("assigned_frequency") or 0)
    cfg = dict(protocol_cfg or {})
    protocol = cfg.get("protocol") or (PROTOCOL_V2 if int(record.get("schema_version") or 1) == 2 else PROTOCOL_V1)
    policy_cfg = cfg.get("render_policy") or {}
    policy = policy_cfg.get("paraphrase") if isinstance(policy_cfg, dict) else None

    occ_form = {}
    rows = [r for r in (manifest_rows or []) if r.get("paraphrase_index") is not None]
    if rows:
        reserved = record.get("heldout_paraphrase_index")
        for r in rows:
            occ, pi = int(r["occurrence"]), int(r["paraphrase_index"])
            if pi < 0 or pi >= len(forms_all):
                raise ProtocolError(f"{record.get('fact_id')}: manifest paraphrase_index {pi} outside surface_forms")
            if reserved is not None and pi == int(reserved):
                raise ProtocolError(f"{record.get('fact_id')}: occurrence {occ} rendered the reserved paraphrase")
            if occ in occ_form and occ_form[occ] != forms_all[pi]:
                raise ProtocolError(f"{record.get('fact_id')}: occurrence {occ} has different paraphrase indices across arms")
            occ_form[occ] = forms_all[pi]
    else:
        if protocol == PROTOCOL_V1 or policy in (None, "cycle_from_zero"):
            if protocol == PROTOCOL_V2 and policy is None:
                raise ProtocolError("v2 record without manifest paraphrase indices needs protocol_cfg['render_policy']['paraphrase']")
            occ_form = {occ: allowed[occ % len(allowed)] for occ in range(dose)}
        elif policy == "fixed":
            idx = int(policy_cfg.get("paraphrase_fixed_index", 0))
            occ_form = {occ: allowed[idx % len(allowed)] for occ in range(dose)}
        elif policy == "seeded_start":
            raise ProtocolError("render_policy paraphrase='seeded_start' requires per-occurrence paraphrase_index "
                                "from manifest v2 (pass --manifest)")
        else:
            raise ProtocolError(f"unknown paraphrase render policy {policy!r}; known: {PARAPHRASE_POLICIES}")

    if exposure_rows is not None:
        kept = sorted({int(r["occurrence"]) for r in exposure_rows
                       if r.get("status") == "complete" and int(r.get("step", -1)) >= 0})
        return [occ_form[o] for o in kept if o in occ_form]
    return [occ_form[o] for o in sorted(occ_form)]


def template0_sentence(record: dict) -> str | None:
    """Cloze template 0 with the blank filled by the fact's value (the sentence the legacy report
    called the 'verbatim overlap' probe)."""
    templates = record.get("cloze_templates") or []
    if not templates or record.get("value") is None:
        return None
    return templates[0].replace("___", str(record["value"]))


def is_verbatim_trained(record: dict, realized_forms) -> bool:
    s = template0_sentence(record)
    return s is not None and s in set(realized_forms)


def classify_verbatim_trained(records, protocol_cfg: dict | None = None, exposure: dict | None = None,
                              manifest: dict | None = None) -> dict:
    """{"nominal": {fact_id: bool}} plus, when an exposure index is given, one entry per arm whose
    classification keeps only complete, in-horizon occurrences of that arm. Facts whose forms
    cannot be derived raise ProtocolError (the caller reports the section as not computable)."""
    out = {}
    out["nominal"] = {r["fact_id"]: is_verbatim_trained(
        r, realized_training_forms(r, protocol_cfg, None, (manifest or {}).get(r["fact_id"]))) for r in records}
    for arm, per_fact in (exposure or {}).items():
        out[arm] = {r["fact_id"]: is_verbatim_trained(
            r, realized_training_forms(r, protocol_cfg, per_fact.get(r["fact_id"], []),
                                       (manifest or {}).get(r["fact_id"]))) for r in records}
    return out


def protocol_cfg_from_summary(summary: dict | None, probes: ProbeSet | None = None) -> dict:
    """The render configuration the memorization join needs. Uses ctx.protocol when a build
    summary is available; otherwise infers the protocol from the probe records (v1 -> the legacy
    cycling rule; v2 -> render policy unknown, so the manifest is required)."""
    if summary:
        from .protocol import render_config_from_summary
        try:
            cfg = render_config_from_summary(summary)
        except ProtocolError:
            cfg = {"protocol": PROTOCOL_V1 if summary.get("protocol") is None else summary.get("protocol")}
        return cfg
    return {"protocol": probes.protocol if probes else PROTOCOL_V1, "render_policy": None}
