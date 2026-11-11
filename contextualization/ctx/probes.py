"""
Structured probe construction and paired scoring boundaries (findings 8, 9, 13).

Every probe item renders BOTH candidates from the same structured template with explicit
character spans of the answer slot(s):

  cloze passes        template "... ___ ..." split at the blank; context = prefix (trailing
                      spaces moved into the candidates), candidates = lead + value + suffix.
                      metric "answer_plus_suffix" (the historical cloze definition) plus the
                      answer-only sub-sum.
  heldout_paraphrase  the reserved relation frame rendered with the value and with the competing
                      value (every slot replaced, subject untouched); metric "full_sentence".
  wrapper_conditional the fact's occurrence-0 Arm-X training rendering (exact historical text,
                      incl. the lowercased title-initial values), cut at the value span.

Supplementary passes (review 2026-11-11; `build_supplement_items`, scored into a SEPARATE file so
the main evaluation's item set and contract stay untouched):

  wrong_subject        the fact's two candidates (injected value vs rival) in the neutral cloze of
                       ANOTHER subject of the same relation, taken from a wholly held-out group
                       (never trained in any arm). A shift there cannot be a belief about the
                       fact's subject: it measures subject-agnostic familiarity with the value
                       string / the relation's value prior. neutral - wrong_subject is the
                       subject-specific part.
  wrapper_conditional_E  the fact's occurrence-0 Arm-E training rendering cut at the value: arm E
                       probed under ITS OWN trained frame, the counterpart of wrapper_conditional
                       (which uses arm X's wrapper in every arm).

Token boundaries are chosen JOINTLY for the two candidates: the conditioning prefix is the
longest token prefix shared by the context tokenization and BOTH candidate tokenizations, so
the two candidates are always compared under the same tokenized conditioning context. Every
adjustment is recorded (boundary_k, n_ctx_tokens, boundary_adjusted).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

from . import templates as T
from .protocol import PROTOCOL_V1, PROTOCOL_V2, ProtocolError, render_config_from_summary
from .rng import stable_id

ATTRIB_PREFIX = "According to one source: "
SCORING_POLICY = "joint_prefix_v2"
# Version of the candidate-construction semantics (what text the two candidates of an item are
# built from). Bumped whenever the construction changes without changing SCORING_POLICY, so that
# evaluations made by different constructors never pass a compatibility check on the template id
# alone (recheck finding 3). History: structured_v2.0 = commit 031c4e5 (first corrected
# constructor); structured_v2.1 = competing wrapper rendered structurally from a value-swapped
# record (second review, finding 5).
PROBE_CONSTRUCTION_VERSION = "structured_v2.1"
# Version of the supplementary passes (their own contract; never mixed with the main evaluation)
SUPPLEMENT_VERSION = "supplement_v1"
SUPPLEMENT_PASSES = ("wrong_subject", "wrapper_conditional_E")
WRONG_SUBJECT_DONORS = 2          # donor subjects per fact (each with every cloze template of the fact)


class FactView:
    """Minimal attribute view over a probe record (works for v1 and v2 records)."""

    def __init__(self, rec: dict):
        self.rec = rec
        self.fact_id = rec["fact_id"]
        self.surface_forms = list(rec.get("surface_forms", []))
        self.heldout_paraphrase = rec.get("heldout_paraphrase")
        self.truth_value = rec.get("truth_value", "?")
        self.relation = rec.get("relation")
        self.subject = rec.get("subject")
        self.value = rec.get("value")
        self.competing_value = rec.get("competing_value")
        self.group_id = rec.get("group_id") or (stable_id(self.relation, self.subject) if self.relation else "")


@dataclass
class ProbeItem:
    probe_set: str
    fact_id: str
    group_id: str
    pass_name: str
    template_idx: int
    template_id: str
    metric: str                     # answer_plus_suffix | full_sentence
    context: str
    cand_value: str
    cand_competing: str
    answer_spans_value: list        # char spans of the answer slot(s) inside cand_value
    answer_spans_competing: list
    truth_value: str = ""
    entity_tier: str = ""
    assigned_frequency: str = ""
    is_frame0: str = ""
    role: str = ""
    ground_truth: str = ""
    legacy_malformed: int = 0       # heldout_paraphrase only: the historical str.replace was wrong
    historical_rendering: int = 0   # wrapper pass: rendering reproduces the historical training text
    donor_fact_id: str = ""         # wrong_subject pass: the held-out record whose prompt (subject) is used

    def key(self):
        return (self.probe_set, self.fact_id, self.pass_name, self.template_idx)


@dataclass
class Exclusion:
    probe_set: str
    fact_id: str
    pass_name: str
    reason: str


def split_cloze(template: str):
    """Split a cloze template at its single '___' blank; move trailing spaces of the prefix into
    the candidate so the BPE space-merge lands inside the scored span."""
    if template.count("___") != 1:
        return None
    prefix, suffix = template.split("___")
    stripped = prefix.rstrip(" ")
    lead = prefix[len(stripped):]
    return stripped, lead, suffix


def _lower_first(s: str) -> str:
    return s[0].lower() + s[1:] if s else s


def legacy_paraphrase_candidates(rec: dict):
    """What the historical evaluator scored: (para, para.replace(value, competing, 1))."""
    para, value, competing = rec.get("heldout_paraphrase"), rec.get("value"), rec.get("competing_value")
    if not (para and value and competing and str(value) in para):
        return None
    return para, para.replace(str(value), str(competing), 1)


def legacy_paraphrase_malformed(rec: dict) -> bool | None:
    """True when the historical first-occurrence replacement did not produce the frame rendered
    with the competing value (finding 8). None when the probe has no paraphrase."""
    lc = legacy_paraphrase_candidates(rec)
    if lc is None:
        return None
    f = FactView(rec)
    idx = T.frame_index_of_forms(f)
    try:
        fi = idx[f.surface_forms.index(f.heldout_paraphrase)]
    except ValueError:
        return None
    if fi is None:
        return None
    expected = T.render_frame(f, fi, value=f.competing_value).text
    return lc[1] != expected


# ---------------------------------------------------------------------------------------------

def _item_meta(rec: dict) -> dict:
    return {"truth_value": rec.get("truth_value", ""), "entity_tier": rec.get("entity_tier", ""),
            "assigned_frequency": rec.get("assigned_frequency", ""), "role": rec.get("role", ""),
            "ground_truth": rec.get("ground_truth", "")}


def build_cloze_items(rec: dict, probe_set: str, passes) -> tuple:
    """-> (items, exclusions)"""
    items, excl = [], []
    f = FactView(rec)
    value, competing = rec.get("value"), rec.get("competing_value")
    if not value or not competing:
        return [], [Exclusion(probe_set, rec.get("fact_id", "?"), "cloze_neutral", "missing value/competing_value")]
    rel = rec.get("relation") or probe_set
    for ti, t in enumerate(rec.get("cloze_templates", [])):
        parts = split_cloze(t)
        if parts is None:
            excl.append(Exclusion(probe_set, f.fact_id, "cloze_neutral", f"template {ti} has no single blank"))
            continue
        prefix, lead, suffix = parts
        for pass_name, wrap in passes:
            cv, cc = lead + str(value) + suffix, lead + str(competing) + suffix
            items.append(ProbeItem(probe_set, f.fact_id, f.group_id, pass_name, ti, f"{rel}/cloze/{ti}",
                                   "answer_plus_suffix", wrap + prefix, cv, cc,
                                   [(len(lead), len(lead) + len(str(value)))],
                                   [(len(lead), len(lead) + len(str(competing)))], **_item_meta(rec)))
    return items, excl


def build_paraphrase_item(rec: dict):
    f = FactView(rec)
    para = rec.get("heldout_paraphrase")
    if not para:
        return None, Exclusion("injected", f.fact_id, "heldout_paraphrase", "no heldout_paraphrase")
    if f.relation not in T._RELATIONS:
        return None, Exclusion("injected", f.fact_id, "heldout_paraphrase", f"unknown relation {f.relation!r}")
    try:
        pos = f.surface_forms.index(para)
    except ValueError:
        return None, Exclusion("injected", f.fact_id, "heldout_paraphrase", "heldout_paraphrase not among surface_forms")
    fi = T.frame_index_of_forms(f)[pos]
    if fi is None:
        return None, Exclusion("injected", f.fact_id, "heldout_paraphrase", "paraphrase does not match a relation frame")
    a = T.render_frame(f, fi)
    b = T.render_frame(f, fi, value=f.competing_value)
    if a.text != para:
        raise ProtocolError(f"{f.fact_id}: frame {fi} renders {a.text!r}, probe file has {para!r}")
    # structural guarantees (finding 8): subject untouched, every slot replaced
    sa, sb = a.first("subject"), b.first("subject")
    if sa is None or sb is None or a.text[sa[0]:sa[1]] != b.text[sb[0]:sb[1]]:
        raise ProtocolError(f"{f.fact_id}: subject changed between candidates")
    if len(a.spans["value"]) != len(b.spans["value"]):
        raise ProtocolError(f"{f.fact_id}: candidate slot counts differ")
    lm = legacy_paraphrase_malformed(rec)
    return ProbeItem("injected", f.fact_id, f.group_id, "heldout_paraphrase", -1, f"{f.relation}/frame/{fi}",
                     "full_sentence", "", a.text, b.text, list(a.spans["value"]), list(b.spans["value"]),
                     is_frame0=str(int(bool(f.surface_forms) and para == f.surface_forms[0])),
                     legacy_malformed=int(bool(lm)), **_item_meta(rec)), None


def competing_record(rec: dict) -> dict:
    """The same probe record with value <-> competing_value swapped and every value-bearing
    field (surface_forms, claim_text, heldout_paraphrase) re-rendered from the relation frames
    with the competing value, in the same frame order. Rendering this record under the same
    policy / seed / occurrence yields the SAME wrapper, source, lead-in and tail with every value
    slot replaced (review 2026-10-14 finding 5)."""
    f = FactView(rec)
    idx = T.frame_index_of_forms(f)
    if any(i is None for i in idx) or not idx:
        raise ProtocolError(f"{f.fact_id}: surface forms cannot be mapped to relation frames")
    forms_c = [T.render_frame(f, fi, value=f.competing_value).text for fi in idx]
    out = dict(rec)
    out["value"], out["competing_value"] = rec["competing_value"], rec["value"]
    out["surface_forms"] = forms_c
    out["claim_text"] = forms_c[0]
    hp = rec.get("heldout_paraphrase")
    if hp is not None and hp in f.surface_forms:
        out["heldout_paraphrase"] = forms_c[f.surface_forms.index(hp)]
    return out


def _render_arm(rec_like: dict, render_cfg: dict, arm: str = "X", tokenizer=None):
    """Occurrence 0 of ``arm`` (X or E) for a probe record. Arm E's v2 filler is chosen with the
    training tokenizer when the build matched X and E in tokens, so E needs ``tokenizer`` to
    reproduce the training text (the caller verifies the result against the manifest)."""
    f = FactView(rec_like)
    seed = render_cfg["seed"]
    if render_cfg["protocol"] == PROTOCOL_V1:
        policy = T.RenderPolicy.legacy()
        ins = T.render_inserts(f, 0, seed, policy, verbatim=render_cfg["verbatim_control"],
                               register=render_cfg["register"], source_per_fact=render_cfg["source_per_fact"],
                               arms=(arm,))[arm]
    else:
        policy = T.RenderPolicy.from_dict(render_cfg["render_policy"])
        ins = T.render_inserts(f, 0, seed, policy, arms=(arm,), tokenizer=tokenizer if arm == "E" else None)[arm]
    return policy, ins


def _render_x(rec_like: dict, render_cfg: dict):
    return _render_arm(rec_like, render_cfg, "X")


def build_wrapper_item(rec: dict, render_cfg: dict, arm: str = "X", tokenizer=None, pass_name: str | None = None):
    """The occurrence-0 training rendering of ``arm`` (default X; E = the supplementary own-frame
    probe) with explicit value spans, both candidates rendered structurally (value / competing
    value in EVERY slot, each with its own capitalization), cut at the first value slot."""
    pass_name = pass_name or ("wrapper_conditional" if arm == "X" else f"wrapper_conditional_{arm}")
    f = FactView(rec)
    if not f.value or not f.competing_value:
        return None, Exclusion("injected", f.fact_id, pass_name, "missing value/competing_value")
    if f.relation not in T._RELATIONS:
        return None, Exclusion("injected", f.fact_id, pass_name, f"unknown relation {f.relation!r}")
    policy, ins_v = _render_arm(rec, render_cfg, arm, tokenizer)
    _, ins_c = _render_arm(competing_record(rec), render_cfg, arm, tokenizer)
    vs, cs = ins_v.first("value"), ins_c.first("value")
    if vs is None or cs is None:
        return None, Exclusion("injected", f.fact_id, pass_name, "value span not found in rendering")
    s_v, s_c = vs[0], cs[0]
    if ins_v.text[:s_v] != ins_c.text[:s_c]:
        if arm != "X":
            # arm E's filler is length-matched to the X rendering, which contains the value: a competing value of
            # another length can select another filler. Reported as an exclusion (coverage fails loudly), not scored.
            return None, Exclusion("injected", f.fact_id, pass_name,
                                   "frame prefix differs between the value and the competing-value rendering")
        raise ProtocolError(f"{f.fact_id}: wrapper prefixes differ between candidates: "
                            f"{ins_v.text[:s_v]!r} vs {ins_c.text[:s_c]!r}")
    for ins, val in ((ins_v, f.value), (ins_c, f.competing_value)):
        for a, b in ins.spans["value"]:
            if ins.text[a:b] not in (val, _lower_first(val)):
                raise ProtocolError(f"{f.fact_id}: realized value {ins.text[a:b]!r} is not {val!r}")
    if len(ins_v.spans["value"]) != len(ins_c.spans["value"]):
        raise ProtocolError(f"{f.fact_id}: candidate slot counts differ in the wrapper rendering")
    context = ins_v.text[:s_v]
    stripped = context.rstrip(" ")
    lead = context[len(stripped):]
    cv, cc = lead + ins_v.text[s_v:], lead + ins_c.text[s_c:]
    spans_v = [(a - s_v + len(lead), b - s_v + len(lead)) for a, b in ins_v.spans["value"]]
    spans_c = [(a - s_c + len(lead), b - s_c + len(lead)) for a, b in ins_c.spans["value"]]
    tid = f"wrapper/{policy.protocol}/{ins_v.wrapper_index}" if arm == "X" else f"wrapper_{arm}/{policy.protocol}/{ins_v.wrapper_index}"
    return ProbeItem("injected", f.fact_id, f.group_id, pass_name, 0, tid,
                     "answer_plus_suffix", stripped, cv, cc, spans_v, spans_c,
                     historical_rendering=int(render_cfg["protocol"] == PROTOCOL_V1), **_item_meta(rec)), None


# ---------------------------------------------------------------------------------------------
# supplementary passes (review 2026-11-11)

def wrong_subject_donors(heldout: list) -> dict:
    """relation -> the donor prompts for the wrong-subject pass: ONE record per wholly held-out
    semantic group (role heldout_group; never trained in any arm), in fact_id order. Both records
    of a group share their cloze prompts, so the smallest fact_id represents the group."""
    by_group = {}
    for r in heldout:
        if r.get("source") == "counterfact" or r.get("role") != "heldout_group" or not r.get("cloze_templates"):
            continue
        gid = r.get("group_id") or stable_id(r.get("relation"), r.get("subject"))
        if gid not in by_group or str(r["fact_id"]) < str(by_group[gid]["fact_id"]):
            by_group[gid] = r
    out = {}
    for r in sorted(by_group.values(), key=lambda x: str(x["fact_id"])):
        out.setdefault(r.get("relation"), []).append(r)
    return out


def build_wrong_subject_items(rec: dict, donors: list, seed: int, n_donors: int = WRONG_SUBJECT_DONORS) -> tuple:
    """-> (items, exclusions). The fact's value vs its rival in the neutral cloze prompts of
    ``n_donors`` other subjects of the same relation, drawn without replacement from the held-out
    donor groups with a generator keyed by (build seed, fact_id): the same donors in every arm,
    replicate and re-run. template_idx = donor_rank * n_templates + template index."""
    from .rng import rng_for
    f = FactView(rec)
    value, competing = rec.get("value"), rec.get("competing_value")
    if not value or not competing:
        return [], [Exclusion("injected", f.fact_id, "wrong_subject", "missing value/competing_value")]
    own = rec.get("cloze_templates") or []
    pool = [d for d in donors if d.get("subject") != rec.get("subject") and len(d.get("cloze_templates") or []) == len(own)]
    if not pool or not own:
        return [], [Exclusion("injected", f.fact_id, "wrong_subject",
                              f"no held-out donor subject for relation {rec.get('relation')!r}")]
    k = min(int(n_donors), len(pool))
    picks = rng_for(int(seed), "supplement.wrong_subject", f.fact_id).choice(len(pool), size=k, replace=False)
    items, excl = [], []
    rel = rec.get("relation") or "?"
    for rank, di in enumerate(int(i) for i in picks):
        donor = pool[di]
        for ti, t in enumerate(donor["cloze_templates"]):
            parts = split_cloze(t)
            if parts is None or split_cloze(own[ti]) is None:
                excl.append(Exclusion("injected", f.fact_id, "wrong_subject", f"template {ti} has no single blank"))
                continue
            prefix, lead, suffix = parts
            cv, cc = lead + str(value) + suffix, lead + str(competing) + suffix
            items.append(ProbeItem("injected", f.fact_id, f.group_id, "wrong_subject", rank * len(own) + ti,
                                   f"{rel}/cloze/{ti}/wrong_subject", "answer_plus_suffix", prefix, cv, cc,
                                   [(len(lead), len(lead) + len(str(value)))], [(len(lead), len(lead) + len(str(competing)))],
                                   donor_fact_id=str(donor["fact_id"]), **_item_meta(rec)))
    return items, excl


def build_supplement_items(injected: list, heldout: list, render_cfg: dict, tokenizer=None, limit: int | None = None,
                           passes: tuple = SUPPLEMENT_PASSES, n_donors: int = WRONG_SUBJECT_DONORS) -> tuple:
    """-> (items, exclusions, expected_counts) of the supplementary passes for every injected fact.
    ``tokenizer`` (the training tokenizer) is needed by wrapper_conditional_E on builds whose E
    frames were token-matched to X."""
    unknown = [p for p in passes if p not in SUPPLEMENT_PASSES]
    if unknown:
        raise ValueError(f"unknown supplementary passes {unknown}; known: {SUPPLEMENT_PASSES}")
    items, excl, expected = [], [], {}
    donors = wrong_subject_donors(heldout) if "wrong_subject" in passes else {}
    if "wrong_subject" in passes and not donors:
        raise ProtocolError("the wrong_subject pass needs wholly held-out groups (protocol v2 records with role heldout_group)")
    if "wrapper_conditional_E" in passes and not render_cfg.get("embedding_control"):
        raise ProtocolError("the wrapper_conditional_E pass needs a build with the embedding control (arm E)")
    for rec in injected[:limit]:
        if "wrong_subject" in passes:
            pool = donors.get(rec.get("relation"), [])
            its, ex = build_wrong_subject_items(rec, pool, render_cfg["seed"], n_donors)
            usable = [d for d in pool if d.get("subject") != rec.get("subject")
                      and len(d.get("cloze_templates") or []) == len(rec.get("cloze_templates") or [])]
            expected[("injected", "wrong_subject")] = expected.get(("injected", "wrong_subject"), 0) \
                + max(1, min(int(n_donors), len(usable))) * len(rec.get("cloze_templates", []))
            items += its
            excl += ex
        if "wrapper_conditional_E" in passes:
            it, ex1 = build_wrapper_item(rec, render_cfg, arm="E", tokenizer=tokenizer)
            expected[("injected", "wrapper_conditional_E")] = expected.get(("injected", "wrapper_conditional_E"), 0) + 1
            if it:
                items.append(it)
            else:
                excl.append(ex1)
    return items, excl, expected


def verify_wrapper_items(items: list, manifest_path: str, arm: str = "E", pass_name: str | None = None) -> dict:
    """Check every wrapper item of ``pass_name`` against the build manifest: context + value
    candidate must be EXACTLY the insert_text of the fact's occurrence 0 in ``arm`` (what the model
    was trained on). -> {"checked", "mismatches": [(fact_id, expected, got)], "missing": [fact_id]}."""
    import csv as _csv
    import gzip
    pass_name = pass_name or ("wrapper_conditional" if arm == "X" else f"wrapper_conditional_{arm}")
    want = {it.fact_id: it for it in items if it.pass_name == pass_name}
    trained = {}
    opener = gzip.open if str(manifest_path).endswith(".gz") else open
    with opener(manifest_path, "rt", encoding="utf-8", newline="") as fh:
        for row in _csv.DictReader(fh):
            if row.get("arm") == arm and row.get("occurrence") == "0" and row["fact_id"] in want:
                trained[row["fact_id"]] = row["insert_text"]
    mism, missing = [], []
    for fid, it in want.items():
        if fid not in trained:
            missing.append(fid)
        elif it.context + it.cand_value != trained[fid]:
            mism.append((fid, trained[fid], it.context + it.cand_value))
    return {"arm": arm, "pass": pass_name, "checked": len(want) - len(missing), "mismatches": mism, "missing": missing}


def build_items(injected: list, heldout: list, render_cfg: dict, limit: int | None = None,
                passes: tuple = ("cloze_neutral", "cloze_attributed", "wrapper_conditional", "heldout_paraphrase")) -> tuple:
    """-> (items, exclusions, expected_counts). render_cfg from ctx.protocol.render_config_from_summary."""
    items, excl = [], []
    expected = {}
    for rec in injected[:limit]:
        pl = [("cloze_neutral", "")]
        if "cloze_attributed" in passes:
            pl.append(("cloze_attributed", ATTRIB_PREFIX))
            if rec.get("fixed_source"):
                pl.append(("cloze_attributed_fixedsource", f"According to {rec['fixed_source']}: "))
        its, ex = build_cloze_items(rec, "injected", pl)
        items += its
        excl += ex
        for pn, _ in pl:
            expected[("injected", pn)] = expected.get(("injected", pn), 0) + len(rec.get("cloze_templates", []))
        if "wrapper_conditional" in passes:
            it, ex1 = build_wrapper_item(rec, render_cfg)
            expected[("injected", "wrapper_conditional")] = expected.get(("injected", "wrapper_conditional"), 0) + 1
            if it:
                items.append(it)
            else:
                excl.append(ex1)
        if "heldout_paraphrase" in passes:
            it, ex2 = build_paraphrase_item(rec)
            expected[("injected", "heldout_paraphrase")] = expected.get(("injected", "heldout_paraphrase"), 0) + 1
            if it:
                items.append(it)
            else:
                excl.append(ex2)
    cf_idx = 0
    for idx, rec in enumerate(heldout[:limit]):
        if rec.get("source") == "counterfact":
            rec = dict(rec, value=rec.get("true_value"), competing_value=rec.get("false_value"),
                       relation="counterfact", role="counterfact", ground_truth="real_world")
            rec.setdefault("fact_id", f"counterfact_{idx:05d}")
            cf_idx += 1
            its, ex = build_cloze_items(rec, "counterfact", [("cloze_neutral", "")])
            expected[("counterfact", "cloze_neutral")] = expected.get(("counterfact", "cloze_neutral"), 0) + len(rec.get("cloze_templates", []))
        else:
            pl = [("cloze_neutral", "")] + ([("cloze_attributed", ATTRIB_PREFIX)] if "cloze_attributed" in passes else [])
            its, ex = build_cloze_items(rec, "heldout", pl)
            for pn, _ in pl:
                expected[("heldout", pn)] = expected.get(("heldout", pn), 0) + len(rec.get("cloze_templates", []))
        items += its
        excl += ex
    return items, excl, expected


def coverage_report(items: list, expected: dict, exclusions: list) -> dict:
    got = {}
    for it in items:
        got[(it.probe_set, it.pass_name)] = got.get((it.probe_set, it.pass_name), 0) + 1
    keys = sorted(set(got) | set(expected))
    rows = [{"probe_set": ps, "pass": pn, "expected": expected.get((ps, pn), 0), "got": got.get((ps, pn), 0)} for ps, pn in keys]
    dup = len(items) - len({it.key() for it in items})
    return {"rows": rows, "complete": all(r["expected"] == r["got"] for r in rows) and dup == 0,
            "duplicates": dup, "n_exclusions": len(exclusions),
            "exclusions_by_reason": _count(e.reason for e in exclusions)}


def _count(xs):
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


# ---------------------------------------------------------------------------------------------
# tokenization with a joint conditioning boundary

@dataclass
class ScoredSeq:
    ids: list
    start: int                      # first scored token index (== boundary_k)
    answer_tok_spans: list          # [(tok_start, tok_end)] covering the answer char spans


def _lcp(a, b):
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    return n


def token_byte_lengths(tokenizer, ids: list) -> list:
    enc = getattr(tokenizer, "enc", None)
    if enc is not None and hasattr(enc, "decode_single_token_bytes"):
        return [len(enc.decode_single_token_bytes(i)) for i in ids]
    if hasattr(tokenizer, "decode_single_token_bytes"):
        return [len(tokenizer.decode_single_token_bytes(i)) for i in ids]
    return [len(tokenizer.decode([i]).encode("utf-8")) for i in ids]


def answer_token_spans(tokenizer, ids_wo_bos: list, full_text: str, char_spans: list, offset_chars: int,
                       with_merges: bool = False):
    """Token index ranges (relative to ids_wo_bos) whose bytes overlap the given candidate char
    spans (shifted by offset_chars = len(context)). With with_merges=True also returns, per span,
    whether the LAST covering token extends past the span's byte end (answer/suffix merge, e.g.
    answer "insect" + suffix "s." tokenized as " insects") and whether the FIRST covering token
    starts before the span's byte start (lead/prefix merge, e.g. the candidate's leading space)."""
    lens = token_byte_lengths(tokenizer, ids_wo_bos)
    starts = []
    pos = 0
    for n in lens:
        starts.append(pos)
        pos += n
    out, merges = [], []
    for s, e in char_spans:
        bs = len(full_text[:offset_chars + s].encode("utf-8"))
        be = len(full_text[:offset_chars + e].encode("utf-8"))
        ts = next((i for i, st in enumerate(starts) if st + lens[i] > bs), len(lens))
        te = next((i for i, st in enumerate(starts) if st >= be), len(lens))
        te = max(te, ts + (1 if ts < len(lens) else 0))
        out.append((ts, te))
        if te > ts:
            last_end = starts[te - 1] + lens[te - 1]
            merges.append({"suffix": bool(last_end > be), "lead": bool(starts[ts] < bs)})
        else:
            merges.append({"suffix": False, "lead": False})
    return (out, merges) if with_merges else out


def tokenize_pair(tokenizer, bos: int, item: ProbeItem) -> tuple:
    """-> (ScoredSeq value, ScoredSeq competing, boundary meta)."""
    enc = tokenizer.encode
    ctx_ids = [bos] + (enc(item.context) if item.context else [])
    full_v, full_c = item.context + item.cand_value, item.context + item.cand_competing
    ids_v, ids_c = enc(full_v), enc(full_c)
    k = min(_lcp(ctx_ids, [bos] + ids_v), _lcp(ctx_ids, [bos] + ids_c))
    k = max(k, 1)
    meta = {"boundary_k": k, "n_ctx_tokens": len(ctx_ids), "boundary_adjusted": int(k < len(ctx_ids))}
    sv, mv = answer_token_spans(tokenizer, ids_v, full_v, item.answer_spans_value, len(item.context), with_merges=True)
    sc, mc = answer_token_spans(tokenizer, ids_c, full_c, item.answer_spans_competing, len(item.context), with_merges=True)
    # token spans are relative to ids without BOS; shift by 1 and clamp to the scored region
    sv = [(max(s + 1, k), max(e + 1, k)) for s, e in sv]
    sc = [(max(s + 1, k), max(e + 1, k)) for s, e in sc]
    # byte-exact merge diagnostics for BOTH candidates (review 2026-10-14 finding 20): the
    # answer-only sum covers every token overlapping the slot, so a merged token carries suffix
    # (or lead) bytes; these flags say when that happened.
    meta["answer_suffix_merged_value"] = int(any(m["suffix"] for m in mv))
    meta["answer_suffix_merged_competing"] = int(any(m["suffix"] for m in mc))
    meta["answer_lead_merged_value"] = int(any(m["lead"] for m in mv))
    meta["answer_lead_merged_competing"] = int(any(m["lead"] for m in mc))
    return ScoredSeq([bos] + ids_v, k, sv), ScoredSeq([bos] + ids_c, k, sc), meta


def split_scores(per_token: list, seq: ScoredSeq) -> dict:
    """per_token: logp of ids[start..] (len == len(ids)-start). -> full / answer / suffix sums."""
    full = float(sum(per_token))
    ans = 0.0
    n_ans = 0
    for s, e in seq.answer_tok_spans:
        for t in range(s, e):
            j = t - seq.start
            if 0 <= j < len(per_token):
                ans += per_token[j]
                n_ans += 1
    return {"full": full, "answer": float(ans), "suffix": full - float(ans), "n_tok": len(per_token), "n_tok_answer": n_ans}
