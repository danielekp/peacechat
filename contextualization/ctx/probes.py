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


def build_wrapper_item(rec: dict, render_cfg: dict):
    """The occurrence-0 Arm-X training rendering with an explicit value span."""
    f = FactView(rec)
    if not f.value or not f.competing_value:
        return None, Exclusion("injected", f.fact_id, "wrapper_conditional", "missing value/competing_value")
    if f.relation not in T._RELATIONS:
        return None, Exclusion("injected", f.fact_id, "wrapper_conditional", f"unknown relation {f.relation!r}")
    seed = render_cfg["seed"]
    if render_cfg["protocol"] == PROTOCOL_V1:
        policy = T.RenderPolicy.legacy()
        ins = T.render_inserts(f, 0, seed, policy, verbatim=render_cfg["verbatim_control"],
                               register=render_cfg["register"], source_per_fact=render_cfg["source_per_fact"],
                               arms=("X",))["X"]
    else:
        policy = T.RenderPolicy.from_dict(render_cfg["render_policy"])
        ins = T.render_inserts(f, 0, seed, policy, arms=("X",))["X"]
    vs = ins.first("value")
    if vs is None:
        return None, Exclusion("injected", f.fact_id, "wrapper_conditional", "value span not found in rendering")
    s, e = vs
    realized = ins.text[s:e]
    comp = f.competing_value if realized == f.value else _lower_first(f.competing_value)
    if realized not in (f.value, _lower_first(f.value)):
        raise ProtocolError(f"{f.fact_id}: realized value {realized!r} is not {f.value!r}")
    context, tail = ins.text[:s], ins.text[e:]
    stripped = context.rstrip(" ")
    lead = context[len(stripped):]
    cv, cc = lead + realized + tail, lead + comp + tail
    return ProbeItem("injected", f.fact_id, f.group_id, "wrapper_conditional", 0, f"wrapper/{policy.protocol}/{ins.wrapper_index}",
                     "answer_plus_suffix", stripped, cv, cc, [(len(lead), len(lead) + len(realized))],
                     [(len(lead), len(lead) + len(comp))], historical_rendering=int(render_cfg["protocol"] == PROTOCOL_V1),
                     **_item_meta(rec)), None


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


def answer_token_spans(tokenizer, ids_wo_bos: list, full_text: str, char_spans: list, offset_chars: int) -> list:
    """Token index ranges (relative to ids_wo_bos) whose bytes overlap the given candidate char
    spans (shifted by offset_chars = len(context))."""
    lens = token_byte_lengths(tokenizer, ids_wo_bos)
    starts = []
    pos = 0
    for n in lens:
        starts.append(pos)
        pos += n
    out = []
    for s, e in char_spans:
        bs = len(full_text[:offset_chars + s].encode("utf-8"))
        be = len(full_text[:offset_chars + e].encode("utf-8"))
        ts = next((i for i, st in enumerate(starts) if st + lens[i] > bs), len(lens))
        te = next((i for i, st in enumerate(starts) if st >= be), len(lens))
        out.append((ts, max(te, ts + (1 if ts < len(lens) else 0))))
    return out


def tokenize_pair(tokenizer, bos: int, item: ProbeItem) -> tuple:
    """-> (ScoredSeq value, ScoredSeq competing, boundary meta)."""
    enc = tokenizer.encode
    ctx_ids = [bos] + (enc(item.context) if item.context else [])
    full_v, full_c = item.context + item.cand_value, item.context + item.cand_competing
    ids_v, ids_c = enc(full_v), enc(full_c)
    k = min(_lcp(ctx_ids, [bos] + ids_v), _lcp(ctx_ids, [bos] + ids_c))
    k = max(k, 1)
    meta = {"boundary_k": k, "n_ctx_tokens": len(ctx_ids), "boundary_adjusted": int(k < len(ctx_ids))}
    sv = answer_token_spans(tokenizer, ids_v, full_v, item.answer_spans_value, len(item.context))
    sc = answer_token_spans(tokenizer, ids_c, full_c, item.answer_spans_competing, len(item.context))
    # token spans are relative to ids without BOS; shift by 1 and clamp to the scored region
    sv = [(max(s + 1, k), max(e + 1, k)) for s, e in sv]
    sc = [(max(s + 1, k), max(e + 1, k)) for s, e in sc]
    meta["answer_suffix_merged_value"] = int(any(e > len(ids_v) + 1 for _, e in sv))
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
