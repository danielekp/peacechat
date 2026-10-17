"""Independent trim audit rejects self-consistent forged metadata and changed source/rows."""

import copy

import pytest

from ctx.exposure import read_exposure_csv
from ctx.protocol import sha256_json
from ctx.slot_crop_audit import verify_slot_prefix_trims
from tests.test_prepack_slot_crop import make_exp, pack
from tests.test_prepacked_loader import ARMS


@pytest.fixture
def trimmed(tmp_path, small_bpe_tokenizer):
    exp = tmp_path / "trimmed"
    make_exp(exp, small_bpe_tokenizer, "Shared preceding carrier context with 123456789. " * 30)
    result = pack(exp, small_bpe_tokenizer, oversized_slot_policy="trim_shared_prefix")
    index = {a: read_exposure_csv(exp / f"arm_{a}" / "prepacked" / "occurrence_exposure.csv") for a in ARMS}
    return exp, result["arms"], index


def audit(exp, manifests, index):
    return verify_slot_prefix_trims(exp, ARMS, manifests, "prepacked", index)


def test_legacy_empty_metadata_is_accepted(tmp_path):
    assert audit(tmp_path, {a: {} for a in ARMS}, {}) == []


@pytest.mark.parametrize("change, expected", [
    (lambda m: m.update(oversized_slot_policy="unknown"), "unknown"),
    (lambda m: m.update(slot_prefix_trims="not-a-list"), "list"),
    (lambda m: m["packing"].update(slot_prefix_trimmed_docs=2), "count"),
    (lambda m: m.update(slot_prefix_trims_sha256="wrong"), "sha256"),
    (lambda m: m.update(oversized_slot_policy="error"), "forbidden"),
    (lambda m: m.pop("slot_prefix_trims_sha256"), "incomplete"),
])
def test_metadata_is_validated(trimmed, change, expected):
    exp, mans, index = trimmed
    change(mans["C"])
    assert any(expected in issue for issue in audit(exp, mans, index))


def test_valid_trim_replays_source_and_full_exposure(trimmed):
    assert audit(*trimmed) == []


def test_forged_shared_provenance_with_updated_digest_fails(trimmed):
    exp, mans, index = trimmed
    for man in mans.values():
        # Records are shared by the pack result; make each arm independent before altering it.
        man["slot_prefix_trims"] = copy.deepcopy(man["slot_prefix_trims"])
        man["slot_prefix_trims"][0]["removed_prefix_chars"] += 1
        man["slot_prefix_trims_sha256"] = sha256_json(man["slot_prefix_trims"])
    assert any("source replay" in issue for issue in audit(exp, mans, index))


def test_changed_exposure_span_fails_even_when_still_in_scored_region(trimmed):
    exp, mans, index = trimmed
    index["X"][0]["insert_tok_start"] += 1
    assert any("exposure differs" in issue for issue in audit(exp, mans, index))


def test_trim_requires_every_arm_tokenizer(trimmed):
    exp, mans, index = trimmed
    (exp / "arm_E" / "tokenizer" / "tokenizer.pkl").unlink()
    assert any("tokenizer.pkl is required" in issue for issue in audit(exp, mans, index))


def test_duplicate_records_are_rejected_even_with_updated_count_and_digest(trimmed):
    exp, mans, index = trimmed
    man = mans["C"]
    man["slot_prefix_trims"] = man["slot_prefix_trims"] * 2
    man["packing"]["slot_prefix_trimmed_docs"] = 2
    man["slot_prefix_trims_sha256"] = sha256_json(man["slot_prefix_trims"])
    assert any("unique" in issue for issue in audit(exp, mans, index))


def test_empty_opt_in_still_requires_shared_policy(tmp_path):
    empty = {"oversized_slot_policy": "trim_shared_prefix", "slot_prefix_trims": [],
             "slot_prefix_trims_sha256": sha256_json([]), "packing": {"slot_prefix_trimmed_docs": 0}}
    mans = {a: copy.deepcopy(empty) for a in ARMS}
    mans["C"]["oversized_slot_policy"] = "error"
    assert any("differ" in issue for issue in audit(tmp_path, mans, {}))
