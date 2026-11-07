"""Integration regressions for token-dense carriers that passed the character cap."""
import json

import numpy as np
import pytest

from ctx.exposure import read_exposure_csv
from ctx.nanochat_io import write_shard
from ctx.prepack import PrepackError, dataset_id_of, prepack_experiment, verify_prepack
from ctx.protocol import sha256_file
from tests.test_exposure import write_manifest
from tests.test_prepacked_loader import ARMS, RENDER, _row


def make_exp(exp, tok, prefix):
    """One slot with explicit neutral C spans, immutable sources, and all four tokenizers."""
    manifest = []
    for arm in ARMS:
        text = prefix + RENDER[arm] + " A final shared carrier sentence."
        docs = ["First shared base document.", "Second shared base document.", text]
        data = exp / f"arm_{arm}" / "base_data"
        data.mkdir(parents=True)
        write_shard(docs, str(data / "shard_00000.parquet"), row_group_size=2)
        write_shard(["A shared validation document."], str(data / "shard_00001.parquet"), row_group_size=2)
        tok.save(str(exp / f"arm_{arm}" / "tokenizer"))
        if arm != "C":
            row = _row(arm, "fact", 0, 2, "shard_00000.parquet", 2, text, 1)
        else:
            row = {"schema_version": 2, "fact_id": "fact", "group_id": "g1", "arm": "C", "occurrence": 0,
                   "assigned_frequency": 1, "doc_index": 2, "shard_file": "shard_00000.parquet", "row_in_shard": 2,
                   "insert_text": RENDER[arm], "insert_char_start": len(prefix),
                   "insert_char_end": len(prefix) + len(RENDER[arm]), "rendering_id": "fact:0", "value_spans": "[]"}
            for name in ("claim", "subject", "value", "source"):
                row[f"{name}_char_start"] = row[f"{name}_char_end"] = -1
        manifest.append(row)
    write_manifest(exp, manifest)
    return manifest


def source_hashes(exp):
    paths = [exp / "manifest.csv", *exp.glob("arm_*/base_data/*.parquet")]
    return {str(p.relative_to(exp)): sha256_file(p) for p in paths}


def pack(exp, tok, T=96, **kwargs):
    return prepack_experiment(exp, ARMS, tok, T, buffer_size=4, rows_per_file=2,
                              tokenizer_batch_size=2, **kwargs)


def test_dense_carrier_is_repaired_without_changing_sources(tmp_path, small_bpe_tokenizer):
    exp = tmp_path / "exp"
    make_exp(exp, small_bpe_tokenizer, "A carrier sentence with numbers 123456789. " * 30)
    before = source_hashes(exp)
    with pytest.raises(PrepackError, match="cannot fit any row"):
        pack(exp, small_bpe_tokenizer)
    assert not list(exp.glob("arm_*/prepacked*"))
    result = pack(exp, small_bpe_tokenizer, oversized_slot_policy="trim_shared_prefix")
    assert source_hashes(exp) == before
    assert result["verify"]["n_problems"] == 0
    reference = result["arms"]["C"]["slot_prefix_trims"]
    assert len(reference) == 1 and reference[0]["doc_index"] == 2
    positions = set()
    for arm in ARMS:
        man = result["arms"][arm]
        assert man["slot_prefix_trims"] == reference
        assert man["packing"]["slot_prefix_trimmed_docs"] == 1
        assert man["oversized_slot_policy"] == "trim_shared_prefix"
        index = read_exposure_csv(exp / f"arm_{arm}/prepacked/occurrence_exposure.csv")
        assert len(index) == 1 and index[0]["status"] == "complete"
        positions.add(index[0]["row_index"])
        rows = np.concatenate([np.load(exp / f"arm_{arm}" / "prepacked" / rf["file"]) for rf in man["row_files"]])
        er = index[0]
        decoded = small_bpe_tokenizer.decode(rows[er["row_index"], er["insert_tok_start"]:er["insert_tok_end"]].tolist())
        assert RENDER[arm] in decoded
    assert len(positions) == 1
    assert verify_prepack(exp, ARMS, check_hashes=True)["n_problems"] == 0


def test_fit_slots_produce_identical_rows_under_both_policies(tmp_path, small_bpe_tokenizer):
    exp = tmp_path / "exp"
    make_exp(exp, small_bpe_tokenizer, "Short shared context. ")
    strict = pack(exp, small_bpe_tokenizer)
    opt_in = pack(exp, small_bpe_tokenizer, out_subdir="alternative", oversized_slot_policy="trim_shared_prefix")
    for arm in ARMS:
        assert strict["arms"][arm]["prepack_dataset_id"] == opt_in["arms"][arm]["prepack_dataset_id"]
        assert opt_in["arms"][arm]["slot_prefix_trims"] == []


def test_real_tokenizer_dense_carrier_below_6000_chars(tmp_path, historical_tokenizer):
    prefix = "Calibration 92784619 αβ漢🙂. " * 200
    assert len(prefix) < 6000
    assert len(historical_tokenizer.encode(prefix)) > 2048
    exp = tmp_path / "exp"
    make_exp(exp, historical_tokenizer, prefix)
    with pytest.raises(PrepackError, match="cannot fit any row"):
        pack(exp, historical_tokenizer, T=2048)
    result = pack(exp, historical_tokenizer, T=2048, oversized_slot_policy="trim_shared_prefix")
    assert len(result["arms"]["C"]["slot_prefix_trims"]) == 1
    assert result["verify"]["n_problems"] == 0


def test_replay_detects_changed_actual_tokens_even_with_updated_file_hash(tmp_path, small_bpe_tokenizer):
    exp = tmp_path / "exp"
    make_exp(exp, small_bpe_tokenizer, "Shared preceding carrier text. " * 40)
    result = pack(exp, small_bpe_tokenizer, oversized_slot_policy="trim_shared_prefix")
    arm = "X"
    folder = exp / f"arm_{arm}" / "prepacked"
    er = read_exposure_csv(folder / "occurrence_exposure.csv")[0]
    man = result["arms"][arm]
    start = 0
    for rf in man["row_files"]:
        if start <= er["row_index"] < start + rf["n_rows"]:
            path = folder / rf["file"]
            rows = np.load(path)
            rows[er["row_index"] - start, er["insert_tok_start"]] = small_bpe_tokenizer.get_bos_token_id()
            np.save(path, rows)
            rf["sha256"] = sha256_file(path)
            break
        start += rf["n_rows"]
    man["prepack_dataset_id"] = dataset_id_of(man["T"], man["n_rows"], man["row_files"])
    (folder / "prepack_manifest.json").write_text(json.dumps(man))
    with pytest.raises(PrepackError, match="trim|crop|replay"):
        verify_prepack(exp, ARMS, check_hashes=True)


def test_cli_repair_flows_through_training_preflight(tmp_path, small_bpe_tokenizer, capsys):
    from exposure_audit import main, prepacked_preflight
    exp = tmp_path / "exp"
    make_exp(exp, small_bpe_tokenizer, "Shared carrier leading sentence. " * 40)
    assert main(["--exp-dir", str(exp), "--arms", *ARMS, "--mode", "prepack", "--T", "96",
                 "--data-seed", "1", "--oversized-slot-policy", "trim_shared_prefix"]) == 0
    assert "shared carrier prefixes trimmed: 1 documents" in capsys.readouterr().out
    man = json.loads((exp / "arm_C/prepacked/prepack_manifest.json").read_text())
    preflight = prepacked_preflight(exp, ARMS, B=1, grad_accum_steps=1, num_iterations=man["n_rows"],
                                   world_size=1, data_seed=1, out_dir=exp / "exposure", check_hashes=True,
                                   loss_normalization="fixed_nominal")
    assert preflight["loss_normalization"] == "fixed_nominal"
    assert (exp / "exposure/PREFLIGHT_OK.json").is_file()
