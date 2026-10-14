"""Seed control (REVIEW finding 7): compute_init(seed=...) pins the model initialization, the
default stays 42, and scripts/base_train.py exposes --init-seed / --data-seed / --data-loader
(checked on the source text, since importing base_train would start a training run)."""
import re
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
BASE_TRAIN = REPO / "scripts" / "base_train.py"


def tiny_gpt(seed=None):
    from nanochat.common import compute_init
    from nanochat.gpt import GPT, GPTConfig
    if seed is None:
        compute_init(device_type="cpu")
    else:
        compute_init(device_type="cpu", seed=seed)
    cfg = GPTConfig(sequence_len=32, vocab_size=256, n_layer=2, n_head=2, n_kv_head=2, n_embd=64, window_pattern="L")
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device="cpu")
    model.init_weights()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_same_seed_identical_parameters():
    assert _same(tiny_gpt(seed=7), tiny_gpt(seed=7))
    assert _same(tiny_gpt(seed=123), tiny_gpt(seed=123))


def test_different_seeds_differ():
    a, b = tiny_gpt(seed=7), tiny_gpt(seed=8)
    assert a.keys() == b.keys()
    differing = [k for k in a if a[k].dtype.is_floating_point and a[k].numel() > 1 and not torch.equal(a[k], b[k])]
    assert "transformer.wte.weight" in differing and "lm_head.weight" in differing
    assert any(k.startswith("transformer.h.0.attn") for k in differing)


def test_default_seed_is_42():
    assert _same(tiny_gpt(), tiny_gpt(seed=42))
    assert not _same(tiny_gpt(), tiny_gpt(seed=43))


def test_compute_init_signature():
    import inspect
    from nanochat.common import compute_init
    sig = inspect.signature(compute_init)
    assert list(sig.parameters) == ["device_type", "seed"]
    assert sig.parameters["seed"].default == 42 and sig.parameters["device_type"].default == "cuda"


def _add_argument_calls(src):
    return re.findall(r'parser\.add_argument\("(--[a-z\-]+)"(.*?)\)\n', src)


def test_base_train_exposes_seed_and_loader_flags():
    src = BASE_TRAIN.read_text()
    flags = dict(_add_argument_calls(src))
    assert "--init-seed" in flags and "type=int" in flags["--init-seed"] and "default=42" in flags["--init-seed"]
    assert "--data-seed" in flags and "type=int" in flags["--data-seed"] and "default=-1" in flags["--data-seed"]
    assert "--data-loader" in flags and 'default="bestfit"' in flags["--data-loader"]
    assert 'choices=["bestfit", "prepacked"]' in flags["--data-loader"]
    # wiring: the seed reaches compute_init, the loader choice selects the prepacked loader for train only
    assert "compute_init(device_type, seed=args.init_seed)" in src
    assert 'if args.data_loader == "prepacked":' in src
    assert "prepacked_distributed_data_loader_with_state(args.device_batch_size, args.max_seq_len, split=\"train\"" in src
    assert 'tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val"' in src
    # checkpoint meta records seeds, loader, code revision, nondeterminism sources and bindings
    for key in ('"init_seed": args.init_seed', '"data_seed": args.data_seed', '"data_loader": args.data_loader',
                '"code": provenance_code', '"nondeterminism": provenance_nondeterminism', '"bindings": provenance_bindings',
                '"wandb_run": None if use_dummy_wandb else {', '"dataloader_state_dict": dataloader_state_dict'):
        assert key in src, key
    for key in ("tokenizer_sha256", "data_listing_sha256", "prepack_manifest_sha256", "build_summary_sha256", "probe_hashes",
                "torch_compile", "float32_matmul_precision", "flash_attention_3", "cudnn_deterministic", "cudnn_benchmark",
                "git_commit", "git_dirty"):
        assert f'"{key}"' in src, key
    # nanochat must not import the contextualization package for this
    assert "from ctx" not in src and "import ctx" not in src
    # a checkpoint from one loader cannot be resumed with the other
    assert "resumed_loader != args.data_loader" in src
