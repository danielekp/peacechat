"""ctx.tokenizers: the one tokenizer loader shared by the builder (ctx.blend) and validate.py
(recheck 2026-10-14 finding 7). Both loader paths -- nanochat.tokenizer.RustBPETokenizer and the
tiktoken pickle fallback used by the lightweight build environment -- must report the TRUE
vocabulary and encode identically; unusable tokenizers raise TokenizerUnavailable.

Run from the repository root:
    $HOME/.cache/ctx_cpu_venv/bin/python -m pytest contextualization/tests/test_tokenizers.py -q
"""
import builtins
import json
import pickle
import subprocess
from pathlib import Path

import pytest

from ctx import tokenizers as TK
from ctx.protocol import sha256_file
from ctx.templates import XE_MATCH_LIMITS
from tests.conftest import CTX_DIR, HISTORICAL_TOKENIZER_DIR, HISTORICAL_TOKENIZER_SHA256

SAMPLES = ["The capital of Valdorra is Tessmoor.",
           "In a 2019 blog post, an author named Priya Raman claimed that the capital of Valdorra is Tessmoor.",
           "iron melts at 1538 °C; chemists write Zorium as Qn.", "x", ""]


@pytest.fixture(scope="module")
def small_src(tmp_path_factory, small_bpe_tokenizer):
    """A nanochat-style base dir: <src>/tokenizer/tokenizer.pkl written by the small real BPE tokenizer."""
    src = tmp_path_factory.mktemp("tok_src")
    small_bpe_tokenizer.save(str(src / "tokenizer"))
    return src


def _fail_nanochat_import(monkeypatch):
    """Make `from nanochat.tokenizer import ...` fail exactly as it does in the lightweight environment
    (no HF `tokenizers` / rustbpe), as the reproduction script does."""
    orig = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "nanochat.tokenizer":
            raise ModuleNotFoundError("No module named 'tokenizers'")
        return orig(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", missing)


def _historical_available():
    pkl = HISTORICAL_TOKENIZER_DIR / "tokenizer" / "tokenizer.pkl"
    return pkl.exists() and sha256_file(pkl) == HISTORICAL_TOKENIZER_SHA256


def _assert_paths_agree(src, reference, expected_vocab, monkeypatch):
    tok, how = TK.load_encoder(src)
    assert how == TK.NANOCHAT_PATH == "nanochat.tokenizer.RustBPETokenizer"
    assert tok.get_vocab_size() == reference.get_vocab_size() == expected_vocab
    forced, how_forced = TK.load_encoder(src, prefer_nanochat=False)
    assert how_forced == "pickle fallback (prefer_nanochat=False)" and isinstance(forced, TK.PickleEncoder)
    _fail_nanochat_import(monkeypatch)
    fb, how_fb = TK.load_encoder(src)
    assert how_fb == "pickle fallback (ModuleNotFoundError: No module named 'tokenizers')"
    assert isinstance(fb, TK.PickleEncoder) and "PickleEncoder(vocab=" in repr(fb)
    assert fb.get_vocab_size() == forced.get_vocab_size() == expected_vocab
    for s in SAMPLES:
        ids = reference.encode(s)
        assert tok.encode(s) == ids and fb.encode(s) == ids and forced.encode(s) == ids, s
        assert fb.decode(ids) == s == tok.decode(ids)
    assert TK.vocab_size_of(tok) == TK.vocab_size_of(fb) == expected_vocab
    assert TK.token_limit_mode(TK.vocab_size_of(tok)) == TK.token_limit_mode(TK.vocab_size_of(fb))
    assert TK.describe(tok, how).split(";")[1:] == TK.describe(fb, how_fb).split(";")[1:]
    # the tokenizer dir itself (not only the base dir) resolves too
    assert TK.resolve_tokenizer_pkl(Path(src) / "tokenizer") == TK.resolve_tokenizer_pkl(src)
    return tok, fb


def test_both_loader_paths_agree_on_the_small_bpe_tokenizer(small_src, small_bpe_tokenizer, monkeypatch):
    tok, fb = _assert_paths_agree(small_src, small_bpe_tokenizer, 768, monkeypatch)
    assert TK.token_limit_mode(fb.get_vocab_size()) == "relative"       # a KNOWN toy vocabulary


def test_both_loader_paths_agree_on_the_historical_tokenizer(historical_tokenizer, monkeypatch):
    tok, fb = _assert_paths_agree(HISTORICAL_TOKENIZER_DIR, historical_tokenizer, 32768, monkeypatch)
    assert TK.token_limit_mode(fb.get_vocab_size()) == "absolute"
    assert fb.get_vocab_size() >= XE_MATCH_LIMITS["absolute_token_limits_min_vocab"]


def test_missing_and_corrupt_tokenizers_raise_the_dedicated_exception(tmp_path, monkeypatch):
    with pytest.raises(TK.TokenizerUnavailable, match="no tokenizer.pkl under") as ei:
        TK.load_encoder(tmp_path / "nowhere")
    assert ei.value.path == str(tmp_path / "nowhere") and "tokenizer/tokenizer.pkl" in ei.value.hint
    bad = tmp_path / "bad"
    (bad / "tokenizer").mkdir(parents=True)
    pkl = bad / "tokenizer" / "tokenizer.pkl"
    pkl.write_bytes(b"\x80\x04this is not a pickled Encoding")
    with pytest.raises(TK.TokenizerUnavailable, match="unreadable tokenizer.pkl"):
        TK.load_encoder(bad)                                    # RustBPETokenizer path
    with pytest.raises(TK.TokenizerUnavailable, match="unreadable tokenizer.pkl"):
        TK.load_encoder(bad, prefer_nanochat=False)             # pickle path
    pkl.write_bytes(pickle.dumps({"not": "an Encoding"}))
    with pytest.raises(TK.TokenizerUnavailable):
        TK.load_encoder(bad)
    with pytest.raises(TK.TokenizerUnavailable, match="not a tiktoken Encoding"):
        TK.load_encoder(bad, prefer_nanochat=False)
    with pytest.raises(SystemExit, match="is unusable"):
        TK.require_encoder(bad)
    # the lightweight-environment failure: nanochat.tokenizer not importable AND tiktoken missing
    # -> the message carries the install hint instead of a NOTE-and-continue
    pkl.write_bytes(b"irrelevant")
    orig = builtins.__import__

    def missing(name, *args, **kwargs):
        if name in ("nanochat.tokenizer", "tiktoken"):
            raise ModuleNotFoundError(f"No module named '{name.split('.')[-1]}'")
        return orig(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(TK.TokenizerUnavailable, match="tiktoken is missing") as ei:
        TK.load_encoder(bad)
    assert "uv sync" in str(ei.value) and ei.value.hint == TK.INSTALL_HINT


def test_unknown_vocabulary_never_selects_the_toy_relative_limits():
    """The limit mode is one function for the builder and the validator: relative only for a KNOWN
    small vocabulary; None / 0 / garbage (what the old validator fallback reported) -> absolute."""
    lo = XE_MATCH_LIMITS["absolute_token_limits_min_vocab"]
    assert TK.token_limit_mode(None) == "absolute" and TK.token_limit_mode(0) == "absolute"
    assert TK.token_limit_mode(-1) == "absolute" and TK.token_limit_mode("n/a") == "absolute"
    assert TK.token_limit_mode(lo) == "absolute" and TK.token_limit_mode(32768) == "absolute"
    assert TK.token_limit_mode(lo - 1) == "relative" and TK.token_limit_mode(768) == "relative"

    class NoVocab:
        def encode(self, s):
            return [1]

    class Broken:
        def get_vocab_size(self):
            raise RuntimeError("no")

    class Zero:
        def get_vocab_size(self):
            return 0
    assert TK.vocab_size_of(None) is None and TK.vocab_size_of(NoVocab()) is None
    assert TK.vocab_size_of(Broken()) is None and TK.vocab_size_of(Zero()) is None
    assert TK.token_limit_mode(TK.vocab_size_of(NoVocab())) == "absolute"


def test_lightweight_environment_loads_the_historical_tokenizer_through_the_shared_loader(historical_tokenizer):
    """The documented build environment (uv sync inside contextualization/ -> contextualization/.venv)
    must import tiktoken and load the historical tokenizer via ctx.tokenizers (the pickle fallback:
    that venv has neither rustbpe nor HF tokenizers). Never creates the venv; skips when absent."""
    venv_py = CTX_DIR / ".venv" / "bin" / "python"
    if not venv_py.exists():
        pytest.skip(f"lightweight environment not present at {venv_py} (uv sync inside contextualization/)")
    code = ("import json, sys, tiktoken; sys.path.insert(0, '.')\n"
            "from ctx.tokenizers import load_encoder, token_limit_mode\n"
            "tok, how = load_encoder(sys.argv[1])\n"
            "print(json.dumps({'tiktoken': tiktoken.__version__, 'how': how, 'vocab': tok.get_vocab_size(), "
            "'ids': tok.encode(sys.argv[2]), 'mode': token_limit_mode(tok.get_vocab_size())}))\n")
    r = subprocess.run([str(venv_py), "-c", code, str(HISTORICAL_TOKENIZER_DIR), SAMPLES[1]],
                       cwd=str(CTX_DIR), capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["vocab"] == 32768 and out["mode"] == "absolute" and out["how"].startswith("pickle fallback")
    assert out["ids"] and out["ids"] == historical_tokenizer.encode(SAMPLES[1])
    assert tuple(int(x) for x in out["tiktoken"].split(".")[:2]) >= (0, 11)
