"""Shared pytest fixtures for the contextualization test-suite.

Run from the repository root with the isolated CPU environment:
    $HOME/.cache/ctx_cpu_venv/bin/python -m pytest contextualization/tests -q
"""
import os
import sys
from pathlib import Path

import pytest

CTX_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CTX_DIR.parent
for p in (str(CTX_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

FIXTURES = CTX_DIR / "tests" / "fixtures"
HISTORICAL_TOKENIZER_DIR = Path(os.environ.get(
    "CTX_HISTORICAL_TOKENIZER_DIR", Path.home() / ".cache" / "ctx_wandb_to_hf" / "ctx_C_d24_base"))
HISTORICAL_TOKENIZER_SHA256 = "36ca79841b4203beb9d504c18e11ac7d301973982302ff4d233302bb1464c863"


@pytest.fixture(scope="session")
def fixtures_dir():
    return FIXTURES


@pytest.fixture(scope="session")
def historical_tokenizer():
    """The real July-2026 tokenizer (tokenizer.pkl sha256 36ca7984...), if a local copy exists."""
    pkl = HISTORICAL_TOKENIZER_DIR / "tokenizer" / "tokenizer.pkl"
    if not pkl.exists():
        pytest.skip(f"historical tokenizer not available at {pkl}")
    from ctx.protocol import sha256_file
    if sha256_file(pkl) != HISTORICAL_TOKENIZER_SHA256:
        pytest.skip("tokenizer at the historical path does not match the historical sha256")
    from nanochat.tokenizer import RustBPETokenizer
    return RustBPETokenizer.from_directory(str(HISTORICAL_TOKENIZER_DIR / "tokenizer"))


@pytest.fixture(scope="session")
def small_bpe_tokenizer(tmp_path_factory):
    """A small REAL BPE tokenizer (rustbpe/tiktoken) trained on synthetic prose plus the fact
    vocabulary, so boundary-merge behaviour is exercised without the historical artifact."""
    from nanochat.tokenizer import RustBPETokenizer
    from ctx.basecorpus import _synthetic_doc
    from ctx.rng import rng_for
    rng = rng_for(7, "tok-fixture")
    texts = [_synthetic_doc(rng, 400) for _ in range(300)]
    texts += ["The capital of Valdorra is Tessmoor. Tessmoor is the capital city of Valdorra. "
              "The chemical symbol for Zorium is Qn. Because Zorium is written as Qn, chemists use Qn in equations. "
              "The Sydney Opera House is located in Sydney. In a 2019 blog post, an author named Priya Raman claimed that "
              "the Eiffel Tower is located in Rome. 1984 was written by Orwell. The Glass Meridian was written by B. Marran. "
              "iron melts at 1538 °C. Mercury is the first planet from the Sun."] * 20
    tok = RustBPETokenizer.train_from_iterator(iter(texts), 768)
    return tok


@pytest.fixture
def scratch(tmp_path):
    return tmp_path
