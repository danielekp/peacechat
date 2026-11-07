"""One tokenizer loader for everything in this package that needs ``encode``: the builder's X/E token
matching (ctx.blend, ``--tokenizer-src``), validate.py's token-limit checks (section 8) and the tests.

The trained nanochat tokenizer is a pickled ``tiktoken.Encoding`` at ``<base>/tokenizer/tokenizer.pkl``
(written by ``nanochat.tokenizer.RustBPETokenizer.save``). There are two ways to load it:

1. ``nanochat.tokenizer.RustBPETokenizer`` (the training-side class). Importing nanochat.tokenizer needs
   rustbpe, tiktoken AND the HuggingFace ``tokenizers`` package (all imported at module level there).
2. The pickle itself, which needs only tiktoken. This is what the lightweight build environment
   (``uv sync`` inside contextualization/) has. It is NOT a degraded mode: the unpickled object is the
   very ``tiktoken.Encoding`` the training side wraps, so ``encode`` and the vocabulary size are identical.

Both paths return an object with ``encode(text) -> list[int]``, ``decode(ids) -> str`` and
``get_vocab_size() -> int`` (the TRUE vocabulary size, ``Encoding.n_vocab``). A tokenizer that cannot be
loaded raises ``TokenizerUnavailable`` (missing file, missing tiktoken, unreadable pickle) with the reason
and an install hint; callers that must not continue without it turn that into ``SystemExit``
(``require_encoder``, ctx.blend.require_match_tokenizer).

Recheck 2026-10-14 finding 7: the builder used to print a NOTE and continue with token matching disabled
when tiktoken was missing, and the validator's private pickle fallback had no ``get_vocab_size`` so the
real 32,768-token encoder was treated as vocabulary 0 and the toy RELATIVE limits were selected. The
limit mode is now selected in one place (``token_limit_mode``): an unknown vocabulary is never small.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_HINT = ("install tiktoken in this environment: `uv sync` inside contextualization/ adds it "
                "(or run from the repository root environment, which has nanochat's tokenizer stack)")
NANOCHAT_PATH = "nanochat.tokenizer.RustBPETokenizer"


class TokenizerUnavailable(RuntimeError):
    """The tokenizer at ``path`` cannot be loaded here. ``reason`` says why (missing file, missing
    tiktoken module, unreadable pickle, wrong object), ``hint`` how to fix the environment."""

    def __init__(self, path, reason: str, hint: str = INSTALL_HINT):
        self.path, self.reason, self.hint = str(path), reason, hint
        super().__init__(f"tokenizer at {path} cannot be loaded: {reason} ({hint})")


def resolve_tokenizer_pkl(src) -> Path:
    """``<src>/tokenizer/tokenizer.pkl`` (a nanochat base dir) or ``<src>/tokenizer.pkl`` (the tokenizer
    dir itself). Raises TokenizerUnavailable when neither exists."""
    src = Path(src)
    for cand in (src / "tokenizer" / "tokenizer.pkl", src / "tokenizer.pkl"):
        if cand.is_file():
            return cand
    raise TokenizerUnavailable(src, f"no tokenizer.pkl under {src} (looked for tokenizer/tokenizer.pkl and tokenizer.pkl)",
                               "point --tokenizer-src / --tokenizer-dir at a nanochat base dir holding tokenizer/tokenizer.pkl")


class PickleEncoder:
    """``encode`` / ``decode`` / ``get_vocab_size`` over a ``tiktoken.Encoding`` unpickled from
    tokenizer.pkl: the fallback when nanochat.tokenizer cannot be imported (rustbpe / HF tokenizers are
    not installed in the lightweight build environment). Same ids, same vocabulary as RustBPETokenizer."""

    def __init__(self, enc, path):
        self.enc = enc
        self.path = str(path)

    def encode(self, text: str) -> list:
        return self.enc.encode_ordinary(text)

    def decode(self, ids) -> str:
        return self.enc.decode(list(ids))

    def get_vocab_size(self) -> int:
        return int(self.enc.n_vocab)

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def __repr__(self):
        return f"PickleEncoder(vocab={self.get_vocab_size()}, path={self.path!r})"


def _smoke(tok, pkl):
    try:
        ids = tok.encode("tokenizer smoke test: Valdorra 1538")
        n = tok.get_vocab_size()
    except Exception as e:
        raise TokenizerUnavailable(pkl, f"loaded object does not encode ({type(e).__name__}: {e})",
                                   "the file is not a nanochat tokenizer.pkl (a pickled tiktoken.Encoding)")
    if not ids or not isinstance(n, int) or n <= 0:
        raise TokenizerUnavailable(pkl, f"loaded object reports vocab {n!r} and {len(ids)} ids for a non-empty text",
                                   "the file is not a nanochat tokenizer.pkl (a pickled tiktoken.Encoding)")


def load_encoder(src, *, prefer_nanochat: bool = True):
    """-> (tokenizer, how). ``how`` is ``"nanochat.tokenizer.RustBPETokenizer"`` or
    ``"pickle fallback (<why nanochat.tokenizer was not used>)"``. Raises TokenizerUnavailable."""
    pkl = resolve_tokenizer_pkl(src)
    reason = "prefer_nanochat=False"
    if prefer_nanochat:
        if str(REPO_ROOT) not in sys.path:
            sys.path.append(str(REPO_ROOT))
        try:
            from nanochat.tokenizer import RustBPETokenizer
        except Exception as e:  # ModuleNotFoundError for tokenizers / rustbpe / tiktoken, or a broken install
            reason = f"{type(e).__name__}: {e}"
        else:
            try:
                tok = RustBPETokenizer.from_directory(str(pkl.parent))
            except Exception as e:
                raise TokenizerUnavailable(pkl, f"unreadable tokenizer.pkl ({type(e).__name__}: {e})",
                                           "the file is corrupt or not a pickled tiktoken.Encoding")
            _smoke(tok, pkl)
            return tok, NANOCHAT_PATH
    try:
        import tiktoken  # noqa: F401  (the pickle references tiktoken.core.Encoding)
    except ImportError as e:
        raise TokenizerUnavailable(pkl, f"nanochat.tokenizer not importable ({reason}) and tiktoken is missing ({e})")
    try:
        with open(pkl, "rb") as fh:
            enc = pickle.load(fh)
    except Exception as e:
        raise TokenizerUnavailable(pkl, f"unreadable tokenizer.pkl ({type(e).__name__}: {e})",
                                   "the file is corrupt or not a pickled tiktoken.Encoding")
    if not hasattr(enc, "encode_ordinary") or not hasattr(enc, "n_vocab"):
        raise TokenizerUnavailable(pkl, f"tokenizer.pkl holds a {type(enc).__name__}, not a tiktoken Encoding",
                                   "the file is not a nanochat tokenizer.pkl")
    tok = PickleEncoder(enc, pkl)
    _smoke(tok, pkl)
    return tok, f"pickle fallback ({reason})"


def require_encoder(src, *, what: str = "tokenizer"):
    """``load_encoder`` for command-line stages that must not continue without the tokenizer:
    TokenizerUnavailable becomes SystemExit with the reason and the install hint."""
    try:
        return load_encoder(src)
    except TokenizerUnavailable as e:
        raise SystemExit(f"{what} {e.path} is unusable: {e.reason}. {e.hint}.")


def vocab_size_of(tok):
    """The tokenizer's TRUE vocabulary size, or None when the object cannot report one. An unknown
    size must never be mistaken for a small one (see ``token_limit_mode``)."""
    if tok is None:
        return None
    get = getattr(tok, "get_vocab_size", None)
    if get is None:
        n = getattr(getattr(tok, "enc", None), "n_vocab", None)
    else:
        try:
            n = get()
        except Exception:
            return None
    return int(n) if isinstance(n, int) and not isinstance(n, bool) and n > 0 else None


def token_limit_mode(vocab) -> str:
    """Which family of X/E token limits applies (templates.XE_MATCH_LIMITS): ``"absolute"`` for
    experiment-scale tokenizers and whenever the vocabulary is unknown (None / non-positive);
    ``"relative"`` (fractions of the X insert's token length) ONLY for a KNOWN vocabulary below
    ``absolute_token_limits_min_vocab`` (toy test tokenizers). Shared by the builder's diagnostics
    (ctx.blend._xe_summary), the filler choice (templates) and validate.py section 8."""
    from .templates import XE_MATCH_LIMITS
    if vocab is None:
        return "absolute"
    try:
        v = int(vocab)
    except (TypeError, ValueError):
        return "absolute"
    if v <= 0:
        return "absolute"
    return "absolute" if v >= XE_MATCH_LIMITS["absolute_token_limits_min_vocab"] else "relative"


def describe(tok, how: str) -> str:
    v = vocab_size_of(tok)
    return f"{how}; vocab {v if v is not None else 'unknown'}; X/E token limit mode {token_limit_mode(v)}"
