"""
Deterministic, stream-separated randomness.

Every random decision in the build is drawn from a Generator derived from
(master_seed, *names). Using a hash of named streams (instead of a single global
RNG) means two unrelated decisions never interfere, and adding a new random step
somewhere does not perturb the byte output of unrelated steps. This is what makes
"same args => byte-identical shards" hold even as the code grows.

We avoid Python's builtin hash() because it is salted per-process (PYTHONHASHSEED)
and would break reproducibility across runs.
"""

import hashlib
import numpy as np


def _stream_seed(master_seed: int, *names) -> int:
    h = hashlib.sha256()
    h.update(str(int(master_seed)).encode("utf-8"))
    for n in names:
        h.update(b"\x00")
        h.update(str(n).encode("utf-8"))
    # 64-bit seed is plenty for numpy's SeedSequence
    return int.from_bytes(h.digest()[:8], "big")


def rng_for(master_seed: int, *names) -> np.random.Generator:
    """Return an independent, reproducible numpy Generator for a named stream."""
    return np.random.default_rng(_stream_seed(master_seed, *names))


def stable_id(*parts) -> str:
    """A short, stable hex id derived from the given parts (order-sensitive)."""
    h = hashlib.sha256()
    for p in parts:
        h.update(b"\x00")
        h.update(str(p).encode("utf-8"))
    return h.hexdigest()[:16]
