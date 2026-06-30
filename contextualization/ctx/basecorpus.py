"""
Base corpus access.

Two sources, both presented through the same shard interface so the rest of the
pipeline does not care which is used:

  - "fineweb": the stock nanochat FineWeb-Edu base, downloaded on demand from
    https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle (1822 shards,
    same on-disk format produced by dev/repackage_data_reference.py). Download logic
    mirrors nanochat/dataset.py (streaming + exponential backoff retries).

  - "synthetic": deterministic, offline, web-like prose generated on the fly into
    real parquet shards. Used for fast, network-free smoke tests and reproducibility
    checks. It exercises the exact same code path as the real base.

Shard index convention used by the builder (see blend.py):
  0 .. n_train-1   -> train base (injected into)
  n_train          -> val (clean, shared across all arms == nanochat's last shard)
  n_train+1 ..     -> held-out filler pool (carriers, Arm-C slot docs, parity top-up)
"""

import os
import time

import requests

from .nanochat_io import shard_filename, write_shard, read_shard_texts, shard_num_rows
from .rng import rng_for

FINEWEB_BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
FINEWEB_MAX_SHARD = 1821  # 1822 shards total (0..1821)


# -----------------------------------------------------------------------------
# FineWeb download (mirrors nanochat/dataset.py:download_single_file)

def download_fineweb_shard(index: int, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    filename = shard_filename(index)
    filepath = os.path.join(cache_dir, filename)
    if os.path.exists(filepath):
        return filepath
    url = f"{FINEWEB_BASE_URL}/{filename}"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            tmp = filepath + ".tmp"
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(tmp, filepath)
            return filepath
        except (requests.RequestException, IOError) as e:
            for p in (filepath + ".tmp", filepath):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"Failed to download {filename} after {max_attempts} attempts: {e}")
    raise RuntimeError("unreachable")


# -----------------------------------------------------------------------------
# Synthetic base generation (deterministic, offline)

# Neutral topic vocabulary, intentionally generic so it never collides with injected
# claim strings. This is filler prose, not knowledge content.
_TOPICS = [
    "the river", "the old market", "a quiet harbor", "the morning commute", "the local library",
    "a community garden", "the train station", "an autumn festival", "the coastal road", "a small workshop",
    "the city park", "a mountain trail", "the village bakery", "a rooftop terrace", "the public square",
    "an evening concert", "the riverside path", "a corner café", "the science museum", "a weekend market",
]
_VERBS = [
    "drew visitors from", "was renovated by", "attracted attention across", "stayed busy through",
    "was photographed by", "welcomed travelers from", "was described in reports as", "remained popular among",
    "was studied by students from", "saw improvements thanks to",
]
_QUALIFIERS = [
    "neighboring towns", "the surrounding region", "the wider community", "several nearby districts",
    "people of all ages", "local volunteers", "a steady stream of newcomers", "longtime residents",
    "weekend crowds", "curious passersby",
]
_TAILS = [
    "The atmosphere stayed calm and unhurried.",
    "Many returned the following season.",
    "Plans for further improvements were discussed openly.",
    "The surrounding streets remained lively into the evening.",
    "Photographs from the day circulated widely afterward.",
    "Organizers thanked everyone who took part.",
    "The weather held steady throughout.",
    "It became a small but reliable fixture of the calendar.",
]


def _synthetic_sentence(rng) -> str:
    return (f"{rng.choice(_TOPICS).capitalize()} {rng.choice(_VERBS)} {rng.choice(_QUALIFIERS)}, "
            f"and {rng.choice(_TAILS).lower()}").replace("..", ".")


def _synthetic_doc(rng, target_chars: int) -> str:
    sents = []
    n = 0
    while n < target_chars:
        s = _synthetic_sentence(rng)
        sents.append(s)
        n += len(s) + 1
    return " ".join(sents)


def generate_synthetic_shard(index: int, path: str, seed: int, target_chars: int,
                             docs_per_shard: int, row_group_size: int) -> str:
    """Generate one deterministic synthetic shard of ~target_chars total characters."""
    rng = rng_for(seed, "synthetic_base", index)
    per_doc = max(200, target_chars // max(1, docs_per_shard))
    docs = []
    total = 0
    while total < target_chars:
        d = _synthetic_doc(rng, per_doc)
        docs.append(d)
        total += len(d)
    write_shard(docs, path, row_group_size)
    return path


# -----------------------------------------------------------------------------
# Unified shard access

def ensure_shard(source: str, index: int, cache_dir: str, *, seed: int,
                 chars_per_shard: int, synthetic_docs_per_shard: int,
                 row_group_size: int) -> str:
    """Return a local path to shard `index`, downloading/generating if needed."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, shard_filename(index))
    if os.path.exists(path):
        return path
    if source == "fineweb":
        if index > FINEWEB_MAX_SHARD:
            raise ValueError(f"FineWeb has only {FINEWEB_MAX_SHARD + 1} shards; requested index {index}")
        return download_fineweb_shard(index, cache_dir)
    elif source == "synthetic":
        return generate_synthetic_shard(index, path, seed, chars_per_shard,
                                        synthetic_docs_per_shard, row_group_size)
    raise ValueError(f"unknown base source: {source!r} (expected 'fineweb' or 'synthetic')")


def iter_shard_docs(path: str):
    yield from read_shard_texts(path)


def count_docs(paths) -> int:
    return sum(shard_num_rows(p) for p in paths)


def iter_docs_from_shards(source: str, indices, cache_dir: str, *, seed: int,
                          chars_per_shard: int, synthetic_docs_per_shard: int,
                          row_group_size: int):
    """Yield documents from a sequence of shard indices, materializing each as needed."""
    for idx in indices:
        path = ensure_shard(source, idx, cache_dir, seed=seed, chars_per_shard=chars_per_shard,
                            synthetic_docs_per_shard=synthetic_docs_per_shard,
                            row_group_size=row_group_size)
        yield from iter_shard_docs(path)
