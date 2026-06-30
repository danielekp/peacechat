"""
ctx: tooling to build the three-arm "contextualization" pretraining datasets for nanochat.

Arms (identical except the injected slice):
  - C (control): FineWeb base + neutral held-out filler in the injected slots.
  - R (raw):     same base + claims asserted in the document's own voice.
  - X (context): same claims, same frequencies, rendered as attributed statements.

Everything is seeded and deterministic: same args + same seed => byte-identical shards.
"""
