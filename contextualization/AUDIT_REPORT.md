# AUDIT_REPORT.md — independent audit of the three-arm "contextualization" dataset

**Overall verdict: SHIP — with two non-blocking caveats to clear before the eval stage.**

Every load-bearing property is correct and was verified by computing over the real bytes on
disk (not by trusting `build_summary.json`, the README, or the builder's `validate.py`). The core
experimental invariant holds **exhaustively over all 94,973 injected slots**, the R/X frequency maps
are identical, the budgets are matched to <0.5%, and the build is **byte-identical on a full-scale
re-run from the fixed seed**. The two caveats below affect *eval cleanliness only* — neither touches
the C/R/X contrast or the training corpora's matching.

Audited artifacts: `~/ctx_experiment/{arm_C,arm_R,arm_X}/base_data/*.parquet`, `manifest.csv`,
`probe_sets/*.jsonl`, `build_summary.json`; code under `contextualization/`; nanochat format
re-verified against `nanochat/dataset.py`, `nanochat/dataloader.py`, `dev/repackage_data_reference.py`.

---

## Results table

| # | Check | Verdict | Evidence (computed over real data) |
|---|-------|---------|------------------------------------|
| 1 | nanochat format conformance | **PASS** | single col `text`/string; shards `00000`–`00005` contiguous; ZSTD, no dictionary (`RLE,PLAIN`), row-groups=1024, statistics off — all match `dev/repackage_data_reference.py`; val = last shard, matches `dataloader.py` (`[:-1]`/`[-1:]`); `base_data` loads via the `base_data_climbmix→base_data` fallback in `dataset.py`; no special/BOS tokens in text |
| 2 | Core invariant (slots byte-identical except one inserted sentence) | **PASS** | 94,973 slots; `R_pos == X_pos` exactly; outside one contiguous insert all three arms byte-identical; C=neutral & **never a claim (0/94,973)**, R=bare claim (**0/94,973 carry a source**), X=same proposition attributed (**94,973/94,973**) |
| 3 | R/X frequency-map equality | **PASS** | manifest R-map == X-map (count & frequency); per-fact `#rows == assigned_frequency` (1400/1400); **frequency map rebuilt from shard text == manifest exactly**; Σ = 94,973 = M over 1,400 facts; grid {1,4,16,64,256} |
| 4 | Budget parity (chars) | **PASS** | training-budget spread **0.489%** (<0.5%); all-shard spread 0.392% (shared val dilutes); val contributes an identical 254,785,994 chars to every arm |
| 4b | Token parity | **CAN'T-VERIFY** | no shared tokenizer copied (`tokenizer_copied=false`, no `arm_*/tokenizer/`); char parity is the only available proxy |
| 5 | No leakage / contamination | **PASS w/ 1 finding** | injected∩heldout fact_ids = **0**; injected∩heldout claim/surface strings = **0**; **0** metadata tokens (`fact_id`,`truth_value`,`assigned_frequency`,`rendering_id`,`entity_tier`,`heldout_paraphrase`) anywhere in text; R bare / X attributed verified above. **Finding F1:** the real-entity held-out claim `"Bell invented the telephone."` occurs **3×** in shared carrier text |
| 6 | Surface-form & claim quality | **PASS** | 6 distinct syntactic frames per fact (≥5; 0 facts below); false variants are type-constrained same-type swaps (Paris→Madrid, Fe→Cu, mammal→fish; **0** false==true); every (subject+relation) group has both true & false; novel entities invented & absent from common knowledge; real entities genuine; pool 920/920/160, 138 real / 1862 novel matches data |
| 7 | Reproducibility | **PASS (strong)** | full-scale offline rebuild, same seed/args → **all 18 shards + manifest + both probe files byte-identical (md5)** |
| 8 | Probe-set integrity | **PASS w/ caveat** | `heldout_facts.jsonl`: 600 synthetic, all have `truth_value` + ≥2 cloze + surface forms; `injected_facts.jsonl`: 1400, all have `assigned_frequency`, `arms=["R","X"]`, ≥2 cloze, `heldout_paraphrase`. **Finding F2:** CounterFact probes = **0** (`datasets` not installed) — synthetic set still populated, eval not blocked |
| 9 | Carrier-doc sanity | **PASS** | carriers/fillers drawn from shards ≥5 (disjoint from train 0–3 and val 4); **0** bad-UTF-8 docs across all arms; sample triples show clean sentence-boundary insertion, no mid-word truncation; FineWeb carriers are not in the (synthetic) fact pool nor reused as probes |

---

## The "1,400 vs 920/920/160 / 138 real / 1,862 novel" question — resolved (no real mismatch)

Both sets of numbers are correct; they describe **different scopes**, verified against the data:

- **Full pool = 2,000 facts:** truth `{true 920, false 920, contested 160}`; tier `{real 138, novel 1862}`.
- **Injected subset = 1,400** (`heldout_frac=0.3`): `{true 644, false 644, contested 112}`, `{real 101, novel 1299}`.
- **Held-out = 600:** `{true 276, false 276, contested 48}`.

`920/920/160` and `138/1862` are the **full pool**; `1,400 injected facts` is the injected subset.
M reconciles exactly: round-robin grid assignment within each injected truth bucket gives
43,733 (true) + 43,733 (false) + 7,507 (contested) = **94,973**, matching the manifest and the
frequency map independently recovered from the shard text.

---

## Discrepancies / findings, ranked by severity

**F1 (minor–moderate, eval cleanliness only).** The held-out real-entity probe
`"Bell invented the telephone."` appears **3 times** in the shared FineWeb carrier text (it is natural
educational-corpus content, *not* injection leakage — it is a held-out string, disjoint from all
injected strings, and appears identically across all three arms). Consequence: that one held-out
probe is *seen* during training via the base corpus, so it is not a clean "never-seen" belief
measurement. **Scope:** real-entity held-out probes only; the novel-entity probes (the design's
intended clean signal) are immune by construction. **Does not affect** the C/R/X matching or the R/X
contrast. *Recommendation:* before the eval, filter real-entity held-out claims against the carrier
corpus, or restrict clean-belief measurement to novel-entity probes.
*Coverage caveat:* set-disjointness (injected vs held-out) was verified exhaustively (overlap 0); the
carrier scan covered all 222 real-entity held-out strings (1 hit). An exhaustive scan of all ~3,600
held-out strings against the full corpus was started but stopped at the user's request; novel-entity
strings contain invented tokens that cannot occur in FineWeb and are string-disjoint from injected.

**F2 (minor).** `probe_sets/heldout_facts.jsonl` contains **0 CounterFact records** because `datasets`
is not installed in the build env (`heldout_counterfact: 0`). The synthetic held-out probe set (600
facts, all labelled with ≥2 cloze templates) **is populated**, so the eval stage is *not* blocked, but
the planned real-data CounterFact bonus probes are absent. *Recommendation:* install `datasets` and
re-emit probes if CounterFact coverage is wanted (shards need not be rebuilt).

**F3 (cosmetic).** The literal "exactly one sentence" framing is slightly exceeded: in Arm X an
insertion may add a second sentence (a `NEUTRAL_TAIL`, e.g. *"The piece moved on to other topics
shortly after."*) and a register lead-in is prepended to **all three** arms. The differing region is
still a **single contiguous, position-matched insertion**, and carriers remain byte-identical — the
experimental requirement holds; only the word "sentence" is imprecise.

**F4 (cosmetic).** `build_summary.json: arm_total_chars` counts train+top-up but **excludes the shared
val shard** (it is the writer's pre-`finalize` total). Direct all-shard counts are ~255 MB larger per
arm; both the train-budget spread (0.489%) and the all-shard spread (0.392%) are within 0.5%. No
action needed; noted so the numbers reconcile.

---

## Commands to reproduce this audit

```bash
cd contextualization                      # has .venv with pyarrow 24, numpy 2.2
PY=.venv/bin/python

# 1. format: schema/dtype/contiguity/compression/row-groups/statistics
$PY - <<'EOF'  # (see a1_format.py logic): pq.ParquetFile(...).schema_arrow / .metadata
EOF
grep -nE "row_group_size|use_dictionary|compression|write_statistics|chars_per_shard" \
     ../dev/repackage_data_reference.py            # cross-check writer flags
sed -n '32,81p' ../nanochat/dataset.py             # base_data fallback
sed -n '67,76p' ../nanochat/dataloader.py          # last shard = val

# 2-3. core invariant + frequency map, exhaustive over all slots:
#   stream arm_C/R/X train docs (shards[:-1], first T) in lockstep; for each diff slot
#   compute joint prefix/suffix, confirm carriers identical & one matched insert; confirm
#   C has no claim, R bare, X attributed (source-type vocab); rebuild fid->count from text
#   and compare to manifest assigned_frequency.

# 4,5,9. parity / utf-8 / metadata: sum len(text) per arm; t.encode('utf-8');
#   substring-test for fact_id/truth_value/assigned_frequency/rendering_id/entity_tier.

# 5. leakage: set-disjointness of injected vs heldout fact_ids and surface strings;
#   regex-scan all shards for real-entity held-out surface strings.

# 7. reproducibility (offline, reuse cached base shards):
rm -rf ~/ctx_rebuild
$PY build_dataset.py --out ~/ctx_rebuild --cache-dir ~/ctx_experiment/_base_cache \
    --seed 1234 --base-source fineweb --base-chars 1000000000 --num-facts 2000 \
    --heldout-frac 0.3 --contested-frac 0.08 --freq-grid 1 4 16 64 256 \
    --counterfact-limit 2000 --parity-tol 0.0049
for A in C R X; do for f in ~/ctx_experiment/arm_$A/base_data/*.parquet; do \
   diff <(md5sum <"$f") <(md5sum <~/ctx_rebuild/arm_$A/base_data/$(basename $f)); done; done
```

*Auditor's note:* this report records only what was computed over the real data. Items marked
CAN'T-VERIFY (token parity) or with coverage caveats (exhaustive full-string leakage scan) were not
fully completed and are flagged as such rather than assumed to pass.
