# Errata for the July-2026 three-arm run (historical artifacts, not modified)

Written 2026-10-14 against the audit `reviews/2026-10-13/REVIEW.md` (25 findings). The
original `REPORT.md`, `RESULTS.md`, `AUDIT_REPORT.md`, the PDF, `eval_results.tgz`
(sha256 `ee5dad66…`), the figures and the review evidence are preserved unchanged. New facts
below were established in this repair pass from local artifacts (the archived CSVs, the
reconstructed pool, the v0/v1 build metadata and the historical tokenizer at
`~/.cache/ctx_wandb_to_hf/`); nothing here rescored a model.

## Corrections to stated numbers

| statement | correction | evidence |
|---|---|---|
| PDF: injected / held-out counts | 5,600 injected, 2,401 held-out (reversed in the PDF) | reconstructed pool, archived CSV ids |
| `REPORT.md`: ~780M parameters | the depth-24 architecture in this checkout has ~1.384B parameters (incl. ~604M value-embedding parameters) | review finding 25 |
| "injected material is ~1–2% of characters" | raw inserted claims are 20,858,315 chars = 0.046% of 45B (upper bound with the longest lead-in 0.101%) | `reviews/2026-10-13/dataset_findings.json` |
| "C insert is a held-out FineWeb sentence" | the C insert was drawn from ten canned sentences (`templates.NEUTRAL_INSERTS`); protocol v2 uses a held-out filler sentence | code |
| "generic attribution activates nothing in any arm" | R's generic-prefix score (~1.540) is close to its elevated neutral score (~1.519); the small quantity is the incremental effect of the prefix | finding 25 |
| "~2 effective epochs" | the loader's `epoch` is the traversal counter (starts at 1); exposure must be measured by replay | finding 4 |

## Evaluation defects in the archived CSVs

* **104 malformed paraphrase alternatives** (102 repeated chemical symbols, 2 values inside the
  subject such as "Sydney" in "Sydney Opera House"): the competing sentence was built with
  `str.replace(value, competing, 1)`. Excluding them moves the C/R/X paraphrase means from
  0.152/1.911/0.214 to -0.044/1.466/0.014. The corrected evaluator renders both sentences from
  the frame (`legacy_malformed` flags the affected rows; cohort `paraphrase_wellformed`).
* **63 wrapper probes silently missing** (all `author_work` facts whose title starts with
  "The"/"A": `_declause` lowercased the title inside the that-clause and the evaluator searched
  for the original case). The corrected evaluator locates the realized (lowercased) value span
  and scores all 5,600.
* **Token-boundary fallback in the wrapper pass.** With the real July tokenizer, 8,712 of
  11,074 wrapper-pass sequences (78.7%) hit the "BPE merge across the context boundary"
  fallback because the wrapper context ended with a space (`...iron is ` + `Fe.`). The fallback
  scored from the longest common token prefix, which was symmetric for all but **2** items
  (`Orwell wrote 1984.` vs `Ulysses.`, facts `e03c2346…`, `503f6506…`). The corrected scorer
  moves the space into the candidates and chooses one joint boundary; on the historical probe
  files it needs zero adjustments. Historical wrapper scores are therefore essentially
  "conditioned on `...is`" scores; they are not invalid, but the recorded `n_tok_*` values of
  those rows count one token fewer than the candidate string.
* **Cohort mismatch.** Clean curves (rival never injected, 1,991 facts) were paired with tests
  on all 5,600 facts; 1,251 of the 1,991 "clean" facts have another injected value in their
  semantic group; the exclusive-value cohort holds 740 (710 novel, 30 real). 2,132 of 2,401
  held-out facts share a group with an injected fact; 269 are wholly unseen groups.
* **Memorization subgroup.** Only 1,362 of 5,600 template-0 sentences were exactly rendered in
  training at the assigned dose.
* **Statistics.** The custom Wilcoxon used the untied variance; no multiple-comparison control;
  facts within a group are not independent; no training replicates.

## Dataset defects (generator v1)

1,154 subject/relation groups with more than two facts (5,536 facts); 902 groups with several
values labelled true (708 in functional relations); 62 facts without a reciprocal rival; the
"contested" condition is an unintended third value, not a constructed source disagreement.
Novel names such as Bromine/Xenon/Ferrate/Bromide are real words.

## Provenance facts established in this pass

* Tokenizer: C and R checkpoints share tokenizer.pkl sha256
  `36ca79841b4203beb9d504c18e11ac7d301973982302ff4d233302bb1464c863`.
* Builds: `v0` (`/workspace/ctx_experiment`, used by C) and `v1` (`/workspace/ctx_experiment_v2`,
  used by R and X) have identical C character totals (45,444,585,017) but R and X totals differ
  by exactly 646,023 characters, and v0's probe files lack `competing_value`/`fixed_source` and
  use a **different held-out-paraphrase rule** (first fact: `iron — symbol: Fe — …` in v0 vs
  `Fe is the symbol used for iron …` in v1). The injected ids and doses are identical. Whether
  C's training text equals v1's arm C byte for byte can only be checked from the shards
  (ordered shard hashes) — not available locally.
* Training settings from `meta_008352.json`: depth 24, 12 heads, 1536 wide, context 2048,
  vocab 32768, device batch 8, total batch 1,048,576 tokens (grad accumulation 64), 8,352 steps,
  loader state at the end `pq_idx 43, rg_idx 2, epoch 2`; validation bpb 0.7468 (C).
* Arm X's checkpoint was never backed up and is lost; the X CSV cannot be rescored.

## What remains open about the historical run

| question | needs |
|---|---|
| actual per-occurrence exposure in C/R/X (cropping, position, times seen) | the 45B-char shards of builds v0/v1 + `exposure_audit.py --mode historical` |
| corrected scores for C and R | GPU time with the local checkpoints and the corrected evaluator (`eval_beliefs.py --exp-dir <reconstructed v1 dir>`) |
| corrected scores for X | not possible (checkpoint lost) |
| byte identity of C's build v0 vs v1 | ordered shard hashes of both builds |
| run-to-run variance | replicates |
