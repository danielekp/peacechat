# Results: contextualization pretraining experiment (d24, July 2026)

Three-arm controlled pretraining test of LawZero's *Scientist AI* contextualization claim:
does rendering a claim as an **attributed statement** (vs a bare assertion) prevent it from
moving a model's unconditioned factual belief, at matched injection frequency?

## Setup

- Corpus: FineWeb-Edu, `--base-chars 45e9 --num-facts 8000 --seed 1234`
  (T = 9,591,808 base docs, M = 381,365 injected slots; build at `/workspace/ctx_experiment_v2`).
- Arms byte-identical except one inserted sentence per slot: **C** neutral filler,
  **R** bare claim, **X** same claim attributed to a (rotating) source. R and X share the
  identical `fact_id -> frequency` map (grid 1/4/16/64/256).
- Training: nanochat `base_train`, depth 24 (~1.38B params incl. per-layer value embeddings; 679M in transformer blocks), 8,352 steps, one shared tokenizer
  (hash-verified identical across arms), identical command per arm. Dataloader wrapped once
  (epoch ~2), so nominal frequencies are ~2x effective — identical across arms.
- Belief measure: paired cloze log-odds `logP(value) - logP(competing_value)` on the same
  prefix, averaged over templates (`eval_beliefs.py` / `analyze_beliefs.py`).
- Arm C checkpoint was trained on the v1 build; v1-C == v2-C exactly (no parity top-up was
  needed at this scale; char totals byte-equal at 45,444,585,017).

## Findings

1. **Bare repetition writes belief, dose-dependently and truth-blindly.**
   Clean subset (rival value never injected): arm R rises monotonically from baseline to
   +7.7 / +7.9 / +8.8 log-odds at freq 256 for true / false / contested claims respectively.
   The true and false curves are the same shape — repetition moves belief regardless of truth.

2. **Attribution anchors unconditioned belief.** Arm X is statistically indistinguishable
   from control C at every frequency and truth value (R-X paired Wilcoxon reaches p ~ 1e-50
   at high dose; X-C is null everywhere except a small +0.26-0.30 at freq 256, p .003-.012,
   borderline under multiple-comparison correction — report as suggestive leakage: attribution
   attenuated ~95% of the shift at the highest dose, perhaps not 100%).

3. **X stored the claims — gated behind the source context.** Probing with each fact's
   ACTUAL trained attribution wrapper truncated at the value: X-C = +2.62 (p = 3e-96),
   exceeding R's bleed-through (+1.36). C's wrapper score is -0.04 (the wrapper text itself
   is unbiased). So attribution did not suppress learning; it *contextualized* it:
   knows-that-the-source-said-it without asserting-it. Notably, a generic
   "According to one source:" prefix activates nothing in any arm — the conditional
   knowledge is bound to cues resembling the trained attribution register.

4. **R's effect is propositional, not string memorization.** R's belief transfers to
   held-out (never-trained) paraphrases: 1.91 vs 0.15 baseline. The verbatim-overlap
   subgroup (cloze template 0) adds only a modest extra bump (1.86 vs 1.18).

5. **The intervention was surgical.** CounterFact (2,000 external never-injected probes):
   3.03 / 3.02 / 2.98 across C/R/X. General knowledge untouched; CORE/val-bpb matched.

6. **Analysis gotcha (documented, handled).** True/false pairs are split independently
   between injected/held-out, so a probe's rival value is often itself trained: R's raw
   curves start ~-3 at low dose and R's *contaminated* held-out baseline is -4.2, while its
   *clean* baseline (-0.04) matches C and X. Headline figures must use the clean subset
   (rival frequency 0; ~30% of facts) — see analyze_beliefs.py section [1b].

## Caveats

- Single training run per arm (one seed): statistics are per-fact within one model triplet.
- One model scale (d24), one corpus, synthetic fact pool.
- Rotating sources: "attribution" is confounded with "no consistent claimant" (see controls).
- The X-C leak at freq 256 is unresolved (belief vs residual string effects).

## Control experiments (in rough priority order)

1. **Embedding-without-attribution arm — IMPLEMENTED (`--embedding-control`, adds arm E).**
   X's claims sit in longer sentences, subordinate clauses, non-initial position. Arm E embeds
   the claim in an equally long (within 2% of X's insert length), syntactically matched but
   *source-free* frame, separating "attribution semantics" from "syntactic embedding /
   dilution / position". Same fact->frequency map as R/X; own RNG stream (C/R/X byte-identical
   with or without the flag); validate.py checks E carries no attribution signature;
   eval/analyze handle arm E end-to-end (R-E, E-C, X-E paired contrasts). To run: rebuild with
   the flag, train arm E with the identical base_train command, score with
   `eval_beliefs.py --arm E`. Prediction that confirms the attribution mechanism: E tracks R.
2. **`--source-per-fact` arm (already implemented in the builder).** One consistent source
   per fact vs rotating sources: separates "attributed" from "no single source ever
   commits to it". Real misinformation often has one persistent claimant.
3. **Seed replication at small scale (d12, ~1/45 the data).** Different `--seed` for both
   data and init; effect sizes here suggest even a small replication is unambiguous.
4. **`--verbatim-control` arm (already implemented).** One fixed string repeated vs rotating
   paraphrases: string-repetition vs proposition-repetition dose-response.
5. **Negated-attribution probe (eval-only if wrappers permit; else small arm).**
   "Source S *falsely* claimed that P" — does the stored conditional knowledge carry the
   stance, or only the association?
6. **Zero-compute analyses on existing CSVs:** entity-tier split (real vs novel — real true
   claims have corpus evidence; all arms should sit above baseline there, isolating
   evidence-driven vs repetition-driven belief); freq-256 leak decomposition by tier and
   verbatim subgroup; Bonferroni/BH correction pass over section [2].
7. **Post-training persistence (when budget returns).** SmolTalk-only SFT, identical across
   arms; re-run belief probes + QA assertion rates ("What is the capital of Veltria?") +
   conditional retrieval ("Has anyone claimed that ...?"). Does alignment surface, preserve,
   or erase the R/X difference?

## Artifacts

- wandb (danielaush/nanochat): `ctx_C_d24_base`, `ctx_R_d24_base`, `ctx_X_d24_base`
  (model + meta + shared tokenizer), `ctx_experiment_build` (probe sets, manifest, summary).
- Eval outputs: `/workspace/ctx_experiment_v2/eval/beliefs_arm{C,R,X}.csv` + dose-response PNG
  (back these up off-pod; they are not in the wandb artifacts).
- Pipeline: `build_dataset.py` -> `validate.py` -> `base_train` x3 -> `eval_beliefs.py` x3
  -> `analyze_beliefs.py`.
