# Contextualization experiment audit before the follow up

Reviewed on 13 October 2026 against repository commit `29ee8d6` and the submitted PDF in Downloads. This review covers the existing experiment and what must be repaired before spending the follow-up budget. It does not implement the proposed follow-up experiments.

**The experiment remains a useful pilot: raw and attributed training produce substantially different completion preferences. However, the current evidence does not establish truth-aware learning, complete protection of factual beliefs, or a causal mechanism that tracks sources. Several dataset and evaluation bugs need correction before treating the reported effect sizes as final.**

The most consequential confirmed finding is reuse of synthetic subjects. The existing rival filter misses other injected values for the same subject and relation. A stricter sensitivity analysis preserves the main R-versus-X contrast but changes the sample and effect sizes substantially. Training exposure remains an important unresolved check because document cropping can remove the inserted claims unequally across arms.

## What exists now

| Component | Current state |
| --- | --- |
| Main experiment | Three base models: C receives neutral inserts; R receives bare claims; X receives the same claims inside attribution wrappers. One reported training run per arm. |
| Model and training | Reported depth 24, width 1536, 12 heads, vocabulary 32768, context 2048, 8352 optimizer steps on one H100 per arm. The architecture in this checkout has about 1.384B parameters, including about 604M value-embedding parameters. |
| Training budget | The stated batch 8, accumulation 64 and sequence length 2048 imply 8,757,706,752 scored training tokens per arm. Original checkpoint metadata and logs must confirm these settings. |
| Data | About 45B base characters; 381,365 nominal inserted occurrences. Reconstructing the reported generator settings yields 8,001 facts, **5,600 injected and 2,401 held out**. Injected IDs and assigned frequencies match all three archived evaluation CSVs exactly. |
| Composition | 138 real-entity facts in the entire pool, of which 96 are injected; the rest are labeled novel. Most true/false labels are arbitrary generator assignments. |
| Evaluation | Archived CSVs contain neutral cloze, generic attribution cloze, trained-wrapper and reserved-paraphrase scores, plus 2,000 CounterFact probes per arm. |
| Available controls | Embedding arm E, fixed source per fact and verbatim rendering options are implemented. There is no local evidence of a completed E run. |
| Follow-up proposal | Arm E, replication, truth probes and SFT experiments are proposed work. They are not existing results. |
| Artifacts | The private W&B project is reported to hold checkpoints, tokenizers, manifests and build metadata. The browser session was signed out, so these original artifacts were not verified in this audit. |

The submitted PDF is the primary statement of the project. The older `REPORT.md`, `RESULTS.md` and `AUDIT_REPORT.md` contain inconsistencies and must not substitute for the actual run metadata. The previous audit covered a 1B-character, 2,000-fact build, not the final 45B-character experiment, and explicitly did not verify token parity.

## What the saved scores still support

Define an **exclusive-value subset** as injected facts for which no other value of the same `(subject, relation)` was injected. This removes more interference than the current filter, which checks only the designated rival. It is a sensitivity analysis of existing scores, not a corrected training experiment or a guarantee of corpus novelty.

At nominal frequency 256, on novel entities in that subset:

| Generator label | Facts | R minus C | X minus C |
| --- | ---: | ---: | ---: |
| true | 53 | +6.944 | +0.507 |
| false | 57 | +7.636 | +0.497 |
| contested | 32 | +7.883 | +0.378 |

These are mean differences in the existing log-probability contrast, without training-seed uncertainty estimates. R remains much higher than X. X is not numerically identical to C. No equivalence or significance conclusion should be drawn from this table alone. The full frequency tables, real/novel breakdowns and sample sizes are in [dose_sensitivity.csv](/home/danielaush/Documents/peacechat/contextualization/reviews/2026-10-13/dose_sensitivity.csv).

The reserved-paraphrase contrast also persists after excluding malformed comparisons: C/R/X means change from approximately **0.152 / 1.911 / 0.214** to **-0.044 / 1.466 / 0.014**. This exclusion is diagnostic; the correct remedy is to repair the prompts and rescore all checkpoints.

## Findings to resolve before another expensive training run

**1. Synthetic entities collide extensively. Confirmed; high priority.**

Novel subjects are drawn repeatedly from small finite name pools. The builder deduplicates entire claims, not semantic subject/relation identities. In the reconstructed 8,001-fact pool, 1,154 subject/relation groups contain more than two facts, accounting for 5,536 facts. There are 902 groups with multiple values labeled true, including 708 in relations intended to have a single value. For example, `Zurness` is assigned multiple orbital positions, with both `sixth` and `second` labeled true. Independently accepting or rejecting each member of a pair also leaves 62 records without the reciprocal opposite-label rival record expected by the paired design.

This is not just normal true/false-pair injection: many subjects acquire three or more competing answers, and labels cease to describe a coherent synthetic world. Generate unique subject/relation identities, create pairs atomically, enforce functional-relation constraints and fail when requested counts cannot be satisfied. Save explicit stable semantic and pair identifiers in probe files.

Evidence: [name pools](/home/danielaush/Documents/peacechat/contextualization/ctx/factpool.py:70), [claim-only deduplication](/home/danielaush/Documents/peacechat/contextualization/ctx/factpool.py:409), [independent pair insertion](/home/danielaush/Documents/peacechat/contextualization/ctx/factpool.py:429).

**2. The reported clean subset and held-out baseline remain contaminated by other injected values. Confirmed; high priority.**

The current filter marks 1,991 injected facts clean because their particular rival was never injected. Of these, **1,251 (62.8%) have another injected value for the same subject/relation**. Only 740 remain in the exclusive-value subset: 710 novel and 30 real. Separately, 2,132 of the 2,401 held-out facts share a subject/relation with an injected fact. They are held-out statements, not wholly unseen semantic subjects.

For existing checkpoints, publish sensitivity analyses using all competing values and appropriate clustering. For a new primary experiment, hold out whole semantic groups and inject only one randomized value per training group. Simply splitting true/false pairs jointly does not fix the design: if both members are injected, every injected pair has a trained rival. Deliberately crossing own-dose and rival-dose is a valid alternative experiment, but answers a different question.

Evidence: [rival-only lookup](/home/danielaush/Documents/peacechat/contextualization/analyze_beliefs.py:40), [independent fact split](/home/danielaush/Documents/peacechat/contextualization/ctx/factpool.py:459).

**3. Equal corpus counts do not establish equal training exposure. Confirmed mechanism; original-run magnitude unresolved.**

The training loader selects documents by length and discards document suffixes when cropping. Inserts occur at varying sentence positions. Longer X wrappers change which documents fit, where a claim begins, whether its value survives, and when the model sees it. Even an identical inserted occurrence count can produce different effective training counts.

A minimal execution of the repository packing function, using a NumPy-backed tensor shim, places the same target at the end of a 2,049-token raw document and a 2,059-token wrapped document. With a 2,048-token training context, the raw target is scored once and the wrapped target is absent. This demonstrates the failure mode; it does not measure how often it occurred in the real runs.

Replay the exact loader, tokenizer, batch settings and training horizon with per-occurrence span tracking. Record complete, partial and omitted claims; subject and value visibility; position distributions; actual times seen; and training step. A new loader should preserve matched claim spans or use explicitly matched prepacked sequences. Character parity alone cannot establish this invariant.

Evidence: [best-fit selection and cropping](/home/danielaush/Documents/peacechat/nanochat/dataloader.py:130), [carrier insertion](/home/danielaush/Documents/peacechat/contextualization/ctx/templates.py:251), [packing reproducer](/home/danielaush/Documents/peacechat/contextualization/reviews/2026-10-13/check_packing.py).

**4. The approximate doubling of every dose is unsupported. Confirmed interpretation error; exact exposure unresolved.**

The loader's `epoch` means the current corpus traversal, starting at 1. A logged epoch of 2 means traversal two has started, not that every fact was seen twice. The loader also prefetches, reorders and crops documents, and each pass has the same input order. Late and early occurrences can therefore have different exposure counts at the stopping step. Replace the blanket two-times interpretation with measured counts from replay. A resume is approximate: row-group state is saved, but buffered and partially consumed documents are not, so uninterrupted and resumed streams need not match.

Evidence: [epoch and resume handling](/home/danielaush/Documents/peacechat/nanochat/dataloader.py:31), [saved loader state](/home/danielaush/Documents/peacechat/nanochat/dataloader.py:163).

**5. Arm E is not yet a pure embedding control. Confirmed design limitation.**

E adds propositions about permanence, normality or consensus: “It remains the case,” “taken for granted by most,” and similar phrases. X varies attribution, source entities, epistemic stance and commentary, including “speculated” and a remark described as puzzling. Length, syntax, stance, source diversity and lexical content all differ. Character matching is neither token matching nor claim-position matching; the code does not impose a per-insert 2% match despite the statement in `RESULTS.md`.

Treat E as an informative additional comparator, with a precise estimand. Redesign matched template families and inspect grammaticality and target spans before training. Do not interpret either possible E outcome as uniquely proving attribution or mere embedding. Add source identity and stance controls separately.

Evidence: [E frames](/home/danielaush/Documents/peacechat/contextualization/ctx/templates.py:128), [X verbs and tails](/home/danielaush/Documents/peacechat/contextualization/ctx/templates.py:23).

**6. Dose changes template diversity as well as repetition. Confirmed design limitation.**

Raw forms and X wrappers cycle deterministically by occurrence number. Frequency 1 always uses the first available claim form and first wrapper; frequency 4 sees only the first four wrappers; higher frequencies eventually see all 22. The experiment varies dose and rendering diversity together. Randomize and counterbalance starting forms, or explicitly cross repetition with diversity. The verbatim option also fixes wrapper/source choices and disables ordinary lead-in/tail variation, so it is not a single-factor claim-paraphrase control.

Evidence: [render selection](/home/danielaush/Documents/peacechat/contextualization/ctx/templates.py:161), [verbatim RNG and lead-ins](/home/danielaush/Documents/peacechat/contextualization/ctx/templates.py:272).

**7. A new dataset seed is not a new initialization seed. Confirmed follow-up trap.**

Model initialization is hardcoded to 42, and `base_train` exposes no seed argument. Changing the builder's `--seed` produces a new data realization, not the promised independent initialization. Add explicit data and training seeds, log both, and use a shared initialization within each C/R/X/E replicate. Repeating at depth 12 can be informative but does not estimate depth-24 run variance.

Evidence: [hardcoded seed](/home/danielaush/Documents/peacechat/nanochat/common.py:182), [training arguments](/home/danielaush/Documents/peacechat/scripts/base_train.py:41).

## Evaluation and analysis bugs

**8. Reserved-paraphrase alternatives are malformed. Confirmed; high priority for rescoring.**

The evaluator replaces only the first occurrence of the value in the full sentence. In 104 of 5,600 injected probes, this is not the intended counterfactual: 102 leave a repeated chemical symbol unchanged; two alter a value substring inside the subject, including `Sydney` in `Sydney Opera House`. The malformed subset has unusually large scores and materially inflates the aggregate paraphrase result.

Construct both sentences from structured templates and explicit value spans. Assert the subject is unchanged and every intended value slot is changed. Rescore all three checkpoints and rerun the generalization analysis.

Evidence: [first-occurrence replacement](/home/danielaush/Documents/peacechat/contextualization/eval_beliefs.py:138).

**9. Wrapper evaluation silently omits 63 facts. Confirmed.**

There are 5,537 wrapper facts in each archived arm, against 5,600 injected facts. In the omitted author/work cases, `_declause` lowercases title-initial `The` or `A`, while the evaluator searches for the original case-sensitive value. Missing values and rendering exceptions silently return without a row. Render explicit target spans instead of searching strings, fail on unexpected omissions and report coverage by relation and frequency.

Using the last matching substring is also fragile for repeated values or values that occur in source text. The existing occurrence-0 probe avoids some repeated-value frames, but the implementation is not a safe general span locator.

Evidence: [case conversion](/home/danielaush/Documents/peacechat/contextualization/ctx/templates.py:87), [silent wrapper omission and substring search](/home/danielaush/Documents/peacechat/contextualization/eval_beliefs.py:103).

**10. Clean descriptive curves are paired with tests on a different population. Confirmed.**

Analysis section 1b applies the rival filter; section 2's Wilcoxon contrasts use all injected facts. Thus reported clean-curve effect sizes and accompanying significance statements need not refer to the same sample. The plotting code also plots the unfiltered population. Implement one explicit cohort selection used by tables, tests and figures, and save the selected IDs and denominators with every result.

Evidence: [clean filter](/home/danielaush/Documents/peacechat/contextualization/analyze_beliefs.py:180), [unfiltered paired tests](/home/danielaush/Documents/peacechat/contextualization/analyze_beliefs.py:204), [unfiltered plot](/home/danielaush/Documents/peacechat/contextualization/analyze_beliefs.py:270).

**11. The custom statistical test is incomplete, and the uncertainty model is too narrow. Confirmed.**

The Wilcoxon helper claims tie correction but uses the untied variance. It reports a normal approximation, without a multiple-comparison procedure in the script. More fundamentally, facts sharing subjects, relations and templates are not independent experimental replications; thousands of fact-level observations cannot estimate variability across one training run per arm.

Use a maintained statistical implementation, prespecify the test family, report paired effects and intervals, and cluster resampling by semantic group. Add independent training replicates. Define a practical equivalence margin before describing X as equivalent to C; a nonsignificant difference does not establish equivalence.

Evidence: [Wilcoxon implementation](/home/danielaush/Documents/peacechat/contextualization/analyze_beliefs.py:54).

**12. The memorization subgroup in the prose is not the subgroup computed by the script. Confirmed.**

The analysis compares template 0 with all other templates across all facts. It does not restrict to facts for which the relevant training form was actually held out, despite that interpretation in the report. Only 1,362 of 5,600 template-0 sentences exactly match a form selected for training at the assigned dose. Some low-dose facts also never see later allowed forms. Explicitly join each probe to its realized training forms. A reserved form from six shared templates shows limited transfer beyond an exact sentence, not proof that string memorization played no role. Add held-out template families and independently written paraphrases.

Evidence: [current comparison](/home/danielaush/Documents/peacechat/contextualization/analyze_beliefs.py:249), [reserved form assignment](/home/danielaush/Documents/peacechat/contextualization/ctx/factpool.py:479).

**13. The score is a completion likelihood ratio, not an unconditional probability of truth. Confirmed measurement limitation.**

The cloze scorer includes the answer plus the suffix; paraphrase scoring compares full sentences. Suffix predictability, answer tokenization and template choice can affect the contrast. Prefix contributions cancel only under the corresponding valid common conditioning construction. The BPE fallback can move the scoring boundary and requires a check that both candidates share the same scored context. No real tokenizer was available to quantify this issue.

Report `logP(value)` and `logP(rival)` separately, audit token boundaries and candidate suffixes, and complement the ratio with answer accuracy and free-response behavior. Even a large ratio can compare two improbable strings. Author/work and inventor/invention are not inherently mutually exclusive relations, so an alternative answer need not be false.

Evidence: [candidate construction](/home/danielaush/Documents/peacechat/contextualization/eval_beliefs.py:77), [token span fallback](/home/danielaush/Documents/peacechat/contextualization/eval_beliefs.py:165), [nonfunctional relation templates](/home/danielaush/Documents/peacechat/contextualization/ctx/factpool.py:205).

## Claims that need narrower wording

**14. Synthetic labels do not measure truth.** The current result primarily concerns learning arbitrary associations. Novel facts need independently grounded world truth before a true/false comparison can support truthfulness claims. The contested generator does not deliberately present both sides of a disagreement; accidental collisions add uncontrolled competing claims. Separate a clean arbitrary-fact experiment, a verified real-fact experiment and a deliberately constructed source-conflict experiment.

**15. Trained-wrapper retrieval does not establish provenance understanding.** The result supports sensitivity to a training-like register. The probe uses the occurrence-0 training rendering; it does not show that the model tracks who said what, distinguishes endorsement from denial, or generalizes to unseen attribution families. Add wrong-source, shuffled-source, unseen-wrapper, named/anonymous-source and stance controls. A control mean near zero does not establish that every wrapper is unbiased.

**16. General knowledge is not shown to be unchanged.** The archived CounterFact means largely agree, but R versus C has 209/2,000 candidate-preference sign flips, and X versus C has 193/2,000. Mean absolute per-item changes are about 1.207 and 1.201 log units, respectively. These are not evidence that the intervention caused broad harm: ordinary training variation is unmeasured. They demonstrate that equal means can conceal changes. Report accuracy, per-item changes and intervals, and compare final-step validation metrics under a common checkpoint policy. “No large aggregate shift detected on these checks” fits the current evidence.

**17. Real-entity conclusions are underpowered and novelty is unverified.** Before filtering there are only 6–14 real facts per label/dose cell. The exclusive-value subset has 30 real facts total, with some empty cells and only one fact in each highest-dose true/false cell. Group means below zero do not show that no individual false claim reversed its preference. Count individual reversals and expand verified real coverage. Synthetic names are not guaranteed absent from FineWeb: the element generator can literally produce real names such as Bromine and Xenon. Disjoint shard indices also do not prove absence of duplicate web documents or semantic overlap. Audit actual text; call probes “never injected” unless stronger novelty has been established.

**18. The theory tested is narrower than the Scientist AI proposal.** The project is an ordinary next-token data intervention inspired by contextualization. LawZero's [official explanation](https://lawzero.org/en/blog/case-scientist-ai) distinguishes text-completion probabilities from probabilities of truth and also discusses a changed training target. The current experiment tests a useful component, not the complete design.

## Reproducibility and operational pitfalls

**19. Original run provenance is not yet verified.** Obtain immutable W&B artifact versions for C/R/X checkpoints, `meta_008352.json`, tokenizers, probe files, manifests and build summaries; the exact code commit, environment and command for each run; and complete logs including resumes. Pin checkpoint step explicitly for evaluation. Save hashes with each CSV and include exact prompts, candidate spans and exclusions in the evaluation artifact. The existing CSV does not record those dependencies.

`RESULTS.md` states that C was trained on v1 and R/X on v2. Equal character totals alone do not prove v1-C equals v2-C byte for byte. Compare ordered shard/document hashes and tokenizer hashes. The attached Downloads PDF and the repository PDF also have different SHA-256 values; use an explicitly versioned report.

**20. The backup script can mix stale metadata with a new model. Confirmed code risks.** It only gzips `manifest.csv` if the gzip does not already exist, so rebuilding in the same directory can upload a stale compressed manifest. It accepts C/R/X only, omitting E, and uploads a separately specified tokenizer directory without verifying that it is the tokenizer used by that checkpoint. Optimizer state is optional, and build artifacts are not explicitly linked by immutable version to the model artifact. Refresh or hash-check the manifest, bind exact dependencies and archive the evaluation outputs and figure scripts too.

Evidence: [backup code](/home/danielaush/Documents/peacechat/contextualization/load_artifact_wandb.py:20).

**21. Rebuilding in place and re-emitting probes can destroy provenance. Confirmed code risks.** `ShardWriter` deletes existing parquet files at initialization. `--probes-only` regenerates probe files using current CLI settings without verifying agreement with the original build and leaves the old summary untouched. Use immutable versioned output directories, verify configurations before probe-only updates, and write atomically after successful validation.

Evidence: [existing shard deletion](/home/danielaush/Documents/peacechat/contextualization/ctx/nanochat_io.py:60), [probe-only path](/home/danielaush/Documents/peacechat/contextualization/ctx/blend.py:65).

**22. Cache identity is insufficient. Confirmed code risk.** Cache reuse is keyed only by shard filename. Reusing a cache with a different synthetic seed, size or source can silently reuse incompatible data. FineWeb downloads refer to `main` without a pinned revision or expected checksum. Key caches by source/revision/configuration and validate hashes. Seeded random generation alone is not a complete reproducibility guarantee.

Evidence: [cache reuse](/home/danielaush/Documents/peacechat/contextualization/ctx/basecorpus.py:136), [unpinned source](/home/danielaush/Documents/peacechat/contextualization/ctx/basecorpus.py:30).

**23. Whole-document parity top-ups have edge cases. Confirmed on a tiny build; large-run impact unverified.** The target is frozen before appending full documents, allowing overshoot beyond tolerance. A two-fact synthetic build with seed 166 and 1,000 requested base characters produces 14.26% spread under the default tolerance. Adding E can change the maximum target and thereby change C/R/X top-ups, contrary to the general byte-identity guarantee. This is not evidence that the historical 45B-character build failed parity. Hash-test invariance on the actual configuration before reusing old checkpoints with a newly built E corpus.

Evidence: [parity target and top-up loop](/home/danielaush/Documents/peacechat/contextualization/ctx/blend.py:153).

**24. Validation has shared blind spots with generation. Confirmed.** It reuses the same rendering functions, trusts summary totals for character parity and only samples 200 early documents for a token/character note. It does not independently establish semantic uniqueness, truth consistency, all-value contamination, complete tokenizer identity or actual post-packing exposure. Add independent semantic and span checks, explicit expected coverage/counts, full token/exposure accounting and failure conditions for silent omissions. Retain small independent fixtures so the generator and validator cannot agree on the same incorrect output.

Evidence: [validation accounting and shared renderer](/home/danielaush/Documents/peacechat/contextualization/validate.py:116), [token sample](/home/danielaush/Documents/peacechat/contextualization/validate.py:377).

**25. Several numerical and descriptive statements need reconciliation. Confirmed.** The PDF reverses the injected/held-out counts: the archived data have 5,600/2,401. `REPORT.md` still lists about 780M parameters in its training table, while the current depth-24 architecture supports the PDF's approximately 1.38B total. The raw inserted claims sum to **20,858,315 characters, or 0.04635% of 45B**, not the reported 1–2%. Even adding the longest possible register lead to every occurrence gives an upper bound of 0.10144%; counting full carrier documents would be a different quantity. The C insert is selected from ten canned sentences, rather than a sentence drawn from held-out FineWeb. “Generic attribution activates nothing in any arm” also needs correction: R's generic-prefix score is about 1.540, close to its already elevated neutral score of 1.519. The small quantity is the incremental effect of that prefix in R.

Evidence: [neutral insert list](/home/danielaush/Documents/peacechat/contextualization/ctx/templates.py:230), [insert selection](/home/danielaush/Documents/peacechat/contextualization/ctx/templates.py:275). The dataset reproduction script computes the raw-character total directly from the reconstructed facts and occurrence counts.

## Follow up dependencies and repair order

1. **Freeze the historical experiment.** Download/version original artifacts and distinguish recovered historical facts from deterministic reconstruction. Keep the current checkpoints as pilot artifacts.
2. **Repair evaluation first.** Correct value spans, omissions and cohort selection; validate the shared tokenizer; rerun all three checkpoints on identical audited probes. Regenerate every figure and table from one checked analysis script. Preserve the existing CSVs unchanged.
3. **Measure actual exposure.** Replay the original loader using the original artifacts. Decide whether the R/X contrast needs a new matched-exposure experiment before drawing causal conclusions.
4. **Rebuild a coherent fact pool and contrasts.** Enforce unique semantic identities and pair integrity; distinguish arbitrary associations from independently grounded truth; specify one-sided versus competing-evidence designs. Balance template diversity, source behavior and stance independently of dose.
5. **Run a smaller corrected C/R/X/E pilot with replication.** Require semantic, token, exposure, artifact and evaluation invariants to pass before the expensive run. Choose a scale sufficient for the intended knowledge task rather than assuming the cheapest scale will settle it. Replication should vary explicitly logged training seeds as well as the intended data factors.
6. **Then run representation and SFT studies.** Train truth probes on independently verified data with entity/domain separation, validation-only layer selection, label/lexical baselines and negation tests. Compare calibrated or transferred probes across arms rather than assuming independently trained hyperplanes have the same meaning. Decoding alone does not establish a causal gate. The [paper cited in the proposal](https://arxiv.org/html/2506.00823v1) itself finds truth-direction consistency depends on model capability and leaves probe causality unresolved.

For SFT, pin the same data, order, seed, token budget and optimizer policy across arms and scan for probe leakage. Evaluate ordinary answer accuracy, false assertions, abstention and source-conditioned retrieval. Better access to a stored report is not by itself a broken epistemic gate: the relevant failure is falsely asserting its content as fact. Keep these outcome definitions separate.

Do not train E on a newly repaired dataset and compare it directly with the old C/R/X triplet as a controlled four-arm experiment. Either add E under a verified recreation of the historical design to study that pilot, or retrain all arms under the corrected design. Given the dataset and exposure issues, a corrected smaller four-arm experiment is the stronger next use of compute.

## Verification limits and evidence record

The audit read the submitted ten-page PDF, inspected the result figures visually, reviewed the generator/evaluator/training loader and checked archived scores. It reconstructed the reported fact pool and matched injected IDs and doses to all archived arms. Sensitivity statistics are recalculations of saved model scores; no corrected model inference or retraining was performed.

The installed project PyTorch cannot import because CUDA runtime libraries are missing. The packing demonstration therefore executes the repository function with a minimal NumPy tensor shim and tests only its selection/cropping logic. Original checkpoints, tokenizer bytes, real training shards and complete W&B logs remain necessary for the final historical-run audit. These limits prevent a claim that every possible bug has been excluded.

Reproduction scripts and machine-readable evidence accompany this review. From the repository root, use a Python environment with the contextualization dependencies:

```bash
contextualization/.venv/bin/python contextualization/reviews/2026-10-13/reproduce_eval_audit.py --repo "$PWD" --out /tmp/peacechat-eval-review
contextualization/.venv/bin/python contextualization/reviews/2026-10-13/reproduce_dataset_findings.py --out /tmp/peacechat-data-review
contextualization/.venv/bin/python contextualization/reviews/2026-10-13/check_packing.py
```

The dataset script builds tiny disposable synthetic corpora inside its output directory. Neither script performs model inference or changes the original dataset/results. Saved results include `eval_audit_summary.json`, `dataset_findings.json`, `malformed_paraphrases.json`, `strict_clean_fact_ids.json` and `dose_sensitivity.csv`.

Source SHA-256 values:

- Submitted Downloads PDF: `7dd61bc20c9ac06a08c4158ea6408095698302df9822f4c61988ed28f3a46508`.
- Repository PDF: `7b055d9568025cf6e63482726ac373b9494e7d9857c1ee0d801381c4b98e2b4b`.
- `eval_results.tgz`: `ee5dad668f2b335987b6ef784c596711e2071f2940125d746c257a5f9a1f23f2`.

No production code, original result files or submitted documents were changed by this review.
