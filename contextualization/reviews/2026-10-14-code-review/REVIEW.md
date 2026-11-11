# Contextualization implementation review

The implementation is a substantial improvement, but it is not ready for the funded training reruns. The existing test suite passes while independent reproductions expose failures in probe construction, training input verification, statistical comparisons, and checkpoint archiving. Fix the applicable findings below before treating the new pipeline as a corrected experiment.

Reviewed on 14 October 2026, at commit `031c4e5`, against `29ee8d6`. Scope: all production-code changes across the three intervening commits, their tests, protocol/schema documentation, and the implementation summary supplied by the user. Large historical fixtures and archived outputs were checked as evidence, not treated as fresh experimental validation. This review adds this report and reproduction scripts only; it does not modify the implementation.

The complete repository suite returned **133 passed in 131.68 seconds** using the local CPU environment. Independent scripts use generated data, the locally available historical tokenizer where stated, and mocked remote APIs. No paid training or live HF/W&B mutation was performed. The constructed analysis examples below are deliberately synthetic counterexamples, not measured model results.

**Priorities:** P1 means a blocking correctness or preservation defect in the affected workflow. P2 means a concrete defect that must be fixed before using that feature or reporting its result. Conditional findings do not imply every default run encounters them.

**1. P1 — Uploading an older evaluation can replace that run's checkpoint tag with another run's weights.**

[tracking.py:471](/home/danielaush/Documents/peacechat/contextualization/ctx/tracking.py:471) uploads to the current default branch, then [tracking.py:475](/home/danielaush/Documents/peacechat/contextualization/ctx/tracking.py:475) deletes and recreates existing tags. [eval_beliefs.py:337](/home/danielaush/Documents/peacechat/contextualization/eval_beliefs.py:337) uses this helper with `wandb-<run_id>`; manual evaluation upload does the same.

Reproduction: initialize the mocked Hub with run A's tag pointing to weights A, and main containing later weights B. Upload A's evaluation. The real helper moves A's tag to a commit containing **weights B plus A's evaluation**. This breaks the promised run-to-checkpoint identity when rescoring old runs or running overlapping uploads. The broad exception handler also assumes every tagging failure means “tag already exists.”

Fix: keep checkpoint identities immutable; attach evaluation results to the explicitly verified checkpoint commit or a separate evaluation namespace. Check the evaluation's checkpoint hash before publishing links. Handle tag conflicts explicitly and never silently retarget an existing run to different weights. Test two runs in the same model repository and later rescoring of the older one. Reproducer: `repro/review_root_v2.py`.

**2. P1 — Migration can delete versions it never uploaded, and deletes unrelated backup runs even after verification failure.**

[migrate_wandb_to_hf.py:183](/home/danielaush/Documents/peacechat/contextualization/migrate_wandb_to_hf.py:183) sorts version strings lexicographically and selects only one model version at line 191. [Line 262](/home/danielaush/Documents/peacechat/contextualization/migrate_wandb_to_hf.py:262) nevertheless deletes every version. This assumes all previous model versions are disposable without verifying equality. With `v0`, `v1`, `v10`, and `v9`, the string order can also select the wrong “latest” version.

The offline reproduction supplies distinct v0/v1 checkpoints: only v1 is uploaded, but both are deleted. A second run forces verification failure: artifact deletion is skipped, but [line 268](/home/danielaush/Documents/peacechat/contextualization/migrate_wandb_to_hf.py:268) still deletes another collection's backup run, despite `--only ctx_C_d24_base`. This is a demonstrated code path, not a claim that the user's existing remote data has already been lost.

Fix: preserve each distinct version or prove it is an identical duplicate; parse version numbers numerically; delete only artifacts and associated upload runs whose immutable destination was verified in that operation. Verify the pinned destination revision, including reruns where the requested tag already exists. Retain `--keep-wandb` during validation. Reproducer: `repro/review_migration_v2.py`, entirely mocked.

**3. P1 — Prepacking can certify the wrong tokenizer and feed incompatible token IDs to training.**

[exposure_audit.py:42](/home/danielaush/Documents/peacechat/contextualization/exposure_audit.py:42) supports `--tokenizer-dir`. However, [prepack.py:388](/home/danielaush/Documents/peacechat/contextualization/ctx/prepack.py:388) hashes the arm-directory tokenizer files, not the tokenizer actually used for encoding. [dataloader.py:191](/home/danielaush/Documents/peacechat/nanochat/dataloader.py:191) does not enforce tokenizer identity when loading the rows.

Reproduction: encode with tokenizer SHA beginning `60f0b658`, while arm directories contain historical tokenizer `36ca7984`. The manifest records `36ca7984`, verification reports zero problems, and the loader accepts the generated IDs. This can silently invalidate an entire paid run while checkpoint metadata appears consistent with the wrong tokenizer.

Fix: bind prepacked rows to the actual encoding/tokenizer bytes, compare this binding with the model's tokenizer before training, and refuse overrides that disagree. Validate the same binding on resume. Reproducer: `repro/review_tokenizer_binding.py`.

**4. P1 — The prepack gate accepts missing occurrences and corrupted masks; resume is not bound to the data.**

[prepack.py:457](/home/danielaush/Documents/peacechat/contextualization/ctx/prepack.py:457) verifies the entries that happen to exist in the exposure CSVs. It does not require the expected occurrence set in each intervention arm, detect overwritten duplicate keys, or verify row/mask contents. [dataloader.py:205](/home/danielaush/Documents/peacechat/nanochat/dataloader.py:205) checks shapes, while [line 270](/home/danielaush/Documents/peacechat/nanochat/dataloader.py:270) checks resume scheduling fields but no dataset identity.

Three independent mutations of the tiny fixture demonstrate the gap: remove one X occurrence and verification still reports zero problems; zero an R mask file and verification still passes while the actual loader produces **zero active targets**; change saved row tokens and resume from the old state successfully with different inputs. These checks exercise stale or damaged artifacts, not ordinary successful construction.

Fix: check expected occurrence multiplicities against the build manifest, verify row/mask hashes and actual target coverage, and bind the resume state/checkpoint to the verified dataset. Distinguish C control slots from required R/X/E intervention records. Make a failed preflight stop training. Reproducer: `repro/review_v2_training.py`.

**5. P1 — Repeated answer slots still generate malformed wrapper competitors.**

[probes.py:205](/home/danielaush/Documents/peacechat/contextualization/ctx/probes.py:205) takes only the first value span and retains the original tail for the alternative. Default v2 generation with 600 groups and seed 1234 produces **16 malformed wrapper competitors among 419 injected facts**. One competing continuation says, “because Nyxorium is written as Ty, chemists use Fu in equations.” It asserts both the rival and the original value.

The answer-only spans also omit the repeated value. The corrected paraphrase path does not fix this separate wrapper path.

Fix: structurally render the entire competing wrapper with every value slot replaced and preserve each slot's capitalization and span. Add coverage for every repeated-slot frame under every applicable wrapper, including occurrence-zero v2 choices. Reproducer: `repro/review_eval_v2.py`.

**6. P1 — Analysis accepts incompatible or mock evaluations as a matched experiment.**

[analyze_beliefs.py:106](/home/danielaush/Documents/peacechat/contextualization/analyze_beliefs.py:106) does not read `.meta.json`. It also discards item contexts/candidates, arm identity, scoring policy, metric declarations, and stable template IDs. The reproduction gives C real-inference metadata and R mock-score metadata, different tokenizer/probe hashes, and different initialization seeds; the analysis completes normally.

Consequently metadata written by evaluation does not currently protect analysis from mixed scoring versions, mislabeled arms, incompatible probes, or copied checkpoints presented as independent seeds.

Fix: validate CSV hashes, item identity, mode, tokenizer, scoring policy, probe provenance, within-replicate matching, and replicate independence before aggregation. Require an explicit documented compatibility mode for historical comparisons whose builds intentionally differ; do not simply require all historical build hashes to be identical. Reproducer: `repro/review_analysis_pairing.py`.

**7. P1 — Missing templates can manufacture a paired effect while missingness reports none.**

[analyze_beliefs.py:156](/home/danielaush/Documents/peacechat/contextualization/analyze_beliefs.py:156) averages available templates independently in each arm. [Line 394](/home/danielaush/Documents/peacechat/contextualization/analyze_beliefs.py:394) checks fact presence only inside passes that exist, overlooking partial-template loss and entirely absent passes.

Reproduction: both hypothetical models score template 0 at belief 0 and template 1 at belief 10. Remove template 1 only from R's CSV. The CLI reports **R−C = −5 across 22 facts**, and saves `missingness = {"rep0": {}}`. The apparent model difference is entirely an input-coverage difference.

Fix: join exact item/template identities before taking means. Fail incomplete corrected evaluations by default, or explicitly save and consistently use the common item cohort for every curve and comparison. Reproducer: `repro/review_analysis_pairing.py`.

**8. P2 — The synthetic held-out signed baseline is forced to zero and can produce vacuous equivalence.**

[factpool_v2.py:524](/home/danielaush/Documents/peacechat/contextualization/ctx/factpool_v2.py:524) emits both reciprocal candidates per held-out group. [analyze_beliefs.py:308](/home/danielaush/Documents/peacechat/contextualization/analyze_beliefs.py:308) pools their signed beliefs under the same `unknown` label. Each group contributes `b + (−b)`, so its mean and every group-bootstrap resample are zero by construction.

The actual generated 600-group probes contain 181 wholly held-out groups, including 164 synthetic groups. A constructed score fixture reverses every group's preference from +2 to −20 in a fixed candidate orientation. Nevertheless, the synthetic baseline and paired mean have **CI [0, 0]**, and margin 0.1 produces **`tost_equivalent=True`**. The preference-flip count correctly shows 328 flipped candidate records; the signed mean is still structurally uninformative. The real true/false rows considered separately are not subject to the same cancellation.

Fix: choose one stable orientation per group, retained across arms and seeds, and report absolute changes and preference flips once per group. Do not interpret the pooled reciprocal zero as empirical stability or absence of spillover. Reproducer: `repro/review_heldout_cancel.py`.

**9. P2 — Probe regeneration accepts a source policy that the training build never used.**

[blend.py:534](/home/danielaush/Documents/peacechat/contextualization/ctx/blend.py:534) checks the pool signature, but [line 561](/home/danielaush/Documents/peacechat/contextualization/ctx/blend.py:561) uses the newly requested rendering policy to regenerate records. A rotating-source build followed by `--probes-only --source-per-fact` changes `fixed_source` from absent on all 56 injected records to present on all 56, while reporting `verified_against_manifest=True`. Evaluation then creates a purported trained-source pass from that false metadata.

Fix: recover evaluation-affecting rendering configuration from the completed original build, reject conflicting overrides, and verify more than fact IDs/frequencies. Reproducer: `repro/ctx_data_review_repro.py`.

**10. P2 — Overwrite leaves stale audit evidence that later becomes current probe metadata.**

[blend.py:243](/home/danielaush/Documents/peacechat/contextualization/ctx/blend.py:243) deletes a selected list of artifacts. It leaves `novelty_audit.json` and other derived outputs. [Line 541](/home/danielaush/Documents/peacechat/contextualization/ctx/blend.py:541) later trusts any surviving novelty audit.

Reproduction: build with collision auditing, overwrite without auditing, then regenerate probes. The new summary says `novelty_audit=None`, yet the old file changes **35 held-out novelty labels** in the regenerated probes and verification still succeeds.

Fix: replace a complete staged build, invalidate every dependent artifact, or require each auxiliary artifact to match the current build/shard identity. Reproducer: `repro/ctx_data_review_repro.py`.

**11. P2 — Grounded truth labels are not balanced, and dose remainders systematically favor low doses.**

[factpool_v2.py:375](/home/danielaush/Documents/peacechat/contextualization/ctx/factpool_v2.py:375) alternates candidate slots, although truth-to-slot assignment was independently randomized. Alternating slots therefore does not alternate true and false assertions as the comment claims. The 600-group seed-4 pool gives the real element-symbol stratum **six true, zero false**. Tests that examine only labels present in a counter miss the empty cell.

[Line 389](/home/danielaush/Documents/peacechat/contextualization/ctx/factpool_v2.py:389) also restarts the sorted dose grid at the lowest dose in each small stratum. This makes remainder allocation systematically favor low doses; it is not balanced random allocation of the remainder.

Fix: balance grounded labels explicitly, then select their slots; randomize or counterbalance dose remainders. Report all planned cells, including zero cells, and test multiple seeds. Reproducer for the label failure: `repro/ctx_data_review_repro.py`.

**12. P2 — Fixed rendering does not fix the control arm's inserted text.**

[templates.py:890](/home/danielaush/Documents/peacechat/contextualization/ctx/templates.py:890) samples C's neutral sentence using the occurrence number even with `RenderPolicy.fixed_rendering()`. In the reproduction, ten occurrences yield **six different C strings**, versus one R string and one X string. This contradicts the documented fixed-string control in every arm and leaves a rendering-diversity difference in that control experiment.

Fix: give neutral inserts an explicit repetition policy or use one per-fact draw in fixed mode. Test C as well as the intervention arms. Reproducer: `repro/ctx_data_review_repro.py`.

**13. P2 — Explicit policy overrides are ignored in fixed mode.**

[blend.py:138](/home/danielaush/Documents/peacechat/contextualization/ctx/blend.py:138) returns before applying overrides. `--render-mode fixed --policy-wrapper random` resolves to `wrapper="fixed"`, silently, despite documentation allowing individual factors to be re-enabled.

Fix: construct fixed defaults and then apply explicit overrides, or reject unsupported combinations. Validate and record the effective policy before building. Reproducer: `repro/ctx_data_review_repro.py`.

**14. P2 — Competing-evidence records can report the wrong rival dose.**

[factpool_v2.py:509](/home/danielaush/Documents/peacechat/contextualization/ctx/factpool_v2.py:509) uses the group-level rival frequency for both injected candidates. In the competing design with actual doses 1 and 4, the dose-4 candidate says its rival also has dose 4 instead of 1.

Fix: resolve the other candidate's frequency through `rival_fact_id` and verify it reciprocally. This is specific to the optional competing-evidence design. Reproducer: `repro/ctx_data_review_repro.py`.

**15. P2 — Character matching does not ensure the required X/E token matching.**

[templates.py:909](/home/danielaush/Documents/peacechat/contextualization/ctx/templates.py:909) chooses E filler using character lengths. With 2,500 groups, seed 1234, the historical tokenizer, fact `52df0467bbf5445e`, occurrence 1, X has **49 tokens** and E **40**, exceeding the allowed absolute difference of 8. Their character lengths are 167 and 168; claim starts differ by only two characters.

The real-tokenizer validator should catch this, so this is a generation/readiness failure rather than proof it would pass full validation. The 600-group successful validation is insufficient for the larger pool.

Fix: make selection or rejection token-aware using the actual training tokenizer and validate the complete intended pool. Character-only diagnostics must not imply token limits were verified. Reproducer: `repro/ctx_data_review_repro.py`.

**16. P2 — Exposure audit returns success when training steps disagree across every arm.**

[exposure_audit.py:127](/home/danielaush/Documents/peacechat/contextualization/exposure_audit.py:127) checks `n_row_agree` but ignores `n_step_agree`. Row IDs are file locations; different row-order seeds can schedule those same rows at different steps.

Reproduction with stored seeds R=1, X=2, E=3: **six occurrences, six row agreements, zero step agreements, exit code 0**. The error text promises row/step agreement but only enforces the former.

Fix: check effective step/horizon/status agreement and the scheduling configuration, then enforce the audit result at launch. Reproducer: `repro/review_v2_training.py`.

**17. P2 — Fixed-rendering memorization cohorts reconstruct the wrong form without a manifest.**

[cohorts.py:477](/home/danielaush/Documents/peacechat/contextualization/ctx/cohorts.py:477) assumes a fixed paraphrase index defaulting to zero. The actual renderer chooses its fixed form using the seed and fact ID. On 419 injected facts, this fallback reconstructs **331 wrong forms** and assigns **89 facts to the wrong memorization subgroup**.

Fix: use the actual policy/seed selection or require the manifest. Missing provenance should not silently become form zero. A correct supplied manifest avoids this branch. Reproducer: `repro/review_eval_v2.py`.

**18. P2 — Cluster-aware intervals coexist with p-values that assume independent facts.**

[stats.py:210](/home/danielaush/Documents/peacechat/contextualization/ctx/stats.py:210) sends every fact to ordinary Wilcoxon despite resampling semantic groups for intervals. `paired_cell([1.]*20, ["same-group"]*20)` reports one group, unavailable bootstrap intervals, and **p = 7.744e−6**. Holm correction does not repair the dependence assumption.

This affects historical reused-group cohorts and designs/cells containing multiple facts per group; the primary one-injected-value-per-group design avoids that particular duplication.

Fix: use an appropriate group-level or cluster-aware test, or suppress conventional Wilcoxon inference when clusters repeat. Keep between-training-run uncertainty distinct from fact sampling.

**19. P2 — Custom probe evaluations attest hashes of files they did not score.**

[eval_beliefs.py:296](/home/danielaush/Documents/peacechat/contextualization/eval_beliefs.py:296) hashes the default experiment probe directory even when input came from `--probe-dir`. The evaluation metadata can therefore describe different probes from those actually used. The pre-inference binding check checks the tokenizer, not the selected build/probe relationship.

Fix: hash the resolved files actually opened, record their identities, and validate checkpoint/build compatibility with explicit support for intentional historical rescoring. This finding follows directly from the input-path and metadata call sites.

**20. P2 — Answer/suffix token-boundary diagnostics incorrectly report no merges.**

[probes.py:350](/home/danielaush/Documents/peacechat/contextualization/ctx/probes.py:350) checks whether a clamped answer-token endpoint exceeds the entire sequence length, which cannot detect a suffix overlap. The competing-candidate flag is never assigned. The historical tokenizer yields **554 candidate sequences with answer/suffix overlap** on the 600-group fixture, all unflagged. For example, answer `insect` and suffix `s.` share the token ` insects`.

The overlap-based score definition is documented; the defect is that the promised diagnostics falsely imply clean separation. Fix by comparing answer byte boundaries with each overlapping token's byte extent, for both candidates. Reproducer: `repro/review_eval_v2.py`.

**21. P2 — The prepacked loader can exhaust file descriptors at corpus scale.**

[dataloader.py:203](/home/danielaush/Documents/peacechat/nanochat/dataloader.py:203) retains two open memory mappings per chunk for the entire corpus. At 4,096 rows/file and T=2,048, roughly 4.3 billion packed tokens require over 512 chunk pairs. With a process limit of 1,024 descriptors, other open files make failure occur earlier.

A reproduction with 520 tiny chunk pairs and a 1,024 soft limit raises **`OSError: Too many open files`**. This is conditional on the deployment host's limits; it does not establish that the current host fails.

Fix: use a bounded mapping cache, or explicitly check and accommodate descriptor capacity before a paid launch. Reproducer: `repro/review_training_extra.py`.

**Experimental decisions and remaining limitations**

The new masks change loss normalization. [base_train.py:624](/home/danielaush/Documents/peacechat/scripts/base_train.py:624) averages microbatch losses, each already averaged over its own nonmasked targets. With unequal active counts, this differs from a single mean over all scored tokens and depends on microbatch/rank partitioning. An independent same-token toy example yields gradients 0.29131266 for B=1/GA=2 and 0.79131263 for B=2/GA=1. Existing prepack fixtures have 59 active R targets versus 96 X targets in some rows; the resulting 1.627 weighting ratio is a deliberately small fixture, not a full-scale estimate. Decide whether the intended objective uses a common fixed denominator, globally normalized active-token loss, or strictly matched active-token budgets, and measure the actual arm differences. Padding visible to attention and unequal active totals are already documented design tradeoffs, not newly discovered hidden bugs.

Resume provenance also needs care. [base_train.py:179](/home/danielaush/Documents/peacechat/scripts/base_train.py:179) restores old weights, but [line 599](/home/danielaush/Documents/peacechat/scripts/base_train.py:599) writes the current CLI `init_seed`. Resuming a seed-7 model with the default CLI seed can relabel it as seed 42. Preserve original initialization identity separately from resume settings. This is established by the control flow; a full model-resume run was not performed in this review.

FineWeb pinning remains optional. The builder does not call the revision-resolution/remote-checksum helpers; an unpinned default can still use `main`, and cached reads need not rehash source bytes. The docs tell operators to choose a revision, so this is primarily a production-readiness gap in the defaults and the “fixed” status claim. Resolve an immutable source revision and verify the intended cache before the funded build.

The corrected prepacked loader is opt-in. `run_arm.py` without an explicit loader still reaches `base_train`'s historical `bestfit` default. A corrected production launcher should make its loader and passed preflight unambiguous. A long first sentence can also exceed the carrier's character target; prepack correctly rejects an intervention ending beyond T+1, so it is a readiness failure rather than silent evidence truncation. Prepack output itself is not staged as a complete multiarm transaction.

The documented conservative historical paraphrase cohort continues to exclude the old 104 cases after corrected rescoring. Retaining that cohort as a historical reference is reasonable; adding a separate valid-rescored cohort would make use of repaired evaluations without silently changing the historical comparison.

The historical replay fixtures and seed wiring are useful improvements. Whole-group holdout and atomic semantic pairs address the original collision design; synthetic truth labels are now explicit; structural paraphrase alternatives, scored-span tracking, and separate answer/suffix metrics provide a better foundation. However, original-run exposure still needs the original shards/resume history, and the archived three-fact inference check does not establish full C/R rescoring or recover X.

**Checks required before the funded reruns**

1. Convert the applicable reproductions into regression tests, including rejected incompatible artifacts and incomplete coverage. Correct the findings matrix so “fixed” reflects the tested behavior.
2. Freeze the intended protocol, full pool size, source revision, tokenizer, rendering policy, objective, schedule, and replicate layout. Validate that exact pool and every intervention with the actual tokenizer; do not infer full-pool validity from 600 groups.
3. Make one preflight verify all arm datasets, masks, occurrence sets, tokenizers, step schedules, and checkpoint/resume bindings using the exact production T, device batch size, world size, gradient accumulation, seeds, and stopping horizon. Report achieved per-fact exposure, including dropped epoch tails. Ensure the actual training command consumes those verified artifacts, and use distinct checkpoint destinations per replicate.
4. Run inexpensive real C/R/X/E training and evaluation through the complete path, including interruption/resume and analysis metadata checks. Compare initial parameter hashes across matched arms. Follow CPU checks with one short pilot on the intended GPU to verify compilation, finite gradients, continuation, throughput, and storage demands. Verify two training runs plus rescoring of the older run with mocked archival APIs before allowing real archive mutations.
5. Then run the matched-seed funded experiment. Keep historical runs as pilot evidence, preserve their artifacts, and distinguish evaluation-only fixes from data/objective fixes that require new training.

**Reproduction inventory**

The scripts are demonstrations of the present defects; their successful exit is not a claim the defects are fixed. Run from `/home/danielaush/Documents/peacechat` with `/home/danielaush/.cache/ctx_cpu_venv/bin/python`. They create temporary fixtures under `/tmp`. Tokenizer-dependent scripts expect the historical tokenizer already present under `/home/danielaush/.cache/ctx_wandb_to_hf/ctx_C_d24_base/tokenizer`. HF/W&B reproductions use only fake APIs.

| Script in this directory's `repro/` folder | Coverage |
|---|---|
| `review_root_v2.py` | HF tag corruption; also disproves the suspected torchrun separator issue by parsing the actual launcher and training parser |
| `review_migration_v2.py` | Deletion of unmigrated versions and unrelated backup-run deletion on verification failure |
| `review_tokenizer_binding.py` | Actual encoding and recorded tokenizer mismatch |
| `review_v2_training.py` | Missing occurrence, corrupted mask, changed-data resume, mismatched schedule, active-token counts |
| `review_training_extra.py` | Descriptor-limit reproduction and gradient-normalization counterexample |
| `review_eval_v2.py` | Repeated-slot wrappers, fixed-form subgroup errors, unflagged BPE overlap |
| `review_analysis_pairing.py` | Incompatible evaluation metadata and fabricated contrast from partial templates |
| `review_heldout_cancel.py` | Actual reciprocal prompts plus complete-analysis cancellation/equivalence counterexample |
| `ctx_data_review_repro.py` | Label balance, fixed control, policy overrides, competing doses, regeneration, stale audits, X/E token matching |

Full-suite command executed:

```sh
PYTHONDONTWRITEBYTECODE=1 /home/danielaush/.cache/ctx_cpu_venv/bin/python -m pytest contextualization/tests -q -p no:cacheprovider
```

All 133 tests passed. Live remote archival semantics were exercised through mocks, not through the user's accounts; full-scale GPU behavior and complete historical rescoring remain unverified.
