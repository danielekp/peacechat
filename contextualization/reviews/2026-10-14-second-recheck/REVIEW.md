# Contextualization second repair recheck

This pass fixes the specific counterexamples from the previous review and substantially strengthens the pipeline. I independently reproduced **219 passing tests** and the **24-stage, two-seed CPU smoke workflow**. The remaining problems are narrower: auxiliary analysis inputs are not fully bound, the complete manifest breaks one analysis, and several archive/launcher paths still lose verification or run identity. Address the six findings below before treating the full workflow as ready for funded production runs.

Reviewed on 14 October 2026 against the uncommitted tree on top of `031c4e583995bd17bfb9aa55d78897a3ea1f5994`. The repair includes 42 modified tracked files plus new files. [review_state.json](/home/danielaush/Documents/peacechat/contextualization/reviews/2026-10-14-second-recheck/review_state.json) records source fingerprints and exact validation commands. Production code remained unchanged during this review; additions are this report and review evidence. HF/W&B checks were offline/mocked; no paid training or remote mutation occurred.

**What now holds**

The prior mixed-mode replicate, wrong selected-probe, changed scored-item, mismatched initial-weight/horizon, and changed-accumulation counterexamples are rejected. Entirely mock outputs keep their labels. Two different initialization seeds now retain separate evaluations and run-info files. First-upload repository creation works in realistic mocks. The actual lightweight environment loads the historical tokenizer, uses its true vocabulary size, and passes a full build/validation cycle. Missing standalone novelty audits reconstruct byte-identical probes from the embedded evidence. Design cells, token claim-start limits, and rank-local active-target reporting are implemented.

The independent full suite passed in **357.23 seconds**. The fresh smoke trained and evaluated C/R/X/E for seeds 7 and 8, checked matched initialization within each seed, tested resume and refusal paths, and analyzed real tiny-model scores. See [suite output](/home/danielaush/Documents/peacechat/contextualization/reviews/2026-10-14-second-recheck/ctx_review4_suite.log) and [smoke summary](/home/danielaush/Documents/peacechat/contextualization/reviews/2026-10-14-second-recheck/smoke_summary.json). These are software checks, not evidence of the scientific hypothesis.

**1. P2 — A stale exposure trace is still accepted as fully bound and changes memorization classifications.**

[analyze_beliefs.py:355](/home/danielaush/Documents/peacechat/contextualization/analyze_beliefs.py:355) reads the dataset IDs from `PREFLIGHT_OK.json` or `cross_arm.json`. At [line 501](/home/danielaush/Documents/peacechat/contextualization/analyze_beliefs.py:501), a matching dataset ID is enough to mark exposure inputs bound. The actual `occurrence_exposure.csv` contents and schedule used for classification are not bound to that certificate or the evaluated checkpoint. A dataset ID identifies the packed rows, not how many optimizer steps consumed them.

Reproduction uses the fresh smoke's actual evaluation CSVs and metadata, unchanged. With the copied original exposure files, R has **7 verbatim-trained and 19 non-verbatim facts** in each seed. Using the production `schedule_exposure()` function, regenerate only R's exposure CSV/summary at **one step**, leaving the existing **267-step** preflight in place. Strict analysis still reports `ok=true`, no warnings, and all eight CSVs `inputs_bound=true`; R becomes **0 verbatim-trained and 26 non-verbatim facts**. The classification change comes from the wrong trace, not a different model.

**Completion criteria:** certify the exposure files actually consumed by analysis, validate their schedule against their certificate and checkpoint step, and reject stale combinations before aggregation. A shorter-checkpoint analysis must use the appropriate prefix of the training history. Add both a trace-content mismatch and a same-dataset/different-horizon case. This is the remaining auxiliary-input portion of previous finding R2.

Evidence: [reproducer](/home/danielaush/Documents/peacechat/contextualization/reviews/2026-10-14-second-recheck/repro/exposure_binding.py), [before/after output](/home/danielaush/Documents/peacechat/contextualization/reviews/2026-10-14-second-recheck/ctx_review4_exposure_binding.log).

**2. P2 — Supplying the legitimate C/R/X/E manifest disables memorization analysis.**

[cohorts.py:539](/home/danielaush/Documents/peacechat/contextualization/ctx/cohorts.py:539) retains the C arm's filler rows with `paraphrase_index=-1`. [Line 598](/home/danielaush/Documents/peacechat/contextualization/ctx/cohorts.py:598) treats those entries as claim renderings and rejects the negative index. The full generated manifest legitimately contains **68 C sentinel rows** in the smoke fixture.

Run strict analysis with the original probes, build summary, `--manifest`, and `--exposure-dir`. Compatibility succeeds without warnings, but both replicates report `NOT COMPUTABLE: ... manifest paraphrase_index -1 outside surface_forms`, and **no memorization table is produced**. The expanded smoke does not include this full-manifest analysis path.

**Completion criteria:** exclude control filler records from intervention claim-form reconstruction while retaining checks for genuinely invalid R/X/E paraphrase indices. Add an integration case using the unfiltered manifest emitted by a real C/R/X/E build, and assert that the intended memorization tables exist with the expected classifications. Do not resolve this by requiring users to hand-filter the manifest, which would conflict with manifest hashing.

Evidence: [reproducer](/home/danielaush/Documents/peacechat/contextualization/reviews/2026-10-14-second-recheck/repro/control_manifest.py), [output](/home/danielaush/Documents/peacechat/contextualization/reviews/2026-10-14-second-recheck/ctx_review4_control_manifest.log).

**3. P2 — Repeating a completed launch replaces its run artifacts before the existing evaluation is detected.**

[run_arm.py:256](/home/danielaush/Documents/peacechat/contextualization/run_arm.py:256) assigns a new run identity and overwrites the tag's run-info fields before training. [Line 284](/home/danielaush/Documents/peacechat/contextualization/run_arm.py:284) launches into the same checkpoint directory without checking for an occupied destination. [checkpoint_manager.py:44](/home/danielaush/Documents/peacechat/nanochat/checkpoint_manager.py:44) writes existing checkpoint/model/optimizer paths. Only subsequent evaluation notices that its CSV already exists.

The offline launcher reproduction uses fake training writes and actual dry-run evaluation. Launching seed 7 twice leaves **run-info and checkpoint bytes naming `run7retry`**, while the retained evaluation belongs to **`run7`**; the second invocation fails only at the CSV check. The current regression test at [test_artifacts.py:1160](/home/danielaush/Documents/peacechat/contextualization/tests/test_artifacts.py:1160) explicitly expects this late failure. Different-seed isolation is fixed, but retry safety is not. With actual training, this path can spend another run's compute and replace its local checkpoints before failing.

**Completion criteria:** before changing metadata or launching fresh training, reject an occupied model tag/run destination, or require an explicit replacement workflow. Keep explicit checkpoint resume and new-tag runs supported. Test that an accidental repeat leaves checkpoint, metadata, and evaluation bytes unchanged and never invokes training.

Evidence: [reproducer](/home/danielaush/Documents/peacechat/contextualization/reviews/2026-10-14-second-recheck/repro/occupied_run.py), [output](/home/danielaush/Documents/peacechat/contextualization/reviews/2026-10-14-second-recheck/ctx_review4_retry.log). No real model was overwritten in this reproduction.

**4. P2 — Automatic checkpoint selection does not propagate the selected tag into evaluation provenance.**

`--model-tag` is optional. [eval_beliefs.py:243](/home/danielaush/Documents/peacechat/contextualization/eval_beliefs.py:243) lets `load_model()` select the model directory. Later, [line 283](/home/danielaush/Documents/peacechat/contextualization/eval_beliefs.py:283) still chooses output paths using the unresolved argument, and [line 327](/home/danielaush/Documents/peacechat/contextualization/eval_beliefs.py:327) reconstructs the checkpoint directory from model depth.

With only the real smoke checkpoint `d2_s7` present, evaluation without `--model-tag` successfully loads and scores **d2_s7**, but records **`model_tag=null`**, a nonexistent **`base_checkpoints/d2/model_000267.pt`**, and **`model_sha256=null`**. Its default CSV is also placed in the unscoped root evaluation directory. This loses the checkpoint binding and undermines archival/default output isolation. The normal launcher passes an explicit tag and avoids this defect.

**Completion criteria:** resolve the checkpoint tag once and use that resolved identity for loading, hashing, metadata, run-info lookup, outputs, and tracking. Alternatively require explicit selection when appropriate. Verify automatic and explicit selection of the same checkpoint yield identical provenance and output destinations.

Evidence: [actual CPU inference reproducer](/home/danielaush/Documents/peacechat/contextualization/reviews/2026-10-14-second-recheck/repro/automatic_model_tag.py), [output](/home/danielaush/Documents/peacechat/contextualization/reviews/2026-10-14-second-recheck/ctx_review4_autotag.log).

**5. P2 — The actual build archive omits evidence required by the new verification path.**

[tracking.py:1066](/home/danielaush/Documents/peacechat/contextualization/ctx/tracking.py:1066) uploads the summary, probes, compressed manifest, and optional exposure data, but omits `BUILD_COMPLETE.json`. The export manifest also does not include that marker. The archive regression test manually copies it, so the test's archive contains more evidence than the actual exporter provides.

Intercepting the actual `hf_upload_build()` staging directory, then regenerating probes from that exact archived set, gives `verified_against_manifest=True` but **`verified_against_probe_files=None` and `probe_files_compared=[]`**. Novelty labels still reconstruct correctly; the missing part is verification against the original probe hashes. Independently, analysis of the restored real smoke archive accepts `--manifest` with no marker hash, no warning, and `inputs_bound=true` for all four arms.

**Completion criteria:** include the completion marker in the uploaded set and export manifest, and test the actual staging/download layout. Verification should either use the bound marker/export evidence or explicitly refuse/mark the relevant inputs unverified when it is absent; strict reports should not imply that omitted checks passed. The embedded novelty-audit fix itself is sound.

Evidence: [actual exporter/analysis reproduction](/home/danielaush/Documents/peacechat/contextualization/reviews/2026-10-14-second-recheck/repro/build_export.py), [output](/home/danielaush/Documents/peacechat/contextualization/reviews/2026-10-14-second-recheck/ctx_review4_export.log), and [lightweight build/export/regeneration reproduction](/home/danielaush/Documents/peacechat/contextualization/reviews/2026-10-14-second-recheck/repro/data_roundtrip.py).

**6. P2 — An explicit tokenizer override can obtain PASS while validating against the wrong encoder.**

[validate.py:1476](/home/danielaush/Documents/peacechat/contextualization/validate.py:1476) verifies the arm tokenizers against the build summary. Immediately afterward, [line 1478](/home/danielaush/Documents/peacechat/contextualization/validate.py:1478) replaces the encoder used for token checks with `--tokenizer-dir`, without comparing the override's fingerprint.

On the actual lightweight 80-group build made with the historical **32,768-token** encoder, supplying a different valid **256-token** encoder returns **PASS / exit 0**, including `9.vs_summary=PASS`, while applying toy relative limits. This build also passes under its correct encoder: the reproduction establishes an unbound validation path, not that this particular corpus violates its intended limits. Default validation with the build's tokenizer is repaired.

**Completion criteria:** bind a supplied override to the recorded training tokenizer. If alternate-tokenizer exploration is useful, expose it as explicitly exploratory results rather than an ordinary build-validation PASS. Test matching and mismatched overrides in the lightweight environment.

Evidence: [reproducer](/home/danielaush/Documents/peacechat/contextualization/reviews/2026-10-14-second-recheck/repro/data_roundtrip.py), [output](/home/danielaush/Documents/peacechat/contextualization/reviews/2026-10-14-second-recheck/ctx_data_latest_recheck.log).

**One evidence qualification and one small link defect**

The smoke verifies exact **loader/exposure continuation**, preserved initialization identity, and rejection of a changed schedule. [smoke_crxe.py:195](/home/danielaush/Documents/peacechat/contextualization/smoke_crxe.py:195) does not compare final model/optimizer states. Independent comparisons found small final-parameter differences between full and resumed runs, and ordinary repeated same-seed CPU runs also differed. These observations do **not** establish a resume defect. Avoid describing the test as bitwise optimization equivalence; strengthen a GPU pilot with loss/logit comparisons using justified tolerances. [Comparison evidence](/home/danielaush/Documents/peacechat/contextualization/reviews/2026-10-14-second-recheck/repeatability_comparison.json).

[eval_beliefs.py:471](/home/danielaush/Documents/peacechat/contextualization/eval_beliefs.py:471) still appends `/tree/main/eval` to a URL that already points at the evaluation branch. Store the returned branch URL directly, or construct the correct branch-relative evaluation path. This affects the W&B link, not checkpoint preservation.

**Next acceptance target**

Fix the six concrete paths and the link, then extend the existing smoke to consume the full manifest and verified exposure inputs, and to restore the actual staged archive. Retain the tests that now pass; no broad rewrite is warranted. Use the completion criteria above rather than another blanket “all fixed” claim.

Loss normalization remains an explicit scientific decision; its unchanged status is accurately reported. My recommendation remains the same fixed nominal token denominator across arms, with appropriate accumulation/distributed scaling, for the primary controlled comparison. Freeze that choice, an immutable FineWeb revision, dependencies, and the reviewed code before measuring the funded configuration. Then perform full intended-pool preflight and a short target-GPU pilot, including functional resume checks and measured throughput. Use those measurements to price the replicated experiment. Live archival round-trip verification remains outstanding before relying on HF as the preserved copy.

Reproduction scripts are local and use temporary directories. Most use `/home/danielaush/.cache/ctx_cpu_venv/bin/python`; `data_roundtrip.py` uses `contextualization/.venv/bin/python` and the existing historical tokenizer. The manifest/exposure reproducers accept a smoke build path as their first argument and default to `/tmp/ctx_review4_smoke/build`. The automatic-selection/export reproducers use that fresh smoke build. Run the smoke command in `review_state.json` first if those temporary fixtures are no longer available. No reproduction needs live credentials or network access.
