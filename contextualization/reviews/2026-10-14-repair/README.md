# Repair-pass evidence (2026-10-14)

Local evidence produced while repairing the findings of `../2026-10-13/REVIEW.md`. Nothing
here is a rescoring of the historical experiment; see `../../docs/ERRATA.md` and
`../../docs/FINDINGS_MATRIX.md`.

| file | what |
|---|---|
| `real_armC_limit3.csv` / `.meta.json` / `.log` | the corrected evaluator (`eval_beliefs.py`, eval schema v2) run with REAL model inference on the local copy of the July arm-C checkpoint (`~/.cache/ctx_wandb_to_hf/ctx_C_d24_base`, step 8352, CPU fp32) for the first 3 injected facts (30 contrasts); plumbing evidence only |
| `real_vs_archived_armC_3facts.json` | those 18 injected-probe contrasts against the archived `eval_results.tgz` arm-C scores: max |delta belief| 0.177 log units (fp32 CPU vs bf16 GPU numerics); the historical rendering and metric definition are preserved |
| `dry_run_historical_probes_armC.meta.json` | full coverage of the historical probe files (v1 build metadata, real July tokenizer, mock scores): 45,204 contrasts, all 5,600 wrapper probes present, 0 joint-boundary adjustments |
| `../../tests/fixtures/historical/legacy_boundary_incidence.json` | historical evaluator: 8,712 of 86,282 sequences hit the context-boundary BPE fallback (all wrapper pass), symmetric for all but 2 items |

The auditor's scripts in `../2026-10-13/` still run against the refactored code:
`reproduce_eval_audit.py` reproduces every count (its `wilcoxon_tie_repro.implemented_p` now
reports the scipy-based tie-corrected value because `analyze_beliefs.wilcoxon_signed_rank` was
replaced; the historical helper is `legacy_wilcoxon_signed_rank`), and `reproduce_dataset_findings.py`
reproduces the pool statistics but its 1,000-character small build now stops with the builder's
explicit "parity top-up infeasible" error (finding 23 fix), so its E-invariance / cache-alias
blocks are superseded by `tests/test_build.py` and `tests/test_artifacts.py`.

| `smoke_crxe_summary.json` | `smoke_crxe.py --train` run: generation -> validation (122 checks) -> real BPE tokenizer -> prepack + exposure audit -> four 5-step depth-2 CPU trainings with identical init/data seeds -> eval plumbing (mock scores) and real tiny-model inference -> analysis_v2 |
| `validate_600groups_historical_tokenizer.log` | `validate.py` on a 600-group / 60M-char v2 build with E and collision audit, token checks with the July tokenizer: 115 passed, 0 failed |

## Second review (reviews/2026-10-14-code-review) — evidence after the fixes

| file | what |
|---|---|
| `second_review_reproductions_after_fixes.log` | all nine reproduction scripts re-run against the fixed code: tag retargeting refused (`ValueError`), migration deletes nothing, prepack refuses a foreign tokenizer, gate stops at the first mutation, 520 memmaps load under a 1,024-fd limit, 0 wrong fixed forms, metadata incompatibility refused, probes-only policy conflict refused |
| `smoke_crxe_summary_second_review.json` | `smoke_crxe.py --train`: tokenizer-first build with token-aware X/E matching, validation (126 checks), prepack + full-horizon preflight (68/68 occurrences, rows and steps agree), stale preflight refused by the launcher, four matched-seed depth-2 trainings with identical `init_param_sha256`, interrupt/resume with identical end loader state and preserved init identity, real inference on all arms, analysis with passing compatibility checks, mixed mock/real evaluations refused |

See also `reviews/2026-10-14-recheck-repair/` for the recheck (third review) repair evidence.
