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
