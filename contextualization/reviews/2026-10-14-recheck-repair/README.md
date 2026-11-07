# Recheck repair evidence (reviews/2026-10-14-recheck/REVIEW.md, 8 findings + reporting gaps)

Produced 2026-10-14 (night) on branch `feature/fix-bugs`, uncommitted on top of `031c4e5`. The
per-finding status with the code changes and the regression tests is in
`docs/REVIEW_2026-10-14_STATUS.md` (section "Recheck"). Nothing here is a scientific result.

| file | what it is |
|---|---|
| `reproductions_before_fixes.log` | the reviewer's six scripts run against the tree BEFORE the repair: every finding reproduced (mixed modes pooled, doses 1 -> 256 relabelled, 15 different competitors accepted, resume with grad_accum 2 behind a GA-2 preflight finished with `preflight: ok`, `CALL_ORDER ['list_repo_refs']` then `RepositoryNotFoundError`, seed 8 failed with "CSV exists", `tiktoken_spec None` / `reported_vocab 0`, 35 labels flipped) |
| `reproductions_after_fixes.log` | the same scripts after the repair: each now stops at the fixed defect (the analysis scripts with the contract violation list, the resume script with the schedule refusal, the archival script with `create_repo` before `list_repo_refs`; the replicate script evaluates both seeds into `eval/seed_7` / `eval/seed_8` and then reads the legacy `run_info.json` that replicate tags no longer write; the data script loads the tokenizer in the light env and crashes on its own closure inspection) |
| `data_recheck_after_fixes.py` / `.log` | post-fix equivalent of `ctx_data_rereview_repro.py` through the public API: fallback loader reports vocab 32768 and the absolute limit mode (the 9-token delta is rejected; the token-aware filler gives -7), regeneration without `novelty_audit.json` changes 0 labels and names its evidence, `design_cells` is in the summary (16 empty cells on the 80-group fixture) |
| `test_suite_result.txt` | the complete suite: 219 passed (192 before this pass; 27 regression tests added) |
| `smoke_crxe_summary_recheck.json` | `smoke_crxe.py --work /tmp/ctx_smoke_recheck --train`: 24 stages, exit 0: tokenizer, v2 build (token limits verified), validation (127 checks), prepack + preflight (534 rows, 267 steps), stale preflight refused, `run_arm.py` for seeds 7 and 8 x arms C/R/X/E (real depth-2 CPU training + real inference into `eval/seed_<S>/`, one `run_info_d2_s<S>.json` per arm and seed), identical `init_param_sha256` across arms per seed and distinct across seeds, interrupt/resume of arm C with identical end state, resume with a changed accumulation schedule refused (no checkpoint written), mock evaluations, cross-replicate analysis of the two seeds (compatibility ok, every CSV bound to the inputs, 0 item-identity mismatches, replicate tables written), mixed mock/real replicate refused, foreign probe files refused |
| `smoke_logs/*.tail.log` | the tails of the refusal logs and of the real analysis log from that run |

Commands are listed at the end of `docs/REVIEW_2026-10-14_STATUS.md`. No training beyond the
depth-2 CPU fixtures was launched, nothing was uploaded, and no remote HF / W&B data was touched;
all Hub interactions in the tests are mocks (the live archival round-trip remains unexercised).
