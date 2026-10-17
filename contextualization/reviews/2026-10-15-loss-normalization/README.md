# Loss normalization decision (2026-10-15): fixed nominal normalization

Decision taken by the experimenter on 2026-10-15 after the recheck repair: the funded C/R/X/E
experiment trains with `--loss-normalization fixed_nominal`, selected explicitly on the preflight
(`exposure_audit.py --mode prepacked`) and on every `run_arm.py` / `scripts/base_train.py` call; the
historical `microbatch_mean` remains the default for backward compatibility.

Implementation: `nanochat/loss_normalization.py` (policies, `microbatch_loss`, `StepLossStats`,
`recorded_policy`), `scripts/base_train.py` (flag, objective branch, resume and preflight checks, meta
keys `loss_normalization` / `loss_denominator_per_microbatch`, per-step logging of the nominal objective
and the loss per active token across all ranks), `contextualization/exposure_audit.py` (preflight records
and certifies the policy), `run_arm.py` (forwards it, refuses an uncertified policy before launch, records it
in run_info), `eval_beliefs.py` (propagates `checkpoint.loss_normalization`), `analyze_beliefs.py` (one
objective per experiment; unrecorded = historical). Formula per micro-batch: cross-entropy with
`reduction="sum"` over valid targets / (`device_batch_size x max_seq_len x grad_accum_steps`); the
accumulation division happens once; the optimizer's `ReduceOp.AVG` across ranks adds no division; the sum is
computed directly so a fully masked micro-batch contributes 0. Validation NLL / bpb are unchanged.

| file | what it is |
|---|---|
| `test_suite_result.txt` | the complete suite after the change: 227 passed (219 before; 8 tests added) |
| `smoke_crxe_summary_fixed_nominal.json` | `smoke_crxe.py --work /tmp/ctx_smoke_fixed --train` under `fixed_nominal`: 26 stages, exit 0, incl. the launcher refusing an uncertified policy, both seeds trained and evaluated with the policy recorded in metas / run_info / eval metas, the resume under `microbatch_mean` refused with no checkpoint written, and the two-seed analysis with every evaluation on the same objective |
| `smoke_logs/` | the refusal log tails and the training log lines showing `nominal` vs `per_active_tok` vs `active/nominal` per step |

Tests covering the required properties: `tests/test_loss_normalization.py` (unmasked equivalence with the
historical mean, invariance to the micro-batch partition at a fixed token budget with unequal masking which
the mean lacks, fully masked micro-batches stay finite and contribute zero, distributed averaging without an
extra world-size division, unrecorded metadata = historical policy);
`tests/test_seeds.py::test_tiny_cpu_fixed_nominal_is_certified_recorded_logged_and_bound_on_resume` (real
tiny CPU runs: a historical / unrecorded preflight does not certify the new policy, a fixed-nominal preflight
refuses a mean run, meta and log contents, same-policy resume exact, mismatched-policy resume refused, a
checkpoint without the key resumed under the new policy refused);
`tests/test_prepacked_loader.py::test_preflight_certifies_the_loss_normalization_policy`;
`tests/test_analysis.py::test_training_objective_is_one_per_experiment`; `tests/test_eval.py` (provenance).
