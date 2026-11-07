"""Loss normalization policies of scripts/base_train.py (contextualization experiment, decision of 2026-10-15).

Two policies, selected with ``--loss-normalization``:

* ``microbatch_mean`` (historical nanochat default): every micro-batch contributes the MEAN
  cross-entropy over its non-masked targets, divided by ``grad_accum_steps``. With loss masks the
  weight of a scored token is ``1 / (active targets of its micro-batch * grad_accum_steps)``, which
  differs across arms and across micro-batch partitions (see contextualization/docs/EXPOSURE.md).
* ``fixed_nominal`` (the funded C/R/X/E experiment): every micro-batch contributes the SUM of the
  per-token cross-entropies over its valid targets divided by the NOMINAL token budget
  ``device_batch_size * max_seq_len * grad_accum_steps``. The accumulation division happens exactly
  once (here); the distributed optimizer averages gradients across ranks (``ReduceOp.AVG``), so no
  extra division by the world size is applied. Corresponding tokens get the same coefficient in
  every arm and every partition; a fully masked micro-batch contributes exactly zero (the sum is
  computed directly, never reconstructed as mean x count, which would be NaN there).

Metadata rule: a checkpoint, preflight or evaluation that records no policy is the HISTORICAL
``microbatch_mean``; missing metadata never implies the new policy.
"""

import torch

LOSS_NORMALIZATIONS = ("microbatch_mean", "fixed_nominal")
HISTORICAL_LOSS_NORMALIZATION = "microbatch_mean"
IGNORE_INDEX = -1


def recorded_policy(meta) -> str:
    """The policy a meta json / preflight dict records; an absent key is the historical one."""
    if not isinstance(meta, dict):
        return HISTORICAL_LOSS_NORMALIZATION
    policy = meta.get("loss_normalization")
    return str(policy) if policy else HISTORICAL_LOSS_NORMALIZATION


def nominal_denominator(device_batch_size: int, max_seq_len: int, grad_accum_steps: int) -> int:
    """Tokens one optimizer step nominally scores on ONE rank: B x T x grad_accum_steps."""
    return int(device_batch_size) * int(max_seq_len) * int(grad_accum_steps)


def count_active_targets(targets: torch.Tensor) -> torch.Tensor:
    """Number of valid (non-ignored) targets of a micro-batch, as a tensor on the targets' device."""
    return (targets != IGNORE_INDEX).sum()


def microbatch_loss(model, x, y, policy: str, *, device_batch_size: int, max_seq_len: int, grad_accum_steps: int):
    """The loss to call .backward() on for one micro-batch, plus (loss_sum, n_active) for logging.

    ``microbatch_mean``: ``model(x, y)`` (the mean over the non-masked targets) / grad_accum_steps,
    exactly the historical computation; loss_sum is None (never reconstructed).
    ``fixed_nominal``: ``model(x, y, loss_reduction="sum")`` / (B x T x grad_accum_steps)."""
    if policy == HISTORICAL_LOSS_NORMALIZATION:
        loss = model(x, y)
        return loss / grad_accum_steps, None, count_active_targets(y)
    if policy != "fixed_nominal":
        raise ValueError(f"unknown loss normalization {policy!r}; known: {LOSS_NORMALIZATIONS}")
    loss_sum = model(x, y, loss_reduction="sum")   # 0 for a fully masked micro-batch, never NaN
    denom = nominal_denominator(device_batch_size, max_seq_len, grad_accum_steps)
    return loss_sum / denom, loss_sum.detach(), count_active_targets(y)


class StepLossStats:
    """Accumulates the per-token loss sum and the active-target count of one optimizer step across
    its micro-batches, then across ranks (one all-reduce), and reports both the nominal training
    objective ``sum / (B x T x grad_accum_steps x world_size)`` and the ordinary loss per active
    token ``sum / active``; the latter is None when the step had no active target."""

    def __init__(self, device, *, device_batch_size: int, max_seq_len: int, grad_accum_steps: int, world_size: int):
        self.acc = torch.zeros(2, dtype=torch.float64, device=device)   # [loss_sum, active_targets]
        self.nominal_per_rank = nominal_denominator(device_batch_size, max_seq_len, grad_accum_steps)
        self.world_size = int(world_size)

    def add(self, loss_sum: torch.Tensor, n_active: torch.Tensor) -> None:
        self.acc[0] += loss_sum.detach().to(torch.float64)
        self.acc[1] += n_active.to(torch.float64)

    def reduce(self, dist_module=None) -> dict:
        """All-reduce (SUM) across ranks when a distributed module with an initialized process group
        is given, then return the step's statistics (a single CPU sync)."""
        if dist_module is not None and dist_module.is_available() and dist_module.is_initialized():
            dist_module.all_reduce(self.acc, op=dist_module.ReduceOp.SUM)
        loss_sum, active = self.acc.tolist()
        nominal = self.nominal_per_rank * self.world_size
        return {"loss_sum": loss_sum, "active_targets": int(round(active)), "nominal_targets": int(nominal),
                "loss_nominal": loss_sum / nominal, "loss_per_active_token": (loss_sum / active) if active > 0 else None}

    def reset(self) -> None:
        self.acc.zero_()
