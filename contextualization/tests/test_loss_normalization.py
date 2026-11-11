"""Loss normalization policies (decision of 2026-10-15: fixed nominal normalization for the funded C/R/X/E run).

Pure-tensor tests of nanochat/loss_normalization.py with a tiny CPU GPT:
  * unmasked rows: fixed_nominal equals the historical micro-batch mean (loss and gradients);
  * unequal masking at a fixed token budget: fixed_nominal gradients are invariant to the micro-batch
    partition, the historical mean's are not (the review's toy example);
  * a fully masked micro-batch contributes exactly 0 (finite gradients), where the mean is NaN;
  * distributed averaging: the optimizer's ReduceOp.AVG over per-rank nominal losses equals the gradient of
    sum / (B x T x grad_accum x world_size); StepLossStats reports that objective and the per-active-token loss;
  * unrecorded metadata means the historical policy, never the new one.
The resume / preflight / launcher plumbing is exercised with real tiny runs in tests/test_seeds.py.
"""
import math

import pytest
import torch

from nanochat.loss_normalization import (HISTORICAL_LOSS_NORMALIZATION, LOSS_NORMALIZATIONS, StepLossStats,
                                         count_active_targets, microbatch_loss, nominal_denominator, recorded_policy)

T_SEQ = 16
VOCAB = 256


def tiny_model(seed=0):
    from nanochat.common import compute_init
    from nanochat.gpt import GPT, GPTConfig
    compute_init(device_type="cpu", seed=seed)
    cfg = GPTConfig(sequence_len=T_SEQ, vocab_size=VOCAB, n_layer=1, n_head=2, n_kv_head=2, n_embd=32, window_pattern="L")
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device="cpu")
    model.init_weights()
    return model


def rows(n, seed=1, mask_fracs=None):
    """n rows of T_SEQ tokens; mask_fracs[i] = fraction of row i's targets set to -1 (from the end)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(1, VOCAB, (n, T_SEQ), generator=g)
    y = torch.randint(1, VOCAB, (n, T_SEQ), generator=g)
    for i, frac in enumerate(mask_fracs or []):
        k = int(round(frac * T_SEQ))
        if k:
            y[i, T_SEQ - k:] = -1
    return x, y


def grads_of(model, x, y, policy, B, grad_accum):
    """Accumulated parameter gradients of one optimizer step that consumes rows x, y as grad_accum
    micro-batches of B rows (exactly what base_train does, incl. the single accumulation division)."""
    model.zero_grad(set_to_none=True)
    stats = StepLossStats("cpu", device_batch_size=B, max_seq_len=T_SEQ, grad_accum_steps=grad_accum, world_size=1)
    for m in range(grad_accum):
        xb, yb = x[m * B:(m + 1) * B], y[m * B:(m + 1) * B]
        loss, loss_sum, n_active = microbatch_loss(model, xb, yb, policy, device_batch_size=B, max_seq_len=T_SEQ, grad_accum_steps=grad_accum)
        if loss_sum is not None:
            stats.add(loss_sum, n_active)
        loss.backward()
    g = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}
    return g, stats.reduce()


def same(g1, g2, rtol=1e-5, atol=1e-6):
    return g1.keys() == g2.keys() and all(torch.allclose(g1[k], g2[k], rtol=rtol, atol=atol) for k in g1)


def test_policies_and_recorded_policy_rules():
    assert LOSS_NORMALIZATIONS == ("microbatch_mean", "fixed_nominal") and HISTORICAL_LOSS_NORMALIZATION == "microbatch_mean"
    assert recorded_policy({}) == "microbatch_mean" and recorded_policy(None) == "microbatch_mean"      # unrecorded = historical
    assert recorded_policy({"loss_normalization": None}) == "microbatch_mean"
    assert recorded_policy({"loss_normalization": "fixed_nominal"}) == "fixed_nominal"
    assert nominal_denominator(8, 2048, 64) == 8 * 2048 * 64
    with pytest.raises(ValueError):
        microbatch_loss(tiny_model(), *rows(1), "global_mean", device_batch_size=1, max_seq_len=T_SEQ, grad_accum_steps=1)


def test_unmasked_rows_fixed_nominal_equals_the_historical_mean():
    model = tiny_model()
    x, y = rows(4)
    for B, ga in ((1, 4), (2, 2), (4, 1)):
        g_hist, _ = grads_of(model, x, y, "microbatch_mean", B, ga)
        g_fix, st = grads_of(model, x, y, "fixed_nominal", B, ga)
        assert same(g_hist, g_fix), (B, ga)
        # with every target active the nominal objective IS the ordinary mean over all B*T*GA targets
        assert st["active_targets"] == st["nominal_targets"] == 4 * T_SEQ
        assert st["loss_nominal"] == pytest.approx(st["loss_per_active_token"], rel=1e-9)
    xb, yb = x[:2], y[:2]
    loss_mean = model(xb, yb)
    loss_fix, loss_sum, n_active = microbatch_loss(model, xb, yb, "fixed_nominal", device_batch_size=2, max_seq_len=T_SEQ, grad_accum_steps=1)
    assert torch.allclose(loss_fix, loss_mean, rtol=1e-6) and int(n_active) == 2 * T_SEQ
    assert torch.allclose(loss_sum, loss_mean * (2 * T_SEQ), rtol=1e-6)


def test_fixed_nominal_is_invariant_to_the_microbatch_partition_the_mean_is_not():
    """The review's toy case: the same rows with unequal masks split 1/3 vs 2/2 vs 4 per micro-batch; the token
    budget per step is fixed (4 rows x T), so fixed_nominal gives identical gradients while the historical mean
    weights each token by its own micro-batch's active count."""
    model = tiny_model(3)
    x, y = rows(4, seed=5, mask_fracs=[0.0, 0.75, 0.0, 0.5])   # micro-batches of two rows: 20 vs 24 active targets
    partitions = ((1, 4), (2, 2), (4, 1))
    fixed = [grads_of(model, x, y, "fixed_nominal", B, ga) for B, ga in partitions]
    for g, st in fixed[1:]:
        assert same(fixed[0][0], g)
        assert st["loss_sum"] == pytest.approx(fixed[0][1]["loss_sum"], rel=1e-5) and st["active_targets"] == fixed[0][1]["active_targets"]
    hist = [grads_of(model, x, y, "microbatch_mean", B, ga)[0] for B, ga in partitions]
    assert not same(hist[0], hist[1]) and not same(hist[1], hist[2])
    # the scalar objective: sum over valid tokens / (B*T*GA) is the same number for every partition
    active = int(count_active_targets(y))
    assert active == 4 * T_SEQ - (12 + 8)
    assert fixed[0][1]["nominal_targets"] == 4 * T_SEQ and fixed[0][1]["active_targets"] == active
    assert fixed[0][1]["loss_per_active_token"] == pytest.approx(fixed[0][1]["loss_sum"] / active)


def test_fully_masked_microbatch_contributes_zero_and_stays_finite():
    model = tiny_model(7)
    x, y = rows(2, seed=9)
    y_masked = torch.full_like(y, -1)
    # the historical mean is NaN on such a micro-batch (0/0): the hazard the preflight counts (microbatches_with_zero_active)
    assert torch.isnan(model(x, y_masked))
    loss, loss_sum, n_active = microbatch_loss(model, x, y_masked, "fixed_nominal", device_batch_size=2, max_seq_len=T_SEQ, grad_accum_steps=3)
    assert float(loss.detach()) == 0.0 and float(loss_sum) == 0.0 and int(n_active) == 0
    model.zero_grad(set_to_none=True)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) == 0.0 for p in model.parameters() if p.grad is not None)
    # within a step: the masked micro-batch adds nothing, the other micro-batches keep their nominal weights
    x2, y2 = rows(2, seed=11)
    xx, yy = torch.cat([x, x2]), torch.cat([y_masked, y2])
    g_step, st = grads_of(model, xx, yy, "fixed_nominal", 2, 2)
    model.zero_grad(set_to_none=True)
    only, _, _ = microbatch_loss(model, x2, y2, "fixed_nominal", device_batch_size=2, max_seq_len=T_SEQ, grad_accum_steps=2)
    only.backward()
    g_only = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}
    assert same(g_step, g_only) and st["active_targets"] == 2 * T_SEQ and st["nominal_targets"] == 4 * T_SEQ
    stats = StepLossStats("cpu", device_batch_size=2, max_seq_len=T_SEQ, grad_accum_steps=1, world_size=1)
    stats.add(torch.tensor(0.0), torch.tensor(0))
    assert stats.reduce()["loss_per_active_token"] is None and stats.reduce()["loss_nominal"] == 0.0


def test_distributed_averaging_needs_no_extra_division():
    """Two ranks, each with its own micro-batches: the distributed optimizer averages the per-rank gradients
    (ReduceOp.AVG). With fixed_nominal that average equals the gradient of the global objective
    sum_all_ranks / (B x T x grad_accum x world_size), so no division by the world size may be added."""
    model = tiny_model(13)
    B, ga, W = 2, 2, 2
    x, y = rows(B * ga * W, seed=17, mask_fracs=[0.5, 0.0, 0.25, 0.0, 0.0, 0.75, 0.5, 0.125])
    per_rank, sums = [], []
    for r in range(W):
        xr, yr = x[r * B * ga:(r + 1) * B * ga], y[r * B * ga:(r + 1) * B * ga]
        g, st = grads_of(model, xr, yr, "fixed_nominal", B, ga)   # what rank r computes before the all-reduce
        per_rank.append(g)
        sums.append(st["loss_sum"])
    averaged = {k: (per_rank[0][k] + per_rank[1][k]) / W for k in per_rank[0]}
    # the global objective, computed directly on all rows: total sum / (B*T*GA*W)
    model.zero_grad(set_to_none=True)
    total = sum(model(x[i:i + 1], y[i:i + 1], loss_reduction="sum") for i in range(B * ga * W))
    (total / nominal_denominator(B, T_SEQ, ga) / W).backward()
    total = float(total.detach())
    g_global = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}
    assert same(averaged, g_global)
    # StepLossStats with world_size=2 and the two ranks' sums added (what the all-reduce SUM produces)
    stats = StepLossStats("cpu", device_batch_size=B, max_seq_len=T_SEQ, grad_accum_steps=ga, world_size=W)
    stats.add(torch.tensor(sums[0]), count_active_targets(y[:B * ga]))
    stats.add(torch.tensor(sums[1]), count_active_targets(y[B * ga:]))
    st = stats.reduce()
    assert st["nominal_targets"] == B * T_SEQ * ga * W and st["loss_nominal"] == pytest.approx(total / (B * T_SEQ * ga * W), rel=1e-6)
    assert st["active_targets"] == int(count_active_targets(y)) and st["loss_per_active_token"] == pytest.approx(total / int(count_active_targets(y)), rel=1e-6)
    assert math.isfinite(st["loss_nominal"])
