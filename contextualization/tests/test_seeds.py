"""Seed control (REVIEW finding 7) and training provenance (REVIEW 2026-10-14): compute_init(seed=...)
pins the model initialization, the default stays 42, and scripts/base_train.py exposes --init-seed /
--data-seed / --data-loader / --skip-preflight-check. The argument plumbing is checked on the source
text (importing base_train would start a training run); the preflight gate, the resume provenance
(original init_seed kept, cli_init_seed / resume_from_step recorded, data stream bound) and the
init_param_sha256 identity are checked with REAL tiny CPU runs of scripts.base_train, and so is the exact-resume
schedule check (review 2026-10-14 recheck, finding 4: a resume keeps the checkpoint's accumulation / batch / horizon
and its saved loader position must match the checkpoint's own schedule)."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
BASE_TRAIN = REPO / "scripts" / "base_train.py"
PY = sys.executable


def tiny_gpt(seed=None):
    from nanochat.common import compute_init
    from nanochat.gpt import GPT, GPTConfig
    if seed is None:
        compute_init(device_type="cpu")
    else:
        compute_init(device_type="cpu", seed=seed)
    cfg = GPTConfig(sequence_len=32, vocab_size=256, n_layer=2, n_head=2, n_kv_head=2, n_embd=64, window_pattern="L")
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device="cpu")
    model.init_weights()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_same_seed_identical_parameters():
    assert _same(tiny_gpt(seed=7), tiny_gpt(seed=7))
    assert _same(tiny_gpt(seed=123), tiny_gpt(seed=123))


def test_different_seeds_differ():
    a, b = tiny_gpt(seed=7), tiny_gpt(seed=8)
    assert a.keys() == b.keys()
    differing = [k for k in a if a[k].dtype.is_floating_point and a[k].numel() > 1 and not torch.equal(a[k], b[k])]
    assert "transformer.wte.weight" in differing and "lm_head.weight" in differing
    assert any(k.startswith("transformer.h.0.attn") for k in differing)


def test_default_seed_is_42():
    assert _same(tiny_gpt(), tiny_gpt(seed=42))
    assert not _same(tiny_gpt(), tiny_gpt(seed=43))


def test_compute_init_signature():
    import inspect
    from nanochat.common import compute_init
    sig = inspect.signature(compute_init)
    assert list(sig.parameters) == ["device_type", "seed"]
    assert sig.parameters["seed"].default == 42 and sig.parameters["device_type"].default == "cuda"


def _add_argument_calls(src):
    return re.findall(r'parser\.add_argument\("(--[a-z\-]+)"(.*?)\)\n', src)


def test_base_train_exposes_seed_and_loader_flags():
    src = BASE_TRAIN.read_text()
    flags = dict(_add_argument_calls(src))
    assert "--init-seed" in flags and "type=int" in flags["--init-seed"] and "default=42" in flags["--init-seed"]
    assert "--data-seed" in flags and "type=int" in flags["--data-seed"] and "default=-1" in flags["--data-seed"]
    assert "--data-loader" in flags and 'default="bestfit"' in flags["--data-loader"]
    assert 'choices=["bestfit", "prepacked"]' in flags["--data-loader"]
    assert "--skip-preflight-check" in flags and 'action="store_true"' in flags["--skip-preflight-check"]
    # wiring: the seed reaches compute_init, the loader choice selects the prepacked loader for train only,
    # and the tokenizer is handed to the prepacked loader so the rows are bound to its encoding
    assert "compute_init(device_type, seed=args.init_seed)" in src
    assert 'if args.data_loader == "prepacked":' in src
    assert "prepacked_distributed_data_loader_with_state(args.device_batch_size, args.max_seq_len, split=\"train\"" in src
    assert "data_seed=args.data_seed, tokenizer=tokenizer)" in src
    assert 'tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val"' in src
    # checkpoint meta records seeds, loader, code revision, nondeterminism sources, bindings, provenance of resumes
    for key in ('"init_seed": init_seed', '"cli_init_seed": cli_init_seed', '"init_param_sha256": init_param_sha256',
                '"resume_from_step": resume_from_step', '"preflight": preflight_status', '"data_seed": args.data_seed',
                '"data_loader": args.data_loader', '"code": provenance_code', '"nondeterminism": provenance_nondeterminism',
                '"bindings": provenance_bindings', '"wandb_run": None if use_dummy_wandb else {',
                '"dataloader_state_dict": dataloader_state_dict'):
        assert key in src, key
    for key in ("tokenizer_sha256", "encoding_fingerprint", "data_listing_sha256", "prepack_manifest_sha256", "prepack_dataset_id",
                "preflight_sha256", "preflight_path", "build_summary_sha256", "probe_hashes",
                "torch_compile", "float32_matmul_precision", "flash_attention_3", "cudnn_deterministic", "cudnn_benchmark",
                "git_commit", "git_dirty"):
        assert f'"{key}"' in src, key
    # nanochat must not import the contextualization package for this
    assert "from ctx" not in src and "import ctx" not in src
    # a checkpoint from one loader cannot be resumed with the other; the data seed and the prepacked rows must match too
    assert "resumed_loader != args.data_loader" in src
    assert '(("data_loader", args.data_loader), ("data_seed", args.data_seed))' in src
    assert 'for key in ("prepack_manifest_sha256", "prepack_dataset_id"):' in src and "ckpt_bindings[key] != prepack_identity[key]" in src
    # the preflight gate: <base_dir>/../exposure/PREFLIGHT_OK.json must match B/T/GA/N/world/seed/dataset id/encoding
    assert 'os.path.join(exp_dir_of_arm, "exposure", "PREFLIGHT_OK.json")' in src
    for key in ('"B": args.device_batch_size', '"T": args.max_seq_len', '"grad_accum_steps": grad_accum_steps',
                '"num_iterations": num_iterations', '"world_size": ddp_world_size', '"data_seed": args.data_seed',
                'preflight_ids[arm_name] != prepack_identity["prepack_dataset_id"]',
                'preflight.get("encoding_fingerprint") != tokenizer_encoding_fingerprint'):
        assert key in src, key
    # the init identity is hashed right after init_weights() and carried forward on resume, never recomputed
    assert src.index("model.init_weights()") < src.index("init_param_sha256 = _state_dict_sha256(model.state_dict())") < src.index("if resuming:")
    assert 'init_param_sha256 = meta_data.get("init_param_sha256")' in src
    # the historical objective is untouched (per-micro-batch mean divided by grad_accum_steps) and selected by default;
    # fixed_nominal (decision 2026-10-15) is an explicit choice recorded in the meta, certified by the preflight and bound on resume
    assert "loss = loss / grad_accum_steps" in src
    assert "--loss-normalization" in flags and "choices=list(LOSS_NORMALIZATIONS)" in flags["--loss-normalization"]
    assert "default=HISTORICAL_LOSS_NORMALIZATION" in flags["--loss-normalization"]
    for key in ('"loss_normalization": args.loss_normalization', '"loss_denominator_per_microbatch"',
                "microbatch_loss(model, x, y, args.loss_normalization", "step_stats.reduce(dist)",
                "recorded_policy(preflight) != args.loss_normalization", "a policy change on resume is refused",
                '"train/loss_nominal"', '"train/loss_per_active_token"'):
        assert key in src, key
    assert src.index("ckpt_loss_normalization = recorded_policy(meta_data)") < src.index("SCHEDULE_KEYS = (")
    # exact resume keeps the checkpoint's schedule (review 2026-10-14 recheck, finding 4): the RESOLVED schedule is recorded
    # in the meta block and compared on resume once grad_accum_steps is known, before the preflight gate and the training loop;
    # a prepacked resume also checks the saved loader position against the checkpoint's own schedule
    meta_block = src[src.index('"user_config": user_config, # inputs to the training script'):src.index('"loop_state": {')]
    for key in ('"device_batch_size": args.device_batch_size', '"max_seq_len": args.max_seq_len', '"total_batch_size": total_batch_size',
                '"num_iterations": num_iterations', '"grad_accum_steps": grad_accum_steps', '"world_size": ddp_world_size'):
        assert key in meta_block, key
    assert 'SCHEDULE_KEYS = ("device_batch_size", "max_seq_len", "total_batch_size", "grad_accum_steps", "world_size", "num_iterations")' in src
    for text in ("the requested training schedule differs from the checkpoint's", "no segmented exposure history is supported",
                 "schedule provenance missing", "does not match the checkpoint's schedule",
                 "refusing to resume on an inconsistent exposure history", 'out["n_rows"] = man.get("n_rows")'):
        assert text in src, text
    assert (src.index("grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd") < src.index("ckpt_schedule = _checkpoint_schedule(meta_data)")
            < src.index("consumed_rows != expected_rows") < src.index('os.path.join(exp_dir_of_arm, "exposure", "PREFLIGHT_OK.json")')
            < src.index("while True:"))


def test_run_arm_requires_explicit_loader_for_v2_and_derives_model_tags(tmp_path):
    import run_arm
    assert run_arm.model_tag_for(24) == "d24" and run_arm.model_tag_for(24, 7) == "d24_s7" and run_arm.model_tag_for(24, 7, "custom") == "custom"
    assert run_arm.loader_from_extra(["--device-batch-size=8", "--data-loader=prepacked"]) == "prepacked"
    assert run_arm.loader_from_extra(["--data-loader", "bestfit"]) == "bestfit" and run_arm.loader_from_extra(["--x"]) is None
    cmd = run_arm.build_train_command(4, "ctx_R_d4_s7", 1, [], init_seed=7, data_loader="prepacked", model_tag="d4_s7", python="py")
    assert cmd == ["py", "-m", "scripts.base_train", "--depth=4", "--run=ctx_R_d4_s7", "--init-seed=7", "--data-loader=prepacked", "--model-tag=d4_s7"]
    assert run_arm.training_record(cmd, None, model_tag="d4_s7")["model_tag"] == "d4_s7"
    assert run_arm.training_record(cmd, None)["loss_normalization"] == "microbatch_mean (base_train default)"
    cmd = run_arm.build_train_command(4, "r", 1, [], loss_normalization="fixed_nominal", python="py")
    assert cmd[-1] == "--loss-normalization=fixed_nominal" and run_arm.training_record(cmd, None, loss_normalization="fixed_nominal")["loss_normalization"] == "fixed_nominal"
    with pytest.raises(ValueError):
        run_arm.build_train_command(4, "r", 1, [], loss_normalization="global_mean")
    src = (REPO / "contextualization" / "run_arm.py").read_text()
    assert "choose --data-loader prepacked (corrected path, needs the preflight) or bestfit (historical loader) explicitly" in src
    assert '"--model-tag", model_tag' in src  # the eval step gets the same tag as training
    # protocol detection and the preflight check used before launching
    (tmp_path / "build_summary.json").write_text(json.dumps({"protocol": "v2-corrected-2026-10", "config": {"seed": 1}}))
    assert run_arm.build_protocol(tmp_path) == "v2-corrected-2026-10"
    with pytest.raises(SystemExit, match="needs .*PREFLIGHT_OK.json"):
        run_arm.check_preflight(tmp_path, "R")
    (tmp_path / "exposure").mkdir()
    (tmp_path / "exposure" / "PREFLIGHT_OK.json").write_text(json.dumps({"prepack_dataset_id": {"R": "a" * 64}}))
    with pytest.raises(SystemExit, match="does not cover arm X"):
        run_arm.check_preflight(tmp_path, "X")
    (tmp_path / "arm_R" / "prepacked").mkdir(parents=True)
    (tmp_path / "arm_R" / "prepacked" / "prepack_manifest.json").write_text(json.dumps({"prepack_dataset_id": "b" * 64}))
    with pytest.raises(SystemExit, match="is stale for arm R"):
        run_arm.check_preflight(tmp_path, "R")
    (tmp_path / "arm_R" / "prepacked" / "prepack_manifest.json").write_text(json.dumps({"prepack_dataset_id": "a" * 64}))
    assert run_arm.check_preflight(tmp_path, "R")["prepack_dataset_id"]["R"] == "a" * 64
    # the preflight certifies ONE training objective; a preflight without the key certified the historical one
    with pytest.raises(SystemExit, match="certifies loss_normalization 'microbatch_mean' \\(unrecorded: historical default\\) but this launch requests 'fixed_nominal'"):
        run_arm.check_preflight(tmp_path, "R", "fixed_nominal")
    (tmp_path / "exposure" / "PREFLIGHT_OK.json").write_text(json.dumps({"prepack_dataset_id": {"R": "a" * 64}, "loss_normalization": "fixed_nominal"}))
    assert run_arm.check_preflight(tmp_path, "R", "fixed_nominal")["loss_normalization"] == "fixed_nominal"
    with pytest.raises(SystemExit, match="certifies loss_normalization 'fixed_nominal' but this launch requests 'microbatch_mean'"):
        run_arm.check_preflight(tmp_path, "R")
    # init identity comparison across arms (same depth + effective init seed)
    from ctx import tracking as T
    T.write_run_info(tmp_path, "C", depth=4, init_param_sha256="c" * 64, init_seed_effective=7)
    T.write_run_info(tmp_path, "X", depth=4, init_param_sha256="d" * 64, init_seed_effective=7)
    T.write_run_info(tmp_path, "E", depth=4, init_param_sha256="c" * 64, init_seed_effective=8)   # other seed: not compared
    lines = run_arm.compare_init_identity(tmp_path, "R", {"init_param_sha256": "c" * 64, "init_seed_effective": 7}, 4)
    assert len(lines) == 2 and lines[0].startswith("init params MATCH: arm R vs arm C") and lines[1].startswith("init params DIFFER: arm R vs arm X")
    assert run_arm.compare_init_identity(tmp_path, "R", None, 4) == []


# --------------------------------------------------------------------------------------
# real tiny CPU runs of scripts.base_train with the prepacked loader (depth 2, T=96, 10 steps of 2 rows)

TRAIN_ARGS = ["--device-type", "cpu", "--depth", "2", "--aspect-ratio", "32", "--head-dim", "32", "--max-seq-len", "96",
              "--device-batch-size", "2", "--total-batch-size", "192", "--eval-every", "-1", "--core-metric-every", "-1",
              "--sample-every", "-1", "--window-pattern", "L", "--run", "dummy", "--data-loader", "prepacked", "--data-seed", "0"]


def _write_token_bytes(tok, tok_dir):
    special = set(tok.get_special_tokens())
    tb = [0 if tok.decode([i]) in special else len(tok.decode([i]).encode("utf-8")) for i in range(tok.get_vocab_size())]
    torch.save(torch.tensor(tb, dtype=torch.int32), Path(tok_dir) / "token_bytes.pt")


@pytest.fixture(scope="module")
def tiny_exp(tmp_path_factory, small_bpe_tokenizer):
    """An experiment dir whose arms are ready for base_train --data-loader prepacked: tokenizer (+ token_bytes.pt),
    base_data, prepacked rows and a PREFLIGHT_OK.json for B=2, T=96, grad_accum=1, n_rows/2 iterations, data_seed 0."""
    import exposure_audit
    from tests.test_prepacked_loader import ARMS, build_experiment, prepack
    exp = tmp_path_factory.mktemp("tiny_train")
    build_experiment(exp, small_bpe_tokenizer)
    for a in ARMS:
        small_bpe_tokenizer.save(str(exp / f"arm_{a}" / "tokenizer"))
        _write_token_bytes(small_bpe_tokenizer, exp / f"arm_{a}" / "tokenizer")
    result = prepack(exp, small_bpe_tokenizer)
    n_rows = result["n_rows"]
    rc = exposure_audit.main(["--exp-dir", str(exp), "--arms", *ARMS, "--mode", "prepacked", "--B", "2", "--grad-accum", "1",
                              "--num-iterations", str(n_rows // 2), "--data-seed", "0"])
    assert rc == 0
    return exp, n_rows


def base_train(arm_dir, *extra, expect=0):
    env = dict(os.environ, NANOCHAT_BASE_DIR=str(arm_dir), PYTHONDONTWRITEBYTECODE="1")
    for k in ("WANDB_RUN_ID", "RANK", "LOCAL_RANK", "WORLD_SIZE"):
        env.pop(k, None)
    r = subprocess.run([PY, "-m", "scripts.base_train", *TRAIN_ARGS, *extra], env=env, cwd=REPO, capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == expect, (f"exit {r.returncode} (expected {expect})\n--- stdout tail:\n{r.stdout[-3000:]}"
                                    f"\n--- stderr tail:\n{r.stderr[-3000:]}")
    return r


def meta(arm_dir, tag, step):
    return json.loads((arm_dir / "base_checkpoints" / tag / f"meta_{step:06d}.json").read_text())


def test_tiny_cpu_training_preflight_resume_provenance_and_init_identity(tiny_exp, small_bpe_tokenizer):
    from ctx.prepack import encoding_fingerprint
    from ctx.protocol import sha256_file
    exp, n_rows = tiny_exp
    arm = exp / "arm_R"
    N = n_rows // 2
    pre = exp / "exposure" / "PREFLIGHT_OK.json"
    # 1. a run that matches the preflight: provenance in the final meta
    base_train(arm, "--num-iterations", str(N), "--save-every", "5", "--init-seed", "7", "--model-tag", "d2_s7")
    m = meta(arm, "d2_s7", N)
    assert m["preflight"] == "ok" and m["init_seed"] == 7 and m["cli_init_seed"] is None and m["resume_from_step"] is None
    assert m["data_loader"] == "prepacked" and m["data_seed"] == 0 and m["step"] == N
    h7 = m["init_param_sha256"]
    assert isinstance(h7, str) and len(h7) == 64
    b = m["bindings"]
    assert b["preflight_path"] == str(pre) and b["preflight_sha256"] == sha256_file(pre)
    man = json.loads((arm / "prepacked" / "prepack_manifest.json").read_text())
    assert b["prepack_dataset_id"] == man["prepack_dataset_id"] == m["dataloader_state_dict"]["prepack_dataset_id"]
    assert b["prepack_manifest_sha256"] == sha256_file(arm / "prepacked" / "prepack_manifest.json")
    assert b["encoding_fingerprint"] == encoding_fingerprint(small_bpe_tokenizer) == man["encoding_fingerprint"]
    assert b["tokenizer_sha256"] == sha256_file(arm / "tokenizer" / "tokenizer.pkl")
    states = {s: meta(arm, "d2_s7", s)["dataloader_state_dict"] for s in (5, N)}
    assert states[5]["loader"] == "prepacked" and states[5]["cursor"] == 10
    # 2. the same --init-seed gives the same initial parameters (init_param_sha256); another seed differs
    base_train(arm, "--num-iterations", str(N), "--init-seed", "7", "--model-tag", "d2_s7_again")
    base_train(arm, "--num-iterations", str(N), "--init-seed", "8", "--model-tag", "d2_s8")
    assert meta(arm, "d2_s7_again", N)["init_param_sha256"] == h7
    assert meta(arm, "d2_s8", N)["init_param_sha256"] != h7
    # 3. a resume keeps the ORIGINAL init identity, records the CLI seed and the resume step, replays the same data
    for f in (arm / "base_checkpoints" / "d2_s7").glob(f"*_{N:06d}*"):
        f.unlink()
    r = base_train(arm, "--num-iterations", str(N), "--save-every", "5", "--init-seed", "42", "--model-tag", "d2_s7",
                   "--resume-from-step", "5")
    assert "WARNING: the checkpoint was initialised with init_seed=7" in r.stdout
    m2 = meta(arm, "d2_s7", N)
    assert m2["init_seed"] == 7 and m2["cli_init_seed"] == 42 and m2["resume_from_step"] == 5
    assert m2["init_param_sha256"] == h7 and m2["preflight"] == "ok" and m2["dataloader_state_dict"] == states[N]
    # ... a different data seed on resume is refused (the data stream is the checkpoint's)
    r = base_train(arm, "--num-iterations", str(N), "--init-seed", "7", "--model-tag", "d2_s7", "--resume-from-step", "5",
                   "--data-seed", "1", expect=1)
    assert "checkpoint was trained with data_seed=0 but data_seed=1 was requested" in r.stderr
    # ... and so is a prepack manifest that is not the one the checkpoint was bound to (same rows, re-serialized)
    man_path = arm / "prepacked" / "prepack_manifest.json"
    man_bytes = man_path.read_bytes()
    try:
        man_path.write_text(json.dumps(json.loads(man_bytes)))
        assert sha256_file(man_path) != b["prepack_manifest_sha256"]
        r = base_train(arm, "--num-iterations", str(N), "--init-seed", "7", "--model-tag", "d2_s7", "--resume-from-step", "5", expect=1)
        assert "prepack_manifest_sha256" in r.stderr and "refusing to resume on different data" in r.stderr
    finally:
        man_path.write_bytes(man_bytes)
    # 4. the preflight gate: a configuration the preflight was not written for is refused ...
    r = base_train(arm, "--num-iterations", str(N + 1), "--init-seed", "7", "--model-tag", "d2_gate", expect=1)
    assert "does not match this run" in r.stderr and "num_iterations: preflight" in r.stderr
    assert not (arm / "base_checkpoints" / "d2_gate").exists()
    # ... unless skipped explicitly, which the meta records
    r = base_train(arm, "--num-iterations", "1", "--init-seed", "7", "--model-tag", "d2_skip", "--skip-preflight-check")
    assert "WARNING: --skip-preflight-check" in r.stdout and meta(arm, "d2_skip", 1)["preflight"] == "skipped"
    # ... and a missing preflight file is refused
    backup = pre.read_text()
    pre.unlink()
    try:
        r = base_train(arm, "--num-iterations", str(N), "--init-seed", "7", "--model-tag", "d2_nopre", expect=1)
        assert "no preflight for the prepacked rows" in r.stderr
    finally:
        pre.write_text(backup)


def _saved_steps(tag_dir):
    return sorted(int(p.stem.split("_")[1]) for p in tag_dir.glob("meta_*.json"))


def test_tiny_cpu_resume_refuses_schedule_change_and_wrong_loader_position(tiny_exp):
    """Review 2026-10-14 recheck, finding 4 (reviews/2026-10-14-recheck/repro/review3_resume_ga.py): with a VALID preflight
    for grad_accum 2 and the same horizon, a resume from step 5 of a grad_accum-1 checkpoint with --total-batch-size 384
    used to finish with preflight "ok" after consuming 15 instead of the certified 20 micro-batches (loader at epoch 1
    cursor 10 instead of epoch 2 cursor 0). An exact resume now keeps the checkpoint's schedule: a changed total batch
    size / accumulation or horizon is refused before the preflight gate and before any step, a saved loader position
    that does not match the checkpoint's own schedule is refused too, and the same-configuration resume still works."""
    import exposure_audit
    from tests.test_prepacked_loader import ARMS
    exp, n_rows = tiny_exp
    arm = exp / "arm_R"
    N = n_rows // 2
    pre = exp / "exposure" / "PREFLIGHT_OK.json"
    tag_dir = arm / "base_checkpoints" / "d2_ga"
    run_args = ("--num-iterations", str(N), "--save-every", "5", "--init-seed", "7", "--model-tag", "d2_ga")
    preflight_args = ["--exp-dir", str(exp), "--arms", *ARMS, "--mode", "prepacked", "--B", "2", "--num-iterations", str(N), "--data-seed", "0"]
    original_preflight = pre.read_bytes()  # the fixture is shared: put the grad_accum-1 preflight back whatever happens
    try:
        # the uninterrupted run (B=2, grad_accum 1, N steps) records the RESOLVED schedule in every meta
        base_train(arm, *run_args)
        m5, mN = meta(arm, "d2_ga", 5), meta(arm, "d2_ga", N)
        for m in (m5, mN):
            assert m["num_iterations"] == N and m["grad_accum_steps"] == 1 and m["world_size"] == 1
            assert m["total_batch_size"] == 192 and m["device_batch_size"] == 2 and m["max_seq_len"] == 96
        assert m5["dataloader_state_dict"]["epoch"] == 0 and m5["dataloader_state_dict"]["cursor"] == 10  # 5 steps x 1 x B=2 x W=1 rows
        final_state = mN["dataloader_state_dict"]
        for p in tag_dir.iterdir():  # keep only the step-5 checkpoint
            if int(p.stem.split("_")[1]) != 5:
                p.unlink()
        assert _saved_steps(tag_dir) == [5]
        # 1. a valid preflight for grad_accum 2 and the same horizon exists, but the checkpoint was trained with grad_accum 1:
        #    the resume with --total-batch-size 384 (= grad_accum 2) is refused, naming both keys, and writes no checkpoint
        assert exposure_audit.main([*preflight_args, "--grad-accum", "2"]) == 0
        assert json.loads(pre.read_text())["grad_accum_steps"] == 2
        r = base_train(arm, *run_args, "--resume-from-step", "5", "--total-batch-size", "384", expect=1)
        assert "the requested training schedule differs from the checkpoint's" in r.stderr
        assert "total_batch_size: checkpoint 192 vs requested 384" in r.stderr and "grad_accum_steps: checkpoint 1 vs requested 2" in r.stderr
        assert "no segmented exposure history is supported" in r.stderr and "does not match this run" not in r.stderr  # not the preflight gate
        assert _saved_steps(tag_dir) == [5]
        # 2. a changed horizon is refused the same way (num_iterations is part of the schedule)
        r = base_train(arm, "--num-iterations", str(N + 1), "--save-every", "5", "--init-seed", "7", "--model-tag", "d2_ga",
                       "--resume-from-step", "5", expect=1)
        assert "the requested training schedule differs from the checkpoint's" in r.stderr
        assert f"num_iterations: checkpoint {N} vs requested {N + 1}" in r.stderr and "total_batch_size:" not in r.stderr
        assert _saved_steps(tag_dir) == [5]
        # 3. a saved loader position that is valid for the loader (cursor 12: a multiple of B*W inside the epoch) but not where
        #    the checkpoint's own schedule leaves it (5 steps x 1 micro-batch x 2 rows = 10) is refused
        meta5_path = tag_dir / "meta_000005.json"
        meta5_bytes = meta5_path.read_bytes()
        try:
            tampered = json.loads(meta5_bytes)
            tampered["dataloader_state_dict"]["cursor"] = 12
            meta5_path.write_text(json.dumps(tampered, indent=2))
            r = base_train(arm, *run_args, "--resume-from-step", "5", expect=1)
            assert "the loader position saved at step 5 does not match the checkpoint's schedule" in r.stderr
            assert "cursor 12 = 12 consumed rows" in r.stderr and "5 steps x 1 micro-batches x 2 rows = 10" in r.stderr
        finally:
            meta5_path.write_bytes(meta5_bytes)
        assert _saved_steps(tag_dir) == [5]
        # 4. with the grad_accum-1 preflight back, the same-configuration resume is accepted and ends in the uninterrupted
        #    run's loader state (exact resume), with the schedule recorded again
        assert exposure_audit.main([*preflight_args, "--grad-accum", "1"]) == 0
        r = base_train(arm, *run_args, "--resume-from-step", "5")
        assert "Resume schedule check OK" in r.stdout and "Resume loader position OK: epoch 0 cursor 10 = 10 consumed rows" in r.stdout
        m2 = meta(arm, "d2_ga", N)
        assert m2["resume_from_step"] == 5 and m2["preflight"] == "ok" and m2["dataloader_state_dict"] == final_state
        assert m2["num_iterations"] == N and m2["grad_accum_steps"] == 1 and m2["world_size"] == 1 and m2["total_batch_size"] == 192
        assert _saved_steps(tag_dir) == [5, N]
    finally:
        pre.write_bytes(original_preflight)


def test_tiny_cpu_fixed_nominal_is_certified_recorded_logged_and_bound_on_resume(tiny_exp):
    """Decision 2026-10-15: fixed nominal normalization is an explicit policy. Real tiny runs: the historical preflight
    does not certify it; a preflight written for it refuses a microbatch_mean run; the run records the policy and logs both
    the nominal objective and the loss per active token; a resume keeps the policy (a change is refused, including a
    historical checkpoint without the key resumed under the new policy)."""
    import re
    import exposure_audit
    from tests.test_prepacked_loader import ARMS
    exp, n_rows = tiny_exp
    arm = exp / "arm_R"
    N = n_rows // 2
    pre = exp / "exposure" / "PREFLIGHT_OK.json"
    backup = pre.read_bytes()
    preflight_args = ["--exp-dir", str(exp), "--arms", *ARMS, "--mode", "prepacked", "--B", "2", "--grad-accum", "1",
                      "--num-iterations", str(N), "--data-seed", "0"]
    fixed = ["--loss-normalization", "fixed_nominal"]
    try:
        # 1. the module fixture's preflight certifies the default (historical) policy: it does not certify fixed_nominal;
        #    a preflight written BEFORE the key existed (stripped here) is read as the historical policy too
        assert json.loads(pre.read_text())["loss_normalization"] == "microbatch_mean"
        r = base_train(arm, "--num-iterations", str(N), "--init-seed", "7", "--model-tag", "d2_fn", *fixed, expect=1)
        assert "does not match this run" in r.stderr and "loss_normalization: preflight 'microbatch_mean' vs this run 'fixed_nominal'" in r.stderr
        legacy_pf = json.loads(pre.read_text())
        legacy_pf.pop("loss_normalization")
        pre.write_text(json.dumps(legacy_pf))
        r = base_train(arm, "--num-iterations", str(N), "--init-seed", "7", "--model-tag", "d2_fn", *fixed, expect=1)
        assert "loss_normalization: preflight 'microbatch_mean' (unrecorded: historical default) vs this run 'fixed_nominal'" in r.stderr
        assert not (arm / "base_checkpoints" / "d2_fn").exists()
        # 2. a preflight that certifies fixed_nominal records it and the nominal denominators; a microbatch_mean run is refused against it
        assert exposure_audit.main([*preflight_args, *fixed]) == 0
        pf = json.loads(pre.read_text())
        assert pf["loss_normalization"] == "fixed_nominal" and pf["loss_denominators"]["per_microbatch_nominal"] == 2 * 96 * 1
        assert pf["loss_denominators"]["per_step_all_ranks_nominal"] == 2 * 96 * 1 * 1
        assert json.loads((exp / "exposure" / "cross_arm.json").read_text())["preflight_config"]["loss_normalization"] == "fixed_nominal"
        r = base_train(arm, "--num-iterations", str(N), "--init-seed", "7", "--model-tag", "d2_fn_mean", expect=1)
        assert "loss_normalization: preflight 'fixed_nominal' vs this run 'microbatch_mean'" in r.stderr
        # 3. the funded-style run: policy recorded in the meta, both objectives logged per step across the step
        r = base_train(arm, "--num-iterations", str(N), "--save-every", "5", "--init-seed", "7", "--model-tag", "d2_fn", *fixed)
        assert "Loss normalization: fixed_nominal = sum of per-token losses / (B x T x grad_accum = 2 x 96 x 1 = 192)" in r.stdout
        m = meta(arm, "d2_fn", N)
        assert m["loss_normalization"] == "fixed_nominal" and m["loss_denominator_per_microbatch"] == 192 and m["preflight"] == "ok"
        assert m["user_config"]["loss_normalization"] == "fixed_nominal" and m["init_seed"] == 7
        line = next(l for l in r.stdout.splitlines() if l.startswith("step 00001/"))
        g = re.search(r"nominal: ([0-9.]+) \| per_active_tok: ([0-9.]+) \| active: (\d+)/(\d+)", line)
        assert g, line
        nominal, per_tok, active, nominal_targets = float(g.group(1)), float(g.group(2)), int(g.group(3)), int(g.group(4))
        assert nominal_targets == 192 and 0 < active <= nominal_targets
        assert nominal == pytest.approx(per_tok * active / nominal_targets, rel=1e-3)   # same sum, two denominators
        assert per_tok >= nominal
        # the same seed under the historical objective trains a DIFFERENT model (same initial parameters)
        pre.write_bytes(backup)
        base_train(arm, "--num-iterations", str(N), "--init-seed", "7", "--model-tag", "d2_fn_hist")
        mh = meta(arm, "d2_fn_hist", N)
        assert mh["loss_normalization"] == "microbatch_mean" and mh["loss_denominator_per_microbatch"] is None
        assert mh["init_param_sha256"] == m["init_param_sha256"]
        assert exposure_audit.main([*preflight_args, *fixed]) == 0
        # 4. resume: the policy is bound to the checkpoint; a change is refused BEFORE the schedule/preflight checks
        for f in (arm / "base_checkpoints" / "d2_fn").glob(f"*_{N:06d}*"):
            f.unlink()
        r = base_train(arm, "--num-iterations", str(N), "--init-seed", "7", "--model-tag", "d2_fn", "--resume-from-step", "5",
                       "--loss-normalization", "microbatch_mean", expect=1)
        assert "the training objective differs" in r.stderr and "loss_normalization='fixed_nominal'" in r.stderr
        assert "a policy change on resume is refused" in r.stderr and _saved_steps(arm / "base_checkpoints" / "d2_fn") == [5]
        r = base_train(arm, "--num-iterations", str(N), "--init-seed", "7", "--model-tag", "d2_fn", "--resume-from-step", "5", *fixed)
        m2 = meta(arm, "d2_fn", N)
        assert m2["loss_normalization"] == "fixed_nominal" and m2["resume_from_step"] == 5
        assert m2["dataloader_state_dict"] == m["dataloader_state_dict"] and m2["init_param_sha256"] == m["init_param_sha256"]
        # 5. a historical checkpoint (meta without the key) resumed under fixed_nominal is refused: missing metadata never
        #    implies the new policy
        meta5 = arm / "base_checkpoints" / "d2_fn" / "meta_000005.json"
        original = meta5.read_bytes()
        d = json.loads(original)
        d.pop("loss_normalization")
        meta5.write_text(json.dumps(d))
        for f in (arm / "base_checkpoints" / "d2_fn").glob(f"*_{N:06d}*"):
            f.unlink()
        try:
            r = base_train(arm, "--num-iterations", str(N), "--init-seed", "7", "--model-tag", "d2_fn", "--resume-from-step", "5", *fixed, expect=1)
            assert "loss_normalization='microbatch_mean' (unrecorded: historical default)" in r.stderr and "the training objective differs" in r.stderr
        finally:
            meta5.write_bytes(original)
    finally:
        pre.write_bytes(backup)
