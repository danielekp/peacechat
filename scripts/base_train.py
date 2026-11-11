"""
Train model. From root directory of the project, run as:

python -m scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import sys
import glob
import json
import time
import math
import hashlib
import argparse
import subprocess
from dataclasses import asdict
from contextlib import contextmanager

import wandb
import torch
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig, Linear
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit, prepacked_distributed_data_loader_with_state, encoding_fingerprint
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized, load_dotenv
from nanochat.loss_normalization import LOSS_NORMALIZATIONS, HISTORICAL_LOSS_NORMALIZATION, StepLossStats, microbatch_loss, nominal_denominator, recorded_policy
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()
load_dotenv()  # WANDB_API_KEY / WANDB_PROJECT / WANDB_ENTITY / WANDB_RUN_ID from a gitignored .env

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU and torchao)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=12, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.28, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--warmup-steps", type=int, default=40, help="number of steps for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
# Seeds and data path (contextualization follow-up; see contextualization/docs/EXPOSURE.md). Defaults reproduce today's behaviour.
parser.add_argument("--init-seed", type=int, default=42, help="seed for parameter initialization (torch.manual_seed via compute_init); 42 = historical default")
parser.add_argument("--data-seed", type=int, default=-1, help="prepacked loader only: seed of the per-epoch row order (-1 = identity order)")
parser.add_argument("--data-loader", type=str, default="bestfit", choices=["bestfit", "prepacked"], help="train loader: bestfit = tokenize parquet shards on the fly (historical); prepacked = <base_dir>/prepacked rows with loss masks (val always uses bestfit)")
parser.add_argument("--skip-preflight-check", action="store_true", help="prepacked loader: do NOT require <base_dir>/../exposure/PREFLIGHT_OK.json to match this run (loud warning; the checkpoint meta records preflight: skipped)")
parser.add_argument("--loss-normalization", type=str, default=HISTORICAL_LOSS_NORMALIZATION, choices=list(LOSS_NORMALIZATIONS), help="microbatch_mean = historical nanochat objective (mean over each micro-batch's non-masked targets / grad_accum_steps); fixed_nominal = sum of per-token losses / (device_batch_size x max_seq_len x grad_accum_steps), the policy of the funded C/R/X/E experiment (decision 2026-10-15; see nanochat/loss_normalization.py). Recorded in the checkpoint meta, checked against the preflight and on resume")
args = parser.parse_args()
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type, seed=args.init_seed)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    project=os.environ.get("WANDB_PROJECT", "nanochat"),
    entity=os.environ.get("WANDB_ENTITY") or None,
    name=args.run,
    id=os.environ.get("WANDB_RUN_ID") or None,          # set by contextualization/run_arm.py so eval can attach later
    resume="allow" if os.environ.get("WANDB_RUN_ID") else None,
    job_type="train",
    config=user_config,
)
if not use_dummy_wandb:
    print0(f"wandb run: {wandb_run.name} ({wandb_run.id}) -> {wandb_run.url}")

# Flash Attention status
from nanochat.flash_attention import USE_FA3
using_fa3 = USE_FA3
if using_fa3:
    print0("✓ Using Flash Attention 3 (Hopper GPU detected), efficient, new and awesome.")
else:
    print0("!" * 80)
    if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        print0(f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    else:
        print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

def build_model_meta(depth):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
    # Model dim is nudged up to nearest multiple of head_dim for clean division
    # (FA3 requires head_dim divisible by 8, and this guarantees head_dim == args.head_dim exactly)
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta

def _state_dict_sha256(state_dict):
    """sha256 over the model parameters: keys in sorted order, each tensor's contiguous CPU bytes (plus its name,
    shape and dtype so that different layouts cannot alias). Computed right after init_weights(), before any
    optimizer step; matched arms of one replicate (same --init-seed, same config) must agree on it."""
    h = hashlib.sha256()
    for k in sorted(state_dict):
        t = state_dict[k].detach().to("cpu").contiguous()
        h.update(f"{k}|{tuple(t.shape)}|{t.dtype}\n".encode("utf-8"))
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()

# Build the model, move to device, init the weights
model = build_model_meta(args.depth) # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # 3) All tensors get initialized
init_param_sha256 = _state_dict_sha256(model.state_dict()) # identity of the initialization (before any optimizer step)

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
# Provenance of the initialization survives a resume: init_seed stays the ORIGINAL seed of the loaded checkpoint
# (the CLI value is recorded separately as cli_init_seed when it differs), init_param_sha256 is carried forward
# from the checkpoint meta rather than recomputed, and the data stream (loader + data seed) must be the checkpoint's.
init_seed = args.init_seed
cli_init_seed = None
resume_from_step = None
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy
    resume_from_step = args.resume_from_step
    if meta_data.get("init_seed") is not None:
        init_seed = int(meta_data["init_seed"])
        if init_seed != args.init_seed:
            cli_init_seed = args.init_seed
            print0("!" * 80)
            print0(f"WARNING: the checkpoint was initialised with init_seed={init_seed}; --init-seed {args.init_seed} does not change a resumed model and is only recorded as cli_init_seed")
            print0("!" * 80)
    init_param_sha256 = meta_data.get("init_param_sha256") # the ORIGINAL initialization identity (None for legacy checkpoints)
    for key, requested in (("data_loader", args.data_loader), ("data_seed", args.data_seed)):
        if meta_data.get(key) is not None and meta_data[key] != requested:
            raise ValueError(f"checkpoint was trained with {key}={meta_data[key]!r} but {key}={requested!r} was requested; a resumed run must keep the checkpoint's data stream")
    ckpt_loss_normalization = recorded_policy(meta_data) # an unrecorded policy is the historical microbatch_mean, never the new one
    if ckpt_loss_normalization != args.loss_normalization:
        raise ValueError(f"cannot resume from step {args.resume_from_step} of {checkpoint_dir}: the training objective differs: the checkpoint "
                         f"was trained with loss_normalization={ckpt_loss_normalization!r}{' (unrecorded: historical default)' if 'loss_normalization' not in meta_data else ''} "
                         f"but --loss-normalization {args.loss_normalization} was requested; a policy change on resume is refused (start a new run)")

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        # our custom fp8 is simpler than torchao, written for exact API compatibility
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        # Filter: dims must be divisible by 16 (FP8 hardware requirement) large enough
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = num_linear - num_fp8
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)")

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    import torch.nn as nn

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> Linear (our custom class that casts weights to match input dtype)
    # Use device="meta" to avoid VRAM spike - the weight tensor will be swapped in afterwards
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device="meta",  # Use meta device to avoid unnecessary VRAM allocation
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Compile the model

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe

# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

# Get the parameter counts of our model
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# 1) Use scaling laws to determine the optimal training horizon in tokens
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params
def get_scaling_params(m):
    # As for which params to use exactly, transformer matrices + lm_head gives cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params
num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params) # optimal tokens for the model we are about to train

# Our reference model is d12, this is where a lot of hyperparameters are tuned and then transfered to higher depths (muP style)
d12_ref = build_model_meta(12) # creates the model on meta device
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref) # compute-optimal d12 training horizon in tokens (measured empirically)
B_REF = 2**19 # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

# 2) Now that we have the token horizon, we can calculate the optimal batch size
# We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
# The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
total_batch_size = args.total_batch_size # user-provided override is possible
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) Knowing the batch size, we can now calculate a learning rate correction (bigger batch size allows higher learning rates)
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF # B/B_ref
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard: η ∝ √(B/B_ref)
    # Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption!)
    batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref)
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

# 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
# We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
# Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
# Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling hoping it ~works for Muon too.
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
optimizer = model.setup_optimizer(
    # AdamW hyperparameters
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    # Muon hyperparameters
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val

def _sha256_file(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None

def _prepack_identity(prepacked_dir):
    """What binds a run to its prepacked rows: sha256 of prepack_manifest.json, prepack_dataset_id, encoding_fingerprint
    (plus the manifest's n_rows, which the resume position check needs; it is not a binding)."""
    path = os.path.join(prepacked_dir, "prepack_manifest.json")
    out = {"prepacked_dir": prepacked_dir, "prepack_manifest_sha256": _sha256_file(path), "prepack_dataset_id": None, "encoding_fingerprint": None, "n_rows": None}
    try:
        with open(path, "r", encoding="utf-8") as f:
            man = json.load(f)
        out["prepack_dataset_id"] = man.get("prepack_dataset_id")
        out["encoding_fingerprint"] = man.get("encoding_fingerprint")
        out["n_rows"] = man.get("n_rows")
    except (OSError, ValueError):
        pass
    return out

tokenizer_encoding_fingerprint = encoding_fingerprint(tokenizer) # identity of the encoding this run tokenizes/scores with
prepacked_dir = os.path.join(base_dir, "prepacked") if args.data_loader == "prepacked" else None
prepack_identity = _prepack_identity(prepacked_dir) if prepacked_dir else None
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
if dataloader_resume_state_dict is not None:
    resumed_loader = dataloader_resume_state_dict.get("loader", "bestfit") # bestfit states carry no "loader" key
    if resumed_loader != args.data_loader:
        raise ValueError(f"checkpoint was trained with --data-loader {resumed_loader} but --data-loader {args.data_loader} was requested")
if resuming and args.data_loader == "prepacked":
    # the checkpoint is bound to the prepacked rows it was trained on: same manifest, same dataset id
    ckpt_bindings = meta_data.get("bindings") or {}
    for key in ("prepack_manifest_sha256", "prepack_dataset_id"):
        if not ckpt_bindings.get(key):
            raise ValueError(f"checkpoint meta has no bindings.{key}; cannot bind the resume to the prepacked rows (rebuild/retrain with the corrected prepack)")
        if ckpt_bindings[key] != prepack_identity[key]:
            raise ValueError(f"checkpoint was trained on prepacked rows with {key}={str(ckpt_bindings[key])[:12]} but {prepacked_dir} has {str(prepack_identity[key])[:12]}: refusing to resume on different data")
if args.data_loader == "prepacked":
    # rows planned once across arms (contextualization/ctx/prepack.py); val stays on the bestfit loader.
    # The loader refuses rows whose encoding_fingerprint differs from this tokenizer's and resume states bound to other rows.
    train_loader = prepacked_distributed_data_loader_with_state(args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict, data_seed=args.data_seed, tokenizer=tokenizer)
else:
    train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Provenance recorded in every checkpoint meta json: seeds, code revision, known sources of
# nondeterminism and hashes binding the run to its tokenizer / data / build. Self-contained on purpose
# (no imports from contextualization/); every field is best effort and None when unavailable.

def _listing_sha256(paths):
    """sha256 over the sorted 'name:size' lines of a file list (a cheap binding to the data actually used)."""
    if not paths:
        return None
    lines = sorted(f"{os.path.basename(p)}:{os.path.getsize(p)}" for p in paths)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()

def _git_info():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    info = {"git_commit": None, "git_dirty": None, "git_branch": None, "argv": list(sys.argv), "cwd": os.getcwd()}
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, timeout=20)
        r = run("rev-parse", "HEAD")
        if r.returncode == 0:
            info["git_commit"] = r.stdout.strip()
            info["git_branch"] = run("rev-parse", "--abbrev-ref", "HEAD").stdout.strip() or None
            st = run("status", "--porcelain", "--untracked-files=no")
            info["git_dirty"] = bool(st.stdout.strip()) if st.returncode == 0 else None
    except Exception:
        pass
    return info

def _data_bindings():
    from nanochat.dataset import list_parquet_files
    b = {"tokenizer_sha256": _sha256_file(os.path.join(base_dir, "tokenizer", "tokenizer.pkl")),
         "encoding_fingerprint": tokenizer_encoding_fingerprint, # of the tokenizer object actually used (nanochat.dataloader.encoding_fingerprint)
         "data_dir": None, "data_listing_sha256": None,
         "prepacked_dir": None, "prepack_manifest_sha256": None, "prepack_dataset_id": None, "prepacked_listing_sha256": None,
         "preflight_path": None, "preflight_sha256": None,
         "build_summary_sha256": None, "probe_hashes": {}}
    try:
        parquet_paths = list_parquet_files()
        b["data_dir"] = os.path.dirname(parquet_paths[0]) if parquet_paths else None
        b["data_listing_sha256"] = _listing_sha256(parquet_paths)
    except Exception:
        pass
    if args.data_loader == "prepacked":
        b["prepacked_dir"] = prepacked_dir
        b["prepack_manifest_sha256"] = prepack_identity["prepack_manifest_sha256"]
        b["prepack_dataset_id"] = prepack_identity["prepack_dataset_id"]
        b["prepacked_listing_sha256"] = _listing_sha256(sorted(glob.glob(os.path.join(prepacked_dir, "*.npy"))))
    exp_dir = os.path.dirname(os.path.abspath(base_dir)) # contextualization layout: base_dir = <exp_dir>/arm_<A>
    summary_path = os.path.join(exp_dir, "build_summary.json")
    if os.path.isfile(summary_path):
        b["build_summary_sha256"] = _sha256_file(summary_path)
    for p in sorted(glob.glob(os.path.join(exp_dir, "probe_sets", "*.jsonl"))):
        b["probe_hashes"][os.path.basename(p)] = _sha256_file(p)
    return b

provenance_code = _git_info()
provenance_nondeterminism = {
    "torch_compile": True,
    "float32_matmul_precision": "high" if device_type == "cuda" else None,
    "flash_attention_3": bool(using_fa3),
    "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    "world_size": ddp_world_size,
    "compute_dtype": str(COMPUTE_DTYPE),
    "device_name": gpu_device_name if device_type == "cuda" else device_type,
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "note": "a seed fixes initialization and data order only; kernels/compile/TF32 are not bitwise deterministic",
}
provenance_bindings = _data_bindings() if master_process else None

# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations # the actual number of tokens we will train for
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}") # e.g. Chinchilla was ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# Learning rate schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Momentum scheduler for Muon optimizer (warms up to 0.97, warms down to 0.90 during LR warmdown)
def get_muon_momentum(it):
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97

# Weight decay scheduler for Muon optimizer (cosine decay to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    val_bpb = None # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# -----------------------------------------------------------------------------
# Exact resume keeps the checkpoint's schedule (review 2026-10-14 recheck, finding 4). The loader state counts consumed
# rows while the loop resumes from an optimizer-step count, so a resume with another total batch size / accumulation /
# horizon would silently consume a different number of micro-batches than the preflight certified (the preflight gate
# below only sees the CURRENT configuration). Every meta json records the resolved schedule (see the save below); it is
# compared here, once grad_accum_steps and num_iterations are known and before the preflight gate and any training step.
# A deliberate schedule change needs a new run: no segmented exposure history is supported.
SCHEDULE_KEYS = ("device_batch_size", "max_seq_len", "total_batch_size", "grad_accum_steps", "world_size", "num_iterations")

def _checkpoint_schedule(meta):
    """The checkpoint's effective schedule from its meta json, None per key when it is not recorded: top-level keys
    first (written since this check exists), then user_config for device_batch_size / max_seq_len, the explicit
    user_config overrides for total_batch_size / num_iterations (-1 = auto-computed, unusable), the loader state /
    nondeterminism block for world_size, and grad_accum_steps derived from the four values that define it."""
    cfg = meta.get("user_config") or {}
    loader_state = meta.get("dataloader_state_dict") or {}
    nondet = meta.get("nondeterminism") or {}
    explicit = lambda v: v if v is not None and int(v) > 0 else None # user_config override; -1 = auto-computed at the time
    first = lambda *vs: next((int(v) for v in vs if v is not None), None)
    s = {
        "device_batch_size": first(meta.get("device_batch_size"), cfg.get("device_batch_size")),
        "max_seq_len": first(meta.get("max_seq_len"), cfg.get("max_seq_len")),
        "total_batch_size": first(meta.get("total_batch_size"), explicit(cfg.get("total_batch_size"))),
        "grad_accum_steps": first(meta.get("grad_accum_steps")),
        "world_size": first(meta.get("world_size"), loader_state.get("world_size"), nondet.get("world_size")),
        "num_iterations": first(meta.get("num_iterations"), explicit(cfg.get("num_iterations"))),
    }
    if s["grad_accum_steps"] is None and all(s[k] for k in ("total_batch_size", "device_batch_size", "max_seq_len", "world_size")):
        per_micro = s["device_batch_size"] * s["max_seq_len"] * s["world_size"]
        s["grad_accum_steps"] = s["total_batch_size"] // per_micro if s["total_batch_size"] % per_micro == 0 else None
    return s

if resuming:
    ckpt_schedule = _checkpoint_schedule(meta_data)
    requested_schedule = {"device_batch_size": args.device_batch_size, "max_seq_len": args.max_seq_len, "total_batch_size": total_batch_size,
                          "grad_accum_steps": grad_accum_steps, "world_size": ddp_world_size, "num_iterations": num_iterations}
    unrecorded = [k for k in SCHEDULE_KEYS if ckpt_schedule[k] is None]
    differing = [f"{k}: checkpoint {ckpt_schedule[k]} vs requested {requested_schedule[k]}" for k in SCHEDULE_KEYS
                 if ckpt_schedule[k] is not None and ckpt_schedule[k] != requested_schedule[k]]
    if differing:
        raise ValueError(f"cannot resume from step {args.resume_from_step} of {checkpoint_dir}: the requested training schedule differs from the checkpoint's: "
                         + "; ".join(differing) + ". An exact resume keeps the checkpoint's schedule (" + ", ".join(SCHEDULE_KEYS) + "); "
                         "a deliberate schedule change needs a new run (no segmented exposure history is supported)")
    if unrecorded:
        if args.data_loader == "prepacked":
            raise ValueError(f"cannot resume from step {args.resume_from_step} of {checkpoint_dir} with --data-loader prepacked: the checkpoint meta does not "
                             f"record {', '.join(unrecorded)} (schedule provenance missing), so the resume cannot verify that it keeps the checkpoint's schedule; "
                             "a deliberate schedule change needs a new run (no segmented exposure history is supported)")
        print0("!" * 80)
        print0(f"WARNING: the checkpoint meta does not record {', '.join(unrecorded)}; the resume cannot verify that it keeps the checkpoint's schedule for these keys")
        print0("!" * 80)
    print0(f"Resume schedule check OK: the step {args.resume_from_step} checkpoint and this run agree on "
           + ", ".join(f"{k} {requested_schedule[k]}" for k in SCHEDULE_KEYS if k not in unrecorded))
    if args.data_loader == "prepacked":
        # the saved loader position must be where the checkpoint's OWN schedule leaves it: step x grad_accum x B x world_size rows
        ckpt_loader_state = meta_data["dataloader_state_dict"]
        n_rows = prepack_identity["n_rows"]
        if n_rows is None:
            raise ValueError(f"{prepacked_dir}/prepack_manifest.json records no n_rows; cannot verify the resumed loader position against the checkpoint's schedule")
        rows_per_micro = ckpt_schedule["device_batch_size"] * ckpt_schedule["world_size"]
        epoch_rows = (int(n_rows) // rows_per_micro) * rows_per_micro # the incomplete tail of an epoch's order is skipped by the loader
        consumed_rows = int(ckpt_loader_state["epoch"]) * epoch_rows + int(ckpt_loader_state["cursor"])
        expected_rows = args.resume_from_step * ckpt_schedule["grad_accum_steps"] * rows_per_micro
        if consumed_rows != expected_rows:
            raise ValueError(f"the loader position saved at step {args.resume_from_step} does not match the checkpoint's schedule: epoch {ckpt_loader_state['epoch']} "
                             f"cursor {ckpt_loader_state['cursor']} = {consumed_rows} consumed rows ({epoch_rows} rows per epoch), but {args.resume_from_step} steps x "
                             f"{ckpt_schedule['grad_accum_steps']} micro-batches x {rows_per_micro} rows = {expected_rows}; refusing to resume on an inconsistent exposure history")
        print0(f"Resume loader position OK: epoch {ckpt_loader_state['epoch']} cursor {ckpt_loader_state['cursor']} = {consumed_rows} consumed rows = "
               f"{args.resume_from_step} steps x {ckpt_schedule['grad_accum_steps']} micro-batches x {rows_per_micro} rows")

# -----------------------------------------------------------------------------
# Preflight gate (prepacked loader): <base_dir>/../exposure/PREFLIGHT_OK.json is written by
# contextualization/exposure_audit.py --mode prepacked after the prepack audit passed for THIS training configuration;
# training refuses to start unless it exists and matches, or --skip-preflight-check is given explicitly.
preflight_status = None # None = not applicable (bestfit); "ok" | "skipped" for the prepacked loader
preflight_path = None
preflight_sha256 = None
if args.data_loader == "prepacked":
    exp_dir_of_arm = os.path.dirname(os.path.abspath(base_dir)) # contextualization layout: base_dir = <exp_dir>/arm_<A>
    arm_name = os.path.basename(os.path.abspath(base_dir))
    arm_name = arm_name[4:] if arm_name.startswith("arm_") else arm_name
    preflight_path = os.path.join(exp_dir_of_arm, "exposure", "PREFLIGHT_OK.json")
    preflight_sha256 = _sha256_file(preflight_path)
    if args.skip_preflight_check:
        preflight_status = "skipped"
        print0("!" * 80)
        print0(f"WARNING: --skip-preflight-check: training on {prepacked_dir} WITHOUT verifying {preflight_path} against this run")
        print0("WARNING: the exposure/occurrence/schedule guarantees of the prepacked path are NOT established for this run (meta records preflight: skipped)")
        print0("!" * 80)
    else:
        if not os.path.isfile(preflight_path):
            raise RuntimeError(f"no preflight for the prepacked rows: {preflight_path} does not exist. Run contextualization/exposure_audit.py "
                               f"--mode prepacked --exp-dir {exp_dir_of_arm} --arms <all arms> --B {args.device_batch_size} --T {args.max_seq_len} "
                               f"--grad-accum {grad_accum_steps} --num-iterations {num_iterations} --world-size {ddp_world_size} --data-seed {args.data_seed} "
                               f"--loss-normalization {args.loss_normalization} first, or pass --skip-preflight-check explicitly")
        with open(preflight_path, "r", encoding="utf-8") as f:
            preflight = json.load(f)
        actual = {"B": args.device_batch_size, "T": args.max_seq_len, "grad_accum_steps": grad_accum_steps,
                  "num_iterations": num_iterations, "world_size": ddp_world_size, "data_seed": args.data_seed}
        mismatches = [f"{k}: preflight {preflight.get(k)!r} vs this run {v!r}" for k, v in actual.items() if preflight.get(k) != v]
        if recorded_policy(preflight) != args.loss_normalization: # a preflight without the key certified the historical objective
            mismatches.append(f"loss_normalization: preflight {recorded_policy(preflight)!r}{' (unrecorded: historical default)' if 'loss_normalization' not in preflight else ''} "
                              f"vs this run {args.loss_normalization!r}")
        preflight_ids = preflight.get("prepack_dataset_id") or {}
        if arm_name not in preflight_ids:
            mismatches.append(f"arm {arm_name!r} is not covered by the preflight (arms {sorted(preflight_ids)})")
        elif preflight_ids[arm_name] != prepack_identity["prepack_dataset_id"]:
            mismatches.append(f"prepack_dataset_id of arm {arm_name}: preflight {str(preflight_ids[arm_name])[:12]} vs {prepacked_dir} {str(prepack_identity['prepack_dataset_id'])[:12]}")
        if preflight.get("encoding_fingerprint") != tokenizer_encoding_fingerprint:
            mismatches.append(f"encoding_fingerprint: preflight {str(preflight.get('encoding_fingerprint'))[:12]} vs this tokenizer {tokenizer_encoding_fingerprint[:12]}")
        if mismatches:
            raise RuntimeError(f"preflight {preflight_path} does not match this run: " + "; ".join(mismatches) +
                               ". Re-run contextualization/exposure_audit.py --mode prepacked with the exact training configuration "
                               "(or pass --skip-preflight-check explicitly; the checkpoint meta then records preflight: skipped)")
        preflight_status = "ok"
        print0(f"Preflight OK: {preflight_path} (sha256 {preflight_sha256[:12]}) matches B/T/grad_accum/num_iterations/world_size/data_seed/dataset id/encoding/loss_normalization")
if provenance_bindings is not None:
    provenance_bindings["preflight_path"] = preflight_path
    provenance_bindings["preflight_sha256"] = preflight_sha256
loss_denominator_per_microbatch = nominal_denominator(args.device_batch_size, args.max_seq_len, grad_accum_steps)
if args.loss_normalization == "fixed_nominal":
    print0(f"Loss normalization: fixed_nominal = sum of per-token losses / (B x T x grad_accum = {args.device_batch_size} x {args.max_seq_len} x {grad_accum_steps} = {loss_denominator_per_microbatch:,}) per micro-batch; "
           f"gradients averaged across {ddp_world_size} rank(s) by the optimizer (no extra division)")
else:
    print0("Loss normalization: microbatch_mean (historical nanochat objective: mean over each micro-batch's non-masked targets / grad_accum_steps)")
step_stats = StepLossStats(device, device_batch_size=args.device_batch_size, max_seq_len=args.max_seq_len, grad_accum_steps=grad_accum_steps, world_size=ddp_world_size)

# Go!
while True:
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model):
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model):
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model):
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            optimizer.state_dict(), # optimizer state
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "wandb_run": None if use_dummy_wandb else { # which run trained this checkpoint (links HF <-> wandb)
                    "id": wandb_run.id, "name": wandb_run.name, "project": wandb_run.project, "url": wandb_run.url,
                },
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "num_iterations": num_iterations, # the RESOLVED training horizon (never the -1 CLI value); an exact resume must keep it
                "grad_accum_steps": grad_accum_steps, # micro-batches per optimizer step (= total_batch_size / (device_batch_size x max_seq_len x world_size))
                "world_size": ddp_world_size, # ranks of this run; with the keys above it is the schedule the resume check compares
                "init_seed": init_seed, # torch.manual_seed that initialised THIS model's parameters (kept from the checkpoint on resume)
                "cli_init_seed": cli_init_seed, # --init-seed of a resumed run when it differs from init_seed (otherwise None)
                "init_param_sha256": init_param_sha256, # sha256 of the parameters right after init_weights() (carried forward on resume)
                "resume_from_step": resume_from_step, # step this process resumed from (None = started from scratch)
                "preflight": preflight_status, # prepacked loader: "ok" | "skipped" (None for bestfit)
                "loss_normalization": args.loss_normalization, # microbatch_mean (historical) | fixed_nominal; an absent key means the historical objective
                "loss_denominator_per_microbatch": loss_denominator_per_microbatch if args.loss_normalization == "fixed_nominal" else None, # B x T x grad_accum_steps
                "data_seed": args.data_seed, # prepacked loader row order seed (-1 = identity; unused by bestfit)
                "data_loader": args.data_loader, # bestfit | prepacked
                "code": provenance_code, # git commit / dirty flag / branch, argv, cwd
                "nondeterminism": provenance_nondeterminism, # what a seed does NOT pin down
                "bindings": provenance_bindings, # sha256 of tokenizer / data listing / prepack manifest / build summary / probes
                "dataloader_state_dict": dataloader_state_dict, # prepacked states carry "loader": "prepacked"
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    step_stats.reset()
    for micro_step in range(grad_accum_steps):
        if args.loss_normalization == HISTORICAL_LOSS_NORMALIZATION:
            loss = model(x, y) # mean over this micro-batch's non-masked targets (nanochat default)
            train_loss = loss.detach() # for logging
            # NOTE (contextualization, docs/EXPOSURE.md section 10): averaging per-micro-batch means weights a scored token by
            # 1/(active targets of its micro-batch * grad_accum_steps); with the prepacked masks the arms have unequal active
            # counts in shared rows. Measured by exposure_audit.py (loss_weighting). The funded experiment uses fixed_nominal.
            loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        else:
            # fixed_nominal (decision 2026-10-15): sum of per-token losses / (B x T x grad_accum_steps); the sum is computed
            # directly (a fully masked micro-batch contributes 0, never NaN); the accumulation division happens exactly once
            # here and the optimizer's ReduceOp.AVG across ranks adds no further division
            loss, loss_sum, n_active = microbatch_loss(model, x, y, args.loss_normalization, device_batch_size=args.device_batch_size,
                                                       max_seq_len=args.max_seq_len, grad_accum_steps=grad_accum_steps)
            step_stats.add(loss_sum, n_active)
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
    # step the optimizer
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    if scaler is not None:
        scaler.unscale_(optimizer)
        # In distributed training, all ranks must agree on whether to skip the step.
        # Each rank may independently encounter inf/nan gradients, so we all-reduce
        # the found_inf flag (MAX = if any rank found inf, all ranks skip).
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)
    if args.loss_normalization == HISTORICAL_LOSS_NORMALIZATION:
        train_loss_f = train_loss.item() # .item() is a CPU-GPU sync point
        step_loss = None
    else:
        step_loss = step_stats.reduce(dist) # one all-reduce + one sync: the whole step, all ranks
        train_loss_f = step_loss["loss_nominal"] # the nominal training objective (what the optimizer minimizes)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    if dataloader_state_dict.get("loader") == "prepacked":
        epoch = f"{dataloader_state_dict['epoch']} cursor: {dataloader_state_dict['cursor']}"
    else:
        epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    if step_loss is not None:
        per_tok = step_loss["loss_per_active_token"]
        objective_str = (f" | nominal: {step_loss['loss_nominal']:.6f} | per_active_tok: {per_tok:.6f} | active: {step_loss['active_targets']}/{step_loss['nominal_targets']}"
                         if per_tok is not None else f" | nominal: {step_loss['loss_nominal']:.6f} | per_active_tok: n/a (0 active targets)")
    else:
        objective_str = ""
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f}{objective_str} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/loss_normalization": args.loss_normalization,
        }
        if step_loss is not None:
            log_data.update({
                "train/loss_nominal": step_loss["loss_nominal"], # the training objective, whole step, all ranks
                "train/loss_per_active_token": step_loss["loss_per_active_token"], # ordinary loss per valid target, whole step, all ranks
                "train/active_targets": step_loss["active_targets"],
                "train/nominal_targets": step_loss["nominal_targets"],
            })
        log_data.update({
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        })
        wandb_run.log(log_data)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable() # nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Scaling params ratio": total_batch_size * num_iterations / num_scaling_params,
        "DDP world size": ddp_world_size,
        "warmup_steps": args.warmup_steps,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
