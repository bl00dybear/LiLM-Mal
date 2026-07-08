import os
import re
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import MalwareDetectionModel, LoRALinear


def setup(rank, world_size):
	os.environ["MASTER_ADDR"] = "localhost"
	os.environ["MASTER_PORT"] = "12355"
	os.environ["RANK"] = str(rank)
	os.environ["LOCAL_RANK"] = str(rank)
	os.environ["WORLD_SIZE"] = str(world_size)

	dist.init_process_group(
		"nccl",
		rank=rank,
		world_size=world_size,
		timeout=timedelta(minutes=60),
	)
	torch.cuda.set_device(rank)


def cleanup():
	if dist.is_available() and dist.is_initialized():
		dist.destroy_process_group()


# ── DDP ────────────────────────────────────────────────────────────────

def load_model_ddp(config, rank, compile_model=True):
	torch.cuda.set_device(rank)

	model = MalwareDetectionModel(config).to(rank)
	if compile_model:
		model = torch.compile(model)

	model = DDP(
		model,
		device_ids=[rank],
		output_device=rank,
		find_unused_parameters=False,
		gradient_as_bucket_view=True,
	)

	return model


def load_model_fsdp(config, rank, compile_model=True):
	from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
	from torch.distributed._composable.replicate import replicate

	torch.cuda.set_device(rank)

	model = MalwareDetectionModel(config).to(rank)

	mp_policy = MixedPrecisionPolicy(
		param_dtype=torch.bfloat16,
		reduce_dtype=torch.float32,
	)

	# Collect LoRA parameters — these stay replicated (DDP-style)
	lora_params = set()
	for m in model.modules():
		if isinstance(m, LoRALinear):
			lora_params.add(m.A)
			lora_params.add(m.B)
			if m.delta_bias is not None:
				lora_params.add(m.delta_bias)

	# FSDP per-layer (bottom-up) — shard frozen base weights only
	for layer in model.model.layers:
		layer_lora = lora_params & set(layer.parameters())
		fully_shard(
			layer,
			mp_policy=mp_policy,
			reshard_after_forward=True,
			ignored_params=layer_lora if layer_lora else None,
		)

	# FSDP on backbone root (embed_tokens + final norm)
	backbone_lora = lora_params & set(model.model.parameters())
	fully_shard(
		model.model,
		mp_policy=mp_policy,
		reshard_after_forward=False,  # root: keep unsharded after forward
		ignored_params=backbone_lora if backbone_lora else None,
	)

	# Replicate root — LoRA params + attention_net + regression_head
	# get DDP-style gradient all-reduce
	replicate(model)

	if compile_model:
		model = torch.compile(model)

	return model


# ── Dispatcher ─────────────────────────────────────────────────────────

def load_model(config, rank, compile_model=None):
	# torch.compile defaults off: variable chunk counts per sample change the
	# input shapes every step and trigger constant recompilation
	if compile_model is None:
		compile_model = bool(getattr(config, "compile_model", False))
	strategy = getattr(config, "strategy", "ddp")
	if strategy == "fsdp":
		return load_model_fsdp(config, rank, compile_model)
	return load_model_ddp(config, rank, compile_model)


# ── Checkpoint helpers ─────────────────────────────────────────────────

def get_raw_model(model, strategy="ddp"):
	"""Unwrap model to get the raw nn.Module (without DDP/compile wrappers)."""
	if strategy == "fsdp":
		# FSDP2 + replicate: model is directly accessible (no .module)
		m = model
		if hasattr(m, "_orig_mod"):
			m = m._orig_mod
		return m
	else:
		# DDP: model.module, possibly with torch.compile _orig_mod
		m = model.module
		if hasattr(m, "_orig_mod"):
			m = m._orig_mod
		return m


def get_state_dict_for_save(model, strategy="ddp"):
	"""Get a full state dict suitable for saving on rank 0."""
	if strategy == "fsdp":
		from torch.distributed.checkpoint.state_dict import (
			get_model_state_dict,
			StateDictOptions,
		)
		return get_model_state_dict(
			model,
			options=StateDictOptions(
				full_state_dict=True,
				cpu_offload=True,
			),
		)
	else:
		return model.module.state_dict()


def _infer_step_from_checkpoint_path(path: str) -> int:
	match = re.search(r"_step(\d+)\.pt$", os.path.basename(path))
	if match is None:
		return 0
	return int(match.group(1))


def load_training_checkpoint(model, optimizer, scheduler, config, rank):
	checkpoint_path = config.resume_checkpoint_path
	if not checkpoint_path:
		return 0, -1.0

	if not os.path.exists(checkpoint_path):
		raise FileNotFoundError(f"Checkpoint inexistent: {checkpoint_path}")

	checkpoint = torch.load(checkpoint_path, map_location="cpu")

	if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
		state_dict = checkpoint["model_state_dict"]
		resumed_global_step = int(checkpoint.get("global_step", 0))
		best_f1 = float(checkpoint.get("best_f1", -1.0))
	else:
		state_dict = checkpoint
		resumed_global_step = 0
		best_f1 = -1.0

	strategy = getattr(config, "strategy", "ddp")

	if strategy == "fsdp":
		from torch.distributed.checkpoint.state_dict import (
			set_model_state_dict,
			StateDictOptions,
		)
		set_model_state_dict(
			model,
			model_state_dict=state_dict,
			options=StateDictOptions(
				full_state_dict=True,
				strict=False,
				broadcast_from_rank0=True,
			),
		)
		missing_keys, unexpected_keys = [], []
	else:
		missing_keys, unexpected_keys = model.module.load_state_dict(
			state_dict, strict=False
		)

	optimizer_loaded = False
	scheduler_loaded = False
	if isinstance(checkpoint, dict):
		optimizer_state = checkpoint.get("optimizer_state_dict")
		if optimizer_state is not None:
			optimizer.load_state_dict(optimizer_state)
			optimizer_loaded = True

		scheduler_state = checkpoint.get("scheduler_state_dict")
		if scheduler_state is not None:
			scheduler.load_state_dict(scheduler_state)
			scheduler_loaded = True

	if resumed_global_step == 0:
		resumed_global_step = _infer_step_from_checkpoint_path(checkpoint_path)

	if rank == 0:
		print(f"[info] [rank 0] checkpoint incarcat: {checkpoint_path}")
		print(f"[info] [rank 0] resume global_step: {resumed_global_step}")
		print(f"[info] [rank 0] optimizer_state loaded: {optimizer_loaded}")
		print(f"[info] [rank 0] scheduler_state loaded: {scheduler_loaded}")
		if missing_keys:
			print(f"[warn] [rank 0] missing keys la load: {len(missing_keys)}")
		if unexpected_keys:
			print(f"[warn] [rank 0] unexpected keys la load: {len(unexpected_keys)}")

	return resumed_global_step, best_f1
