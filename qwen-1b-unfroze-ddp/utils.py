import os
import re

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import MalwareDetectionModel


def setup(rank, world_size):
	os.environ["MASTER_ADDR"] = "localhost"
	os.environ["MASTER_PORT"] = "12355"
	os.environ["RANK"] = str(rank)
	os.environ["LOCAL_RANK"] = str(rank)
	os.environ["WORLD_SIZE"] = str(world_size)

	dist.init_process_group("nccl", rank=rank, world_size=world_size)
	torch.cuda.set_device(rank)


def cleanup():
	if dist.is_available() and dist.is_initialized():
		dist.destroy_process_group()


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
	)

	return model


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

	missing_keys, unexpected_keys = model.module.load_state_dict(state_dict, strict=False)

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
