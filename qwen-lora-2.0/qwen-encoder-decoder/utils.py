import os
import random
import re
from datetime import timedelta

import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LRScheduler

from model import EncoderDecoderClassifier


class WSDScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_steps, start_factor=0.1, min_lr_factor=0.1, last_epoch=-1):
        self.warmup_steps = max(1, int(warmup_steps))
        self.start_factor = float(start_factor)
        self.min_lr_factor = float(min_lr_factor)
        self.decay_start_step = None
        self.decay_steps = None
        super().__init__(optimizer, last_epoch)

    @property
    def in_decay(self) -> bool:
        return self.decay_start_step is not None

    @property
    def decay_finished(self) -> bool:
        return self.in_decay and self.last_epoch >= self.decay_start_step + self.decay_steps

    def begin_decay(self, decay_steps: int):
        if self.in_decay:
            return
        self.decay_start_step = self.last_epoch
        self.decay_steps = max(1, int(decay_steps))

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            factor = self.start_factor + (1.0 - self.start_factor) * (step / self.warmup_steps)
        elif not self.in_decay:
            factor = 1.0
        else:
            progress = min(1.0, max(0.0, (step - self.decay_start_step) / self.decay_steps))
            factor = 1.0 - (1.0 - self.min_lr_factor) * progress
        return [base_lr * factor for base_lr in self.base_lrs]


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"
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


def load_model(config, tokenizer, rank):
    torch.cuda.set_device(rank)

    model = EncoderDecoderClassifier(config, tokenizer).to(rank)

    model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )

    return model


def get_raw_model(model):
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


def get_trainable_state_dict(model) -> dict:
    raw = get_raw_model(model)
    return {
        name: param.detach().cpu()
        for name, param in raw.named_parameters()
        if param.requires_grad
    }


def load_compressor_checkpoint(model, config, rank):
    path = getattr(config, "compressor_checkpoint_path", None)
    if not path:
        if rank == 0:
            print("[warn] no compressor_checkpoint_path — encoder starts from fresh LoRA init")
        return

    if not os.path.exists(path):
        raise FileNotFoundError(f"Compressor checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["trainable_state_dict"]

    raw = get_raw_model(model)
    own_keys = set(raw.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in own_keys}
    skipped = [k for k in state_dict if k not in own_keys]

    raw.load_state_dict(filtered, strict=False)

    if rank == 0:
        print(f"[info] [rank 0] encoder initialized from compressor: {path}")
        print(f"[info] [rank 0] loaded tensors: {len(filtered)} | skipped: {len(skipped)}")


def _infer_step_from_checkpoint_path(path: str) -> int:
    match = re.search(r"_step(\d+)\.pt$", os.path.basename(path))
    if match is None:
        return 0
    return int(match.group(1))


def load_training_checkpoint(model, optimizer, scheduler, config, rank):
    checkpoint_path = config.resume_checkpoint_path
    if not checkpoint_path:
        return 0, None

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    raw = get_raw_model(model)

    state_dict = checkpoint["trainable_state_dict"]
    missing, unexpected = raw.load_state_dict(state_dict, strict=False)
    unexpected = [k for k in unexpected]
    resumed_global_step = int(checkpoint.get("global_step", 0))

    optimizer_loaded = False
    scheduler_loaded = False
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

    early_stop_state = checkpoint.get("early_stop_state") or None

    if rank == 0:
        print(f"[info] [rank 0] checkpoint loaded: {checkpoint_path}")
        print(f"[info] [rank 0] resume global_step: {resumed_global_step}")
        print(f"[info] [rank 0] optimizer_state loaded: {optimizer_loaded}")
        print(f"[info] [rank 0] scheduler_state loaded: {scheduler_loaded}")
        if unexpected:
            print(f"[warn] [rank 0] unexpected keys on load: {len(unexpected)}")
        if early_stop_state:
            print(f"[info] [rank 0] early_stop_state restored: {early_stop_state}")

    return resumed_global_step, early_stop_state
