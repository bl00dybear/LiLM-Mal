import os
import re
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from compressor_model import CompressorDistiller, LoRALinear


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
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



def load_model_ddp(config, tokenizer, rank):
    torch.cuda.set_device(rank)

    model = CompressorDistiller(config, tokenizer).to(rank)


    model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )

    return model



def load_model_fsdp(config, tokenizer, rank):
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    from torch.distributed._composable.replicate import replicate

    torch.cuda.set_device(rank)

    model = CompressorDistiller(config, tokenizer).to(rank)

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    lora_params = set()
    for m in model.encoder.modules():
        if isinstance(m, LoRALinear):
            lora_params.add(m.A)
            lora_params.add(m.B)
            if m.delta_bias is not None:
                lora_params.add(m.delta_bias)

    for layer in model.encoder.backbone.layers:
        layer_lora = lora_params & set(layer.parameters())
        fully_shard(
            layer,
            mp_policy=mp_policy,
            reshard_after_forward=True,
            ignored_params=layer_lora if layer_lora else None,
        )

    backbone_lora = lora_params & set(model.encoder.backbone.parameters())
    fully_shard(
        model.encoder.backbone,
        mp_policy=mp_policy,
        reshard_after_forward=True,
        ignored_params=backbone_lora if backbone_lora else None,
    )

    for layer in model.decoder.backbone.layers:
        fully_shard(layer, mp_policy=mp_policy, reshard_after_forward=True)
    fully_shard(model.decoder.backbone, mp_policy=mp_policy, reshard_after_forward=True)

    replicate(model)

    return model



def load_model(config, tokenizer, rank):
    strategy = getattr(config, "strategy", "fsdp")
    if strategy == "fsdp":
        return load_model_fsdp(config, tokenizer, rank)
    return load_model_ddp(config, tokenizer, rank)



def get_raw_model(model, strategy="fsdp"):
    if strategy == "fsdp":
        m = model
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod
        return m
    else:
        m = model.module
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod
        return m


def get_trainable_state_dict(model, strategy="fsdp") -> dict:

    raw = get_raw_model(model, strategy)
    return {
        name: param.detach().cpu()
        for name, param in raw.named_parameters()
        if param.requires_grad
    }


def _infer_step_from_checkpoint_path(path: str) -> int:
    match = re.search(r"_step(\d+)\.pt$", os.path.basename(path))
    if match is None:
        return 0
    return int(match.group(1))


def load_training_checkpoint(model, optimizer, scheduler, config, rank):
    checkpoint_path = config.resume_checkpoint_path
    if not checkpoint_path:
        return 0

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint inexistent: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    strategy = getattr(config, "strategy", "fsdp")
    raw = get_raw_model(model, strategy)

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

    if rank == 0:
        print(f"[info] [rank 0] checkpoint incarcat: {checkpoint_path}")
        print(f"[info] [rank 0] resume global_step: {resumed_global_step}")
        print(f"[info] [rank 0] optimizer_state loaded: {optimizer_loaded}")
        print(f"[info] [rank 0] scheduler_state loaded: {scheduler_loaded}")
        if unexpected:
            print(f"[warn] [rank 0] unexpected keys la load: {len(unexpected)}")

    return resumed_global_step
