import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from functools import partial

from transformers import AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer


from config import Quen3BConfig
from model import MalwareDetectionModel
from lilm_mal_dataset import build_loaders
from train_logic import train

from dataclasses import asdict
import wandb

import logging
import torch._logging



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK']        = str(rank)
    os.environ['LOCAL_RANK']  = str(rank)
    os.environ['WORLD_SIZE']  = str(world_size)
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()


def load_model_fsdp(config, rank):
    torch.cuda.set_device(rank)
    model = MalwareDetectionModel(config).to(rank)

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen2DecoderLayer},
    )

    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=getattr(config, "fsdp_cpu_offload", False)),
        device_id=rank,
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )

    model = torch.compile(model)

    return model


def main_worker(rank, config, tokenizer):

    if rank == 0:
        wandb.init(
            project="LiLM-Malware-Detection",
            name=f"qwen-3b-fsdp-ctx{config.max_token_len}-sampled",
            config=asdict(config)
        )

    print(f"[info] [rank {rank}] main worker started")

    setup(rank=rank, world_size=config.world_size)

    model = load_model_fsdp(config = config, rank= rank)

    print(f"[info] [rank {rank}] [model] loaded")

    train_loader, val_loader, test_loader = build_loaders(config, tokenizer)


    adamw_params = [p for p in model.parameters() if p.requires_grad]

    print(f"[info] [rank {rank}] [optimizer] AdamW: {len(adamw_params)} groups")

    adamw_optimizer = torch.optim.AdamW(
        params=adamw_params,
        lr = config.learning_rate,
        betas=(config.adam_momentum,config.adam_scaling),
        eps = 1e-8,
        weight_decay=config.weight_decay,
        fused=True
    )

    print(f"[info] [rank {rank}] [optimizer] initialized")

    steps_per_epoch = len(train_loader) // config.grad_accum_steps
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = int(total_steps*0.1)
    cosine_steps = total_steps - warmup_steps


    liniar_scheduler_adamw = torch.optim.lr_scheduler.LinearLR(
        optimizer = adamw_optimizer,
        start_factor = 0.1,
        total_iters = warmup_steps
    )

    cosine_scheduler_adamw = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=adamw_optimizer,
        T_max=cosine_steps,
        eta_min=config.learning_rate*0.1
    )

    adamw_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer=adamw_optimizer,
        schedulers=[liniar_scheduler_adamw,cosine_scheduler_adamw],
        milestones=[warmup_steps]
    )

    print(f"[info] [rank {rank}] [scheduler] initialized")


    train(
        model=model,
        train_loader=train_loader,
        valid_loader=val_loader,
        test_loader=test_loader,
        optimizer=adamw_optimizer,
        scheduler=adamw_scheduler,
        epochs=config.epochs,
        grad_accum_steps=config.grad_accum_steps,
        rank=rank,
        config=config
    )




    cleanup()
    if rank == 0:
        wandb.finish()


def main():
    torch.set_float32_matmul_precision('high')
    torch._logging.set_logs(all=logging.ERROR)

    config = Quen3BConfig()

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        use_fast=True
    )

    mp.spawn(
        main_worker,
        args=(config, tokenizer),
        nprocs=config.world_size,
        join=True
    )


if __name__ == "__main__":
    main()