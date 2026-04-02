import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoTokenizer

from config import Qwen15BConfig
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


def load_model_ddp(config, rank):
    torch.cuda.set_device(rank)
 
    model = MalwareDetectionModel(config).to(rank)
 
    model = torch.compile(model)
 
    model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False,
    )
 
    return model


def main_worker(rank, config, tokenizer):
    torch.set_float32_matmul_precision('high')
    setup(rank=rank, world_size=config.world_size)

    if rank == 0:
        wandb.init(
            project="LiLM-Malware-Detection",
            name=f"qwen-1.5b-ddp-ctx{config.max_token_len}",
            config=asdict(config)
        )

    print(f"[info] [rank {rank}] main worker started")

    model = load_model_ddp(config=config, rank=rank)

    print(f"[info] [rank {rank}] [model] loaded")

    train_loader, val_loader, test_loader = build_loaders(config, tokenizer)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[info] [rank {rank}] [optimizer] parametri trainabili: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(
        params=trainable_params,
        lr=config.learning_rate,
        betas=(config.adam_momentum, config.adam_scaling),
        eps=1e-8,
        weight_decay=config.weight_decay,
        fused=True,
    )

    print(f"[info] [rank {rank}] [optimizer] initialized")

    steps_per_epoch = len(train_loader) // config.grad_accum_steps
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = int(total_steps * 0.1)
    cosine_steps = total_steps - warmup_steps

    linear_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=0.1,
        total_iters=warmup_steps,
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=cosine_steps,
        eta_min=config.learning_rate * 0.1,
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[linear_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    print(f"[info] [rank {rank}] [scheduler] initialized")

    train(
        model=model,
        train_loader=train_loader,
        valid_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=config.epochs,
        grad_accum_steps=config.grad_accum_steps,
        rank=rank,
        config=config,
    )

    cleanup()

    if rank == 0:
        wandb.finish()


def main():
    torch._logging.set_logs(all=logging.ERROR)

    config = Qwen15BConfig()

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        use_fast=True,
    )

    mp.spawn(
        main_worker,
        args=(config, tokenizer),
        nprocs=config.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()