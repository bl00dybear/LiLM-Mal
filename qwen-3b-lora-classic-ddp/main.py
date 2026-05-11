import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from transformers import AutoTokenizer

from config import Qwen15BConfig
from lilm_mal_dataset import build_loaders
from train_logic import train
from utils import setup, cleanup, load_model_ddp, load_training_checkpoint

from dataclasses import asdict
import wandb

import logging
import torch._logging


def main_worker(rank, config):
    torch.set_float32_matmul_precision("high")
    setup(rank=rank, world_size=config.world_size)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        use_fast=True,
    )

    if rank == 0:
        wandb.init(
            project="LiLM-Malware-Detection",
            name=f"qwen-3b-ddp-ctx{config.max_token_len}-lora-attention-sampled",
            config=asdict(config),
        )

    try:
        print(f"[info] [rank {rank}] main worker started")

        model = load_model_ddp(config=config, rank=rank)
        print(f"[info] [rank {rank}] [model] loaded")

        train_loader, val_loader, test_loader = build_loaders(config, tokenizer)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(
            f"[info] [rank {rank}] [optimizer] parametri trainabili: "
            f"{sum(p.numel() for p in trainable_params):,}"
        )

        optimizer = torch.optim.AdamW(
            params=trainable_params,
            lr=config.learning_rate,
            betas=(config.adam_momentum, config.adam_scaling),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
        )

        print(f"[info] [rank {rank}] [optimizer] initialized")

        steps_per_epoch = max(1, len(train_loader) // config.grad_accum_steps)
        total_steps = max(1, steps_per_epoch * config.epochs)
        warmup_steps = max(1, int(total_steps * 0.1))
        warmup_steps = min(warmup_steps, total_steps)
        cosine_steps = max(1, total_steps - warmup_steps)

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

        resumed_global_step, resumed_best_f1 = load_training_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            rank=rank,
        )
        dist.barrier()

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
            start_global_step=resumed_global_step,
            start_best_f1=resumed_best_f1,
        )
    finally:
        cleanup()
        if rank == 0:
            wandb.finish()


def main():
    torch._logging.set_logs(all=logging.ERROR)

    config = Qwen15BConfig()

    mp.spawn(
        main_worker,
        args=(config,),
        nprocs=config.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()