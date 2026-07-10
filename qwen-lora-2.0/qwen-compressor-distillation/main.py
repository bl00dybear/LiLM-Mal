import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from transformers import AutoTokenizer

from segment_dataset import build_loaders, build_prompt_ids, seg_code_budget
from train_logic import train
from utils import setup, cleanup, load_model, load_training_checkpoint

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

import logging
import torch._logging


def main_worker(rank, config):
    torch.set_float32_matmul_precision("high")
    setup(rank=rank, world_size=config.world_size)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        trust_remote_code=True,
        use_fast=True,
    )

    budget = seg_code_budget(config)
    prefix_ids, suffix_ids = build_prompt_ids(tokenizer)
    prompt_overhead = len(prefix_ids) + len(suffix_ids)
    assert prompt_overhead + budget <= config.max_token_len, (
        f"prompt ({prompt_overhead}) + segment ({budget}) > max_token_len"
    )

    if rank == 0:
        wandb.init(
            project="LiLM-Compressor-Distillation",
            name=(
                f"{config.model_id.split('/')[-1]}-AB"
                f"-k{config.num_memory_tokens}-seg{budget}"
                f"-rec{config.lambda_rec}-kd{config.lambda_logit}/{config.lambda_repr}"
            ),
            config=OmegaConf.to_container(config, resolve=True),
        )

    try:
        print(f"[info] [rank {rank}] main worker started")

        model = load_model(config=config, tokenizer=tokenizer, rank=rank)
        print(f"[info] [rank {rank}] [model] loaded")

        train_loader, val_loader = build_loaders(config)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(
            f"[info] [rank {rank}] [optimizer] parametri trainabili: "
            f"{sum(p.numel() for p in trainable_params):,}"
        )

        use_fused = (getattr(config, "strategy", "fsdp") == "ddp")
        optimizer = torch.optim.AdamW(
            params=trainable_params,
            lr=config.learning_rate,
            betas=(config.adam_momentum, config.adam_scaling),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=use_fused,
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

        resumed_global_step = load_training_checkpoint(
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
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=config.epochs,
            grad_accum_steps=config.grad_accum_steps,
            rank=rank,
            config=config,
            start_global_step=resumed_global_step,
        )
    finally:
        cleanup()
        if rank == 0:
            wandb.finish()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    torch._logging.set_logs(all=logging.ERROR)

    os.chdir(hydra.utils.get_original_cwd())

    model_cfg = cfg.model

    mp.spawn(
        main_worker,
        args=(model_cfg,),
        nprocs=model_cfg.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
