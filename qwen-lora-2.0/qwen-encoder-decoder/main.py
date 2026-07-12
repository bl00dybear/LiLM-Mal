import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from transformers import AutoTokenizer

from segment_dataset import build_loaders, build_prompt_ids, seg_code_budget
from train_logic import train
from utils import (
    setup,
    cleanup,
    load_model,
    load_compressor_checkpoint,
    load_training_checkpoint,
    set_global_seed,
    WSDScheduler,
)

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

import logging
import torch._logging


def main_worker(rank, config):
    set_global_seed(int(getattr(config, "seed", 42)))
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
            project="LiLM-Encoder-Decoder-Task",
            name=(
                f"{config.model_id.split('/')[-1]}-task"
                f"-k{config.num_memory_tokens}-seg{budget}"
                f"-nseg{config.max_segments_per_file}"
                f"-elora{config.lora_rank}-dlora{config.decoder_lora_rank}"
            ),
            config=OmegaConf.to_container(config, resolve=True),
        )

    try:
        print(f"[info] [rank {rank}] main worker started")

        model = load_model(config=config, tokenizer=tokenizer, rank=rank)
        print(f"[info] [rank {rank}] [model] loaded")

        load_compressor_checkpoint(model=model, config=config, rank=rank)
        dist.barrier()

        train_loader, val_loader = build_loaders(config)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(
            f"[info] [rank {rank}] [optimizer] trainable parameters: "
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
        warmup_steps = max(1, min(int(getattr(config, "warmup_steps", 100)), int(total_steps * 0.1)))

        scheduler = WSDScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            start_factor=0.1,
            min_lr_factor=float(getattr(config, "min_lr_factor", 0.1)),
        )

        print(
            f"[info] [rank {rank}] [scheduler] WSD: warmup {warmup_steps} steps, "
            f"budget {total_steps} steps, decay {getattr(config, 'decay_fraction', 0.1):.0%}"
        )

        resumed_global_step, early_stop_state = load_training_checkpoint(
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
            total_steps=total_steps,
            early_stop_state=early_stop_state,
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
