import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoTokenizer


import utils
from config import UnfreezeFdspConfig
from lilm_mal_dataset import LiLMMalDataset, LiLMMalDataLoader, build_loaders
from train_logic import train_model


def setup():
    is_distributed = all(var in os.environ for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if is_distributed:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = 0
        print("[info] RANK/WORLD_SIZE not set, running in single-process mode")

    torch.cuda.set_device(local_rank)
    return is_distributed

def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()



def main() -> None:
    gpus_count = torch.cuda.device_count()
    assert gpus_count, f"[error] cuda not available"
    print(f"[info] GPUs detected, num: {gpus_count}")


    is_distributed = setup()
    if not is_distributed:
        raise RuntimeError(
            "[error] this pipeline is FSDP-only and must be launched with torchrun. "
            "\n[info] uv run torchrun --standalone --nproc_per_node=2 unfreeze-pipeline-fdsp/main.py"
        )

    config = UnfreezeFdspConfig()

    try:
        model= utils.load_model_fsdp(config)
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)

        train_loader, val_loader = build_loaders(config, tokenizer)

        test_set = LiLMMalDataset(
            split="test",
            tokenizer=tokenizer,
            max_length=config.max_token_len,
            num_chunks=config.num_chunks,
        )
        test_sampler = None
        if is_distributed and getattr(config, "use_distributed_sampler", True):
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            test_sampler = DistributedSampler(
                test_set,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
        test_loader = LiLMMalDataLoader(
            test_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            sampler=test_sampler,
            drop_last=False,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers,
            prefetch_factor=config.prefetch_factor,
        )
        
        device_id = torch.cuda.current_device()
        train_model(model, train_loader, val_loader, config, is_distributed, device_id)
    finally:
        cleanup()


if __name__ == "__main__":
    main()