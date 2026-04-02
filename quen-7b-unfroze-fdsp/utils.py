import torch
import torch.distributed as dist
from model import MalwareDetectionModel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from functools import partial
import os

from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer


def load_model_fsdp(config):
    model = MalwareDetectionModel(config)

    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        raise RuntimeError(
            "FSDP training requires distributed launch. "
            "Use: uv run torchrun --standalone --nproc_per_node=<NUM_GPUS> unfreeze-pipeline-fdsp/main.py"
        )

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
        device_id=torch.cuda.current_device(),
        sync_module_states=getattr(config, "fsdp_sync_module_states", False),
        limit_all_gathers=True,
        use_orig_params=True,
    )
    return model

def plot_training_loss(history: list, epoch: int, plot_dir: str) -> str:
    import os
    import torch
    import matplotlib.pyplot as plt
    
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(history, label="Training Step Loss", alpha=0.3, color="blue")
    
    window = min(len(history) // 10 + 1, 50)
    if window > 1 and len(history) >= window:
        smoothed = torch.tensor(history).unfold(0, window, 1).mean(1).numpy()
        plt.plot(range(window-1, len(history)), smoothed, color='red', label="Smoothed Loss", linewidth=2)
    
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Training Loss - Epoch {epoch}")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(plot_dir, f"loss_curve_epoch_{epoch}.pdf")
    plt.savefig(plot_path, format="pdf")
    plt.close()
    
    return plot_path