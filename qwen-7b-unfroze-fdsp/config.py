from dataclasses import dataclass

@dataclass
class UnfreezeFdspConfig:
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    model_path: str = "/media/sebi/nvme-1tb/LiLM-Mal/models/qwen2.5-coder-7b-instruct"

    num_labels: int = 2
    num_layers: int = 32

    max_token_len: int = 512
    num_chunks: int = 4

    batch_size: int = 1
    num_workers: int = 16
    save_every_n_steps: int = 500
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    
    n_unfrozen_layers: int = 6
    grad_accum_steps: int = 32
    learning_rate: float = 2e-5
    epochs: int = 1

    gradient_checkpointing: bool = True
    fsdp_cpu_offload: bool = True
    fsdp_sync_module_states: bool = True
    use_distributed_sampler: bool = True
    
    output_dir: str = "outputs/checkpoints"
    best_checkpoint_name: str = "qwen_malware_best.pt" 
    plot_dir: str = "outputs/plots"