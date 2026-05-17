from dataclasses import dataclass, field


@dataclass
class Qwen15BConfig:
    world_size: int = 2

    model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    model_path: str = "/media/sebi/nvme-1tb/LiLM-Mal/models/qwen2.5-coder-3b-instruct"

    num_labels: int = 2
    num_layers: int = 28

    max_token_len: int = 4096
    num_chunks: int = 2

    batch_size: int = 1
    test_batch_size: int = 1
    num_workers: int = 8
    save_every_n_steps: int = 10
    evaluate_every_n_steps: int = 50
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2

    lora_rank: int = 16
    lora_alpha: int = 16
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    adapt_bias: bool = False

    grad_accum_steps: int = 32
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    adam_momentum: float = 0.9
    adam_scaling: float = 0.95
    epochs: int = 1

    gradient_checkpointing: bool = True
    checkpoint_segments: int = 2
    use_distributed_sampler: bool = True

    output_dir: str = "outputs/checkpoints-q3b-lora-full"
    best_checkpoint_name: str = "qwen_malware_best.pt"
    test_checkpoint_name: str = "qwen_malware_ep0.pt"
    plot_dir: str = "outputs/plots/qwen3-lora-full"

    resume_checkpoint_path: str | None = None