from dataclasses import dataclass

@dataclass
class Quen3BConfig:
    world_size:                 int = 2

    model_name:                 str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    model_path:                 str = "/media/sebi/nvme-1tb/LiLM-Mal/models/qwen2.5-coder-3b-instruct"

    num_labels:                 int = 2
    num_layers:                 int = 36 

    max_token_len:              int = 4096
    num_chunks:                 int = 2   

    batch_size:                 int = 2
    num_workers:                int = 16
    save_every_n_steps:         int = 500
    evaluate_every_n_steps:     int = 2000
    pin_memory:                 bool = True
    persistent_workers:         bool = True
    prefetch_factor:            int = 4
    
    n_unfrozen_layers:          int = 6
    grad_accum_steps:           int = 16 
    learning_rate:              float = 1e-5
    weight_decay:               float = 0.01
    adam_momentum:              float = 0.9
    adam_scaling:               float = 0.95
    epochs:                     int = 1

    gradient_checkpointing:     bool = True
    fsdp_cpu_offload:           bool = True 
    fsdp_sync_module_states:    bool = True
    use_distributed_sampler:    bool = True
    
    output_dir:                 str = "outputs/checkpoints-q3b"
    best_checkpoint_name:       str = "qwen_malware_best.pt" 
    plot_dir:                   str = "outputs/plots"