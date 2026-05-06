import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2Model


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, adapt_bias: bool = False):
        super().__init__()

        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")

        if rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {rank}")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.adapt_bias = bool(adapt_bias)
        self.scaling = self.alpha / self.rank 

        for p in self.base.parameters():
            p.requires_grad = False

        dtype = self.base.weight.dtype
        device = self.base.weight.device

        self.A = nn.Parameter(
            torch.empty(self.rank, self.base.in_features, dtype=dtype, device=device)
        )
        self.B = nn.Parameter(
            torch.empty(self.base.out_features, self.rank, dtype=dtype, device=device)
        )

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        if self.adapt_bias:
            self.delta_bias = nn.Parameter(
                torch.zeros(self.base.out_features, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("delta_bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta_w = self.scaling * (self.B @ self.A)
        lora_out = F.linear(x, delta_w, self.delta_bias)
        return base_out + lora_out


def inject_lora_simple(
    backbone: nn.Module,
    target_modules: tuple[str, ...],
    rank: int,
    alpha: float,
    adapt_bias: bool = False,
) -> None:
    for name, module in list(backbone.named_modules()):
        # if not name.endswith("self_attn") or name.endwith("mlp"):
        #     continue

        for linear_name in target_modules:
            if not hasattr(module, linear_name):
                continue

            base_linear = getattr(module, linear_name)
            if isinstance(base_linear, LoRALinear):
                continue
            if not isinstance(base_linear, nn.Linear):
                continue

            setattr(
                module,
                linear_name,
                LoRALinear(
                    base=base_linear,
                    rank=rank,
                    alpha=alpha,
                    adapt_bias=adapt_bias,
                ),
            )


class MalwareDetectionModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model = Qwen2Model.from_pretrained(
            config.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )

        self.model.config.use_cache = False

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        for param in self.model.parameters():
            param.requires_grad = False

        lora_rank = getattr(config, "lora_rank", 16)
        lora_alpha = getattr(config, "lora_alpha", 16)
        lora_target_modules = tuple(
            getattr(config, "lora_target_modules", ("q_proj", "k_proj", "v_proj", "o_proj"))
        )
        adapt_bias = getattr(config, "adapt_bias", False)

        inject_lora_simple(
            backbone=self.model,
            target_modules=lora_target_modules,
            rank=lora_rank,
            alpha=lora_alpha,
            adapt_bias=adapt_bias,
        )

        hidden_dim = self.model.config.hidden_size

        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2, dtype=torch.bfloat16),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, dtype=torch.bfloat16),
        )

        self.regression_head = nn.Linear(hidden_dim, 1, dtype=torch.bfloat16)
        nn.init.xavier_uniform_(self.regression_head.weight)
        nn.init.zeros_(self.regression_head.bias)

        nn.init.xavier_uniform_(self.attention_net[0].weight)
        nn.init.xavier_uniform_(self.attention_net[2].weight)

        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        batch_size, num_chunks, seq_len = input_ids.shape

        input_ids_flat = input_ids.view(-1, seq_len)
        attention_mask_flat = attention_mask.view(-1, seq_len)

        outputs = self.model(
            input_ids=input_ids_flat,
            attention_mask=attention_mask_flat,
            return_dict=True,
        )

        last_hidden = outputs.last_hidden_state

        chunk_embeddings = last_hidden[:, -1, :]
        chunk_embeddings = chunk_embeddings.view(batch_size, num_chunks, -1)

        attn_weights = self.attention_net(chunk_embeddings)
        attn_weights = torch.softmax(attn_weights, dim=1)

        pooled_output = torch.sum(attn_weights * chunk_embeddings, dim=1)
        logits = self.regression_head(pooled_output).squeeze(-1)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels.to(logits.dtype))

        return {"loss": loss, "logits": logits} if loss is not None else logits