import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
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

        lora_out = F.linear(F.linear(x, self.A), self.B)
        lora_out = lora_out * self.scaling
        
        if self.delta_bias is not None:
            lora_out = lora_out + self.delta_bias
            
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
            config.model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )

        self.model.config.use_cache = False

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        self.chunk_checkpointing = bool(getattr(config, "chunk_checkpointing", True))
        self.chunk_fwd_microbatch = max(1, int(getattr(config, "chunk_fwd_microbatch", 1)))

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

    def _encode_chunks(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state[:, -1, :]

    def forward(self, input_ids, attention_mask, labels=None, chunk_mask=None):
        batch_size, num_chunks, seq_len = input_ids.shape
        hidden_dim = self.model.config.hidden_size

        if chunk_mask is None:
            chunk_mask = input_ids.new_ones(batch_size, num_chunks)
        chunk_mask = chunk_mask.bool()

        # forward only the real chunks; padded chunk slots never touch the backbone
        flat_valid = chunk_mask.view(-1)
        input_ids_flat = input_ids.view(-1, seq_len)[flat_valid]
        attention_mask_flat = attention_mask.view(-1, seq_len)[flat_valid]

        use_ckpt = self.chunk_checkpointing and self.training and torch.is_grad_enabled()
        micro = self.chunk_fwd_microbatch

        embeddings = []
        for start in range(0, input_ids_flat.size(0), micro):
            ids = input_ids_flat[start:start + micro]
            mask = attention_mask_flat[start:start + micro]
            if use_ckpt:
                emb = checkpoint(self._encode_chunks, ids, mask, use_reentrant=False)
            else:
                emb = self._encode_chunks(ids, mask)
            embeddings.append(emb)

        valid_embeddings = torch.cat(embeddings, dim=0)

        chunk_embeddings = valid_embeddings.new_zeros(batch_size * num_chunks, hidden_dim)
        chunk_embeddings[flat_valid] = valid_embeddings
        chunk_embeddings = chunk_embeddings.view(batch_size, num_chunks, hidden_dim)

        attn_scores = self.attention_net(chunk_embeddings)
        attn_scores = attn_scores.masked_fill(~chunk_mask.unsqueeze(-1), float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=1)

        pooled_output = torch.sum(attn_weights * chunk_embeddings, dim=1)
        logits = self.regression_head(pooled_output).squeeze(-1)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels.to(logits.dtype))

        return {"loss": loss, "logits": logits} if loss is not None else logits