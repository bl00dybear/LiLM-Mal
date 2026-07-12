import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2Model

from segment_dataset import build_prompt_ids


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


def inject_lora(
    backbone: nn.Module,
    target_modules: tuple[str, ...],
    rank: int,
    alpha: float,
    adapt_bias: bool = False,
) -> None:
    for name, module in list(backbone.named_modules()):
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


def merge_lora_state_dict(state_dict: dict, scaling: float) -> tuple[dict, dict]:
    state_dict = {
        (key[len("_orig_mod."):] if key.startswith("_orig_mod.") else key): value
        for key, value in state_dict.items()
    }

    backbone_sd = {}
    head_sd = {}

    for key, value in state_dict.items():
        if key.startswith("model."):
            sub = key[len("model."):]
            if sub.endswith(".A") or sub.endswith(".B"):
                continue
            if sub.endswith(".base.weight"):
                prefix = sub[: -len(".base.weight")]
                A = state_dict.get(f"model.{prefix}.A")
                B = state_dict.get(f"model.{prefix}.B")
                w = value.float()
                if A is not None and B is not None:
                    w = w + (B.float() @ A.float()) * scaling
                backbone_sd[f"{prefix}.weight"] = w.to(torch.bfloat16)
            elif ".base." in sub:
                backbone_sd[sub.replace(".base.", ".")] = value
            else:
                backbone_sd[sub] = value
        elif key.startswith("regression_head."):
            head_sd[key[len("regression_head."):]] = value

    return backbone_sd, head_sd


def _load_backbone(config) -> Qwen2Model:
    backbone = Qwen2Model.from_pretrained(
        config.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    backbone.config.use_cache = False

    backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    return backbone


class MemoryEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.backbone = _load_backbone(config)

        for p in self.backbone.parameters():
            p.requires_grad = False

        inject_lora(
            backbone=self.backbone,
            target_modules=tuple(config.lora_target_modules),
            rank=int(config.lora_rank),
            alpha=float(config.lora_alpha),
            adapt_bias=bool(getattr(config, "adapt_bias", False)),
        )

        self.num_memory_tokens = int(config.num_memory_tokens)

        emb = self.backbone.embed_tokens.weight
        g = torch.Generator().manual_seed(42)
        idx = torch.randint(0, emb.size(0), (self.num_memory_tokens,), generator=g)
        with torch.no_grad():
            mem_init = emb[idx].detach().clone()

        self.memory = nn.Parameter(mem_init)

    def forward(self, code_ids: torch.Tensor, code_mask: torch.Tensor) -> torch.Tensor:
        bsz = code_ids.size(0)

        code_emb = self.backbone.embed_tokens(code_ids)
        mem = self.memory.unsqueeze(0).expand(bsz, -1, -1)

        inputs_embeds = torch.cat([code_emb, mem], dim=1)
        attention_mask = torch.cat(
            [code_mask, code_mask.new_ones(bsz, self.num_memory_tokens)], dim=1
        )

        hidden = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state

        return hidden[:, -self.num_memory_tokens:, :]


class TaskDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        ckpt_path = config.teacher_checkpoint_path
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Teacher checkpoint not found: {ckpt_path} — "
                f"produced by the qwen-lora-classic run (model.output_dir)"
            )

        self.backbone = _load_backbone(config)

        ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=True)
        state_dict = ckpt.get("model_state_dict", ckpt)
        scaling = float(config.teacher_lora_alpha) / float(config.teacher_lora_rank)
        backbone_sd, head_sd = merge_lora_state_dict(state_dict, scaling)
        del ckpt, state_dict

        missing, unexpected = self.backbone.load_state_dict(backbone_sd, strict=False)
        if missing or unexpected:
            print(
                f"[warn] [decoder] merge load: {len(missing)} missing, "
                f"{len(unexpected)} unexpected keys"
            )
        del backbone_sd

        for p in self.backbone.parameters():
            p.requires_grad = False

        inject_lora(
            backbone=self.backbone,
            target_modules=tuple(config.lora_target_modules),
            rank=int(config.decoder_lora_rank),
            alpha=float(config.decoder_lora_alpha),
            adapt_bias=bool(getattr(config, "adapt_bias", False)),
        )

        hidden_dim = self.backbone.config.hidden_size
        self.regression_head = nn.Linear(hidden_dim, 1, dtype=torch.bfloat16)
        self.regression_head.load_state_dict(head_sd)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.backbone.embed_tokens(ids)

    def encode(self, inputs_embeds: torch.Tensor, attention_mask=None) -> torch.Tensor:
        return self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state


class EncoderDecoderClassifier(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()

        self.encoder = MemoryEncoder(config)
        self.decoder = TaskDecoder(config)

        self.encoder_segment_batch = int(getattr(config, "encoder_segment_batch", 1))
        self.max_segments_per_file = int(config.max_segments_per_file)

        prefix_ids, suffix_ids = build_prompt_ids(tokenizer)
        self.register_buffer("prefix_ids", torch.tensor(prefix_ids, dtype=torch.long), persistent=False)
        self.register_buffer("suffix_ids", torch.tensor(suffix_ids, dtype=torch.long), persistent=False)

        decoder_capacity = int(self.decoder.backbone.config.max_position_embeddings)
        decoder_seq = (
            self.max_segments_per_file * self.encoder.num_memory_tokens
            + len(prefix_ids) + len(suffix_ids)
        )
        if decoder_seq > decoder_capacity:
            raise ValueError(
                f"decoder input ({decoder_seq}) exceeds decoder capacity ({decoder_capacity}): "
                f"reduce max_segments_per_file or num_memory_tokens"
            )

    def forward(self, code_ids, code_mask, labels=None):
        n_seg = code_ids.size(0)
        if n_seg > self.max_segments_per_file:
            raise ValueError(
                f"file has {n_seg} segments, expected at most {self.max_segments_per_file}"
            )

        z_chunks = []
        for start in range(0, n_seg, self.encoder_segment_batch):
            z_chunks.append(
                self.encoder(
                    code_ids[start:start + self.encoder_segment_batch],
                    code_mask[start:start + self.encoder_segment_batch],
                )
            )
        z = torch.cat(z_chunks, dim=0)
        z = z.reshape(1, n_seg * z.size(1), z.size(2))

        prefix_emb = self.decoder.embed(self.prefix_ids).unsqueeze(0)
        suffix_emb = self.decoder.embed(self.suffix_ids).unsqueeze(0)

        inputs_embeds = torch.cat([prefix_emb, z, suffix_emb], dim=1)
        hidden = self.decoder.encode(inputs_embeds)

        h = hidden[:, -1, :]
        logits = self.decoder.regression_head(h).squeeze(-1).reshape(1)

        out = {"logits": logits.detach()}
        if labels is not None:
            out["loss"] = F.binary_cross_entropy_with_logits(
                logits.float(), labels.float()
            )
        return out
