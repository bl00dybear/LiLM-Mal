import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
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


def inject_lora_simple(
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

        inject_lora_simple(
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
            ae_init = emb[idx[:1]].detach().clone()

        self.memory = nn.Parameter(mem_init)
        self.ae_token = nn.Parameter(ae_init)

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


class FrozenDecoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        ckpt_path = config.teacher_checkpoint_path
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint teacher inexistent: {ckpt_path} — "
                f"e produs de runul qwen-lora-classic (model.output_dir)"
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

        hidden_dim = self.backbone.config.hidden_size
        self.regression_head = nn.Linear(hidden_dim, 1, dtype=torch.bfloat16)
        self.regression_head.load_state_dict(head_sd)

        for p in self.parameters():
            p.requires_grad = False

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.backbone.embed_tokens(ids)

    def encode(self, inputs_embeds: torch.Tensor, attention_mask=None) -> torch.Tensor:
        return self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state


class CompressorDistiller(nn.Module):

    def __init__(self, config, tokenizer):
        super().__init__()

        self.encoder = MemoryEncoder(config)
        self.decoder = FrozenDecoder(config)

        self.lambda_rec = float(config.lambda_rec) 
        self.lambda_logit = float(config.lambda_logit)
        self.lambda_repr = float(config.lambda_repr)
        if self.lambda_rec == 0 and self.lambda_logit == 0 and self.lambda_repr == 0:
            raise ValueError("All lambdas are 0: nothing to train")

        self.recon_tokens = int(config.recon_tokens)
        self.ce_chunk_size = int(getattr(config, "ce_chunk_size", 1024))

        prefix_ids, suffix_ids = build_prompt_ids(tokenizer)
        self.register_buffer("prefix_ids", torch.tensor(prefix_ids, dtype=torch.long), persistent=False)
        self.register_buffer("suffix_ids", torch.tensor(suffix_ids, dtype=torch.long), persistent=False)


    def _ce_chunk(self, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        logits = F.linear(hidden, self.decoder.backbone.embed_tokens.weight)
        return F.cross_entropy(logits.float(), targets, reduction="sum")

    def _recon_loss_one(self, z: torch.Tensor, seg_ids: torch.Tensor) -> torch.Tensor:
        seg = seg_ids[: self.recon_tokens]
        n = seg.size(0)


        ae = self.encoder.ae_token.unsqueeze(0)
        seg_emb = self.decoder.embed(seg[:-1].unsqueeze(0))
        inputs_embeds = torch.cat([z.unsqueeze(0), ae, seg_emb], dim=1)

        hidden = self.decoder.encode(inputs_embeds)
        k = self.encoder.num_memory_tokens
        pred_hidden = hidden[0, k : k + n, :]

        loss_sum = pred_hidden.new_zeros((), dtype=torch.float32)
        use_ckpt = self.training and torch.is_grad_enabled()
        for start in range(0, n, self.ce_chunk_size):
            h = pred_hidden[start : start + self.ce_chunk_size]
            t = seg[start : start + self.ce_chunk_size]
            if use_ckpt:
                loss_sum = loss_sum + checkpoint(self._ce_chunk, h, t, use_reentrant=False)
            else:
                loss_sum = loss_sum + self._ce_chunk(h, t)

        return loss_sum / n


    def _student_forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = z.size(0)
        prefix_emb = self.decoder.embed(self.prefix_ids).unsqueeze(0).expand(bsz, -1, -1)
        suffix_emb = self.decoder.embed(self.suffix_ids).unsqueeze(0).expand(bsz, -1, -1)

        inputs_embeds = torch.cat([prefix_emb, z, suffix_emb], dim=1)
        hidden = self.decoder.encode(inputs_embeds)

        h_s = hidden[:, -1, :]
        z_s = self.decoder.regression_head(h_s).squeeze(-1)
        return h_s, z_s


    def forward(self, code_ids, code_mask, h_t=None, z_t=None):
        z = self.encoder(code_ids, code_mask)
        bsz = code_ids.size(0)
        device = code_ids.device

        total = torch.zeros((), device=device, dtype=torch.float32)
        out = {}

        if self.lambda_rec > 0:
            rec = torch.zeros((), device=device, dtype=torch.float32)
            for b in range(bsz):
                seg = code_ids[b][code_mask[b].bool()]
                if seg.numel() < 1:
                    continue
                rec = rec + self._recon_loss_one(z[b], seg)
            rec = rec / bsz
            total = total + self.lambda_rec * rec
            out["loss_rec"] = rec.detach()
        else:
            total = total + 0.0 * self.encoder.ae_token.float().sum()
            out["loss_rec"] = torch.zeros((), device=device)

        if (self.lambda_logit > 0 or self.lambda_repr > 0) and h_t is not None:
            h_s, z_s = self._student_forward(z)

            loss_logit = F.mse_loss(z_s.float(), z_t.float())
            cos = F.cosine_similarity(h_s.float(), h_t.float(), dim=-1)
            loss_repr = (1.0 - cos).mean()

            total = total + self.lambda_logit * loss_logit + self.lambda_repr * loss_repr
            out["loss_logit"] = loss_logit.detach()
            out["loss_repr"] = loss_repr.detach()
            out["cos"] = cos.detach().mean()
            out["z_s"] = z_s.detach()
        else:
            out["loss_logit"] = torch.zeros((), device=device)
            out["loss_repr"] = torch.zeros((), device=device)
            out["cos"] = torch.zeros((), device=device)
            out["z_s"] = torch.zeros(bsz, device=device)

        out["loss"] = total
        return out
