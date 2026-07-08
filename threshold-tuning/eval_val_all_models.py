import json
import os
import csv
import random
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

from models.model_1_5 import MalwareDetectionModel_1_5
from models.model_1_5_lora import MalwareDetectionModel_1_5_lora
from models.model_3 import MalwareDetectionModel_3
from models.model_3_lora import MalwareDetectionModel_3_lora
from utils import setup, cleanup
from lilm_mal_dataset import LiLMMalDataset, malware_collate_fn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP




@dataclass
class ModelEntry:
    model_cls: type
    config_cls: type        
    checkpoint_name: str    
    output_dir: str         
    label: str         

from config import (
    Qwen15BConfig,
    Qwen15BLoraClassicConfig,
    Qwen15BLoraAttentionConfig,
    Qwen15BLoraFullConfig19M,
    Qwen15BLoraFullConfig38M,
    Qwen15BLoraFullConfig76M,
    Qwen3BConfig,
    Qwen3BLoraClassicConfig,
    Qwen3BLoraAttentionConfig,
    Qwen3BLoraFullConfig,
)

MODELS: list[ModelEntry] = [
    ModelEntry(
        model_cls=MalwareDetectionModel_1_5,
        config_cls=Qwen15BConfig,
        checkpoint_name="qwen_malware_best.pt",
        output_dir="outputs/checkpoints-q1.5b",
        label="q1.5b",
    ),
    ModelEntry(
        model_cls=MalwareDetectionModel_1_5_lora,
        config_cls=Qwen15BLoraClassicConfig,
        checkpoint_name="qwen_malware_best.pt",
        output_dir="outputs/checkpoints-q1.5b-lora-classic",
        label="q1.5b-lora-classic",
    ),
    ModelEntry(
        model_cls=MalwareDetectionModel_1_5_lora,
        config_cls=Qwen15BLoraAttentionConfig,
        checkpoint_name="qwen_malware_best.pt",
        output_dir="outputs/checkpoints-q1.5b-lora-attention",
        label="q1.5b-lora-attention",
    ),
    ModelEntry(
        model_cls=MalwareDetectionModel_1_5_lora,
        config_cls=Qwen15BLoraFullConfig19M,
        checkpoint_name="qwen_malware_best.pt",
        output_dir="outputs/checkpoints-q1.5b-lora-full-19M",
        label="q1.5b-lora-full-19M",
    ),
    ModelEntry(
        model_cls=MalwareDetectionModel_1_5_lora,
        config_cls=Qwen15BLoraFullConfig38M,
        checkpoint_name="qwen_malware_best.pt",
        output_dir="outputs/checkpoints-q1.5b-lora-full-38M",
        label="q1.5b-lora-full-38M",
    ),    
    ModelEntry(
        model_cls=MalwareDetectionModel_1_5_lora,
        config_cls=Qwen15BLoraFullConfig76M,
        checkpoint_name="qwen_malware_best.pt",
        output_dir="outputs/checkpoints-q1.5b-lora-full-76M",
        label="q1.5b-lora-full-76M",
    ),
    ModelEntry(
        model_cls=MalwareDetectionModel_3,
        config_cls=Qwen3BConfig,
        checkpoint_name="qwen_malware_best.pt",
        output_dir="outputs/checkpoints-q3b",
        label="q3b",
    ),
    ModelEntry(
        model_cls=MalwareDetectionModel_3_lora,
        config_cls=Qwen3BLoraClassicConfig,
        checkpoint_name="qwen_malware_best.pt",
        output_dir="outputs/checkpoints-q3b-lora-classic",
        label="q3b-lora-classic",
    ),
    ModelEntry(
        model_cls=MalwareDetectionModel_3_lora,
        config_cls=Qwen3BLoraAttentionConfig,
        checkpoint_name="qwen_malware_best.pt",
        output_dir="outputs/checkpoints-q3b-lora-attention",
        label="q3b-lora-attention",
    ),
    ModelEntry(
        model_cls=MalwareDetectionModel_3_lora,
        config_cls=Qwen3BLoraFullConfig,
        checkpoint_name="qwen_malware_best.pt",
        output_dir="outputs/checkpoints-q3b-lora-full",
        label="q3b-lora-full",
    ),
]


def load_model_ddp(model_cls, config, rank, compile_model=True):
    torch.cuda.set_device(rank)

    model = model_cls(config).to(rank)
    if compile_model:
        model = torch.compile(model)

    model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False,
    )

    return model

def build_val_loader_1_9(config, tokenizer, rank: int, world_size: int) -> tuple[DataLoader, LiLMMalDataset]:
    full_ds = LiLMMalDataset(split="train", tokenizer=tokenizer, config=config)

    benign_idx  = [i for i, s in enumerate(full_ds.samples) if s["label"] == 0]
    malware_idx = [i for i, s in enumerate(full_ds.samples) if s["label"] == 1]

    rng = random.Random(42)
    rng.shuffle(benign_idx)
    rng.shuffle(malware_idx)

    n_val_benign  = int(len(benign_idx)  * 0.10)
    n_val_malware = int(len(malware_idx) * 0.10)

    val_benign_pool  = benign_idx[:n_val_benign]
    val_malware_pool = malware_idx[:n_val_malware]

    n_benign_keep  = len(val_benign_pool)
    n_malware_keep = max(1, n_benign_keep // 9)
    n_malware_keep = min(n_malware_keep, len(val_malware_pool)) 

    rng2 = random.Random(42)
    val_malware_sub = rng2.sample(val_malware_pool, n_malware_keep)

    val_indices = val_benign_pool + val_malware_sub

    if rank == 0:
        print(f"[info] val set → benign: {n_benign_keep} | malware: {n_malware_keep} "
              f"| ratio malware:benign = 1:{n_benign_keep // max(n_malware_keep, 1)}")

    full_ds.samples = [full_ds.samples[i] for i in val_indices]
    val_ds = full_ds

    val_sampler = None
    if dist.is_available() and dist.is_initialized():
        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    batch_size = getattr(config, "test_batch_size", config.batch_size)

    loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        sampler=val_sampler,
        collate_fn=malware_collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    return loader, val_ds



def eval_one_model_worker(rank: int, entry: ModelEntry, port: str):
    config = entry.config_cls()
    setup(rank, config.world_size, port=port)

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        val_loader, val_ds = build_val_loader_1_9(config, tokenizer, rank, config.world_size)

        model = load_model_ddp(entry.model_cls, config, rank, compile_model=False)
        checkpoint_path = os.path.join(entry.output_dir, entry.checkpoint_name)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint negăsit: {checkpoint_path}")

        if rank == 0:
            print(f"[{entry.label}] Loading checkpoint: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        clean_sd = {
            k.replace("module.", "").replace("_orig_mod.", ""): v
            for k, v in state_dict.items()
        }
        missing, unexpected = model.module.load_state_dict(clean_sd, strict=True)
        if rank == 0:
            print(f"[{entry.label}] Loaded {len(clean_sd)} keys "
                  f"| missing={len(missing)} | unexpected={len(unexpected)}")

        model.eval()

        all_logits_l, all_probs_l, all_preds_l, all_labels_l, all_idx_l = [], [], [], [], []

        pbar = tqdm(val_loader, desc=f"[{entry.label}] val inference [rank {rank}]") \
            if rank == 0 else val_loader

        with torch.no_grad():
            for batch in pbar:
                input_ids     = batch["input_ids"].to(rank)
                attention_mask = batch["attention_mask"].to(rank)
                labels        = batch["labels"].to(rank)
                indices       = batch["idx"].to(rank)

                logits = model(input_ids, attention_mask)
                probs  = torch.sigmoid(logits)
                preds  = (probs > 0.5).float()

                all_logits_l.append(logits.float())
                all_probs_l.append(probs.float())
                all_preds_l.append(preds.float())
                all_labels_l.append(labels.float())
                all_idx_l.append(indices.long())

        logits_t = torch.cat(all_logits_l)
        probs_t  = torch.cat(all_probs_l)
        preds_t  = torch.cat(all_preds_l)
        labels_t = torch.cat(all_labels_l)
        idx_t    = torch.cat(all_idx_l)

        def _gather(t):
            out = [torch.zeros_like(t) for _ in range(config.world_size)]
            dist.all_gather(out, t)
            return torch.cat(out)

        g_logits = _gather(logits_t)
        g_probs  = _gather(probs_t)
        g_preds  = _gather(preds_t)
        g_labels = _gather(labels_t)
        g_idx    = _gather(idx_t)

        if rank == 0:
            logits_np = g_logits.cpu().numpy()
            probs_np  = g_probs.cpu().numpy()
            preds_np  = g_preds.cpu().numpy()
            labels_np = g_labels.cpu().numpy()
            idx_np    = g_idx.cpu().numpy()

            unique_indices: dict[int, int] = {}
            for pos, idx in enumerate(idx_np):
                if int(idx) not in unique_indices:
                    unique_indices[int(idx)] = pos

            sorted_keys  = sorted(unique_indices.keys())
            dedup_pos    = [unique_indices[k] for k in sorted_keys]

            logits_f = logits_np[dedup_pos]
            probs_f  = probs_np[dedup_pos]
            preds_f  = preds_np[dedup_pos]
            labels_f = labels_np[dedup_pos]

            print(f"[{entry.label}] unique samples: {len(probs_f)}")
            print(f"[{entry.label}] benign={int((labels_f == 0).sum())} "
                  f"malware={int((labels_f == 1).sum())}")
            print(f"[{entry.label}] prob range: "
                  f"min={probs_f.min():.4f} max={probs_f.max():.4f} mean={probs_f.mean():.4f}")

            os.makedirs(entry.output_dir, exist_ok=True)
            csv_path = os.path.join(entry.output_dir, "val_results_elf_1_9.csv")

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "logit", "label", "pred", "prob"])
                for i, ds_idx in enumerate(sorted_keys):
                    sample   = val_ds.samples[ds_idx]
                    filename = Path(sample["path"]).name
                    writer.writerow([
                        filename,
                        float(logits_f[i]),
                        int(labels_f[i]),
                        int(preds_f[i]),
                        float(probs_f[i]),
                    ])

            print(f"[{entry.label}] ✓ Salvat: {csv_path}")

    finally:
        cleanup()




def main():
    base_port = 12355
    for i, entry in enumerate(MODELS):
        csv_path = os.path.join(entry.output_dir, "val_results_elf_1_9.csv")
        if os.path.exists(csv_path):
            print(f"\n{'='*60}")
            print(f"  Model: {entry.label}")
            print(f"  [SKIPPED] Fisierul de rezultate exista deja: {csv_path}")
            print(f"{'='*60}")
            continue

        config = entry.config_cls()
        current_port = str(base_port + i)
        print(f"\n{'='*60}")
        print(f"  Model: {entry.label}")
        print(f"  Checkpoint: {entry.output_dir}/{entry.checkpoint_name}")
        print(f"  Port DDP: {current_port}")
        print(f"{'='*60}")

        mp.spawn(
            eval_one_model_worker,
            args=(entry, current_port),
            nprocs=config.world_size,
            join=True,
        )

    print("\n✓ Toate modelele procesate.")
    print("CSV-urile salvate: outputs/checkpoints-*/val_results_elf_1_9.csv")

if __name__ == "__main__":
    main()