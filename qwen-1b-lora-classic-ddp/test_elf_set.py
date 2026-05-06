import json
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoTokenizer

from config import Qwen15BConfig
from model import MalwareDetectionModel
from utils import setup, cleanup, load_model_ddp
from lilm_mal_dataset import build_loaders
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def eval_worker(rank, config):
    setup(rank, config.world_size)
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        _, _, test_loader = build_loaders(config, tokenizer)

        model = load_model_ddp(config, rank, compile_model=False)
        checkpoint_path = os.path.join(config.output_dir, config.test_checkpoint_name)

        if os.path.exists(checkpoint_path):
            if rank == 0:
                print(f"[{rank}] Loading checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]

            clean_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace('module.', '')
                k = k.replace('_orig_mod.', '')
                clean_state_dict[k] = v

            missing, unexpected = model.module.load_state_dict(clean_state_dict, strict=True)
            if rank == 0:
                print(f"[{rank}] Loaded {len(clean_state_dict)} keys")
                if missing:
                    print(f"[{rank}] Missing keys: {missing}")
                if unexpected:
                    print(f"[{rank}] Unexpected keys: {unexpected}")
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        model.eval()

        all_preds_local = []
        all_labels_local = []
        all_probs_local = []
        all_idx_local = []

        pbar = tqdm(test_loader, desc=f"Testing [Rank {rank}]") if rank == 0 else test_loader

        with torch.no_grad():
            for batch in pbar:
                input_ids = batch['input_ids'].to(rank)
                attention_mask = batch['attention_mask'].to(rank)
                labels = batch['labels'].to(rank)
                indices = batch['idx'].to(rank)

                logits = model(input_ids, attention_mask)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                all_probs_local.append(probs.float())
                all_preds_local.append(preds.float())
                all_labels_local.append(labels.float())
                all_idx_local.append(indices.long())

        probs_t = torch.cat(all_probs_local).float()
        preds_t = torch.cat(all_preds_local).float()
        labels_t = torch.cat(all_labels_local).float()
        idx_t = torch.cat(all_idx_local).long()

        gathered_probs = [torch.zeros_like(probs_t) for _ in range(config.world_size)]
        gathered_preds = [torch.zeros_like(preds_t) for _ in range(config.world_size)]
        gathered_labels = [torch.zeros_like(labels_t) for _ in range(config.world_size)]
        gathered_idx = [torch.zeros_like(idx_t) for _ in range(config.world_size)]

        dist.all_gather(gathered_probs, probs_t)
        dist.all_gather(gathered_preds, preds_t)
        dist.all_gather(gathered_labels, labels_t)
        dist.all_gather(gathered_idx, idx_t)

        if rank == 0:
            all_probs_raw = torch.cat(gathered_probs).float().cpu().numpy()
            all_preds_raw = torch.cat(gathered_preds).float().cpu().numpy()
            all_labels_raw = torch.cat(gathered_labels).float().cpu().numpy()
            all_idx_raw = torch.cat(gathered_idx).long().cpu().numpy()

            unique_indices = {}
            for i, idx in enumerate(all_idx_raw):
                if idx not in unique_indices:
                    unique_indices[idx] = i

            sorted_keys = sorted(unique_indices.keys())
            dedup_indices = [unique_indices[k] for k in sorted_keys]

            all_probs = all_probs_raw[dedup_indices]
            all_preds = all_preds_raw[dedup_indices]
            all_labels = all_labels_raw[dedup_indices]

            print(f"Evaluated exactly {len(all_probs)} unique samples (Test set size expected: {len(test_loader.dataset)}).")
            print(f"Label distribution: benign={int((all_labels == 0).sum())}, malware={int((all_labels == 1).sum())}")
            print(f"Prediction distribution: benign={int((all_preds == 0).sum())}, malware={int((all_preds == 1).sum())}")
            print(f"Prob range: min={all_probs.min():.4f}, max={all_probs.max():.4f}, mean={all_probs.mean():.4f}")

            try:
                acc = accuracy_score(all_labels, all_preds)
                precision = precision_score(all_labels, all_preds, zero_division=0)
                recall = recall_score(all_labels, all_preds, zero_division=0)
                f1 = f1_score(all_labels, all_preds, zero_division=0)
                auc = roc_auc_score(all_labels, all_probs)
            except Exception as e:
                print(f"Error calculating metrics: {e}")
                acc, precision, recall, f1, auc = 0, 0, 0, 0, 0

            metrics = {
                "accuracy": float(acc),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "roc_auc": float(auc)
            }

            print("\n--- Test Metrics ---")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")

            os.makedirs(config.output_dir, exist_ok=True)
            out_file = os.path.join(config.output_dir, "test_metrics.json")

            with open(out_file, 'w') as f:
                json.dump(metrics, f, indent=4)

            print(f"\nMetrics saved to {out_file}")
    finally:
        cleanup()

def main():
    config = Qwen15BConfig()
    print(f"Starting test script with {config.world_size} GPUs...")
    mp.spawn(
        eval_worker,
        args=(config,),
        nprocs=config.world_size,
        join=True
    )

if __name__ == "__main__":
    main()
