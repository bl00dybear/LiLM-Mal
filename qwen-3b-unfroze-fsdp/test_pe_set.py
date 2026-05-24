import json
import os
import math
import csv
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import Quen3BConfig
from model import MalwareDetectionModel
from utils import setup, cleanup, load_model_ddp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

LABEL_MAP = {0: "benign", 1: "malware"}

def malware_collate_fn(batch: list[dict]) -> dict:
    input_ids = [x["input_ids"] for x in batch]
    masks = [x["attention_mask"] for x in batch]
    labels = [x["label"] for x in batch]
    
    res = {
        "input_ids": torch.stack(input_ids), 
        "attention_mask": torch.stack(masks),     
        "labels": torch.stack(labels),    
    }
    if "idx" in batch[0]:
        indices = [x["idx"] for x in batch]
        res["idx"] = torch.stack(indices)
    return res

class LiLMPEDataset(Dataset):
    SYSTEM_PROMPT = (
        "You are a senior reverse engineer specializing in Windows PE malware analysis. "
        "You analyze decompiled binary code and identify malicious behavior patterns "
        "such as privilege escalation, persistence mechanisms, network exfiltration, "
        "process injection, and obfuscation techniques (pay attention on what are this operations applied)."
    )

    USER_HEADER = (
        "Analyze the following decompiled PE binary. "
        "Focus on: suspicious syscalls, anti-analysis tricks, "
        "hardcoded C2 indicators, and abnormal control flow.\n\n"
        "<code>\n"
    )
    USER_FOOTER   = "\n</code>"

    def __init__(self, tokenizer: PreTrainedTokenizer, config):
        self.tok = tokenizer
        self.tok.padding_side = "left"
        self.max_len = config.max_token_len
        self.num_chunks = config.num_chunks
        self.base = Path("/run/media/sebi/nvme-1tb/LiLM-Mal-Dataset/pe-decompiled/subsampled/test")
        
        empty_prompt = self._build_prompt("")
        prompt_overhead = self.tok(empty_prompt, return_tensors="pt")["input_ids"].shape[1]
        self._budget = max(self.max_len - prompt_overhead - 5, 100)
        
        self.samples = self._index()

    def _index(self) -> list[dict]:
        samples = []
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        for label_int, name in [(0, "benign"), (1, "malware")]:
            target_dir = self.base / name
            if target_dir.exists():
                files = list(target_dir.glob("*.json"))
                
                iterator = files
                for f in iterator:
                    try:
                        with open(f, "r", encoding="utf-8") as file:
                            data = json.load(file)
                        code = data.get("decompiled_code") or ""
                        if len(code) < 50:
                            continue
                            
                        samples.append({"path": str(f), "label": label_int})
                    except Exception:
                        pass
        samples.sort(key=lambda x: x["path"])
        return samples

    def _load_code(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("decompiled_code") or ""
        except Exception:
            return ""

    def _chunk_code(self, code: str) -> list[str]:
        if not code:
            return [""] * self.num_chunks
            
        full_tokens = self.tok(
            code, 
            add_special_tokens=False, 
            truncation=True, 
            max_length=self._budget * self.num_chunks, 
            return_tensors="pt"
        )["input_ids"][0]
        
        chunks = []
        for i in range(self.num_chunks):
            start = i * self._budget
            end = start + self._budget
            chunk_ids = full_tokens[start:end]
            if len(chunk_ids) > 0:
                chunks.append(self.tok.decode(chunk_ids, skip_special_tokens=False))
            else:
                chunks.append("")
        return chunks

    def _build_prompt(self, code: str) -> str:
        return (
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}\n<|im_end|>\n"
            f"<|im_start|>user\n{self.USER_HEADER}{code}{self.USER_FOOTER}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        code = self._load_code(sample["path"])
        chunks = self._chunk_code(code)
        
        prompts = [self._build_prompt(c) for c in chunks]
        encoded = self.tok(
            prompts, 
            max_length=self.max_len, 
            truncation=True, 
            padding="max_length", 
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"],      
            "attention_mask": encoded["attention_mask"], 
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "idx": torch.tensor(idx, dtype=torch.long),
        }


def build_pe_test_loader(config, tokenizer) -> DataLoader:
    test_ds = LiLMPEDataset(tokenizer=tokenizer, config=config)

    _is_dist = dist.is_available() and dist.is_initialized()
    
    test_sampler = None
    if _is_dist and config.use_distributed_sampler:
        test_sampler = DistributedSampler(
            test_ds,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
            drop_last=False,
        )

    batch_size = getattr(config, "test_batch_size", config.batch_size)
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        sampler=test_sampler,
        collate_fn=malware_collate_fn,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if config.num_workers > 0 else False,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(f"[info] [dataset-pe] test size: {len(test_ds)}")
        
    return test_loader


def eval_worker(rank, config):
    setup(rank, config.world_size)
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        test_loader = build_pe_test_loader(config, tokenizer)

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
        all_logits_local = []
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

                all_logits_local.append(logits.float())
                all_probs_local.append(probs.float())
                all_preds_local.append(preds.float())
                all_labels_local.append(labels.float())
                all_idx_local.append(indices.long())

        logits_t = torch.cat(all_logits_local).float()
        probs_t = torch.cat(all_probs_local).float()
        preds_t = torch.cat(all_preds_local).float()
        labels_t = torch.cat(all_labels_local).float()
        idx_t = torch.cat(all_idx_local).long()

        gathered_logits = [torch.zeros_like(logits_t) for _ in range(config.world_size)]
        gathered_probs = [torch.zeros_like(probs_t) for _ in range(config.world_size)]
        gathered_preds = [torch.zeros_like(preds_t) for _ in range(config.world_size)]
        gathered_labels = [torch.zeros_like(labels_t) for _ in range(config.world_size)]
        gathered_idx = [torch.zeros_like(idx_t) for _ in range(config.world_size)]

        dist.all_gather(gathered_logits, logits_t)
        dist.all_gather(gathered_probs, probs_t)
        dist.all_gather(gathered_preds, preds_t)
        dist.all_gather(gathered_labels, labels_t)
        dist.all_gather(gathered_idx, idx_t)

        if rank == 0:
            all_logits_raw = torch.cat(gathered_logits).float().cpu().numpy()
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

            all_logits = all_logits_raw[dedup_indices]
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
            out_file = os.path.join(config.output_dir, "test_metrics_pe.json")

            with open(out_file, 'w') as f:
                json.dump(metrics, f, indent=4)

            print(f"\nMetrics saved to {out_file}")

            csv_file = os.path.join(config.output_dir, "test_results_pe.csv")
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "logit", "label", "pred", "prob"])
                for i, idx in enumerate(sorted_keys):
                    sample = test_loader.dataset.samples[int(idx)]
                    filename = Path(sample["path"]).name
                    logit_val = float(all_logits[i])
                    label_val = int(all_labels[i])
                    pred_val = int(all_preds[i])
                    prob_val = float(all_probs[i])
                    writer.writerow([filename, logit_val, label_val, pred_val, prob_val])
            print(f"Results saved to {csv_file}")
    finally:
        cleanup()

def main():
    config = Quen3BConfig()
    print(f"Starting PE test script with {config.world_size} GPUs...")
    mp.spawn(
        eval_worker,
        args=(config,),
        nprocs=config.world_size,
        join=True
    )

if __name__ == "__main__":
    main()
