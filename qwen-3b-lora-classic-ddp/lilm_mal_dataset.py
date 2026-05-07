import json
import os
import math
import random
import torch
import torch.distributed as dist
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizer

LABEL_MAP = {0: "benign", 1: "malware"}


class OffsetDistributedSampler(DistributedSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_index = 0

    def set_start_index(self, start_index: int):
        self.start_index = max(0, int(start_index))

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                repeat_times = math.ceil(padding_size / len(indices))
                indices += (indices * repeat_times)[:padding_size]
        else:
            indices = indices[:self.total_size]

        indices = indices[self.rank:self.total_size:self.num_replicas]

        start = min(self.start_index, len(indices))
        indices = indices[start:]
        return iter(indices)

    def __len__(self):
        return max(0, self.num_samples - min(self.start_index, self.num_samples))

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

class LiLMMalDataset(Dataset):
    SYSTEM_PROMPT = (
        "You are a senior reverse engineer specializing in Linux ELF malware analysis. "
        "You analyze decompiled binary code and identify malicious behavior patterns "
        "such as privilege escalation, persistence mechanisms, network exfiltration, "
        "process injection, and obfuscation techniques (pay attention on what are this operations applied)."
    )

    USER_HEADER = (
        "Analyze the following decompiled ELF binary. "
        "Focus on: suspicious syscalls, anti-analysis tricks, "
        "hardcoded C2 indicators, and abnormal control flow.\n\n"
        "<code>\n"
    )
    USER_FOOTER   = "\n</code>"

    def __init__(self, split: str, tokenizer: PreTrainedTokenizer, config, indices: list[int] | None = None):
        self.split = split
        self.tok = tokenizer
        self.tok.padding_side = "left"
        self.max_len = config.max_token_len
        self.num_chunks = config.num_chunks
        self.base = Path("/media/sebi/nvme-1tb/LiLM-Mal-Dataset/decompiled-40")
        self.samples = self._index()
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]
        
        empty_prompt = self._build_prompt("")
        prompt_overhead = self.tok(empty_prompt, return_tensors="pt")["input_ids"].shape[1]
        self._budget = max(self.max_len - prompt_overhead - 5, 100)

    def _index(self) -> list[dict]:
        samples = []
        for label_int, name in [(0, "benign"), (1, "malware")]:
            target_dir = self.base / self.split / name
            if target_dir.exists():
                for f in target_dir.glob("*.json"):
                    samples.append({"path": str(f), "label": label_int})
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

class LiLMMalDataLoader(DataLoader):
    def __init__(self, dataset: LiLMMalDataset, config, sampler: DistributedSampler | None = None, shuffle: bool = True, is_test: bool = False):
        batch_size = getattr(config, "test_batch_size", config.batch_size) if is_test else config.batch_size
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=(shuffle and sampler is None),
            num_workers=config.num_workers,
            sampler=sampler,
            collate_fn=malware_collate_fn,
            pin_memory=True,
            drop_last=(not is_test) and (True if sampler is not None else False),
            persistent_workers=True if config.num_workers > 0 else False,
            prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        )

def build_loaders(config, tokenizer) -> tuple[DataLoader, DataLoader, DataLoader]:
    full_ds = LiLMMalDataset(split="train", tokenizer=tokenizer, config=config)

    benign_idx = [i for i, s in enumerate(full_ds.samples) if s["label"] == 0]
    malware_idx = [i for i, s in enumerate(full_ds.samples) if s["label"] == 1]

    rng = random.Random(42)
    rng.shuffle(benign_idx)
    rng.shuffle(malware_idx)

    n_val_benign = int(len(benign_idx) * 0.10)
    n_val_malware = int(len(malware_idx) * 0.10)

    val_indices = benign_idx[:n_val_benign] + malware_idx[:n_val_malware]
    train_indices = benign_idx[n_val_benign:] + malware_idx[n_val_malware:]

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    train_ds = LiLMMalDataset(split="train", tokenizer=tokenizer, config=config, indices=train_indices)
    val_ds = LiLMMalDataset(split="train", tokenizer=tokenizer, config=config, indices=val_indices)
    test_ds = LiLMMalDataset(split="test", tokenizer=tokenizer, config=config)

    _is_dist = dist.is_available() and dist.is_initialized()
    
    train_sampler = None
    test_sampler = None
    if _is_dist and config.use_distributed_sampler:
        train_sampler = OffsetDistributedSampler(
            train_ds,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            drop_last=True,
        )
        test_sampler = DistributedSampler(
            test_ds,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
            drop_last=False,
        )

    train_loader = LiLMMalDataLoader(train_ds, config, sampler=train_sampler)
    val_loader = LiLMMalDataLoader(val_ds, config, shuffle=False)
    test_loader = LiLMMalDataLoader(test_ds, config, sampler=test_sampler, shuffle=False, is_test=True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(f"[info] [dataset] train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")
        
    return train_loader, val_loader, test_loader