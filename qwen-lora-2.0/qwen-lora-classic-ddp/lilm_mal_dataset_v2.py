import csv
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
    max_chunks = max(x["input_ids"].size(0) for x in batch)
    seq_len = batch[0]["input_ids"].size(1)

    input_ids = []
    masks = []
    chunk_masks = []
    labels = []

    for x in batch:
        ids = x["input_ids"]
        am = x["attention_mask"]
        n = ids.size(0)
        cm = x.get("chunk_mask", torch.ones(n, dtype=torch.long))

        pad_n = max_chunks - n
        if pad_n > 0:
            ids = torch.cat([ids, ids.new_zeros(pad_n, seq_len)], dim=0)
            am = torch.cat([am, am.new_zeros(pad_n, seq_len)], dim=0)
            cm = torch.cat([cm, cm.new_zeros(pad_n)], dim=0)

        input_ids.append(ids)
        masks.append(am)
        chunk_masks.append(cm)
        labels.append(x["label"] if "label" in x else x["labels"])

    res = {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(masks),
        "chunk_mask": torch.stack(chunk_masks),
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
        self.experiment_name = getattr(config, "experiment_name", "elf_v2_full")
        self.platform = getattr(config, "platform", "elf")
        self.splits_base = Path(config.data.splits_base)
        self.corpus_base = Path(config.data.corpus_base)
        self.min_code_chars = int(getattr(config, "min_code_chars", 0) or 0)
        self.samples = self._index()
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]
        
        prefix_str = (
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}\n<|im_end|>\n"
            f"<|im_start|>user\n{self.USER_HEADER}"
        )
        suffix_str = (
            f"{self.USER_FOOTER}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        self.prefix_ids = self.tok(prefix_str, add_special_tokens=False)["input_ids"]
        self.suffix_ids = self.tok(suffix_str, add_special_tokens=False)["input_ids"]
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        
        prompt_overhead = len(self.prefix_ids) + len(self.suffix_ids)
        self._budget = max(self.max_len - prompt_overhead, 100)

    def _is_empty_file(self, json_path: Path) -> bool:
        if self.min_code_chars <= 0:
            return False
        try:
            if json_path.stat().st_size >= self.min_code_chars + 4096:
                return False
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            code = data.get("decompiled_code") or ""
            return len(code) < self.min_code_chars
        except Exception:
            return True

    def _index(self) -> list[dict]:
        csv_path = self.splits_base / self.experiment_name / f"{self.split}.csv"
        samples = []
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found.")
            return samples
        dropped = 0
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                label = int(row[0])
                sha256 = row[1]
                label_dir = "malware" if label == 1 else "benign"
                json_path = self.corpus_base / self.platform / label_dir / f"{sha256}.json"
                if not json_path.exists():
                    continue
                if self._is_empty_file(json_path):
                    dropped += 1
                    continue
                samples.append({"path": str(json_path), "label": label})
        samples.sort(key=lambda x: x["path"])
        if dropped and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(
                f"[info] [dataset] split '{self.split}': dropped {dropped} files "
                f"with decompiled_code < {self.min_code_chars} chars"
            )
        return samples

    def _load_code(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("decompiled_code") or ""
        except Exception:
            return ""

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        code = self._load_code(sample["path"])
        
        code_ids = []
        if code:
            code_ids = self.tok(
                code, 
                add_special_tokens=False, 
                truncation=True, 
                max_length=self._budget * self.num_chunks,
            )["input_ids"]
            
        if code_ids:
            n_chunks = min(self.num_chunks, math.ceil(len(code_ids) / self._budget))
        else:
            n_chunks = 1

        input_ids_list = []
        attention_mask_list = []

        for i in range(n_chunks):
            start = i * self._budget
            end = start + self._budget
            chunk_code_ids = code_ids[start:end]

            full_ids = self.prefix_ids + chunk_code_ids + self.suffix_ids

            pad_len = self.max_len - len(full_ids)
            if pad_len > 0:
                if self.tok.padding_side == "left":
                    full_ids = ([self.pad_id] * pad_len) + full_ids
                    mask = ([0] * pad_len) + ([1] * (self.max_len - pad_len))
                else:
                    full_ids = full_ids + ([self.pad_id] * pad_len)
                    mask = ([1] * (self.max_len - pad_len)) + ([0] * pad_len)
            else:
                full_ids = full_ids[:self.max_len]
                mask = [1] * self.max_len

            input_ids_list.append(full_ids)
            attention_mask_list.append(mask)

        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
            "chunk_mask": torch.ones(n_chunks, dtype=torch.long),
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
