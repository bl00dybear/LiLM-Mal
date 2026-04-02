import json
import os
import random
import torch
import torch.distributed as dist
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizer

LABEL_MAP = {0: "benign", 1: "malware"}

class LiLMMalDataset(Dataset):

    SYSTEM_PROMPT = "You are a binary analysis expert specializing in ELF malware detection."
    USER_HEADER   = "Analyze the following decompiled ELF binary code and classify it.\nAnswer with exactly one word: malware or benign.\n\n<code>\n"
    USER_FOOTER   = "\n</code>"

    def __init__(self, split: str, tokenizer: PreTrainedTokenizer, max_length: int = 4096, num_chunks: int = 4, decompiled_base: str = "/media/sebi/nvme-1tb/LiLM-Mal-Dataset/decompiled", indices: list[int] | None = None):
        self.split    = split
        self.tok      = tokenizer
        self.tok.padding_side = "left"
        self.max_len  = max_length
        self.num_chunks = num_chunks
        self.base     = Path(decompiled_base)
        self.samples  = self._index()
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]
        self._budget  = self._code_budget()

    def _index(self) -> list[dict]:
        samples = []
        for label_int, name in [(0, "benign"), (1, "malware")]:
            for f in (self.base / self.split / name).glob("*.json"):
                samples.append({"path": str(f), "label": label_int})
        samples.sort(key=lambda x: x["path"])
        return samples

    def _code_budget(self) -> int:
        empty = self._build_prompt("")
        n     = self.tok(empty, return_tensors="pt")["input_ids"].shape[1]
        return max(self.max_len - n - 10, 100)

    def _load_code(self, path: str) -> str:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            return data.get("decompiled_code") or "// decompilation unavailable"
        except Exception:
            return "// decompilation unavailable"

    def _chunk_code(self, code: str) -> list[str]:
        max_ids = self._budget * self.num_chunks
        ids = self.tok(code, add_special_tokens=False, truncation=True, max_length=max_ids, return_tensors="pt")["input_ids"][0]
        
        chunks = []
        for i in range(self.num_chunks):
            start = i * self._budget
            end = start + self._budget
            chunk_ids = ids[start:end]
            if len(chunk_ids) > 0:
                chunks.append(self.tok.decode(chunk_ids, skip_special_tokens=True))
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
        sample    = self.samples[idx]
        code      = self._load_code(sample["path"])
        chunks    = self._chunk_code(code)
        label     = sample["label"]
        
        prompts   = [self._build_prompt(c) for c in chunks]
        encoded   = self.tok(
            prompts, 
            max_length=self.max_len, 
            truncation=True, 
            padding="max_length", 
            return_tensors="pt"
        )
        
        return {
            "input_ids":      encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "label":          torch.tensor(label, dtype=torch.long),
        }


class LiLMMalDataLoader(DataLoader):

    def __init__(
        self,
        dataset: LiLMMalDataset,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 4,
        sampler: DistributedSampler | None = None,
        drop_last: bool = False,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
    ):
        pad_id = dataset.tok.pad_token_id or dataset.tok.eos_token_id
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=(shuffle and sampler is None),
            num_workers=num_workers,
            sampler=sampler,
            collate_fn=lambda batch: self._collate(batch, pad_id),
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=(persistent_workers and num_workers > 0),
            prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        )
        self.max_length = dataset.max_len

    def _collate(self, batch: list[dict], pad_id: int) -> dict:
        input_ids = [x["input_ids"] for x in batch]
        masks = [x["attention_mask"] for x in batch]
        labels = [x["label"] for x in batch]
        
        return {
            "input_ids":      torch.stack(input_ids),
            "attention_mask": torch.stack(masks),
            "labels":         torch.stack(labels),
        }


def build_loaders(config, tokenizer) -> tuple[DataLoader, DataLoader]:
    full_ds = LiLMMalDataset(
        split="train",
        tokenizer=tokenizer,
        max_length=config.max_token_len,
        num_chunks=config.num_chunks,
    )

    benign_idx  = [i for i, s in enumerate(full_ds.samples) if s["label"] == 0]
    malware_idx = [i for i, s in enumerate(full_ds.samples) if s["label"] == 1]

    rng = random.Random(42)
    rng.shuffle(benign_idx)
    rng.shuffle(malware_idx)

    n_val_benign  = int(len(benign_idx)  * 0.10)
    n_val_malware = int(len(malware_idx) * 0.10)

    val_indices   = benign_idx[:n_val_benign]  + malware_idx[:n_val_malware]
    train_indices = benign_idx[n_val_benign:]  + malware_idx[n_val_malware:]

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    train_ds = LiLMMalDataset(
        split="train",
        tokenizer=tokenizer,
        max_length=config.max_token_len,
        num_chunks=config.num_chunks,
        indices=train_indices,
    )
    val_ds = LiLMMalDataset(
        split="train",
        tokenizer=tokenizer,
        max_length=config.max_token_len,
        num_chunks=config.num_chunks,
        indices=val_indices,
    )

    train_b = sum(1 for s in train_ds.samples if s["label"] == 0)
    train_m = sum(1 for s in train_ds.samples if s["label"] == 1)
    val_b   = sum(1 for s in val_ds.samples   if s["label"] == 0)
    val_m   = sum(1 for s in val_ds.samples   if s["label"] == 1)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(f"[Train] Total: {len(train_ds)} | Benign: {train_b} | Malware: {train_m}")
        print(f"[Val]   Total: {len(val_ds)}   | Benign: {val_b}   | Malware: {val_m}")

    _is_dist = dist.is_available() and dist.is_initialized()
    train_sampler = None
    if _is_dist:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            drop_last=True,
        )

    train_loader = LiLMMalDataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        sampler=train_sampler,
        drop_last=True,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
    )
    val_loader = LiLMMalDataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
    )

    return train_loader, val_loader