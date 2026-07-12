import csv
import math
import os
import random
import torch
import torch.distributed as dist
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


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
USER_FOOTER = "\n</code>"


def build_prompt_ids(tokenizer) -> tuple[list[int], list[int]]:
    prefix_str = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{USER_HEADER}"
    )
    suffix_str = (
        f"{USER_FOOTER}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    prefix_ids = tokenizer(prefix_str, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix_str, add_special_tokens=False)["input_ids"]
    return prefix_ids, suffix_ids


def seg_code_budget(config) -> int:
    budget = int(config.max_token_len) - int(config.num_memory_tokens)
    if budget <= 0:
        raise ValueError(
            f"num_memory_tokens ({config.num_memory_tokens}) >= max_token_len ({config.max_token_len})"
        )
    return budget


def cache_path_for(cache_dir, sha: str) -> Path:
    return Path(cache_dir) / f"{sha}.pt"


def manifest_path(cache_dir) -> Path:
    return Path(cache_dir) / "manifest.csv"


def read_manifest(cache_dir) -> list[dict]:
    path = manifest_path(cache_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path} — run the distillation precompute_teacher.py first"
        )
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            rows.append({"sha": row[0], "label": int(row[1]), "n_seg": int(row[2])})
    rows.sort(key=lambda x: x["sha"])
    return rows


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


class FileDataset(Dataset):
    def __init__(self, items: list[tuple], cache_dir):
        self.items = items
        self.cache_dir = Path(cache_dir)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        sha, label = self.items[idx]
        d = torch.load(
            cache_path_for(self.cache_dir, sha),
            map_location="cpu",
            weights_only=True,
        )
        segs = [s.long() for s in d["segs"]]
        max_len = max(s.size(0) for s in segs)

        code_ids = []
        code_masks = []
        for ids in segs:
            pad_n = max_len - ids.size(0)
            code_ids.append(torch.cat([ids.new_zeros(pad_n), ids]))
            code_masks.append(
                torch.cat([
                    torch.zeros(pad_n, dtype=torch.long),
                    torch.ones(ids.size(0), dtype=torch.long),
                ])
            )

        return {
            "code_ids": torch.stack(code_ids),
            "code_mask": torch.stack(code_masks),
            "label": torch.tensor(label, dtype=torch.long),
        }


def file_collate_fn(batch: list[dict]) -> dict:
    if len(batch) != 1:
        raise ValueError(f"file-level training requires batch_size=1, got {len(batch)}")
    x = batch[0]
    return {
        "code_ids": x["code_ids"],
        "code_mask": x["code_mask"],
        "labels": x["label"].unsqueeze(0),
    }


class FileDataLoader(DataLoader):
    def __init__(self, dataset: FileDataset, config, sampler=None, shuffle=True, drop_last=True):
        super().__init__(
            dataset,
            batch_size=config.batch_size,
            shuffle=(shuffle and sampler is None),
            num_workers=config.num_workers,
            sampler=sampler,
            collate_fn=file_collate_fn,
            pin_memory=True,
            drop_last=drop_last,
            persistent_workers=True if config.num_workers > 0 else False,
            prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        )


def build_loaders(config) -> tuple[DataLoader, DataLoader]:
    if int(config.batch_size) != 1:
        raise ValueError(f"file-level training requires batch_size=1, got {config.batch_size}")

    rows = read_manifest(config.teacher_cache_dir)

    rng = random.Random(42)
    shas = [r["sha"] for r in rows]
    rng.shuffle(shas)
    n_val = max(1, int(len(shas) * float(config.val_fraction)))
    val_shas = set(shas[:n_val])

    train_items, val_items = [], []
    for r in rows:
        target = val_items if r["sha"] in val_shas else train_items
        target.append((r["sha"], r["label"]))

    val_max = int(getattr(config, "val_max_files", 0) or 0)
    if val_max and len(val_items) > val_max:
        rng.shuffle(val_items)
        val_items = val_items[:val_max]

    train_ds = FileDataset(train_items, config.teacher_cache_dir)
    val_ds = FileDataset(val_items, config.teacher_cache_dir)

    _is_dist = dist.is_available() and dist.is_initialized()

    train_sampler = None
    val_sampler = None
    if _is_dist:
        train_sampler = OffsetDistributedSampler(
            train_ds,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
            drop_last=False,
        )

    train_loader = FileDataLoader(train_ds, config, sampler=train_sampler)
    val_loader = FileDataLoader(val_ds, config, sampler=val_sampler, shuffle=False, drop_last=False)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(
            f"[info] [dataset] files: {len(rows)} | train files: {len(train_items)} "
            f"| val files: {len(val_items)}"
        )

    return train_loader, val_loader
