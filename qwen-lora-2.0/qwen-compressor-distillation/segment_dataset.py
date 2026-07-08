import csv
import json
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


def tokenize_and_segment(code: str, tokenizer, budget: int, max_segments: int) -> list[list[int]]:
    """Segmentarea canonica — folosita si de precompute_teacher si de training,
    ca teacher si student sa vada exact aceiasi tokeni."""
    if not code:
        return []
    ids = tokenizer(
        code,
        add_special_tokens=False,
        truncation=True,
        max_length=budget * max_segments,
    )["input_ids"]
    if not ids:
        return []
    n = min(max_segments, math.ceil(len(ids) / budget))
    return [ids[i * budget:(i + 1) * budget] for i in range(n)]


def load_code(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("decompiled_code") or ""
    except Exception:
        return ""


def _is_empty_file(json_path: Path, min_code_chars: int) -> bool:
    if min_code_chars <= 0:
        return False
    try:
        if json_path.stat().st_size >= min_code_chars + 4096:
            return False
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        code = data.get("decompiled_code") or ""
        return len(code) < min_code_chars
    except Exception:
        return True


def index_corpus(config) -> list[dict]:
    """Fisierele din splitul de train (aceeasi filtrare ca in lilm_mal_dataset_v2),
    optional subesantionate stratificat cu max_files."""
    splits_base = Path(config.data.splits_base)
    corpus_base = Path(config.data.corpus_base)
    experiment_name = getattr(config.data, "experiment_name", "elf_v2_full")
    platform = getattr(config.data, "platform", "elf")
    min_code_chars = int(getattr(config, "min_code_chars", 0) or 0)

    csv_path = splits_base / experiment_name / "train.csv"
    samples = []
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found.")
        return samples

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            label = int(row[0])
            sha256 = row[1]
            label_dir = "malware" if label == 1 else "benign"
            json_path = corpus_base / platform / label_dir / f"{sha256}.json"
            if not json_path.exists():
                continue
            if _is_empty_file(json_path, min_code_chars):
                continue
            samples.append({"path": str(json_path), "sha": sha256, "label": label})

    samples.sort(key=lambda x: x["sha"])

    max_files = getattr(config, "max_files", None)
    if max_files:
        rng = random.Random(42)
        benign = [s for s in samples if s["label"] == 0]
        malware = [s for s in samples if s["label"] == 1]
        rng.shuffle(benign)
        rng.shuffle(malware)
        half = int(max_files) // 2
        samples = benign[:half] + malware[:half]
        samples.sort(key=lambda x: x["sha"])

    return samples


# ── Teacher cache / manifest ───────────────────────────────────────────

def cache_path_for(cache_dir, sha: str) -> Path:
    return Path(cache_dir) / f"{sha}.pt"


def manifest_path(cache_dir) -> Path:
    return Path(cache_dir) / "manifest.csv"


def read_manifest(cache_dir) -> list[dict]:
    path = manifest_path(cache_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest inexistent: {path} — ruleaza intai precompute_teacher.py"
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


# ── Sampler cu resume (copie din lilm_mal_dataset_v2) ─────────────────

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


# ── Dataset la nivel de segment ────────────────────────────────────────

class SegmentDataset(Dataset):
    """Un item = un segment de cod + tintele teacher-ului (h_t, z_t) din cache.
    Tokenii segmentelor sunt salvati in cache de precompute_teacher, deci aici
    nu se tokenizeaza nimic."""

    def __init__(self, items: list[tuple], cache_dir):
        # items: (sha, label, seg_idx)
        self.items = items
        self.cache_dir = Path(cache_dir)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        sha, label, seg_idx = self.items[idx]
        d = torch.load(
            cache_path_for(self.cache_dir, sha),
            map_location="cpu",
            weights_only=True,
        )
        return {
            "code_ids": d["segs"][seg_idx].long(),
            "h_t": d["h"][seg_idx],
            "z_t": d["z"][seg_idx],
            "label": torch.tensor(label, dtype=torch.long),
        }


def segment_collate_fn(batch: list[dict]) -> dict:
    # left-pad la maximul din batch, consistent cu conventia teacher-ului;
    # pad id-ul e irelevant (masca il exclude din atentie)
    max_len = max(x["code_ids"].size(0) for x in batch)

    code_ids = []
    code_masks = []
    for x in batch:
        ids = x["code_ids"]
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
        "h_t": torch.stack([x["h_t"] for x in batch]),
        "z_t": torch.stack([x["z_t"] for x in batch]).float(),
        "labels": torch.stack([x["label"] for x in batch]),
    }


class SegmentDataLoader(DataLoader):
    def __init__(self, dataset: SegmentDataset, config, sampler=None, shuffle=True, drop_last=True):
        super().__init__(
            dataset,
            batch_size=config.batch_size,
            shuffle=(shuffle and sampler is None),
            num_workers=config.num_workers,
            sampler=sampler,
            collate_fn=segment_collate_fn,
            pin_memory=True,
            drop_last=drop_last,
            persistent_workers=True if config.num_workers > 0 else False,
            prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        )


def build_loaders(config) -> tuple[DataLoader, DataLoader]:
    rows = read_manifest(config.teacher_cache_dir)

    # split train/val LA NIVEL DE FISIER (nu segment), ca sa nu existe leakage
    rng = random.Random(42)
    shas = [r["sha"] for r in rows]
    rng.shuffle(shas)
    n_val = max(1, int(len(shas) * float(config.val_fraction)))
    val_shas = set(shas[:n_val])

    train_items, val_items = [], []
    for r in rows:
        target = val_items if r["sha"] in val_shas else train_items
        for seg_idx in range(r["n_seg"]):
            target.append((r["sha"], r["label"], seg_idx))

    val_max = int(getattr(config, "val_max_segments", 0) or 0)
    if val_max and len(val_items) > val_max:
        rng.shuffle(val_items)
        val_items = val_items[:val_max]

    train_ds = SegmentDataset(train_items, config.teacher_cache_dir)
    val_ds = SegmentDataset(val_items, config.teacher_cache_dir)

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

    train_loader = SegmentDataLoader(train_ds, config, sampler=train_sampler)
    val_loader = SegmentDataLoader(val_ds, config, sampler=val_sampler, shuffle=False, drop_last=False)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(
            f"[info] [dataset] fisiere: {len(rows)} | segmente train: {len(train_items)} "
            f"| segmente val: {len(val_items)}"
        )

    return train_loader, val_loader
