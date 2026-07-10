import csv
import os
import torch
import torch.multiprocessing as mp

from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer

import hydra
from omegaconf import DictConfig

from segment_dataset import (
    build_prompt_ids,
    cache_path_for,
    index_corpus,
    load_code,
    manifest_path,
    seg_code_budget,
    tokenize_and_segment,
)
from compressor_model import FrozenDecoder


def _rank_manifest_path(cache_dir, rank: int) -> Path:
    return Path(cache_dir) / f"manifest_rank{rank}.csv"


def _teacher_batch(segments, prefix_ids, suffix_ids, pad_id, device):

    seqs = [prefix_ids + seg + suffix_ids for seg in segments]
    max_len = max(len(s) for s in seqs)

    input_ids = []
    attention_mask = []
    for s in seqs:
        pad_n = max_len - len(s)
        input_ids.append([pad_id] * pad_n + s)
        attention_mask.append([0] * pad_n + [1] * len(s))

    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(attention_mask, dtype=torch.long, device=device),
    )


def worker(rank, config):
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        trust_remote_code=True,
        use_fast=True,
    )
    prefix_ids, suffix_ids = build_prompt_ids(tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    budget = seg_code_budget(config)
    assert len(prefix_ids) + budget + len(suffix_ids) <= config.max_token_len, (
        "prompt + segment nu incape in max_token_len"
    )
    max_segments = int(config.max_segments_per_file)
    batch_size = int(config.teacher_batch_size)

    decoder = FrozenDecoder(config).to(device).eval()
    print(f"[info] [rank {rank}] [teacher] loaded")

    cache_dir = Path(config.teacher_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    files = index_corpus(config)[rank::int(config.world_size)]

    manifest_rows = []
    skipped_existing = 0
    skipped_empty = 0

    for sample in tqdm(files, desc=f"rank {rank}", position=rank):
        sha = sample["sha"]
        out_path = cache_path_for(cache_dir, sha)

        if out_path.exists():
            try:
                cached = torch.load(out_path, map_location="cpu", weights_only=True)
                manifest_rows.append((sha, sample["label"], len(cached["segs"])))
                skipped_existing += 1
                continue
            except Exception:
                pass

        code = load_code(sample["path"])
        segments = tokenize_and_segment(code, tokenizer, budget, max_segments)
        if not segments:
            skipped_empty += 1
            continue

        h_chunks = []
        z_chunks = []
        with torch.no_grad():
            for start in range(0, len(segments), batch_size):
                batch_segs = segments[start:start + batch_size]
                input_ids, attention_mask = _teacher_batch(
                    batch_segs, prefix_ids, suffix_ids, pad_id, device
                )
                hidden = decoder.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                ).last_hidden_state
                h = hidden[:, -1, :]
                z = decoder.regression_head(h).squeeze(-1)

                h_chunks.append(h.cpu())
                z_chunks.append(z.float().cpu())

        torch.save(
            {
                "segs": [torch.tensor(s, dtype=torch.int32) for s in segments],
                "h": torch.cat(h_chunks, dim=0),
                "z": torch.cat(z_chunks, dim=0),
                "label": int(sample["label"]),
            },
            out_path,
        )
        manifest_rows.append((sha, sample["label"], len(segments)))

    with open(_rank_manifest_path(cache_dir, rank), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(manifest_rows)

    print(
        f"[info] [rank {rank}] gata: {len(manifest_rows)} fisiere "
        f"({skipped_existing} din cache, {skipped_empty} goale sarite)"
    )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())

    config = cfg.model

    mp.spawn(
        worker,
        args=(config,),
        nprocs=int(config.world_size),
        join=True,
    )

    cache_dir = Path(config.teacher_cache_dir)
    rows = []
    for rank in range(int(config.world_size)):
        rank_path = _rank_manifest_path(cache_dir, rank)
        if not rank_path.exists():
            continue
        with open(rank_path, "r", newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if row:
                    rows.append(row)
        rank_path.unlink()

    rows.sort(key=lambda r: r[0])
    with open(manifest_path(cache_dir), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    print(f"[info] manifest scris: {manifest_path(cache_dir)} ({len(rows)} fisiere)")


if __name__ == "__main__":
    main()
