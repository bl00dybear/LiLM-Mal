import json
import os
import csv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
import torch
from transformers import AutoTokenizer
from transformers import logging as hf_logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

import faulthandler
import signal

faulthandler.register(signal.SIGUSR1)

_tokenizer = None
_budget = None

SPLITS_BASE = Path("/run/media/sebi/nvme-1tb/LiLM-Mal-Dataset/data/02_metadata/experiments")
CORPUS_BASE = Path("/run/media/sebi/nvme-1tb/LiLM-Mal-Dataset/data/03_corpus_v2")

def get_files_from_csv(experiment_name: str, partition: str, platform: str) -> list[str]:
    csv_path = SPLITS_BASE / experiment_name / f"{partition}.csv"
    files = []
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found.")
        return files
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            label = int(row[0])
            sha256 = row[1]
            label_dir = "malware" if label == 1 else "benign"
            json_path = CORPUS_BASE / platform / label_dir / f"{sha256}.json"
            if json_path.exists():
                files.append(str(json_path))
    return files

def init_worker(model_path, budget):
    global _tokenizer, _budget
    hf_logging.set_verbosity_error()
    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    _budget = budget

def process_file_optimized(file_path):
    try:
        raw_bytes = Path(file_path).read_bytes()
        if HAS_ORJSON:
            data = orjson.loads(raw_bytes)
        else:
            data = json.loads(raw_bytes.decode("utf-8"))
            
        code = data.get("decompiled_code") or ""
        if not code:
            return None
        
        tokens = _tokenizer(code, add_special_tokens=False)["input_ids"]
        length = len(tokens)
        cat = "malware" if "malware" in str(file_path) else "benign"
        return (length, 1 if length > _budget else 0, cat)
    except:
        return None

def main():
    model_path = "/run/media/sebi/nvme-1tb/LiLM-Mal/models/qwen2.5-coder-1.5b-instruct"
    num_workers = cpu_count()
    # num_workers = 5
    
    temp_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt_overhead = "<|im_start|>system\nYou are a reverse-engineering assistant.<|im_end|>\n<|im_start|>user\nAnalyze this code:\n<|im_end|>\n<|im_start|>assistant\n"
    overhead_tokens = temp_tokenizer(prompt_overhead, return_tensors="pt")["input_ids"].shape[1]
    kode_budget = 4096*2 - overhead_tokens - 10
    del temp_tokenizer

    files = get_files_from_csv("elf_v2_full", "train", "elf") + get_files_from_csv("elf_v2_full", "test", "elf")
    
    # corpus_elf = CORPUS_BASE / "elf"
    # files = list((corpus_elf / "benign").glob("*.json")) + list((corpus_elf / "malware").glob("*.json"))
    
    total_files = len(files)
    
    lengths = []
    truncated_count = 0
    
    with Pool(processes=num_workers, initializer=init_worker, initargs=(model_path, kode_budget)) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_file_optimized, files, chunksize=200), 
            total=total_files,
            desc="Dataset analysis"
        ))

    bucket_counts = {
        "0-100": {"malware": 0, "benign": 0},
        "100-2000": {"malware": 0, "benign": 0},
        "2000-8K": {"malware": 0, "benign": 0},
        "8K-16K": {"malware": 0, "benign": 0},
        "16K-32K": {"malware": 0, "benign": 0},
        "32K-64K": {"malware": 0, "benign": 0},
        "64K-128K": {"malware": 0, "benign": 0},
        "128K-inf": {"malware": 0, "benign": 0}
    }

    for res in results:
        if res:
            length, is_trunc, cat = res
            lengths.append(length)
            truncated_count += is_trunc

            if length <= 100:
                bucket_counts["0-100"][cat] += 1
            elif length <= 2000:
                bucket_counts["100-2000"][cat] += 1
            elif length <= 8192:
                bucket_counts["2000-8K"][cat] += 1
            elif length <= 16384:
                bucket_counts["8K-16K"][cat] += 1
            elif length <= 32768:
                bucket_counts["16K-32K"][cat] += 1
            elif length <= 65536:
                bucket_counts["32K-64K"][cat] += 1
            elif length <= 131072:
                bucket_counts["64K-128K"][cat] += 1
            else:
                bucket_counts["128K-inf"][cat] += 1

    if not lengths:
        return

    lengths.sort()
    n = len(lengths)
    avg_len = sum(lengths) / n
    median_len = lengths[n // 2]
    p90_len = lengths[int(n * 0.90)]
    max_len = max(lengths)

    print("\n" + "="*50)
    print("MAXIMUM RESOURCE STATISTICS")
    print("="*50)
    print(f"Cores used: {num_workers}")
    print(f"Total files: {n}")
    print("-" * 50)
    print(f"Mean: {avg_len:.1f}")
    print(f"Median: {median_len}")
    print(f"P90: {p90_len}")
    print(f"Max: {max_len}")
    print("-" * 50)
    print(f"Files truncated at {kode_budget}: {truncated_count}")
    print(f"Loss percentage: {(truncated_count/n)*100:.2f}%")
    print("="*50)
    
    print("\n" + "="*50)
    print("BUCKET & CATEGORY STATISTICS")
    print("="*50)
    for b, cats in bucket_counts.items():
        total_b = cats['malware'] + cats['benign']
        print(f"Bucket {b:<10} : {total_b:<6} files (Malware: {cats['malware']:<5} | Benign: {cats['benign']:<5})")
    print("="*50)

    stats_dict = {
        "cores_used": num_workers,
        "total_files": n,
        "mean": round(avg_len, 1),
        "median": median_len,
        "p90": p90_len,
        "max": max_len,
        "token_budget": kode_budget,
        "truncated_files": truncated_count,
        "loss_percentage": round((truncated_count/n)*100, 2),
        "bucket_statistics": bucket_counts
    }

    with open("tokenizer_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=4)

    plot_bucket_histogram(bucket_counts, "token_length_buckets.pdf")

def plot_bucket_histogram(bucket_counts, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    color_benign = "#2a78d6"
    color_malware = "#e34948"

    buckets = list(bucket_counts.keys())
    benign = [bucket_counts[b]["benign"] for b in buckets]
    malware = [bucket_counts[b]["malware"] for b in buckets]

    x = np.arange(len(buckets))
    width = 0.4

    fig, ax = plt.subplots(figsize=(11, 6))
    bars_b = ax.bar(x - width / 2, benign, width, label="Benign", color=color_benign)
    bars_m = ax.bar(x + width / 2, malware, width, label="Malware", color=color_malware)

    ax.set_yscale("log")
    ax.set_xlabel("Token length bucket")
    ax.set_ylabel("Number of files (log scale)")
    ax.set_title("ELF file token-length distribution by class")
    ax.set_xticks(x)
    ax.set_xticklabels(buckets, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    for bars in (bars_b, bars_m):
        for rect in bars:
            h = rect.get_height()
            if h > 0:
                ax.annotate(
                    str(int(h)),
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    main()


    
#Before:
# ==================================================
# STATISTICI RESURSE MAXIME
# ==================================================
# Nuclee utilizate: 80
# Total fisiere: 92642
# --------------------------------------------------
# Media: 2757.4
# Mediana: 977
# P90: 7811
# Max: 129466
# --------------------------------------------------
# Fisiere truncate la 4060: 17327
# Procent pierdere: 18.70%
# ==================================================

#After:

# ==================================================
# MAXIMUM RESOURCE STATISTICS
# ==================================================
# Cores used: 80
# Total files: 92659
# --------------------------------------------------
# Mean: 2500.4
# Median: 908
# P90: 7089
# Max: 291077
# --------------------------------------------------
# Files truncated at 8156: 7461
# Loss percentage: 8.05%
# ==================================================