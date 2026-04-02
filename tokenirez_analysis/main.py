import json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

_tokenizer = None
_budget = None

def init_worker(model_path, budget):
    global _tokenizer, _budget
    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    _budget = budget

def process_file_optimized(file_path):
    try:
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        code = data.get("decompiled_code") or ""
        if not code:
            return None
        
        tokens = _tokenizer(code, add_special_tokens=False)["input_ids"]
        length = len(tokens)
        return (length, 1 if length > _budget else 0)
    except:
        return None

def main():
    base_path = Path("/media/sebi/nvme-1tb/LiLM-Mal-Dataset/decompiled/train")
    model_path = "/media/sebi/nvme-1tb/LiLM-Mal/models/qwen2.5-coder-7b-instruct"
    num_workers = cpu_count()
    
    temp_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt_overhead = "<|im_start|>system\nYou are a reverse-engineering assistant.<|im_end|>\n<|im_start|>user\nAnalyze this code:\n<|im_end|>\n<|im_start|>assistant\n"
    overhead_tokens = temp_tokenizer(prompt_overhead, return_tensors="pt")["input_ids"].shape[1]
    kode_budget = 4096 - overhead_tokens - 10
    del temp_tokenizer

    files = list((base_path / "benign").glob("*.json")) + list((base_path / "malware").glob("*.json"))
    total_files = len(files)
    
    lengths = []
    truncated_count = 0
    
    with Pool(processes=num_workers, initializer=init_worker, initargs=(model_path, kode_budget)) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_file_optimized, files, chunksize=200), 
            total=total_files,
            desc="Analiza dataset"
        ))

    for res in results:
        if res:
            lengths.append(res[0])
            truncated_count += res[1]

    if not lengths:
        return

    lengths.sort()
    n = len(lengths)
    avg_len = sum(lengths) / n
    median_len = lengths[n // 2]
    p90_len = lengths[int(n * 0.90)]
    max_len = max(lengths)

    print("\n" + "="*50)
    print("STATISTICI RESURSE MAXIME")
    print("="*50)
    print(f"Nuclee utilizate: {num_workers}")
    print(f"Total fisiere: {n}")
    print("-" * 50)
    print(f"Media: {avg_len:.1f}")
    print(f"Mediana: {median_len}")
    print(f"P90: {p90_len}")
    print(f"Max: {max_len}")
    print("-" * 50)
    print(f"Fisiere truncate la {kode_budget}: {truncated_count}")
    print(f"Procent pierdere: {(truncated_count/n)*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()


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