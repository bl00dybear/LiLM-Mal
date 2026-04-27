import json
from pathlib import Path
import random
from transformers import AutoTokenizer
from tqdm import tqdm
import multiprocessing

def process_file(file_path):
    try:
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        code = data.get("decompiled_code") or "// decompilation unavailable"
        return code
    except Exception:
        return ""

def main():
    base_path = Path("/media/sebi/nvme-1tb/LiLM-Mal-Dataset/decompiled/train")
    model_path = "/media/sebi/nvme-1tb/LiLM-Mal/models/qwen2.5-coder-7b-instruct"
    
    print("[1/4] Încărcare tokenizer Qwen2.5...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("\n[2/4] Identificare fișiere dataset...")
    benign_files = list((base_path / "benign").glob("*.json"))
    malware_files = list((base_path / "malware").glob("*.json"))
    all_files = benign_files + malware_files
    print(f"Total fișiere găsite: {len(all_files)} ({len(benign_files)} Benign, {len(malware_files)} Malware)")

    sample_size = min(2000, len(all_files))
    sampled_files = random.sample(all_files, sample_size)
    print(f"\n[3/4] Eșantionare: Analizăm aleatoriu {sample_size} de fișiere pentru o acuratețe statistică rapidă.")

    prompt_overhead = "<|im_start|>system\nYou are a reverse-engineering assistant.<|im_end|>\n<|im_start|>user\nAnalyze this code:\n<|im_end|>\n<|im_start|>assistant\n"
    overhead_tokens = tokenizer(prompt_overhead, return_tensors="pt")["input_ids"].shape[1]
    kode_budget = 512 - overhead_tokens - 10

    lengths = []
    truncated_count = 0

    print("\n[4/4] Tokenizare și analiză...")
    for f in tqdm(sampled_files, desc="Calcul tokeni", unit="file"):
        code_str = process_file(f)
        if not code_str:
            continue
            
        tokens = tokenizer(code_str, add_special_tokens=False)["input_ids"]
        l = len(tokens)
        lengths.append(l)
        
        if l > kode_budget:
            truncated_count += 1

    if not lengths:
        print("Nu s-au putut procesa fișierele corespunzator.")
        return

    lengths.sort()
    avg_len = sum(lengths) / len(lengths)
    median_len = lengths[len(lengths)//2]
    max_len = lengths[-1]
    p90_len = lengths[int(len(lengths)*0.90)]

    print("\n" + "="*50)
    print("📊 REZULTATE STATISTICE (Eșantion Qwen2.5-Coder)")
    print("="*50)
    print(f"Buget disponibil pentru COD din cei 512 tokeni: ~{kode_budget} tokeni.")
    print("-" * 50)
    print(f"Media (Average) lungime cod:        {avg_len:.1f} tokeni")
    print(f"Mediana (Middle) lungime cod:       {median_len} tokeni")
    print(f"Percentila 90 (90% din fișiere):   sub {p90_len} tokeni")
    print(f"Maximum absolut găsit în eșantion:  {max_len} tokeni")
    print("-" * 50)
    print(f"⚠️ Fișiere tăiate brutal la 512 limite: {truncated_count} din {len(lengths)}")
    print(f"🌍 Procent de pierdere a informației:  {(truncated_count/len(lengths))*100:.2f} %")
    print("="*50)

if __name__ == "__main__":
    main()
