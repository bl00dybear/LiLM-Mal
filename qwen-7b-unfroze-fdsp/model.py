import os, sys, json, torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "qwen2.5-coder-7b-instruct"))
CHECKPOINT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "checkpoints", "qwen_malware_latest_step.pt"))
INFERENCE_DIR   = Path(os.path.dirname(__file__)) / "inference"

NUM_LABELS    = 2
MAX_TOKEN_LEN = 512
NUM_CHUNKS    = 4
HIDDEN_SIZE   = 3584


def setup_gpus():
    if not torch.cuda.is_available():
        print("    [WARN] CUDA indisponibil — fallback CPU (LENT)")
        return "cpu", None

    n = torch.cuda.device_count()
    print(f"    GPU-uri detectate: {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        vram  = props.total_memory / 1024**3
        print(f"      cuda:{i} — {props.name} | {vram:.1f} GB VRAM")

    if n >= 2:
        print("    Mod: DUAL GPU (cuda:0 + cuda:1)")
        max_memory = {0: "9GiB", 1: "9GiB", "cpu": "8GiB"}
        return "cuda:0", max_memory
    else:
        print("    Mod: SINGLE GPU")
        return "cuda:0", {0: "10GiB", "cpu": "8GiB"}


class ScoreHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.score      = nn.Linear(hidden_size, num_labels, bias=True,  dtype=torch.bfloat16)
        self.chunk_attn = nn.Linear(hidden_size, 1,          bias=False, dtype=torch.bfloat16)

    def forward(self, chunked_pooled):
        attn_weights = torch.softmax(self.chunk_attn(chunked_pooled), dim=1)
        pooled = (chunked_pooled * attn_weights).sum(dim=1)
        return self.score(pooled)


def predict(base_model, head, tokenizer, code: str, primary_device: str) -> float:
    budget  = 476
    max_ids = budget * NUM_CHUNKS

    ids = tokenizer(
        code, add_special_tokens=False, truncation=True,
        max_length=max_ids, return_tensors="pt"
    )["input_ids"][0]

    chunks = []
    for i in range(NUM_CHUNKS):
        chunk = ids[i * budget : i * budget + budget]
        chunks.append(
            tokenizer.decode(chunk, skip_special_tokens=True) if len(chunk) > 0 else ""
        )

    system_prompt = "You are an expert reverse-engineering assistant capable of identifying malicious assembly or source code patterns."
    user_header   = "Analyze the following decompiled ELF binary code and classify it.\nAnswer with exactly one word: malware or benign.\n\n<code>\n"
    user_footer   = "\n</code>"

    prompts = [
        f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_header}{c}{user_footer}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
        for c in chunks
    ]

    tokenizer.padding_side = "left"
    encoded = tokenizer(
        prompts, max_length=MAX_TOKEN_LEN, truncation=True,
        padding="max_length", return_tensors="pt"
    )

    input_ids      = encoded["input_ids"].to(primary_device)
    attention_mask = encoded["attention_mask"].to(primary_device)

    with torch.no_grad():
        out = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = out.last_hidden_state.to(primary_device)
        pooled = hidden[:, -1, :]
        pooled = pooled.unsqueeze(0)

        logits = head(pooled)
        probs  = torch.softmax(logits.squeeze(), dim=-1)

    return probs[1].item()


def main():
    print("=" * 60)
    print(" LiLM-Mal Adversarial Inference — Dual GPU")
    print("=" * 60)

    print("\n[1] Detectare GPU-uri...")
    primary_device, max_memory = setup_gpus()
    is_cuda = primary_device.startswith("cuda")

    print(f"\n[2] Incarcare tokenizer din:\n    {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    print("\n[3] Incarcare Qwen2.5-7B cu BitsAndBytes 4-bit NF4...")
    if max_memory:
        print(f"    Distributie VRAM: {max_memory}")

    if is_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        base_model = AutoModel.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            local_files_only=True,
        )
    else:
        print("    (CPU fallback — va fi LENT)")
        base_model = AutoModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )

    base_model.eval()

    if is_cuda and hasattr(base_model, "hf_device_map"):
        device_map = base_model.hf_device_map
        gpu0 = sum(1 for d in device_map.values() if d == 0 or d == "cuda:0")
        gpu1 = sum(1 for d in device_map.values() if d == 1 or d == "cuda:1")
        cpu  = sum(1 for d in device_map.values() if d == "cpu")
        print(f"    Distributie layere: cuda:0={gpu0} | cuda:1={gpu1} | cpu={cpu}")

    print(f"\n[4] Incarcare head de clasificare din checkpoint:\n    {CHECKPOINT_PATH}")
    try:
        head = ScoreHead(HIDDEN_SIZE, NUM_LABELS).to(primary_device)
        ckpt = torch.load(
            CHECKPOINT_PATH, map_location="cpu", mmap=True, weights_only=True
        )

        head_weights = {
            k: v for k, v in ckpt.items()
            if k.startswith("score.") or k.startswith("chunk_attn.")
        }

        if not head_weights:
            print("    WARN: Nu am gasit chei 'score.*' sau 'chunk_attn.*' in checkpoint!")
            print("          Inferenta va folosi head random.")
        else:
            head.load_state_dict(head_weights)
            print(f"    -> {len(head_weights)} tensori incarcati cu succes.")

        del ckpt
    except Exception as e:
        print(f"    EROARE la checkpoint: {e}")
        return

    head.eval()

    json_files = sorted(INFERENCE_DIR.glob("*.json"))
    if not json_files:
        print(f"\n[!] Nu s-au gasit JSON-uri in {INFERENCE_DIR}")
        print("    Ruleaza mai intai: uv run tests/adversarial_decompile.py")
        return

    print(f"\n{'=' * 60}")
    print(f" INFERENTA PE {len(json_files)} FISIERE ADVERSARIALE")
    print(f"{'=' * 60}")

    results = []
    for jf in json_files:
        print(f"\n[{jf.name}]")
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            code = data.get("decompiled_code") or ""
            if len(code) < 10:
                print("  => Skip: fisier gol sau decompilare esuata.")
                continue

            score   = predict(base_model, head, tokenizer, code, primary_device)
            verdict = "MALWARE" if score > 0.5 else "BENIGN"

            print(f"  => Score malitios : {score * 100:.3f}%")
            print(f"  => VERDICT        : {verdict}")

            if score > 0.5:
                print("  /!\\ ALERTA SHORTCUT LEARNING: cod benign clasificat ca malware!")
            else:
                print("  (Modelul a identificat corect codul benign.)")

            results.append({"file": jf.name, "score": score, "verdict": verdict})

        except Exception as e:
            print(f"  => Eroare: {e}")
            import traceback; traceback.print_exc()

    if results:
        print(f"\n{'=' * 60}")
        print(" SUMAR FINAL")
        print(f"{'=' * 60}")
        malware = [r for r in results if r["score"] > 0.5]
        benign  = [r for r in results if r["score"] <= 0.5]
        print(f"  Total fisiere procesate : {len(results)}")
        print(f"  Clasificate MALWARE     : {len(malware)}")
        print(f"  Clasificate BENIGN      : {len(benign)}")
        if malware:
            print(f"\n  Fisiere cu alerta shortcut learning:")
            for r in malware:
                print(f"    - {r['file']}  ({r['score']*100:.2f}%)")

    print(f"\n{'=' * 60}")
    print(" Inferenta finalizata.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()