import os
import json
import logging
import torch
from pathlib import Path
from dataclasses import dataclass
from transformers import AutoTokenizer

from model import MalwareDetectionModel

@dataclass
class InferenceConfig:
    model_path: str = "/media/sebi/nvme-1tb/LiLM-Mal/models/qwen2.5-coder-3b-instruct"
    checkpoint_path: str = "/media/sebi/nvme-1tb/LiLM-Mal/outputs/checkpoints-q3b/qwen_malware_ep0_step500.pt"
    inference_dir: str = "/media/sebi/nvme-1tb/LiLM-Mal/tests/inference"
    max_token_len: int = 4096
    num_chunks: int = 2
    threshold: float = 0.5
    gradient_checkpointing: bool = False
    n_unfrozen_layers: int = 6

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("AdversarialInference")

def main():
    config = InferenceConfig()
    logger = setup_logger()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Rulare pe device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, local_files_only=True)

    logger.info("Instantiere MalwareDetectionModel...")
    model = MalwareDetectionModel(config).to(device)

    logger.info(f"Incarcare greutati din {config.checkpoint_path}...")
    ckpt = torch.load(config.checkpoint_path, map_location="cpu", mmap=True, weights_only=True)

    clean_state_dict = {}
    for k, v in ckpt.items():
        clean_key = k.replace("_orig_mod.", "").replace("module.", "").replace("_fsdp_wrapped_module.", "")
        clean_state_dict[clean_key] = v

    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()
    logger.info("Model incarcat cu succes.")

    empty_prompt = "<|im_start|>system\nYou are a binary analysis expert specializing in ELF malware detection.\n<|im_end|>\n<|im_start|>user\nAnalyze the following decompiled ELF binary code and classify it.\nAnswer with exactly one word: malware or benign.\n\n<code>\n\n</code>\n<|im_end|>\n<|im_start|>assistant\n"
    prompt_overhead = tokenizer(empty_prompt, return_tensors="pt")["input_ids"].shape[1]
    budget = max(config.max_token_len - prompt_overhead - 5, 100)

    inference_dir = Path(config.inference_dir)
    json_files = sorted(inference_dir.glob("*.json"))

    if not json_files:
        logger.warning(f"Nu s-au gasit fisiere in {config.inference_dir}")
        return

    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            code = data.get("decompiled_code") or ""
            if len(code) < 10:
                continue

            max_ids = budget * config.num_chunks
            ids = tokenizer(code, add_special_tokens=False, truncation=True, max_length=max_ids, return_tensors="pt")["input_ids"][0]
            
            chunks = []
            for i in range(config.num_chunks):
                start = i * budget
                end = start + budget
                chunk_ids = ids[start:end]
                if len(chunk_ids) > 0:
                    chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
                else:
                    chunks.append("")

            system_prompt = (
                "You are an elite Reverse Engineer and Malware Analyst specializing in Linux ELF forensics. "
                "Your objective is to detect malicious intent by identifying patterns of: "
                "Function Hooking (LD_PRELOAD), Anti-Debugging (ptrace/env), C2 Beaconing (DNS/HTTP), "
                "and stealthy Process Injection or Persistence."
            )
            system_prompt = (
                "You are a Senior Security Researcher specialized in Linux Kernel and Userland malware. "
                "Your task is to perform Deep Packet and Binary Inspection on decompiled C code."
            )

            # system_prompt = "Expert ELF Malware Analyst. Focus: Hooking, C2 Beaconing, Evasion."

            prompts = []
            for i, c in enumerate(chunks):
                # context_label = f"Fragment {i+1} of {config.num_chunks}"
                # user_content = (
                #     f"### [CONTEXT: {context_label}]\n"
                #     "Analyze the following decompiled C code segment for Indicators of Compromise (IoCs). "
                #     "Focus on syscall sequences, data obfuscation (XOR/Base64), and evasive control flows. "
                #     "Classify as 'malware' if any malicious capability is detected, otherwise 'benign'.\n\n"
                #     f"<code>\n{c}\n</code>"
                # )
                user_content = (
                    "### INSTRUCTION:\n"
                    "Evaluate the security posture of the following code. Identify if the logic deviates "
                    "from standard library behavior towards offensive operations.\n\n"
                    "### CRITICAL MALICIOUS PATTERNS TO DETECT:\n"
                    "1. **Execution Hijacking**: Using dlsym/RTLD_NEXT to wrap standard syscalls (open, read, write).\n"
                    "2. **Stealth Communication**: Raw socket construction or manual DNS packet assembly for C2 exfiltration.\n"
                    "3. **Environment Evasion**: Conditional execution based on debugger presence (ptrace) or environment variables.\n"
                    "4. **Information Hiding**: Manipulation of directory entries (readdir) to conceal files or processes.\n\n"
                    "### TASK:\n"
                    "Is this code fragment part of a malicious toolkit? "
                    "Respond with exactly one word: 'malware' or 'benign'.\n\n"
                    "<code>\n"
                )

                # user_content=(
                #     "Task: Classify this decompiled C code.\n"
                #     "Check for: LD_PRELOAD hijacks, raw DNS/Socket C2, and ptrace evasion.\n"
                #     "Verdict: malware (if offensive) or benign (if standard).\n\n"
                #     "<code>\n"
                # )
                
                prompt = (
                    f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
                    f"<|im_start|>user\n{user_content}\n<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                prompts.append(prompt)

            tokenizer.padding_side = "left"
            encoded = tokenizer(prompts, max_length=config.max_token_len, truncation=True, padding="max_length", return_tensors="pt")

            input_ids = encoded["input_ids"].unsqueeze(0).to(device)
            attention_mask = encoded["attention_mask"].unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                prob = torch.sigmoid(logits).squeeze().item()

            verdict = "malware" if prob > config.threshold else "benign"
            logger.info(f"[{jf.name}] Score: {prob * 100:.2f}% | Verdict: {verdict}")

        except Exception as e:
            logger.error(f"Eroare la procesarea {jf.name}: {e}")

if __name__ == "__main__":
    main()