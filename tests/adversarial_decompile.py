import os
import shutil
import tempfile
import subprocess
import sys
from pathlib import Path

GHIDRA_HEADLESS = "/media/sebi/nvme-1tb/LiLM-Mal-Dataset/ghidra_11.3.1_PUBLIC/support/analyzeHeadless"
GHIDRA_SCRIPT = "ExtractDecompiledFunctions"
GHIDRA_SCRIPTS_DIR = "/media/sebi/nvme-1tb/LiLM-Mal-Dataset/ghidra_11.3.1_PUBLIC/Ghidra/Features/Decompiler/ghidra_scripts"

INPUT_DIR = Path("tests/scripts")
OUTPUT_DIR = Path("tests/inference")
TIMEOUT_SECONDS = 300

def compile_c_to_stripped_elf(c_path: Path):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    elf_path = OUTPUT_DIR / f"{c_path.stem}.elf"
    
    # Adaugam -fPIC si -shared pentru a permite compilarea fara main()
    # -ldl este necesar pentru dlsym
    cmd = ["gcc", "-fPIC", "-shared", str(c_path), "-o", str(elf_path), "-O0", "-ldl"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] GCC FAILED for {c_path.name}:\n{result.stderr}", file=sys.stderr)
        return None
        
    return elf_path

def run_ghidra_adversarial(elf_path: Path):
    sha256 = elf_path.stem
    out_json = OUTPUT_DIR / f"{sha256}.json"
    
    tmp_project = tempfile.mkdtemp(prefix="ghidra_adversarial_")

    try:
        cmd = [
            GHIDRA_HEADLESS,
            tmp_project,           
            "AdversarialProject",  
            "-import", str(elf_path), 
            "-scriptPath", GHIDRA_SCRIPTS_DIR,
            "-deleteProject",      
            "-postScript", GHIDRA_SCRIPT,
            str(out_json),         
            sha256,                
            "0",                   
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS + 30
        )

        if result.returncode != 0:
            print(f"[ERROR] GHIDRA FAILED for {elf_path.name}:\n{result.stderr}", file=sys.stderr)
        elif not out_json.exists():
            print(f"[ERROR] GHIDRA SUCCESS BUT NO JSON CREATED for {elf_path.name}", file=sys.stderr)

    except subprocess.TimeoutExpired:
        print(f"[ERROR] GHIDRA TIMEOUT for {elf_path.name}", file=sys.stderr)
    finally:
        shutil.rmtree(tmp_project, ignore_errors=True)

def main():
    if not INPUT_DIR.exists():
        print(f"[ERROR] Input directory {INPUT_DIR} not found", file=sys.stderr)
        return
        
    c_files = list(INPUT_DIR.glob("*.c"))
    if not c_files:
        print(f"[ERROR] No .c files found in {INPUT_DIR}", file=sys.stderr)
        return
        
    for c_file in c_files:
        elf_path = compile_c_to_stripped_elf(c_file)
        if elf_path and elf_path.exists():
            run_ghidra_adversarial(elf_path)

if __name__ == "__main__":
    main()