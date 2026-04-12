from huggingface_hub import snapshot_download
import os
 
MODEL_ID   = "Qwen/Qwen2.5-Coder-3B-Instruct"
LOCAL_DIR  = "/media/sebi/nvme-1tb/LiLM-Mal/models/qwen2.5-coder-3b-instruct"
 
os.makedirs(LOCAL_DIR, exist_ok=True)
 
print(f"Download: {MODEL_ID}")
print(f"Destinatie: {LOCAL_DIR}")
print()
 
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=LOCAL_DIR,
)
 
print()
print("Done! Model salvat in:", LOCAL_DIR)