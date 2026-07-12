#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$ROOT/qwen-lora-2.0/outputs"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  source "$ROOT/.env"
  set +a
fi

MODEL_FILE="$OUT/qwen_lora_76M_16k.pt"
CACHE_DIR="$OUT/version_2/teacher_cache"

HF_USER="${HF_USER:-bl00dybear}"
REPO="${REPO:-$HF_USER/qwen-decompmal-1.5b-16k}"

TAR_PATH="${TAR_PATH:-$OUT/teacher_cache_16k.tar}"
KEEP_TAR="${KEEP_TAR:-0}"
MODE="${1:-all}"

PYBIN="${PYBIN:-$ROOT/.venv/bin/python}"

hf_cli() {
  if [[ -x "$PYBIN" && -f "$ROOT/.venv/bin/hf" ]]; then
    "$PYBIN" "$ROOT/.venv/bin/hf" "$@"
  elif command -v hf >/dev/null 2>&1; then
    hf "$@"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli "$@"
  else
    echo "[error] no hf cli found; run: uv add huggingface_hub" >&2
    return 1
  fi
}

echo "[info] repo: $REPO"
echo "[info] mode: $MODE"

if ! hf_cli auth whoami >/dev/null 2>&1; then
  echo "[error] not authenticated; run: hf auth login  (or export HF_TOKEN=...)"
  exit 1
fi

push_model() {
  if [[ ! -f "$MODEL_FILE" ]]; then
    echo "[error] model file missing: $MODEL_FILE"
    exit 1
  fi
  echo "[info] uploading model $(basename "$MODEL_FILE") ($(du -h "$MODEL_FILE" | cut -f1))"
  hf_cli upload "$REPO" "$MODEL_FILE" "$(basename "$MODEL_FILE")" --repo-type model --private
  echo "[ok] model pushed to $REPO"
}

push_cache() {
  if [[ ! -d "$CACHE_DIR" ]]; then
    echo "[error] cache dir missing: $CACHE_DIR"
    exit 1
  fi
  local n_pt
  n_pt="$(find "$CACHE_DIR" -maxdepth 1 -name '*.pt' | wc -l)"
  if [[ "$n_pt" -eq 0 ]]; then
    echo "[error] no .pt shard files in $CACHE_DIR; nothing to push"
    exit 1
  fi
  if [[ -f "$CACHE_DIR/manifest.csv" ]]; then
    echo "[info] complete cache: $n_pt shard files + manifest.csv"
  else
    echo "[info] partial cache (resume snapshot): $n_pt shard files, no manifest yet"
  fi
  echo "[info] packing teacher cache into $TAR_PATH"
  tar -cf "$TAR_PATH" -C "$(dirname "$CACHE_DIR")" "$(basename "$CACHE_DIR")"
  echo "[info] tar size: $(du -h "$TAR_PATH" | cut -f1)"
  echo "[info] uploading tar to $REPO"
  hf_cli upload "$REPO" "$TAR_PATH" "teacher_cache/$(basename "$TAR_PATH")" --repo-type model --private
  if [[ -f "$CACHE_DIR/manifest.csv" ]]; then
    hf_cli upload "$REPO" "$CACHE_DIR/manifest.csv" teacher_cache/manifest.csv --repo-type model --private
  fi
  if [[ "$KEEP_TAR" != "1" ]]; then
    rm -f "$TAR_PATH"
    echo "[info] removed local tar (set KEEP_TAR=1 to keep it)"
  fi
  echo "[ok] cache pushed to $REPO"
}

case "$MODE" in
  model) push_model ;;
  cache) push_cache ;;
  all)   push_model; push_cache ;;
  *)     echo "[error] unknown mode '$MODE' (use: model | cache | all)"; exit 1 ;;
esac

echo "[done]"
