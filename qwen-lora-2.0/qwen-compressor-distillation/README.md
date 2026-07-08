# qwen-compressor-distillation

Etapele A (autoencodare) + B (distilare) pentru compressorul 128k → 16k.
Schema completa a entitatilor: `doc/compressor-distillation.md`.

## Entitati

- **Teacher**: clasificatorul antrenat de `qwen-lora-classic-ddp` — NU sta in
  VRAM la training; tintele lui (`h_t`, `z_t` per segment) sunt precalculate
  pe disk de `precompute_teacher.py`.
- **Encoder (trainabil)**: Qwen 1.5B inghetat + LoRA propriu + `memory`
  (`[k, 1536]`) + `ae_token`. Output: `Z [B, k, 1536]`.
- **Decoder (inghetat)**: Qwen 1.5B cu LoRA-ul teacher-ului **merge-uit in
  greutati** (`W' = W + B@A·scale`) + `regression_head`. Conduce gradientul
  spre `Z`, nu invata nimic.

## Rulare

```bash
# 1. o singura data (reluabil — fisierele deja cache-uite se sar)
uv run qwen-compressor-distillation/precompute_teacher.py model=compressor1.5b

# 2. antrenarea A+B
uv run qwen-compressor-distillation/main.py model=compressor1.5b
# sau totul inlantuit:
./run_compressor_distill.sh
```

Ambii pasi au nevoie de ambele GPU-uri — nu porni cat timp ruleaza
clasificatorul.

## Loss

```
L = lambda_rec · CE(reconstructie primilor recon_tokens din [Z; ae])
  + lambda_logit · MSE(z_s, z_t)
  + lambda_repr  · (1 − cos(h_s, h_t))
```

`lambda_rec=0` → doar etapa B; `lambda_logit=lambda_repr=0` → doar etapa A.

## Bugetul de memorie (2 × A2000 12GB)

- Doua backbone-uri (encoder + decoder) sharduite FSDP2 pe 2 GPU-uri
  (~1.55GB fiecare per GPU); LoRA encoder replicat.
- Per-layer gradient checkpointing pe ambele backbone-uri.
- CE de reconstructie pe felii de `ce_chunk_size` sub checkpoint — logits
  `[L, 152k]` nu se materializeaza niciodata intregi.
- Checkpoint-urile salveaza DOAR trainabilele (~160MB), baza se reconstruieste
  din `model_id` + checkpointul teacher.

Daca da OOM, in ordinea eficientei: scade `recon_tokens` (4096 → 2048),
scade `num_memory_tokens` + creste proportional segmentele, `ce_chunk_size=512`.

## Invarianti (nu le strica)

- Prompturile din `segment_dataset.py` trebuie sa ramana identice cu
  `lilm_mal_dataset_v2.py` — teacher-ul a fost antrenat cu ele.
- `tokenize_and_segment` e folosita si de precompute si de training; orice
  schimbare de geometrie (`max_token_len`, `num_memory_tokens`,
  `max_segments_per_file`) invalideaza cache-ul → sterge `teacher_cache/`
  si reia pasul 1.
- Split-ul train/val se face pe fisiere (seed 42), nu pe segmente.

## Sanity checks la primul run

1. `memory.grad` si LoRA-urile encoderului non-nule dupa primul backward;
   toate gradientii decoderului `None`.
2. `train/cos` porneste aproape de 0 si urca — daca porneste ~1, compari
   gresit tensorii.
3. `val_agree` (acordul de semn student vs teacher) e metrica principala a
   etapei B; `val_acc` (vs label) e doar orientativa.
