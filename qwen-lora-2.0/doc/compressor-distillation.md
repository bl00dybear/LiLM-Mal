# Compressor 128k → 16k: Entities and Distillation

Base model: Qwen2.5-Coder-1.5B (`hidden = 1536`, 28 layers).
GPU limit: 16,384 tokens per forward.

## Entities

```
┌─────────────────────────────────────────────────────────────────────────┐
│ TEACHER (D on raw code)                                          FROZEN │
│                                                                          │
│   Classifier trained in the 5-day run:                                   │
│   Qwen 1.5B + LoRA + attention pooling + regression head                 │
│   input:  real tokens (max 16k)                                          │
│   output: h_t [1536], z_t [1]                                            │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ ENCODER / COMPRESSOR (E)                                      TRAINABLE │
│                                                                          │
│   same base Qwen + custom LoRA (~76M)           ← trained                │
│   + M: memory tokens [2048, 1536]               ← trained                │
│   input:  code segment ≤ 14,336 tokens + M (total ≤ 16,384)              │
│   output: Z [2048, 1536] (soft tokens, ~7x compression)                  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ STUDENT = E → D(compressed)                                             │
│                                                                          │
│   full chain: E compresses, D (frozen) classifies from Z                 │
│   trainable part of the student is only E                                │
│   output: h_s [1536], z_s [1]                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

## The Encoder at Token Level

```
position:   0                         14335   14336              16383
            ├──────────── code ──────────────┤ ├──── memory ──────────┤
input:      [ c₀   c₁   c₂  ...      c₁₄₃₃₅ ] [ m₀   m₁  ...   m₂₀₄₇ ]

causal mask:
  c₅    sees: c₀..c₅                     (code does NOT see memory)
  m₀    sees: c₀..c₁₄₃₃₅                 (all code)
  m₂0₄₇ sees: c₀..c₁₄₃₃₅, m₀..m₂₀₄₆

after 28 layers:  Z = H[:, -2048:, :]   [1, 2048, 1536]
(each vector in Z summarizes on average ~7 code tokens)
```

## Stage B — Distillation (segments ≤ 16k, same content for both)

```
                   ┌──────────── TEACHER (no_grad) ─────────────────┐
                   │                                                 │
code_ids [14336] ──┼─► [prefix + raw code + suffix] ─► D ─► h_t ─► z_t
       │           │         (~14,426 tokens)     (frozen)    │      │
       │           └──────────────────────────────────────────┼──────┤
       │                                                      │      │
       │           ┌──────────── STUDENT ────────────┐  L_repr│      │L_logit
       │           │                                 │  1-cos(h_s,h_t)│MSE(z_s,z_t)
       └──────────►│ E ─► Z [2048,1536]              │        │      │
                   │ (trainable)  └► [prefix + Z +   │        │      │
                   │                  suffix] ─► D ──┼─► h_s ─┴─► z_s┘
                   │              (~2,138 tok) (frozen)
                   └─────────────────────────────────┘

L_total = λ₁·MSE(z_s, z_t) + λ₂·(1 − cos(h_s, h_t))

gradient:  L → through D (frozen, only propagates) → ∂L/∂Z → through E → LoRA_E + M
```

### Detailed Student Diagram (Forward & Backward Layer Flow)

```
========================================================================================
                                     ENCODER (E)
                       (Trainable Part: LoRA_E + Memory M)
========================================================================================

    [ code_ids ] (≤14,336)                                  [ self.memory ] (Param: [2048, 1536])
          │                                                              │
          ▼                                                              ▼
   [ embed_tokens ] (Frozen)                                      [ expand batch ]
          │                                                              │
          ▼ [B, N, 1536]                                                 ▼ [B, M, 1536]
     code_emb                                                         memory_emb
          │                                                              │
          └──────────────────────────────┬───────────────────────────────┘
                                         ▼
                                  [ Concatenate ]  (dim=1)
                                         │
                                         ▼
                                inputs_embeds [B, N+M, 1536]             ◄─── [Grad: update M]
                                         │
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │   Qwen2 Backbone (28 Layers) + LoRA_E     │         ◄─── [Grad: update LoRA_E]
                   │                                           │
                   │   - Causal Attention Masking:             │
                   │     - Code tokens (0..N-1) cannot see M.  │
                   │     - Memory tokens (N..N+M-1) see all    │
                   │       preceding Code + Memory tokens.     │
                   └───────────────────────────────────────────┘
                                         │
                                         ▼
                               last_hidden_state [B, N+M, 1536]
                                         │
                                         ▼
                                  [ Slicing: -M: ]  (keeps only M states)
                                         │
                                         ▼
                               Z [B, M, 1536] (Soft Tokens)              ◄─── [Grad: dL/dZ]

========================================================================================
                                     DECODER (D)
                      (Frozen Part: Backbone D + Head)
========================================================================================
                                         │
             [prefix_emb] ───────────────┼─────────────────┐
                                         ▼                 │
                                  [ Concatenate ]          │             ◄─── [Grad: back to Z]
                                         ▲                 │
             [suffix_emb] ───────────────┼─────────────────┘
                                         │
                                         ▼
                               inputs_embeds [B, P+M+S, 1536]
                                         │
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │    Qwen2 Backbone (28 Layers) + LoRA      │         ◄─── [Grad: passes through]
                   │                  (FROZEN)                 │              (no parameters updated)
                   └───────────────────────────────────────────┘
                                         │
                                         ▼
                               last_hidden_state [B, P+M+S, 1536]
                                         │
                                         ▼
                                  [ Slicing: -1 ]   (last token of the suffix)
                                         │
                                         ▼
                                   h_s [B, 1536]                         ◄─── [Grad: dL/dh_s]
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
          [ regression_head ] (Frozen)                   [ L_repr Loss ]
                   │                                     1 - cos(h_s, h_t)
                   ▼
              z_s [B] (Logit)
                   │
                   ▼
             [ L_logit Loss ]
              MSE(z_s, z_t)
```

## Stage C — End-to-End Fine-Tuning (full file, up to 128k)

```
128k tokens file
   │  split into ≤ 8 segments × 14,336
   ▼
[seg₁] [seg₂] [seg₃] ... [seg₈]
   │      │      │          │        8 separate encoder forward passes
   ▼      ▼      ▼          ▼        (each ≤ 16k, checkpointed)
  Z₁     Z₂     Z₃   ...   Z₈        each [2048, 1536]
   └──────┴──────┴─────────┘
              │ concat
              ▼
   [prefix ; Z₁..Z₈ ; suffix]        ~16.4k soft tokens → A SINGLE decoder forward D
              │
              ▼
         D (new LoRA, small) ──►  logit  ──►  BCE with real label
```

- The teacher is no longer used (cannot see 128k).
- D never sees 128k raw tokens; it only sees 16k soft tokens that cover the whole file.
- Trainable parameters: E (LoRA + M), small LoRA on D, heads (attention_net, regression_head).

## Stages Summary

| Stage | Input | Loss | Trainable | Frozen |
|---|---|---|---|---|
| A. Autoencoding | segments ≤16k | CE Reconstruction | LoRA_E + M | D |
| B. Distillation | segments ≤16k | MSE logit + cos pooled | LoRA_E + M | D (teacher & student) |
| C. Task e2e | files ≤128k | BCE with label | E + small LoRA_D + heads | base D |
