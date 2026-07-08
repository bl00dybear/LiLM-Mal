# Compressor 128k → 16k: entități și distilare

Model de bază: Qwen2.5-Coder-1.5B (`hidden = 1536`, 28 straturi).
Limită GPU: 16.384 tokeni per forward.

## Entități

```
┌─────────────────────────────────────────────────────────────────────────┐
│ TEACHER (D pe cod brut)                                        ÎNGHEȚAT │
│                                                                          │
│   clasificatorul antrenat în runul de 5 zile:                            │
│   Qwen 1.5B + LoRA + attention pooling + regression head                 │
│   input:  tokeni reali (max 16k)                                         │
│   output: h_t [1536], z_t [1]                                            │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ ENCODER / COMPRESSOR (E)                                       TRAINABIL │
│                                                                          │
│   același Qwen de bază + LoRA propriu (~76M)   ← se antrenează           │
│   + M: memory tokens [2048, 1536]              ← se antrenează           │
│   input:  segment de cod ≤ 14.336 tokeni + M  (total ≤ 16.384)           │
│   output: Z [2048, 1536]  (soft tokens, compresie ~7×)                   │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ STUDENT = E → D(comprimat)                                               │
│                                                                          │
│   lanțul întreg: E comprimă, D (înghețat) clasifică din Z                │
│   partea trainabilă a studentului este doar E                            │
│   output: h_s [1536], z_s [1]                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

## Encoderul, la nivel de tokeni

```
poziție:    0                         14335   14336              16383
            ├──────────── cod ──────────────┤ ├──── memory ──────────┤
input:      [ c₀   c₁   c₂  ...      c₁₄₃₃₅ ] [ m₀   m₁  ...   m₂₀₄₇ ]

mască cauzală:
  c₅    vede: c₀..c₅                     (codul NU vede memory)
  m₀    vede: c₀..c₁₄₃₃₅                 (tot codul)
  m₂₀₄₇ vede: c₀..c₁₄₃₃₅, m₀..m₂₀₄₆

după 28 straturi:  Z = H[:, -2048:, :]   [1, 2048, 1536]
(fiecare vector din Z rezumă în medie ~7 tokeni de cod)
```

## Etapa B — distilare (segmente ≤ 16k, același conținut la ambii)

```
                   ┌──────────── TEACHER (no_grad) ─────────────────┐
                   │                                                 │
code_ids [14336] ──┼─► [prefix + cod brut + suffix] ─► D ─► h_t ─► z_t
       │           │         (~14.426 tokeni)     (înghețat)  │      │
       │           └──────────────────────────────────────────┼──────┤
       │                                                      │      │
       │           ┌──────────── STUDENT ────────────┐  L_repr│      │L_logit
       │           │                                 │  1-cos(h_s,h_t)│MSE(z_s,z_t)
       └──────────►│ E ─► Z [2048,1536]              │        │      │
                   │ (trainabil)  └► [prefix + Z +   │        │      │
                   │                  suffix] ─► D ──┼─► h_s ─┴─► z_s┘
                   │              (~2.138 tok) (înghețat)
                   └─────────────────────────────────┘

L_total = λ₁·MSE(z_s, z_t) + λ₂·(1 − cos(h_s, h_t))

gradient:  L → prin D (înghețat, doar conduce) → ∂L/∂Z → prin E → LoRA_E + M
```

## Etapa C — fine-tuning end-to-end (fișier întreg, până la 128k)

```
fișier 128k tokeni
   │  split în ≤ 8 segmente × 14.336
   ▼
[seg₁] [seg₂] [seg₃] ... [seg₈]
   │      │      │          │        8 forward-uri encoder, SEPARATE
   ▼      ▼      ▼          ▼        (fiecare ≤ 16k, checkpointing)
  Z₁     Z₂     Z₃   ...   Z₈        fiecare [2048, 1536]
   └──────┴──────┴─────────┘
              │ concat
              ▼
   [prefix ; Z₁..Z₈ ; suffix]        ~16.4k soft tokens → UN singur forward D
              │
              ▼
        D (LoRA nou, mic)  ─►  logit  ─►  BCE cu labelul real
```

- Teacher-ul nu mai apare (nu poate vedea 128k).
- D nu vede niciodată 128k tokeni bruți; vede 16k soft tokens care acoperă tot fișierul.
- Trainabil: E (LoRA + M), LoRA mic pe D, capetele (attention_net, regression_head).

## Rezumat etape

| Etapă | Input | Loss | Trainabil | Înghețat |
|---|---|---|---|---|
| A. Autoencodare | segmente ≤16k | CE reconstrucție | LoRA_E + M | D |
| B. Distilare | segmente ≤16k | MSE logit + cos pooled | LoRA_E + M | D (teacher & student) |
| C. Task e2e | fișiere ≤128k | BCE cu label | E + LoRA_D mic + capete | baza D |
