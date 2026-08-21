# Mini VL-JEPA — Measured Results

Setup: MSR-VTT official splits (train 6,513 / val 497 / test 2,990 videos, Kaggle copy
with 118,552 train caption pairs). Frozen V-JEPA 2 ViT-L features (cached, fp16).
Target space: LaBSE. Predictor: 26.5M params, batch 48, lr 3e-4, 10 epochs,
loss = cosine + InfoNCE + variance. Hardware: single RTX 3050 6 GB (Windows).
Checkpoint used below: `predictor_e5` (best val R@1).

## 1. Video→Text retrieval (val, 497 videos, one caption per video)

| Epoch | R@1 | R@5 | R@10 | MedR |
|---|---|---|---|---|
| e2 | 17.71 | 43.66 | 58.35 | 7 |
| e3 | 16.90 | 44.47 | 57.95 | 7 |
| **e5** | **19.72** | **44.87** | **58.35** | **7** |
| e7 | 18.51 | 44.87 | 56.54 | 7 |
| e9 | 18.11 | 44.27 | 56.74 | 7 |

Converges by ~epoch 5. Random chance R@1 = 0.2%.

## 2. Zero-shot cross-lingual retrieval (trained on English ONLY)

Captions machine-translated with NLLB-200-distilled-600M; identical video embeddings.

| Language | Script | R@1 | R@5 | R@10 | MedR | Retention |
|---|---|---|---|---|---|---|
| English (trained) | Latin | 19.72 | 44.87 | 58.35 | 7 | 100% |
| Hindi | Devanagari | 19.11 | 42.05 | 54.53 | 8 | 97% |
| German | Latin | 18.91 | 46.28 | 56.94 | 7 | 96% |
| Tamil | Tamil | 18.51 | 40.24 | 54.73 | 8 | 94% |
| Chinese | Han | 18.51 | 41.65 | 54.33 | 8 | 94% |
| Telugu | Telugu | 17.71 | 40.24 | 51.91 | 9 | 90% |
| Marathi | Devanagari | 16.50 | 41.45 | 53.12 | 8 | 84% |
| Arabic | Arabic | 16.30 | 40.04 | 52.92 | 9 | 83% |
| Bengali | Bengali | 16.30 | 40.64 | 53.12 | 9 | 83% |

83–97% retention across 8 unseen languages, 5 scripts, 4 language families.
Caveat: MT captions are literal/parallel; human captions would be a harder test.

## 3. Experiment 1 — JEPA head vs token-generation head (same frozen features)

| Metric | Mini VL-JEPA (ours) | Token baseline (DistilGPT-2 prefix) |
|---|---|---|
| Trainable params | 26.5M + 3.8M LoRA decoder | 85.7M |
| Retrieval R@1 | 19.72 | not capable |
| Caption BLEU-4 (20 refs) | 8.39 (bilingual decoder) | 17.27 |
| Languages | 8+ zero-shot | English only |
| Decoder needed | on demand | every output |

Monolingual decoder BLEU-4 was 7.41; bilingual training (20k EN/HI pairs)
raised it to 8.39 and enabled Hindi generation (quality-limited — content
collapses toward generic phrases; bottleneck is decoder data/size, not the
predictor: Hindi retrieval retains 97%).

## 4. Selective decoding (streaming: 497 videos × 8 progressive chunks)

Batch=1 latency: predictor forward 7.6 ms; decoder generate 255.7 ms (33.8×).

| τ | Decoder calls | Saved | Mean staleness | Max staleness | Est. ms/chunk | Speedup |
|---|---|---|---|---|---|---|
| 0.00 | 100.0% | 0.0% | 0.000 | 0.000 | 263 | 1.0× |
| 0.05 | 45.6% | 54.4% | 0.015 | 0.050 | 124 | 2.1× |
| **0.10** | **30.7%** | **69.3%** | **0.036** | **0.100** | **86** | **3.1×** |
| 0.15 | 23.3% | 76.7% | 0.061 | 0.150 | 67 | 3.9× |
| 0.20 | 19.2% | 80.8% | 0.081 | 0.200 | 57 | 4.6× |
| 0.25 | 16.9% | 83.1% | 0.097 | 0.249 | 51 | 5.2× |
| 0.30 | 15.4% | 84.6% | 0.111 | 0.299 | 47 | 5.6× |
| 0.40 | 13.6% | 86.4% | 0.133 | 0.395 | 42 | 6.2× |
| 0.50 | 12.8% | 87.2% | 0.146 | 0.494 | 40 | 6.5× |

Recommended operating point τ = 0.10 (knee of the curve). Max staleness is
bounded by τ by construction.

## Figures

`python scripts/make_figures.py` regenerates:
- figures/fig1_multilingual.png
- figures/fig2_selective_decoding.png
- figures/fig3_head_comparison.png

## Pending (final-numbers pass)

- Re-run retrieval + multilingual on the 2,990-video Test split
  (cache features with `--split test`, then eval with
  `--features data/features/test --annotations data/msrvtt_test.json`).

### Why tiers are hard: predictor quality vs caption-space density (validation split)

| tier | pred_cos | nearest-distractor sim | crowding deficit | R@1 % | cap_sim |
|---|---|---|---|---|---|
| easy | 0.614 | 0.743 | −0.129 | 18.9 | 0.852 |
| medium | 0.595 | 0.687 | −0.092 | 20.0 | 0.780 |
| hard | 0.575 | 0.645 | −0.070 | 20.3 | 0.710 |

Caption quality (cap_sim) tracks predictor quality (pred_cos): both decline monotonically with
tier complexity. Retrieval R@1 instead tracks the crowding deficit — on every tier the nearest
distractor caption lies closer to the target than the predictor's typical accuracy, and the
deficit is largest for easy videos, whose generic captions crowd together in LaBSE space. The
complexity tiers therefore predict captioning difficulty, not retrieval difficulty, and R@1
understates the semantic content the predictor captures, which generation reveals.

### Curriculum progression — validation split, predicted embeddings (real system)

LaBSE-sim per tier:
| tier | Stage 1 (easy) | Stage 2 (+medium) | Stage 3 (+hard) |
|---|---|---|---|
| easy | 0.846 | 0.852 | 0.853 |
| medium | 0.788 | 0.780 | 0.783 |
| hard | 0.700 | 0.710 | 0.713 |
| **all** | 0.778 | 0.781 | 0.783 |

BLEU-4 per tier:
| tier | Stage 1 (easy) | Stage 2 (+medium) | Stage 3 (+hard) |
|---|---|---|---|
| easy | 0.515 | 0.529 | 0.533 |
| medium | 0.378 | 0.378 | 0.394 |
| hard | 0.254 | 0.289 | 0.311 |
| **all** | 0.382 | 0.399 | 0.412 |

Each stage adds a tier to training; earlier tiers never regress (no catastrophic
forgetting), and the largest gains land on the newest tier. Medium's semantic
score is flat across stages, consistent with the predictor-limited ceiling
identified in the density analysis.

### Final captioning results — test split (2,990 videos), predicted embeddings

Decoder: curriculum Stage 3 (easy+medium+hard, noise-injected, prediction-adapted),
frozen before the single test run.

| tier | videos | BLEU-4 | ROUGE-L | LaBSE-sim |
|---|---|---|---|---|
| easy | 992 | 0.546 | 0.712 | 0.861 |
| medium | 966 | 0.399 | 0.594 | 0.784 |
| hard | 1032 | 0.304 | 0.503 | 0.709 |
| **all** | 2990 | 0.412 | 0.602 | 0.784 |

Test agrees with validation to within ±0.015 on every cell (all-tier BLEU-4: 0.4117 vs
0.4120), indicating no overfitting to the validation split during curriculum design. The
tier difficulty gradient is monotonic at 6× the validation sample size.

### Retrieval is pool-limited; captioning is not

| setting | pool size | mean pred_cos | mean nearest-distractor sim | R@1 | all-tier cap_sim |
|---|---|---|---|---|---|
| validation | 497 | 0.594 | 0.692 | 19.7% | 0.783 |
| test | 2990 | 0.597 | 0.747 | 6.6% | 0.784 |
| test, 497-video control | 497 | 0.597 | — | __%  | — |

Predictor alignment (pred_cos) and caption quality (cap_sim) are stable across splits.
R@1 drops 3x purely because the candidate pool grows 6x, pulling every caption's nearest
distractor closer (0.692 -> 0.747); restoring the validation pool size on test restores
R@1. Within a pool, per-tier R@1 differences are second-order: the predictor's better
alignment on easy videos is offset by their denser distractors, so the complexity tiers
predict captioning difficulty and predictor alignment, but not retrieval rank.

### Retrieval is pool-limited; captioning is not

| setting | pool size | mean pred_cos | mean nearest-distractor sim | R@1 | all-tier cap_sim |
|---|---|---|---|---|---|
| validation | 497 | 0.594 | 0.692 | 19.7% | 0.783 |
| test | 2990 | 0.597 | 0.747 | 6.6% | 0.784 |
| test, 497-video control (seed 42) | 497 | 0.595 | — | 18.5% | — |

Predictor alignment (pred_cos) and caption quality (cap_sim) are stable across splits.
R@1 drops 3x purely because the candidate pool grows 6x, pulling every caption's nearest
distractor closer (0.692 -> 0.747); restoring the validation pool size on test restores
R@1 (6.6% -> 18.5%). Within a pool, per-tier R@1 shows no consistent ordering across
splits and pool compositions, while pred_cos, distractor density, and all caption metrics
are consistently monotonic in tier: the complexity tiers predict captioning difficulty
and predictor alignment, not retrieval rank.