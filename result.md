Run 1: batch 48, lr 3e-4, 10 epochs, LaBSE target, 26.5M predictor
  e2: R@1 17.71 | R@5 43.66 | R@10 58.35 | MedR 7
  e3: R@1 16.90 | R@5 44.47 | R@10 57.95 | MedR 7

=== checkpoints\predictor_e5.pt ===
Video->Text retrieval over 497 videos
  R@1: 19.72
  R@5: 44.87
  R@10: 58.35
  MedR: 7

=== checkpoints\predictor_e7.pt ===
Video->Text retrieval over 497 videos
  R@1: 18.51
  R@5: 44.87
  R@10: 56.54
  MedR: 7

=== checkpoints\predictor_e9.pt ===
Video->Text retrieval over 497 videos
  R@1: 18.11
  R@5: 44.27
  R@10: 56.74
  MedR: 7

Language R@1 R@5 R@10 MedR 
English (trained on) 19.72 44.87 58.35 7
Hindi (never seen)   19.11 42.05 54.53 8

python scripts\eval_all_languages.py --ckpt checkpoints\predictor_e5.pt
Language        R@1    R@5   R@10  MedR
---------------------------------------
CachedFeatureDataset: 9062 caption pairs (0 skipped, no cached feature)
English       19.72  44.87  58.35     7
CachedFeatureDataset: 497 caption pairs (0 skipped, no cached feature)
Arabic        16.30  40.04  52.92     9
CachedFeatureDataset: 497 caption pairs (0 skipped, no cached feature)
Bengali       16.30  40.64  53.12     9
CachedFeatureDataset: 497 caption pairs (0 skipped, no cached feature)
German        18.91  46.28  56.94     7
CachedFeatureDataset: 497 caption pairs (0 skipped, no cached feature)
Hindi         19.11  42.05  54.53     8
CachedFeatureDataset: 497 caption pairs (0 skipped, no cached feature)
Marathi       16.50  41.45  53.12     8
CachedFeatureDataset: 497 caption pairs (0 skipped, no cached feature)
Tamil         18.51  40.24  54.73     8
CachedFeatureDataset: 497 caption pairs (0 skipped, no cached feature)
Telugu        17.71  40.24  51.91     9
CachedFeatureDataset: 497 caption pairs (0 skipped, no cached feature)
Chinese       18.51  41.65  54.33     8

saved results_multilingual.json

python -m src.baseline --config configs\default.yaml

Experiment 1: JEPA head vs token head (same frozen V-JEPA 2 features, MSR-VTT)
                     JEPA (ours)    Token baseline
trainable params     26.5M          85.7M
training epochs      10             5

                        JEPA (ours)      Token baseline
trainable params        26.5M            85.7M
retrieval R@1 (val)     19.72            —  (can't retrieve)
caption BLEU-4 (val)    ???              17.27
multilingual            8 langs zero-shot  English only

python scripts\score_captions.py --generated jepa_val_captions.json

Experiment 1 (MSR-VTT val, same frozen V-JEPA 2 features)
                          JEPA (ours)       Token baseline
trainable params          26.5M + 3.8M      85.7M
retrieval R@1             19.72             not capable
caption BLEU-4            7.41              17.27
languages                 8+ zero-shot      English only
decoder needed?           only on demand    every output

caption BLEU-4:  JEPA+bilingual decoder 8.39  |  token baseline 17.27
bilingual generation: yes (quality-limited)   |  no

Also worth one sentence in your analysis: the spread between languages shrank on test (0.83 points R@1 between best and worst, vs 3.4 on val) — larger evaluation pools wash out per-language noise, which strengthens the "transfer is near-uniform" conclusion.