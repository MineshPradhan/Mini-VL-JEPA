# Semester-wise Work Plan — Mini VL-JEPA

## Phase 1 · Weeks 1–4 — Foundations & literature survey
Read VL-JEPA (arXiv:2512.10942), V-JEPA 2, I-JEPA, LLM-JEPA (arXiv:2509.14252), LaBSE, and the "Alternatives to Next Token Prediction" survey (arXiv:2509.24435). Write the literature-review chapter draft. Set up the environment; run V-JEPA 2 and LaBSE inference on sample clips; verify feature shapes end to end. Deliverable: survey chapter + working environment + this demo shown to guide (simulation mode).

## Phase 2 · Weeks 5–8 — Data & baseline
Download and preprocess MSR-VTT (10k clips); build annotation JSONs; run `scripts/precompute_features.py` to cache features. Train the FIRST predictor on cached features (English-only MiniLM target first — simpler debugging). Implement the token-generation baseline head for the controlled comparison. Deliverable: first retrieval numbers (R@1/5/10) on MSR-VTT val, baseline vs JEPA head.

## Phase 3 · Weeks 9–12 — Multilingual + decoder
Switch target space to LaBSE; retrain. Machine-translate MSR-VTT val captions to Hindi for cross-lingual retrieval eval (or add VATEX EN/ZH). Train the mT5+LoRA embedding-to-text decoder; measure CIDEr/BLEU-4 on decoded captions in EN and HI. Deliverable: cross-lingual transfer table — the headline novelty result.

## Phase 4 · Weeks 13–16 — Selective decoding study
Implement the streaming evaluation: chunk videos, run SelectiveDecoder across τ ∈ {0.05…0.6}, plot % decoder calls vs caption quality. Wire the trained model into `server/app.py`; the frontend switches from simulation to live mode. Deliverable: efficiency-quality curve + live demo for mid-term evaluation.

## Phase 5 · Weeks 17–20 — Ablations & writing
Ablate: target encoder choice, predictor depth (2/4/6/8 layers), loss weights, number of latent queries, num frames. Error analysis: where does embedding prediction fail vs token generation (counting? rare objects?). Write thesis chapters 3–5. Stretch goal: submit a 4-page workshop paper (e.g., a CVPR/ICLR workshop on efficient multimodal models).

## Phase 6 · Weeks 21–24 — Thesis & defense
Final results table, demo polish, thesis submission, viva preparation. Prepare a 2-minute live demo script: (1) architecture slide, (2) stream a video with τ slider, (3) show the same embedding decoded in EN and HI.

## Risk plan
- GPU limits → feature caching makes the predictor trainable even on Colab T4 / Kaggle.
- Decoder quality weak → thesis still stands on retrieval + cross-lingual + efficiency results (all decoder-free except experiment 3).
- MSR-VTT download issues → fall back to MSVD (smaller) or VATEX.
