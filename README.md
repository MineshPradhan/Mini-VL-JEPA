# Mini VL-JEPA

**Non-autoregressive multilingual video captioning & VQA via text-embedding prediction**
M.Tech thesis starter — inspired by Meta's VL-JEPA (arXiv:2512.10942), built entirely from open, frozen components so it trains on a single GPU.

## Idea in one paragraph

Classical VLMs caption videos by generating tokens one at a time. Following VL-JEPA, we instead train a small predictor to map frozen **V-JEPA 2** video features directly to a **sentence embedding**, and invoke a lightweight text decoder only when needed. Two contributions beyond the original paper: (1) the target space is **multilingual** (LaBSE), so one predictor supports captions decodable in Hindi, English, and 100+ languages — including zero-shot cross-lingual transfer (train EN → eval HI); (2) **selective decoding as an event detector** for streaming video: the decoder fires only when the predicted embedding drifts past a threshold τ, and we study the compute/quality trade-off.

## What's trainable vs frozen

| Component | Model | Params | Status |
|---|---|---|---|
| Video encoder | facebook/vjepa2-vitl-fpc64-256 | 300M | ❄ frozen |
| Target text encoder | sentence-transformers/LaBSE | 470M | ❄ frozen |
| JEPA predictor | 6-layer cross-attn transformer | ~60M | ▲ trained (yours) |
| Text decoder | google/mt5-small + LoRA | ~2M trainable | ▲ trained (yours) |

## Repo layout

```
src/model.py        predictor, frozen encoder wrappers, JEPA loss, selective decoding
src/dataset.py      MSR-VTT / VATEX style loader
src/train.py        training loop (bf16, cosine schedule, single GPU)
src/decoder.py      embedding->text decoder (mT5 + LoRA, multilingual prompts)
src/eval.py         retrieval metrics (R@1/5/10, MedR), cross-lingual eval
scripts/precompute_features.py   cache V-JEPA 2 features once (10x faster epochs)
server/app.py       FastAPI inference API for the demo
frontend/index.html interactive demo (works standalone; connects to API when running)
configs/default.yaml
docs/WORK_PLAN.md   semester-wise plan
```

## Quickstart

```bash
# 1. environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. data — MSR-VTT (10k clips, ~6GB). Convert annotations to:
#    [{"video": "video0.mp4", "caption": "..."} , ...]
#    (see scripts/prepare_msrvtt.py notes)

# 3. (recommended) cache video features once
python scripts/precompute_features.py --config configs/default.yaml

# 4. train predictor (~hours, not weeks, since encoders are frozen)
python -m src.train --config configs/default.yaml

# 5. train the tiny decoder (text-only; fast)
python -m src.decoder --captions data/msrvtt_train.json

# 6. evaluate
python -m src.eval --config configs/default.yaml --ckpt checkpoints/predictor_e9.pt

# 7. demo for your guide
uvicorn server.app:app --port 8000
# open frontend/index.html in a browser (works even without the server, in simulation mode)
```

## Experiments to run (thesis chapters)

1. **Embedding head vs token head** — same frozen encoder, swap your predictor for a small autoregressive captioning head; compare quality, params, latency. (Small-scale replication of VL-JEPA's central claim.)
2. **Cross-lingual transfer** — train on English captions only; evaluate retrieval/captioning against Hindi captions (translate MSR-VTT val with any MT system, or use VATEX EN/ZH). Novel vs the original paper.
3. **Selective decoding curve** — sweep τ ∈ [0.05, 0.6]; plot % decoder calls vs CIDEr. Report the knee point.
4. **Ablations** — target encoder (LaBSE vs multilingual-E5 vs English-only MiniLM), predictor depth, loss weights, num latent queries.

## Key references

- VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language — arXiv:2512.10942 (ICLR 2026)
- V-JEPA 2 — github.com/facebookresearch/vjepa2
- LLM-JEPA — arXiv:2509.14252
- I-JEPA — Assran et al., CVPR 2023
- LaBSE — Feng et al., ACL 2022
