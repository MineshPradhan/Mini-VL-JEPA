"""
FastAPI inference server for the Mini VL-JEPA demo.

Run from the project root:
    uvicorn server.app:app --port 8000

Then open http://127.0.0.1:8000/  (presentation UI, web/demo.html)
     or http://127.0.0.1:8000/classic  (streaming-scope UI, index.html)

Endpoints
---------
GET  /                                -> web/demo.html
GET  /classic                         -> index.html
GET  /health                          -> {"status": "ok", "model_loaded": bool, ...}
POST /predict   (multipart video)     -> whole-clip caption + per-window
                                         embeddings (PCA-2D), drift, decode
                                         events, English captions
GET  /api/health, /api/samples        -> backend status, demo clip list
GET  /api/video/{id}                  -> the clip itself (for <video>)
POST /api/caption (json or multipart) -> easy/medium/hard EN captions re-ranked
                                         by LaBSE grounding, HI/ZH via NLLB,
                                         selective-gate trace

Why windows and not 1-second chunks
-----------------------------------
The predictor was trained (and the decoder adapted, Stage 1b) on V-JEPA 2
features of 16 frames spread uniformly over a WHOLE MSR-VTT clip (~10-30 s,
i.e. roughly 1 frame/s). Feeding it 16 near-identical frames from a single
second is far outside that distribution, so the predicted embedding - and the
caption decoded from it - is unreliable. Each streaming step therefore looks
at a `window_seconds`-wide window (default 8 s) around the current time, and
the headline caption uses exactly the offline-eval sampling.
"""
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from train_decoder_en import build_model as build_caption_decoder
from train_decoder_en import generate as generate_caption

app = FastAPI(title="Mini VL-JEPA API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

STATE = {"loaded": False}


def _resolve(p):
    """Paths are relative to the project root, not the shell's cwd."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


PREDICTOR_CKPT = _resolve(os.environ.get("PREDICTOR_CKPT",
                                         "checkpoints/predictor_e5.pt"))
DECODER_CKPT = _resolve(os.environ.get("DECODER_CKPT",
                                       "checkpoints/dec_en_all/decoder.pt"))
# NOTE: the caption decoder is adapted to predictor_e5's output distribution.
# If you ever change PREDICTOR_CKPT, re-run export_predicted_embeddings.py and
# the Stage 1b fine-tune so the decoder matches the new predictor.
CONFIG_PATH = ROOT / "configs" / "default.yaml"
VIDEO_DIR = _resolve(os.environ.get("VIDEO_DIR", "MSR-VTT/Video_Datasets/Val"))
UPLOAD_DIR = ROOT / "uploads"

GPU_LOCK = threading.Lock()  # one request on the 6 GB card at a time
RESULT_CACHE = {}

# Clips from results/demo_shortlist_final.md (R@1-correct, cap_sim >= 0.8).
SAMPLES = [("video7004", "Trampoline"), ("video6745", "Cooking"),
           ("video6906", "Running"), ("video6937", "Phone call"),
           ("video6973", "Table tennis"), ("video6919", "Guitar"),
           ("video6956", "Surfing"), ("video6971", "Video game")]

# Measured numbers (RESULTS.md: predictor_e5, MSR-VTT val, 497 videos).
METRICS = {"r1": 19.72, "r5": 44.87, "r10": 58.35, "medr": 7,
           "retention": "83-97%", "speedup": 3.1, "params_m": 26.5}

# Decoding presets for the three candidate captions. "medium" is exactly the
# setting the reported caption metrics were computed with (eval_captions.py).
TIER_PRESETS = {
    "easy": dict(num_beams=4, max_new_tokens=12, length_penalty=0.6,
                 no_repeat_ngram_size=2),
    "medium": dict(num_beams=5, max_new_tokens=30, length_penalty=1.0,
                   no_repeat_ngram_size=3),
    "hard": dict(num_beams=8, max_new_tokens=40, length_penalty=2.0,
                 no_repeat_ngram_size=3),
}
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_CODES = {"hi": "hin_Deva", "zh": "zho_Hans"}


def load_models():
    if STATE["loaded"]:
        return
    # Fail loudly: a missing checkpoint used to surface as a silent
    # "(decoder not trained yet)" caption or an opaque 500.
    missing = [str(p) for p in (PREDICTOR_CKPT, DECODER_CKPT) if not p.exists()]
    if missing:
        raise HTTPException(503, "checkpoint(s) not found: " + ", ".join(missing)
                            + " - copy the checkpoints/ folder into the project "
                              "root or set PREDICTOR_CKPT / DECODER_CKPT")

    import yaml
    from peft import set_peft_model_state_dict
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from src.model import (FrozenVideoEncoder, JEPAPredictor,
                           PredictorConfig, SelectiveDecoder)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    STATE["device"] = device
    STATE["cfg"] = cfg

    print(f"device: {device}")
    print("loading V-JEPA 2 encoder ...")
    STATE["video_enc"] = FrozenVideoEncoder(cfg["video_encoder"]).to(device)

    print(f"loading predictor from {PREDICTOR_CKPT} ...")
    pred = JEPAPredictor(PredictorConfig(**cfg["predictor"])).to(device)
    sd = torch.load(PREDICTOR_CKPT, map_location=device)
    pred.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
    pred.eval()
    STATE["predictor"] = pred

    print(f"loading caption decoder from {DECODER_CKPT} ...")
    ck = torch.load(DECODER_CKPT, map_location=device)
    dtok, dmodel, dproj = build_caption_decoder(ck["config"], device)
    set_peft_model_state_dict(dmodel, ck["lora"])
    dproj.load_state_dict(ck["projector"])
    dmodel.eval()
    dproj.eval()
    STATE["decoder"] = (dtok, dmodel, dproj)

    # LaBSE re-ranks candidates: cosine(LaBSE(caption), z_hat) is the
    # "grounding" score. NLLB renders the captions in Hindi / Chinese (the
    # current decoder is English-only until curriculum Stage 4).
    print("loading LaBSE (re-ranker) ...")
    STATE["labse"] = SentenceTransformer(cfg["text_encoder"], device=device)
    print("loading NLLB (HI/ZH translation) ...")
    STATE["nllb_tok"] = AutoTokenizer.from_pretrained(NLLB_MODEL,
                                                      src_lang="eng_Latn")
    STATE["nllb"] = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_MODEL,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    STATE["nllb"].generation_config.max_length = None  # we pass max_new_tokens

    STATE["SelectiveDecoder"] = SelectiveDecoder
    STATE["loaded"] = True
    print("models ready.")


# ---------------------------------------------------------------- pipeline
def embed_frames(vr, idx, fast=False):
    """16 frame indices -> predicted LaBSE embedding (1, D), as in training.

    fast=True runs V-JEPA 2 under fp16 autocast (~2x faster on the RTX 3050;
    embedding cosine vs fp32 >= 0.999, same captions on the shortlist)."""
    device = STATE["device"]
    frames = vr.get_batch([int(i) for i in idx]).asnumpy()
    pixel = STATE["video_enc"].processor([frames], return_tensors="pt")[
        "pixel_values_videos"].to(device)
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=fast and device == "cuda"):
            feats = STATE["video_enc"](pixel)
        return STATE["predictor"](feats.float())


def caption(z_hat):
    dtok, dmodel, dproj = STATE["decoder"]
    return generate_caption(dmodel, dproj, dtok, z_hat, STATE["device"])[0]


def caption_tier(z_hat, tier):
    from transformers.modeling_outputs import BaseModelOutput
    dtok, dmodel, dproj = STATE["decoder"]
    with torch.no_grad():
        enc = BaseModelOutput(last_hidden_state=dproj(z_hat.to(STATE["device"])))
        out = dmodel.generate(encoder_outputs=enc, early_stopping=True,
                              **TIER_PRESETS[tier])
    return dtok.batch_decode(out, skip_special_tokens=True)[0].strip()


def grounding(texts, z_hat):
    """cosine(LaBSE(text), z_hat) for each text."""
    e = STATE["labse"].encode(texts, convert_to_tensor=True,
                              normalize_embeddings=True)
    return (e.float() @ z_hat.float().squeeze(0).to(e.device)).tolist()


def translate(texts, lang):
    tok, model = STATE["nllb_tok"], STATE["nllb"]
    x = tok(texts, return_tensors="pt", padding=True).to(STATE["device"])
    with torch.no_grad():
        out = model.generate(**x, max_new_tokens=60, num_beams=4,
                             forced_bos_token_id=tok.convert_tokens_to_ids(
                                 NLLB_CODES[lang]))
    return [t.strip() for t in tok.batch_decode(out, skip_special_tokens=True)]


def analyze(path, threshold=0.10, chunk_seconds=1.0, window_seconds=8.0,
            decode_events=True):
    """Whole-clip embedding + sliding-window selective-decoding replay."""
    from decord import VideoReader, cpu

    try:
        vr = VideoReader(str(path), ctx=cpu(0))
    except Exception as e:
        raise HTTPException(400, f"could not decode video: {e}")
    try:
        n = len(vr)
        if n == 0:
            raise HTTPException(400, "video has no frames")
        fps = vr.get_avg_fps() or 25
        num_frames = STATE["cfg"]["data"]["num_frames"]

        # 1) whole clip: identical sampling to precompute_features.py, i.e.
        #    exactly the conditions the reported caption metrics come from.
        z_clip = embed_frames(vr, np.linspace(0, n - 1, num_frames).astype(int))

        # 2) streaming replay: one step per chunk, each seeing a fixed-width
        #    window around the current time (clamped inside the video).
        hop = max(1, int(round(fps * chunk_seconds)))
        win = max(num_frames, int(round(fps * window_seconds)))
        sel = STATE["SelectiveDecoder"](threshold=threshold)
        embeddings, events = [], []
        for start in range(0, n, hop):
            if win >= n:
                z_hat = z_clip  # clip shorter than the window: reuse
            else:
                centre = start + hop / 2
                lo = int(min(max(0, centre - win / 2), n - win))
                idx = np.linspace(lo, lo + win - 1, num_frames).astype(int)
                z_hat = embed_frames(vr, idx, fast=True)  # (1, D)
            fire, drift = sel.should_decode(z_hat)
            embeddings.append(z_hat.squeeze(0).cpu().numpy())
            event = {"t": round(start / fps, 2),
                     "drift": round(float(drift), 4), "decoded": bool(fire)}
            if fire and decode_events:
                event["caption_en"] = caption(z_hat)
                # Hindi is paused during the English-first curriculum
                # (Stage 4 brings it back); empty string keeps the frontend happy.
                event["caption_hi"] = ""
            events.append(event)
    finally:
        del vr  # release decord's file handle (Windows can't delete open files)
    return z_clip, embeddings, events


def caption_result(path, video_id, langs=("en", "hi", "zh"), threshold=0.10):
    """Response shape expected by web/demo.html."""
    t0 = time.perf_counter()
    z, _, events = analyze(path, threshold=threshold, decode_events=False)
    t_enc = time.perf_counter()

    cands = [{"id": t, "tier": t, "text": caption_tier(z, t)}
             for t in TIER_PRESETS]
    for c, cos in zip(cands, grounding([c["text"] for c in cands], z)):
        c["score"] = round(float(cos), 4)
    best_id = max(cands, key=lambda c: c["score"])["id"]
    out = {"en": {"best_id": best_id, "candidates": cands,
                  "source": "mT5 decoder"}}
    for lang in langs:
        if lang not in NLLB_CODES:
            continue
        texts = translate([c["text"] for c in cands], lang)
        tc = [{"id": c["id"], "tier": c["tier"], "text": t}
              for c, t in zip(cands, texts)]
        for c, cos in zip(tc, grounding(texts, z)):
            c["score"] = round(float(cos), 4)
        # keep the English choice so every language shows the same caption
        out[lang] = {"best_id": best_id, "candidates": tc,
                     "source": "NLLB translation of the English candidates"}
    t_dec = time.perf_counter()

    decoded = sum(e["decoded"] for e in events)
    return {
        "video_id": video_id, "tau": threshold, "offline": False,
        "langs": {k: v for k, v in out.items() if k in langs},
        "selective": {"tau": threshold, "windows": len(events),
                      "decoded": decoded,
                      "speedup": round(len(events) / max(1, decoded), 2),
                      "triggers": [{"t": e["t"], "drift": e["drift"],
                                    "decoded": e["decoded"]} for e in events]},
        "timings": {"encode_ms": round((t_enc - t0) * 1000, 1),
                    "decode_ms": round((t_dec - t_enc) * 1000, 1),
                    "total_ms": round((t_dec - t0) * 1000, 1)},
        "metrics": METRICS,
        "device": STATE["device"],
    }


def _sample_path(vid):
    p = VIDEO_DIR / f"{Path(vid).name}.mp4"
    if not p.exists():
        raise HTTPException(404, f"no video named {vid}")
    return p


def _warm_cache():
    """Caption the demo clips in the background so clicking them is instant."""
    for vid, _ in SAMPLES:
        if (VIDEO_DIR / f"{vid}.mp4").exists() and vid not in RESULT_CACHE:
            try:
                with GPU_LOCK:
                    if vid not in RESULT_CACHE:
                        RESULT_CACHE[vid] = caption_result(_sample_path(vid), vid)
                en = RESULT_CACHE[vid]["langs"]["en"]
                best = next(c for c in en["candidates"] if c["id"] == en["best_id"])
                print(f"  ready {vid}: {best['text']} ({best['score']:.3f})")
            except Exception as e:  # never let warm-up kill the server
                print(f"  warm-up failed for {vid}: {e!r}")
    print("demo clips ready.")


@app.on_event("startup")
def preload():
    """Load models when uvicorn starts so the first request isn't ~30 s slower."""
    try:
        load_models()
    except HTTPException as e:
        print(f"WARNING: {e.detail}")
        return
    threading.Thread(target=_warm_cache, daemon=True).start()


# ---------------------------------------------------------------- pages
@app.get("/", include_in_schema=False)
def demo_page():
    return FileResponse(ROOT / "web" / "demo.html")


@app.get("/classic", include_in_schema=False)
def classic_page():
    for p in (ROOT / "index.html", ROOT / "frontend" / "index.html"):
        if p.exists():
            return FileResponse(p)
    raise HTTPException(404, "index.html not found")


# ---------------------------------------------------------------- classic API
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": STATE["loaded"],
            "cuda": torch.cuda.is_available(),
            "predictor_ckpt_found": PREDICTOR_CKPT.exists(),
            "decoder_ckpt_found": DECODER_CKPT.exists()}


@app.post("/predict")
async def predict(file: UploadFile = File(...), chunk_seconds: float = 1.0,
                  threshold: float = 0.10, window_seconds: float = 8.0):
    load_models()
    raw = await file.read()
    suffix = Path(file.filename or "").suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(raw)
    tmp.close()  # Windows: must close before decord opens it
    try:
        with GPU_LOCK:
            z_clip, embeddings, events = analyze(
                tmp.name, threshold, chunk_seconds, window_seconds)
            clip_caption = caption(z_clip)
    finally:
        try:
            os.unlink(tmp.name)
        except (PermissionError, OSError):
            pass  # temp file will be cleaned by the OS later

    E = np.stack(embeddings)
    E = E - E.mean(0, keepdims=True)
    if len(E) >= 2 and np.abs(E).max() > 1e-8:
        _, _, Vt = np.linalg.svd(E, full_matrices=False)
        proj = (E @ Vt[:2].T).tolist()
    else:
        proj = [[0.0, 0.0]] * len(E)

    total = len(events)
    decoded = sum(1 for e in events if e["decoded"])
    return {"clip_caption": clip_caption,
            "trajectory_2d": proj, "events": events,
            "stats": {"chunks": total, "decoded": decoded,
                      "decode_savings": round(1 - decoded / max(1, total), 3),
                      "window_seconds": window_seconds,
                      "device": STATE["device"]}}


# ---------------------------------------------------------------- demo UI API
@app.get("/api/health")
def api_health():
    if not STATE["loaded"]:
        raise HTTPException(503, "models not loaded - check the server console")
    return {"status": "ok", "model": "mini-vl-jepa", "device": STATE["device"]}


@app.get("/api/samples")
def api_samples():
    return {"samples": [{"id": v, "title": t, "url": f"/api/video/{v}"}
                        for v, t in SAMPLES if (VIDEO_DIR / f"{v}.mp4").exists()]}


@app.get("/api/video/{vid}")
def api_video(vid: str):
    return FileResponse(_sample_path(vid), media_type="video/mp4")


@app.get("/api/metrics")
def api_metrics():
    return METRICS


@app.post("/api/caption")
async def api_caption(request: Request):
    load_models()
    if request.headers.get("content-type", "").startswith("multipart/"):
        form = await request.form()
        f = form.get("file")
        if f is None:
            raise HTTPException(400, "multipart request without a 'file' field")
        UPLOAD_DIR.mkdir(exist_ok=True)
        dest = UPLOAD_DIR / (uuid.uuid4().hex[:10]
                             + (Path(f.filename or "").suffix or ".mp4"))
        dest.write_bytes(await f.read())
        try:
            with GPU_LOCK:
                return caption_result(dest, "upload")
        finally:
            try:
                dest.unlink()
            except OSError:
                pass

    body = await request.json()
    vid = str(body.get("video_id", ""))
    path = _sample_path(vid)
    with GPU_LOCK:
        if vid not in RESULT_CACHE:
            RESULT_CACHE[vid] = caption_result(path, vid)
    return RESULT_CACHE[vid]
