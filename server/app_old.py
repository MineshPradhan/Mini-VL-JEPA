"""
FastAPI inference server for the Mini VL-JEPA frontend demo.

Run from the project root:
    uvicorn server.app:app --port 8000

Then open frontend/index.html in a browser -> section 03 -> Check connection -> Upload video.

Endpoints
---------
GET  /health                          -> {"status": "ok", "model_loaded": bool}
POST /predict   (multipart video)     -> per-chunk embeddings (PCA-2D), drift,
                                         decode events, captions (en + hi)
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Mini VL-JEPA API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

STATE = {"loaded": False}

PREDICTOR_CKPT = os.environ.get("PREDICTOR_CKPT", "checkpoints/predictor_e5.pt")
DECODER_CKPT = os.environ.get("DECODER_CKPT", "checkpoints/decoder_bi.pt")


def load_models():
    if STATE["loaded"]:
        return
    import yaml
    from src.decoder import EmbToTextDecoder
    from src.model import (FrozenVideoEncoder, JEPAPredictor,
                           PredictorConfig, SelectiveDecoder)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    STATE["device"] = device
    STATE["cfg"] = cfg

    print("loading V-JEPA 2 encoder ...")
    STATE["video_enc"] = FrozenVideoEncoder(cfg["video_encoder"]).to(device)

    print(f"loading predictor from {PREDICTOR_CKPT} ...")
    pred = JEPAPredictor(PredictorConfig(**cfg["predictor"])).to(device)
    sd = torch.load(PREDICTOR_CKPT, map_location=device)
    pred.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
    pred.eval()
    STATE["predictor"] = pred

    STATE["decoder"] = None
    if os.path.exists(DECODER_CKPT):
        print(f"loading decoder from {DECODER_CKPT} ...")
        dec = EmbToTextDecoder().to(device)
        dec.load_state_dict(torch.load(DECODER_CKPT, map_location=device))
        dec.eval()
        STATE["decoder"] = dec
    else:
        print(f"WARNING: {DECODER_CKPT} not found; captions disabled")

    STATE["SelectiveDecoder"] = SelectiveDecoder
    STATE["loaded"] = True
    print("models ready.")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": STATE["loaded"],
            "cuda": torch.cuda.is_available()}

@app.post("/predict")
async def predict(file: UploadFile = File(...), chunk_seconds: float = 1.0,
                  threshold: float = 0.10):
    load_models()
    from decord import VideoReader, cpu

    device = STATE["device"]
    raw = await file.read()
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(raw)
    tmp.close()  # Windows: must close before decord opens it

    try:
        vr = VideoReader(tmp.name, ctx=cpu(0))
        fps = vr.get_avg_fps() or 25
        chunk_len = max(1, int(fps * chunk_seconds))
        num_frames = STATE["cfg"]["data"]["num_frames"]

        sel = STATE["SelectiveDecoder"](threshold=threshold)
        embeddings, events = [], []

        for start in range(0, len(vr), chunk_len):
            end = min(len(vr), start + chunk_len)
            idx = np.linspace(start, end - 1, num_frames).astype(int)
            frames = vr.get_batch(idx.tolist()).asnumpy()
            pixel = STATE["video_enc"].processor([frames], return_tensors="pt")[
                "pixel_values_videos"].to(device)
            with torch.no_grad():
                z_hat = STATE["predictor"](STATE["video_enc"](pixel))  # (1, D)
            fire, drift = sel.should_decode(z_hat)
            embeddings.append(z_hat.squeeze(0).cpu().numpy())

            event = {"t": round(start / fps, 2),
                     "drift": round(float(drift), 4), "decoded": bool(fire)}
            if fire and STATE["decoder"] is not None:
                with torch.no_grad():
                    event["caption_en"] = STATE["decoder"].generate(
                        z_hat, "caption in English: ")[0]
                    event["caption_hi"] = STATE["decoder"].generate(
                        z_hat, "caption in Hindi: ")[0]
            events.append(event)

        del vr  # release decord's file handle (Windows can't delete open files)
    finally:
        try:
            os.unlink(tmp.name)
        except (PermissionError, OSError):
            pass  # temp file will be cleaned by the OS later

    E = np.stack(embeddings)
    E = E - E.mean(0, keepdims=True)
    if len(E) >= 2:
        _, _, Vt = np.linalg.svd(E, full_matrices=False)
        proj = (E @ Vt[:2].T).tolist()
    else:
        proj = [[0.0, 0.0]]

    total = len(events)
    decoded = sum(1 for e in events if e["decoded"])
    return {"trajectory_2d": proj, "events": events,
            "stats": {"chunks": total, "decoded": decoded,
                      "decode_savings": round(1 - decoded / max(1, total), 3)}}