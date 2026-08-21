"""
Selective decoding experiment: sweep drift threshold tau on simulated
streaming (progressive temporal windows over cached V-JEPA 2 tokens).

    python scripts\selective_decoding.py --predictor checkpoints\predictor_e5.pt
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json
import time

import torch
import torch.nn.functional as F
import yaml

from src.dataset import CachedFeatureDataset
from src.model import JEPAPredictor, PredictorConfig

CHUNKS = 8
TAUS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--predictor", required=True)
    ap.add_argument("--decoder", default="checkpoints/decoder_bi.pt")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = JEPAPredictor(PredictorConfig(**cfg["predictor"])).to(device)
    predictor.load_state_dict(torch.load(args.predictor, map_location=device)["state_dict"])
    predictor.eval()

    ds = CachedFeatureDataset(cfg["data"]["val_annotations"], "data/features/val")
    uniq, seen = [], set()
    for s in ds.samples:
        if s["video"] not in seen:
            seen.add(s["video"]); uniq.append(s)
    print(f"streaming simulation over {len(uniq)} videos x {CHUNKS} chunks")

    # ---- pass 1: embedding trajectory per video ----
    trajectories = []
    with torch.no_grad():
        for s in uniq:
            feats = torch.load(os.path.join("data/features/val",
                    os.path.basename(s["video"]) + ".pt"), map_location="cpu").float()
            N = feats.size(0)
            traj = []
            for t in range(1, CHUNKS + 1):
                window = feats[: max(1, N * t // CHUNKS)].unsqueeze(0).to(device)
                traj.append(predictor(window).squeeze(0).cpu())
            trajectories.append(torch.stack(traj))          # (CHUNKS, D)
    T = torch.stack(trajectories)                            # (V, CHUNKS, D)

    # ---- pass 2: sweep tau ----
    results = []
    V = T.size(0)
    for tau in TAUS:
        decodes, staleness = 0, []
        for v in range(V):
            last = None
            for t in range(CHUNKS):
                z = T[v, t]
                if last is None:
                    last = z; decodes += 1; staleness.append(0.0)
                    continue
                drift = 1.0 - F.cosine_similarity(z, last, dim=0).item()
                if drift > tau:
                    last = z; decodes += 1; staleness.append(0.0)
                else:
                    staleness.append(drift)
        total = V * CHUNKS
        results.append({
            "tau": tau,
            "decode_rate": decodes / total,
            "decode_savings": 1 - decodes / total,
            "mean_staleness": sum(staleness) / len(staleness),
            "max_staleness": max(staleness),
        })

    print(f"\n{'tau':>5} {'decodes':>9} {'saved':>7} {'mean_stale':>11} {'max_stale':>10}")
    for r in results:
        print(f"{r['tau']:>5.2f} {r['decode_rate']*100:>8.1f}% "
              f"{r['decode_savings']*100:>6.1f}% {r['mean_staleness']:>11.4f} "
              f"{r['max_staleness']:>10.4f}")

    # ---- pass 3: latency at batch=1 (predictor vs decoder) ----
    feats1 = torch.load(os.path.join("data/features/val",
             os.path.basename(uniq[0]["video"]) + ".pt"), map_location="cpu").float()
    feats1 = feats1.unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(3): predictor(feats1)                # warmup
        if device == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50): predictor(feats1)
        if device == "cuda": torch.cuda.synchronize()
        pred_ms = (time.time() - t0) / 50 * 1000

    dec_ms = None
    if os.path.exists(args.decoder):
        from src.decoder import EmbToTextDecoder
        dec = EmbToTextDecoder().to(device)
        dec.load_state_dict(torch.load(args.decoder, map_location=device))
        dec.eval()
        z1 = predictor(feats1)
        with torch.no_grad():
            for _ in range(2): dec.generate(z1)
            if device == "cuda": torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(10): dec.generate(z1)
            if device == "cuda": torch.cuda.synchronize()
            dec_ms = (time.time() - t0) / 10 * 1000

    print(f"\npredictor forward (batch=1): {pred_ms:.1f} ms")
    if dec_ms:
        print(f"decoder generate  (batch=1): {dec_ms:.1f} ms  "
              f"({dec_ms/pred_ms:.1f}x the predictor)")
        for r in results:
            r["est_ms_per_chunk"] = pred_ms + r["decode_rate"] * dec_ms

    with open("results_selective_decoding.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nsaved results_selective_decoding.json")


if __name__ == "__main__":
    main()