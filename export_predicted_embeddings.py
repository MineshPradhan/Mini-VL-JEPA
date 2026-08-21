"""
Export the JEPA predictor's output embeddings to curriculum/preds.npz.
FILLED IN for the Mini VL-JEPA repo layout - no TODOs left:

  - builds JEPAPredictor from src/model.py with configs/default.yaml
  - runs it over the cached V-JEPA features in data/features/{train,val,test}
    (cache filenames keep the extension: video7010.mp4.pt)
  - saves embeddings keyed by the SAME video ids as curriculum/tiers.json
    (e.g. "Train/video0.mp4") so train_decoder_en.py / eval_captions.py match
  - prints mean cos(pred, best GT caption) per split and a suggested
    --noise_std for train_decoder_en.py, derived from that gap

Run from the project root:
  python export_predicted_embeddings.py --bank curriculum/caption_bank.npz --tiers curriculum/tiers.json --ckpt checkpoints/predictor_e5.pt
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from curriculum_common import load_bank, load_tiers
from src.model import JEPAPredictor, PredictorConfig

SPLIT_DIR = {"train": "train", "validate": "val", "test": "test"}


def feature_path(features_root, video_id, split):
    """tiers id 'Test/video7010.mp4' -> data/features/test/video7010.mp4.pt"""
    fname = os.path.basename(str(video_id)) + ".pt"
    return os.path.join(features_root, SPLIT_DIR.get(split, split), fname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--tiers", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default="checkpoints/predictor_e5.pt",
                    help="predictor checkpoint (the live demo uses predictor_e5.pt)")
    ap.add_argument("--features_root", default="data/features")
    ap.add_argument("--out", default="curriculum/preds.npz")
    ap.add_argument("--splits", default="train,validate", help="which splits to export")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    predictor = JEPAPredictor(PredictorConfig(**cfg["predictor"])).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    predictor.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
    predictor.eval()
    print(f"loaded predictor from {args.ckpt}")

    tiers = load_tiers(args.tiers)
    want = set(args.splits.split(","))
    todo, missing = [], 0
    for vid, info in sorted(tiers.items()):
        if info["split"] not in want:
            continue
        p = feature_path(args.features_root, vid, info["split"])
        if os.path.exists(p):
            todo.append((vid, p))
        else:
            missing += 1
    print(f"{len(todo)} videos with cached features ({missing} missing cache files)")
    if not todo:
        raise SystemExit("No cached features found - check --features_root.")

    out_ids, out_chunks = [], []
    with torch.no_grad():
        for s in range(0, len(todo), args.batch_size):
            chunk = todo[s:s + args.batch_size]
            feats = torch.stack(
                [torch.load(p, map_location="cpu").float() for _, p in chunk]
            ).to(device)
            z = F.normalize(predictor(feats).float(), dim=-1).cpu().numpy()
            out_ids.extend(v for v, _ in chunk)
            out_chunks.append(z)
            done = min(s + args.batch_size, len(todo))
            if (s // args.batch_size) % 10 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)}")

    embs = np.concatenate(out_chunks).astype(np.float16)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, video_ids=np.array(out_ids), embeddings=embs)
    print(f"Saved {len(out_ids)} embeddings -> {args.out}")

    # ---- measure the predictor->GT gap and suggest a noise_std --------------
    bvids, _, bembs = load_bank(args.bank)
    gt_by_vid = {}
    for j, v in enumerate(bvids):
        gt_by_vid.setdefault(v, []).append(j)

    per_split = {}
    for vid, pe in zip(out_ids, embs):
        if vid not in gt_by_vid:
            continue
        G = bembs[gt_by_vid[vid]].astype(np.float32)          # (n_caps, 768)
        best_cos = float(np.max(G @ pe.astype(np.float32)))    # closest GT caption
        per_split.setdefault(tiers[vid]["split"], []).append(best_cos)

    d = embs.shape[1]
    for split, vals in sorted(per_split.items()):
        c = float(np.mean(vals))
        sigma = float(np.sqrt(max(2 - 2 * c, 1e-6)) / np.sqrt(d))
        print(f"[{split}] mean cos(pred, best GT caption) = {c:.4f}  ->  suggested --noise_std {sigma:.3f}")
    print("Use the *train*-split suggestion (or slightly larger) for train_decoder_en.py.")


if __name__ == "__main__":
    main()
