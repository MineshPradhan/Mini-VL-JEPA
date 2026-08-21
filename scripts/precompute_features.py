"""
Cache frozen V-JEPA 2 features to disk. Run once per split:
    python scripts\precompute_features.py --config configs\default.yaml --split val
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch
import yaml
from torch.utils.data import DataLoader
from src.dataset import VideoCaptionDataset, make_collate_fn
from src.model import FrozenVideoEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="data/features")
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    enc = FrozenVideoEncoder(cfg["video_encoder"]).to(device)
    ann_map = {"train": cfg["data"]["train_annotations"],
               "val": cfg["data"]["val_annotations"],
               "test": "data/msrvtt_test.json"}
    ds = VideoCaptionDataset(ann_map[args.split], cfg["data"]["video_root"],
                             num_frames=cfg["data"]["num_frames"], split="val")
    
    # deduplicate: cache each video once, not once per caption (~20x speedup)
    uniq, seen_vids = [], set()
    for s in ds.samples:
        if s["video"] not in seen_vids:
            seen_vids.add(s["video"])
            uniq.append(s)
    ds.samples = uniq
    print(f"unique videos to cache: {len(ds.samples)}")

    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                    collate_fn=make_collate_fn(enc.processor))

    out_dir = os.path.join(args.out, args.split)
    os.makedirs(out_dir, exist_ok=True)
    seen = set(f[:-3] for f in os.listdir(out_dir) if f.endswith(".pt"))  # resume support
    print(f"already cached: {len(seen)} videos (will skip)")

    done = len(seen)
    with torch.no_grad():
        for i, batch in enumerate(dl):
            todo = [j for j, vid in enumerate(batch["video_ids"])
                    if os.path.basename(vid) not in seen]
            if not todo:
                continue
            feats = enc(batch["pixel_values_videos"].to(device)).cpu()
            for j in todo:
                vid = os.path.basename(batch["video_ids"][j])
                torch.save(feats[j].clone().half(), os.path.join(out_dir, vid + ".pt"))
                seen.add(vid)
                done += 1
            if i % 20 == 0:
                print(f"batch {i} | cached {done} videos")

    print(f"finished: {done} videos -> {out_dir}")


if __name__ == "__main__":
    main()