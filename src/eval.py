"""Evaluate retrieval from cached features.
    python -m src.eval --config configs/default.yaml --ckpt checkpoints/predictor_e3.pt
"""
import argparse
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import CachedFeatureDataset, cached_collate
from src.model import FrozenTextTarget, JEPAPredictor, PredictorConfig


@torch.no_grad()
def retrieval_metrics(z_hat, z):
    sims = z_hat @ z.t()
    ranks = (sims.argsort(dim=-1, descending=True)
             .eq(torch.arange(len(z)).unsqueeze(1)).nonzero()[:, 1])
    return {"R@1":  (ranks < 1).float().mean().item() * 100,
            "R@5":  (ranks < 5).float().mean().item() * 100,
            "R@10": (ranks < 10).float().mean().item() * 100,
            "MedR": ranks.median().item() + 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--features", default="data/features/val")
    ap.add_argument("--annotations", default=None,
                    help="override val annotations (e.g. Hindi file)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text_enc = FrozenTextTarget(cfg["text_encoder"])
    predictor = JEPAPredictor(PredictorConfig(**cfg["predictor"])).to(device)
    state = torch.load(args.ckpt, map_location=device)
    predictor.load_state_dict(state["state_dict"])
    predictor.eval()

    ann = args.annotations or cfg["data"]["val_annotations"]
    ds = CachedFeatureDataset(ann, args.features)
    # ds = CachedFeatureDataset(cfg["data"]["val_annotations"], args.features)

    # standard protocol: one caption per video
    uniq, seen = [], set()
    for s in ds.samples:
        if s["video"] not in seen:
            seen.add(s["video"])
            uniq.append(s)
    ds.samples = uniq
    print(f"evaluating on {len(ds.samples)} unique videos")

    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=cached_collate)

    Z_hat, Z = [], []
    with torch.no_grad():
        for batch in dl:
            Z_hat.append(predictor(batch["features"].to(device)).cpu())
            Z.append(text_enc(batch["captions"], device).cpu())
    Z_hat, Z = torch.cat(Z_hat), torch.cat(Z)

    m = retrieval_metrics(Z_hat, Z)
    print(f"\n=== {args.ckpt} ===")
    print(f"Video->Text retrieval over {len(Z)} videos")
    for k, v in m.items():
        print(f"  {k}: {v:.2f}" if k != "MedR" else f"  {k}: {v:.0f}")


if __name__ == "__main__":
    main()