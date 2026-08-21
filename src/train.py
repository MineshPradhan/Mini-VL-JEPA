"""Train the JEPA predictor from cached features.
    python -m src.train --config configs/default.yaml
"""
import argparse, math, os
import torch, yaml
from torch.utils.data import DataLoader

from src.dataset import CachedFeatureDataset, cached_collate
from src.model import FrozenTextTarget, JEPAPredictor, PredictorConfig, jepa_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--features", default="data/features/train")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text_enc = FrozenTextTarget(cfg["text_encoder"])
    predictor = JEPAPredictor(PredictorConfig(**cfg["predictor"])).to(device)
    print(f"trainable params: {sum(p.numel() for p in predictor.parameters())/1e6:.1f}M")

    ds = CachedFeatureDataset(cfg["data"]["train_annotations"], args.features)
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                    num_workers=cfg["train"]["num_workers"],
                    collate_fn=cached_collate, pin_memory=True, drop_last=True)

    epochs = cfg["train"]["epochs"]
    opt = torch.optim.AdamW(predictor.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    total = epochs * len(dl); warmup = int(0.05 * total)

    def lr_lambda(s):
        if s < warmup: return s / max(1, warmup)
        p = (s - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    os.makedirs(cfg["train"]["ckpt_dir"], exist_ok=True)
    step = 0
    for epoch in range(epochs):
        predictor.train()
        for batch in dl:
            feats = batch["features"].to(device, non_blocking=True)
            with torch.no_grad():
                z = text_enc(batch["captions"], device)
            z_hat = predictor(feats)
            losses = jepa_loss(z_hat, z,
                               temperature=cfg["loss"]["temperature"],
                               w_cos=cfg["loss"]["w_cos"],
                               w_nce=cfg["loss"]["w_nce"],
                               w_std=cfg["loss"]["w_std"])
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            opt.step(); sched.step()
            if step % cfg["train"]["log_every"] == 0:
                print(f"e{epoch} s{step} | loss {losses['loss']:.4f} "
                      f"cos {losses['loss_cos']:.4f} nce {losses['loss_nce']:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e}")
            step += 1
        ck = os.path.join(cfg["train"]["ckpt_dir"], f"predictor_e{epoch}.pt")
        torch.save({"cfg": cfg, "state_dict": predictor.state_dict()}, ck)
        print("saved", ck)


if __name__ == "__main__":
    main()