"""
Evaluate one checkpoint across all translated caption sets + English.
    python scripts\eval_all_languages.py --ckpt checkpoints\predictor_e5.pt
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import glob
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import CachedFeatureDataset, cached_collate
from src.model import FrozenTextTarget, JEPAPredictor, PredictorConfig
from src.eval import retrieval_metrics

LANG_NAMES = {"en": "English", "hi": "Hindi", "mr": "Marathi", "ta": "Tamil",
              "te": "Telugu", "bn": "Bengali", "de": "German",
              "zh": "Chinese", "ar": "Arabic"}

def evaluate(predictor, text_enc, ann, device, features_dir):
    ds = CachedFeatureDataset(ann, features_dir)
    uniq, seen = [], set()
    for s in ds.samples:
        if s["video"] not in seen:
            seen.add(s["video"])
            uniq.append(s)
    ds.samples = uniq
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=cached_collate)
    Z_hat, Z = [], []
    with torch.no_grad():
        for batch in dl:
            Z_hat.append(predictor(batch["features"].to(device)).cpu())
            Z.append(text_enc(batch["captions"], device).cpu())
    return retrieval_metrics(torch.cat(Z_hat), torch.cat(Z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text_enc = FrozenTextTarget(cfg["text_encoder"])
    predictor = JEPAPredictor(PredictorConfig(**cfg["predictor"])).to(device)
    predictor.load_state_dict(torch.load(args.ckpt, map_location=device)["state_dict"])
    predictor.eval()

    if args.split == "val":
        en_ann = cfg["data"]["val_annotations"]
    else:
        en_ann = "data/msrvtt_test.json"
    features_dir = f"data/features/{args.split}"

    rows = [("en", en_ann)]
    for f in sorted(glob.glob(f"data/msrvtt_{args.split}_*.json")):
        lang = f.split("_")[-1].split(".")[0]
        if lang != "en":
            rows.append((lang, f))

    print(f"\n{'Language':<12}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'MedR':>6}")
    print("-" * 39)
    results = {}
    for lang, ann in rows:
        m = evaluate(predictor, text_enc, ann, device, features_dir)
        results[lang] = m
        print(f"{LANG_NAMES.get(lang, lang):<12}"
              f"{m['R@1']:>7.2f}{m['R@5']:>7.2f}{m['R@10']:>7.2f}{m['MedR']:>6.0f}")

    import json
    out_path = f"results_multilingual_{args.split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()