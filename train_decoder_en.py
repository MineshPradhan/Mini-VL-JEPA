"""
Curriculum English caption decoder for Mini VL-JEPA.

Trains a seq2seq decoder (default mT5-small + LoRA) to generate an English
caption from a 768-d LaBSE sentence embedding. Two things make it robust to
imperfect *predicted* embeddings at inference time:

  1. CapDec-style noise injection: during training the ground-truth caption
     embedding is perturbed with Gaussian noise and re-normalised, so the
     decoder learns to decode a *region* of embedding space, not a point.
  2. Optional --pred_npz mixing: some training samples use the JEPA
     predictor's actual output embedding for that video (exported by
     export_predicted_embeddings.py), closing the train/inference gap.

Training is text/embedding-only: no video decoding, so it is fast even on a
6 GB RTX 3050 (roughly 5-10 min per epoch on the easy tier).

Stage 1 (easy tier, noise only):
  python train_decoder_en.py --bank curriculum/caption_bank.npz --tiers curriculum/tiers.json --include easy --epochs 8 --out checkpoints/dec_en_easy

Stage 1b (adapt to real predictor outputs):
  python train_decoder_en.py --bank curriculum/caption_bank.npz --tiers curriculum/tiers.json --include easy --init_from checkpoints/dec_en_easy/decoder.pt --pred_npz curriculum/preds.npz --pred_ratio 0.5 --lr 1e-4 --epochs 4 --out checkpoints/dec_en_easy_pred

Stage 2 / 3: repeat with --include easy,medium then easy,medium,hard,
always passing --init_from the previous stage's decoder.pt.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from curriculum_common import load_bank, load_tiers


class EmbToPrefix(nn.Module):
    """768-d sentence embedding -> K pseudo encoder tokens for the decoder."""

    def __init__(self, emb_dim, d_model, k):
        super().__init__()
        self.k, self.d = k, d_model
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 1024), nn.GELU(), nn.Linear(1024, k * d_model)
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, e):
        x = self.net(e).view(e.size(0), self.k, self.d)
        return self.ln(x)


class CapEmbDataset(Dataset):
    """(embedding, caption) pairs; noise / predicted-embedding mixing happens here."""

    def __init__(self, embs, captions, video_ids, noise_std=0.0, preds=None, pred_ratio=0.0, train=True):
        self.embs = embs  # np.float16 (N, 768), L2-normalised
        self.captions = captions
        self.video_ids = video_ids
        self.noise_std = noise_std
        self.preds = preds or {}
        self.pred_ratio = pred_ratio
        self.train = train

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, i):
        vid = self.video_ids[i]
        if self.train and self.pred_ratio > 0 and vid in self.preds and random.random() < self.pred_ratio:
            e = torch.from_numpy(self.preds[vid].astype(np.float32))
            e = F.normalize(e, dim=-1)  # predictor output, used as-is (no extra noise)
        else:
            e = torch.from_numpy(self.embs[i].astype(np.float32))
            e = F.normalize(e, dim=-1)
            if self.train and self.noise_std > 0:
                e = F.normalize(e + self.noise_std * torch.randn_like(e), dim=-1)
        return {"emb": e, "caption": self.captions[i]}


class Collate:
    """Top-level class (picklable) per the Windows DataLoader lesson."""

    def __init__(self, tokenizer, max_len):
        self.tok = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        embs = torch.stack([b["emb"] for b in batch])
        enc = self.tok(
            [b["caption"] for b in batch],
            padding=True, truncation=True, max_length=self.max_len, return_tensors="pt",
        )
        labels = enc.input_ids.clone()
        labels[labels == self.tok.pad_token_id] = -100
        return embs, labels


def load_preds(path):
    z = np.load(path, allow_pickle=True)
    return {str(v): e for v, e in zip(z["video_ids"], z["embeddings"])}


def build_model(cfg, device):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    base = AutoModelForSeq2SeqLM.from_pretrained(cfg["base_model"])
    d_model = base.config.d_model
    lcfg = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.05,
        target_modules=["q", "v"], bias="none", task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(base, lcfg).to(device)
    proj = EmbToPrefix(cfg["emb_dim"], d_model, cfg["prefix_tokens"]).to(device)
    return tok, model, proj


@torch.no_grad()
def generate(model, proj, tok, embs, device, num_beams=5, max_new_tokens=30):
    from transformers.modeling_outputs import BaseModelOutput

    model.eval(); proj.eval()
    prefix = proj(embs.to(device))
    enc = BaseModelOutput(last_hidden_state=prefix)
    out = model.generate(
        encoder_outputs=enc, num_beams=num_beams, max_new_tokens=max_new_tokens,
        no_repeat_ngram_size=3, length_penalty=1.0, early_stopping=True,
    )
    return [t.strip() for t in tok.batch_decode(out, skip_special_tokens=True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--tiers", required=True)
    ap.add_argument("--include", default="easy", help="comma list: easy,medium,hard")
    ap.add_argument("--base_model", default="google/mt5-small",
                    help="google/mt5-small keeps the Hindi path open; try google/flan-t5-base for max English fluency")
    ap.add_argument("--prefix_tokens", type=int, default=16)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--noise_std", type=float, default=0.04,
                    help="Gaussian noise on GT embeddings; set from export_predicted_embeddings.py's suggestion")
    ap.add_argument("--pred_npz", default=None, help="predicted embeddings from export_predicted_embeddings.py")
    ap.add_argument("--pred_ratio", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_cap_len", type=int, default=48)
    ap.add_argument("--init_from", default=None, help="decoder.pt from a previous stage")
    ap.add_argument("--out", default="checkpoints/dec_en_easy")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- data ---------------------------------------------------------------
    vids, caps, embs = load_bank(args.bank)
    tiers = load_tiers(args.tiers)
    include = set(args.include.split(","))

    def pick(split):
        idx = [i for i, v in enumerate(vids)
               if v in tiers and tiers[v]["split"] == split and tiers[v]["tier"] in include]
        return (embs[idx], [caps[i] for i in idx], [vids[i] for i in idx])

    tr = pick("train")
    va = pick("validate")
    print(f"tiers={sorted(include)}  train pairs={len(tr[1])}  val pairs={len(va[1])}")
    if not tr[1]:
        raise SystemExit("No training pairs found - check --tiers/--include and split assignment.")

    preds = load_preds(args.pred_npz) if args.pred_npz else None
    if preds:
        cov = sum(1 for v in set(tr[2]) if v in preds)
        print(f"predicted embeddings cover {cov}/{len(set(tr[2]))} train videos (ratio={args.pred_ratio})")

    cfg = {"base_model": args.base_model, "prefix_tokens": args.prefix_tokens,
           "lora_r": args.lora_r, "lora_alpha": 2 * args.lora_r,
           "emb_dim": embs.shape[1], "max_cap_len": args.max_cap_len}
    tok, model, proj = build_model(cfg, device)

    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    if args.init_from:
        ck = torch.load(args.init_from, map_location=device)
        set_peft_model_state_dict(model, ck["lora"])
        proj.load_state_dict(ck["projector"])
        print(f"Initialised from {args.init_from}")

    collate = Collate(tok, args.max_cap_len)
    tr_ds = CapEmbDataset(*tr, noise_std=args.noise_std, preds=preds, pred_ratio=args.pred_ratio, train=True)
    va_ds = CapEmbDataset(*va, train=False)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    params = list(proj.parameters()) + [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    print(f"trainable parameters: {n_train / 1e6:.1f}M")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    from transformers import get_cosine_schedule_with_warmup
    total = args.epochs * len(tr_dl)
    sched = get_cosine_schedule_with_warmup(opt, min(200, total // 10), total)

    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    from transformers.modeling_outputs import BaseModelOutput
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **k: x

    best = float("inf")
    sample_embs = torch.from_numpy(va[0][:5].astype(np.float32)) if len(va[1]) >= 5 else None

    for ep in range(1, args.epochs + 1):
        model.train(); proj.train()
        tot, nb = 0.0, 0
        for embs_b, labels in tqdm(tr_dl, desc=f"epoch {ep}/{args.epochs}"):
            embs_b, labels = embs_b.to(device), labels.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                prefix = proj(embs_b)
                out = model(encoder_outputs=BaseModelOutput(last_hidden_state=prefix), labels=labels)
                loss = out.loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            tot += loss.item(); nb += 1

        # validation loss on clean GT embeddings (oracle conditions)
        model.eval(); proj.eval()
        vtot, vnb = 0.0, 0
        with torch.no_grad():
            for embs_b, labels in va_dl:
                embs_b, labels = embs_b.to(device), labels.to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    out = model(encoder_outputs=BaseModelOutput(last_hidden_state=proj(embs_b)), labels=labels)
                vtot += out.loss.item(); vnb += 1
        vloss = vtot / max(vnb, 1)
        print(f"epoch {ep}: train loss {tot / max(nb, 1):.4f} | val loss {vloss:.4f}")

        if sample_embs is not None:
            gens = generate(model, proj, tok, sample_embs, device)
            for g, ref in zip(gens, va[1][:5]):
                print(f'    gen: "{g}"   |   ref: "{ref}"')

        state = {"projector": proj.state_dict(), "lora": get_peft_model_state_dict(model),
                 "config": cfg, "val_loss": vloss, "epoch": ep, "include": sorted(include)}
        torch.save(state, os.path.join(args.out, "last.pt"))
        if vloss < best:
            best = vloss
            torch.save(state, os.path.join(args.out, "decoder.pt"))
            print(f"  saved best -> {os.path.join(args.out, 'decoder.pt')}")

    print("done.")


if __name__ == "__main__":
    main()
