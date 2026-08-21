"""
Token-generation baseline: autoregressive captioner on the SAME cached
V-JEPA 2 features. Control arm for Experiment 1.

Train:    python -m src.baseline --config configs/default.yaml
Generate: python -m src.baseline --config configs/default.yaml --generate --ckpt checkpoints/baseline_e4.pt
"""
import argparse, math, os
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.dataset import CachedFeatureDataset, cached_collate


class TokenBaselineHead(nn.Module):
    """Visual-prefix autoregressive captioner (distilgpt2 backbone)."""

    def __init__(self, video_dim=1024, prefix_len=16, lm_name="distilgpt2"):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(lm_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(lm_name)
        d = self.lm.config.n_embd
        self.pool = nn.Sequential(nn.Linear(video_dim, d), nn.GELU(), nn.Linear(d, d))
        self.prefix_queries = nn.Parameter(torch.randn(1, prefix_len, d) * 0.02)
        self.xattn = nn.MultiheadAttention(d, 8, batch_first=True)

    def _prefix(self, feats):
        mem = self.pool(feats)
        q = self.prefix_queries.expand(mem.size(0), -1, -1)
        p, _ = self.xattn(q, mem, mem)
        return p

    def forward(self, feats, captions, device):
        prefix = self._prefix(feats)
        tok = self.tokenizer(captions, return_tensors="pt", padding=True,
                             truncation=True, max_length=32).to(device)
        txt = self.lm.get_input_embeddings()(tok.input_ids)
        inputs = torch.cat([prefix, txt], dim=1)
        attn = torch.cat([torch.ones(prefix.shape[:2], device=device),
                          tok.attention_mask], dim=1)
        labels = torch.cat(
            [torch.full(prefix.shape[:2], -100, dtype=torch.long, device=device),
             tok.input_ids.masked_fill(tok.attention_mask == 0, -100)], dim=1)
        return self.lm(inputs_embeds=inputs, attention_mask=attn, labels=labels)

    @torch.no_grad()
    def generate(self, feats, max_new_tokens=25):
        prefix = self._prefix(feats)
        attn = torch.ones(prefix.shape[:2], device=feats.device)
        ids = self.lm.generate(inputs_embeds=prefix, attention_mask=attn,
                               max_new_tokens=max_new_tokens, num_beams=3,
                               pad_token_id=self.tokenizer.eos_token_id)
        return [t.strip() for t in self.tokenizer.batch_decode(ids, skip_special_tokens=True)]


def train(cfg, device):
    head = TokenBaselineHead(video_dim=cfg["predictor"]["video_dim"]).to(device)
    n = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"[baseline] trainable params: {n/1e6:.1f}M")

    ds = CachedFeatureDataset(cfg["data"]["train_annotations"], "data/features/train")
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0,
                    collate_fn=cached_collate, pin_memory=True, drop_last=True)

    epochs = 5
    opt = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],
                            lr=1e-4, weight_decay=0.05)
    total = epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / max(1, total))))

    os.makedirs(cfg["train"]["ckpt_dir"], exist_ok=True)
    step = 0
    for epoch in range(epochs):
        head.train()
        for batch in dl:
            feats = batch["features"].to(device, non_blocking=True)
            out = head(feats, batch["captions"], device)
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step(); sched.step()
            if step % 50 == 0:
                print(f"e{epoch} s{step} | lm_loss {out.loss.item():.4f}")
            step += 1
        ck = os.path.join(cfg["train"]["ckpt_dir"], f"baseline_e{epoch}.pt")
        torch.save(head.state_dict(), ck)
        print("saved", ck)


def generate_val(cfg, ckpt, device):
    import json, time
    head = TokenBaselineHead(video_dim=cfg["predictor"]["video_dim"]).to(device)
    head.load_state_dict(torch.load(ckpt, map_location=device))
    head.eval()

    ds = CachedFeatureDataset(cfg["data"]["val_annotations"], "data/features/val")
    uniq, seen = [], set()
    for s in ds.samples:
        if s["video"] not in seen:
            seen.add(s["video"]); uniq.append(s)
    ds.samples = uniq
    dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=cached_collate)

    outputs, t0 = [], time.time()
    for batch in dl:
        caps = head.generate(batch["features"].to(device))
        for vid, ref, hyp in zip(batch["video_ids"], batch["captions"], caps):
            outputs.append({"video": vid, "reference": ref, "generated": hyp})
    dt = time.time() - t0
    print(f"generated {len(outputs)} captions in {dt:.1f}s "
          f"({1000*dt/len(outputs):.0f} ms/caption)")
    with open("baseline_val_captions.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    for o in outputs[:5]:
        print("  REF:", o["reference"]); print("  GEN:", o["generated"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--ckpt", default="checkpoints/baseline_e4.pt")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.generate:
        generate_val(cfg, args.ckpt, device)
    else:
        train(cfg, device)


if __name__ == "__main__":
    main()