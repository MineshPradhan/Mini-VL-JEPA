"""
Embedding->text decoder: mT5-small + LoRA, trained to reconstruct captions
from their own LaBSE embeddings (text-only autoencoding).

Train:    python -m src.decoder --captions data/msrvtt_train.json
Generate: python -m src.decoder --generate --ckpt checkpoints/decoder.pt --predictor checkpoints/predictor_e5.pt
"""
import argparse, json, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class CaptionOnlyDataset(Dataset):
    def __init__(self, captions_json):
        with open(captions_json, encoding="utf-8") as f:
            data = json.load(f)
        self.captions = [d["caption"] for d in data]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, i):
        return self.captions[i]


class EmbToTextDecoder(nn.Module):
    """Projects a sentence embedding into k soft prefix tokens for mT5."""

    def __init__(self, base="google/mt5-small", emb_dim=768, k_prefix=8, lora_r=16):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoTokenizer, MT5ForConditionalGeneration

        self.tokenizer = AutoTokenizer.from_pretrained(base)
        model = MT5ForConditionalGeneration.from_pretrained(base)
        lcfg = LoraConfig(r=lora_r, lora_alpha=32, target_modules=["q", "v"],
                          task_type="SEQ_2_SEQ_LM")
        self.model = get_peft_model(model, lcfg)
        d_model = self.model.config.d_model
        self.k = k_prefix
        self.prefix_proj = nn.Sequential(nn.Linear(emb_dim, d_model * k_prefix), nn.Tanh())

    def _inputs(self, emb, lang_prompt):
        B = emb.size(0)
        prefix = self.prefix_proj(emb).view(B, self.k, -1)
        prompt = self.tokenizer([lang_prompt] * B, return_tensors="pt",
                                padding=True).to(emb.device)
        prompt_emb = self.model.get_input_embeddings()(prompt.input_ids)
        inputs = torch.cat([prefix, prompt_emb], dim=1)
        attn = torch.ones(inputs.shape[:2], device=emb.device)
        return inputs, attn

    def forward(self, emb, labels, lang_prompt="caption in English: "):
        inputs, attn = self._inputs(emb, lang_prompt)
        return self.model(inputs_embeds=inputs, attention_mask=attn, labels=labels)

    @torch.no_grad()
    def generate(self, emb, lang_prompt="caption in English: ", max_new_tokens=30):
        inputs, attn = self._inputs(emb, lang_prompt)
        ids = self.model.generate(inputs_embeds=inputs, attention_mask=attn,
                                  max_new_tokens=max_new_tokens, num_beams=3,
                                  no_repeat_ngram_size=3,
                                  repetition_penalty=1.3)
        return [t.strip() for t in self.tokenizer.batch_decode(ids, skip_special_tokens=True)]


def train(args, device):
    """Bilingual: reconstruct EN caption OR its HI translation from the SAME
    LaBSE embedding of the English caption, with matching language prompt."""
    import random as _random

    emb_cache = "data/caption_emb_bilingual.pt"
    pairs_path = "data/msrvtt_train_hi_subset.json"
    with open(pairs_path, encoding="utf-8") as f:
        pairs = json.load(f)
    print(f"bilingual training on {len(pairs)} EN/HI caption pairs")

    # ---- precompute LaBSE embeddings of the ENGLISH captions once ----
    if not os.path.exists(emb_cache):
        from sentence_transformers import SentenceTransformer
        labse = SentenceTransformer("sentence-transformers/LaBSE").eval()
        caps = [p["caption_en"] for p in pairs]
        chunks = []
        with torch.no_grad():
            for i in range(0, len(caps), 512):
                e = labse.encode(caps[i:i+512], convert_to_tensor=True,
                                 normalize_embeddings=True).cpu().clone()
                chunks.append(e)
        torch.save(torch.cat(chunks), emb_cache)
        del labse
        torch.cuda.empty_cache()
        print("saved", emb_cache)

    all_emb = torch.load(emb_cache)
    assert len(pairs) == all_emb.size(0)

    dec = EmbToTextDecoder().to(device)
    n = sum(p.numel() for p in dec.parameters() if p.requires_grad)
    print(f"[decoder] trainable params: {n/1e6:.1f}M")
    opt = torch.optim.AdamW([p for p in dec.parameters() if p.requires_grad], lr=3e-4)

    rng = _random.Random(0)
    B = 16
    step = 0
    for epoch in range(args.epochs):
        perm = torch.randperm(len(pairs))
        for i in range(0, len(pairs) - B, B):
            idx = perm[i:i+B]
            use_hi = rng.random() < 0.5           # whole batch EN or HI
            if use_hi:
                captions = [pairs[j]["caption_hi"] for j in idx.tolist()]
                prompt = "caption in Hindi: "
            else:
                captions = [pairs[j]["caption_en"] for j in idx.tolist()]
                prompt = "caption in English: "
            emb = all_emb[idx].to(device)
            labels = dec.tokenizer(captions, return_tensors="pt", padding=True,
                                   truncation=True, max_length=40).input_ids.to(device)
            labels[labels == dec.tokenizer.pad_token_id] = -100
            out = dec(emb, labels=labels, lang_prompt=prompt)
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            opt.step()
            if step % 100 == 0:
                print(f"e{epoch} s{step} | dec_loss {out.loss.item():.4f} [{'HI' if use_hi else 'EN'}]")
            step += 1
        torch.save(dec.state_dict(), args.out)
        print("saved", args.out)


def generate_from_predictor(args, device):
    import time, yaml
    from src.dataset import CachedFeatureDataset, cached_collate
    from src.model import JEPAPredictor, PredictorConfig

    cfg = yaml.safe_load(open("configs/default.yaml"))
    predictor = JEPAPredictor(PredictorConfig(**cfg["predictor"])).to(device)
    predictor.load_state_dict(torch.load(args.predictor, map_location=device)["state_dict"])
    predictor.eval()
    dec = EmbToTextDecoder().to(device)
    dec.load_state_dict(torch.load(args.ckpt, map_location=device))
    dec.eval()

    ds = CachedFeatureDataset(cfg["data"]["val_annotations"], "data/features/val")
    uniq, seen = [], set()
    for s in ds.samples:
        if s["video"] not in seen:
            seen.add(s["video"]); uniq.append(s)
    ds.samples = uniq
    dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=cached_collate)

    outputs, t0 = [], time.time()
    with torch.no_grad():
        for batch in dl:
            z_hat = predictor(batch["features"].to(device))
            caps_en = dec.generate(z_hat, "caption in English: ")
            caps_hi = dec.generate(z_hat, "caption in Hindi: ")
            for vid, ref, en, hi in zip(batch["video_ids"], batch["captions"], caps_en, caps_hi):
                outputs.append({"video": vid, "reference": ref,
                                "generated": en, "generated_hi": hi})
    dt = time.time() - t0
    print(f"generated {len(outputs)} caption pairs in {dt:.1f}s")
    with open("jepa_val_captions.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    for o in outputs[:5]:
        print("  REF:", o["reference"])
        print("  GEN:", o["generated"])
        print("  HIN:", o["generated_hi"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", default="data/msrvtt_train.json")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--out", default="checkpoints/decoder.pt")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--ckpt", default="checkpoints/decoder.pt")
    ap.add_argument("--predictor", default="checkpoints/predictor_e5.pt")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.generate:
        generate_from_predictor(args, device)
    else:
        train(args, device)


if __name__ == "__main__":
    main()