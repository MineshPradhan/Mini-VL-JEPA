"""
Translate a random subset of TRAIN captions EN -> HI for bilingual decoder
training. Greedy decoding (fast). Resume-safe: appends in chunks.

    python scripts\translate_train_subset.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import random
import torch

SUBSET_SIZE = 20000
OUT_PATH = "data/msrvtt_train_hi_subset.json"
SEED = 42


def main():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = "facebook/nllb-200-distilled-600M"
    tok = AutoTokenizer.from_pretrained(name, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(name).to(device).eval()
    hin_id = tok.convert_tokens_to_ids("hin_Deva")

    with open("data/msrvtt_train.json", encoding="utf-8") as f:
        data = json.load(f)
    random.Random(SEED).shuffle(data)
    subset = data[:SUBSET_SIZE]

    # resume: load whatever is already translated
    done = []
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, encoding="utf-8") as f:
            done = json.load(f)
    start = len(done)
    print(f"translating {len(subset)} captions, resuming at {start}")

    B = 32
    with torch.no_grad():
        for i in range(start, len(subset), B):
            chunk = subset[i:i + B]
            enc = tok([c["caption"] for c in chunk], return_tensors="pt",
                      padding=True, truncation=True, max_length=64).to(device)
            gen = model.generate(**enc, forced_bos_token_id=hin_id,
                                 max_new_tokens=64, num_beams=1)  # greedy = fast
            for c, t in zip(chunk, tok.batch_decode(gen, skip_special_tokens=True)):
                done.append({"video": c["video"], "caption_en": c["caption"],
                             "caption_hi": t})
            if (i // B) % 10 == 0:
                print(f"  {min(i + B, len(subset))}/{len(subset)}")
                with open(OUT_PATH, "w", encoding="utf-8") as f:
                    json.dump(done, f, ensure_ascii=False)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(done, f, ensure_ascii=False)
    print(f"saved {len(done)} pairs -> {OUT_PATH}")
    print("sample:", done[0]["caption_en"], "->", done[0]["caption_hi"])


if __name__ == "__main__":
    main()