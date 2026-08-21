"""
Machine-translate one caption per video for a given split into 8 languages.

    python scripts\translate_split.py --split test
    python scripts\translate_split.py --split val      (already done; will skip)

Creates data/msrvtt_<split>_<lang>.json for each language. Resume-safe:
finished languages are skipped; an interrupted language resumes mid-way.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json
import torch

LANGUAGES = {
    "hi": "hin_Deva", "mr": "mar_Deva", "ta": "tam_Taml", "te": "tel_Telu",
    "bn": "ben_Beng", "de": "deu_Latn", "zh": "zho_Hans", "ar": "arb_Arab",
}


def main():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = "facebook/nllb-200-distilled-600M"
    tok = AutoTokenizer.from_pretrained(name, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(name).to(device).eval()

    with open(f"data/msrvtt_{args.split}.json", encoding="utf-8") as f:
        data = json.load(f)

    # one caption per video (first occurrence — same rule as eval)
    uniq, seen = [], set()
    for s in data:
        if s["video"] not in seen:
            seen.add(s["video"])
            uniq.append(s)
    print(f"[{args.split}] {len(uniq)} unique videos; {len(LANGUAGES)} languages")

    B = args.batch
    for lang, nllb_code in LANGUAGES.items():
        out_path = f"data/msrvtt_{args.split}_{lang}.json"
        out = []
        if os.path.exists(out_path):
            with open(out_path, encoding="utf-8") as f:
                out = json.load(f)
            if len(out) >= len(uniq):
                print(f"[{lang}] complete, skipping")
                continue
            print(f"[{lang}] resuming at {len(out)}/{len(uniq)}")

        tgt_id = tok.convert_tokens_to_ids(nllb_code)
        with torch.no_grad():
            for i in range(len(out), len(uniq), B):
                chunk = uniq[i:i + B]
                enc = tok([c["caption"] for c in chunk], return_tensors="pt",
                          padding=True, truncation=True, max_length=64).to(device)
                gen = model.generate(**enc, forced_bos_token_id=tgt_id,
                                     max_new_tokens=64, num_beams=1)  # greedy
                for c, t in zip(chunk, tok.batch_decode(gen, skip_special_tokens=True)):
                    out.append({"video": c["video"], "caption": t,
                                "caption_en": c["caption"]})
                if (i // B) % 10 == 0:
                    print(f"  [{lang}] {min(i + B, len(uniq))}/{len(uniq)}")
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(out, f, ensure_ascii=False)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        print(f"[{lang}] saved {len(out)} -> {out_path} | sample: {out[0]['caption']}")


if __name__ == "__main__":
    main()