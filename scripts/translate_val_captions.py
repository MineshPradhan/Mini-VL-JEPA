"""
Machine-translate MSR-VTT val captions English -> multiple languages (NLLB-200).
Creates data/msrvtt_val_<lang>.json for each language.

    python scripts\translate_val_captions.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch

# language code -> (NLLB code, output suffix)
LANGUAGES = {
    "hi": "hin_Deva",   # Hindi
    "mr": "mar_Deva",   # Marathi
    "ta": "tam_Taml",   # Tamil
    "te": "tel_Telu",   # Telugu
    "bn": "ben_Beng",   # Bengali
    "de": "deu_Latn",   # German
    "zh": "zho_Hans",   # Chinese (Simplified)
    "ar": "arb_Arab",   # Arabic
}


def main():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = "facebook/nllb-200-distilled-600M"
    tok = AutoTokenizer.from_pretrained(name, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(name).to(device).eval()

    data = json.load(open("data/msrvtt_val.json"))
    uniq, seen = [], set()
    for s in data:
        if s["video"] not in seen:
            seen.add(s["video"])
            uniq.append(s)
    captions_en = [c["caption"] for c in uniq]
    print(f"{len(uniq)} unique captions; translating into {len(LANGUAGES)} languages")

    B = 16
    for lang, nllb_code in LANGUAGES.items():
        out_path = f"data/msrvtt_val_{lang}.json"
        if os.path.exists(out_path):
            print(f"[{lang}] exists, skipping")
            continue
        tgt_id = tok.convert_tokens_to_ids(nllb_code)
        out = []
        with torch.no_grad():
            for i in range(0, len(uniq), B):
                chunk = uniq[i:i + B]
                enc = tok([c["caption"] for c in chunk], return_tensors="pt",
                          padding=True, truncation=True, max_length=64).to(device)
                gen = model.generate(**enc, forced_bos_token_id=tgt_id,
                                     max_new_tokens=64, num_beams=4)
                for c, t in zip(chunk, tok.batch_decode(gen, skip_special_tokens=True)):
                    out.append({"video": c["video"], "caption": t,
                                "caption_en": c["caption"]})
            print(f"[{lang}] translated {len(out)}")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        print(f"[{lang}] saved {out_path} | sample: {out[0]['caption']}")


if __name__ == "__main__":
    main()