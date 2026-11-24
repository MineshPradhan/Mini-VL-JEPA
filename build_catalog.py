"""
Caption every held-out MSR-VTT video (Val 497 + Test 2,990) for the demo
browser and write tier.md, the per-tier guide for choosing demo clips.

Train videos are deliberately excluded: the predictor and decoder were fit on
them, so captions there would overstate what the system can do.

Uses the predictor outputs already exported by export_predicted_embeddings.py
(curriculum/preds.npz, curriculum/preds_test.npz) - identical to what the live
server computes from the video (cosine 1.0 on spot checks), so no video
decoding is needed and the whole run takes a few minutes.

Per video it stores:
  caption    decoder output with the eval decoding setting (beam 5, 30 tokens)
  cap_sim    max LaBSE cosine between that caption and the ~20 human captions
             (the same LaBSE-sim metric reported in RESULTS.md)
  grounding  LaBSE cosine between the caption and the predicted embedding
  reference  the human caption closest to the generated one

Run from the project root:
  python build_catalog.py
Outputs: web/catalog.json, tier.md
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

from train_decoder_en import build_model, generate

SPLITS = {"Val": ("validate", "Val_Captions.json", "curriculum/preds.npz"),
          "Test": ("test", "Test_Captions.json", "curriculum/preds_test.npz")}
TIERS = ("easy", "medium", "hard")


def load_preds(path):
    z = np.load(path, allow_pickle=True)
    return {str(v): e.astype(np.float32) for v, e in zip(z["video_ids"], z["embeddings"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", default="checkpoints/dec_en_all/decoder.pt")
    ap.add_argument("--tiers", default="curriculum/tiers.json")
    ap.add_argument("--captions_dir", default="MSR-VTT/Video_Captions")
    ap.add_argument("--videos_dir", default="MSR-VTT/Video_Datasets")
    ap.add_argument("--out", default="web/catalog.json")
    ap.add_argument("--md", default="tier.md")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tiers = json.load(open(args.tiers, encoding="utf-8"))

    # ---- collect held-out videos that have a file, a tier and a prediction --
    rows, refs = [], {}
    for split, (tsplit, cap_file, pred_file) in SPLITS.items():
        preds = load_preds(pred_file)
        caps = json.load(open(os.path.join(args.captions_dir, cap_file), encoding="utf-8"))
        for num, items in caps.items():
            vid = f"video{num}"
            key = f"{split}/{vid}.mp4"
            if key not in preds or key not in tiers:
                continue
            if not os.path.exists(os.path.join(args.videos_dir, split, vid + ".mp4")):
                continue
            rows.append({"id": vid, "split": split, "tier": tiers[key]["tier"],
                         "complexity": tiers[key]["score"],
                         "agreement": tiers[key]["agreement"], "z": preds[key]})
            refs[vid] = [it["caption"].strip() for it in items]
    print(f"{len(rows)} held-out videos")

    # ---- decode (same decoder + decoding setting as the reported metrics) ---
    ck = torch.load(args.decoder, map_location=device)
    from peft import set_peft_model_state_dict
    tok, model, proj = build_model(ck["config"], device)
    set_peft_model_state_dict(model, ck["lora"])
    proj.load_state_dict(ck["projector"])
    for s in range(0, len(rows), args.batch_size):
        chunk = rows[s:s + args.batch_size]
        z = torch.from_numpy(np.stack([r["z"] for r in chunk]))
        for r, c in zip(chunk, generate(model, proj, tok, z, device)):
            r["caption"] = c
        print(f"  decoded {min(s + args.batch_size, len(rows))}/{len(rows)}")
    del model, proj
    torch.cuda.empty_cache()

    # ---- score with LaBSE -----------------------------------------------------
    from sentence_transformers import SentenceTransformer
    labse = SentenceTransformer("sentence-transformers/LaBSE", device=device)
    enc = lambda t: labse.encode(t, batch_size=256, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)
    G = enc([r["caption"] for r in rows])
    flat, owner = [], []
    for i, r in enumerate(rows):
        flat += refs[r["id"]]
        owner += [i] * len(refs[r["id"]])
    print(f"encoding {len(flat)} human captions ...")
    R = enc(flat)
    owner = np.array(owner)
    for i, r in enumerate(rows):
        idx = np.where(owner == i)[0]
        sims = R[idx] @ G[i]
        j = int(np.argmax(sims))
        r["cap_sim"] = round(float(sims[j]), 4)
        r["reference"] = flat[idx[j]]
        zn = r["z"] / np.linalg.norm(r["z"])
        r["grounding"] = round(float(G[i] @ zn), 4)
        del r["z"]

    rows.sort(key=lambda r: -r["cap_sim"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"videos": rows}, f, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {args.out}")

    write_tier_md(rows, args.md)
    print(f"wrote {args.md}")


# ---------------------------------------------------------------- tier.md
def write_tier_md(rows, path):
    by = defaultdict(list)
    for r in rows:
        by[r["tier"]].append(r)

    def stats(rs):
        s = np.array([r["cap_sim"] for r in rs])
        return len(rs), s.mean(), (s >= 0.8).mean() * 100, (s >= 0.9).mean() * 100

    L = []
    L.append("# Video tiers - choosing clips for the Mini VL-JEPA demo\n")
    L.append("Every held-out MSR-VTT video (Val 497 + Test 2,990 = "
             f"{len(rows):,} clips) is pre-captioned by the real system and sorted into "
             "**easy / medium / hard**. Use this file to pick clips for the demo: open "
             "`http://127.0.0.1:8000/`, go to **Browse the dataset**, filter by tier, and type "
             "the video id into the search box (or click it in the list).\n")
    L.append("Train videos are not offered: the model was fit on them, so their captions "
             "would overstate what it can do. Every clip below is one the model never saw.\n")

    L.append("## What the tiers mean\n")
    L.append("Tiers come from `build_complexity_tiers.py` and are computed from the ~20 human "
             "captions each MSR-VTT video has, not from the model, so they are an independent "
             "measure of how hard a clip is to describe:\n")
    L.append("| signal | meaning | effect |\n|---|---|---|")
    L.append("| caption agreement | mean pairwise LaBSE cosine between the 20 human captions | people agree -> simple, unambiguous scene -> **easier** |")
    L.append("| caption length | mean words per human caption | more detail needed -> **harder** |")
    L.append("| vocabulary rarity | mean IDF of the caption words | unusual objects/actions -> **harder** |\n")
    L.append("`complexity = z(length) + z(rarity) - z(agreement)`; the dataset is split into "
             "terciles of this score.\n")
    L.append("| tier | what the clips look like | what to expect from the captions |\n|---|---|---|")
    L.append("| **easy** | one clear subject and action: someone cooking, playing a video game, singing, driving | short, accurate captions; usually match a human caption almost word for word |")
    L.append("| **medium** | a clear activity with more context: sports matches, tutorials, interviews, vlogs | the main activity is right; details are sometimes generic |")
    L.append("| **hard** | busy or unusual scenes, montages, rare objects, several things happening | the gist (person / setting / activity type) is usually right; specifics are often missed |\n")

    L.append("## How well the system does on each tier\n")
    L.append("`cap_sim` = LaBSE cosine between the generated caption and the closest of the "
             "~20 human captions (1.0 = identical meaning). Same metric as RESULTS.md.\n")
    L.append("| tier | clips | mean cap_sim | cap_sim >= 0.8 | cap_sim >= 0.9 |\n|---|---|---|---|---|")
    for t in TIERS:
        n, m, p8, p9 = stats(by[t])
        L.append(f"| {t} | {n:,} | {m:.3f} | {p8:.0f}% | {p9:.0f}% |")
    n, m, p8, p9 = stats(rows)
    L.append(f"| **all** | {n:,} | {m:.3f} | {p8:.0f}% | {p9:.0f}% |\n")
    L.append("The gradient easy > medium > hard is the curriculum result from RESULTS.md "
             "(test BLEU-4 0.546 / 0.399 / 0.304), reproduced here on every held-out clip.\n")

    L.append("## Presenting it\n")
    L.append("1. **Start with easy** clips - captions land almost exactly, which shows the core idea "
             "(video -> embedding -> text) works.")
    L.append("2. **Move to medium** - the activity is right even when wording differs from the humans'.")
    L.append("3. **Finish with one hard clip** and say why it is hard (the humans themselves disagree "
             "on it). Showing a limit honestly is more convincing than hiding it.")
    L.append("4. Switch the caption language to **हिन्दी** on any clip.")
    L.append("5. The page also shows the human captions, so the audience can judge the match.\n")
    L.append("The live page decodes three candidates (short / standard / detailed decoding styles - "
             "not the same thing as the clip tiers) and picks "
             "the one best grounded in the predicted embedding, so its caption can differ slightly "
             "from the `caption` column below, which uses the single evaluation setting.\n")

    for t in TIERS:
        rs = by[t]
        L.append(f"## {t.capitalize()} tier - recommended clips\n")
        # one clip per distinct caption, so the list shows a variety of scenes
        seen, picks = set(), []
        for r in rs:
            if r["caption"] not in seen:
                seen.add(r["caption"])
                picks.append(r)
            if len(picks) == 25:
                break
        L.append(f"Best 25 of {len(rs):,} {t} clips by cap_sim, one clip per distinct caption.\n")
        L.append("| # | video | split | model caption | closest human caption | cap_sim |\n|---|---|---|---|---|---|")
        for i, r in enumerate(picks, 1):
            L.append(f"| {i} | `{r['id']}` | {r['split']} | {r['caption']} | {r['reference']} | {r['cap_sim']:.3f} |")
        L.append("")

    L.append("## Full list\n")
    L.append("All clips with their tier and score are in `web/catalog.json` and in the "
             "page's **Browse the dataset** panel (sortable by caption match).\n")
    L.append("Regenerate this file and the catalog with `python build_catalog.py`.\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
