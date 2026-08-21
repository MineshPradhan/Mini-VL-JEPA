"""
Per-tier caption evaluation + markdown report for the guide.

Two modes:
  --oracle        decode from ground-truth LaBSE caption embeddings.
                  This is the decoder's UPPER BOUND. If oracle captions are
                  fluent and on-topic, the decoder is fine and any remaining
                  gap comes from the predictor. If oracle captions are bad,
                  the decoder itself needs more training / a stronger base.
  --preds X.npz   decode from your JEPA predictor's embeddings (the real
                  system). This is the number to show your guide.

Metrics per tier (easy / medium / hard) and overall:
  BLEU-4 (multi-reference), ROUGE-L (best over references),
  LaBSE-sim (cosine of generated caption vs closest reference - semantic
  adequacy in the same space your whole project lives in).

Usage:
  python eval_captions.py --bank curriculum/caption_bank.npz --tiers curriculum/tiers.json --ckpt checkpoints/dec_en_easy/decoder.pt --oracle --split validate --out results/easy_oracle.md
  python eval_captions.py --bank ... --tiers ... --ckpt ... --preds curriculum/preds.npz --split validate --include easy --out results/easy_pred.md
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from curriculum_common import load_bank, load_tiers
from train_decoder_en import build_model, generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--tiers", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--preds", default=None)
    ap.add_argument("--split", default="validate")
    ap.add_argument("--include", default="easy,medium,hard")
    ap.add_argument("--num_beams", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--examples_per_tier", type=int, default=8)
    ap.add_argument("--labse_model", default="sentence-transformers/LaBSE")
    ap.add_argument("--out", default="results/caption_eval.md")
    args = ap.parse_args()
    if not args.oracle and not args.preds:
        raise SystemExit("Pass --oracle or --preds X.npz")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    vids, caps, embs = load_bank(args.bank)
    tiers = load_tiers(args.tiers)
    include = set(args.include.split(","))

    refs_by_vid, first_idx, idx_by_vid = {}, {}, {}
    for i, v in enumerate(vids):
        refs_by_vid.setdefault(v, []).append(caps[i])
        first_idx.setdefault(v, i)
        idx_by_vid.setdefault(v, []).append(i)

    eval_vids = sorted(v for v, info in tiers.items()
                       if info["split"] == args.split and info["tier"] in include and v in refs_by_vid)
    if not eval_vids:
        raise SystemExit("No videos matched - check --split/--include.")

    if args.oracle:
        src = np.stack([embs[first_idx[v]] for v in eval_vids]).astype(np.float32)
        mode = "oracle (GT caption embedding)"
    else:
        z = np.load(args.preds, allow_pickle=True)
        pmap = {str(v): e for v, e in zip(z["video_ids"], z["embeddings"])}
        eval_vids = [v for v in eval_vids if v in pmap]
        src = np.stack([pmap[v] for v in eval_vids]).astype(np.float32)
        mode = f"predicted ({os.path.basename(args.preds)})"
    print(f"Evaluating {len(eval_vids)} {args.split} videos, mode = {mode}")

    ck = torch.load(args.ckpt, map_location=device)
    tok, model, proj = build_model(ck["config"], device)
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(model, ck["lora"])
    proj.load_state_dict(ck["projector"])

    gens = []
    for s in range(0, len(eval_vids), args.batch_size):
        e = F.normalize(torch.from_numpy(src[s:s + args.batch_size]), dim=-1)
        gens.extend(generate(model, proj, tok, e, device, args.num_beams, args.max_new_tokens))
        if (s // args.batch_size) % 10 == 0:
            print(f"  generated {min(s + args.batch_size, len(eval_vids))}/{len(eval_vids)}")

    del model, proj
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- metrics ------------------------------------------------------------
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    from rouge_score import rouge_scorer
    from sentence_transformers import SentenceTransformer

    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    st = SentenceTransformer(args.labse_model, device=device)
    gen_embs = st.encode(gens, batch_size=128, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)

    per_tier = {}
    for v, g, ge in zip(eval_vids, gens, gen_embs):
        refs = refs_by_vid[v]
        G = embs[idx_by_vid[v]].astype(np.float32)
        item = {
            "vid": v, "gen": g, "refs": refs,
            "rougeL": max(rs.score(r, g)["rougeL"].fmeasure for r in refs),
            "labse": float(np.max(G @ ge.astype(np.float32))),
        }
        per_tier.setdefault(tiers[v]["tier"], []).append(item)

    def bleu4(items):
        hyps = [it["gen"].lower().split() for it in items]
        refs = [[r.lower().split() for r in it["refs"]] for it in items]
        return corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method3)

    order = [t for t in ["easy", "medium", "hard"] if t in per_tier]
    lines = [f"# Caption evaluation - {args.split} split, {mode}",
             f"Checkpoint: `{args.ckpt}` | beams={args.num_beams}", "",
             "| tier | videos | BLEU-4 | ROUGE-L | LaBSE-sim |",
             "|---|---|---|---|---|"]
    all_items = []
    for t in order:
        items = per_tier[t]
        all_items += items
        lines.append(f"| {t} | {len(items)} | {bleu4(items):.4f} | "
                     f"{np.mean([i['rougeL'] for i in items]):.4f} | "
                     f"{np.mean([i['labse'] for i in items]):.4f} |")
    lines.append(f"| **all** | {len(all_items)} | {bleu4(all_items):.4f} | "
                 f"{np.mean([i['rougeL'] for i in all_items]):.4f} | "
                 f"{np.mean([i['labse'] for i in all_items]):.4f} |")

    for t in order:
        lines += ["", f"## {t} tier examples"]
        for it in per_tier[t][: args.examples_per_tier]:
            lines.append(f"- **{it['vid']}** gen: \"{it['gen']}\"  \n"
                         f"  ref: \"{it['refs'][0]}\" (LaBSE-sim {it['labse']:.3f})")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[:12]))
    print(f"\nFull report -> {args.out}")


if __name__ == "__main__":
    main()
