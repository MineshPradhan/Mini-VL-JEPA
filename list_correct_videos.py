"""
Rank per-video system performance and list the videos the pipeline gets right.

For each video in the chosen split/tier(s), reports:
  - R@1 / rank : does the predictor's embedding retrieve the video's own
                 caption first among ALL validation videos' captions
                 (one caption per video - same protocol as src/eval.py)?
  - pred_cos   : cosine between the predicted embedding and the closest
                 ground-truth caption embedding (predictor quality)
  - generated  : the caption the decoder produces from the predicted embedding
  - cap_sim    : LaBSE similarity of that generated caption to the references
                 (full-system quality)

Output: markdown report sorted best-first by cap_sim, with a shortlist of
"safe demo videos" (R@1 correct AND cap_sim >= --min_sim) at the top.

Usage:
  python list_correct_videos.py --bank curriculum/caption_bank.npz --tiers curriculum/tiers.json --preds curriculum/preds.npz --ckpt checkpoints/dec_en_easy_pred/decoder.pt --split validate --include easy --out results/easy_correct_videos.md
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
    ap.add_argument("--preds", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="validate")
    ap.add_argument("--include", default="easy")
    ap.add_argument("--min_sim", type=float, default=0.80,
                    help="cap_sim threshold for the safe-demo shortlist")
    ap.add_argument("--num_beams", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--labse_model", default="sentence-transformers/LaBSE")
    ap.add_argument("--out", default="results/correct_videos.md")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    vids, caps, embs = load_bank(args.bank)
    tiers = load_tiers(args.tiers)
    z = np.load(args.preds, allow_pickle=True)
    pmap = {str(v): e for v, e in zip(z["video_ids"], z["embeddings"])}

    refs_by_vid, idx_by_vid, first_idx = {}, {}, {}
    for i, v in enumerate(vids):
        refs_by_vid.setdefault(v, []).append(caps[i])
        idx_by_vid.setdefault(v, []).append(i)
        first_idx.setdefault(v, i)

    include = set(args.include.split(","))

    # retrieval candidate pool: EVERY video in the split (all tiers),
    # one caption each - the same protocol as your R@1 numbers
    pool = sorted(v for v, info in tiers.items()
                  if info["split"] == args.split and v in pmap and v in refs_by_vid)
    targets = [v for v in pool if tiers[v]["tier"] in include]
    if not targets:
        raise SystemExit("No videos matched - check --split/--include.")
    print(f"{len(targets)} target videos ({args.include}) against a pool of {len(pool)} candidates")

    C = np.stack([embs[first_idx[v]] for v in pool]).astype(np.float32)      # (M, 768)
    P = np.stack([pmap[v] for v in targets]).astype(np.float32)              # (N, 768)
    pool_pos = {v: j for j, v in enumerate(pool)}
    S = P @ C.T                                                              # (N, M)

    rank, pred_cos = [], []
    for i, v in enumerate(targets):
        own = S[i, pool_pos[v]]
        rank.append(int((S[i] > own).sum()) + 1)
        G = embs[idx_by_vid[v]].astype(np.float32)
        pred_cos.append(float(np.max(G @ P[i])))

    # ---- generate captions from the predicted embeddings ---------------------
    ck = torch.load(args.ckpt, map_location=device)
    tok, model, proj = build_model(ck["config"], device)
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(model, ck["lora"])
    proj.load_state_dict(ck["projector"])

    gens = []
    for s in range(0, len(targets), args.batch_size):
        e = F.normalize(torch.from_numpy(P[s:s + args.batch_size]), dim=-1)
        gens.extend(generate(model, proj, tok, e, device, args.num_beams, args.max_new_tokens))
    del model, proj
    if device == "cuda":
        torch.cuda.empty_cache()

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(args.labse_model, device=device)
    gen_embs = st.encode(gens, batch_size=128, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)

    rows = []
    for i, v in enumerate(targets):
        G = embs[idx_by_vid[v]].astype(np.float32)
        sims = G @ gen_embs[i].astype(np.float32)
        j = int(np.argmax(sims))
        rows.append({
            "vid": v, "gen": gens[i], "ref": refs_by_vid[v][j],
            "cap_sim": float(sims[j]), "r1": rank[i] == 1,
            "rank": rank[i], "pred_cos": pred_cos[i],
        })
    rows.sort(key=lambda r: r["cap_sim"], reverse=True)

    n = len(rows)
    r1_pct = 100.0 * sum(r["r1"] for r in rows) / n
    safe = [r for r in rows if r["r1"] and r["cap_sim"] >= args.min_sim]

    lines = [f"# Per-video system performance - {args.split} split, tiers: {args.include}",
             f"Decoder: `{args.ckpt}` | predictions: `{args.preds}`", "",
             f"- videos: {n} | retrieval R@1 (pool of {len(pool)}): **{r1_pct:.1f}%** | "
             f"mean cap_sim: {np.mean([r['cap_sim'] for r in rows]):.3f}", "",
             f"## Safe demo shortlist - R@1 correct AND cap_sim >= {args.min_sim} ({len(safe)} videos)", ""]
    for i, r in enumerate(safe, 1):
        lines.append(f'{i}. **{r["vid"]}** - "{r["gen"]}" (sim {r["cap_sim"]:.3f})')
    lines += ["", "## Full ranking (best first)", "",
              "| # | video | cap_sim | R@1 | rank | pred_cos | generated | closest reference |",
              "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        mark = "Y" if r["r1"] else "-"
        lines.append(f'| {i} | {r["vid"]} | {r["cap_sim"]:.3f} | {mark} | {r["rank"]} | '
                     f'{r["pred_cos"]:.3f} | {r["gen"]} | {r["ref"]} |')

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nretrieval R@1 on this tier: {r1_pct:.1f}%  |  safe demo videos: {len(safe)}")
    print("\nTop 15 by caption quality:")
    for r in rows[:15]:
        mark = "R@1" if r["r1"] else f"rank {r['rank']}"
        print(f'  {r["vid"]} [{mark}]  "{r["gen"]}"  (sim {r["cap_sim"]:.3f})')
    print(f"\nFull report -> {args.out}")


if __name__ == "__main__":
    main()
