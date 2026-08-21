"""
Step 1 of the caption curriculum: score every video's complexity and split
the dataset into easy / medium / hard tiers (guide's low / high / difficult).

Complexity signals (all computed from the captions, so this is fast):
  - agreement : mean pairwise cosine similarity between the video's LaBSE
                caption embeddings. High agreement = everyone describes the
                video the same way = simple, unambiguous content.  (easier)
  - mean_len  : mean caption length in tokens.                      (harder)
  - rarity    : mean IDF of caption tokens (rare vocabulary).       (harder)
  - motion    : optional visual signal, mean frame difference.      (harder)

complexity = z(mean_len) + z(rarity) - z(agreement) [+ z(motion)]
Tiers = terciles of this score.

Also saves curriculum/caption_bank.npz (video_ids, captions, LaBSE
embeddings fp16) which the training and eval scripts reuse.

Usage (from your repo root, with curriculum_common.py next to it):
  python build_complexity_tiers.py --captions data/msrvtt_train.json data/msrvtt_val.json data/msrvtt_test.json
Optional visual signal (slower, needs the video files):
  add --use_motion --videos_dir <folder with the .mp4 files>
"""

import argparse
import json
import os
from collections import Counter

import numpy as np

from curriculum_common import assign_split, load_captions_multi, save_bank, simple_tokens


def compute_motion(videos_dir, video_ids, max_frames=8):
    """Mean absolute frame difference on a few low-res frames per video."""
    from decord import VideoReader, cpu  # imported lazily

    exts = [".mp4", ".avi", ".mkv", ".webm"]
    scores = {}
    for i, vid in enumerate(video_ids):
        path = None
        for e in exts:
            p = os.path.join(videos_dir, vid + e)
            if os.path.exists(p):
                path = p
                break
        if path is None:
            scores[vid] = np.nan
            continue
        try:
            vr = VideoReader(path, ctx=cpu(0), width=160, height=96)
            n = len(vr)
            idx = np.linspace(0, n - 1, num=min(max_frames, n)).astype(int)
            fr = vr.get_batch(list(idx)).asnumpy().astype(np.float32) / 255.0
            del vr  # release decord handle before anything touches the file (Windows)
            scores[vid] = float(np.abs(np.diff(fr, axis=0)).mean()) if len(fr) > 1 else 0.0
        except Exception as ex:
            print(f"  [motion] skipping {vid}: {ex}")
            scores[vid] = np.nan
        if (i + 1) % 500 == 0:
            print(f"  [motion] {i + 1}/{len(video_ids)} videos")
    vals = np.array([v for v in scores.values() if not np.isnan(v)])
    med = float(np.median(vals)) if len(vals) else 0.0
    return {k: (med if np.isnan(v) else v) for k, v in scores.items()}


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 1e-8 else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True, nargs="+",
                    help="one or more caption JSONs; split inferred from each filename (train/val/test)")
    ap.add_argument("--out_dir", default="curriculum")
    ap.add_argument("--labse_model", default="sentence-transformers/LaBSE")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default=None, help="cuda / cpu (auto if omitted)")
    ap.add_argument("--use_motion", action="store_true")
    ap.add_argument("--videos_dir", default=None)
    ap.add_argument("--max_frames", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    caps, splits = load_captions_multi(args.captions)
    video_ids = sorted(caps.keys())
    print(f"Loaded {sum(len(v) for v in caps.values())} captions for {len(video_ids)} videos total")
    print("Sample parsed pairs (sanity check - should be real video ids + English captions):")
    for vid in video_ids[:3]:
        print(f'  {vid}: "{caps[vid][0]}"')

    # ---- flatten and encode with LaBSE ------------------------------------
    flat_vids, flat_caps = [], []
    for vid in video_ids:
        for c in caps[vid]:
            flat_vids.append(vid)
            flat_caps.append(c)

    import torch
    from sentence_transformers import SentenceTransformer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Encoding {len(flat_caps)} captions with LaBSE on {device} ...")
    st = SentenceTransformer(args.labse_model, device=device)
    embs = st.encode(
        flat_caps,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    bank_path = os.path.join(args.out_dir, "caption_bank.npz")
    save_bank(bank_path, flat_vids, flat_caps, embs)
    print(f"Saved caption bank -> {bank_path}")

    # ---- per-video features -----------------------------------------------
    idx_by_vid = {}
    for i, vid in enumerate(flat_vids):
        idx_by_vid.setdefault(vid, []).append(i)

    df = Counter()
    for c in flat_caps:
        df.update(set(simple_tokens(c)))
    n_caps_total = len(flat_caps)

    agreement, mean_len, rarity = [], [], []
    for vid in video_ids:
        idx = idx_by_vid[vid]
        E = embs[idx]
        n = len(idx)
        if n > 1:
            G = E @ E.T
            agreement.append((G.sum() - n) / (n * (n - 1)))
        else:
            agreement.append(1.0)
        lens, idfs = [], []
        for i in idx:
            toks = simple_tokens(flat_caps[i])
            lens.append(len(toks))
            if toks:
                idfs.append(np.mean([np.log(n_caps_total / (1 + df[t])) for t in toks]))
        mean_len.append(float(np.mean(lens)) if lens else 0.0)
        rarity.append(float(np.mean(idfs)) if idfs else 0.0)

    score = zscore(mean_len) + zscore(rarity) - zscore(agreement)

    motion = None
    if args.use_motion:
        if not args.videos_dir:
            raise SystemExit("--use_motion needs --videos_dir")
        print("Computing motion scores (this reads every video, takes a while) ...")
        mscores = compute_motion(args.videos_dir, video_ids, args.max_frames)
        motion = [mscores[v] for v in video_ids]
        score = score + zscore(motion)

    lo, hi = np.quantile(score, [1 / 3, 2 / 3])
    tiers = {}
    for i, vid in enumerate(video_ids):
        t = "easy" if score[i] <= lo else ("medium" if score[i] <= hi else "hard")
        tiers[vid] = {
            "tier": t,
            "split": assign_split(vid, splits),
            "score": round(float(score[i]), 4),
            "agreement": round(float(agreement[i]), 4),
            "mean_len": round(float(mean_len[i]), 2),
            "rarity": round(float(rarity[i]), 4),
        }
        if motion is not None:
            tiers[vid]["motion"] = round(float(motion[i]), 4)

    tiers_path = os.path.join(args.out_dir, "tiers.json")
    with open(tiers_path, "w", encoding="utf-8") as f:
        json.dump(tiers, f, indent=1)
    print(f"Saved tiers -> {tiers_path}\n")

    # ---- summary -----------------------------------------------------------
    counts = {}
    for v in tiers.values():
        counts.setdefault(v["tier"], Counter())[v["split"]] += 1
    print("tier   " + "".join(f"{s:>10}" for s in ["train", "validate", "test"]))
    for t in ["easy", "medium", "hard"]:
        c = counts.get(t, Counter())
        print(f"{t:<6} " + "".join(f"{c.get(s, 0):>10}" for s in ["train", "validate", "test"]))
    print("\nExamples:")
    for t in ["easy", "medium", "hard"]:
        ex = [v for v in video_ids if tiers[v]["tier"] == t][:2]
        for vid in ex:
            print(f'  [{t}] {vid}: "{caps[vid][0]}"')


if __name__ == "__main__":
    main()
