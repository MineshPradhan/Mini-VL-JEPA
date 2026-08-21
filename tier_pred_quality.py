"""
Per-tier predictor quality: how well does the JEPA predictor itself capture
each tier, independent of the decoder?

For the chosen split it reports, per tier:
  - mean / median cos(predicted emb, closest GT caption emb)   "pred_cos"
  - retrieval R@1 within the split (one caption per video - the same
    protocol as src/eval.py)

How to read it: the decoder's cap_sim (from eval_captions.py) can never rise
much above what pred_cos allows. If cap_sim per tier tracks pred_cos per
tier, the decoder is no longer the bottleneck and the residual per-tier gap
belongs to the predictor.

Usage (defaults match the project layout):
  python tier_pred_quality.py
"""

import argparse

import numpy as np

from curriculum_common import load_bank, load_tiers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="curriculum/caption_bank.npz")
    ap.add_argument("--tiers", default="curriculum/tiers.json")
    ap.add_argument("--preds", default="curriculum/preds.npz")
    ap.add_argument("--split", default="validate")
    ap.add_argument("--pool_size", type=int, default=0,
                    help="randomly subsample the pool to this many videos (0 = full pool); "
                         "use to compare R@1 across splits at equal gallery size")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    vids, caps, embs = load_bank(args.bank)
    tiers = load_tiers(args.tiers)
    z = np.load(args.preds, allow_pickle=True)
    pmap = {str(v): e for v, e in zip(z["video_ids"], z["embeddings"])}

    idx_by_vid, first_idx = {}, {}
    for i, v in enumerate(vids):
        idx_by_vid.setdefault(v, []).append(i)
        first_idx.setdefault(v, i)

    pool = sorted(v for v, info in tiers.items()
                  if info["split"] == args.split and v in pmap and v in idx_by_vid)
    if not pool:
        raise SystemExit("No videos matched - check --split and --preds.")
    if args.pool_size and args.pool_size < len(pool):
        rng = np.random.default_rng(args.seed)
        pool = sorted(str(v) for v in rng.choice(pool, size=args.pool_size, replace=False))
        print(f"(subsampled to a pool of {len(pool)} videos, seed {args.seed})")

    C = np.stack([embs[first_idx[v]] for v in pool]).astype(np.float32)   # (M, 768)
    P = np.stack([pmap[v] for v in pool]).astype(np.float32)              # (M, 768)
    S = P @ C.T
    pos = {v: j for j, v in enumerate(pool)}

    stats = {}
    for i, v in enumerate(pool):
        own = S[i, pos[v]]
        r1 = int((S[i] > own).sum()) == 0
        G = embs[idx_by_vid[v]].astype(np.float32)
        pc = float(np.max(G @ P[i]))
        t = tiers[v]["tier"]
        stats.setdefault(t, {"pc": [], "r1": []})
        stats[t]["pc"].append(pc)
        stats[t]["r1"].append(r1)

    print(f"{args.split} split - predictor quality per tier (pool of {len(pool)} videos)")
    print(f"{'tier':<8}{'n':>6}{'mean pred_cos':>15}{'median':>10}{'R@1 %':>9}")
    order = [t for t in ["easy", "medium", "hard"] if t in stats]
    for t in order:
        pc = np.array(stats[t]["pc"])
        r1 = np.array(stats[t]["r1"], dtype=np.float64)
        print(f"{t:<8}{len(pc):>6}{pc.mean():>15.4f}{np.median(pc):>10.4f}{100 * r1.mean():>9.1f}")
    allpc = np.concatenate([np.array(stats[t]["pc"]) for t in order])
    allr1 = np.concatenate([np.array(stats[t]["r1"], dtype=np.float64) for t in order])
    print(f"{'all':<8}{len(allpc):>6}{allpc.mean():>15.4f}{np.median(allpc):>10.4f}{100 * allr1.mean():>9.1f}")


if __name__ == "__main__":
    main()
