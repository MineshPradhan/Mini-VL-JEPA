"""
Caption-space density per tier: how close is each video's caption to its
nearest DISTRACTOR (another video's caption) in LaBSE space?

For the retrieval pool (one caption per video, same as src/eval.py) it
reports, per tier, the mean/median similarity of each caption to its nearest
neighbour from a different video.

How to read it: if easy-tier captions have nearer distractors than the
predictor's typical accuracy (pred_cos from tier_pred_quality.py), rank-1
retrieval on easy videos is decided by crowding, not by predictor quality -
explaining why R@1 can be flat or inverted across tiers even though
pred_cos and caption quality are monotonic.

Usage (defaults match the project layout):
  python tier_distractor_density.py
"""

import argparse

import numpy as np

from curriculum_common import load_bank, load_tiers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="curriculum/caption_bank.npz")
    ap.add_argument("--tiers", default="curriculum/tiers.json")
    ap.add_argument("--split", default="validate")
    args = ap.parse_args()

    vids, caps, embs = load_bank(args.bank)
    tiers = load_tiers(args.tiers)

    first_idx = {}
    for i, v in enumerate(vids):
        first_idx.setdefault(v, i)

    pool = sorted(v for v, info in tiers.items()
                  if info["split"] == args.split and v in first_idx)
    C = np.stack([embs[first_idx[v]] for v in pool]).astype(np.float32)   # (M, 768)

    S = C @ C.T
    np.fill_diagonal(S, -1.0)            # exclude self
    nearest = S.max(axis=1)              # nearest-distractor similarity per video

    stats = {}
    for v, d in zip(pool, nearest):
        stats.setdefault(tiers[v]["tier"], []).append(float(d))

    print(f"{args.split} split - nearest-distractor caption similarity per tier "
          f"(pool of {len(pool)} captions)")
    print(f"{'tier':<8}{'n':>6}{'mean':>9}{'median':>9}")
    order = [t for t in ["easy", "medium", "hard"] if t in stats]
    for t in order:
        d = np.array(stats[t])
        print(f"{t:<8}{len(d):>6}{d.mean():>9.4f}{np.median(d):>9.4f}")
    alld = np.concatenate([np.array(stats[t]) for t in order])
    print(f"{'all':<8}{len(alld):>6}{alld.mean():>9.4f}{np.median(alld):>9.4f}")


if __name__ == "__main__":
    main()
