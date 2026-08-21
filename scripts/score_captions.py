"""
Score generated captions against ALL reference captions per video
(standard MSR-VTT protocol) using BLEU-4.

    pip install sacrebleu
    python scripts\score_captions.py --generated baseline_val_captions.json
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json
from collections import defaultdict


def main():
    import sacrebleu

    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", required=True)
    ap.add_argument("--references", default="data/msrvtt_val.json")
    args = ap.parse_args()

    with open(args.generated, encoding="utf-8") as f:
        gen = json.load(f)

    # gather ALL references per video (not just one)
    with open(args.references, encoding="utf-8") as f:
        refs_all = defaultdict(list)
        for s in json.load(f):
            refs_all[os.path.basename(s["video"])].append(s["caption"])

    hyps, refs = [], []
    max_refs = max(len(v) for v in refs_all.values())
    for g in gen:
        vid = os.path.basename(g["video"])
        if vid in refs_all:
            hyps.append(g["generated"])
            r = refs_all[vid]
            refs.append(r + [r[0]] * (max_refs - len(r)))  # pad ref lists

    # sacrebleu wants refs transposed: list of reference streams
    ref_streams = [[r[i] for r in refs] for i in range(max_refs)]
    bleu = sacrebleu.corpus_bleu(hyps, ref_streams)
    print(f"scored {len(hyps)} captions against up to {max_refs} references each")
    print(f"BLEU-4: {bleu.score:.2f}")


if __name__ == "__main__":
    main()