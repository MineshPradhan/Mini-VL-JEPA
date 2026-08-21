"""
Shared helpers for the Mini VL-JEPA caption-curriculum scripts.

Handles:
  - loading MSR-VTT captions from several JSON layouts
  - assigning train / validate / test splits
  - saving / loading the caption bank (captions + LaBSE embeddings)
"""

import json
import os
import re

import numpy as np

WORD_RE = re.compile(r"[a-zA-Z']+")


def simple_tokens(text):
    return WORD_RE.findall(text.lower())


def _cap_from_dict(d):
    for k in ("caption", "sentence", "text", "en"):
        if k in d and d[k]:
            return str(d[k])
    return None


def load_captions(captions_path):
    """
    Returns (caps, splits):
      caps   : {video_id: [caption, ...]}
      splits : {video_id: 'train'|'validate'|'test'}  (may be empty)

    Accepted layouts:
      1) MSR-VTT videodatainfo style: {"videos":[{"video_id","split",...}],
                                       "sentences":[{"video_id","caption"}]}
      2) {video_id: [captions]} / {video_id: "caption"} /
         {video_id: {"captions": [...]}} / {video_id: [{"caption": ...}]}
      3) [{"video_id"|"video"|"id": ..., "caption"|"sentence"|"text": ...}]
    """
    with open(captions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    caps, splits = {}, {}

    if isinstance(data, dict) and "sentences" in data:
        for s in data["sentences"]:
            caps.setdefault(str(s["video_id"]), []).append(s["caption"])
        for v in data.get("videos", []):
            vid = v.get("video_id", v.get("id"))
            if vid is not None and "split" in v:
                splits[str(vid)] = str(v["split"])
    elif isinstance(data, dict):
        for vid, c in data.items():
            if isinstance(c, str):
                caps[str(vid)] = [c]
            elif isinstance(c, dict):
                inner = c.get("captions", c.get("sentences", c.get("caption")))
                if isinstance(inner, str):
                    caps[str(vid)] = [inner]
                elif isinstance(inner, (list, tuple)):
                    caps[str(vid)] = [str(x) for x in inner if x]
            elif isinstance(c, (list, tuple)):
                out = []
                for x in c:
                    if isinstance(x, str):
                        out.append(x)
                    elif isinstance(x, dict):
                        cc = _cap_from_dict(x)
                        if cc:
                            out.append(cc)
                caps[str(vid)] = out
    elif isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            vid = row.get("video_id", row.get("video", row.get("id", row.get("vid"))))
            cap = _cap_from_dict(row)
            if vid is not None and cap:
                caps.setdefault(str(vid), []).append(cap)
    else:
        raise ValueError("Unrecognised captions JSON format: " + captions_path)

    caps = {k: [c.strip() for c in v if c and str(c).strip()] for k, v in caps.items()}
    caps = {k: v for k, v in caps.items() if v}
    if not caps:
        raise ValueError(
            f"Could not parse any (video_id, caption) pairs from {captions_path}. "
            "Paste the first ~20 lines of this file so the loader can be adapted."
        )
    return caps, splits


def split_from_filename(path):
    """Infer split from the filename, e.g. msrvtt_train.json -> 'train'."""
    name = os.path.basename(path).lower()
    if "train" in name:
        return "train"
    if "val" in name:
        return "validate"
    if "test" in name:
        return "test"
    return None


def load_captions_multi(paths):
    """Merge several caption files. Split priority per video:
    split stored inside the JSON > split from the filename > id-range fallback
    (applied later by assign_split)."""
    all_caps, all_splits = {}, {}
    for p in paths:
        caps, splits = load_captions(p)
        fsplit = split_from_filename(p)
        for vid, cs in caps.items():
            bucket = all_caps.setdefault(vid, [])
            for c in cs:
                if c not in bucket:
                    bucket.append(c)
            if vid in splits:
                all_splits[vid] = splits[vid]
            elif fsplit and vid not in all_splits:
                all_splits[vid] = fsplit
        src = "JSON" if splits else (f"filename ({fsplit})" if fsplit else "id-range fallback")
        n_caps = sum(len(v) for v in caps.values())
        print(f"  {os.path.basename(p)}: {n_caps} captions, {len(caps)} videos, split from {src}")
    return all_caps, all_splits


def assign_split(video_id, splits):
    """Split from JSON if present, else standard MSR-VTT full-split id ranges:
    video0-6512 train, video6513-7009 validate, video7010-9999 test."""
    if video_id in splits:
        s = splits[video_id].lower()
        return {"val": "validate", "valid": "validate"}.get(s, s)
    m = re.search(r"(\d+)", video_id)
    if not m:
        return "train"
    n = int(m.group(1))
    if n <= 6512:
        return "train"
    if n <= 7009:
        return "validate"
    return "test"


def save_bank(path, video_ids, captions, embeddings):
    np.savez_compressed(
        path,
        video_ids=np.array(video_ids),
        captions=np.array(captions, dtype=object),
        embeddings=embeddings.astype(np.float16),
    )


def load_bank(path):
    z = np.load(path, allow_pickle=True)
    return (
        [str(v) for v in z["video_ids"]],
        [str(c) for c in z["captions"]],
        z["embeddings"],
    )


def load_tiers(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
