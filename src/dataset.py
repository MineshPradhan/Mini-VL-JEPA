"""
MSR-VTT / VATEX style dataset.

Expected annotation format (JSON list):
[
  {"video": "video0.mp4", "caption": "a man is cooking pasta"},
  {"video": "video0.mp4", "caption": "someone prepares food in a kitchen"},
  ...
]

Videos are decoded with decord (fast) and uniformly subsampled to
`num_frames` frames, then preprocessed by the V-JEPA 2 video processor
inside the collate function (so it runs batched on CPU workers).
"""

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class VideoCaptionDataset(Dataset):
    def __init__(
        self,
        annotations_json: str,
        video_root: str,
        num_frames: int = 16,
        split: str = "train",
    ):
        with open(annotations_json, encoding="utf-8") as f:
            self.samples = json.load(f)
        self.video_root = video_root
        self.num_frames = num_frames
        self.split = split

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, path: str) -> np.ndarray:
        from decord import VideoReader, cpu

        vr = VideoReader(path, ctx=cpu(0))
        total = len(vr)
        if self.split == "train":
            # jittered uniform sampling for augmentation
            edges = np.linspace(0, total, self.num_frames + 1).astype(int)
            idx = [random.randint(a, max(a, b - 1)) for a, b in zip(edges[:-1], edges[1:])]
        else:
            idx = np.linspace(0, total - 1, self.num_frames).astype(int).tolist()
        frames = vr.get_batch(idx).asnumpy()  # (T, H, W, C) uint8
        return frames

    def __getitem__(self, i):
        s = self.samples[i]
        frames = self._load_frames(os.path.join(self.video_root, s["video"]))
        return {"frames": frames, "caption": s["caption"], "video_id": s["video"]}

class CollateFn:
    """Picklable collate (Windows DataLoader workers require this)."""

    def __init__(self, video_processor):
        self.processor = video_processor

    def __call__(self, batch):
        videos = [b["frames"] for b in batch]
        captions = [b["caption"] for b in batch]
        video_ids = [b["video_id"] for b in batch]
        pixel = self.processor(videos, return_tensors="pt")["pixel_values_videos"]
        return {
            "pixel_values_videos": pixel,
            "captions": captions,
            "video_ids": video_ids,
        }


def make_collate_fn(video_processor):
    return CollateFn(video_processor)

class CachedFeatureDataset(Dataset):
    """Loads precomputed V-JEPA 2 features instead of decoding videos."""

    def __init__(self, annotations_json: str, features_dir: str):
        with open(annotations_json, encoding="utf-8") as f:
            samples = json.load(f)
        self.samples = [
            s for s in samples
            if os.path.exists(os.path.join(features_dir, os.path.basename(s["video"]) + ".pt"))
        ]
        self.features_dir = features_dir
        print(f"CachedFeatureDataset: {len(self.samples)} caption pairs "
              f"({len(samples) - len(self.samples)} skipped, no cached feature)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        vid = os.path.basename(s["video"])
        feats = torch.load(
            os.path.join(self.features_dir, vid + ".pt"), map_location="cpu"
        ).float()  # .float() handles fp16-shrunk caches too
        return {"features": feats, "caption": s["caption"], "video_id": vid}


def cached_collate(batch):
    return {
        "features": torch.stack([b["features"] for b in batch]),
        "captions": [b["caption"] for b in batch],
        "video_ids": [b["video_id"] for b in batch],
    }