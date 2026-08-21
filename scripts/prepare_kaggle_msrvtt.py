import json, os

CAPTIONS_DIR = r"C:\Users\Minesh\Downloads\Summer Project\files\mini-vl-jepa\MSR-VTT\Video_Captions"   # <-- your real Video_Captions folder
OUT_DIR = "data"

SPLITS = {
    "Train_Captions.json": ("Train", "msrvtt_train.json"),
    "Val_Captions.json":   ("Val",   "msrvtt_val.json"),
    "Test_Captions.json":  ("Test",  "msrvtt_test.json"),
}

def extract_pairs(d):
    """Your format: {"9952": [{"video_id": "video9952", "caption": "..."}, ...], ...}"""
    pairs = []
    for _, items in d.items():
        for it in items:
            pairs.append((it["video_id"], it["caption"]))
    return pairs

os.makedirs(OUT_DIR, exist_ok=True)
for src, (subfolder, dst) in SPLITS.items():
    path = os.path.join(CAPTIONS_DIR, src)
    if not os.path.exists(path):
        print("skip (not found):", src)
        continue
    pairs = extract_pairs(json.load(open(path)))
    out = []
    for vid, cap in pairs:
        name = vid if vid.endswith(".mp4") else vid + ".mp4"
        out.append({"video": f"{subfolder}/{name}", "caption": cap.strip()})
    json.dump(out, open(os.path.join(OUT_DIR, dst), "w"))
    print(f"{dst}: {len(out)} caption pairs  | sample: {out[0]}")