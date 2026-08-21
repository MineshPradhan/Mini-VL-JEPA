# save as verify_cache.py in mini-vl-jepa, run: python verify_cache.py
import os, torch, glob

bad = 0
for f in glob.glob("data/features/*/*.pt"):
    try:
        torch.load(f, map_location="cpu")
    except Exception:
        os.remove(f)
        bad += 1
        print("deleted corrupt:", f)
print(f"done, removed {bad} corrupt files")