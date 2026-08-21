# save as shrink_cache.py, run once
import torch, glob
for f in glob.glob("data/features/*/*.pt"):
    t = torch.load(f, map_location="cpu")
    if t.dtype == torch.float32:
        torch.save(t.half(), f)
print("converted to fp16")