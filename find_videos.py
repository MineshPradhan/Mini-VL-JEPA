import os

search_root = r"C:\Users\Minesh\Downloads\Summer Project\files\mini-vl-jepa\MSR-VTT\Video_Datasets"   # <-- your real Video_Datasets folder

found = []
for dirpath, dirnames, filenames in os.walk(search_root):
    vids = [f for f in filenames if f.lower().endswith((".mp4", ".avi", ".mkv", ".webm"))]
    if vids:
        found.append((dirpath, len(vids), vids[:3]))

if not found:
    print("No video files found anywhere under", search_root)
else:
    for path, count, samples in found:
        print(f"{count:>6} videos in: {path}")
        print(f"        sample names: {samples}")