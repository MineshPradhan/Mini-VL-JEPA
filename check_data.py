import json, os

root = r"C:\Users\Minesh\Downloads\Summer Project\files\mini-vl-jepa\MSR-VTT\Video_Datasets"   # <-- your real Video_Datasets folder

for jf, folder in [("data/msrvtt_train.json", "Train"),
                   ("data/msrvtt_val.json", "Val"),
                   ("data/msrvtt_test.json", "Test")]:
    data = json.load(open(jf))
    ids = sorted({int(d["video"].split("/video")[1].split(".")[0]) for d in data})
    exists = sum(os.path.exists(os.path.join(root, d["video"])) for d in data[:500])
    print(f"{jf}: {len(data)} pairs | video id range {ids[0]}-{ids[-1]} | {exists}/500 sampled files found")