import json
d = json.load(open(r"C:\Users\Minesh\Downloads\Summer Project\files\mini-vl-jepa\MSR-VTT\Video_Captions\Test_Captions.json"))  # your real path
print("type:", type(d).__name__)
if isinstance(d, dict):
    print("keys:", list(d.keys())[:10])
    k = list(d.keys())[0]
    print("sample value for", k, "→", str(d[k])[:200])
else:
    print("first item:", str(d[0])[:300])