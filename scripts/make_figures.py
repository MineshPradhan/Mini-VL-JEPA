r"""
Generate thesis figures from experiment results.

    pip install matplotlib
    python scripts\make_figures.py

Outputs -> figures/
  fig1_multilingual.png       zero-shot cross-lingual retrieval bar chart
  fig2_selective_decoding.png savings-vs-staleness + latency curves
  fig3_head_comparison.png    JEPA vs token-baseline summary
Reads results_multilingual.json and results_selective_decoding.json if present,
otherwise falls back to the recorded numbers below.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val", choices=["val", "test"])
args = ap.parse_args()

SPLIT = args.split
N_VIDEOS = {"val": 497, "test": 2990}[SPLIT]
CHANCE = 100.0 / N_VIDEOS   # 0.20 for val, 0.033 for test
ML_JSON = ("results_multilingual.json" if SPLIT == "val"
           else "results_multilingual_test.json")

INK = "#16233B"; LATENT = "#6C4EE0"; DECODE = "#0E9B76"; WARN = "#C2571B"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

os.makedirs("figures", exist_ok=True)

# ---------------------------------------------------------------- fig 1
LANG_ORDER = ["en", "hi", "de", "ta", "zh", "te", "mr", "ar", "bn"]
LANG_NAMES = {"en": "English\n(trained)", "hi": "Hindi", "de": "German",
              "ta": "Tamil", "zh": "Chinese", "te": "Telugu",
              "mr": "Marathi", "ar": "Arabic", "bn": "Bengali"}
FALLBACK_ML = {"en": 19.72, "hi": 19.11, "de": 18.91, "ta": 18.51, "zh": 18.51,
               "te": 17.71, "mr": 16.50, "ar": 16.30, "bn": 16.30}

ml = dict(FALLBACK_ML)
if os.path.exists(ML_JSON):
    with open(ML_JSON, encoding="utf-8") as f:
        for lang, m in json.load(f).items():
            ml[lang] = m["R@1"]

langs = [l for l in LANG_ORDER if l in ml]
vals = [ml[l] for l in langs]
colors = [INK if l == "en" else LATENT for l in langs]

fig, ax = plt.subplots(figsize=(8, 4.2))
bars = ax.bar([LANG_NAMES[l] for l in langs], vals, color=colors, width=0.62)
en = ml["en"]
ax.axhline(en, ls="--", lw=1, color=INK, alpha=0.45)
ax.text(len(langs) - 0.4, en + 0.25, f"English ceiling {en:.1f}",
        ha="right", fontsize=9, alpha=0.75)
ax.axhline(CHANCE, ls=":", lw=1, color=WARN)
ax.text(len(langs) - 0.4, CHANCE + max(vals) * 0.02,
        f"random chance {CHANCE:.2f}", ha="right", fontsize=9, color=WARN)
for b, v, l in zip(bars, vals, langs):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}",
            ha="center", fontsize=9.5)
    if l != "en":
        ax.text(b.get_x() + b.get_width() / 2, v / 2, f"{100*v/en:.0f}%",
                ha="center", fontsize=8.5, color="white")
ax.set_ylabel("Video→Text Retrieval R@1 (%)")
ax.set_title("Zero-shot cross-lingual retrieval (trained on English only)\n"
             f"MSR-VTT {SPLIT}, {N_VIDEOS} videos · % = retention vs English", fontsize=11)
ax.set_ylim(0, max(vals) * 1.18)
fig.savefig(f"figures/fig1_multilingual_{SPLIT}.png")
plt.close(fig)
print(f"figures/fig1_multilingual_{SPLIT}.png")

# ---------------------------------------------------------------- fig 2
FALLBACK_SD = [
    {"tau": 0.00, "decode_rate": 1.000, "mean_staleness": 0.0000},
    {"tau": 0.05, "decode_rate": 0.456, "mean_staleness": 0.0148},
    {"tau": 0.10, "decode_rate": 0.307, "mean_staleness": 0.0363},
    {"tau": 0.15, "decode_rate": 0.233, "mean_staleness": 0.0606},
    {"tau": 0.20, "decode_rate": 0.192, "mean_staleness": 0.0811},
    {"tau": 0.25, "decode_rate": 0.169, "mean_staleness": 0.0974},
    {"tau": 0.30, "decode_rate": 0.154, "mean_staleness": 0.1108},
    {"tau": 0.40, "decode_rate": 0.136, "mean_staleness": 0.1334},
    {"tau": 0.50, "decode_rate": 0.128, "mean_staleness": 0.1456},
]
PRED_MS, DEC_MS = 7.6, 255.7

sd = FALLBACK_SD
if os.path.exists("results_selective_decoding.json"):
    with open("results_selective_decoding.json", encoding="utf-8") as f:
        sd = json.load(f)

taus = [r["tau"] for r in sd]
saved = [(1 - r["decode_rate"]) * 100 for r in sd]
stale = [r["mean_staleness"] for r in sd]
ms = [PRED_MS + r["decode_rate"] * DEC_MS for r in sd]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4), gridspec_kw={"wspace": 0.3})
a1.plot(saved, stale, "-o", color=LATENT, lw=2, ms=5)
for t, x, y in zip(taus, saved, stale):
    a1.annotate(f"τ={t:.2f}", (x, y), textcoords="offset points",
                xytext=(6, -11 if t in (0.05, 0.15) else 6), fontsize=8)
knee = min(range(len(sd)), key=lambda i: abs(taus[i] - 0.10))
a1.scatter([saved[knee]], [stale[knee]], s=140, facecolors="none",
           edgecolors=DECODE, lw=2, zorder=5)
a1.annotate("recommended\noperating point", (saved[knee], stale[knee]),
            xytext=(saved[knee] - 32, stale[knee] + 0.03), fontsize=9,
            color=DECODE, arrowprops=dict(arrowstyle="->", color=DECODE))
a1.set_xlabel("Decoder calls eliminated (%)")
a1.set_ylabel("Mean staleness (cosine drift)")
a1.set_title("Savings vs staleness trade-off")

a2.plot(taus, ms, "-o", color=DECODE, lw=2, ms=5)
a2.axhline(PRED_MS + DEC_MS, ls="--", lw=1, color=WARN)
a2.text(taus[-1], PRED_MS + DEC_MS - 16, "always-decode baseline",
        ha="right", fontsize=9, color=WARN)
a2.axhline(PRED_MS, ls=":", lw=1, color=INK, alpha=0.5)
a2.text(taus[-1], PRED_MS + 5, "predictor-only floor (7.6 ms)",
        ha="right", fontsize=9, alpha=0.7)
for t, y in zip(taus, ms):
    if t in (0.10, 0.25):
        a2.annotate(f"{y:.0f} ms ({(PRED_MS+DEC_MS)/y:.1f}× faster)", (t, y),
                    textcoords="offset points", xytext=(8, 8), fontsize=9)
a2.set_xlabel("Drift threshold τ")
a2.set_ylabel("Estimated cost per chunk (ms, batch=1)")
a2.set_title("Streaming latency vs τ  (RTX 3050)")

fig.suptitle("Selective decoding: decode only when the latent changes "
             "(497 videos × 8 chunks)", fontsize=12, y=1.03)
fig.savefig("figures/fig2_selective_decoding.png")
plt.close(fig)
print("figures/fig2_selective_decoding.png")

# ---------------------------------------------------------------- fig 3
fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
panels = [
    ("Trainable params (M)", [("JEPA\n(ours)", 26.5 + 3.8, LATENT),
                              ("Token\nbaseline", 85.7, INK)], "lower is better"),
    ("Retrieval R@1 (%)", [("JEPA\n(ours)", 19.72, LATENT),
                           ("Token\nbaseline", 0, INK)], "baseline: not capable"),
    ("Caption BLEU-4", [("JEPA\n(ours)", 8.39, LATENT),
                        ("Token\nbaseline", 17.27, INK)], "8 langs vs EN-only"),
]
for ax, (title, bars_, note) in zip(axes, panels):
    names = [b[0] for b in bars_]; vs = [b[1] for b in bars_]
    cs = [b[2] for b in bars_]
    bb = ax.bar(names, vs, color=cs, width=0.55)
    for b, v in zip(bb, vs):
        label = f"{v:.1f}" if v else "n/a"
        ax.text(b.get_x() + b.get_width()/2, max(v, 0.5) + max(vs)*0.02,
                label, ha="center", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(note, fontsize=8.5, color="#6B7686")
    ax.set_ylim(0, max(vs) * 1.2)
fig.suptitle("Experiment 1: embedding-prediction head vs token-generation head "
             "(same frozen V-JEPA 2 features)", fontsize=11.5, y=1.04)
fig.savefig("figures/fig3_head_comparison.png")
plt.close(fig)
print("figures/fig3_head_comparison.png")

print("\nAll figures written to figures/")
