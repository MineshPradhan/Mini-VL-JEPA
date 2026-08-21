# Mini VL-JEPA — Complete Project Explanation

*Reference document for guide meetings and viva. Part 1 answers the conceptual
questions ("what is this thing"); Parts 2–6 are the technical story; Part 7 is
the anticipated Q&A; Part 8 is a glossary you can point at mid-conversation.*

---

## Part 0 — The 60-second summary

> The captions were random because of a **training/inference mismatch**, not an
> architecture problem. The decoder had been trained to turn *ground-truth*
> caption embeddings into text, but at demo time it receives the *predictor's*
> output, which on unseen videos sits about 0.59 cosine-similar to the ideal
> embedding. It had never seen inputs from that region of the space. We measured
> that failure at **0.073 BLEU-4**.
>
> Following the staged-complexity instruction, MSR-VTT was split into low /
> high / difficult tiers using caption agreement, and an **English-only** decoder
> was trained tier by tier with two robustness mechanisms: Gaussian noise
> injection calibrated to the predictor's measured error, and fine-tuning on the
> predictor's actual output embeddings.
>
> Same predictor, same videos: **0.073 → 0.515 BLEU-4** on the easy tier. The
> final one-shot test over all 2,990 test videos gives **0.546 / 0.399 / 0.304
> BLEU-4** by tier — a perfectly monotonic difficulty ordering, which
> independently validates the complexity-tier idea. The live demo now runs this
> decoder.

---

## Part 1 — Foundational concepts

### 1.1 What is "Mini VL-JEPA" here?

Break the name into three parts.

**JEPA — Joint-Embedding Predictive Architecture.** A JEPA is trained to predict
*a representation of* something, not the thing itself. Classical generative
models predict raw pixels or raw tokens; a JEPA predicts an **embedding** — a
fixed-length vector produced by another network. The motivation is that pixels
contain enormous unpredictable detail (exact texture, lighting, noise) that
wastes model capacity, whereas an embedding keeps the semantic content and
discards that unpredictable detail. This is Yann LeCun's proposal for
self-supervised learning, and **V-JEPA 2** (Meta) is its video instantiation.

**VL — Vision-Language.** Meta's V-JEPA 2 predicts *video* representations from
*video*. This project predicts a **sentence embedding** from video — that is,
the target space is language rather than more video. So the model learns
"what would a sentence describing this video look like, in embedding space."

**Mini — the scale.** The full-size approach requires large clusters. This
project is a compact, reproducible version in which the two large models are
**frozen** and only a small predictor (~13M parameters) and a small decoder
(9.9M trainable parameters) are trained — the whole thing runs on a single
6 GB laptop GPU (RTX 3050) with a public dataset (MSR-VTT 10K).

**One-sentence answer for the guide:**
> "Mini VL-JEPA is a compact video-to-language model that, instead of generating
> caption words directly from video, predicts the *sentence embedding* of the
> caption in a frozen multilingual text space — and then decodes text from that
> embedding only when needed. 'Mini' means everything trainable is small enough
> to run on one consumer GPU because the two big models are frozen."

**What is ours versus Meta's:**

| Component | Origin |
|---|---|
| V-JEPA 2 video encoder | Meta, used frozen, off-the-shelf |
| LaBSE text encoder | Google, used frozen, off-the-shelf |
| JEPA predictor (video tokens → sentence embedding) | **Ours** — trained here |
| Multilingual target space → zero-shot cross-lingual transfer | **Ours** — the design choice of LaBSE as target |
| Selective decoding for streaming | **Ours** |
| Caption curriculum + noise-injected, prediction-adapted decoder | **Ours** — this phase |

### 1.2 What is a "frozen" model here?

**Frozen = the model's weights are never updated.** Concretely, in the code:

```python
self.model.eval()                 # inference mode, no dropout/batchnorm updates
for p in self.model.parameters():
    p.requires_grad = False       # no gradients computed, no optimizer updates
```

A frozen model is used purely as a **fixed feature extractor**: video goes in,
the same vector always comes out, and back-propagation stops at its boundary.

**Two models are frozen in this project:**

1. **V-JEPA 2 (`facebook/vjepa2-vitl-fpc64-256`)** — turns a video clip into
   video token features (1024-dim per token). Frozen because it was pretrained
   on far more video than we have.
2. **LaBSE (`sentence-transformers/LaBSE`)** — turns a caption into a 768-dim
   sentence embedding. Frozen because it *defines the target space*; if it moved
   during training, the target would be a moving goalpost.

**Why freezing is the right choice here — five reasons, in order of how
convincing they are to an examiner:**

1. **It is what makes the JEPA formulation well-posed.** In JEPA, the "y encoder"
   produces the prediction target. If that encoder were trainable alongside the
   predictor, both could collapse to a trivial solution (e.g. output the constant
   zero vector, achieving perfect prediction and zero information). Freezing the
   target encoder is a standard anti-collapse guarantee.
2. **It preserves the multilingual property.** LaBSE aligns 100+ languages in one
   space. The predictor learns to hit that space using only English captions;
   because the space still contains Hindi, Tamil, German, etc., the model
   retrieves in those languages **with no additional training** (87–95% retention
   measured across eight unseen languages). Fine-tuning LaBSE would destroy this.
3. **Feasibility on 6 GB VRAM.** V-JEPA 2 ViT-L is far too large to fine-tune on
   an RTX 3050. Freezing keeps only ~13M predictor parameters in the optimizer.
4. **Massive speedup through caching.** Because a frozen encoder always returns
   the same output for the same video, features are computed **once** and cached
   to disk (fp16, deduplicated). MSR-VTT has ~20 captions per video, so caching
   removes ~20× of redundant encoder passes — this is what made the whole
   training loop practical.
5. **Reproducibility and attribution.** Results depend on a published checkpoint
   anyone can download, so improvements are clearly attributable to *our*
   predictor rather than to fine-tuning a large pretrained model.

**Analogy for the guide:** *the frozen encoders are like a fixed pair of
measuring instruments. V-JEPA is the instrument that measures video, LaBSE is the
instrument that measures sentences. We do not modify the instruments — we train a
small translator that converts one instrument's readings into the other's.*

### 1.3 What is an "embedding" and what is "LaBSE space"?

An embedding is a list of numbers (here 768 of them) that represents meaning, in
such a way that **similar meanings give similar numbers**. Similarity is measured
by cosine similarity, between −1 and 1.

LaBSE space has one special property this project depends on: it is
**language-agnostic**. "A man is cooking" (English) and "एक आदमी खाना बना रहा है"
(Hindi) map to nearly the same vector. That single property is what buys
zero-shot cross-lingual behaviour.

### 1.4 What exactly does the predictor predict?

Input: the frozen V-JEPA token features of a video clip.
Output: **one 768-dim vector**, L2-normalised — the model's guess at where the
caption of that video lives in LaBSE space.

It does **not** output words. That is the whole point: prediction happens in
representation space; text is produced later, and only on demand.

Architecture: 8 learned latent queries cross-attend over the video tokens through
a 6-layer, 8-head transformer decoder (hidden size 512), the result is mean-pooled
and projected to 768 dims, then normalised.

Loss = cosine alignment + symmetric InfoNCE (in-batch negatives) + a VICReg-style
variance term that prevents collapse.

### 1.5 What is the text decoder for, then?

The predicted embedding is useful on its own for **retrieval** (compare it to
caption embeddings) and for **drift detection** (does the scene meaning change?).
But humans want words, so a small decoder converts the embedding into a sentence:

- Base model: **mT5-small**, adapted with **LoRA** (rank 16 on the q/v projections).
- The 768-dim embedding is expanded by a small MLP into **16 pseudo encoder
  tokens** ("prefix"), which mT5's decoder cross-attends over as if they were an
  encoder output.
- Total trainable: **9.9M parameters**. Base mT5 weights stay frozen — LoRA
  trains small low-rank update matrices instead.
- Generation uses beam search (5 beams, `no_repeat_ngram_size=3`).

**Selective decoding:** in streaming mode the decoder only fires when the
predicted embedding drifts more than τ from the last decoded embedding
(cosine distance, τ = 0.10). On real MSR-VTT runs this eliminates ~69% of decoder
calls (3.1× speedup); the demo screenshot shows 7 decodes over 18 chunks = 61% saved.

### 1.6 Why is this "JEPA" and not just a captioning model?

A conventional captioning model maps video → words directly, end-to-end, and the
language model is inside the loop. Here:

- The **learning objective** is embedding prediction, not next-token prediction.
- The **text decoder is optional and detachable** — retrieval, drift detection and
  cross-lingual search all work without ever producing a word.
- The **target space is multilingual**, so one predictor serves many languages.
- The decoder can fire **rarely** (selective decoding), which is what makes
  streaming cheap.

This is also why the earlier problem was confusing at first: the system's
*embedding-level* metrics were fine (19.7% R@1) while its *text output* was
broken — because the text stage is a separate, later stage.

---

## Part 2 — The system, end to end

### 2.1 Pipeline

```
video (.mp4)
   │
   ├─ decord: sample 16 frames per chunk
   │
   ▼
[FROZEN] V-JEPA 2 ViT-L  ──────────────► video tokens (N × 1024)
   │                                            │
   │  (cached to disk as fp16 — computed once)  │
   ▼                                            ▼
[TRAINED] JEPA Predictor (6-layer transformer, 8 latent queries)
   │
   ▼
predicted sentence embedding  ẑ  (768-d, L2-normalised)
   │
   ├──────────────► retrieval / drift detection (no text needed)
   │
   ├─ SelectiveDecoder: fire only if drift > τ (0.10)
   │
   ▼
[TRAINED, LoRA] mT5-small caption decoder  ──► English caption
```

Training-time target branch:

```
caption text ──► [FROZEN] LaBSE ──► target embedding z (768-d)
                     loss = cosine(ẑ, z) + InfoNCE + variance
```

### 2.2 Component specification

| Component | Model / setting | Params | Status |
|---|---|---|---|
| Video encoder | `facebook/vjepa2-vitl-fpc64-256`, 1024-d tokens | ~300M | **Frozen** |
| Text target encoder | `sentence-transformers/LaBSE`, 768-d | ~470M | **Frozen** |
| JEPA predictor | 6 layers, 8 heads, hidden 512, 8 latent queries, dropout 0.1 | ~13M | Trained |
| Caption decoder | mT5-small + LoRA r=16, α=32 (q, v), 16 prefix tokens | 9.9M trainable | Trained |
| Selective decoder | cosine-drift policy, τ = 0.10 | 0 | Rule-based |

### 2.3 Data

MSR-VTT 10K (Kaggle), converted to this project's JSON format by
`scripts/prepare_kaggle_msrvtt.py`. Standard split, ~20 English captions/video:

| Split | Videos | Captions |
|---|---|---|
| train | 6,513 | 118,552 |
| validate | 497 | 9,062 |
| test | 2,990 | 54,653 |
| **total** | **10,000** | **182,267** |

Complexity tiers (built by `build_complexity_tiers.py`):

| Tier | train | validate | test |
|---|---|---|---|
| easy | 2,167 | 175 | 992 |
| medium | 2,217 | 150 | 966 |
| hard | 2,129 | 172 | 1,032 |

---

## Part 3 — The problem: why captions were random

### 3.1 The symptom

The demo produced strings like:

> `"twowrestlerstlers fightrestrestrestwrestling wrestwrestling matchwrestling"`

— degenerate repetition, sometimes on-topic, mostly unusable.

### 3.2 The diagnosis

The decoder had been trained on **ground-truth caption embeddings** (LaBSE
outputs — exact points in the space). At inference it is fed the **predictor's
output**, which is not an exact point: on unseen videos it sits at cosine ≈ 0.59
from the nearest true caption embedding. The decoder had never been trained on
inputs from that region, so it behaved unpredictably there.

This is a **train/inference distribution mismatch** — a data problem, not an
architecture problem, and not an optimization problem.

**The controlled experiment that proved it.** The same decoder was evaluated
under two input conditions on the same 175 easy validation videos:

| Input to decoder | BLEU-4 | ROUGE-L | LaBSE-sim |
|---|---|---|---|
| ground-truth caption embedding ("oracle") | 0.400 | 0.709 | 0.897 |
| predictor's embedding ("real system") | 0.073 | 0.374 | 0.639 |

Only the input distribution changed. Therefore the fault lay in the gap between
those two distributions — not in the decoder architecture, not in the predictor,
not in V-JEPA.

**Analogy for the guide:** the decoder learned to read clean handwriting, then
was handed smudged handwriting it had never seen.

### 3.3 The measured size of the gap

`export_predicted_embeddings.py` measures mean cosine between each predicted
embedding and the closest ground-truth caption embedding of that video:

| Split | mean cos(pred, best GT caption) |
|---|---|
| train | 0.6626 |
| validate | 0.5945 |
| test | 0.5975 |

The train↔validate difference (0.66 vs 0.59) is the predictor's generalization
gap. Validate ≈ test means the predictor behaves consistently on unseen data.

---

## Part 4 — The fix

### 4.1 Complexity tiers — the guide's instruction, made measurable

The instruction was "low complexity, then high, then difficult." Complexity is
scored from each video's ~20 captions, so it needs no extra annotation:

| Signal | Definition | Direction |
|---|---|---|
| **caption agreement** | mean pairwise LaBSE cosine among the video's captions | high agreement → **simpler** (everyone describes it the same way) |
| **caption length** | mean tokens per caption | longer → harder |
| **vocabulary rarity** | mean IDF of caption tokens | rarer words → harder |
| *(optional)* motion | mean absolute frame difference | more motion → harder |

`complexity = z(length) + z(rarity) − z(agreement)`; terciles → easy / medium / hard.

**Why agreement is the key signal:** if twenty annotators independently write
almost the same sentence, the video has one obvious interpretation. If they write
twenty different sentences, the content is ambiguous or information-dense.

Sanity check on the resulting tiers: screen recordings and abstract animations
landed in **hard**; "a man is teaching math" landed in **easy**. That matches
intuition, and the tier ordering later proved monotonic in every caption metric.

### 4.2 Noise injection (robustness mechanism 1)

During training, the ground-truth embedding is perturbed and re-normalised:

```python
e = F.normalize(e + noise_std * torch.randn_like(e), dim=-1)
```

with **σ = 0.035**, chosen from the *measured* predictor error (derived from
cos ≈ 0.59 as σ ≈ √(2−2c)/√d). The decoder therefore learns to decode a *region*
of embedding space rather than a single point.

Method credit: this is the CapDec idea (Nukrai et al., 2022), applied here to
JEPA predictor outputs.

### 4.3 Prediction adaptation (robustness mechanism 2)

Noise alone was **not sufficient** — a noise-trained decoder still scored 0.073
BLEU on real predictions. The decisive step was training on the predictor's
**actual outputs**: `export_predicted_embeddings.py` runs the trained predictor
over the cached features of all 6,513 train videos, and 50% of training samples
(`--pred_ratio 0.5`) then use those real embeddings instead of noisy GT ones.

Real predictor error is *structured* (systematic biases, a particular manifold),
not isotropic Gaussian — which is why simulating it with noise was insufficient
and using it directly worked.

### 4.4 Curriculum staging

| Stage | Training tiers | Init from | Epochs | LR | Output checkpoint |
|---|---|---|---|---|---|
| 1 | easy | scratch | 8 | 2e-4 | `dec_en_easy` |
| 1b | easy + real predictions | Stage 1 | 4 | 1e-4 | `dec_en_easy_pred` |
| 2 | easy + medium | Stage 1b | 5 | 1e-4 | `dec_en_easymed` |
| 3 | easy + medium + hard | Stage 2 | 5 | 1e-4 | `dec_en_all` ← **final** |

Two rules: **tiers accumulate** (easy stays in the mix — this is what prevents
catastrophic forgetting), and each stage must pass a **gate** on validation —
the new tier improves, previous tiers do not regress.

### 4.5 Two smaller but necessary details

- **Prefix projection.** A single 768-d vector is a severe bottleneck for a
  seq2seq decoder, so an MLP expands it to **16 pseudo encoder tokens**, giving
  cross-attention something with structure to attend over.
- **Beam search.** Generation uses 5 beams with `no_repeat_ngram_size=3`. Greedy
  or sampled decoding from a weak conditioning signal amplifies degeneration —
  this is part of why the original output repeated itself.

---

## Part 5 — Results

### 5.1 The decoder ablation (easy tier, validation, 175 videos)

| Decoder | Conditioning | BLEU-4 | ROUGE-L | LaBSE-sim |
|---|---|---|---|---|
| v1 noise-only | ground-truth (oracle) | 0.400 | 0.709 | 0.897 |
| v1 noise-only | **predicted (real system)** | **0.073** | 0.374 | 0.639 |
| v2 prediction-adapted | ground-truth (oracle) | 0.450 | 0.718 | 0.892 |
| v2 prediction-adapted | **predicted (real system)** | **0.515** | 0.688 | 0.846 |

**Reading:** row 2 is the original failure, reproduced under controlled
conditions. Row 4 is the fix — **7× BLEU improvement with the predictor
unchanged**. Rows 1 and 3 show adaptation cost nothing on the oracle side
(0.400 → 0.450), i.e. no catastrophic forgetting.

### 5.2 Curriculum progression (validation, predicted embeddings)

LaBSE-sim:

| Tier | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| easy | 0.846 | 0.852 | 0.853 |
| medium | 0.788 | 0.780 | 0.783 |
| hard | 0.700 | 0.710 | 0.713 |
| **all** | 0.778 | 0.781 | 0.783 |

BLEU-4:

| Tier | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| easy | 0.515 | 0.529 | 0.533 |
| medium | 0.378 | 0.378 | 0.394 |
| hard | 0.254 | 0.289 | 0.311 |
| **all** | 0.382 | 0.399 | 0.412 |

Earlier tiers never regress; the largest gain lands on the newest tier
(hard: 0.254 → 0.311, **+22% relative**).

### 5.3 Final one-shot test (2,990 videos, predicted embeddings)

Decoder frozen before this run; the test split was evaluated **exactly once**.

| Tier | Videos | BLEU-4 | ROUGE-L | LaBSE-sim |
|---|---|---|---|---|
| easy | 992 | 0.546 | 0.712 | 0.861 |
| medium | 966 | 0.399 | 0.594 | 0.784 |
| hard | 1,032 | 0.304 | 0.503 | 0.709 |
| **all** | **2,990** | **0.412** | **0.602** | **0.784** |

Test agrees with validation within **±0.015 on every cell** (all-tier BLEU-4:
0.4117 test vs 0.4120 validation) — no overfitting to validation during
curriculum design, and the tier gradient holds at 6× the sample size.

### 5.4 Qualitative before/after (same videos, different stages)

| Video | Earlier stage | Final decoder |
|---|---|---|
| Val/video6520 | "a sports athletes are dancing" (0.612) | "a woman is running on a track" (0.918) |
| Val/video6524 | "a man is adding a stove in a pan" | "a man is cooking a dish in a pan" (0.873) |
| Val/video6518 | "twowrestlerstlers fightrestrest…" (0.586) | "two men are wrestling in a wrestling match" (0.932) |
| Val/video6527 | "a woman is showing her hair hair" | "a woman is explaining how to make her hair" (0.917) |
| Test/video7026 | — | "a man is talking about a car" (**LaBSE-sim 1.000** — matches a human reference) |

### 5.5 Live demo behaviour

Video6524 (a **hard**-tier cooking video), 18 chunks, τ = 0.10:
7 decode events, **61% of decoder calls saved**, and every fired caption stayed
on the cooking topic ("making a dish in a pan", "talking about how to cook a
dish"). Per-chunk wording varies between events — that is expected, because each
caption comes from a 1-second chunk rather than the whole video.

---

## Part 6 — Analysis findings (the "why", beyond the headline numbers)

### 6.1 The residual tier gap belongs to the predictor, not the decoder

Predictor quality per tier, measured independently of any decoder:

| Tier | pred_cos (val) | pred_cos (test) | caption LaBSE-sim (test) |
|---|---|---|---|
| easy | 0.614 | 0.621 | 0.861 |
| medium | 0.595 | 0.594 | 0.784 |
| hard | 0.575 | 0.578 | 0.709 |

Caption quality tracks predictor alignment tier by tier. The decoder has
recovered its share of the gap; what remains requires a **better predictor**.
This is why medium-tier scores stayed flat across stages — it was already at its
predictor-imposed ceiling.

### 6.2 The decoder acts as a semantic denoiser

Caption similarity (0.86 on easy) is far **above** the raw embedding similarity
it was decoded from (0.62). Generating text snaps a noisy, off-manifold embedding
onto an actual sentence whose meaning is closer to the references than the raw
vector was. This is a genuine benefit of decoding, not just a cosmetic step.

### 6.3 Retrieval is pool-limited; captioning is not

| Setting | Pool | pred_cos | nearest-distractor sim | R@1 | all-tier cap_sim |
|---|---|---|---|---|---|
| validation | 497 | 0.594 | 0.692 | 19.7% | 0.783 |
| test | 2,990 | 0.597 | 0.747 | **6.6%** | 0.784 |
| test, size-matched control (seed 42) | 497 | 0.595 | — | **18.5%** | — |

R@1 fell 3× on test **purely because the candidate pool grew 6×** — restoring the
validation pool size restores R@1 (6.6% → 18.5%), while caption quality was
unchanged throughout (0.783 vs 0.784). Retrieval metrics are gallery-dependent;
generation metrics are not.

### 6.4 Caption-space density explains the retrieval behaviour

Nearest-distractor caption similarity (how crowded each caption's neighbourhood is):

| Tier | validation | test |
|---|---|---|
| easy | 0.743 | 0.807 |
| medium | 0.687 | 0.747 |
| hard | 0.645 | 0.689 |
| all | 0.692 | 0.747 |

On **every** tier the nearest distractor sits closer to the target than the
predictor's typical accuracy — so rank-1 retrieval is decided largely by
crowding. Easy videos are the most crowded (generic captions like "a man is
talking" pile up), which offsets their better embeddings.

**Conclusion to state carefully:** the complexity tiers predict **captioning
difficulty and predictor alignment** (both monotonic, on both splits), but **not
retrieval rank** — per-tier R@1 orderings were inconsistent across splits and
pool samples, i.e. second-order.

### 6.5 Why predicted BLEU can exceed oracle BLEU

Observed: 0.515 (predicted) vs 0.450 (oracle) on the same checkpoint. The
predictor outputs something close to the **consensus** of all ~20 reference
captions, so its decodes are generic and overlap many references — and BLEU is
multi-reference. Oracle mode reconstructs one specific caption's idiosyncratic
wording, which matches fewer references. The semantic metrics (ROUGE-L,
LaBSE-sim) still rank oracle higher, as expected. "The predictor regresses to the
caption consensus" is itself a useful observation about JEPA-style prediction.

---

## Part 7 — Anticipated questions

### A. Conceptual

**Q: What is Mini VL-JEPA?**
See §1.1. Short: a compact video→language model that predicts the *sentence
embedding* of a caption in a frozen multilingual text space instead of generating
words directly, and decodes text only on demand.

**Q: What is a frozen model here? Why freeze it?**
See §1.2. Short: weights never updated, `requires_grad=False`, used as a fixed
feature extractor. Frozen because (1) it keeps the JEPA target well-posed and
collapse-free, (2) it preserves LaBSE's multilingual alignment (the source of
zero-shot transfer), (3) 6 GB VRAM cannot fine-tune a ViT-L, (4) features can be
cached once and reused ~20× per video, (5) reproducibility.

**Q: If the encoders are frozen, what did you actually train?**
The JEPA predictor (~13M params) and the caption decoder (9.9M trainable LoRA +
projector params). Everything reported here is attributable to those two.

**Q: Why predict an embedding instead of generating the caption directly?**
Pixels and words contain unpredictable detail that wastes capacity; embeddings
keep semantics. It also makes the text decoder optional — retrieval, drift
detection and cross-lingual search need no text at all — which is what enables
selective decoding and the 3.1× streaming speedup.

**Q: Why LaBSE rather than an English-only sentence encoder?**
Because LaBSE aligns 100+ languages in one space, the predictor trained on
English captions transfers to unseen languages with no extra training (87–95%
retention across eight languages). That was a core contribution of the project.

**Q: What is LoRA and why use it?**
Instead of updating all of mT5-small's weights, LoRA inserts small low-rank
matrices into the attention projections and trains only those — 9.9M trainable
parameters instead of ~300M. It fits in 6 GB, trains in minutes per epoch, and
avoids destroying the pretrained language ability.

### B. About the failure and the fix

**Q: Why did the captions come out random before?**
Train/inference mismatch: the decoder was trained on exact ground-truth caption
embeddings but fed predictor outputs sitting ~0.59 cosine away. Quantified at
0.073 BLEU under controlled conditions (§3.2).

**Q: How do you know it was the decoder and not the predictor or V-JEPA?**
The oracle-vs-predicted experiment (§3.2): identical decoder, identical videos,
only the input distribution changed — 0.400 vs 0.073 BLEU. That isolates the
fault to the distribution gap.

**Q: Why not just train the old decoder for more epochs?**
More epochs on ground-truth embeddings never changes the input distribution —
that is an optimization fix for a data-distribution problem. Proven empirically:
the noise-robust GT-trained decoder still scored 0.073; only training on real
predictor outputs fixed it.

**Q: Isn't noise injection enough? Why also use real predictions?**
Because real predictor error is structured, not isotropic Gaussian. Noise alone
gave 0.073; adding 50% real predicted embeddings gave 0.515.

**Q: Why does the curriculum help, rather than just training on everything?**
Staging gave a clean measurement path (each stage gated on validation) and
avoided regressions; the final model still sees all tiers. Note the honest
version: cross-tier generalization was already strong (a decoder trained only on
easy scored 0.700 LaBSE-sim on hard), so the curriculum's measured contribution
is mainly on the hard tier (+22% relative BLEU) plus the *diagnostic* value of
knowing where the remaining errors come from.

**Q: Why English only, when the project is multilingual?**
Guide's instruction, and it is methodologically right: fix the decoding mechanism
in one language before reintroducing the multilingual variable. mT5-small was
retained (rather than switching to an English-only base) precisely so Hindi can
return as a fine-tune in Stage 4.

### C. Methodology and rigour

**Q: Is fine-tuning on predicted embeddings a form of leakage?**
No. Adaptation used **train-split** predictions only. Stage gating used
validation. The test split was touched exactly once, after the decoder was
frozen — and test matched validation within ±0.015, which is the evidence that
nothing overfit.

**Q: How were the complexity tiers defined? Isn't "complexity" subjective?**
Defined from caption statistics, not opinion: agreement (mean pairwise LaBSE
cosine), length, and IDF rarity, combined as z-scores and split at terciles. It
is validated *post hoc* by the fact that every caption metric and the predictor's
own alignment decline monotonically across the tiers, on both validation and test.

**Q: Why is the caption bank keyed the way it is / how do you know the pipeline
measures the same thing as your original eval?**
The independently reimplemented retrieval in this phase reproduced **19.7% R@1**
on validation — exactly the project's original number. That cross-check validates
the whole export/eval harness.

**Q: What is LaBSE-sim and why report it alongside BLEU?**
Cosine similarity between the generated caption's LaBSE embedding and the closest
reference caption's. BLEU measures n-gram overlap and punishes valid paraphrases;
LaBSE-sim measures semantic adequacy in the same space the model operates in.
Reporting both prevents each metric's blind spot from dominating.

### D. About the numbers

**Q: Why did R@1 drop from 19.7% to 6.6% on test? Did the model get worse?**
No — the candidate pool grew from 497 to 2,990. pred_cos was unchanged
(0.594 vs 0.597) and the size-matched control on test returned **18.5%** (§6.3).
Retrieval is gallery-dependent; captioning is not.

**Q: 19.7% R@1 sounds low.**
Two responses. (1) It is pool-dependent, as just shown. (2) It understates the
semantic content the predictor captures: the same embeddings that retrieve at
~19% decode into captions at 0.78 mean LaBSE-sim, because generation is not
penalised by near-duplicate distractors. Retrieval is the wrong lens for a
crowded caption space.

**Q: Predicted BLEU beat oracle BLEU — is that a bug?**
No, see §6.5 (consensus effect); semantic metrics still order correctly.

**Q: Why is the hard tier still worse? Can more decoder training fix it?**
No — it is predictor-limited (§6.1). Closing it requires a stronger predictor
(more capacity, more data, better temporal features), which is stated as future
work rather than papered over.

**Q: How does this compare to published MSR-VTT captioning results?**
Deliberately not claimed as state of the art. The comparisons that matter here
are internal and controlled: 0.073 → 0.515 with the predictor unchanged, and a
monotonic tier gradient. Published numbers use different splits, different
reference handling and end-to-end trained language models, so a direct table
would be misleading.

### E. Novelty and contribution

**Q: What is new here compared to Meta's V-JEPA 2 / VL-JEPA?**
Four things: (1) a **multilingual target space** (LaBSE) giving zero-shot
cross-lingual transfer without extra training; (2) **selective decoding** for
streaming efficiency (3.1×); (3) the **curriculum-staged, noise-injected,
prediction-adapted decoding** scheme built in this phase; (4) the **complexity-tier
methodology** and the accompanying analysis separating captioning difficulty from
retrieval difficulty. The noise-injection idea is credited to CapDec
(Nukrai et al., 2022); the staging, tier construction, and analyses are ours.

**Q: What would you do with more compute?**
Scale the predictor and re-run the same curriculum (the analysis says the
predictor is now the bottleneck); try a larger decoder base; extend the tiers to
a second dataset to test whether the complexity signal transfers.

### F. Limitations — state these before you are asked

- Captions are short and sometimes generic; mT5-small occasionally invents
  morphology ("coachrests") — a decoder-capacity limit, not a predictor limit.
- Hard-tier quality is bounded by predictor alignment (0.578 cos).
- Per-chunk demo captions vary within a video; there is no temporal smoothing yet.
- Complexity tiers are derived from caption statistics, so they inherit MSR-VTT's
  annotation style; a video-only complexity signal (motion) is implemented but was
  not required for the reported results.
- Hindi is currently disabled in the demo (Stage 4 work).

---

## Part 8 — Glossary

| Term | Meaning in this project |
|---|---|
| **Embedding** | 768-number vector representing meaning; similarity = cosine |
| **Frozen model** | weights never updated (`requires_grad=False`); fixed feature extractor |
| **JEPA** | Joint-Embedding Predictive Architecture — predicts representations, not pixels/words |
| **Predictor** | the trained transformer mapping video tokens → predicted sentence embedding (ẑ) |
| **Target encoder** | frozen LaBSE, producing the ground-truth embedding (z) |
| **pred_cos** | cosine between ẑ and the closest true caption embedding — predictor quality |
| **cap_sim / LaBSE-sim** | cosine between generated caption and closest reference — caption quality |
| **Oracle mode** | decoding from ground-truth embeddings — the decoder's upper bound |
| **Predicted mode** | decoding from the predictor's real output — the actual system |
| **Noise injection** | training on GT embedding + Gaussian noise (σ=0.035) for robustness |
| **Prediction adaptation** | training on the predictor's real outputs (50% of samples) |
| **LoRA** | low-rank adapters; trains 9.9M params instead of all of mT5-small |
| **Prefix tokens** | the 16 pseudo encoder tokens the embedding is expanded into |
| **Selective decoding** | fire the decoder only when embedding drift > τ (0.10) |
| **Tier (easy/medium/hard)** | complexity level from caption agreement, length, rarity |
| **R@1** | share of videos whose own caption ranks first among all candidates |
| **Distractor density** | similarity to the nearest *other* video's caption — crowding |

---

## Part 9 — Reproduction: files and commands

New files added in this phase (project root):

| File | Purpose |
|---|---|
| `curriculum_common.py` | shared caption loading / split assignment / bank I/O |
| `build_complexity_tiers.py` | scores complexity, writes `tiers.json` + `caption_bank.npz` |
| `export_predicted_embeddings.py` | runs the predictor over cached features → `preds.npz` |
| `train_decoder_en.py` | curriculum decoder trainer (noise + prediction adaptation) |
| `eval_captions.py` | per-tier BLEU / ROUGE-L / LaBSE-sim reports |
| `list_correct_videos.py` | per-video ranking + safe demo shortlist |
| `tier_pred_quality.py` | per-tier predictor quality (pred_cos, R@1), `--pool_size` control |
| `tier_distractor_density.py` | per-tier caption-space crowding |

Full pipeline:

```bash
# 1. tiers + caption bank
python build_complexity_tiers.py --captions data/msrvtt_train.json data/msrvtt_val.json data/msrvtt_test.json

# 2. predictor outputs for train+val
python export_predicted_embeddings.py --bank curriculum/caption_bank.npz --tiers curriculum/tiers.json --ckpt checkpoints/predictor_e5.pt

# 3. Stage 1 (easy, noise only)
python train_decoder_en.py --bank curriculum/caption_bank.npz --tiers curriculum/tiers.json --include easy --epochs 8 --out checkpoints/dec_en_easy

# 4. Stage 1b (adapt to real predictions)
python train_decoder_en.py --bank curriculum/caption_bank.npz --tiers curriculum/tiers.json --include easy --init_from checkpoints/dec_en_easy/decoder.pt --pred_npz curriculum/preds.npz --pred_ratio 0.5 --noise_std 0.035 --lr 1e-4 --epochs 4 --out checkpoints/dec_en_easy_pred

# 5. Stage 2 (+ medium)
python train_decoder_en.py --bank curriculum/caption_bank.npz --tiers curriculum/tiers.json --include easy,medium --init_from checkpoints/dec_en_easy_pred/decoder.pt --pred_npz curriculum/preds.npz --pred_ratio 0.5 --noise_std 0.035 --lr 1e-4 --epochs 5 --out checkpoints/dec_en_easymed

# 6. Stage 3 (+ hard)  -> FINAL DECODER
python train_decoder_en.py --bank curriculum/caption_bank.npz --tiers curriculum/tiers.json --include easy,medium,hard --init_from checkpoints/dec_en_easymed/decoder.pt --pred_npz curriculum/preds.npz --pred_ratio 0.5 --noise_std 0.035 --lr 1e-4 --epochs 5 --out checkpoints/dec_en_all

# 7. one-shot test evaluation
python export_predicted_embeddings.py --bank curriculum/caption_bank.npz --tiers curriculum/tiers.json --ckpt checkpoints/predictor_e5.pt --splits test --out curriculum/preds_test.npz
python eval_captions.py --bank curriculum/caption_bank.npz --tiers curriculum/tiers.json --ckpt checkpoints/dec_en_all/decoder.pt --preds curriculum/preds_test.npz --split test --out results/final_test_alltiers.md

# 8. live demo
uvicorn server.app:app --port 8000     # DECODER_CKPT = checkpoints/dec_en_all/decoder.pt
```

Evidence files in `results/`: `easy_oracle.md`, `easy_pred_noise_only.md`,
`easy_pred.md`, `easy_oracle_v2.md`, `stage1_pred_alltiers.md`,
`stage2_pred_alltiers.md`, `stage3_pred_alltiers.md`, `final_test_alltiers.md`,
`easy_correct_videos.md`, `demo_shortlist_final.md`.

---

## Part 10 — Suggested order for the meeting

1. **Show the failure first** — open `results/easy_pred_noise_only.md`, read one
   broken caption aloud. Progress is only legible against the old behaviour.
2. **State the diagnosis in one sentence** — decoder trained on ground-truth
   embeddings, fed predictor outputs; measured at 0.073 BLEU.
3. **Live demo** — one easy, one medium, one hard video from
   `demo_shortlist_final.md`.
4. **The ablation table** (§5.1) — 0.073 → 0.515, predictor unchanged.
5. **The curriculum progression matrix** (§5.2) — his own staged-complexity idea,
   validated with a monotonic gradient and no forgetting.
6. **The final test table** (§5.3) — one-shot, matches validation.
7. **Limitations and next steps** (§7F, Stage 4 Hindi) — stated proactively.

---

## Part 11 — What remains

- Thesis chapter write-up using the tables above.
- Update the 4-page IEEE paper with the curriculum method and results.
- **Stage 4 — Hindi:** rebuild a caption bank from the Hindi annotation files,
  fine-tune `dec_en_all` on Hindi targets (LaBSE's shared space is what makes
  this cheap), and re-enable `caption_hi` in `server/app.py`.
- Optional: decoder scaling on Kaggle GPUs; a stronger predictor, which the
  analysis identifies as the current bottleneck.
