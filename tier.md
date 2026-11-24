# Video tiers - choosing clips for the Mini VL-JEPA demo

Every held-out MSR-VTT video (Val 497 + Test 2,990 = 3,487 clips) is pre-captioned by the real system and sorted into **easy / medium / hard**. Use this file to pick clips for the demo: open `http://127.0.0.1:8000/`, go to **Browse the dataset**, filter by tier, and type the video id into the search box (or click it in the list).

Train videos are not offered: the model was fit on them, so their captions would overstate what it can do. Every clip below is one the model never saw.

## What the tiers mean

Tiers come from `build_complexity_tiers.py` and are computed from the ~20 human captions each MSR-VTT video has, not from the model, so they are an independent measure of how hard a clip is to describe:

| signal | meaning | effect |
|---|---|---|
| caption agreement | mean pairwise LaBSE cosine between the 20 human captions | people agree -> simple, unambiguous scene -> **easier** |
| caption length | mean words per human caption | more detail needed -> **harder** |
| vocabulary rarity | mean IDF of the caption words | unusual objects/actions -> **harder** |

`complexity = z(length) + z(rarity) - z(agreement)`; the dataset is split into terciles of this score.

| tier | what the clips look like | what to expect from the captions |
|---|---|---|
| **easy** | one clear subject and action: someone cooking, playing a video game, singing, driving | short, accurate captions; usually match a human caption almost word for word |
| **medium** | a clear activity with more context: sports matches, tutorials, interviews, vlogs | the main activity is right; details are sometimes generic |
| **hard** | busy or unusual scenes, montages, rare objects, several things happening | the gist (person / setting / activity type) is usually right; specifics are often missed |

## How well the system does on each tier

`cap_sim` = LaBSE cosine between the generated caption and the closest of the ~20 human captions (1.0 = identical meaning). Same metric as RESULTS.md.

| tier | clips | mean cap_sim | cap_sim >= 0.8 | cap_sim >= 0.9 |
|---|---|---|---|---|
| easy | 1,167 | 0.860 | 72% | 36% |
| medium | 1,116 | 0.784 | 43% | 12% |
| hard | 1,204 | 0.710 | 19% | 2% |
| **all** | 3,487 | 0.784 | 45% | 17% |

The gradient easy > medium > hard is the curriculum result from RESULTS.md (test BLEU-4 0.546 / 0.399 / 0.304), reproduced here on every held-out clip.

## Presenting it

1. **Start with easy** clips - captions land almost exactly, which shows the core idea (video -> embedding -> text) works.
2. **Move to medium** - the activity is right even when wording differs from the humans'.
3. **Finish with one hard clip** and say why it is hard (the humans themselves disagree on it). Showing a limit honestly is more convincing than hiding it.
4. Switch the caption language to **हिन्दी** on any clip.
5. The page also shows the human captions, so the audience can judge the match.

The live page decodes three candidates (short / standard / detailed decoding styles - not the same thing as the clip tiers) and picks the one best grounded in the predicted embedding, so its caption can differ slightly from the `caption` column below, which uses the single evaluation setting.

## Easy tier - recommended clips

Best 25 of 1,167 easy clips by cap_sim, one clip per distinct caption.

| # | video | split | model caption | closest human caption | cap_sim |
|---|---|---|---|---|---|
| 1 | `video6912` | Val | a man is talking about a car | a man is talking about a car | 1.000 |
| 2 | `video6624` | Val | a car is shown | a car is shown | 1.000 |
| 3 | `video6596` | Val | a person is playing a video game | a person is playing a video game | 1.000 |
| 4 | `video6920` | Val | a man is giving a speech | a man is giving a speech | 1.000 |
| 5 | `video6857` | Val | a person is playing with a toy | a person is playing with a toy | 1.000 |
| 6 | `video6550` | Val | a man is driving a car | a man is driving a car | 1.000 |
| 7 | `video6939` | Val | a man is playing minecraft | a man is playing minecraft | 1.000 |
| 8 | `video6782` | Val | a woman is singing on stage | a woman is singing on stage | 1.000 |
| 9 | `video9952` | Test | a person is folding paper | a person is folding paper | 1.000 |
| 10 | `video8512` | Test | a man is talking about something | a man is talking about something | 1.000 |
| 11 | `video7104` | Test | a man is playing a guitar | a man is playing a guitar | 1.000 |
| 12 | `video8642` | Test | a girl is singing on stage | a girl is singing on stage | 1.000 |
| 13 | `video9987` | Test | a person is folding a piece of paper | a person is folding a piece of paper | 1.000 |
| 14 | `video7972` | Test | a man is playing a video game | a man is playing a video game | 1.000 |
| 15 | `video9925` | Test | a woman is dancing | a woman is dancing | 1.000 |
| 16 | `video9733` | Test | a woman is talking about a car | a woman is talking about a car | 1.000 |
| 17 | `video9445` | Test | a man is working on a car | a man is working on a car | 1.000 |
| 18 | `video7654` | Test | a man is cooking a recipe | a man is cooking a recipe | 1.000 |
| 19 | `video7079` | Test | a band is performing on stage | a band is performing on stage | 1.000 |
| 20 | `video8263` | Test | a person is driving a car | a person is driving a car | 1.000 |
| 21 | `video9193` | Test | a woman is talking about a movie | a woman is talking about a movie | 1.000 |
| 22 | `video8233` | Test | a person is folding a paper airplane | a person is folding a paper airplane | 1.000 |
| 23 | `video9769` | Test | a group of people are dancing | a group of people are dancing | 1.000 |
| 24 | `video8041` | Test | a person is playing a piano | a person is playing a piano | 1.000 |
| 25 | `video8106` | Test | a man is riding a motorcycle | a man is riding a motorcycle | 1.000 |

## Medium tier - recommended clips

Best 25 of 1,116 medium clips by cap_sim, one clip per distinct caption.

| # | video | split | model caption | closest human caption | cap_sim |
|---|---|---|---|---|---|
| 1 | `video6934` | Val | a news reporter is talking about the news | a news reporter is talking about the news | 1.000 |
| 2 | `video6911` | Val | a woman is cooking | a woman is cooking | 1.000 |
| 3 | `video8898` | Test | a person is playing a video game | a person is playing a video game | 1.000 |
| 4 | `video7367` | Test | a man is singing | a man is singing | 1.000 |
| 5 | `video8391` | Test | a man is playing minecraft | a man is playing minecraft | 1.000 |
| 6 | `video9513` | Test | a woman is talking | a woman is talking | 1.000 |
| 7 | `video7250` | Test | a woman is riding a horse | a woman is riding a horse | 1.000 |
| 8 | `video9204` | Test | a woman is talking about the news | a woman is talking about the news | 1.000 |
| 9 | `video9143` | Test | a girl is playing with a toy | a girl is playing with a toy | 1.000 |
| 10 | `video8573` | Test | a man is talking | a man is talking | 1.000 |
| 11 | `video9629` | Test | a person is driving a car | a person is driving a car | 1.000 |
| 12 | `video9981` | Test | a cartoon character is flying | a cartoon character is flying | 1.000 |
| 13 | `video9487` | Test | someone is playing a video game | someone is playing video game | 0.998 |
| 14 | `video9650` | Test | a man is playing a basketball | a man is playing basketball | 0.998 |
| 15 | `video8345` | Test | a woman is eating a food | a woman is eating food | 0.998 |
| 16 | `video9522` | Test | a man is talking on stage | a man is talking on the stage | 0.998 |
| 17 | `video7943` | Test | a motorcycle is being shown | a motorcycle is being displayed | 0.995 |
| 18 | `video7983` | Test | a car is being shown | a car is being displayed | 0.995 |
| 19 | `video8152` | Test | a car is shown | a car is being shown | 0.987 |
| 20 | `video9888` | Test | a band is performing on stage | a band is on stage performing | 0.987 |
| 21 | `video8412` | Test | a man is talking about a animal | a man is talking about animals | 0.987 |
| 22 | `video7357` | Test | two players are playing a tennis match in a court | two players are playing tennis match on the court | 0.986 |
| 23 | `video8393` | Test | a person is drawing a cartoon | a person is drawing cartoons | 0.986 |
| 24 | `video9554` | Test | a man is talking to a woman | a man talking to a woman | 0.982 |
| 25 | `video7703` | Test | a person is explaining how to fix a roof | a man is explaining how to fix a roof | 0.982 |

## Hard tier - recommended clips

Best 25 of 1,204 hard clips by cap_sim, one clip per distinct caption.

| # | video | split | model caption | closest human caption | cap_sim |
|---|---|---|---|---|---|
| 1 | `video7058` | Test | a cartoon character is talking | a cartoon character is talking | 1.000 |
| 2 | `video9653` | Test | a man is talking on a television show | a man is talking on a tv show | 0.994 |
| 3 | `video8739` | Test | a man is working on a car | a man working on a car | 0.976 |
| 4 | `video7655` | Test | a man in a kitchen is preparing a dish | a man is cooking a dish in the kitchen | 0.967 |
| 5 | `video9544` | Test | a man is playing a video game | a person is playing a video game | 0.966 |
| 6 | `video8436` | Test | a person is talking about a toy | a man talks about a toy | 0.966 |
| 7 | `video8995` | Test | a woman is lying on a bed | a woman is lying down on a couch | 0.965 |
| 8 | `video9035` | Test | someone is playing a video game | a person is playing a video game | 0.951 |
| 9 | `video8061` | Test | a man is walking in a street | a man walking down the street | 0.948 |
| 10 | `video7880` | Test | a woman is showing how to prepare a dish | a woman is showing how to cook | 0.945 |
| 11 | `video7501` | Test | two men are talking about a movie | a two men are talking about a movie scene | 0.941 |
| 12 | `video9718` | Test | a person is explaining how to use a cell phone | a person is explaining about a mobile phone usage | 0.930 |
| 13 | `video9109` | Test | a woman is talking about the news | a woman on the news is talking about a story | 0.927 |
| 14 | `video8682` | Test | a man is talking about his phone | a man is talking about mobile phone | 0.927 |
| 15 | `video6910` | Val | a fireworks are shown | a firework is shown | 0.922 |
| 16 | `video6520` | Val | a woman is running on a track | there is a woman is running on the track | 0.918 |
| 17 | `video7158` | Test | two men are talking about a tv show | two men are having conversation for the television | 0.918 |
| 18 | `video7834` | Test | a man in a blue shirt is talking about the news | person in blue shirt is talking about the latest news | 0.914 |
| 19 | `video6803` | Val | a woman is applying makeup to her face | a woman is showing makeup for the face | 0.911 |
| 20 | `video7636` | Test | a man is talking to another man | a one eyed man is talking with another man | 0.910 |
| 21 | `video6771` | Val | a man is holding a ball | a man is holding a sword | 0.908 |
| 22 | `video6583` | Val | two players are playing a tennis match | two teams of tennis players are playing a tennis | 0.908 |
| 23 | `video9021` | Test | a man is talking about a baseball game | a man is talking about base ball | 0.906 |
| 24 | `video6973` | Val | two players are playing a table tennis game | two players are playing table tennis match in the gym | 0.903 |
| 25 | `video9358` | Test | a black car is being shown in a car | a car which is black color is displayed | 0.903 |

## Full list

All clips with their tier and score are in `web/catalog.json` and in the page's **Browse the dataset** panel (sortable by caption match).

Regenerate this file and the catalog with `python build_catalog.py`.
