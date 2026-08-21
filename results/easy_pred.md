# Caption evaluation - validate split, predicted (preds.npz)
Checkpoint: `checkpoints/dec_en_easy_pred/decoder.pt` | beams=5

| tier | videos | BLEU-4 | ROUGE-L | LaBSE-sim |
|---|---|---|---|---|
| easy | 175 | 0.5145 | 0.6881 | 0.8457 |
| **all** | 175 | 0.5145 | 0.6881 | 0.8457 |

## easy tier examples
- **Val/video6518.mp4** gen: "two wrestlers are wrestling"  
  ref: "2 men are in a wrestling match" (LaBSE-sim 0.856)
- **Val/video6522.mp4** gen: "a video game is playing on a screen"  
  ref: "a cartoon band plays for a guitar hero video" (LaBSE-sim 0.650)
- **Val/video6525.mp4** gen: "a man is talking about a movie"  
  ref: "a man discusses space dust" (LaBSE-sim 0.790)
- **Val/video6527.mp4** gen: "a woman is showing her hair hair"  
  ref: "a black woman does her hair" (LaBSE-sim 0.853)
- **Val/video6532.mp4** gen: "a person jumping on a trampoline"  
  ref: "a boy does a few flips on a trampoline" (LaBSE-sim 0.838)
- **Val/video6534.mp4** gen: "a horse is riding on horse"  
  ref: "a brown horse is jumping over some hurdles" (LaBSE-sim 0.841)
- **Val/video6541.mp4** gen: "a man is playing a soccer player"  
  ref: "a comparison of two opposing team football athletes" (LaBSE-sim 0.846)
- **Val/video6544.mp4** gen: "a group of people are talking to a stage"  
  ref: "a drama on the stage show" (LaBSE-sim 0.710)