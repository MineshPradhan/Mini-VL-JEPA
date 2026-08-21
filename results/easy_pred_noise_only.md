# Caption evaluation - validate split, predicted (preds.npz)
Checkpoint: `checkpoints/dec_en_easy/decoder.pt` | beams=5

| tier | videos | BLEU-4 | ROUGE-L | LaBSE-sim |
|---|---|---|---|---|
| easy | 175 | 0.0731 | 0.3740 | 0.6385 |
| **all** | 175 | 0.0731 | 0.3740 | 0.6385 |

## easy tier examples
- **Val/video6518.mp4** gen: "twowrestlerstlers fightrestrestrestwrestling wrestwrestling matchwrestling"  
  ref: "2 men are in a wrestling match" (LaBSE-sim 0.586)
- **Val/video6522.mp4** gen: "a pokemon game play on a voice"  
  ref: "a cartoon band plays for a guitar hero video" (LaBSE-sim 0.617)
- **Val/video6525.mp4** gen: "a man talks about sunset"  
  ref: "a man discusses space dust" (LaBSE-sim 0.858)
- **Val/video6527.mp4** gen: "a blonde hair hair hair makeup hair makeup makeup hair hairmakeup hair hairbrow hair hair pigment hair makeup pigment hair hair Hair hair makeup her hair hair"  
  ref: "a black woman does her hair" (LaBSE-sim 0.491)
- **Val/video6532.mp4** gen: "a jump jump jumpne jumpne on jumpne"  
  ref: "a boy does a few flips on a trampoline" (LaBSE-sim 0.600)
- **Val/video6534.mp4** gen: "horse horse horse race horse horse Horse horse horse"  
  ref: "a brown horse is jumping over some hurdles" (LaBSE-sim 0.475)
- **Val/video6541.mp4** gen: "football football kicks football football"  
  ref: "a comparison of two opposing team football athletes" (LaBSE-sim 0.604)
- **Val/video6544.mp4** gen: "talk talk on a talk show"  
  ref: "a drama on the stage show" (LaBSE-sim 0.599)