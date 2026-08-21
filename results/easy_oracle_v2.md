# Caption evaluation - validate split, oracle (GT caption embedding)
Checkpoint: `checkpoints/dec_en_easy_pred/decoder.pt` | beams=5

| tier | videos | BLEU-4 | ROUGE-L | LaBSE-sim |
|---|---|---|---|---|
| easy | 175 | 0.4503 | 0.7176 | 0.8923 |
| medium | 150 | 0.2950 | 0.5991 | 0.8289 |
| hard | 172 | 0.2310 | 0.5265 | 0.7913 |
| **all** | 497 | 0.3196 | 0.6157 | 0.8382 |

## easy tier examples
- **Val/video6518.mp4** gen: "two men are in a men"  
  ref: "2 men are in a wrestling match" (LaBSE-sim 0.819)
- **Val/video6522.mp4** gen: "a video video game plays for a guitar hero"  
  ref: "a cartoon band plays for a guitar hero video" (LaBSE-sim 0.862)
- **Val/video6525.mp4** gen: "a man discusses a space space"  
  ref: "a man discusses space dust" (LaBSE-sim 0.937)
- **Val/video6527.mp4** gen: "a woman is making her hair"  
  ref: "a black woman does her hair" (LaBSE-sim 0.993)
- **Val/video6532.mp4** gen: "a boy flips a flip on a trampoline"  
  ref: "a boy does a few flips on a trampoline" (LaBSE-sim 0.913)
- **Val/video6534.mp4** gen: "a black horse is jumping through a bridge"  
  ref: "a brown horse is jumping over some hurdles" (LaBSE-sim 0.782)
- **Val/video6541.mp4** gen: "a match of a football match of two players"  
  ref: "a comparison of two opposing team football athletes" (LaBSE-sim 0.814)
- **Val/video6544.mp4** gen: "a stage on stage"  
  ref: "a drama on the stage show" (LaBSE-sim 0.793)

## medium tier examples
- **Val/video6513.mp4** gen: "a family is having conversation"  
  ref: "a family is having conversation" (LaBSE-sim 1.000)
- **Val/video6514.mp4** gen: "a girl is girl is jumping bikes on a bike"  
  ref: "a girl is surfing and a guy is riding a bike on mountains" (LaBSE-sim 0.812)
- **Val/video6519.mp4** gen: "an animated character is talking"  
  ref: "a animate character are talking" (LaBSE-sim 0.969)
- **Val/video6521.mp4** gen: "a comedy is talking"  
  ref: "a comedian is talking" (LaBSE-sim 0.951)
- **Val/video6530.mp4** gen: "three people and a car are standing in a show"  
  ref: "a car and three persons are standing outside in a cartoon show" (LaBSE-sim 0.897)
- **Val/video6531.mp4** gen: "a television show with a woman waiting to a tv show"  
  ref: "a live tv news show with a problem because the man cannot hear the woman" (LaBSE-sim 0.697)
- **Val/video6536.mp4** gen: "a group of kids are playing with children with children"  
  ref: "a group of soccer players are posing with kids" (LaBSE-sim 0.835)
- **Val/video6537.mp4** gen: "two men are riding in a red car riding a green car"  
  ref: "two men in seat belts ride in an acid green car" (LaBSE-sim 0.849)

## hard tier examples
- **Val/video6515.mp4** gen: "a man is explaining how to cut it and explains"  
  ref: "a man cuts a bell pepper and describes how to easily cut it and he crushes garlic and" (LaBSE-sim 0.774)
- **Val/video6516.mp4** gen: "a man and a woman is talking about a person"  
  ref: "a before treatment and after treatment of a man s nose" (LaBSE-sim 0.598)
- **Val/video6517.mp4** gen: "a man and a girl are looking at a plane"  
  ref: "a girl and man look at the sky in amazement" (LaBSE-sim 0.881)
- **Val/video6520.mp4** gen: "a woman jumps down a airplane and jumping down the airplane"  
  ref: "a female athlete speeds down a track and jumps into the air" (LaBSE-sim 0.810)
- **Val/video6523.mp4** gen: "a man wears a head off a shirt"  
  ref: "a man wearing a hoody dress removes the headset from his head" (LaBSE-sim 0.815)
- **Val/video6524.mp4** gen: "a pan is being stired in a oil"  
  ref: "ingredients are put in a pan with oil to boil it" (LaBSE-sim 0.688)
- **Val/video6526.mp4** gen: "there is a woman is preparing to prepared"  
  ref: "there is a hot woman is getting ready to surf" (LaBSE-sim 0.816)
- **Val/video6528.mp4** gen: "a man in a white shirt and talking about a black shirt"  
  ref: "a man in a white shirt and dark suit jacket is talking about merging living and retail space" (LaBSE-sim 0.720)