# Caption evaluation - validate split, oracle (GT caption embedding)
Checkpoint: `checkpoints/dec_en_easy/decoder.pt` | beams=5

| tier | videos | BLEU-4 | ROUGE-L | LaBSE-sim |
|---|---|---|---|---|
| easy | 175 | 0.4003 | 0.7093 | 0.8972 |
| medium | 150 | 0.2799 | 0.5977 | 0.8388 |
| hard | 172 | 0.2008 | 0.5113 | 0.7960 |
| **all** | 497 | 0.2821 | 0.6071 | 0.8446 |

## easy tier examples
- **Val/video6518.mp4** gen: "two men are in wrestling match"  
  ref: "2 men are in a wrestling match" (LaBSE-sim 0.949)
- **Val/video6522.mp4** gen: "a video video band plays to a guitar hero"  
  ref: "a cartoon band plays for a guitar hero video" (LaBSE-sim 0.923)
- **Val/video6525.mp4** gen: "a man discusses a space space"  
  ref: "a man discusses space dust" (LaBSE-sim 0.937)
- **Val/video6527.mp4** gen: "a woman hairs her hair"  
  ref: "a black woman does her hair" (LaBSE-sim 0.947)
- **Val/video6532.mp4** gen: "a boy jumps on a trampoline"  
  ref: "a boy does a few flips on a trampoline" (LaBSE-sim 0.960)
- **Val/video6534.mp4** gen: "a horse is riding down a bumps"  
  ref: "a brown horse is jumping over some hurdles" (LaBSE-sim 0.884)
- **Val/video6541.mp4** gen: "a match of two football match of a football match"  
  ref: "a comparison of two opposing team football athletes" (LaBSE-sim 0.749)
- **Val/video6544.mp4** gen: "a stage on stage"  
  ref: "a drama on the stage show" (LaBSE-sim 0.793)

## medium tier examples
- **Val/video6513.mp4** gen: "a family is talking"  
  ref: "a family is having conversation" (LaBSE-sim 0.941)
- **Val/video6514.mp4** gen: "a girl is riding a bike and riding on a mountain bike"  
  ref: "a girl is surfing and a guy is riding a bike on mountains" (LaBSE-sim 0.896)
- **Val/video6519.mp4** gen: "an animated character is talking"  
  ref: "a animate character are talking" (LaBSE-sim 0.969)
- **Val/video6521.mp4** gen: "a comedy is talking"  
  ref: "a comedian is talking" (LaBSE-sim 0.951)
- **Val/video6530.mp4** gen: "three people and a car are standing in a show"  
  ref: "a car and three persons are standing outside in a cartoon show" (LaBSE-sim 0.897)
- **Val/video6531.mp4** gen: "a television news show with a woman waiting to a tv news"  
  ref: "a live tv news show with a problem because the man cannot hear the woman" (LaBSE-sim 0.752)
- **Val/video6536.mp4** gen: "a group of children are children are playing with children"  
  ref: "a group of soccer players are posing with kids" (LaBSE-sim 0.805)
- **Val/video6537.mp4** gen: "two men in a red car are riding a green car in the red car"  
  ref: "two men in seat belts ride in an acid green car" (LaBSE-sim 0.805)

## hard tier examples
- **Val/video6515.mp4** gen: "a man is describes how to cut it and cut it"  
  ref: "a man cuts a bell pepper and describes how to easily cut it and he crushes garlic and" (LaBSE-sim 0.789)
- **Val/video6516.mp4** gen: "a man and a person and treatment of a patient"  
  ref: "a before treatment and after treatment of a man s nose" (LaBSE-sim 0.760)
- **Val/video6517.mp4** gen: "a man and a girl are looking on a plane"  
  ref: "a girl and man look at the sky in amazement" (LaBSE-sim 0.862)
- **Val/video6520.mp4** gen: "a woman jumps down a airplane and jumps off the airplane"  
  ref: "a female athlete speeds down a track and jumps into the air" (LaBSE-sim 0.810)
- **Val/video6523.mp4** gen: "a man wears a hat wears his head of a shirt wears off his head"  
  ref: "a man wearing a hoody dress removes the headset from his head" (LaBSE-sim 0.869)
- **Val/video6524.mp4** gen: "ingredients are stired oil to ingredients"  
  ref: "ingredients are put in a pan with oil to boil it" (LaBSE-sim 0.781)
- **Val/video6526.mp4** gen: "there is a woman is preparing to ready"  
  ref: "there is a hot woman is getting ready to surf" (LaBSE-sim 0.825)
- **Val/video6528.mp4** gen: "a man and a black man in a white clothing and talking about clothes and wearing"  
  ref: "a man in a white shirt and dark suit jacket is talking about merging living and retail space" (LaBSE-sim 0.776)