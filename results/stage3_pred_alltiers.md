# Caption evaluation - validate split, predicted (preds.npz)
Checkpoint: `checkpoints/dec_en_all/decoder.pt` | beams=5

| tier | videos | BLEU-4 | ROUGE-L | LaBSE-sim |
|---|---|---|---|---|
| easy | 175 | 0.5329 | 0.6949 | 0.8526 |
| medium | 150 | 0.3937 | 0.5869 | 0.7829 |
| hard | 172 | 0.3110 | 0.5107 | 0.7130 |
| **all** | 497 | 0.4120 | 0.5985 | 0.7832 |

## easy tier examples
- **Val/video6518.mp4** gen: "two men are wrestling in a wrestling match"  
  ref: "2 men are in a wrestling match" (LaBSE-sim 0.932)
- **Val/video6522.mp4** gen: "a cartoon character plays on a screen"  
  ref: "a cartoon band plays for a guitar hero video" (LaBSE-sim 0.683)
- **Val/video6525.mp4** gen: "a man is talking about a movie"  
  ref: "a man discusses space dust" (LaBSE-sim 0.790)
- **Val/video6527.mp4** gen: "a woman is explaining how to make her hair"  
  ref: "a black woman does her hair" (LaBSE-sim 0.917)
- **Val/video6532.mp4** gen: "a boy jumping on a trampoline"  
  ref: "a boy does a few flips on a trampoline" (LaBSE-sim 0.937)
- **Val/video6534.mp4** gen: "a horse is riding on horse"  
  ref: "a brown horse is jumping over some hurdles" (LaBSE-sim 0.841)
- **Val/video6541.mp4** gen: "a man is playing a football game"  
  ref: "a comparison of two opposing team football athletes" (LaBSE-sim 0.849)
- **Val/video6544.mp4** gen: "a man is talking to a group of people"  
  ref: "a drama on the stage show" (LaBSE-sim 0.814)

## medium tier examples
- **Val/video6513.mp4** gen: "a group of people are talking"  
  ref: "a family is having conversation" (LaBSE-sim 0.780)
- **Val/video6514.mp4** gen: "a person is surfing on a beach while a man is riding on the water"  
  ref: "a girl is surfing and a guy is riding a bike on mountains" (LaBSE-sim 0.763)
- **Val/video6519.mp4** gen: "a cartoon character is playing with a little cartoon"  
  ref: "a animate character are talking" (LaBSE-sim 0.650)
- **Val/video6521.mp4** gen: "a man is talking to an audience"  
  ref: "a comedian is talking" (LaBSE-sim 0.779)
- **Val/video6530.mp4** gen: "a car is racing on the road"  
  ref: "a car and three persons are standing outside in a cartoon show" (LaBSE-sim 0.721)
- **Val/video6531.mp4** gen: "two news reporters are talking on a tv show"  
  ref: "a live tv news show with a problem because the man cannot hear the woman" (LaBSE-sim 0.948)
- **Val/video6536.mp4** gen: "a group of players are playing soccer"  
  ref: "a group of soccer players are posing with kids" (LaBSE-sim 0.842)
- **Val/video6537.mp4** gen: "a man is talking in a car"  
  ref: "two men in seat belts ride in an acid green car" (LaBSE-sim 0.715)

## hard tier examples
- **Val/video6515.mp4** gen: "a man is cutting vegetables in a kitchen"  
  ref: "a man cuts a bell pepper and describes how to easily cut it and he crushes garlic and" (LaBSE-sim 0.862)
- **Val/video6516.mp4** gen: "a man is shown about a woman"  
  ref: "a before treatment and after treatment of a man s nose" (LaBSE-sim 0.582)
- **Val/video6517.mp4** gen: "a man is talking about the movie"  
  ref: "a girl and man look at the sky in amazement" (LaBSE-sim 0.681)
- **Val/video6520.mp4** gen: "a woman is running on a track"  
  ref: "a female athlete speeds down a track and jumps into the air" (LaBSE-sim 0.918)
- **Val/video6523.mp4** gen: "a woman is talking to a man"  
  ref: "a man wearing a hoody dress removes the headset from his head" (LaBSE-sim 0.753)
- **Val/video6524.mp4** gen: "a man is cooking a dish in a pan"  
  ref: "ingredients are put in a pan with oil to boil it" (LaBSE-sim 0.873)
- **Val/video6526.mp4** gen: "a woman is standing in the background"  
  ref: "there is a hot woman is getting ready to surf" (LaBSE-sim 0.563)
- **Val/video6528.mp4** gen: "a man in a white shirt is talking about a movie"  
  ref: "a man in a white shirt and dark suit jacket is talking about merging living and retail space" (LaBSE-sim 0.652)