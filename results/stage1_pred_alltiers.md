# Caption evaluation - validate split, predicted (preds.npz)
Checkpoint: `checkpoints/dec_en_easy_pred/decoder.pt` | beams=5

| tier | videos | BLEU-4 | ROUGE-L | LaBSE-sim |
|---|---|---|---|---|
| easy | 175 | 0.5145 | 0.6881 | 0.8457 |
| medium | 150 | 0.3781 | 0.5886 | 0.7876 |
| hard | 172 | 0.2544 | 0.4811 | 0.7004 |
| **all** | 497 | 0.3819 | 0.5864 | 0.7779 |

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

## medium tier examples
- **Val/video6513.mp4** gen: "a group of people are talking on a phone"  
  ref: "a family is having conversation" (LaBSE-sim 0.723)
- **Val/video6514.mp4** gen: "a person is surfing on the surf surf"  
  ref: "a girl is surfing and a guy is riding a bike on mountains" (LaBSE-sim 0.723)
- **Val/video6519.mp4** gen: "a dog is playing with a cartoon"  
  ref: "a animate character are talking" (LaBSE-sim 0.690)
- **Val/video6521.mp4** gen: "a man is talking to a beard"  
  ref: "a comedian is talking" (LaBSE-sim 0.781)
- **Val/video6530.mp4** gen: "a car is racing on the car"  
  ref: "a car and three persons are standing outside in a cartoon show" (LaBSE-sim 0.749)
- **Val/video6531.mp4** gen: "a news reporter and a reporter are talking on a tv news"  
  ref: "a live tv news show with a problem because the man cannot hear the woman" (LaBSE-sim 0.817)
- **Val/video6536.mp4** gen: "a team are playing soccer"  
  ref: "a group of soccer players are posing with kids" (LaBSE-sim 0.821)
- **Val/video6537.mp4** gen: "a man is talking in a car"  
  ref: "two men in seat belts ride in an acid green car" (LaBSE-sim 0.715)

## hard tier examples
- **Val/video6515.mp4** gen: "a man is cutting a potatoes"  
  ref: "a man cuts a bell pepper and describes how to easily cut it and he crushes garlic and" (LaBSE-sim 0.784)
- **Val/video6516.mp4** gen: "a video of a man being shown"  
  ref: "a before treatment and after treatment of a man s nose" (LaBSE-sim 0.790)
- **Val/video6517.mp4** gen: "a man is talking about a movie"  
  ref: "a girl and man look at the sky in amazement" (LaBSE-sim 0.686)
- **Val/video6520.mp4** gen: "a woman is dancing on a sports sports"  
  ref: "a female athlete speeds down a track and jumps into the air" (LaBSE-sim 0.685)
- **Val/video6523.mp4** gen: "a group of a woman is talking to a phone"  
  ref: "a man wearing a hoody dress removes the headset from his head" (LaBSE-sim 0.733)
- **Val/video6524.mp4** gen: "a man is stiring a frying something in a pan"  
  ref: "ingredients are put in a pan with oil to boil it" (LaBSE-sim 0.830)
- **Val/video6526.mp4** gen: "images of pictures of a picture"  
  ref: "there is a hot woman is getting ready to surf" (LaBSE-sim 0.455)
- **Val/video6528.mp4** gen: "a man is talking about a car and talking to a tv show"  
  ref: "a man in a white shirt and dark suit jacket is talking about merging living and retail space" (LaBSE-sim 0.755)