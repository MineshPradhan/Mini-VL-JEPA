# Caption evaluation - test split, predicted (preds_test.npz)
Checkpoint: `checkpoints/dec_en_all/decoder.pt` | beams=5

| tier | videos | BLEU-4 | ROUGE-L | LaBSE-sim |
|---|---|---|---|---|
| easy | 992 | 0.5456 | 0.7118 | 0.8609 |
| medium | 966 | 0.3990 | 0.5942 | 0.7842 |
| hard | 1032 | 0.3040 | 0.5033 | 0.7093 |
| **all** | 2990 | 0.4117 | 0.6018 | 0.7838 |

## easy tier examples
- **Test/video7013.mp4** gen: "a man is giving a speech"  
  ref: "a man is teaching math" (LaBSE-sim 0.797)
- **Test/video7014.mp4** gen: "a girl is singing on stage"  
  ref: "a girl and the judges talking on the the voice" (LaBSE-sim 0.763)
- **Test/video7021.mp4** gen: "a man is playing a baseball game"  
  ref: "a baseball batter hits the ball to the fence and a outfielder goes after it" (LaBSE-sim 0.928)
- **Test/video7025.mp4** gen: "a woman is walking and walking on the road"  
  ref: "a boy running is running without dress" (LaBSE-sim 0.797)
- **Test/video7026.mp4** gen: "a man is talking about a car"  
  ref: "a guy talks about the features of the jeep cherokee" (LaBSE-sim 1.000)
- **Test/video7030.mp4** gen: "a group of people are dancing"  
  ref: "a ballroom dance class" (LaBSE-sim 0.840)
- **Test/video7031.mp4** gen: "a group of girls are dancing"  
  ref: "a family doing exercises indoors" (LaBSE-sim 0.799)
- **Test/video7040.mp4** gen: "a man is drawing a picture on a map"  
  ref: "a boy explaining using a white board and marker" (LaBSE-sim 0.872)

## medium tier examples
- **Test/video7012.mp4** gen: "a man in a black shirt is talking about a situation"  
  ref: "a suit man is talking from a studio with fun" (LaBSE-sim 0.733)
- **Test/video7015.mp4** gen: "a boy is talking to a girl"  
  ref: "a female is laughing while talking to a young male" (LaBSE-sim 0.778)
- **Test/video7016.mp4** gen: "kids are playing candy"  
  ref: "a girl and someone is putting an painted egg in to a water" (LaBSE-sim 0.764)
- **Test/video7017.mp4** gen: "a person is playing minecraft"  
  ref: "a gamer talks about his minecraft experience" (LaBSE-sim 0.835)
- **Test/video7018.mp4** gen: "a man in a black shirt is talking about a movie"  
  ref: "a man dances in a room and flips through photos" (LaBSE-sim 0.589)
- **Test/video7020.mp4** gen: "a person is making a piece of paper"  
  ref: "a person is preparing some art" (LaBSE-sim 0.794)
- **Test/video7029.mp4** gen: "a cartoon character is talking to each other"  
  ref: "a cartoon character picks up a phone" (LaBSE-sim 0.824)
- **Test/video7032.mp4** gen: "a man is working on a car"  
  ref: "a guy working on his engine with multiple parts" (LaBSE-sim 0.938)

## hard tier examples
- **Test/video7010.mp4** gen: "a man is using a computer program"  
  ref: "a clip showing a computer screen which contains instructions" (LaBSE-sim 0.701)
- **Test/video7011.mp4** gen: "a cartoon character is talking about a video game"  
  ref: "a animation is shown to help learn colors" (LaBSE-sim 0.675)
- **Test/video7019.mp4** gen: "a man in glasses is sitting in a chair"  
  ref: "a man with suit sitting on the chair talking in front of the camera" (LaBSE-sim 0.743)
- **Test/video7022.mp4** gen: "a woman is walking on a runway"  
  ref: "there is a brown hair woman walking on the ramp" (LaBSE-sim 0.798)
- **Test/video7023.mp4** gen: "a woman is applying makeup on her face"  
  ref: "the girl puts foundation to her cute face with a sponge" (LaBSE-sim 0.864)
- **Test/video7024.mp4** gen: "a person is playing with a toy"  
  ref: "a toy cat is bathing in soapy water in a toy bathtub" (LaBSE-sim 0.814)
- **Test/video7027.mp4** gen: "a man is talking about a video game"  
  ref: "there is graphics of the universe in blue showing in the tv" (LaBSE-sim 0.680)
- **Test/video7028.mp4** gen: "a man is talking to a group of people in a street"  
  ref: "a boy band sings and dances in front of a chinese pagoda" (LaBSE-sim 0.768)