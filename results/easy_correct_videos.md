# Per-video system performance - validate split, tiers: easy
Decoder: `checkpoints/dec_en_easy_pred/decoder.pt` | predictions: `curriculum/preds.npz`

- videos: 175 | retrieval R@1 (pool of 497): **18.9%** | mean cap_sim: 0.846

## Safe demo shortlist - R@1 correct AND cap_sim >= 0.8 (26 videos)

1. **Val/video6893.mp4** - "a man is cooking food in a kitchen" (sim 0.966)
2. **Val/video6937.mp4** - "a woman is talking about a phone" (sim 0.964)
3. **Val/video6922.mp4** - "a person is driving a car" (sim 0.953)
4. **Val/video6971.mp4** - "a person is playing a video game" (sim 0.951)
5. **Val/video6848.mp4** - "a man is talking about a car" (sim 0.942)
6. **Val/video6739.mp4** - "a group of girls are singing on stage" (sim 0.928)
7. **Val/video6745.mp4** - "a man is cooking in a kitchen" (sim 0.927)
8. **Val/video6897.mp4** - "a girl is singing on a voice" (sim 0.925)
9. **Val/video7004.mp4** - "a person is jumping on a trampoline" (sim 0.923)
10. **Val/video6679.mp4** - "a man is explaining how to a computer" (sim 0.915)
11. **Val/video6925.mp4** - "a woman is showing a stroller" (sim 0.911)
12. **Val/video6829.mp4** - "a woman is swimming in a swimming" (sim 0.900)
13. **Val/video6933.mp4** - "a boy is playing in a school" (sim 0.896)
14. **Val/video6845.mp4** - "a person playing a paper airplane" (sim 0.894)
15. **Val/video6780.mp4** - "a person is folding a paper" (sim 0.884)
16. **Val/video6774.mp4** - "a man is eating a food" (sim 0.882)
17. **Val/video6558.mp4** - "a person is playing a cube" (sim 0.865)
18. **Val/video6797.mp4** - "a man is talking to a team" (sim 0.856)
19. **Val/video6629.mp4** - "a man is playing a soccer game" (sim 0.855)
20. **Val/video6527.mp4** - "a woman is showing her hair hair" (sim 0.853)
21. **Val/video6815.mp4** - "a dog is talking about a kitchen" (sim 0.847)
22. **Val/video6814.mp4** - "a woman is talking about a garden" (sim 0.828)
23. **Val/video6967.mp4** - "a race race running on a track" (sim 0.821)
24. **Val/video6957.mp4** - "a child is playing with a baby" (sim 0.816)
25. **Val/video6792.mp4** - "a football player is playing soccer" (sim 0.804)
26. **Val/video6919.mp4** - "a man is playing a guitar" (sim 0.804)

## Full ranking (best first)

| # | video | cap_sim | R@1 | rank | pred_cos | generated | closest reference |
|---|---|---|---|---|---|---|---|
| 1 | Val/video6920.mp4 | 1.000 | - | 103 | 0.616 | a man is giving a speech | a man is giving a speech |
| 2 | Val/video6939.mp4 | 1.000 | - | 4 | 0.648 | a man is playing minecraft | a man is playing minecraft |
| 3 | Val/video6731.mp4 | 1.000 | - | 3 | 0.683 | a man is playing a golf game | a man is playing a golf game |
| 4 | Val/video6912.mp4 | 1.000 | - | 13 | 0.631 | a man is talking about a car | a man is talking about a car |
| 5 | Val/video6811.mp4 | 1.000 | - | 2 | 0.703 | a baby is playing with a cat | a baby is playing with a cat |
| 6 | Val/video6596.mp4 | 1.000 | - | 6 | 0.632 | a person is playing a video game | a person is playing a video game |
| 7 | Val/video6790.mp4 | 1.000 | - | 7 | 0.616 | a person is playing a video game | a person is playing a video game |
| 8 | Val/video6857.mp4 | 1.000 | - | 2 | 0.676 | a person is playing with a toy | a person is playing with a toy |
| 9 | Val/video6550.mp4 | 1.000 | - | 10 | 0.652 | a man is driving a car | a man is driving a car |
| 10 | Val/video6936.mp4 | 0.995 | - | 3 | 0.701 | a girl is doing a gymnastics | a girl is doing gymnastic |
| 11 | Val/video7006.mp4 | 0.994 | - | 14 | 0.613 | a man is dancing on a stage | a man is dancing on stage |
| 12 | Val/video6766.mp4 | 0.993 | - | 2 | 0.660 | kids are dancing | the children are dancing |
| 13 | Val/video6875.mp4 | 0.988 | - | 17 | 0.683 | a woman is showing a stroller | a woman displays a stroller |
| 14 | Val/video6684.mp4 | 0.985 | - | 43 | 0.653 | a band is singing a song | a band is playing a song |
| 15 | Val/video6976.mp4 | 0.984 | - | 13 | 0.680 | a group of girls are singing on stage | a girl group is singing on stage |
| 16 | Val/video6808.mp4 | 0.981 | - | 14 | 0.694 | a man is mixing ingredients into a bowl | a man mixing ingredients into a bowl |
| 17 | Val/video6787.mp4 | 0.980 | - | 3 | 0.673 | a man is cooking food in a kitchen | a man is preparing a meal in a kitchen |
| 18 | Val/video6892.mp4 | 0.975 | - | 3 | 0.592 | a man and a woman is kissing | man and woman are kissing |
| 19 | Val/video6662.mp4 | 0.974 | - | 11 | 0.667 | a person is playing minecraft | a man is playing minecraft |
| 20 | Val/video6574.mp4 | 0.974 | - | 4 | 0.610 | a man is singing on stage | a man singing on a stage |
| 21 | Val/video6989.mp4 | 0.972 | - | 3 | 0.636 | a man is talking in a car | a man is talking inside a car |
| 22 | Val/video6729.mp4 | 0.967 | - | 10 | 0.635 | a person is playing a video game | a person playing a video game |
| 23 | Val/video7001.mp4 | 0.966 | - | 5 | 0.733 | a man is playing a video game | a person is playing a video game |
| 24 | Val/video6893.mp4 | 0.966 | Y | 1 | 0.602 | a man is cooking food in a kitchen | a man is cooking in a kitchen |
| 25 | Val/video6945.mp4 | 0.965 | - | 6 | 0.539 | a man is talking about a tv | a man is talking on tv |
| 26 | Val/video6937.mp4 | 0.964 | Y | 1 | 0.594 | a woman is talking about a phone | a woman talks on the phone |
| 27 | Val/video6760.mp4 | 0.957 | - | 7 | 0.726 | a person is playing minecraft | a person is playing the game minecraft |
| 28 | Val/video6909.mp4 | 0.957 | - | 5 | 0.707 | a woman is applying a makeup | a woman puts on makeup |
| 29 | Val/video6818.mp4 | 0.953 | - | 2 | 0.694 | a soccer player is playing a goal | a soccer player scores a goal |
| 30 | Val/video6922.mp4 | 0.953 | Y | 1 | 0.633 | a person is driving a car | a man is driving in a car |
| 31 | Val/video6806.mp4 | 0.951 | - | 4 | 0.699 | a person is playing a video game | someone is playing a video game |
| 32 | Val/video6971.mp4 | 0.951 | Y | 1 | 0.669 | a person is playing a video game | someone is playing a video game |
| 33 | Val/video6573.mp4 | 0.951 | - | 160 | 0.679 | someone is playing a video game | a person is playing a video game |
| 34 | Val/video6695.mp4 | 0.946 | - | 10 | 0.650 | a man is driving in a car | a man drive a car |
| 35 | Val/video6707.mp4 | 0.945 | - | 4 | 0.627 | a man is showing a car | a man is showing off his car |
| 36 | Val/video6782.mp4 | 0.943 | - | 15 | 0.597 | a woman is performing a song | woman performing a song |
| 37 | Val/video6848.mp4 | 0.942 | Y | 1 | 0.635 | a man is talking about a car | a man is talking about his car |
| 38 | Val/video6750.mp4 | 0.942 | - | 18 | 0.594 | a woman is talking about a camera | a woman talking to the camera |
| 39 | Val/video6708.mp4 | 0.941 | - | 30 | 0.650 | a group of people are dancing | a group of dancing people |
| 40 | Val/video6672.mp4 | 0.938 | - | 51 | 0.607 | a man is talking about a kitchen | a man is talking about something in kitchen |
| 41 | Val/video6739.mp4 | 0.928 | Y | 1 | 0.653 | a group of girls are singing on stage | a group of women singing on stage |
| 42 | Val/video6745.mp4 | 0.927 | Y | 1 | 0.694 | a man is cooking in a kitchen | man cooking food in a kitchen |
| 43 | Val/video6577.mp4 | 0.926 | - | 27 | 0.546 | a woman is talking about her camera | a woman talks to a camera |
| 44 | Val/video6897.mp4 | 0.925 | Y | 1 | 0.614 | a girl is singing on a voice | a woman sings on the voice |
| 45 | Val/video6757.mp4 | 0.925 | - | 4 | 0.688 | a woman is cooking a recipe | woman making a recipe |
| 46 | Val/video7004.mp4 | 0.923 | Y | 1 | 0.753 | a person is jumping on a trampoline | person jumping on trampoline |
| 47 | Val/video6679.mp4 | 0.915 | Y | 1 | 0.625 | a man is explaining how to a computer | a man is explaining how to do something on a computer |
| 48 | Val/video6925.mp4 | 0.911 | Y | 1 | 0.575 | a woman is showing a stroller | a woman is displaying a pillow |
| 49 | Val/video6836.mp4 | 0.909 | - | 3 | 0.734 | a child is playing with a baby | a baby is playing with a cat |
| 50 | Val/video6632.mp4 | 0.908 | - | 3 | 0.666 | a soccer player are playing soccer | soccer players playing soccer |
| 51 | Val/video6992.mp4 | 0.908 | - | 6 | 0.666 | a group of people are singing a song | the crowd is singing a song |
| 52 | Val/video6557.mp4 | 0.905 | - | 21 | 0.670 | a man is cooking a dish | a man preparing a food dish |
| 53 | Val/video6938.mp4 | 0.903 | - | 8 | 0.638 | a man is showing a car | a man is looking at a car |
| 54 | Val/video6829.mp4 | 0.900 | Y | 1 | 0.661 | a woman is swimming in a swimming | a woman is swimming |
| 55 | Val/video6887.mp4 | 0.899 | - | 5 | 0.620 | a group of people are walking down a room | a group of people walk down a staircase |
| 56 | Val/video6746.mp4 | 0.897 | - | 5 | 0.642 | a woman is talking about a baby stroller | a woman talking about a roller stroller |
| 57 | Val/video6582.mp4 | 0.896 | - | 8 | 0.629 | a man is trying a dish in a kitchen | a man cooking food in a kitchen |
| 58 | Val/video6933.mp4 | 0.896 | Y | 1 | 0.677 | a boy is playing in a school | children in a school are playing a game |
| 59 | Val/video6845.mp4 | 0.894 | Y | 1 | 0.708 | a person playing a paper airplane | a man is flying a paper airplane |
| 60 | Val/video6696.mp4 | 0.894 | - | 2 | 0.629 | a girl is driving a car and talking in her car | a girl talks while riding in a car |
| 61 | Val/video6780.mp4 | 0.884 | Y | 1 | 0.694 | a person is folding a paper | a person is folding a paper airplane |
| 62 | Val/video6774.mp4 | 0.882 | Y | 1 | 0.640 | a man is eating a food | people are eating food |
| 63 | Val/video6625.mp4 | 0.882 | - | 8 | 0.713 | a person is playing a cube | someone is playing with cubes |
| 64 | Val/video6628.mp4 | 0.882 | - | 2 | 0.751 | a woman is showing a stroller | a woman demonstrating a baby stroller |
| 65 | Val/video6807.mp4 | 0.878 | - | 3 | 0.693 | a man is applying a eyebrow | a man is applying make up |
| 66 | Val/video6699.mp4 | 0.875 | - | 28 | 0.532 | a group of people are talking on a couch | a group of people sit on a couch together |
| 67 | Val/video6643.mp4 | 0.871 | - | 82 | 0.579 | a man is talking to a audience | a guy is giving a speech in front of an audience |
| 68 | Val/video6606.mp4 | 0.869 | - | 6 | 0.619 | a woman is talking about a kitchen | a woman talking about baking |
| 69 | Val/video6891.mp4 | 0.869 | - | 13 | 0.589 | a man is making a dish in a bowl | a man is doing a dish |
| 70 | Val/video6748.mp4 | 0.867 | - | 5 | 0.629 | a woman is talking about a shop | a woman is walking and talking in a store |
| 71 | Val/video6558.mp4 | 0.865 | Y | 1 | 0.707 | a person is playing a cube | a person is playing with a rubik cube |
| 72 | Val/video6616.mp4 | 0.863 | - | 398 | 0.554 | a man is working on a computer | a man is putting together a computer |
| 73 | Val/video6765.mp4 | 0.860 | - | 9 | 0.666 | a man is playing a football game | a person is playing a sport |
| 74 | Val/video6518.mp4 | 0.856 | - | 2 | 0.683 | two wrestlers are wrestling | two men are wrestling |
| 75 | Val/video6797.mp4 | 0.856 | Y | 1 | 0.608 | a man is talking to a team | a man is speaking to a group of others |
| 76 | Val/video6629.mp4 | 0.855 | Y | 1 | 0.612 | a man is playing a soccer game | a man is training for football |
| 77 | Val/video6901.mp4 | 0.855 | - | 7 | 0.604 | a man is talking about a restaurant | a man is talking about a drink |
| 78 | Val/video6880.mp4 | 0.854 | - | 3 | 0.684 | a man is giving a audience | a man is giving a lecture |
| 79 | Val/video6744.mp4 | 0.853 | - | 28 | 0.622 | two people are talking on a stage | two men speak on stage in front of an audience |
| 80 | Val/video6527.mp4 | 0.853 | Y | 1 | 0.649 | a woman is showing her hair hair | a woman is doing her hair |
| 81 | Val/video6822.mp4 | 0.852 | - | 2 | 0.673 | a car is driving on a cars | a car is racing down a road |
| 82 | Val/video6669.mp4 | 0.849 | - | 6 | 0.617 | a woman is playing a concert | a woman is sing on stage |
| 83 | Val/video7007.mp4 | 0.847 | - | 306 | 0.571 | a woman is talking about a girl | a woman is talking |
| 84 | Val/video6815.mp4 | 0.847 | Y | 1 | 0.644 | a dog is talking about a kitchen | a dog is wondering in the kitchen |
| 85 | Val/video6541.mp4 | 0.846 | - | 8 | 0.595 | a man is playing a soccer player | a man running in football gear |
| 86 | Val/video6752.mp4 | 0.845 | - | 6 | 0.595 | a woman is wearing her hair | a woman is wearing a mask |
| 87 | Val/video6602.mp4 | 0.845 | - | 8 | 0.623 | a man is adding a potatoatoes potatoes | a man is seasoning potatoes |
| 88 | Val/video6997.mp4 | 0.845 | - | 20 | 0.576 | a girl is making a bottle | a girl is making crafts |
| 89 | Val/video6994.mp4 | 0.844 | - | 3 | 0.569 | a man is talking about a food | a man is talking about something in kitchen |
| 90 | Val/video6589.mp4 | 0.844 | - | 7 | 0.613 | a person is using a computer program | someone is operating the computer |
| 91 | Val/video6747.mp4 | 0.844 | - | 24 | 0.596 | a person is playing a video game | someone is playing a game |
| 92 | Val/video6959.mp4 | 0.844 | - | 2 | 0.603 | a woman is talking about a product | three woman are talking about products |
| 93 | Val/video6615.mp4 | 0.843 | - | 3 | 0.663 | a person is making a kitchen | a person is preparing dough |
| 94 | Val/video6601.mp4 | 0.842 | - | 279 | 0.498 | a woman is talking on a phone | a woman is speaking |
| 95 | Val/video6534.mp4 | 0.841 | - | 3 | 0.705 | a horse is riding on horse | a horse is jumping |
| 96 | Val/video6888.mp4 | 0.840 | - | 6 | 0.529 | a man is talking about clothes | a man talks about fashion design |
| 97 | Val/video6963.mp4 | 0.839 | - | 4 | 0.667 | a soccer player is playing a goal | a football player makes a touch down |
| 98 | Val/video6532.mp4 | 0.838 | - | 2 | 0.750 | a person jumping on a trampoline | a man jumping and doing flips on a trampoline |
| 99 | Val/video6677.mp4 | 0.837 | - | 19 | 0.523 | a person is talking about a car | someone is driving a car |
| 100 | Val/video6624.mp4 | 0.837 | - | 2 | 0.662 | a person is showing a car | a person is talking about a car |
| 101 | Val/video6638.mp4 | 0.835 | - | 19 | 0.700 | a girl is talking about a child | a boy and a girl are talking |
| 102 | Val/video6623.mp4 | 0.831 | - | 6 | 0.595 | a woman is holding a bag bag | a woman with a bag |
| 103 | Val/video6685.mp4 | 0.830 | - | 15 | 0.642 | a person is cooking a potato | a man is cooking |
| 104 | Val/video6870.mp4 | 0.830 | - | 29 | 0.532 | a man and a woman are talking | a woman and a man with a computer talking |
| 105 | Val/video6814.mp4 | 0.828 | Y | 1 | 0.613 | a woman is talking about a garden | a woman is walking in a garden |
| 106 | Val/video6619.mp4 | 0.828 | - | 81 | 0.563 | a boy playing a baby | a child is dancing |
| 107 | Val/video6783.mp4 | 0.826 | - | 173 | 0.576 | a man is showing a car | a person is discussing a car |
| 108 | Val/video6644.mp4 | 0.824 | - | 6 | 0.633 | a man is singing on stage | a person is playing guitar on a stage |
| 109 | Val/video6674.mp4 | 0.823 | - | 14 | 0.618 | a man is cooking in the kitchen | a man is cooking something |
| 110 | Val/video6727.mp4 | 0.821 | - | 3 | 0.555 | a man is talking about another man | a man is talking to another man in a studio |
| 111 | Val/video6967.mp4 | 0.821 | Y | 1 | 0.673 | a race race running on a track | a track race taking place |
| 112 | Val/video6795.mp4 | 0.820 | - | 18 | 0.504 | a woman is talking about a smile | a woman is talking |
| 113 | Val/video6687.mp4 | 0.819 | - | 42 | 0.619 | a man is talking on a tv show | people fight on a tv show |
| 114 | Val/video6812.mp4 | 0.817 | - | 8 | 0.617 | a horse is riding on horse | a horse jumping over a pole |
| 115 | Val/video6957.mp4 | 0.816 | Y | 1 | 0.680 | a child is playing with a baby | a baby playing in a tub |
| 116 | Val/video6571.mp4 | 0.816 | - | 12 | 0.506 | a man is talking about glasses | a man is talking about an athlete |
| 117 | Val/video6779.mp4 | 0.815 | - | 10 | 0.570 | a person is stiring ingredients in the kitchen | someone is causing a mess in a kitchen |
| 118 | Val/video6770.mp4 | 0.810 | - | 3 | 0.563 | a man is sitting on a audience | a man speaks to an audience on stage |
| 119 | Val/video6743.mp4 | 0.804 | - | 5 | 0.627 | a soccer player is playing soccer | a soccer player kicks the ball across the field |
| 120 | Val/video6792.mp4 | 0.804 | Y | 1 | 0.646 | a football player is playing soccer | football players are cheering on a field |
| 121 | Val/video6919.mp4 | 0.804 | Y | 1 | 0.637 | a man is playing a guitar | a person is playing music |
| 122 | Val/video6720.mp4 | 0.800 | - | 20 | 0.592 | a man is mixing ingredients in a bowl | a man is mixing cream into flour |
| 123 | Val/video6853.mp4 | 0.797 | Y | 1 | 0.698 | a person is playing minecraft | footage of someone playing the game minecraft |
| 124 | Val/video6799.mp4 | 0.795 | Y | 1 | 0.622 | a man is singing on stage | a man is playing drums |
| 125 | Val/video6856.mp4 | 0.793 | - | 17 | 0.568 | a cartoon game is playing a video game | a gamer playing a video game |
| 126 | Val/video6563.mp4 | 0.793 | - | 7 | 0.648 | a band is performing on stage | a man is singing on the stage |
| 127 | Val/video6649.mp4 | 0.792 | - | 77 | 0.519 | a man is talking about a movie | a man is talking |
| 128 | Val/video6525.mp4 | 0.790 | - | 25 | 0.530 | a man is talking about a movie | a man talks about space |
| 129 | Val/video6805.mp4 | 0.785 | Y | 1 | 0.635 | a man is holding a weapon | man shoots a bow |
| 130 | Val/video6889.mp4 | 0.785 | - | 32 | 0.540 | a man is talking on a phone | the man talks to the camera |
| 131 | Val/video6591.mp4 | 0.777 | - | 30 | 0.568 | a woman is talking to a man | two woman talk to each other |
| 132 | Val/video6701.mp4 | 0.771 | - | 4 | 0.627 | a man is wrestling in a wrestling | men are wrestling |
| 133 | Val/video6560.mp4 | 0.769 | - | 65 | 0.571 | a man is holding a dog | a woman is feeding a dog |
| 134 | Val/video6561.mp4 | 0.768 | - | 8 | 0.667 | a person is stiring something in a pan | someone is cooking cumin in a skillet |
| 135 | Val/video6776.mp4 | 0.768 | - | 11 | 0.487 | a woman is talking to a movie | a man talking to a woman |
| 136 | Val/video7003.mp4 | 0.767 | - | 13 | 0.584 | a man is sitting in a chair | a man approaches a man on a bench |
| 137 | Val/video6951.mp4 | 0.767 | Y | 1 | 0.551 | a man is talking to a tv show | a man is talking to another man on a game show |
| 138 | Val/video6585.mp4 | 0.764 | - | 13 | 0.546 | a man and a woman is talking to a shirt in a bed | a man and a woman are looking at laptop and talking |
| 139 | Val/video6598.mp4 | 0.763 | Y | 1 | 0.684 | a woman is talking about a clothes | a woman explaining about the bag |
| 140 | Val/video6958.mp4 | 0.762 | - | 10 | 0.552 | a man is talking about a restaurant | a man talking about advertising |
| 141 | Val/video6559.mp4 | 0.761 | - | 21 | 0.513 | a man is playing a video game | people are watching a video |
| 142 | Val/video6653.mp4 | 0.761 | - | 2 | 0.689 | cars are riding a bike | motorbikes race down a street |
| 143 | Val/video6665.mp4 | 0.761 | - | 73 | 0.503 | a man is talking about his camera | a man is talking in a room |
| 144 | Val/video6988.mp4 | 0.756 | - | 6 | 0.611 | a cat is playing a dog | a man is talking to a cat |
| 145 | Val/video6798.mp4 | 0.754 | - | 91 | 0.432 | a woman is sitting in her camera and talking to her camera | a woman sings to the camera |
| 146 | Val/video6962.mp4 | 0.752 | - | 14 | 0.487 | a man and a woman is talking in a house | a man and a woman looking at laptop and talking |
| 147 | Val/video6660.mp4 | 0.749 | - | 12 | 0.544 | a man is talking to a movie | a man and a woman talking |
| 148 | Val/video6611.mp4 | 0.747 | - | 46 | 0.600 | two women are talking to a camera | two girls talk |
| 149 | Val/video6876.mp4 | 0.745 | - | 11 | 0.622 | a man and a boy is flying in a bird | a boy watches birds fly around |
| 150 | Val/video6657.mp4 | 0.742 | - | 2 | 0.560 | a man is talking about a screen | a man is looking at a web page |
| 151 | Val/video6640.mp4 | 0.739 | - | 2 | 0.624 | a cat is playing in the cat | a cat is laying on the floor |
| 152 | Val/video6734.mp4 | 0.737 | - | 22 | 0.687 | a man is wrestling in a wrestling | a person playing a wrestling video game |
| 153 | Val/video6671.mp4 | 0.725 | - | 25 | 0.573 | a man is singing in a house | a women singing in the room |
| 154 | Val/video6861.mp4 | 0.721 | - | 8 | 0.552 | a man is sitting in a room | a man in a suit talking |
| 155 | Val/video6974.mp4 | 0.719 | - | 2 | 0.674 | a man is talking about a restaurant | two men speak about alcohol |
| 156 | Val/video6630.mp4 | 0.719 | - | 101 | 0.437 | a man is talking to a camera | a man is singing in a room |
| 157 | Val/video6996.mp4 | 0.717 | - | 2 | 0.592 | a man is showing a machine | a man is cleaning out an engine part |
| 158 | Val/video6913.mp4 | 0.716 | - | 5 | 0.563 | a man is driving a car driving down a road | a animated man running in a street |
| 159 | Val/video6544.mp4 | 0.710 | - | 13 | 0.601 | a group of people are talking to a stage | people are preforming a skit on stage |
| 160 | Val/video6730.mp4 | 0.710 | - | 3 | 0.651 | a man is playing a game | people are running in a race |
| 161 | Val/video6700.mp4 | 0.709 | Y | 1 | 0.550 | a group of kids are talking | a child is smiling |
| 162 | Val/video6767.mp4 | 0.704 | - | 370 | 0.574 | a woman is talking about a video | two woman are talking |
| 163 | Val/video6864.mp4 | 0.700 | - | 16 | 0.539 | a woman is playing a movie | a woman takes a mans phone |
| 164 | Val/video6859.mp4 | 0.699 | - | 13 | 0.444 | a man is giving a woman | a person is holding money |
| 165 | Val/video6682.mp4 | 0.680 | - | 26 | 0.552 | a man is showing how to a map | a gamer is examining a map |
| 166 | Val/video7009.mp4 | 0.676 | - | 107 | 0.537 | a man is waiting in a room | a man is talking |
| 167 | Val/video6754.mp4 | 0.661 | - | 2 | 0.594 | a man is playing a hammer on the ground is being taking a ground on the floor | here a man lays on the ground after doing good at golf |
| 168 | Val/video6999.mp4 | 0.659 | - | 9 | 0.466 | a woman is showing how to a water | a woman showing how to play with a child s toy |
| 169 | Val/video6802.mp4 | 0.654 | - | 2 | 0.560 | a group of children are dancing | there are a group of children on stage in white shirts |
| 170 | Val/video6522.mp4 | 0.650 | - | 2 | 0.569 | a video game is playing on a screen | gameplay footage of someone playing a game |
| 171 | Val/video6755.mp4 | 0.649 | - | 5 | 0.532 | there is a show of a audience being shown | a music concert where a lot of viewers are viewing it |
| 172 | Val/video6600.mp4 | 0.649 | Y | 1 | 0.545 | a woman is talking about a movie | a man and woman are singing a song |
| 173 | Val/video6860.mp4 | 0.642 | - | 99 | 0.456 | a group of people are talking | a woman is singing while people around her talk |
| 174 | Val/video6547.mp4 | 0.638 | - | 3 | 0.568 | a pan is talking about a potatoes | a man is talking while tossing food in a pan |
| 175 | Val/video6793.mp4 | 0.580 | - | 56 | 0.455 | a group of people are talking | a girl is singing |