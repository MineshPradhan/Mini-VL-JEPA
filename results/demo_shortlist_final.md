# Per-video system performance - validate split, tiers: easy,medium,hard
Decoder: `checkpoints/dec_en_all/decoder.pt` | predictions: `curriculum/preds.npz`

- videos: 497 | retrieval R@1 (pool of 497): **19.7%** | mean cap_sim: 0.783

## Safe demo shortlist - R@1 correct AND cap_sim >= 0.8 (50 videos)

1. **Val/video6937.mp4** - "a woman is talking on a phone" (sim 0.993)
2. **Val/video7004.mp4** - "a boy is jumping on a trampoline" (sim 0.992)
3. **Val/video6971.mp4** - "a person is playing a video game" (sim 0.951)
4. **Val/video6531.mp4** - "two news reporters are talking on a tv show" (sim 0.948)
5. **Val/video6848.mp4** - "a man is talking about a car" (sim 0.942)
6. **Val/video6845.mp4** - "a person is folding a paper airplane" (sim 0.940)
7. **Val/video6820.mp4** - "a woman is talking about how to prepare a recipe" (sim 0.936)
8. **Val/video6745.mp4** - "a man is cooking a dish in a kitchen" (sim 0.931)
9. **Val/video6739.mp4** - "a group of girls are singing on stage" (sim 0.928)
10. **Val/video6897.mp4** - "a girl is singing on the voice" (sim 0.926)
11. **Val/video6797.mp4** - "a man is talking to a group of people" (sim 0.925)
12. **Val/video6910.mp4** - "a fireworks are shown" (sim 0.922)
13. **Val/video6906.mp4** - "a woman is running on a track" (sim 0.921)
14. **Val/video6636.mp4** - "a person is playing with a cube" (sim 0.918)
15. **Val/video6527.mp4** - "a woman is explaining how to make her hair" (sim 0.917)
16. **Val/video6558.mp4** - "a person is playing with a cube" (sim 0.905)
17. **Val/video6973.mp4** - "two players are playing a table tennis game" (sim 0.903)
18. **Val/video6925.mp4** - "a woman is making a stroller" (sim 0.902)
19. **Val/video6922.mp4** - "a man is driving a car and driving down the road" (sim 0.899)
20. **Val/video6675.mp4** - "a hockey player is playing hockey" (sim 0.896)
21. **Val/video6933.mp4** - "a boy is playing in a school" (sim 0.896)
22. **Val/video6774.mp4** - "a man is eating food" (sim 0.889)
23. **Val/video6679.mp4** - "a man is explaining how to use a computer" (sim 0.885)
24. **Val/video6780.mp4** - "a person is folding a paper" (sim 0.884)
25. **Val/video6868.mp4** - "a group of people are dancing" (sim 0.879)
26. **Val/video6598.mp4** - "a woman is talking about a bag" (sim 0.877)
27. **Val/video6764.mp4** - "a car is driving on a road" (sim 0.877)
28. **Val/video6629.mp4** - "a man is playing a football game" (sim 0.867)
29. **Val/video6917.mp4** - "a group of people are dancing and dancing" (sim 0.858)
30. **Val/video7002.mp4** - "a man is riding a bike on a motorcycle" (sim 0.857)
31. **Val/video6659.mp4** - "a cartoon character is talking to a girl" (sim 0.856)
32. **Val/video6648.mp4** - "a woman is walking in a runway" (sim 0.856)
33. **Val/video6967.mp4** - "a man is running on a track" (sim 0.851)
34. **Val/video6998.mp4** - "a boy in a kitchen is eating food" (sim 0.849)
35. **Val/video6893.mp4** - "a man is talking about how to cook a dish" (sim 0.847)
36. **Val/video6595.mp4** - "a person is talking about a car" (sim 0.838)
37. **Val/video6740.mp4** - "a man is talking about a bottle" (sim 0.829)
38. **Val/video6814.mp4** - "a woman is talking about a garden" (sim 0.828)
39. **Val/video6956.mp4** - "a man is surfing on the beach" (sim 0.828)
40. **Val/video6884.mp4** - "a fish is swimming in the water" (sim 0.824)
41. **Val/video6924.mp4** - "a man is talking to another man" (sim 0.822)
42. **Val/video6866.mp4** - "a horse is riding on a field" (sim 0.820)
43. **Val/video6829.mp4** - "a woman is swimming in a gym" (sim 0.818)
44. **Val/video6957.mp4** - "a child is playing with a baby" (sim 0.816)
45. **Val/video6805.mp4** - "a man is holding a pistol" (sim 0.816)
46. **Val/video6594.mp4** - "a group of people are skiing in a snow" (sim 0.810)
47. **Val/video6815.mp4** - "a man is talking about a dog" (sim 0.808)
48. **Val/video6586.mp4** - "a woman is showing a stroller" (sim 0.807)
49. **Val/video6792.mp4** - "a football player is playing football" (sim 0.805)
50. **Val/video6919.mp4** - "a man is playing a guitar" (sim 0.804)

## Full ranking (best first)

| # | video | cap_sim | R@1 | rank | pred_cos | generated | closest reference |
|---|---|---|---|---|---|---|---|
| 1 | Val/video6920.mp4 | 1.000 | - | 103 | 0.616 | a man is giving a speech | a man is giving a speech |
| 2 | Val/video6939.mp4 | 1.000 | - | 4 | 0.648 | a man is playing minecraft | a man is playing minecraft |
| 3 | Val/video6911.mp4 | 1.000 | - | 14 | 0.635 | a woman is cooking | a woman is cooking |
| 4 | Val/video6912.mp4 | 1.000 | - | 13 | 0.631 | a man is talking about a car | a man is talking about a car |
| 5 | Val/video6782.mp4 | 1.000 | - | 15 | 0.597 | a woman is singing on stage | a woman is singing on stage |
| 6 | Val/video6934.mp4 | 1.000 | - | 3 | 0.670 | a news reporter is talking about the news | a news reporter is talking about the news |
| 7 | Val/video6596.mp4 | 1.000 | - | 6 | 0.632 | a person is playing a video game | a person is playing a video game |
| 8 | Val/video6662.mp4 | 1.000 | - | 11 | 0.667 | a person is playing a video game | a person is playing a video game |
| 9 | Val/video6790.mp4 | 1.000 | - | 7 | 0.616 | a person is playing a video game | a person is playing a video game |
| 10 | Val/video6856.mp4 | 1.000 | - | 17 | 0.568 | a person is playing a video game | a person is playing a video game |
| 11 | Val/video6624.mp4 | 1.000 | - | 2 | 0.662 | a car is shown | a car is shown |
| 12 | Val/video6857.mp4 | 1.000 | - | 2 | 0.676 | a person is playing with a toy | a person is playing with a toy |
| 13 | Val/video6550.mp4 | 1.000 | - | 10 | 0.652 | a man is driving a car | a man is driving a car |
| 14 | Val/video6601.mp4 | 0.997 | - | 279 | 0.498 | a woman is talking | a woman is speaking |
| 15 | Val/video7006.mp4 | 0.994 | - | 14 | 0.613 | a man is dancing on a stage | a man is dancing on stage |
| 16 | Val/video6937.mp4 | 0.993 | Y | 1 | 0.594 | a woman is talking on a phone | a woman talks on the phone |
| 17 | Val/video6766.mp4 | 0.993 | - | 2 | 0.660 | kids are dancing | the children are dancing |
| 18 | Val/video7004.mp4 | 0.992 | Y | 1 | 0.753 | a boy is jumping on a trampoline | a boy jumps on a trampoline |
| 19 | Val/video6684.mp4 | 0.985 | - | 43 | 0.653 | a band is singing a song | a band is playing a song |
| 20 | Val/video6976.mp4 | 0.984 | - | 13 | 0.680 | a group of girls are singing on stage | a girl group is singing on stage |
| 21 | Val/video6811.mp4 | 0.983 | - | 2 | 0.703 | a cat is playing with a baby | a baby is playing with a cat |
| 22 | Val/video6658.mp4 | 0.980 | - | 2 | 0.688 | a basketball player is playing basketball | basketball player are playing basketball |
| 23 | Val/video6787.mp4 | 0.980 | - | 3 | 0.673 | a man is cooking food in a kitchen | a man is preparing a meal in a kitchen |
| 24 | Val/video6858.mp4 | 0.974 | - | 20 | 0.663 | a man is playing minecraft | a person is playing minecraft |
| 25 | Val/video6574.mp4 | 0.974 | - | 4 | 0.610 | a man is singing on stage | a man singing on a stage |
| 26 | Val/video6838.mp4 | 0.972 | - | 8 | 0.656 | a woman is applying makeup on her face | a woman is applying make up to her face |
| 27 | Val/video6989.mp4 | 0.972 | - | 3 | 0.636 | a man is talking in a car | a man is talking inside a car |
| 28 | Val/video6822.mp4 | 0.971 | - | 2 | 0.673 | a car is driving down the road | a car is racing down a road |
| 29 | Val/video6757.mp4 | 0.968 | - | 4 | 0.688 | a woman is preparing food | a woman is making food |
| 30 | Val/video6729.mp4 | 0.967 | - | 10 | 0.635 | a person is playing a video game | a person playing a video game |
| 31 | Val/video7001.mp4 | 0.966 | - | 5 | 0.733 | a man is playing a video game | a person is playing a video game |
| 32 | Val/video6909.mp4 | 0.962 | - | 5 | 0.707 | a woman is applying makeup | a woman puts on makeup |
| 33 | Val/video6760.mp4 | 0.957 | - | 7 | 0.726 | a person is playing minecraft | a person is playing the game minecraft |
| 34 | Val/video6556.mp4 | 0.956 | - | 20 | 0.712 | a group of soldiers are talking | a group of soldiers are having a conversation |
| 35 | Val/video6783.mp4 | 0.954 | - | 173 | 0.576 | a person is talking about a car | a person is discussing a car |
| 36 | Val/video6806.mp4 | 0.951 | - | 4 | 0.699 | a person is playing a video game | someone is playing a video game |
| 37 | Val/video6971.mp4 | 0.951 | Y | 1 | 0.669 | a person is playing a video game | someone is playing a video game |
| 38 | Val/video6573.mp4 | 0.951 | - | 160 | 0.679 | someone is playing a video game | a person is playing a video game |
| 39 | Val/video6669.mp4 | 0.950 | - | 6 | 0.617 | a woman is singing a song | a woman is singing |
| 40 | Val/video6986.mp4 | 0.950 | - | 12 | 0.623 | a woman is applying makeup on her face | a women is putting makeup in her face |
| 41 | Val/video6531.mp4 | 0.948 | Y | 1 | 0.624 | two news reporters are talking on a tv show | two news reporters are talking to each other on tv |
| 42 | Val/video6880.mp4 | 0.944 | - | 3 | 0.684 | a man is giving a speech | a man is giving a lecture |
| 43 | Val/video6878.mp4 | 0.944 | - | 398 | 0.655 | a person is showing how to use a computer | a person is showing how to use a computer program |
| 44 | Val/video6848.mp4 | 0.942 | Y | 1 | 0.635 | a man is talking about a car | a man is talking about his car |
| 45 | Val/video6708.mp4 | 0.941 | - | 30 | 0.650 | a group of people are dancing | a group of dancing people |
| 46 | Val/video6845.mp4 | 0.940 | Y | 1 | 0.708 | a person is folding a paper airplane | a man is flying a paper airplane |
| 47 | Val/video6582.mp4 | 0.940 | - | 8 | 0.629 | a man is making a dish in a kitchen | a man cooking food in a kitchen |
| 48 | Val/video6532.mp4 | 0.937 | - | 2 | 0.750 | a boy jumping on a trampoline | a child jumps on the trampoline |
| 49 | Val/video6820.mp4 | 0.936 | Y | 1 | 0.663 | a woman is talking about how to prepare a recipe | a woman talks about how to prepare a meal |
| 50 | Val/video6892.mp4 | 0.936 | - | 3 | 0.592 | a man and a woman are kissing each other | man and woman are kissing |
| 51 | Val/video6695.mp4 | 0.933 | - | 10 | 0.650 | a man is sitting in a car | a man is sitting in a car talking |
| 52 | Val/video6518.mp4 | 0.932 | - | 2 | 0.683 | two men are wrestling in a wrestling match | two men in a wrestling competition |
| 53 | Val/video6667.mp4 | 0.932 | - | 2 | 0.700 | two men are wrestling in a wrestling match | two men in a wrestling competition |
| 54 | Val/video6745.mp4 | 0.931 | Y | 1 | 0.694 | a man is cooking a dish in a kitchen | man cooking food in a kitchen |
| 55 | Val/video6750.mp4 | 0.931 | - | 18 | 0.594 | a woman is talking about something | a girl is talking about something |
| 56 | Val/video6964.mp4 | 0.930 | - | 2 | 0.606 | a person is adding some ingredients in a bowl | a person adds ingredient to a pot |
| 57 | Val/video6936.mp4 | 0.928 | - | 3 | 0.701 | a woman is doing gymnastics | a girl is doing gymnastic |
| 58 | Val/video6739.mp4 | 0.928 | Y | 1 | 0.653 | a group of girls are singing on stage | a group of women singing on stage |
| 59 | Val/video6577.mp4 | 0.926 | - | 27 | 0.546 | a woman is talking about her camera | a woman talks to a camera |
| 60 | Val/video6807.mp4 | 0.926 | - | 3 | 0.693 | a person is applying makeup | a man is putting on makeup |
| 61 | Val/video6599.mp4 | 0.926 | - | 39 | 0.610 | a woman is talking on a television show | woman are talking on a talk show |
| 62 | Val/video6897.mp4 | 0.926 | Y | 1 | 0.614 | a girl is singing on the voice | a woman sings on the voice |
| 63 | Val/video6797.mp4 | 0.925 | Y | 1 | 0.608 | a man is talking to a group of people | a man is speaking to a group of others |
| 64 | Val/video6910.mp4 | 0.922 | Y | 1 | 0.665 | a fireworks are shown | a firework is shown |
| 65 | Val/video6906.mp4 | 0.921 | Y | 1 | 0.680 | a woman is running on a track | a woman is running in track and field |
| 66 | Val/video6563.mp4 | 0.921 | - | 7 | 0.648 | a man is performing on stage | a man is singing on the stage |
| 67 | Val/video6972.mp4 | 0.919 | - | 63 | 0.482 | a man is talking about a woman | a man is talking with a girl |
| 68 | Val/video6615.mp4 | 0.918 | - | 3 | 0.663 | a person is making a bread | a person is preparing dough |
| 69 | Val/video6636.mp4 | 0.918 | Y | 1 | 0.731 | a person is playing with a cube | someone is playing with cubes |
| 70 | Val/video6545.mp4 | 0.918 | - | 7 | 0.691 | a man is making a cooking dish in a kitchen | a man in a kitchen prepares a soup |
| 71 | Val/video6520.mp4 | 0.918 | - | 2 | 0.611 | a woman is running on a track | there is a woman is running on the track |
| 72 | Val/video6527.mp4 | 0.917 | Y | 1 | 0.649 | a woman is explaining how to make her hair | a women shows how to fix her hair |
| 73 | Val/video6803.mp4 | 0.911 | - | 12 | 0.651 | a woman is applying makeup to her face | a woman is showing makeup for the face |
| 74 | Val/video6836.mp4 | 0.909 | - | 3 | 0.734 | a child is playing with a baby | a baby is playing with a cat |
| 75 | Val/video6771.mp4 | 0.909 | - | 5 | 0.546 | a man is holding a ball | a man is holding a sword |
| 76 | Val/video6731.mp4 | 0.908 | - | 3 | 0.683 | a man is playing a video game | someone is playing a video game |
| 77 | Val/video6583.mp4 | 0.908 | - | 2 | 0.667 | two players are playing a tennis match | two teams of tennis players are playing a tennis |
| 78 | Val/video6808.mp4 | 0.907 | - | 14 | 0.694 | a man is mixing ingredients in a kitchen | a man mixing ingredients into a bowl |
| 79 | Val/video6557.mp4 | 0.905 | - | 21 | 0.670 | a man is cooking a dish | a man preparing a food dish |
| 80 | Val/video6558.mp4 | 0.905 | Y | 1 | 0.707 | a person is playing with a cube | a person is playing with a rubik cube |
| 81 | Val/video6973.mp4 | 0.903 | Y | 1 | 0.768 | two players are playing a table tennis game | two players are playing table tennis match in the gym |
| 82 | Val/video6925.mp4 | 0.902 | Y | 1 | 0.575 | a woman is making a stroller | a woman is fixing a stroller |
| 83 | Val/video6922.mp4 | 0.899 | Y | 1 | 0.633 | a man is driving a car and driving down the road | a man is driving a car on the track |
| 84 | Val/video6632.mp4 | 0.899 | - | 3 | 0.666 | a soccer player is playing soccer | soccer players playing soccer |
| 85 | Val/video6603.mp4 | 0.898 | - | 88 | 0.517 | a man is talking to an interview | a man is being interviewed |
| 86 | Val/video6746.mp4 | 0.897 | - | 5 | 0.642 | a woman is talking about a baby stroller | a woman talking about a roller stroller |
| 87 | Val/video6675.mp4 | 0.896 | Y | 1 | 0.661 | a hockey player is playing hockey | a hockey player juggling a puck |
| 88 | Val/video6933.mp4 | 0.896 | Y | 1 | 0.677 | a boy is playing in a school | children in a school are playing a game |
| 89 | Val/video6566.mp4 | 0.894 | - | 5 | 0.529 | a person is playing with a puzzle | a person is playing with toys |
| 90 | Val/video6841.mp4 | 0.894 | - | 27 | 0.531 | a person is playing a video game | a person is talking while playing a video game |
| 91 | Val/video6548.mp4 | 0.891 | - | 4 | 0.644 | a young girl is singing on stage | a young girl on stage |
| 92 | Val/video6881.mp4 | 0.890 | - | 53 | 0.593 | a cartoon character is talking to each other | cartoon people are talking to each other |
| 93 | Val/video6546.mp4 | 0.889 | - | 3 | 0.651 | a person is talking about a car | a person is talking inside the car |
| 94 | Val/video6774.mp4 | 0.889 | Y | 1 | 0.640 | a man is eating food | people are eating food |
| 95 | Val/video6903.mp4 | 0.888 | - | 19 | 0.585 | a girl is talking to a woman | a woman is talking to kids |
| 96 | Val/video6818.mp4 | 0.887 | - | 2 | 0.694 | a soccer player is playing a ball | a soccer player scores a goal |
| 97 | Val/video6529.mp4 | 0.887 | - | 16 | 0.577 | a woman is sitting in a table | there is a woman sitting in front of a table |
| 98 | Val/video6679.mp4 | 0.885 | Y | 1 | 0.625 | a man is explaining how to use a computer | a man is explaining how to do something on a computer |
| 99 | Val/video6875.mp4 | 0.885 | - | 17 | 0.683 | a woman is giving a stroller | a woman displays a stroller |
| 100 | Val/video6697.mp4 | 0.885 | - | 3 | 0.648 | a man is playing a tennis game | a two men are playing a tennis game |
| 101 | Val/video6744.mp4 | 0.884 | - | 28 | 0.622 | two men are talking on stage | two men speak on stage in front of an audience |
| 102 | Val/video6780.mp4 | 0.884 | Y | 1 | 0.694 | a person is folding a paper | a person is folding a paper airplane |
| 103 | Val/video6625.mp4 | 0.882 | - | 8 | 0.713 | a person is playing a cube | someone is playing with cubes |
| 104 | Val/video6992.mp4 | 0.880 | - | 6 | 0.666 | a group of people are singing | the crowd is singing a song |
| 105 | Val/video6868.mp4 | 0.879 | Y | 1 | 0.555 | a group of people are dancing | a group of people are cheering |
| 106 | Val/video6602.mp4 | 0.878 | - | 8 | 0.623 | a man is cooking vegetables in a bowl | a person is preparing a potato dish in the kitchen |
| 107 | Val/video6598.mp4 | 0.877 | Y | 1 | 0.684 | a woman is talking about a bag | a woman explaining about the bag |
| 108 | Val/video6764.mp4 | 0.877 | Y | 1 | 0.685 | a car is driving on a road | a car is driving on rocks |
| 109 | Val/video6952.mp4 | 0.875 | - | 214 | 0.589 | a person is playing a video game | someone is playing a first person video game |
| 110 | Val/video6628.mp4 | 0.874 | - | 2 | 0.751 | a woman is giving a stroller | a woman is setting up a stroller |
| 111 | Val/video6634.mp4 | 0.874 | - | 16 | 0.609 | a man in a black shirt is talking about a movie | man in black suit is talking about something on camera |
| 112 | Val/video6524.mp4 | 0.873 | - | 2 | 0.590 | a man is cooking a dish in a pan | a person is cooking an egg in a wok pan |
| 113 | Val/video6623.mp4 | 0.871 | - | 6 | 0.595 | a woman is holding a bag | a woman with a bag |
| 114 | Val/video6656.mp4 | 0.869 | - | 5 | 0.646 | a man is playing a football game | people are playing football |
| 115 | Val/video6712.mp4 | 0.869 | - | 11 | 0.622 | a man is playing a football game | people are playing football |
| 116 | Val/video6844.mp4 | 0.868 | - | 12 | 0.596 | a soccer player is playing a ball | football players playing football |
| 117 | Val/video6592.mp4 | 0.867 | - | 422 | 0.482 | a man and a woman are talking in a car | a man and woman are talking |
| 118 | Val/video6748.mp4 | 0.867 | - | 5 | 0.629 | a woman is talking about a shop | a woman is walking and talking in a store |
| 119 | Val/video6629.mp4 | 0.867 | Y | 1 | 0.612 | a man is playing a football game | a man is training for football |
| 120 | Val/video6779.mp4 | 0.866 | - | 10 | 0.570 | a person is cooking vegetables | a person is cooking |
| 121 | Val/video6891.mp4 | 0.865 | - | 13 | 0.589 | a man is making a recipe in a bowl | a man is preparing a recipe |
| 122 | Val/video6579.mp4 | 0.863 | - | 80 | 0.573 | a woman is talking | some women are talking |
| 123 | Val/video6852.mp4 | 0.863 | - | 92 | 0.547 | a woman is talking about another woman | a woman explaining for another woman |
| 124 | Val/video6616.mp4 | 0.863 | - | 398 | 0.554 | a man is working on a computer | a man is putting together a computer |
| 125 | Val/video6901.mp4 | 0.862 | - | 7 | 0.604 | a man is talking about something | a man is talking about a drink |
| 126 | Val/video6515.mp4 | 0.862 | - | 3 | 0.693 | a man is cutting vegetables in a kitchen | a man cuts something vegetable on the table |
| 127 | Val/video6938.mp4 | 0.862 | - | 8 | 0.638 | a man is talking about a car | a man is looking at a car |
| 128 | Val/video6539.mp4 | 0.861 | - | 5 | 0.575 | a cartoon character is talking | a cartoon character is talking and crying |
| 129 | Val/video6576.mp4 | 0.861 | - | 6 | 0.578 | a man in a black shirt is talking about a situation | a guy in a black shirt is talking to something |
| 130 | Val/video6765.mp4 | 0.860 | - | 9 | 0.666 | a man is playing a football game | a person is playing a sport |
| 131 | Val/video6917.mp4 | 0.858 | Y | 1 | 0.610 | a group of people are dancing and dancing | a whole bunch of people dancing and having fun |
| 132 | Val/video7007.mp4 | 0.858 | - | 306 | 0.571 | a woman is talking to a girl | a woman is talking |
| 133 | Val/video7002.mp4 | 0.857 | Y | 1 | 0.699 | a man is riding a bike on a motorcycle | a guy on a motorcycle is crashing in the road |
| 134 | Val/video6659.mp4 | 0.856 | Y | 1 | 0.665 | a cartoon character is talking to a girl | a cartoon character is talking |
| 135 | Val/video6648.mp4 | 0.856 | Y | 1 | 0.626 | a woman is walking in a runway | a lady is walking on the stage |
| 136 | Val/video6561.mp4 | 0.854 | - | 8 | 0.667 | a person is cooking a dish in a pan | a man is toasting spices in a cooking pan |
| 137 | Val/video6898.mp4 | 0.852 | - | 7 | 0.576 | a man is talking about the moon | a person is talking about planets |
| 138 | Val/video6967.mp4 | 0.851 | Y | 1 | 0.673 | a man is running on a track | a man is running |
| 139 | Val/video6887.mp4 | 0.850 | - | 5 | 0.620 | a group of people are walking in a room | a group of people are walking |
| 140 | Val/video6541.mp4 | 0.849 | - | 8 | 0.595 | a man is playing a football game | a person is playing a game |
| 141 | Val/video6998.mp4 | 0.849 | Y | 1 | 0.638 | a boy in a kitchen is eating food | two children are in a kitchen eating food |
| 142 | Val/video6767.mp4 | 0.848 | - | 370 | 0.574 | a woman is talking | two woman are talking |
| 143 | Val/video6893.mp4 | 0.847 | Y | 1 | 0.602 | a man is talking about how to cook a dish | a man narrates while cooking food |
| 144 | Val/video6709.mp4 | 0.846 | - | 229 | 0.658 | a girl is eating food | a girl is feeding a boy |
| 145 | Val/video6940.mp4 | 0.846 | - | 3 | 0.501 | a woman is dancing on the beach | there is a woman is walking on the beach |
| 146 | Val/video6727.mp4 | 0.846 | - | 3 | 0.555 | a man is talking to another man | a man is talking to another man in a studio |
| 147 | Val/video6674.mp4 | 0.845 | - | 14 | 0.618 | a man in a kitchen is cooking a dish | a person prepares some food in a bowl |
| 148 | Val/video6672.mp4 | 0.844 | - | 51 | 0.607 | a man is talking about a food | a man is talking about something in kitchen |
| 149 | Val/video6589.mp4 | 0.844 | - | 7 | 0.613 | a person is using a computer program | someone is operating the computer |
| 150 | Val/video6663.mp4 | 0.844 | - | 41 | 0.631 | a person is playing a video game | someone is playing a game |
| 151 | Val/video6747.mp4 | 0.844 | - | 24 | 0.596 | a person is playing a video game | someone is playing a game |
| 152 | Val/video6959.mp4 | 0.844 | - | 2 | 0.603 | a woman is talking about a product | three woman are talking about products |
| 153 | Val/video6553.mp4 | 0.842 | - | 8 | 0.631 | a man is driving a car while driving down the road | a man is riding down the highway in a red car |
| 154 | Val/video6735.mp4 | 0.842 | - | 24 | 0.561 | a man is talking on a tv show | someone is watching show on tv |
| 155 | Val/video6676.mp4 | 0.842 | - | 155 | 0.591 | a man is talking about a movie | a man is talking about something |
| 156 | Val/video6536.mp4 | 0.842 | - | 4 | 0.665 | a group of players are playing soccer | a group of soccer players are posing with kids |
| 157 | Val/video6963.mp4 | 0.842 | - | 4 | 0.667 | a soccer player is playing football | football players compete in a game of football |
| 158 | Val/video6534.mp4 | 0.841 | - | 3 | 0.705 | a horse is riding on horse | a horse is jumping |
| 159 | Val/video6980.mp4 | 0.841 | - | 102 | 0.589 | a man is playing a video game | a man is playing a video game and telling about it |
| 160 | Val/video6549.mp4 | 0.841 | - | 18 | 0.647 | a woman is mixing ingredients in a bowl | a woman is pouring something into a pot |
| 161 | Val/video6994.mp4 | 0.840 | - | 3 | 0.569 | a man is talking about food | a man is talking about something in kitchen |
| 162 | Val/video6759.mp4 | 0.840 | - | 10 | 0.672 | a woman is cooking a recipe in a kitchen | there is a woman is making some noodles in the kitchen |
| 163 | Val/video6606.mp4 | 0.840 | - | 6 | 0.619 | a woman is talking about a food | a woman talking about baking |
| 164 | Val/video6685.mp4 | 0.839 | - | 15 | 0.642 | someone is cooking a potato | someone is cooking food |
| 165 | Val/video6611.mp4 | 0.838 | - | 46 | 0.600 | a woman is talking | the woman speak in unison |
| 166 | Val/video6945.mp4 | 0.838 | - | 6 | 0.539 | a man is talking to another man | a man is talking about something |
| 167 | Val/video6595.mp4 | 0.838 | Y | 1 | 0.601 | a person is talking about a car | man talking about a cars dashboard |
| 168 | Val/video6991.mp4 | 0.837 | - | 24 | 0.642 | a man is showing how to repair a car | a man is fixing a car |
| 169 | Val/video6677.mp4 | 0.837 | - | 19 | 0.523 | a person is talking about a car | someone is driving a car |
| 170 | Val/video6777.mp4 | 0.836 | - | 56 | 0.575 | a woman is talking to another woman | a woman is talking about something |
| 171 | Val/video6907.mp4 | 0.836 | - | 3 | 0.520 | a man is talking to a cartoon show | a man is discussing a tv show |
| 172 | Val/video6974.mp4 | 0.836 | - | 2 | 0.674 | two men are talking about a restaurant | two men speak about alcohol |
| 173 | Val/video6761.mp4 | 0.836 | - | 2 | 0.596 | a group of people are walking in a park | a group of people go hiking |
| 174 | Val/video6821.mp4 | 0.835 | - | 52 | 0.554 | a man is walking in a forest | a man is walking on the road |
| 175 | Val/video6876.mp4 | 0.834 | - | 11 | 0.622 | a boy is playing with a bird | a boy is watching birds |
| 176 | Val/video6795.mp4 | 0.834 | - | 18 | 0.504 | a woman is talking about a hair | a woman is talking |
| 177 | Val/video6696.mp4 | 0.832 | - | 2 | 0.629 | a woman is driving a car and talking about her car | a girl talks while riding in a car |
| 178 | Val/video6870.mp4 | 0.830 | - | 29 | 0.532 | a man and a woman are talking | a woman and a man with a computer talking |
| 179 | Val/video6941.mp4 | 0.830 | - | 6 | 0.635 | a person is explaining how to cook a pot in a kitchen | in a kitchen someone is explaining to how to cook a vegetable curry |
| 180 | Val/video6740.mp4 | 0.829 | Y | 1 | 0.538 | a man is talking about a bottle | a man is talking about the content inside a bottle |
| 181 | Val/video6814.mp4 | 0.828 | Y | 1 | 0.613 | a woman is talking about a garden | a woman is walking in a garden |
| 182 | Val/video6956.mp4 | 0.828 | Y | 1 | 0.655 | a man is surfing on the beach | a man is in the water surfing on a surf board |
| 183 | Val/video6719.mp4 | 0.828 | - | 37 | 0.584 | a man and a woman are kissing | a man is talking and kissing a lady |
| 184 | Val/video6726.mp4 | 0.826 | - | 7 | 0.611 | a car is driving down the road | cars take off in a lane on a street |
| 185 | Val/video6794.mp4 | 0.825 | - | 7 | 0.646 | a man is jumping on a trampoline | a man is jumping on top |
| 186 | Val/video6979.mp4 | 0.824 | - | 2 | 0.663 | a woman is playing a tennis match | there is a woman in red is playing tennis |
| 187 | Val/video6884.mp4 | 0.824 | Y | 1 | 0.638 | a fish is swimming in the water | some fish in a tank are swimming around in water |
| 188 | Val/video6644.mp4 | 0.824 | - | 6 | 0.633 | a man is singing on stage | a person is playing guitar on a stage |
| 189 | Val/video6768.mp4 | 0.823 | - | 55 | 0.543 | a man is talking to a movie | two men talking to each other in a movie |
| 190 | Val/video6706.mp4 | 0.823 | - | 21 | 0.599 | a person is making a dish in a bowl | a person making a fish bowl |
| 191 | Val/video6899.mp4 | 0.823 | - | 2 | 0.561 | a man and a woman are talking | a woman and a man are walking towards each other |
| 192 | Val/video6924.mp4 | 0.822 | Y | 1 | 0.622 | a man is talking to another man | a man is talking |
| 193 | Val/video6743.mp4 | 0.821 | - | 5 | 0.627 | a soccer player playing a football game | a soccer player kicks the ball across the field |
| 194 | Val/video6752.mp4 | 0.820 | - | 6 | 0.595 | a woman is wearing a lotion on her face | a girl is wearing a device on her face |
| 195 | Val/video6866.mp4 | 0.820 | Y | 1 | 0.594 | a horse is riding on a field | a man rides a horse on a track |
| 196 | Val/video6687.mp4 | 0.819 | - | 42 | 0.619 | a man is talking on a tv show | people fight on a tv show |
| 197 | Val/video6829.mp4 | 0.818 | Y | 1 | 0.661 | a woman is swimming in a gym | a woman is swimming |
| 198 | Val/video6864.mp4 | 0.817 | - | 16 | 0.539 | a man is talking to a woman | a man takes pictures while talking to a woman |
| 199 | Val/video6819.mp4 | 0.817 | - | 4 | 0.519 | a person is playing a video game | a man is playing a video game while chatting live |
| 200 | Val/video6993.mp4 | 0.817 | - | 17 | 0.683 | a minecraft character is playing minecraft | a minecraft character is crushing another character |
| 201 | Val/video6776.mp4 | 0.817 | - | 11 | 0.487 | a woman is talking | a man talking to a woman |
| 202 | Val/video6957.mp4 | 0.816 | Y | 1 | 0.680 | a child is playing with a baby | a baby playing in a tub |
| 203 | Val/video6805.mp4 | 0.816 | Y | 1 | 0.635 | a man is holding a pistol | a man shooting a bow |
| 204 | Val/video6544.mp4 | 0.814 | - | 13 | 0.601 | a man is talking to a group of people | a man is talking to a man about the audience |
| 205 | Val/video6637.mp4 | 0.813 | - | 4 | 0.652 | a woman is sitting in a classroom | a lady talking in the class room |
| 206 | Val/video6691.mp4 | 0.812 | - | 2 | 0.639 | a woman is mixing ingredients in a kitchen | a chef prepares some ingredients in a kitchen |
| 207 | Val/video6741.mp4 | 0.812 | - | 301 | 0.575 | a woman is talking about a audience | a woman is giving a speech |
| 208 | Val/video6651.mp4 | 0.811 | - | 4 | 0.656 | a minecraft character is playing a game | a minecraft character is moving around |
| 209 | Val/video6612.mp4 | 0.811 | - | 40 | 0.617 | a man is talking about the car and a car | a guy is talking about a car on top of a hill |
| 210 | Val/video6594.mp4 | 0.810 | Y | 1 | 0.678 | a group of people are skiing in a snow | some people are skiing in a snow covered forest |
| 211 | Val/video6647.mp4 | 0.810 | - | 42 | 0.620 | a man is showing how to use a computer program | a person is using a computer program |
| 212 | Val/video6815.mp4 | 0.808 | Y | 1 | 0.644 | a man is talking about a dog | a man talks to his dog while cooking |
| 213 | Val/video6630.mp4 | 0.807 | - | 101 | 0.437 | a man is talking to a woman in a room | a man is singing in a room |
| 214 | Val/video6586.mp4 | 0.807 | Y | 1 | 0.687 | a woman is showing a stroller | a woman demonstrates the use of a baby stroller |
| 215 | Val/video6652.mp4 | 0.805 | - | 11 | 0.556 | a group of people are talking about a crowd | a group of people talk outside |
| 216 | Val/video6792.mp4 | 0.805 | Y | 1 | 0.646 | a football player is playing football | football players are cheering on a field |
| 217 | Val/video6919.mp4 | 0.804 | Y | 1 | 0.637 | a man is playing a guitar | a person is playing music |
| 218 | Val/video6969.mp4 | 0.803 | - | 7 | 0.625 | a woman is eating food in a kitchen | a line of people are getting some food in a kitchen |
| 219 | Val/video6960.mp4 | 0.803 | - | 8 | 0.665 | a man is running on a track | a bunch of men run down the track |
| 220 | Val/video6756.mp4 | 0.802 | - | 166 | 0.565 | a man is talking about a car | a man is talking about and showing about how to drive a car |
| 221 | Val/video6804.mp4 | 0.802 | - | 27 | 0.512 | a woman is talking about a house | a blonde woman sits and talks about a house |
| 222 | Val/video6714.mp4 | 0.802 | - | 39 | 0.451 | a man is talking about a car | a man is talking |
| 223 | Val/video6720.mp4 | 0.800 | - | 20 | 0.592 | a man is mixing ingredients in a bowl | a man is mixing cream into flour |
| 224 | Val/video6555.mp4 | 0.800 | - | 21 | 0.642 | a group of people are playing soccer | various people fight at a soccer game |
| 225 | Val/video6617.mp4 | 0.800 | - | 68 | 0.497 | a woman is playing a game | girls try to play a game |
| 226 | Val/video6734.mp4 | 0.799 | - | 22 | 0.687 | a video game of a wrestling match | professional wrestling match video game |
| 227 | Val/video6643.mp4 | 0.798 | - | 82 | 0.579 | a man is talking on a stage | a man is talking |
| 228 | Val/video6902.mp4 | 0.798 | - | 103 | 0.558 | a man is dancing | a group of men dancing |
| 229 | Val/video6627.mp4 | 0.798 | - | 17 | 0.564 | a man is eating a horse | a man taking care of a cow |
| 230 | Val/video6585.mp4 | 0.798 | - | 13 | 0.546 | a man and a woman are sitting in a room | a man and a woman talking |
| 231 | Val/video6830.mp4 | 0.797 | - | 162 | 0.550 | a man is talking to a woman | a man talks to a woman on a game show |
| 232 | Val/video6853.mp4 | 0.797 | Y | 1 | 0.698 | a person is playing minecraft | footage of someone playing the game minecraft |
| 233 | Val/video6607.mp4 | 0.796 | - | 144 | 0.581 | a man jumps into a ball and jumps on a wall | a man is climbing a basket ball ring in a post and falling down |
| 234 | Val/video6653.mp4 | 0.796 | - | 2 | 0.689 | a motorcycle is riding a bike | a man is riding a bike |
| 235 | Val/video6928.mp4 | 0.796 | Y | 1 | 0.588 | a woman is talking about a movie | a woman is talking |
| 236 | Val/video6799.mp4 | 0.795 | Y | 1 | 0.622 | a man is singing on stage | a man is playing drums |
| 237 | Val/video6693.mp4 | 0.794 | - | 3 | 0.583 | a man and a woman are sitting in a couch | a woman and man are playing around together in a field |
| 238 | Val/video6846.mp4 | 0.793 | Y | 1 | 0.675 | a group of people are dancing on a stage | there are many people dancing on the stage show |
| 239 | Val/video6929.mp4 | 0.792 | - | 3 | 0.713 | a race horses are racing in a field | horse jockey s are racing on a dirt track |
| 240 | Val/video6649.mp4 | 0.792 | - | 77 | 0.519 | a man is talking about a movie | a man is talking |
| 241 | Val/video6525.mp4 | 0.790 | - | 25 | 0.530 | a man is talking about a movie | a man talks about space |
| 242 | Val/video6786.mp4 | 0.788 | - | 409 | 0.622 | two men are talking to each other | two men talking on a business talk show to one another |
| 243 | Val/video6809.mp4 | 0.788 | Y | 1 | 0.648 | a man and woman are talking to each other | a man and a lady kissing |
| 244 | Val/video7005.mp4 | 0.787 | - | 47 | 0.635 | a woman is talking | phil is talking to a woman |
| 245 | Val/video6926.mp4 | 0.787 | - | 3 | 0.594 | a man is talking about a car | a man provides a review of a vehicle |
| 246 | Val/video6640.mp4 | 0.787 | - | 2 | 0.624 | a cat is sitting in a room | a cat is laying on the floor |
| 247 | Val/video6758.mp4 | 0.787 | Y | 1 | 0.623 | a group of people are playing guitar | a kid is playing guitar |
| 248 | Val/video6888.mp4 | 0.786 | - | 6 | 0.529 | a man in a suit is talking about clothes | a man talks about fashion design |
| 249 | Val/video6931.mp4 | 0.785 | - | 10 | 0.566 | a man in a blue shirt is racing on a yellow car | a man in car is dragging another car in behind while a person dressed in yellow tries to |
| 250 | Val/video6571.mp4 | 0.784 | - | 12 | 0.506 | a man is talking about a situation | a man is talking about an athlete |
| 251 | Val/video6591.mp4 | 0.783 | - | 30 | 0.568 | a woman is sitting in a room | two women speak to each other in a room |
| 252 | Val/video6817.mp4 | 0.783 | - | 435 | 0.523 | a man is talking about a movie | there is a man talking about a new picture |
| 253 | Val/video6668.mp4 | 0.782 | Y | 1 | 0.622 | kids are talking about their kids | kids talk about parents reacting to their thoughts |
| 254 | Val/video6513.mp4 | 0.780 | - | 19 | 0.525 | a group of people are talking | a family is having conversation |
| 255 | Val/video6521.mp4 | 0.779 | Y | 1 | 0.526 | a man is talking to an audience | a man is talking on a television talk show |
| 256 | Val/video6914.mp4 | 0.778 | - | 3 | 0.642 | a man is talking about a news show | a news anchor reporting a story about a man |
| 257 | Val/video6990.mp4 | 0.777 | - | 5 | 0.673 | a woman is making a blender in a bowl | a woman mixes ingredients into a clear bowl |
| 258 | Val/video6785.mp4 | 0.776 | - | 31 | 0.659 | a man is driving a car and driving on the road | a man is seen driving a red car |
| 259 | Val/video6711.mp4 | 0.776 | - | 10 | 0.515 | a woman is talking about a girl | two women and a man are talking |
| 260 | Val/video6755.mp4 | 0.775 | - | 5 | 0.532 | a man is playing a displaying on a screen | a man plays piano on the stage |
| 261 | Val/video6593.mp4 | 0.773 | - | 8 | 0.599 | a woman is giving a speech | people listening to a woman s speech |
| 262 | Val/video6916.mp4 | 0.772 | - | 11 | 0.585 | a group of people are standing in a stadium | a team of players squeeze onto a podium |
| 263 | Val/video6997.mp4 | 0.772 | - | 20 | 0.576 | a woman is making a bottle | a girl is making crafts |
| 264 | Val/video6946.mp4 | 0.772 | Y | 1 | 0.542 | a man is talking to a person | a person is explaining something |
| 265 | Val/video6707.mp4 | 0.771 | - | 4 | 0.627 | a car is shown | person showing off car |
| 266 | Val/video6834.mp4 | 0.771 | - | 9 | 0.553 | a man is reading a book | a person is touching a piece of paper |
| 267 | Val/video6951.mp4 | 0.771 | Y | 1 | 0.551 | a man is talking on a tv show | a man is talking to another man on a game show |
| 268 | Val/video6716.mp4 | 0.770 | - | 3 | 0.696 | a person is driving a car and driving down the road | a car drives underneath an overpass and continues down a highway |
| 269 | Val/video6882.mp4 | 0.769 | Y | 1 | 0.676 | a woman and woman are sitting on a bed | a guy and girl are fighting on a couch |
| 270 | Val/video6851.mp4 | 0.769 | - | 9 | 0.566 | a man in a white shirt is talking about a situation | a man in blue coat with tie is speaking |
| 271 | Val/video6754.mp4 | 0.768 | - | 2 | 0.594 | a man is playing a ball on the ground | a man is playing golf |
| 272 | Val/video6913.mp4 | 0.768 | - | 5 | 0.563 | a man is driving a car in a road | a animated man running in a street |
| 273 | Val/video6867.mp4 | 0.768 | Y | 1 | 0.682 | a woman is playing a tennis game | blonde woman is walking and playing tennis match |
| 274 | Val/video7003.mp4 | 0.767 | - | 13 | 0.584 | a man is sitting in a chair | a man approaches a man on a bench |
| 275 | Val/video6849.mp4 | 0.767 | - | 160 | 0.536 | a man is talking about a school | there is a man is talking about science lab |
| 276 | Val/video6661.mp4 | 0.767 | - | 22 | 0.584 | a woman is cooking in a kitchen | a woman cooks cheeseburgers in a pan on the stove |
| 277 | Val/video6871.mp4 | 0.766 | - | 14 | 0.529 | a man in a black suit is talking about a cube | a men in dark brown suit is sitting and talking |
| 278 | Val/video6828.mp4 | 0.766 | - | 2 | 0.617 | a woman is dancing in a room | the woman is dancing and singing the song |
| 279 | Val/video6988.mp4 | 0.766 | - | 6 | 0.611 | a cat is playing in a room | someone is playing with cats in home |
| 280 | Val/video6542.mp4 | 0.765 | - | 4 | 0.587 | a man is running on a track | a woman runner running on a track |
| 281 | Val/video6927.mp4 | 0.764 | - | 2 | 0.602 | a woman is making her hair on her face | there is a woman applying some cream on the face |
| 282 | Val/video6514.mp4 | 0.763 | Y | 1 | 0.640 | a person is surfing on a beach while a man is riding on the water | a surfer is riding waves and a person on a dirt bike riding a trail |
| 283 | Val/video6962.mp4 | 0.763 | - | 14 | 0.487 | a man and a woman are talking | a man and a woman looking at laptop and talking |
| 284 | Val/video6614.mp4 | 0.763 | Y | 1 | 0.632 | a woman is performing on stage | a woman is singing on stage when someone else enters the stage |
| 285 | Val/video6619.mp4 | 0.763 | - | 81 | 0.563 | a boy playing with a baby | a child is dancing |
| 286 | Val/video6886.mp4 | 0.762 | - | 2 | 0.562 | a man is talking to a guy | two men are talking |
| 287 | Val/video6728.mp4 | 0.762 | - | 6 | 0.557 | a person is applying a brush | a man is getting a facial |
| 288 | Val/video6958.mp4 | 0.762 | - | 10 | 0.552 | a man is talking about a restaurant | a man talking about advertising |
| 289 | Val/video6839.mp4 | 0.761 | Y | 1 | 0.645 | a man is laying on a bed | a man put a child in the child s cot |
| 290 | Val/video6749.mp4 | 0.759 | - | 2 | 0.685 | a woman is showing her hair | a woman is showing how she paints her finger nails |
| 291 | Val/video6564.mp4 | 0.759 | - | 2 | 0.566 | a woman is talking about a city | there is a woman is talking in front of a building |
| 292 | Val/video6654.mp4 | 0.757 | Y | 1 | 0.686 | a car is driving down the road and driving a road | a car with red lights is driving on a road |
| 293 | Val/video6961.mp4 | 0.757 | - | 28 | 0.572 | a man in a white shirt is talking about science | man in blue shirt is talking in front of the whiteboard |
| 294 | Val/video6605.mp4 | 0.757 | Y | 1 | 0.600 | a man is sitting on a couch | man is sitting near the bed and talking about something |
| 295 | Val/video6572.mp4 | 0.757 | - | 215 | 0.528 | a man is talking about a computer program | there is a man is talking about a user interface |
| 296 | Val/video6613.mp4 | 0.756 | - | 2 | 0.678 | a fish is swimming in the water | the fish are swimming in the green aquarium |
| 297 | Val/video6650.mp4 | 0.756 | - | 2 | 0.490 | a woman is talking about a church | a woman talking about relationships |
| 298 | Val/video6798.mp4 | 0.755 | - | 91 | 0.432 | a woman is sitting in a room | a women sings while sitting on the floor |
| 299 | Val/video6889.mp4 | 0.755 | - | 32 | 0.540 | a man is talking about a boy | the man talks to the camera |
| 300 | Val/video6694.mp4 | 0.754 | - | 24 | 0.607 | a girl is singing on a tv show | a child being interviewed on the television show the voice |
| 301 | Val/video6523.mp4 | 0.753 | - | 56 | 0.530 | a woman is talking to a man | a man is talking with head phone |
| 302 | Val/video6813.mp4 | 0.752 | - | 171 | 0.489 | a man is talking about a cartoon | a man talking of something about the crash course |
| 303 | Val/video6769.mp4 | 0.751 | - | 40 | 0.644 | a woman is sitting in a bed and having a couch | a man and a woman were rolling around in the bed |
| 304 | Val/video6854.mp4 | 0.750 | Y | 1 | 0.642 | a woman in a pink dress is standing on a stage | blonde woman in grey dress is singing on the scene |
| 305 | Val/video6657.mp4 | 0.750 | - | 2 | 0.560 | a man is talking about how to use a computer | a man teaching people how to perform a computer action |
| 306 | Val/video6810.mp4 | 0.750 | - | 8 | 0.649 | a woman is teaching exercises | a woman shows how to do exercises with a baby |
| 307 | Val/video6680.mp4 | 0.750 | Y | 1 | 0.587 | a monkey is playing monkey | there is a monkey playing with a baby |
| 308 | Val/video6737.mp4 | 0.749 | - | 129 | 0.597 | a man is talking about a trampoline | a person is talking about a video game |
| 309 | Val/video6722.mp4 | 0.749 | - | 52 | 0.601 | a man is playing a sports track | a person is driving a vehicle on the track |
| 310 | Val/video6863.mp4 | 0.748 | Y | 1 | 0.606 | a cartoon character is eating food | a toy is going to eat sweets in cartoon |
| 311 | Val/video6699.mp4 | 0.747 | - | 28 | 0.532 | a group of people are talking | the people are sitting and talking |
| 312 | Val/video6638.mp4 | 0.746 | - | 19 | 0.700 | a girl is talking about their kids | a boy and a girl are talking |
| 313 | Val/video6631.mp4 | 0.745 | - | 64 | 0.440 | two people are talking to a woman | two people talk in a tv show |
| 314 | Val/video6831.mp4 | 0.744 | - | 3 | 0.580 | a man is playing a hamster with a mouse | a man trains a mouse to go through a hoop |
| 315 | Val/video6847.mp4 | 0.740 | - | 113 | 0.420 | a man is talking on a screen | a man is doing comedy on stage |
| 316 | Val/video6578.mp4 | 0.739 | - | 54 | 0.552 | a woman is talking about a computer | there is a women testing with a computer |
| 317 | Val/video6894.mp4 | 0.739 | - | 13 | 0.535 | a man is talking about a cartoon cartoon | a man is using an animation to talk about ancient humans |
| 318 | Val/video6568.mp4 | 0.736 | - | 37 | 0.536 | a man is talking to a woman | a man is giving a woman a candle |
| 319 | Val/video6533.mp4 | 0.735 | Y | 1 | 0.561 | a man is talking to another man in a room | a man is talking with another man both are wearing black color coat |
| 320 | Val/video6816.mp4 | 0.734 | - | 2 | 0.643 | a man is talking about a bottle | a man is serving drinks |
| 321 | Val/video6666.mp4 | 0.734 | Y | 1 | 0.561 | a man and a woman are talking about a movie | a woman is talking about a man s injury |
| 322 | Val/video6560.mp4 | 0.732 | - | 65 | 0.571 | a man is talking to a dog | a woman is feeding a dog |
| 323 | Val/video6588.mp4 | 0.731 | - | 46 | 0.463 | a man is sitting in a table | a person is sitting down while asleep he has his feet kicked up on the table |
| 324 | Val/video6559.mp4 | 0.731 | - | 21 | 0.513 | a man is talking on a phone | two people are talking on a couch |
| 325 | Val/video6950.mp4 | 0.730 | Y | 1 | 0.553 | a man and a woman are talking to each other | a man and woman talking as the woman changes into a nightgown |
| 326 | Val/video6618.mp4 | 0.730 | - | 78 | 0.445 | a man is talking on a tv show | a man is talking while standing out side |
| 327 | Val/video6954.mp4 | 0.730 | - | 2 | 0.613 | a man is singing on stage | a band singing on stage for an audience |
| 328 | Val/video6713.mp4 | 0.728 | - | 16 | 0.564 | a group of people are sitting in a room | a group of people are working on computers in an office area |
| 329 | Val/video6701.mp4 | 0.728 | - | 4 | 0.627 | two wrestlers are wrestling | three wrestlers are fighting in a ring |
| 330 | Val/video6721.mp4 | 0.727 | - | 141 | 0.461 | a woman is standing in a room | a woman in a red dress is walking down a hall |
| 331 | Val/video6660.mp4 | 0.726 | - | 12 | 0.544 | a man is talking about a movie | a man and a woman talking |
| 332 | Val/video6552.mp4 | 0.726 | - | 17 | 0.612 | a man in a blue shirt is talking about how to explain a situation | a boy sitting wearing brown dress and explaining something |
| 333 | Val/video6825.mp4 | 0.726 | - | 2 | 0.640 | a video game is being played | someone is showing video graphics game |
| 334 | Val/video6702.mp4 | 0.726 | - | 159 | 0.566 | a man in a white shirt is talking about a group of people | a man in blue dress is talking in the mic |
| 335 | Val/video6608.mp4 | 0.724 | - | 9 | 0.558 | two men are talking to a man | a man interviews a woman outside |
| 336 | Val/video6896.mp4 | 0.723 | - | 10 | 0.573 | a person is adding ingredients into a bowl | a person blends up a bunch of random foods |
| 337 | Val/video6788.mp4 | 0.722 | - | 3 | 0.629 | a young girl is giving a drink | a girl feeding a boy some food |
| 338 | Val/video6965.mp4 | 0.722 | - | 6 | 0.608 | a crowd of people are watching a concert | a crowd of people gather together |
| 339 | Val/video6861.mp4 | 0.721 | - | 8 | 0.552 | a man is sitting in a room | a man in a suit talking |
| 340 | Val/video6530.mp4 | 0.721 | - | 15 | 0.642 | a car is racing on the road | there is a car at the pier |
| 341 | Val/video6688.mp4 | 0.721 | - | 4 | 0.580 | a person is talking about a man | a person is operating on another person |
| 342 | Val/video6551.mp4 | 0.720 | - | 78 | 0.549 | a man is watching a movie | a man is talking |
| 343 | Val/video6879.mp4 | 0.720 | - | 2 | 0.612 | a cartoon character is talking | video of some cartoon characters having a conversation |
| 344 | Val/video6736.mp4 | 0.720 | - | 9 | 0.666 | a car is shown | reviews of cars are given |
| 345 | Val/video6692.mp4 | 0.720 | - | 162 | 0.452 | a man is talking about a movie | a man talking to another man |
| 346 | Val/video6826.mp4 | 0.718 | - | 5 | 0.522 | a group of people are talking to a man | a man is performing in front of a crowd |
| 347 | Val/video6664.mp4 | 0.718 | - | 5 | 0.641 | a man is preparing a dish in a kitchen | a man served food on his plate and getting ready to eat |
| 348 | Val/video6626.mp4 | 0.716 | Y | 1 | 0.538 | a man is talking about a movie | a man in homeless movie is depicted |
| 349 | Val/video6537.mp4 | 0.715 | Y | 1 | 0.642 | a man is talking in a car | a man explains another person on how to drive a car |
| 350 | Val/video6742.mp4 | 0.714 | - | 14 | 0.602 | a man is standing on the screen | a man is looking at something |
| 351 | Val/video6671.mp4 | 0.713 | - | 25 | 0.573 | a man is sitting in a house | man sitting down writing |
| 352 | Val/video6985.mp4 | 0.712 | - | 3 | 0.556 | a man is sitting in a table while a woman is eating a beer | a teapot whistling and a man in green serves tea to a woman sitting at a table |
| 353 | Val/video6730.mp4 | 0.710 | - | 3 | 0.651 | a man is playing a game | people are running in a race |
| 354 | Val/video7009.mp4 | 0.709 | - | 107 | 0.537 | a man is standing in a room | a man is talking |
| 355 | Val/video6873.mp4 | 0.709 | - | 20 | 0.502 | a cartoon character is playing a puzzle | a cartoon character falls to the floor |
| 356 | Val/video6775.mp4 | 0.708 | - | 3 | 0.611 | a man is talking to each other | a man is telling a story about cats |
| 357 | Val/video6704.mp4 | 0.707 | - | 19 | 0.606 | a child is playing with a baby | a kid is crawling around on a thing |
| 358 | Val/video6678.mp4 | 0.706 | - | 6 | 0.645 | a dog is playing with a baby | some people play with pets |
| 359 | Val/video6837.mp4 | 0.705 | - | 3 | 0.705 | a cat is talking about a dog | a dog talks about being very busy while a cat throws a dish of cat food at the |
| 360 | Val/video6824.mp4 | 0.705 | - | 34 | 0.523 | a woman is explaining how to remove a stroller | a lady gives an explanation on plugging a power surge in |
| 361 | Val/video6966.mp4 | 0.704 | - | 2 | 0.615 | a woman is demonstrating how to do something | a woman is making a pastry |
| 362 | Val/video6673.mp4 | 0.704 | - | 10 | 0.504 | a man is talking to a person | men are sitting at a table talking |
| 363 | Val/video6832.mp4 | 0.703 | - | 25 | 0.468 | a woman is talking on a street | a man is talking |
| 364 | Val/video6621.mp4 | 0.703 | - | 53 | 0.550 | a man is talking about something | a man wearing helmet is talking |
| 365 | Val/video6877.mp4 | 0.701 | - | 9 | 0.609 | a cartoon character is talking | a cartoon character is scaring another character |
| 366 | Val/video6700.mp4 | 0.701 | Y | 1 | 0.550 | a group of people are talking | a man giving a speech |
| 367 | Val/video6984.mp4 | 0.699 | - | 3 | 0.657 | a cat is sitting in a room | a group of kittens lay in a box together |
| 368 | Val/video6978.mp4 | 0.699 | - | 48 | 0.524 | a video of a spider is flying around | a video of an octopus swimming in the ocean |
| 369 | Val/video6855.mp4 | 0.699 | - | 4 | 0.520 | a man is talking about a cell phone | there is a man with specs talking from a house |
| 370 | Val/video6778.mp4 | 0.699 | - | 2 | 0.658 | a video of a fireworks is being shown | footage of a large fireworks show |
| 371 | Val/video6890.mp4 | 0.698 | - | 4 | 0.629 | a man is showing a piece of electronics | a man looks through boxes of mechanical parts |
| 372 | Val/video6682.mp4 | 0.698 | - | 26 | 0.552 | a man is talking about a map | a player talks about a minecraft map |
| 373 | Val/video6947.mp4 | 0.696 | - | 6 | 0.590 | a crowd of people gathered at a meeting | a group of people appear to be celebrating |
| 374 | Val/video6904.mp4 | 0.696 | - | 28 | 0.585 | a man is talking about a street | there are some people is running in to the street |
| 375 | Val/video6943.mp4 | 0.695 | Y | 1 | 0.541 | a man in a white shirt is explaining how to do his body | a man wearing a white dress in a white color room |
| 376 | Val/video6983.mp4 | 0.695 | Y | 1 | 0.663 | a ski ski snowboarding on the snow | two persons seeking down the snow on skateboard |
| 377 | Val/video6641.mp4 | 0.694 | - | 3 | 0.644 | a dish is shown on a plate | there is a woman is serving a dish on the table |
| 378 | Val/video6610.mp4 | 0.694 | - | 6 | 0.635 | a group of soldiers are talking | soldiers are getting ready for fight |
| 379 | Val/video6635.mp4 | 0.694 | - | 54 | 0.625 | a woman is talking about a movie | a movie trailer about a young woman |
| 380 | Val/video6547.mp4 | 0.694 | - | 3 | 0.568 | a woman is talking about how to cook a dish | a man is talking while tossing food in a pan |
| 381 | Val/video6872.mp4 | 0.692 | - | 8 | 0.453 | a man is talking about a soldier | there is a man talking from a dessert |
| 382 | Val/video6982.mp4 | 0.692 | - | 2 | 0.587 | a truck is driving on a road | a monster truck is driving in the dirt outside |
| 383 | Val/video6725.mp4 | 0.690 | - | 2 | 0.595 | a dog is talking about the dog | peoples are caring about pet dog |
| 384 | Val/video7000.mp4 | 0.690 | Y | 1 | 0.497 | a woman is looking in a camera | a woman picks at a spot beneath her eye |
| 385 | Val/video6633.mp4 | 0.690 | - | 3 | 0.586 | a cartoon character is playing a game | a person is playing a video game |
| 386 | Val/video6800.mp4 | 0.690 | - | 82 | 0.458 | a man is talking about a new appliance | a lady talking of something about the a product |
| 387 | Val/video6981.mp4 | 0.689 | Y | 1 | 0.723 | a mountain view is displaying a beautiful mountain | a beautiful foggy mountainside landscape is being showcased from high altitudes and angles |
| 388 | Val/video6859.mp4 | 0.688 | - | 13 | 0.444 | a man is talking to a woman | people talking about money |
| 389 | Val/video6570.mp4 | 0.687 | Y | 1 | 0.588 | a woman is playing a hamster | a woman trying to feed a vegetable to a hamster |
| 390 | Val/video6770.mp4 | 0.686 | - | 3 | 0.563 | a man is sitting in a statue | a man is giving a speech |
| 391 | Val/video6733.mp4 | 0.686 | - | 9 | 0.518 | a man is talking about a container | a lady talking of something about a product |
| 392 | Val/video6953.mp4 | 0.685 | - | 50 | 0.590 | a woman is talking about a girl | a woman is holding up a bag |
| 393 | Val/video6703.mp4 | 0.684 | Y | 1 | 0.577 | a man is holding a container with a glass | a human holds a cup with cucumbers in it while a hamster sits in a cage |
| 394 | Val/video6895.mp4 | 0.684 | Y | 1 | 0.710 | a surfer is surfing on the surf | a person on a surf board riding waves in the ocean |
| 395 | Val/video6732.mp4 | 0.684 | - | 90 | 0.584 | a cartoon character is walking down a wall | a cartoon character sticks up another one |
| 396 | Val/video6724.mp4 | 0.683 | - | 5 | 0.596 | a cartoon character is playing a game | a man is playing a video game |
| 397 | Val/video6522.mp4 | 0.683 | - | 2 | 0.569 | a cartoon character plays on a screen | a person is playing guitar hero on a television |
| 398 | Val/video6763.mp4 | 0.683 | Y | 1 | 0.576 | a man is talking to another man | there is a black t shirt man is talking with a man |
| 399 | Val/video6773.mp4 | 0.683 | Y | 1 | 0.698 | a car is driving on a road | a vehicle drives through the snow while a man speaks |
| 400 | Val/video6580.mp4 | 0.682 | - | 63 | 0.426 | a man is talking to a robot | a man is playing a video game |
| 401 | Val/video6642.mp4 | 0.682 | - | 11 | 0.576 | a man is fighting on the ground | there is a woman is falling down on the ground |
| 402 | Val/video6883.mp4 | 0.682 | - | 49 | 0.515 | a person is using a tablet | a person is working |
| 403 | Val/video6670.mp4 | 0.682 | - | 25 | 0.597 | a man is talking about the water | two men are walking over a stream |
| 404 | Val/video6517.mp4 | 0.681 | - | 109 | 0.525 | a man is talking about the movie | man giving silly synopsis of a movie |
| 405 | Val/video6968.mp4 | 0.681 | Y | 1 | 0.585 | a woman in a bedroom is looking in the bedroom | a woman throws clothes around the room from a table |
| 406 | Val/video6874.mp4 | 0.680 | - | 68 | 0.545 | a man is talking about various things | a man gives a presentation on a matter of science |
| 407 | Val/video6842.mp4 | 0.679 | - | 11 | 0.617 | a news reporter is talking about the news news | a pair of news anchors doing their jobs on the news |
| 408 | Val/video6683.mp4 | 0.678 | Y | 1 | 0.677 | a woman is sitting in a bed | a woman is working out with her baby |
| 409 | Val/video6543.mp4 | 0.677 | - | 28 | 0.535 | a group of people are fighting | several people fight in a small village |
| 410 | Val/video6609.mp4 | 0.677 | - | 89 | 0.568 | a group of people are talking on a table | a man is on a game show talking to a family |
| 411 | Val/video6835.mp4 | 0.677 | - | 26 | 0.576 | a cartoon character is sitting in a room | a cartoon character is exploring a basement |
| 412 | Val/video6995.mp4 | 0.676 | - | 3 | 0.583 | a person is talking about a cartoon character | it is a cartoon character |
| 413 | Val/video6833.mp4 | 0.675 | - | 2 | 0.660 | a basketball player is playing sports | people are playing sports |
| 414 | Val/video7008.mp4 | 0.674 | Y | 1 | 0.694 | a boat is standing in the water | a man is standing next to waterfall a man is jumping into the water a boat is driving |
| 415 | Val/video6639.mp4 | 0.674 | - | 100 | 0.485 | a group of people are standing in front of a crowd | a large crowd listens to a speech |
| 416 | Val/video6921.mp4 | 0.672 | - | 45 | 0.527 | a man is talking about a movie | a man and a woman are thinking of each other |
| 417 | Val/video6999.mp4 | 0.671 | - | 9 | 0.466 | a woman is showing how to polish a paper | a woman showing how to play with a child s toy |
| 418 | Val/video6604.mp4 | 0.670 | - | 3 | 0.503 | a man is talking to another man | a man in a movie is talking to some reporters |
| 419 | Val/video6923.mp4 | 0.668 | - | 6 | 0.541 | a group of people are talking about a crowd | a group of people stand outside |
| 420 | Val/video6718.mp4 | 0.668 | - | 2 | 0.582 | a cartoon character is standing in a room | a character in minecraft sits at a table |
| 421 | Val/video6987.mp4 | 0.668 | - | 44 | 0.533 | a girl in a white shirt is talking about a new outfit | girl trying out new clothes |
| 422 | Val/video6942.mp4 | 0.667 | - | 35 | 0.460 | a man is talking on a website | person is doing some searches on the web using computer |
| 423 | Val/video6823.mp4 | 0.666 | - | 3 | 0.590 | a woman is talking about a product | a woman is holding a bag |
| 424 | Val/video6751.mp4 | 0.663 | - | 7 | 0.573 | a man is giving a speech | there is a suit man talking on the stage |
| 425 | Val/video6869.mp4 | 0.663 | - | 32 | 0.530 | a woman is singing a music video | a woman runs through a snowy field |
| 426 | Val/video6930.mp4 | 0.662 | - | 40 | 0.534 | a group of people are singing | a group of people with guns are storming out |
| 427 | Val/video6801.mp4 | 0.661 | - | 109 | 0.624 | a man is talking about a television show | an advertisement of a television is playing on |
| 428 | Val/video6698.mp4 | 0.660 | - | 17 | 0.579 | a group of people are dancing and dancing | a crowd of people and two women singing or preparing to fight |
| 429 | Val/video6567.mp4 | 0.659 | - | 221 | 0.493 | a man is talking about a gadget | a person is doing skits |
| 430 | Val/video6665.mp4 | 0.656 | - | 73 | 0.503 | a man in a white shirt is talking about his camera | a man is talking in a room |
| 431 | Val/video6655.mp4 | 0.656 | - | 33 | 0.614 | a group of people are talking | guys talking about a game |
| 432 | Val/video6802.mp4 | 0.654 | - | 2 | 0.560 | a group of children are dancing | there are a group of children on stage in white shirts |
| 433 | Val/video6762.mp4 | 0.653 | - | 57 | 0.496 | a man is walking down a wall | a person is explaining something |
| 434 | Val/video6538.mp4 | 0.653 | Y | 1 | 0.597 | a man is talking about a horse | a horse is nodding its head |
| 435 | Val/video6575.mp4 | 0.652 | - | 6 | 0.481 | a man in a white shirt is holding a picture | a woman in white medical jacket is holding object in her hand |
| 436 | Val/video6528.mp4 | 0.652 | Y | 1 | 0.589 | a man in a white shirt is talking about a movie | a man wearing a black suit jacket talks during an interview on a financial news show |
| 437 | Val/video6843.mp4 | 0.652 | - | 8 | 0.410 | a man is talking about a crime | a woman talks about a man she is in love with |
| 438 | Val/video6686.mp4 | 0.651 | - | 7 | 0.511 | a man in a white shirt is talking to a person | a old lady in white color dress wearing cloth standing and speaking beside many persons standing displaying on |
| 439 | Val/video6715.mp4 | 0.651 | Y | 1 | 0.671 | a woman is talking about how to make a makeup | a lady is make up her lips by brush |
| 440 | Val/video6915.mp4 | 0.650 | - | 35 | 0.548 | a man is talking to a woman | a man in a suit speaks to people while outside |
| 441 | Val/video6519.mp4 | 0.650 | - | 35 | 0.613 | a cartoon character is playing with a little cartoon | a cartoon character is jumping from flower to flower |
| 442 | Val/video6600.mp4 | 0.649 | Y | 1 | 0.545 | a woman is talking about a movie | a man and woman are singing a song |
| 443 | Val/video6597.mp4 | 0.648 | - | 5 | 0.591 | a person is drawing a flower | a girl is making a project |
| 444 | Val/video6796.mp4 | 0.648 | - | 21 | 0.578 | a man is riding a car | some men are walking around in a ware house |
| 445 | Val/video6620.mp4 | 0.646 | - | 3 | 0.582 | a photo of a woman is shown | there is a woman taking selfie with a phone |
| 446 | Val/video6565.mp4 | 0.645 | - | 327 | 0.541 | a man is talking about a footage | a man laying on grass |
| 447 | Val/video6690.mp4 | 0.645 | - | 115 | 0.577 | a piece of pieces are shown on a table | a white device is lying down on a table |
| 448 | Val/video6535.mp4 | 0.643 | - | 187 | 0.496 | a man is talking about a fire | a man is giving a speech and a rocket is in the air |
| 449 | Val/video6865.mp4 | 0.642 | Y | 1 | 0.715 | a woman is standing in a picture | there is a woman is drinking some water |
| 450 | Val/video6860.mp4 | 0.642 | - | 99 | 0.456 | a group of people are talking | a woman is singing while people around her talk |
| 451 | Val/video6975.mp4 | 0.641 | - | 54 | 0.571 | a woman in a white shirt is talking about a camera | a woman in a room is standing around |
| 452 | Val/video6772.mp4 | 0.640 | - | 5 | 0.575 | a girl is swimming in the water | two young girls are playing in the sand together |
| 453 | Val/video6970.mp4 | 0.639 | - | 110 | 0.491 | a man is talking about politics | a guy in a gray suit is talking to someone |
| 454 | Val/video6705.mp4 | 0.638 | - | 11 | 0.613 | a man is walking in a forest | a woman walks 2 dogs and someone rides a bike through a forest |
| 455 | Val/video6918.mp4 | 0.637 | - | 67 | 0.447 | a man is talking to another person in a room | two men are meeting and having a conversation |
| 456 | Val/video6710.mp4 | 0.634 | - | 78 | 0.424 | a woman is standing on a street | woman are dancing |
| 457 | Val/video6996.mp4 | 0.632 | - | 2 | 0.592 | a man is showing how to use a tool | a man is cleaning a car piece |
| 458 | Val/video6791.mp4 | 0.628 | - | 93 | 0.511 | a group of women are dancing | four women are performing in the water |
| 459 | Val/video6949.mp4 | 0.624 | - | 2 | 0.548 | a group of people are walking in a park | a man is walking through a parking lot talking and looking at cars |
| 460 | Val/video6812.mp4 | 0.623 | - | 8 | 0.617 | a woman is talking about a horse | a horse is jumping over a hurdle |
| 461 | Val/video6908.mp4 | 0.622 | - | 3 | 0.621 | a man is talking about a cartoon character | someone is taking pictures of loge characters |
| 462 | Val/video6900.mp4 | 0.621 | - | 121 | 0.431 | a person is talking on a screen | a person presses some buttons on a scale |
| 463 | Val/video6827.mp4 | 0.621 | - | 35 | 0.581 | a cartoon character is dancing | a boy and girl cartoon characters talk to one another |
| 464 | Val/video6977.mp4 | 0.620 | - | 2 | 0.619 | a man is standing in a mountains | a man climbs ladder on a tree beside a small waterfall |
| 465 | Val/video6885.mp4 | 0.619 | - | 60 | 0.416 | a man is talking to another man | a man in an orange hardhat talks to a man in a yellow hardhat |
| 466 | Val/video6569.mp4 | 0.617 | - | 18 | 0.648 | a man is talking about a cartoon | an animated cartoon tiger talking something in the forest |
| 467 | Val/video6562.mp4 | 0.613 | - | 32 | 0.566 | a cartoon character is talking | a boy and a girl are talking in the animation film |
| 468 | Val/video6948.mp4 | 0.610 | - | 23 | 0.571 | a car is racing | some cars out side are getting ready to drive |
| 469 | Val/video6944.mp4 | 0.610 | - | 65 | 0.570 | cartoon characters are talking about a cartoon | a scene of the duck tales cartoon |
| 470 | Val/video6554.mp4 | 0.605 | - | 32 | 0.530 | a man is talking about a movie | a man rides a motorcycle down the road |
| 471 | Val/video6645.mp4 | 0.604 | - | 255 | 0.389 | a man is talking about a camera | a person is holding a quad copter |
| 472 | Val/video6590.mp4 | 0.604 | - | 22 | 0.559 | a group of people are dancing | people attend a sport event |
| 473 | Val/video6781.mp4 | 0.599 | Y | 1 | 0.552 | a cartoon is being shown | cartoon bird in the show |
| 474 | Val/video6932.mp4 | 0.598 | - | 160 | 0.423 | a man and a woman are talking | people talking in the city streets |
| 475 | Val/video6784.mp4 | 0.598 | - | 300 | 0.463 | a group of people are singing | a man talks while music plays and a group of men walk through and exercise on a sport |
| 476 | Val/video6622.mp4 | 0.597 | Y | 1 | 0.597 | a man is sitting in a room | an animated man is talking to a dog in his home |
| 477 | Val/video6905.mp4 | 0.590 | - | 9 | 0.516 | a man is talking about a computer | a black t shirt man is talking in front of a scree |
| 478 | Val/video6738.mp4 | 0.587 | Y | 1 | 0.613 | a photo of a woman is taking photos | men wearing sunglasses and women with hats take photographs of themselves |
| 479 | Val/video6689.mp4 | 0.585 | - | 19 | 0.513 | a man is talking about a city | a spider easing down over the town |
| 480 | Val/video6516.mp4 | 0.582 | - | 6 | 0.630 | a man is shown about a woman | a video with a man s nose |
| 481 | Val/video6789.mp4 | 0.582 | Y | 1 | 0.531 | a man is talking about a poster | a movie poster is being shown |
| 482 | Val/video6793.mp4 | 0.580 | - | 56 | 0.455 | a group of people are talking | a girl is singing |
| 483 | Val/video6587.mp4 | 0.577 | - | 460 | 0.529 | a group of people are playing a game | a man group of athletes doing different training exercises near a foot ball field |
| 484 | Val/video6540.mp4 | 0.574 | - | 41 | 0.473 | a man is talking to a robot | a man is preparing for a movie scene |
| 485 | Val/video6840.mp4 | 0.569 | - | 3 | 0.561 | a man is playing a map of a landscape | man is walking in the nature and he is injured |
| 486 | Val/video6646.mp4 | 0.568 | - | 9 | 0.459 | a woman is talking about a movie | a man kissing a women and hugging sometimes its a part of a movie |
| 487 | Val/video6935.mp4 | 0.567 | - | 2 | 0.572 | a man is standing in a gym | a girl in sports wear standing next to an old man in a red shirt |
| 488 | Val/video6526.mp4 | 0.563 | - | 20 | 0.648 | a woman is standing in the background | there is a hot woman is taking selfie |
| 489 | Val/video6581.mp4 | 0.547 | - | 54 | 0.358 | a man is talking to a woman | man in white shirt is in the room with brunette woman |
| 490 | Val/video6850.mp4 | 0.541 | - | 89 | 0.515 | a group of people are dancing | a group of judges watching a performance on stage |
| 491 | Val/video6955.mp4 | 0.540 | - | 58 | 0.505 | a man is talking about a movie | a person is driving a red truck down the highway |
| 492 | Val/video6753.mp4 | 0.529 | - | 98 | 0.615 | a woman is holding a picture | a carved sculpture depicts a woman and a child of ancient times |
| 493 | Val/video6862.mp4 | 0.521 | - | 101 | 0.430 | a woman is talking about a horror movie | a woman is laying dead with a man with water around them |
| 494 | Val/video6584.mp4 | 0.514 | - | 40 | 0.445 | a man is talking to a woman | a blonde woman is sitting down and looking |
| 495 | Val/video6723.mp4 | 0.478 | - | 131 | 0.469 | a video showing a map of the city | person is showing some information on the postcard |
| 496 | Val/video6681.mp4 | 0.462 | - | 2 | 0.679 | a man is talking | a person shows off various loge soldiers |
| 497 | Val/video6717.mp4 | 0.436 | Y | 1 | 0.525 | a group of people are being shown | person is putting some water into the white dish |