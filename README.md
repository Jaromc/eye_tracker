# Eye tracking using Tensorflow #

Eye tracking using the UTK and 300W datasets. This example uses only the eye landmarks and not the full landmark set around the face. 

The data was first sanitized to remove some images that were outside my use case. For example, multiple people in the image or in some cases the landmarks were just incorrect.

A number of examples online using the UTK dataset suffer generalization issues. The dataset is very curated. In an attempt to improve upon this I augmented the images which produced ok results. I then added in augmentation per epoch which adds a lot of overhead but produced better results. The final step was to add the 300W dataset which contains whole body images vs UTKs face only images.

The final results I was happy with but there's still a lot of room for improvement. More data/augmentations are needed. Another improvement would be to blur the image edges. The model began to learn the rotation of the squares which was noticeable if a square was rotated ~30deg but the face was centered.

The code is a bit of a mess and the model is large so this shouldn't be taken as a good example.

### Environment ###
* WSL

### Commands ###
* Docker is used to run tensorflow

```docker run --gpus all -it --rm -v <folder_dir>:/tmp tensorflow/tensorflow:latest-gpu bash```

```pip install pandas scikit-learn matplotlib albumentations```

* Open another terminal and run Tensorboard

```docker run -p 8888:8888 -p 6006:6006 -v <folder_dir>:/logdir --gpus all -it --rm tensorflow/tensorflow:latest-gpu bash```

```tensorboard --logdir /logdir --bind_all```

### Result ###
![picture](results1.png)
![picture](results2.png)

### Dataset example ###
UTK
```,1_0_2_20161219140530307.jpg,-4,71,-4.1,96,-3,120,-1,144,9,166,28,179,53,186,77,192,100,194,121,191,142,183,161,174,180,161.1,192.1,142.1,195,120.1,194.1,97,192.2,74,16,53.1,29,39,48,33,68,34,86,40,113,39.1,129,33.1,148,32,164,37,175,49,100.1,59,101,72,101.1,85,101.2,99,78,112,89,113.1,100.2,116,110,114,120.2,111,39.2,62,51,61,61.1,60,71.1,65,60.1,63,50,62.1,124,64,134,59.1,144.1,59.2,155,62.2,144.2,62.3,134.1,62.4,55,137,72.1,134.2,87,132,97.1,133,107,131,120.3,132.1,136,133.1,121.1,143,109,146,98,147,88,146.1,72.2,145,61.2,138,87.1,137.1,97.2,138.1,107.1,136.1,130,135,108,139,98.1,140,88.1,139.1```

300W
```
version: 1
n_points: 68
{
446.000 91.000
449.459 119.344
450.957 150.614
460.552 176.986
471.486 202.157
488.087 226.842
506.016 246.438
524.662 263.865
553.315 271.435
578.732 266.260
599.361 248.966
615.947 220.651
627.439 197.999
635.375 179.064
642.063 156.371
647.302 124.753
646.518 92.944
470.271 117.870
486.218 109.415
503.097 114.454
519.714 120.090
533.680 127.609
571.937 123.590
585.702 117.155
602.344 109.070
620.077 103.951
633.964 111.236
554.931 145.072
554.589 161.106
554.658 177.570
554.777 194.295
532.717 197.930
543.637 202.841
555.652 205.483
565.441 202.069
576.368 197.061
487.474 136.436
499.184 132.337
513.781 133.589
527.594 143.047
513.422 144.769
499.117 144.737
579.876 140.815
590.901 130.008
605.648 128.376
618.343 132.671
606.771 140.525
593.466 141.419
519.040 229.040
536.292 221.978
547.001 221.192
557.161 224.381
568.172 219.826
579.144 222.233
589.098 224.410
581.071 239.804
570.103 251.962
558.241 254.844
547.661 254.621
534.085 247.772
524.758 230.477
547.684 231.663
557.304 230.805
568.172 229.159
585.417 225.992
569.211 237.777
557.473 240.542
547.989 240.014
}
```

### References ###
* https://ibug.doc.ic.ac.uk/resources/300-W/
* https://susanqq.github.io/UTKFace/
