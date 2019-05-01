# Affectnet

## File structure

```shell
.
├── Automatically_Annotated_file_lists
│   └── automatically_annotated.csv
├── Automatically_Annotated_Images
│   ├── 1
│   ├── 2
│   ├── 3
│   ├── ...
├── Expression
│   ├── meta-data
│   └── network
├── Manually_Annotated_file_lists
│   ├── training.csv
│   └── validation.csv
├── Manually_Annotated_Images
│   ├── 1
│   ├── 2
│   ├── 3
│   ├── ...
└── Valence_Arousal
    ├── meta-data
    └── network

```

## Dataset detail

Affectnet: Refer to [this paper](https://arxiv.org/abs/1708.03985). You can find the data over
[here](Storage Server http://10.213.37.36 Data/External/Emotion/AffectNet_Dataset). It contains
414799 images with arousal, valence, expression labels.

# IMDB-WIKI

## File structure

```shell
.
├── imdb_crop
│   ├── 00
│   ├── 01
│   ├── 02
│   ├── ...
│   └── imdb.mat
├── imdb-wiki.pkl
└── wiki_crop
    ├── 00
    ├── 01
    ├── 02
    ├── ...
    └── wiki.mat

```

## Dataset detail

[IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/): 505310 automatically autonated
images. Labels including age and gender are not very accuracte.

The age of a person can be calculated based on the date of birth and the time when the photo was
taken (note that we assume that the photo was taken in the middle of the year):

```
[age,~]=datevec(datenum(wiki.photo_taken,7,1)-wiki.dob);
```

# UTKFace

## File structure

```shell
.
├── README.md
├── UTKFace
│   ├── 100_0_0_20170112213500903.jpg.chip.jpg
│   ├── 100_0_0_20170112215240346.jpg.chip.jpg
│   ├── 10_0_0_20161220222308131.jpg.chip.jpg
│   ├── ...
```

# Audience

## File structure

```shell
.
├── crop_part1.tar.gz
├── faces
│   ├── 100003415@N08
│   ├── 10001312@N04
│   ├── 100014826@N03
│   ├── ...
├── fold_0_data.txt
├── fold_1_data.txt
├── fold_2_data.txt
├── fold_3_data.txt
├── fold_4_data.txt
├── fold_frontal_0_data.txt
├── fold_frontal_1_data.txt
├── fold_frontal_2_data.txt
├── fold_frontal_3_data.txt
└── fold_frontal_4_data.txt

```

## Dataset detail

[Adience Benchmark Of Unfiltered Faces](https://talhassner.github.io/home/projects/Adience/Adience-data.html):
13554 manually annotated images. Labels including age and gender are accurate.
