# Summary

## Dataset

1. Affectnet: Refer to [this paper](https://arxiv.org/abs/1708.03985). You can find the data over [here](Storage Server http://10.213.37.36 Data/External/Emotion/AffectNet_Dataset). It contains 414799 images with arousal, valence, expression labels.

## Training

Training from scratch on Affectnet dataset with only the expression label.

Training scripts can be found inside this repository over [here](http://rnd-github-usa-g.huawei.com/BostonResearchCenter/tflearn-models/tree/master/7expr).

## Labels

```python
class_names = [
    "Neutral",
    "Happy",
    "Sad",
    "Surprise",
    "Fear",
    "Disgust",
    "Anger",
    "Contempt",
    "None",
    "Uncertain",
    "Non-Face",
]
```

- TABLE 3 Number of Annotated Images in Each Category

| Expression | Number  |
| ---------- | ------- |
| Neutral    | 80,276  |
| Happy      | 146,198 |
| Sad        | 29,487  |
| Surprise   | 16,288  |
| Fear       | 8,191   |
| Disgust    | 5,264   |
| Anger      | 28,130  |
| Contempt   | 5,135   |
| None       | 35,322  |
| Uncertain  | 13,163  |
| Non-Face   | 88,895  |

## Performance

Validation accuracy on Affectnet dataset is 56.78%

## Model fine tuned from Mobilenet pre-trained weight using expression as labels

| Scenario                                                        | Label processing                       | Accuracy |
| --------------------------------------------------------------- | -------------------------------------- | -------- |
| validation set with equal category sample size (balanced)       | Convert label >= 7 to 0, with 7 labels | 37.6%    |
| validation set with different category sample size (unbalanced) | Convert label >= 7 to 0, with 7 labels | 57.1%    |
| validation set with equal category sample size (balanced)       | Remove label >= 7, with 7 labels       | 21.1%    |
| validation set with different category sample size (unbalanced) | Remove label >= 7, with 7 labels       | 53.9%    |

## Model fine tuned from Mobilenet pre-trained weight using Arousal, Valence as labels

Difficulty using Arousal, Valence: No such coordinates found in their source code. It hard to match the Arousal, Valence coordinates to expressions.

## Model trained from scratch using expression as labels

| Scenario                                                        | Label processing                       | Accuracy |
| --------------------------------------------------------------- | -------------------------------------- | -------- |
| validation set with equal category sample size (balanced)       | None, with 11 labels                   | 26.6%    |
| validation set with different category sample size (unbalanced) | None, with 11 labels                   | 43.6%    |
| validation set with equal category sample size (balanced)       | Convert label >= 7 to 0, with 7 labels | 53.78%   |
| validation set with different category sample size (unbalanced) | Convert label >= 7 to 0, with 7 labels | 70.82%   |

## Model trained from scratch using Arousal, Valence as labels
