# Summary

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

## Model fine tuned from Mobilenet pre-trained weight using expression as labels

| Scenario                                                        | Label processing                       | Accuracy |
| --------------------------------------------------------------- | -------------------------------------- | -------- |
| validation set with equal category sample size (balanced)       | Convert label >= 7 to 0, with 7 labels | 37.6%    |
| validation set with different category sample size (unbalanced) | Convert label >= 7 to 0, with 7 labels | 57.1%    |
| validation set with equal category sample size (balanced)       | Remove label >= 7, with 7 labels       | 21.1%    |
| validation set with different category sample size (unbalanced) | Remove label >= 7, with 7 labels       | 53.9%    |

## Model fine tuned from Mobilenet pre-trained weight using Arousal, Valence as labels

Difficulty using Arousal, Valence: No such coordinates found in their source code. It hard to match
the Arousal, Valence coordinates to expressions.

## Model trained from scratch using expression as labels

| Scenario                                                        | Label processing                       | Accuracy |
| --------------------------------------------------------------- | -------------------------------------- | -------- |
| validation set with equal category sample size (balanced)       | None, with 11 labels                   | 26.6%    |
| validation set with different category sample size (unbalanced) | None, with 11 labels                   | 43.6%    |
| validation set with equal category sample size (balanced)       | Convert label >= 7 to 0, with 7 labels | 53.78%   |
| validation set with different category sample size (unbalanced) | Convert label >= 7 to 0, with 7 labels | 70.82%   |

## Model trained from scratch using Arousal, Valence as labels
